# SPDX-License-Identifier: Apache-2.0
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import zarr

ic = pytest.importorskip("icechunk")

from arrayroute.preparation import Preparation  # noqa: E402
from arrayroute.retention import ReleasedPinError, RetentionManager, RetentionPin  # noqa: E402
from arrayroute.snapshot import FilesystemRepository, Snapshot, SnapshotReadSession  # noqa: E402


@pytest.fixture
def data(tmp_path):
    location = FilesystemRepository(str(tmp_path / "repository"))
    repo = ic.Repository.create(
        ic.local_filesystem_storage(location.path),
        config=ic.RepositoryConfig(inline_chunk_threshold_bytes=0),
    )
    session = repo.writable_session("main")
    values = np.arange(128, dtype="f4")
    zarr.group(session.store).create_array("u", data=values, chunks=(16,))
    original = Snapshot(location, session.commit("source"))
    newer = repo.writable_session("main")
    zarr.open_array(newer.store, path="u", mode="r+")[:] = -values
    newest = Snapshot(location, newer.commit("replace source"))
    return repo, original, newest, values


def collect(repo):
    # Only these isolated fixtures are quiescent. A production cutoff must predate every
    # in-flight writer; a future cutoff would be unsafe with concurrent writers.
    cutoff = datetime.now(UTC) + timedelta(seconds=1)
    repo.expire_snapshots(cutoff, delete_expired_branches=False, delete_expired_tags=False)
    return repo.garbage_collect(cutoff)


def test_two_consumers_survive_cleanup_until_both_release(data):
    repo, original, _, values = data
    manager = RetentionManager(original.location)
    training = manager.retain(original, purpose="training")
    report = manager.retain(original, purpose="report")
    collect(repo)
    session = SnapshotReadSession(original)
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as workers:
        assert (
            workers.submit(session.read, "u", (slice(None),)).result().tobytes() == values.tobytes()
        )
    assert manager.release(training)
    assert not manager.release(training)
    collect(repo)
    assert SnapshotReadSession(original).read("u", (slice(None),)).tobytes() == values.tobytes()
    assert manager.release(report)
    summary = collect(repo)
    assert summary.chunks_deleted > 0 and summary.bytes_deleted > 0
    with pytest.raises(ic.SnapshotNotFoundError):
        SnapshotReadSession(original).open("u")
    with pytest.raises(ic.SnapshotNotFoundError):
        manager.retain(original, purpose="reader")


def test_retry_is_idempotent_but_cannot_retarget_or_resurrect_a_pin(data):
    _, original, newest, _ = data
    manager = RetentionManager(original.location)
    token = "a" * 32
    pin = manager.retain(original, purpose="training", token=token)
    assert manager.retain(original, purpose="training", token=token) == pin
    with pytest.raises(ValueError, match="another revision"):
        manager.retain(newest, purpose="training", token=token)
    with pytest.raises(ValueError, match="does not match"):
        manager.release(RetentionPin(newest, token, "training"))
    manager.verify(pin)
    manager.release(pin)
    with pytest.raises(ReleasedPinError):
        manager.retain(original, purpose="training", token=token)


def test_inventory_restart_and_repository_scope(data, tmp_path):
    repo, original, _, _ = data
    manager = RetentionManager(original.location)
    pin = manager.retain(original, purpose="report")
    repo.create_tag("user-publication", original.snapshot_id)
    restored = pickle.loads(pickle.dumps(manager))
    assert restored._writes_checked is False
    assert list(restored.pins()) == [pin]
    restored.verify(pickle.loads(pickle.dumps(pin)))
    wrong = RetentionManager(FilesystemRepository(str(tmp_path / "other")))
    for operation in [wrong.verify, wrong.release]:
        with pytest.raises(ValueError, match="another repository"):
            operation(pin)
    with pytest.raises(ValueError, match="another repository"):
        wrong.retain(original, purpose="reader")


@pytest.mark.parametrize("token", ["", "../" + "a" * 32, "A" * 32, "a" * 31])
def test_invalid_tokens_are_rejected_before_storage(data, token):
    _, original, _, _ = data
    with pytest.raises(ValueError, match="token"):
        RetentionManager(original.location).retain(original, purpose="reader", token=token)


def test_preparation_input_and_publication_survive_history_cleanup(data):
    repo, original, _, values = data
    manager = RetentionManager(original.location)
    job = Preparation.create(
        original,
        branch="prepare",
        job="negative",
        source="u",
        output="negative_u",
        chunks=(16,),
        task_shape=(32,),
        halo=(0,),
        dtype="f4",
        operation="negative-v1",
        parameters={},
        attributes={"units": "m/s", "dims": ["x"], "association": "cell"},
    )
    collect(repo)
    manager.verify(job.input_pin)
    with ThreadPoolExecutor(2) as pool:
        assert job.run(np.negative, pool) == 4
    published = job.publish()
    pin = job.output_pin(published)
    repo.delete_branch("prepare")
    collect(repo)
    assert SnapshotReadSession(original).read("u", (slice(None),)).tobytes() == values.tobytes()
    assert (
        SnapshotReadSession(published).read("negative_u", (slice(None),)).tobytes()
        == (-values).tobytes()
    )
    manager.verify(pin)
    manager.release(pin)
    manager.release(job.input_pin)
    collect(repo)
    with pytest.raises(ic.SnapshotNotFoundError):
        SnapshotReadSession(published).open("negative_u")


def test_lost_publication_tag_response_can_resume(data, monkeypatch):
    import arrayroute.preparation as module

    _, original, _, _ = data
    job = Preparation.create(
        original,
        branch="prepare",
        job="negative",
        source="u",
        output="negative_u",
        chunks=(16,),
        task_shape=(32,),
        halo=(0,),
        dtype="f4",
        operation="negative-v1",
        parameters={},
        attributes={},
    )
    with ThreadPoolExecutor(1) as pool:
        job.run(np.negative, pool)
    actual_retain = module._retain

    def lost_response(repo, pin):
        actual_retain(repo, pin)
        raise ConnectionError("injected lost response after tag commit")

    monkeypatch.setattr(module, "_retain", lost_response)
    with pytest.raises(ConnectionError):
        job.publish()
    monkeypatch.setattr(module, "_retain", actual_retain)
    resumed = Preparation.resume(
        original.location, branch="prepare", job="negative", operation="negative-v1", parameters={}
    )
    result = resumed.publish()
    RetentionManager(original.location).verify(resumed.output_pin(result))
