# SPDX-License-Identifier: Apache-2.0
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
import zarr

ic = pytest.importorskip("icechunk")

from arrayroute.remote import S3Store  # noqa: E402
from arrayroute.snapshot import (  # noqa: E402
    FilesystemRepository,
    S3Repository,
    Snapshot,
    SnapshotReadSession,
)


@pytest.fixture
def repository(tmp_path):
    location = FilesystemRepository(str(tmp_path / "repository"))
    repo = ic.Repository.create(ic.local_filesystem_storage(location.path))
    session = repo.writable_session("main")
    group = zarr.group(store=session.store)
    values = np.random.default_rng(46).standard_normal((4, 17, 13, 9, 3), dtype=np.float32)
    for name in ["velocity", "reference"]:
        group.create_array(name, data=values, chunks=(1, 8, 8, 8, 3))
    group.attrs.update({"units": "m/s", "time_units": "s"})
    snapshot_id = session.commit("original simulation outputs")
    return location, repo, values, snapshot_id


def test_revision_survives_new_commits_eviction_and_spawn(repository):
    location, repo, values, sid = repository
    pinned = Snapshot.resolve(location)
    assert pinned.snapshot_id == sid
    session = SnapshotReadSession(pinned, max_open_arrays=1)
    selection = (1, slice(None), 3, slice(None), 2)
    np.testing.assert_array_equal(session.read("velocity", selection), values[selection])
    newer = repo.writable_session("main")
    zarr.open_group(newer.store, mode="a")["velocity"][:] = 99
    next_id = newer.commit("replace field")
    assert next_id != sid
    session.read("reference", selection)
    assert list(session._readers) == ["reference"]
    session.clear()
    np.testing.assert_array_equal(session.read("velocity", selection), values[selection])
    clone = pickle.loads(pickle.dumps(session))
    assert clone._session is None and not clone._readers
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(session.read, "velocity", selection) for _ in range(2)]
        for result in futures:
            np.testing.assert_array_equal(result.result(), values[selection])
    latest = SnapshotReadSession(Snapshot.resolve(location))
    np.testing.assert_array_equal(latest.read("velocity", selection), 99)


def test_uncommitted_updates_are_invisible_and_readers_cannot_write(repository):
    location, repo, values, _ = repository
    writer = repo.writable_session("main")
    zarr.open_group(writer.store, mode="a")["velocity"][:] = 7
    session = SnapshotReadSession(Snapshot.resolve(location))
    reader = session.open("velocity")
    np.testing.assert_array_equal(reader.read((0,) + (slice(None),) * 4), values[0])
    with pytest.raises(ic.ReadOnlyError):
        reader.backend.array[:] = 5
    writer.discard_changes()
    np.testing.assert_array_equal(session.read("velocity", (0,) + (slice(None),) * 4), values[0])


def test_budget_and_sparse_selection_preserve_bits(repository):
    location, _, values, _ = repository
    session = SnapshotReadSession(Snapshot.resolve(location), buffer_bytes=24 << 10)
    selection = (slice(None), slice(None, None, 7), 3, slice(None, None, 2), slice(None))
    assert session.read("velocity", selection).tobytes() == values[selection].tobytes()
    with pytest.raises(MemoryError):
        session.read("velocity", (slice(None),) * 5)
    result = np.empty_like(values)
    for tile, block in session.open("velocity").stream((slice(None),) * 5):
        result[tile.destination] = block
    assert result.tobytes() == values.tobytes()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"buffer_bytes": 0},
        {"cache_bytes": -1},
        {"max_open_arrays": 0},
        {"metadata_entries": 0},
        {"concurrency": 0},
        {"range_request_bytes": 0},
        {"range_request_bytes": -1},
        {"range_request_bytes": 1 << 64},
    ],
)
def test_limits_fail_before_io(kwargs):
    with pytest.raises(ValueError):
        SnapshotReadSession(Snapshot(FilesystemRepository("missing"), "abc"), **kwargs)


def test_reject_untrusted_paths_and_missing_revisions(repository):
    location, _, _, _ = repository
    session = SnapshotReadSession(Snapshot.resolve(location))
    for name in ["../velocity", "/velocity", "a//b", "a\\b", "a\x00b"]:
        with pytest.raises(ValueError):
            session.open(name)
    with pytest.raises(ic.SnapshotNotFoundError):
        SnapshotReadSession(Snapshot(location, "00000000000000000000")).open("velocity")
    with pytest.raises(ValueError):
        S3Repository(S3Store("test-bucket", endpoint="http://localhost:9000"))
    remote = S3Repository(S3Store("test-bucket", credentials="environment"))
    assert pickle.loads(pickle.dumps(remote)) == remote


def test_local_range_default_does_not_override_remote_transport_policy():
    local = SnapshotReadSession(Snapshot(FilesystemRepository("missing"), "abc"))
    remote = SnapshotReadSession(Snapshot(S3Repository(S3Store("test-bucket")), "abc"))
    assert local.effective_range_request_bytes == 1 << 20
    assert remote.effective_range_request_bytes is None
    tuned = SnapshotReadSession(remote.snapshot, range_request_bytes=8 << 20)
    assert tuned.effective_range_request_bytes == 8 << 20


@pytest.mark.parametrize("range_bytes", [4096, 1 << 20, 32 << 20])
def test_read_range_policy_preserves_revision_and_repository_settings(repository, range_bytes):
    location, repo, values, sid = repository
    before = repo.config
    session = SnapshotReadSession(Snapshot(location, sid), range_request_bytes=range_bytes)
    selection = (slice(None), slice(1, 17, 3), 5, slice(None), 0)
    assert session.read("velocity", selection).tobytes() == values[selection].tobytes()
    restored = pickle.loads(pickle.dumps(session))
    assert restored.range_request_bytes == range_bytes
    assert restored.read("velocity", selection).tobytes() == values[selection].tobytes()
    after = ic.Repository.open(ic.local_filesystem_storage(location.path)).config
    assert before == after


def test_virtual_chunk_access_is_not_implicitly_authorized(tmp_path):
    from icechunk.config import RepositoryConfig
    from icechunk.storage import local_filesystem_store
    from icechunk.virtual import VirtualChunkContainer

    target = tmp_path / "external.bin"
    target.write_bytes(np.arange(4, dtype=np.float32).tobytes())
    location = FilesystemRepository(str(tmp_path / "repository"))
    config = RepositoryConfig.default()
    config.set_virtual_chunk_container(
        VirtualChunkContainer(tmp_path.as_uri() + "/", local_filesystem_store(str(tmp_path)))
    )
    repo = ic.Repository.create(ic.local_filesystem_storage(location.path), config=config)
    writer = repo.writable_session("main")
    zarr.group(writer.store).create_array(
        "field", shape=(4,), dtype="float32", chunks=(4,), compressors=None
    )
    writer.store.set_virtual_ref("field/c/0", target.as_uri(), offset=0, length=16)
    sid = writer.commit("external object reference")
    with pytest.raises(ic.IcechunkError):
        SnapshotReadSession(Snapshot(location, sid)).read("field", (slice(None),))


@pytest.mark.parametrize("coalesce", [False, True])
def test_row_batches_replay_order_and_bits_after_branch_changes(repository, coalesce):
    from arrayroute.reader import ReadStats

    location, repo, values, sid = repository
    session = SnapshotReadSession(Snapshot(location, sid), buffer_bytes=64 << 10)
    indices = [3, 0, -1, 2, 0]
    tail = (slice(None, None, 3), 2, slice(1, None, 2), slice(None))
    expected = np.stack([values[(i, *tail)] for i in indices])
    writer = repo.writable_session("main")
    zarr.open_group(writer.store, mode="a")["velocity"][:] = -123
    writer.commit("replace current branch")
    for candidate in (session, pickle.loads(pickle.dumps(session))):
        stats = ReadStats()
        result = candidate.read_rows("velocity", indices, tail=tail,
                                     coalesce=coalesce, stats=stats)
        assert result.tobytes() == expected.tobytes()
        assert stats.requests > 0
        candidate.clear()
        assert candidate.read_rows("velocity", indices, tail=tail).tobytes() == expected.tobytes()


def test_thread_sessions_share_budget_for_real_snapshot_batches(repository, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from arrayroute.backends import ZarrBackend
    from arrayroute.budget import ReadBudget

    location, _, values, sid = repository
    budget = ReadBudget(64 << 10)
    barrier = threading.Barrier(6)
    native_read = ZarrBackend.read

    def checked_read(backend, selection):
        state = budget.snapshot()
        assert 0 < state["used_bytes"] <= budget.max_bytes
        return native_read(backend, selection)

    monkeypatch.setattr(ZarrBackend, "read", checked_read)

    def gather(rank):
        session = SnapshotReadSession(Snapshot(location, sid), read_budget=budget)
        assert session.open("velocity").buffer_bytes == budget.max_bytes
        barrier.wait(timeout=10)
        indices = [rank % 4, 0, -1, rank % 4]
        tail = (slice(None, None, 2), 4, slice(None), 1)
        result = session.read_rows("velocity", indices, tail=tail)
        assert result.tobytes() == np.stack([values[(i, *tail)] for i in indices]).tobytes()

    with ThreadPoolExecutor(6) as executor:
        list(executor.map(gather, range(6)))
    assert budget.snapshot()["used_bytes"] == 0
    assert 0 < budget.snapshot()["peak_bytes"] <= budget.max_bytes


def test_snapshot_budget_lifecycle_cancel_error_and_spawn(repository, monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from arrayroute.budget import ReadBudget

    location, _, _, sid = repository
    budget = ReadBudget(64 << 10)
    session = SnapshotReadSession(Snapshot(location, sid), read_budget=budget)
    reader = session.open("velocity")
    assert reader.shared_budget is budget
    with pytest.raises(TypeError, match="after spawn"):
        pickle.dumps(session)
    session.clear()
    assert session.open("velocity").shared_budget is budget
    stream = session.open("velocity").stream((slice(None),) * 5)
    next(stream)
    assert budget.snapshot()["used_bytes"] > 0
    stream.close()
    assert budget.snapshot()["used_bytes"] == 0

    cancel = threading.Event()
    with ThreadPoolExecutor(1) as executor:
        with budget.reserve(budget.max_bytes):
            pending = executor.submit(session.read_rows, "velocity", [0],
                                      cancelled=cancel.is_set)
            deadline = time.monotonic() + 5
            while budget.snapshot()["waiting"] != 1:
                assert time.monotonic() < deadline
                time.sleep(.001)
            cancel.set()
            with pytest.raises(InterruptedError):
                pending.result(timeout=5)
    assert budget.snapshot()["used_bytes"] == budget.snapshot()["waiting"] == 0

    def broken_read(_):
        raise OSError("unavailable object")

    monkeypatch.setattr(session.open("velocity").backend, "read", broken_read)
    with pytest.raises(OSError, match="unavailable"):
        session.read_rows("velocity", [0])
    assert budget.snapshot()["used_bytes"] == 0


def test_snapshot_rows_reject_oversized_and_invalid_batches_before_payload(repository, monkeypatch):
    from arrayroute.budget import ReadBudget

    location, _, _, sid = repository
    session = SnapshotReadSession(Snapshot(location, sid), read_budget=ReadBudget(8192))

    def unexpected_read(_):
        pytest.fail("invalid selection reached payload I/O")

    monkeypatch.setattr(session.open("velocity").backend, "read", unexpected_read)
    with pytest.raises(MemoryError):
        session.read_rows("velocity", [0, 1])
    with pytest.raises(TypeError):
        session.read_rows("velocity", [True], tail=(0, 0, 0, 0))
    with pytest.raises(IndexError):
        session.read_rows("velocity", [4], tail=(0, 0, 0, 0))
    with pytest.raises(TypeError, match="ReadBudget"):
        SnapshotReadSession(Snapshot(location, sid), read_budget=1)
