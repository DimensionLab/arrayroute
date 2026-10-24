# SPDX-License-Identifier: Apache-2.0
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pytest
import zarr

ic = pytest.importorskip("icechunk")

from arrayroute.preparation import (  # noqa: E402
    PartitionGrid,
    Preparation,
    PreparationCancelledError,
    PreparationError,
)
from arrayroute.snapshot import FilesystemRepository, Snapshot, SnapshotReadSession  # noqa: E402


@pytest.fixture
def data(tmp_path):
    location = FilesystemRepository(str(tmp_path / "repository"))
    repo = ic.Repository.create(ic.local_filesystem_storage(location.path))
    session = repo.writable_session("main")
    values = np.arange(17 * 13, dtype="f4").reshape(17, 13)
    zarr.group(session.store).create_array(
        "velocity", data=values, chunks=(4, 4), attributes={"units": "m/s"}
    )
    sid = session.commit("source")
    return Snapshot(location, sid), repo, values


def prepare(snapshot, **kwargs):
    return Preparation.create(
        snapshot,
        branch="work",
        job="gradient",
        source="velocity",
        output="prepared",
        chunks=(4, 4),
        task_shape=(8, 8),
        halo=(1, 1),
        dtype="f4",
        operation="gradient-v1",
        parameters={"spacing_m": 2},
        attributes={"units": "1/s", "dims": ["x", "y"]},
        **kwargs,
    )


def derivative(values):
    return np.gradient(values, 2, axis=0)


def test_partition_ownership_and_halos():
    grid = PartitionGrid((17, 13), (4, 4), (8, 8), (1, 2))
    count = np.zeros(grid.shape, dtype=int)
    for number in range(grid.count):
        part = grid.partition(number)
        count[part.output] += 1
        for sl, chunk in zip(part.output, grid.chunks, strict=True):
            assert sl.start % chunk == 0
        values = np.arange(count.size).reshape(count.shape)
        np.testing.assert_array_equal(values[part.read][part.core], values[part.output])
    np.testing.assert_array_equal(count, 1)
    with pytest.raises(IndexError):
        grid.partition(-1)
    with pytest.raises(ValueError, match="multiples"):
        PartitionGrid((20,), (4,), (7,), (0,))
    # 4 PiB logical float32 shape is planned, not allocated or claimed as measured I/O.
    huge = PartitionGrid(
        (1024, 1024, 1024, 1024, 1024), (1, 64, 64, 64, 4), (1, 64, 64, 64, 4), (0, 1, 1, 1, 0)
    )
    assert huge.count == 1024 * 16**3 * 256
    assert huge.partition(huge.count - 1).output[-1] == slice(1020, 1024)


def test_halo_transform_preserves_input_revision_and_units(data):
    snapshot, repo, values = data
    job = prepare(snapshot)
    change = repo.writable_session("main")
    zarr.open_array(change.store, path="velocity", mode="r+")[:] = 999
    change.commit("new solver output must not affect in-flight preparation")
    with ThreadPoolExecutor(3) as pool:
        assert job.run(derivative, pool, batch_tasks=3) == job.grid.count
        assert job.run(derivative, pool) == 0
    published = job.publish()
    reader = SnapshotReadSession(published)
    result = reader.read("prepared", (slice(None),) * 2)
    np.testing.assert_array_equal(result, derivative(values))
    assert reader.attributes("prepared")["units"] == "1/s"
    assert job.publish() == published
    assert repo.lookup_branch("main") != published.snapshot_id


def test_failed_worker_checkpoints_success_and_resume_skips_done(data):
    snapshot, _, values = data
    job = prepare(snapshot)
    with pytest.raises(ValueError, match="incomplete"):
        job.publish()
    seen = []

    def fail_one(block):
        seen.append(float(block[0, 0]))
        if block[0, 0] == 0:
            raise RuntimeError("injected worker failure")
        return -block

    with ThreadPoolExecutor(3) as pool:
        with pytest.raises(PreparationError) as caught:
            job.run(fail_one, pool, batch_tasks=4)
    assert caught.value.tasks == [0]
    assert list(job.pending()) == [0, 4, 5]
    with pytest.raises(ValueError, match="do not match"):
        Preparation.resume(
            snapshot.location,
            branch="work",
            job="gradient",
            operation="different",
            parameters={"spacing_m": 2},
        )
    resumed = Preparation.resume(
        snapshot.location,
        branch="work",
        job="gradient",
        operation="gradient-v1",
        parameters={"spacing_m": 2},
    )
    with ThreadPoolExecutor(2) as pool:
        assert resumed.run(np.negative, pool, batch_tasks=2) == 3
    result = SnapshotReadSession(resumed.publish()).read("prepared", (slice(None),) * 2)
    assert result.tobytes() == (-values).tobytes()
    assert len(seen) == 4


@pytest.mark.parametrize("verification_workers", [1, 4])
def test_spawned_workers_and_digest_failure(data, verification_workers):
    snapshot, repo, values = data
    job = prepare(snapshot)
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert job.run(np.negative, pool, batch_tasks=2) == 6
    published = job.publish(verification_workers=verification_workers)
    np.testing.assert_array_equal(
        SnapshotReadSession(published).read("prepared", (slice(None),) * 2), -values
    )
    changed = repo.writable_session("work")
    zarr.open_array(changed.store, path="prepared", mode="r+")[0, 0] = 3
    changed.commit("simulate corrupt/replaced output after completion")
    with pytest.raises(ValueError, match="digest mismatch"):
        job.publish(verification_workers=verification_workers)


def test_parallel_publication_reuses_revision_and_shares_budget(data, monkeypatch):
    import arrayroute.preparation as preparation

    snapshot, _, _ = data
    job = prepare(snapshot)
    with ThreadPoolExecutor(2) as pool:
        job.run(np.negative, pool)
    published = job.publish()
    observed = []
    verify = preparation._verify_partition

    def observe(*args):
        observed.append(args[-1])
        return verify(*args)

    monkeypatch.setattr(preparation, "_verify_partition", observe)
    assert job.publish(verification_workers=3, buffer_bytes=3 << 20) == published
    assert observed == [1 << 20] * job.grid.count
    with pytest.raises(MemoryError, match="budget"):
        job.publish(verification_workers=3, buffer_bytes=300)
    with pytest.raises(ValueError, match="positive"):
        job.publish(verification_workers=0)


def test_bad_results_budget_and_cancellation_never_mark_done(data):
    snapshot, _, _ = data
    job = prepare(snapshot)
    with ThreadPoolExecutor(2) as pool:
        for operator, budget in [
            (np.negative, 1),
            (lambda v: v[0], 1 << 20),
            (lambda v: v.astype("f8"), 1 << 20),
        ]:
            with pytest.raises(PreparationError):
                job.run(operator, pool, batch_tasks=2, buffer_bytes=budget)
            assert len(list(job.pending())) == 6
        with pytest.raises(InterruptedError):
            job.run(np.negative, pool, cancelled=lambda: True)
        assert len(list(job.pending())) == 6


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_wait_deadline_does_not_submit(data, timeout):
    snapshot, _, _ = data
    job = prepare(snapshot)
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(ValueError, match="finite and positive"):
            job.run(np.negative, pool, batch_timeout_seconds=timeout)
    assert len(list(job.pending())) == job.grid.count


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_stalled_worker_checkpoints_success_and_late_result_is_not_published(data, stop):
    snapshot, repo, values = data
    job = prepare(snapshot)
    release = threading.Event()
    submitted = []
    pool = ThreadPoolExecutor(2)

    class RecordedExecutor:
        def submit(self, *args):
            future = pool.submit(*args)
            submitted.append(future)
            return future

    def stalled(block):
        if block[0, 0] == 0:
            assert release.wait(10), "Test must release its stalled worker"
            return block + 999  # Deliberately wrong late work must never replace the retry.
        return -block

    try:
        if stop == "cancel":
            with pytest.raises(PreparationCancelledError) as caught:
                job.run(stalled, RecordedExecutor(), batch_tasks=2,
                        cancelled=lambda: len(submitted) == 2 and submitted[1].done())
        else:
            with pytest.raises(PreparationError) as caught:
                job.run(stalled, RecordedExecutor(), batch_tasks=2, batch_timeout_seconds=1)
            assert caught.value.reason == "timeout" and caught.value.tasks == [0]
        assert caught.value.snapshot_id == repo.lookup_branch("work")
        assert submitted[0].running()
        assert list(job.pending()) == [0, 2, 3, 4, 5]
        resumed = Preparation.resume(snapshot.location, branch="work", job="gradient",
                                     operation="gradient-v1", parameters={"spacing_m": 2})
        with ThreadPoolExecutor(2) as replacement:
            assert resumed.run(np.negative, replacement, batch_tasks=2) == 5
        published = resumed.publish()
        release.set()
        submitted[0].result(timeout=10)
        assert repo.lookup_branch("work") == published.snapshot_id
        actual = SnapshotReadSession(published).read("prepared", (slice(None),) * 2)
        np.testing.assert_array_equal(actual, -values)
        assert list(resumed.pending()) == []
    finally:
        release.set()
        pool.shutdown(wait=True, cancel_futures=True)


def test_lost_worker_result_is_invisible_and_retried(data):
    from arrayroute.preparation import _transform

    snapshot, repo, values = data
    job = prepare(snapshot)
    parent = repo.writable_session("work")
    _transform(parent.fork(), snapshot, job.config, job.grid.partition(0), np.negative, 1 << 20)
    # Simulate losing the worker result before the coordinator merges/commits it.
    assert len(list(job.pending())) == job.grid.count
    current = SnapshotReadSession(Snapshot.resolve(snapshot.location, branch="work"))
    np.testing.assert_array_equal(current.read("prepared", (slice(None),) * 2), 0)
    with ThreadPoolExecutor(2) as pool:
        assert job.run(np.negative, pool) == job.grid.count
    with pytest.raises(MemoryError, match="budget"):
        job.publish(buffer_bytes=1)
    np.testing.assert_array_equal(
        SnapshotReadSession(job.publish()).read("prepared", (slice(None),) * 2), -values
    )


def test_reused_worker_switches_pinned_input_revision(data):
    snapshot, repo, values = data
    first = prepare(snapshot)
    change = repo.writable_session("main")
    zarr.open_array(change.store, path="velocity", mode="r+")[:] = values + 10
    newer = Snapshot(snapshot.location, change.commit("changed input"))
    second = Preparation.create(
        newer,
        branch="second",
        job="second",
        source="velocity",
        output="prepared",
        chunks=(4, 4),
        task_shape=(8, 8),
        halo=(1, 1),
        dtype="f4",
        operation="negative-v1",
        parameters={},
        attributes={"units": "m/s"},
    )
    with ThreadPoolExecutor(1) as pool:
        first.run(np.negative, pool)
        second.run(np.negative, pool)
    for job, expected in [(first, -values), (second, -(values + 10))]:
        actual = SnapshotReadSession(job.publish()).read("prepared", (slice(None),) * 2)
        np.testing.assert_array_equal(actual, expected)
