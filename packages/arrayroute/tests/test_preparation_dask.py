# SPDX-License-Identifier: Apache-2.0
"""Real Dask processes, including a killed worker, over a shared local repository."""
import logging
import os

import numpy as np
import pytest
import zarr

ic = pytest.importorskip("icechunk")
distributed = pytest.importorskip("distributed")

from arrayroute.preparation import Preparation  # noqa: E402
from arrayroute.snapshot import FilesystemRepository, Snapshot, SnapshotReadSession  # noqa: E402


@pytest.mark.parametrize("kill_worker", [False, True])
def test_dask_preparation_and_worker_recovery(tmp_path, kill_worker):
    location = FilesystemRepository(str(tmp_path / "repository"))
    repo = ic.Repository.create(ic.local_filesystem_storage(location.path))
    source = repo.writable_session("main")
    values = np.arange(5 * 17 * 13, dtype="float32").reshape(5, 17, 13)
    zarr.group(source.store).create_array("velocity", data=values, chunks=(1, 4, 4),
                                         attributes={"units": "m/s"})
    snapshot = Snapshot(location, source.commit("input"))
    job = Preparation.create(snapshot, branch="work", job="gradient", source="velocity",
                             output="gradient", chunks=(1, 4, 4), task_shape=(1, 8, 8),
                             halo=(0, 1, 1), dtype="float32", operation="gradient-v1",
                             parameters={"spacing_m": 2}, attributes={"units": "1/s"})
    marker = tmp_path / "crashed-once"
    # Closures model notebook-defined operators without relying on pytest's import path.
    marker_path = str(marker)

    def gradient(values):
        return np.gradient(values, 2, axis=1)

    def operator(values):
        if kill_worker and values.flat[0] == 0:
            try:
                fd = os.open(marker_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
                os._exit(79)
        return gradient(values)
    with distributed.LocalCluster(n_workers=2, threads_per_worker=1, processes=True,
                                  host="127.0.0.1", dashboard_address="127.0.0.1:0",
                                  memory_limit=0, silence_logs=logging.ERROR) as cluster:
        with distributed.Client(cluster) as client:
            pids = set(client.run(os.getpid).values())
            assert len(pids) == 2 and os.getpid() not in pids
            with client.get_executor(pure=False, retries=1) as executor:
                assert job.run(operator, executor, batch_tasks=5,
                               batch_timeout_seconds=30) == job.grid.count
                assert job.run(operator, executor, batch_timeout_seconds=30) == 0
    assert marker.exists() == kill_worker
    published = job.publish()
    reader = SnapshotReadSession(published)
    np.testing.assert_array_equal(reader.read("gradient", (slice(None),) * 3), gradient(values))
    assert reader.attributes("gradient")["units"] == "1/s"
    assert repo.lookup_branch("main") == snapshot.snapshot_id
    assert list(job.pending()) == []
