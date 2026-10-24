# SPDX-License-Identifier: Apache-2.0
"""Local checkpoint/retry experiment; reports measured costs, not scale extrapolations."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np

from .preparation import Preparation, PreparationError
from .snapshot import FilesystemRepository, Snapshot, SnapshotReadSession


@contextmanager
def _executor(kind: str, workers: int):
    with ExitStack() as stack:
        if kind == "dask":
            from distributed import Client, LocalCluster

            cluster = stack.enter_context(LocalCluster(
                n_workers=workers, threads_per_worker=1, processes=True,
                host="127.0.0.1", dashboard_address="127.0.0.1:0", memory_limit=0,
                silence_logs=logging.ERROR))
            client = stack.enter_context(Client(cluster))
            executor = client.get_executor(pure=False, retries=1)
        elif kind == "spawn":
            executor = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
        else:
            executor = ThreadPoolExecutor(workers)
        yield stack.enter_context(executor)


def main() -> None:
    import icechunk as ic
    import zarr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-dask", action="store_true",
                        help="Also measure one/four Dask worker processes; requires the dask extra")
    parser.add_argument("--task-size", type=int, default=64,
                        help="Spatial task edge, a positive multiple of 32")
    parser.add_argument("--omit-single-task-commits", action="store_true",
                        help="Omit the batch=1 control for larger materialized campaigns")
    args = parser.parse_args()
    if min(args.frames, args.repeats) < 1 or args.size < 16:
        parser.error("Positive frames/repeats and spatial size >= 16 required")
    if args.task_size < 32 or args.task_size % 32:
        parser.error("Task edge must be a positive multiple of 32")
    if args.include_dask:
        import distributed  # noqa: F401
    args.root.mkdir(parents=True, exist_ok=False)
    location = FilesystemRepository(str(args.root / "repository"))
    repo = ic.Repository.create(ic.local_filesystem_storage(location.path))
    session = repo.writable_session("main")
    shape = (args.frames, args.size, args.size, args.size)
    array = zarr.group(session.store).create_array(
        "velocity",
        shape=shape,
        chunks=(1, 32, 32, 32),
        dtype="f4",
        attributes={"units": "m/s", "dims": ["time", "x", "y", "z"]},
    )
    expected = hashlib.sha256()
    for frame in range(args.frames):
        values = np.random.default_rng(frame).standard_normal(shape[1:], dtype=np.float32)
        array[frame] = values
        expected.update(memoryview(-values).cast("B"))
    snapshot = Snapshot(location, session.commit("synthetic velocity source"))
    rows = []
    modes = [("thread", 1, 1), ("thread", 1, 16), ("thread", 4, 16), ("spawn", 4, 16)]
    if args.omit_single_task_commits:
        modes.remove(("thread", 1, 1))
    if args.include_dask:
        modes.extend([("dask", 1, 16), ("dask", 4, 16)])
    for repeat in range(args.repeats):
        order = list(modes)
        np.random.default_rng(128 + repeat).shuffle(order)
        for kind, workers, batch in order:
            name = f"{kind}-{workers}-{batch}-{repeat}"
            start = time.perf_counter()
            job = Preparation.create(
                snapshot,
                branch=name,
                job=name,
                source="velocity",
                output="reversed_velocity",
                chunks=(1, 32, 32, 32),
                task_shape=(1, args.task_size, args.task_size, args.task_size),
                halo=(0, 0, 0, 0),
                dtype="f4",
                operation="numpy-negative-v1",
                parameters={},
                attributes={"units": "m/s", "dims": ["time", "x", "y", "z"]},
            )
            initialized = time.perf_counter()
            with _executor(kind, workers) as pool:
                done = job.run(np.negative, pool, batch_tasks=batch, batch_timeout_seconds=120)
            transformed = time.perf_counter()
            published = job.publish()
            verified = time.perf_counter()
            actual = hashlib.sha256()
            reader = SnapshotReadSession(published)
            for frame in range(args.frames):
                actual.update(
                    memoryview(
                        reader.read("reversed_velocity", (frame,) + (slice(None),) * 3)
                    ).cast("B")
                )
            assert actual.digest() == expected.digest()
            rows.append(
                dict(
                    name=name,
                    repeat=repeat,
                    executor=kind,
                    workers=workers,
                    batch_tasks=batch,
                    tasks=done,
                    initialization_seconds=initialized - start,
                    transform_checkpoint_seconds=transformed - initialized,
                    publication_verify_seconds=verified - transformed,
                    verified_sha256=actual.hexdigest(),
                    snapshot_id=published.snapshot_id,
                )
            )
            (args.root / "measurements.json").write_text(
                json.dumps(
                    dict(
                        unique_input_bytes=int(np.prod(shape)) * 4,
                        shape=shape,
                        input_snapshot=snapshot.snapshot_id,
                        rows=rows,
                        caveats=[
                            "Synthetic local pilot, uncontrolled OS cache",
                            "Workers share one host",
                            "Includes worker startup and checkpoint costs",
                            "Not a TB/PB benchmark",
                        ],
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps(rows[-1]), flush=True)

    # A worker error is repeatable evidence separate from the timing campaign.
    job = Preparation.create(
        snapshot,
        branch="recovery",
        job="recovery",
        source="velocity",
        output="recovered",
        chunks=(1, 32, 32, 32),
        task_shape=(1, args.task_size, args.task_size, args.task_size),
        halo=(0, 0, 0, 0),
        dtype="f4",
        operation="negative-v1",
        parameters={},
        attributes={"units": "m/s"},
    )

    def fail_selected(values):
        if values[0, 0, 0, 0] > 0:
            raise RuntimeError("deliberately fail tasks whose first value is positive")
        return -values

    with ThreadPoolExecutor(4) as pool:
        try:
            job.run(fail_selected, pool, batch_tasks=16)
        except PreparationError as exc:
            failure = dict(failed_tasks=exc.tasks, checkpoint=exc.snapshot_id)
        else:
            raise AssertionError("Seed must exercise worker failure")
    pending = sum(1 for _ in job.pending())
    resumed = Preparation.resume(
        location, branch="recovery", job="recovery", operation="negative-v1", parameters={}
    )
    with ThreadPoolExecutor(4) as pool:
        retried = resumed.run(np.negative, pool, batch_tasks=16)
    published = resumed.publish()
    actual = hashlib.sha256()
    reader = SnapshotReadSession(published)
    for frame in range(args.frames):
        actual.update(memoryview(reader.read("recovered", (frame,) + (slice(None),) * 3)).cast("B"))
    assert actual.digest() == expected.digest()
    (args.root / "recovery.json").write_text(
        json.dumps(
            dict(
                **failure,
                pending_after_failure=pending,
                tasks_run_after_resume=retried,
                total_tasks=job.grid.count,
                snapshot_id=published.snapshot_id,
                verified_sha256=actual.hexdigest(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
