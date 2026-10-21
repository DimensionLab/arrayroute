# SPDX-License-Identifier: Apache-2.0
"""Resumable, chunk-owned transforms over pinned Icechunk revisions.

One trusted coordinator owns a work branch. Workers may write disjoint output chunks;
only the coordinator merges them and commits output plus receipts atomically. Work
branches are staging areas, not published datasets. This is not a worker sandbox or
multi-coordinator lease. Use spawn for process executors, never POSIX fork.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Executor
from dataclasses import dataclass
from itertools import islice
from operator import index
from typing import Any

import numpy as np

from .reader import ArrayReader
from .remote import _path
from .snapshot import RepositoryLocation, S3Repository, Snapshot, SnapshotReadSession, _storage


@dataclass(frozen=True)
class Partition:
    number: int
    output: tuple[slice, ...]
    read: tuple[slice, ...]
    core: tuple[slice, ...]


@dataclass(frozen=True)
class PartitionGrid:
    """O(rank) plan; tasks own whole chunks while their read halos may overlap."""

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    task_shape: tuple[int, ...]
    halo: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("shape", "chunks", "task_shape", "halo"):
            object.__setattr__(self, name, tuple(index(v) for v in getattr(self, name)))
        if (
            not self.shape
            or len({len(self.shape), len(self.chunks), len(self.task_shape), len(self.halo)}) != 1
        ):
            raise ValueError("Shape, chunks, task shape and halo must have the same nonzero rank")
        if min(*self.shape, *self.chunks, *self.task_shape) < 1 or min(self.halo) < 0:
            raise ValueError("Dimensions must be positive and halos nonnegative")
        if any(t % c for t, c in zip(self.task_shape, self.chunks, strict=True)):
            raise ValueError("Task dimensions must be multiples of output chunk dimensions")

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple((n + t - 1) // t for n, t in zip(self.shape, self.task_shape, strict=True))

    @property
    def count(self) -> int:
        return math.prod(self.counts)

    def partition(self, number: int) -> Partition:
        number = index(number)
        if not 0 <= number < self.count:
            raise IndexError("Partition is outside the task grid")
        rest = number
        coords = []
        for count in reversed(self.counts):
            rest, coord = divmod(rest, count)
            coords.append(coord)
        output, read, core = [], [], []
        for coord, n, t, h in zip(
            reversed(coords), self.shape, self.task_shape, self.halo, strict=True
        ):
            lo, hi = coord * t, min(n, (coord + 1) * t)
            start, stop = max(0, lo - h), min(n, hi + h)
            output.append(slice(lo, hi))
            read.append(slice(start, stop))
            core.append(slice(lo - start, hi - start))
        return Partition(number, tuple(output), tuple(read), tuple(core))


def _verify_storage(location: RepositoryLocation) -> None:
    if isinstance(location, S3Repository):
        from .s3_writes import verify_s3_writes

        verify_s3_writes(
            location.store,
            allow_http=location.allow_http,
            force_path_style=location.force_path_style,
        )


def _repository(location: RepositoryLocation):
    import icechunk as ic

    return ic.Repository.open(_storage(location), authorize_virtual_chunk_access={})


def _writable_repository(location: RepositoryLocation):
    repo = _repository(location)
    if isinstance(location, S3Repository):
        from .s3_writes import UnsafeObjectStoreError

        settings = repo.config.storage
        # None inherits the S3 backend's safe defaults. An explicit False disables
        # conditional writes even when the endpoint itself passed the capability probe.
        if settings is not None and (
            settings.unsafe_use_conditional_create is False
            or settings.unsafe_use_conditional_update is False
        ):
            raise UnsafeObjectStoreError(
                "Repository configuration disables conditional writes; restore safe settings "
                "before preparing data"
            )
    return repo


def _digest(value: np.ndarray) -> bytes:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).digest()


_worker = threading.local()


def _input_reader(snapshot: Snapshot, source: str, buffer_bytes: int) -> ArrayReader:
    # One pinned source per worker thread. Reuse native indexes across tasks without
    # sharing mutable Python cache state or retaining multiple repositories indefinitely.
    key = (snapshot, source, buffer_bytes)
    if getattr(_worker, "key", None) != key:
        previous = getattr(_worker, "session", None)
        if previous is not None:
            previous.clear()
        session = SnapshotReadSession(snapshot, buffer_bytes=buffer_bytes, max_open_arrays=1)
        reader = session.open(source)
        _worker.session, _worker.reader, _worker.key = session, reader, key
    return _worker.reader


def _transform(
    fork: Any,
    snapshot: Snapshot,
    config: dict,
    part: Partition,
    operator: Callable[[np.ndarray], np.ndarray],
    buffer_bytes: int,
):
    import zarr

    reader = _input_reader(snapshot, config["source"], buffer_bytes)
    request = reader.plan(part.read)
    dtype = np.dtype(config["dtype"])
    expanded_bytes = math.prod(request.output_shape) * dtype.itemsize
    core_bytes = math.prod(s.stop - s.start for s in part.output) * dtype.itemsize
    # Covers planned read buffers, returned transform and contiguous write copy. Operator
    # scratch, native codec caches and runtime allocations need separate worker headroom.
    required = request.buffer_bytes_upper_bound + expanded_bytes + core_bytes
    if required > buffer_bytes:
        raise MemoryError(
            f"Partition {part.number} needs {required} planned bytes; budget {buffer_bytes}"
        )
    values = reader.read(part.read)
    transformed = operator(values)
    if not isinstance(transformed, np.ndarray) or transformed.shape != values.shape:
        raise ValueError("Operator must return an ndarray with the expanded input shape")
    if transformed.dtype != dtype:
        raise ValueError("Operator returned a dtype different from the declared output dtype")
    core = np.ascontiguousarray(transformed[part.core])
    digest = _digest(core)
    zarr.open_array(fork.store, path=config["output"], mode="r+")[part.output] = core
    return fork, part.number, digest


class PreparationError(RuntimeError):
    """Successful tasks are checkpointed; reopen the work branch to retry missing tasks."""

    def __init__(self, tasks: list[int], snapshot_id: str):
        self.tasks, self.snapshot_id = tasks, snapshot_id
        super().__init__(f"Preparation failed for tasks {tasks}; checkpoint {snapshot_id}")


class Preparation:
    """Shape-preserving numerical transform with explicit operation provenance.

    The caller supplies trusted Python code and its version/configuration. These are not
    automatically fingerprinted. Operators return the full halo-expanded input shape;
    the framework crops the core. Halo size and physical boundary treatment remain the
    operator author's responsibility. New outputs are unsharded to enforce chunk ownership.
    """

    def __init__(self, location: RepositoryLocation, branch: str, job: str, config: dict):
        self.location, self.branch, self.job, self.config = location, branch, job, config
        self.grid = PartitionGrid(
            *(tuple(config[k]) for k in ("shape", "chunks", "task_shape", "halo"))
        )
        self.snapshot = Snapshot(location, config["input_snapshot"])
        self._writes_checked = False

    def __getstate__(self) -> dict:
        return {**vars(self), "_writes_checked": False}

    def _check_writes(self) -> None:
        if not self._writes_checked:
            _verify_storage(self.location)
            self._writes_checked = True

    @property
    def path(self) -> str:
        return f"_arrayroute/preparations/{self.job}"

    @classmethod
    def create(
        cls,
        snapshot: Snapshot,
        *,
        branch: str,
        job: str,
        source: str,
        output: str,
        chunks: tuple[int, ...],
        task_shape: tuple[int, ...],
        halo: tuple[int, ...],
        dtype: str,
        operation: str,
        parameters: dict,
        attributes: dict,
    ) -> Preparation:
        import zarr

        if not re.fullmatch(r"[a-zA-Z0-9_-]+", job) or not operation.strip():
            raise ValueError("A simple job identifier and versioned operation name are required")
        source, output = _path(source), _path(output)
        if any(not p or p.split("/")[0] == "_arrayroute" for p in (source, output)):
            raise ValueError("Source/output must be named arrays outside _arrayroute")
        output_dtype = np.dtype(dtype)
        if output_dtype.kind not in "biufc":
            raise ValueError("Only fixed-size numerical output dtypes are supported")
        _verify_storage(snapshot.location)
        reader = SnapshotReadSession(snapshot).open(source)
        grid = PartitionGrid(reader.backend.spec.shape, chunks, task_shape, halo)
        config = json.loads(
            json.dumps(
                dict(
                    version=1,
                    input_snapshot=snapshot.snapshot_id,
                    source=source,
                    output=output,
                    shape=grid.shape,
                    chunks=grid.chunks,
                    task_shape=grid.task_shape,
                    halo=grid.halo,
                    dtype=output_dtype.str,
                    operation=operation,
                    parameters=parameters,
                ),
                allow_nan=False,
            )
        )
        attributes = json.loads(json.dumps(attributes, allow_nan=False))
        repo = _writable_repository(snapshot.location)
        # Reserve a new branch; never reset an existing branch or overwrite an array.
        repo.create_branch(branch, snapshot.snapshot_id)
        session = repo.writable_session(branch)
        group = zarr.open_group(session.store, mode="a")
        group.create_array(
            output, shape=grid.shape, chunks=grid.chunks, dtype=output_dtype, attributes=attributes
        )
        state = group.create_group(f"_arrayroute/preparations/{job}")
        state.attrs.update({"config": config, "state": "preparing"})
        state.create_array(
            "receipts", shape=(grid.count, 33), chunks=(1024, 33), dtype="u1", fill_value=0
        )
        session.commit(f"Initialize preparation {job}")
        prepared = cls(snapshot.location, branch, job, config)
        prepared._writes_checked = True
        return prepared

    @classmethod
    def resume(
        cls,
        location: RepositoryLocation,
        *,
        branch: str,
        job: str,
        operation: str,
        parameters: dict,
    ) -> Preparation:
        import zarr

        if not re.fullmatch(r"[a-zA-Z0-9_-]+", job):
            raise ValueError("Invalid job identifier")
        session = _repository(location).readonly_session(branch=branch)
        group = zarr.open_group(session.store, path=f"_arrayroute/preparations/{job}", mode="r")
        raw = group.attrs["config"]
        if not isinstance(raw, dict):
            raise ValueError("Invalid preparation configuration")
        config = dict(raw)
        if (
            config["version"] != 1
            or config["operation"] != operation
            or config["parameters"] != parameters
        ):
            raise ValueError("Preparation operation/version/parameters do not match")
        return cls(location, branch, job, config)

    def _receipts(self, session: Any):
        import zarr

        return zarr.open_array(session.store, path=f"{self.path}/receipts", mode="r")

    def pending(self) -> Iterator[int]:
        """Scan progress in bounded blocks; never materialize all task IDs."""
        session = _repository(self.location).readonly_session(branch=self.branch)
        receipts = self._receipts(session)
        for start in range(0, self.grid.count, 1024):
            flags = receipts[start : start + 1024, 0]
            for offset in np.flatnonzero(flags == 0):
                yield start + int(offset)

    def run(
        self,
        operator: Callable[[np.ndarray], np.ndarray],
        executor: Executor,
        *,
        batch_tasks: int = 8,
        buffer_bytes: int = 128 << 20,
        cancelled: Callable[[], bool] | None = None,
    ) -> int:
        """Checkpoint bounded batches. Cancellation is observed between batches.

        A failed batch commits its successful workers before raising PreparationError.
        A lost coordinator can reopen the branch: uncommitted work is simply retried.
        Use one coordinator per branch; executor/worker lifecycle belongs to the caller.
        """
        import zarr

        batch_tasks, buffer_bytes = index(batch_tasks), index(buffer_bytes)
        if min(batch_tasks, buffer_bytes) < 1:
            raise ValueError("Batch size and planned buffer budget must be positive")
        self._check_writes()
        repo = _writable_repository(self.location)
        pending = self.pending()
        completed = 0
        while batch := list(islice(pending, batch_tasks)):
            if cancelled and cancelled():
                raise InterruptedError("Preparation cancelled between checkpoints")
            session = repo.writable_session(self.branch)
            futures = []
            failed = []
            first_error: Exception | None = None
            # Fork all tasks before merging: each starts from the same committed state.
            for number in batch:
                try:
                    future = executor.submit(
                        _transform,
                        session.fork(),
                        self.snapshot,
                        self.config,
                        self.grid.partition(number),
                        operator,
                        buffer_bytes,
                    )
                    futures.append((number, future))
                except Exception as exc:
                    failed.append(number)
                    first_error = first_error or exc
            results = []
            for number, future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    failed.append(number)
                    first_error = first_error or exc
            if results:
                session.merge(*(result[0] for result in results))
                receipts = zarr.open_array(session.store, path=f"{self.path}/receipts", mode="r+")
                blocks: dict[int, list[tuple[int, bytes]]] = {}
                for _, number, digest in results:
                    blocks.setdefault(number // 1024, []).append((number, digest))
                for block, entries in blocks.items():
                    start = block * 1024
                    rows = np.asarray(receipts[start : start + 1024, :])
                    for number, digest in entries:
                        rows[number - start, :] = np.frombuffer(b"\x01" + digest, dtype="u1")
                    receipts[start : start + len(rows), :] = rows
                session.commit(f"Prepare {self.job}: {len(results)} tasks")
                completed += len(results)
            if failed:
                raise PreparationError(failed, repo.lookup_branch(self.branch)) from first_error
        return completed

    def publish(self, *, buffer_bytes: int = 128 << 20) -> Snapshot:
        """Require all receipts and reread/hash each output core before marking complete.

        Returns an immutable revision; no mutable public branch is switched. Retaining
        this revision against garbage collection is the caller's responsibility.
        """
        import zarr

        self._check_writes()
        repo = _writable_repository(self.location)
        session = repo.writable_session(self.branch)
        receipts = self._receipts(session)
        output = SnapshotReadSession(
            Snapshot(self.location, session.snapshot_id), buffer_bytes=buffer_bytes
        ).open(self.config["output"])
        for start in range(0, self.grid.count, 1024):
            rows = np.asarray(receipts[start : start + 1024, :])
            for offset, row in enumerate(rows):
                number = start + offset
                if row[0] != 1:
                    raise ValueError(f"Preparation incomplete: task {number} has no receipt")
                selection = self.grid.partition(number).output
                plan = output.plan(selection)
                if plan.buffer_bytes_upper_bound + plan.output_bytes > buffer_bytes:
                    raise MemoryError("Publication verification exceeds planned buffer budget")
                if _digest(output.read(selection)) != row[1:].tobytes():
                    raise ValueError(f"Output digest mismatch for task {number}")
        group = zarr.open_group(session.store, path=self.path, mode="a")
        if group.attrs["state"] == "complete":
            return Snapshot(self.location, session.snapshot_id)
        group.attrs["state"] = "complete"
        return Snapshot(self.location, session.commit(f"Publish verified preparation {self.job}"))
