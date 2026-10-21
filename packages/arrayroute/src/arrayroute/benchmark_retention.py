# SPDX-License-Identifier: Apache-2.0
"""Measure durable consumer reference overhead on a fresh, disposable local repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .retention import RetentionManager
from .snapshot import FilesystemRepository, Snapshot, SnapshotReadSession


def _timings(samples: list[float]) -> dict:
    return {
        "count": len(samples),
        "total_seconds": sum(samples),
        "median_seconds": float(np.median(samples)),
        "p95_seconds": float(np.percentile(samples, 95)),
        "max_seconds": max(samples),
    }


def benchmark(root: Path, *, consumers: int = 128) -> dict:
    """All destructive tests stay in a new repository; existing paths are refused."""
    import icechunk as ic
    import zarr

    if not 2 <= consumers <= 100_000:
        raise ValueError("Consumers must be between 2 and 100000")
    root.mkdir(parents=True, exist_ok=False)
    location = FilesystemRepository(str(root))
    repo = ic.Repository.create(
        ic.local_filesystem_storage(location.path),
        config=ic.RepositoryConfig(inline_chunk_threshold_bytes=0),
    )
    values = np.random.default_rng(914).standard_normal((256, 1024), dtype=np.float32)
    session = repo.writable_session("main")
    zarr.group(session.store).create_array("u", data=values, chunks=(32, 128))
    original = Snapshot(location, session.commit("original"))
    replacement = repo.writable_session("main")
    zarr.open_array(replacement.store, path="u", mode="r+")[:] = -values
    replacement.commit("new simulation result")
    manager = RetentionManager(location)
    pins, acquired, released = [], [], []
    for _ in range(consumers):
        start = time.perf_counter()
        pins.append(manager.retain(original, purpose="training"))
        acquired.append(time.perf_counter() - start)
    start = time.perf_counter()
    inventory = list(manager.pins())
    inventory_seconds = time.perf_counter() - start
    if set(inventory) != set(pins):
        raise AssertionError("Reference inventory differs from the created consumers")
    files = [p for p in root.rglob("*") if p.is_file()]
    stored_bytes = sum(p.stat().st_size for p in files)
    half = consumers // 2
    for pin in pins[:half]:
        start = time.perf_counter()
        if not manager.release(pin):
            raise AssertionError("Consumer was already released")
        released.append(time.perf_counter() - start)
    # No writer remains in this isolated repository. Never use a current-time cutoff
    # against production storage while sessions may still own uncommitted objects.
    cutoff = datetime.now(UTC)
    repo.expire_snapshots(cutoff, delete_expired_branches=False, delete_expired_tags=False)
    protected_gc = repo.garbage_collect(cutoff)
    actual = SnapshotReadSession(original).read("u", (slice(None), slice(None)))
    expected_hash = hashlib.sha256(values.tobytes()).hexdigest()
    actual_hash = hashlib.sha256(actual.tobytes()).hexdigest()
    if expected_hash != actual_hash:
        raise AssertionError("Retained values changed during cleanup")
    for pin in pins[half:]:
        start = time.perf_counter()
        manager.release(pin)
        released.append(time.perf_counter() - start)
    cutoff = datetime.now(UTC)
    repo.expire_snapshots(cutoff, delete_expired_branches=False, delete_expired_tags=False)
    reclaimed = repo.garbage_collect(cutoff)
    if reclaimed.chunks_deleted < 1 or reclaimed.objects_failed_to_delete:
        raise AssertionError("Unreferenced test chunks were not fully reclaimed")
    try:
        SnapshotReadSession(original).open("u")
    except ic.SnapshotNotFoundError:
        pass
    else:
        raise AssertionError("Released snapshot still exists")
    return {
        "protocol": "local-retention-v1",
        "unique_input_bytes": values.nbytes,
        "consumers": consumers,
        "acquire": _timings(acquired),
        "release": _timings(released),
        "inventory_seconds": inventory_seconds,
        "stored_bytes_with_all_references": stored_bytes,
        "stored_files_with_all_references": len(files),
        "protected_gc_bytes_deleted": protected_gc.bytes_deleted,
        "released_gc_bytes_deleted": reclaimed.bytes_deleted,
        "released_gc_chunks_deleted": reclaimed.chunks_deleted,
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
        "unreferenced_revision_reclaimed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--consumers", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Benchmark output already exists")
    result = benchmark(args.root, consumers=args.consumers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
