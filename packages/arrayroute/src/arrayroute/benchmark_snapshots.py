# SPDX-License-Identifier: Apache-2.0
"""Fresh-process snapshot read overhead and one-chunk revision storage experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def worker(
    root: Path, mode: str, concurrency: int, range_request_bytes: int | None = None
) -> None:
    from .session import LocalReadSession
    from .snapshot import FilesystemRepository, Snapshot, SnapshotReadSession

    fixture = json.loads((root / "fixture.json").read_text())
    start = time.perf_counter()
    effective_range_bytes = None
    if mode == "snapshot":
        session = SnapshotReadSession(
            Snapshot(FilesystemRepository(str(root / "repository")), fixture["snapshot_id"]),
            concurrency=concurrency,
            range_request_bytes=range_request_bytes,
        )
        reader = session.open("velocity")
        effective_range_bytes = session.effective_range_request_bytes
    else:
        reader = LocalReadSession(mode).open(root / "plain.zarr", "velocity")
    opened = time.perf_counter() - start
    checks, seconds = [], 0.0
    for t in fixture["order"]:
        start = time.perf_counter()
        frame = reader.read((t,) + (slice(None),) * 4)
        seconds += time.perf_counter() - start
        checks.append(hashlib.sha256(frame.tobytes()).hexdigest())
        del frame
    assert checks == [fixture["hashes"][t] for t in fixture["order"]]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "mode": mode,
                "open_s": opened,
                "read_s": seconds,
                "process_peak_rss_bytes": rss * (1 if sys.platform == "darwin" else 1024),
                "verified_frames": len(checks),
                "verified": True,
                "snapshot_concurrency": concurrency if mode == "snapshot" else None,
                "range_request_bytes": range_request_bytes if mode == "snapshot" else None,
                "effective_range_request_bytes": effective_range_bytes,
            }
        )
    )


def experiment(
    root: Path, frames: int, edge: int, repeats: int, concurrency: int, chunk_edge: int = 32
) -> None:
    import icechunk as ic
    import zarr
    from zarr.codecs import BloscCodec

    root.mkdir(parents=True, exist_ok=False)
    repo = ic.Repository.create(ic.local_filesystem_storage(str(root / "repository")))
    session = repo.writable_session("main")
    shape, chunks = (frames, edge, edge, edge, 3), (1, chunk_edge, chunk_edge, chunk_edge, 3)
    options = {
        "shape": shape,
        "chunks": chunks,
        "dtype": "float32",
        "compressors": [BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")],
    }
    plain = zarr.open_group(root / "plain.zarr", mode="w").create_array("velocity", **options)
    versioned = zarr.group(session.store).create_array("velocity", **options)
    hashes, rng = [], np.random.default_rng(841)
    for t in range(frames):
        value = rng.standard_normal(shape[1:], dtype=np.float32)
        plain[t] = value
        versioned[t] = value
        hashes.append(hashlib.sha256(value.tobytes()).hexdigest())
    sid = session.commit("initial synthetic field")
    size = lambda p: sum(f.stat().st_size for f in p.rglob("*") if f.is_file())  # noqa: E731
    first_bytes = size(root / "repository")
    change = repo.writable_session("main")
    changed = zarr.open_array(change.store, path="velocity", mode="a")
    changed[0, :chunk_edge, :chunk_edge, :chunk_edge, :] = 1
    next_sid = change.commit("change one spatial chunk")
    second_bytes = size(root / "repository")
    order = list(range(frames))
    random.Random(27).shuffle(order)
    fixture = {
        "shape": shape,
        "chunks": chunks,
        "logical_bytes": int(np.prod(shape)) * 4,
        "hashes": hashes,
        "order": order,
        "snapshot_id": sid,
        "new_snapshot_id": next_sid,
        "first_repository_bytes": first_bytes,
        "second_repository_bytes": second_bytes,
        "plain_bytes": size(root / "plain.zarr"),
        "array_metadata": plain.metadata.to_dict(),
        "versions": {
            n: importlib.metadata.version(n)
            for n in ["arrayroute", "icechunk", "zarr", "tensorstore", "numpy"]
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "cache": "fresh processes; warm/uncontrolled OS cache",
        "snapshot_concurrency": concurrency,
        "timing": "reader open and frame reads; hashes excluded; RSS includes whole process",
    }
    (root / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    for source in [Path(__file__), Path(__file__).with_name("snapshot.py")]:
        (root / source.name).write_bytes(source.read_bytes())
    jobs = [
        (mode, repeat)
        for mode in ["zarr-python", "tensorstore", "snapshot"]
        for repeat in range(repeats)
    ]
    random.Random(81).shuffle(jobs)
    for mode, repeat in jobs:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "arrayroute.benchmark_snapshots",
                str(root),
                "--worker",
                mode,
                "--concurrency",
                str(concurrency),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        record = {**json.loads(result.stdout), "repeat": repeat}
        with (root / "measurements.jsonl").open("a") as out:
            out.write(json.dumps(record) + "\n")
        (root / f"{mode}-{repeat}.stderr").write_text(result.stderr)
        print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--edge", type=int, default=96)
    parser.add_argument("--chunk-edge", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--range-request-bytes", type=int)
    parser.add_argument("--worker", choices=["zarr-python", "tensorstore", "snapshot"])
    args = parser.parse_args()
    if min(args.frames, args.edge, args.chunk_edge, args.repeats, args.concurrency) < 1:
        parser.error("Positive frames, edge, repeats and concurrency are required")
    if args.range_request_bytes is not None and args.worker != "snapshot":
        parser.error("--range-request-bytes applies to --worker snapshot")
    if args.worker:
        worker(args.root, args.worker, args.concurrency, args.range_request_bytes)
    else:
        experiment(
            args.root, args.frames, args.edge, args.repeats, args.concurrency, args.chunk_edge
        )


if __name__ == "__main__":
    main()
