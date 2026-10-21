# SPDX-License-Identifier: Apache-2.0
"""Byte-verified, spawned-worker scans of completed ArrayRoute fixtures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .permutation import Permutation
from .session import LocalReadSession


def _worker(root: str, output: str, settings: dict, rank: int) -> dict:
    started = time.perf_counter()
    directory = Path(output)
    fixture = json.loads((Path(root) / "fixture.json").read_text())
    order = Permutation(fixture["shape"][0], seed=settings["seed"])
    session = LocalReadSession(
        settings["engine"],
        buffer_bytes=settings["buffer_bytes"],
        max_open_arrays=1,
        tensorstore_threads=settings["threads_per_worker"],
    )
    reader = session.open(Path(root) / f"{settings['layout']}.zarr", "")
    frames, logical_bytes, read_seconds, verify_seconds = 0, 0, 0.0, 0.0
    with (directory / f"rank-{rank}.jsonl").open("x") as log:
        for position in range(rank, len(order), settings["workers"]):
            if (directory / "CANCELLED").exists():
                raise InterruptedError("Another scan worker failed or the scan was cancelled")
            frame = order[position]
            before = time.perf_counter()
            values = reader.read((frame,) + (slice(None),) * (len(fixture["shape"]) - 1))
            read_s = time.perf_counter() - before
            before = time.perf_counter()
            digest = hashlib.sha256(values.tobytes()).hexdigest()
            if digest != fixture["frame_sha256"][frame]:
                raise ValueError(f"Corrupt frame {frame} in {settings['layout']}")
            verify_s = time.perf_counter() - before
            record = {
                "position": position,
                "frame": frame,
                "bytes": values.nbytes,
                "read_s": read_s,
                "verify_s": verify_s,
                "sha256": digest,
            }
            frames += 1
            logical_bytes += values.nbytes
            read_seconds += read_s
            verify_seconds += verify_s
            log.write(json.dumps(record) + "\n")
            log.flush()
            del values
    return {
        "rank": rank,
        "frames": frames,
        "verified_logical_bytes": logical_bytes,
        "read_seconds": read_seconds,
        "verify_seconds": verify_seconds,
        "worker_seconds": time.perf_counter() - started,
    }


def scan_parallel(
    root: Path,
    output: Path,
    *,
    engine: str = "tensorstore",
    layout: str = "sharded",
    workers: int = 2,
    threads_per_worker: int = 2,
    buffer_bytes: int = 256 << 20,
    seed: int = 991,
) -> dict:
    """Scan a fixed fixture once, without padded samples or a global shuffled index.

    Processes use spawn. The reported aggregate rate includes startup, metadata, reads,
    SHA-256 verification and per-frame evidence writes. It is logical throughput, not
    physical network traffic. Each process has its own buffer and engine thread limits.
    """
    if any(type(v) is not int or v < 1 for v in (workers, threads_per_worker, buffer_bytes)):
        raise ValueError("Worker, thread and buffer limits must be positive integers")
    LocalReadSession(engine, buffer_bytes=buffer_bytes, tensorstore_threads=threads_per_worker)
    root, output = root.resolve(), output.resolve()
    receipt = (root / "fixture.json").read_bytes()
    fixture = json.loads(receipt)
    if layout not in {"frame", "brick", "sharded"} or layout not in fixture["layouts"]:
        raise ValueError("Layout missing from completed fixture")
    frames = fixture["shape"][0]
    Permutation(frames, seed=seed)
    if frames < workers or len(fixture["frame_sha256"]) != frames:
        raise ValueError("Need at least one frame per worker and a complete hash manifest")
    settings = {
        "engine": engine,
        "layout": layout,
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "buffer_bytes": buffer_bytes,
        "seed": seed,
        "permutation": Permutation.algorithm,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "settings.json").write_text(json.dumps(settings, indent=2))
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        pending = [
            pool.submit(_worker, str(root), str(output), settings, r) for r in range(workers)
        ]
        try:
            for task in as_completed(pending):
                results.append(task.result())
        except BaseException:
            (output / "CANCELLED").write_text("Scan interrupted or a worker failed\n")
            for task in pending:
                task.cancel()
            raise
    elapsed = time.perf_counter() - started
    verified = sum(result["verified_logical_bytes"] for result in results)
    if sum(result["frames"] for result in results) != frames:
        raise ValueError("Incomplete frame coverage")
    if verified != fixture["logical_bytes_per_layout"]:
        raise ValueError("Verified bytes disagree with the completed fixture")
    if (
        hashlib.sha256((root / "fixture.json").read_bytes()).digest()
        != hashlib.sha256(receipt).digest()
    ):
        raise ValueError("Fixture manifest changed during the scan")
    packages = ("numpy", "arrayroute", {"zarr-python": "zarr"}.get(engine, engine))
    report = {
        "settings": settings,
        "frames": frames,
        "verified_logical_bytes": verified,
        "elapsed_seconds": elapsed,
        "verified_logical_GB_s": verified / elapsed / 1e9,
        "workers": sorted(results, key=lambda result: result["rank"]),
        "fixture_sha256": hashlib.sha256(receipt).hexdigest(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "environment": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "versions": {p: importlib.metadata.version(p) for p in packages},
            "threads": {k: os.getenv(k) for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS")},
        },
        "method": "One host; spawn workers; disjoint permutation positions; "
        "every frame SHA-256 checked. "
        "Wall time includes spawn, metadata, reads, verification and evidence writes. "
        "OS cache uncontrolled; physical transport not measured. "
        "TensorStore cache disabled; its copy/I/O pools each use threads_per_worker. "
        "Other engines retain process-level thread policies. "
        "Memory includes workers times reader budget plus hash copies and backend scratch.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--engine", choices=["tensorstore", "zarr-python", "zarrista"], default="tensorstore"
    )
    parser.add_argument("--layout", choices=["frame", "brick", "sharded"], default="sharded")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--buffer-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=991)
    args = parser.parse_args()
    report = scan_parallel(
        args.root,
        args.output,
        engine=args.engine,
        layout=args.layout,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        buffer_bytes=args.buffer_mib << 20,
        seed=args.seed,
    )
    if not math.isfinite(report["verified_logical_GB_s"]):
        raise ArithmeticError("Nonfinite throughput measurement")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
