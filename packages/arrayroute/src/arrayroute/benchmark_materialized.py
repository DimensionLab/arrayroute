# SPDX-License-Identifier: Apache-2.0
"""Verify the synthetic layout benchmark without waiting for file inventory.

Regenerate each frame from its seed. This is deliberately specific to the fixture
in ``arrayroute.benchmark``; it is not a verifier for arbitrary scientific data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from .backends import TensorStoreBackend, ZarrBackend, ZarristaBackend
from .execution import ProcessingCancelledError, ordered_map
from .permutation import Permutation
from .reader import ArrayReader, Backend


def reference_frame(frame: int, edge: int) -> np.ndarray:
    rng = np.random.default_rng(83491 + frame)
    value = rng.standard_normal((edge, edge, edge, 3), dtype=np.float32)
    x = np.arange(edge, dtype=np.float32) / edge
    value *= np.float32(0.03)
    value += np.sin(x[:, None, None, None] * 12 + frame * 0.1)
    return value


def digest(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def open_backend(path: Path, engine: str, *, read_threads: int = 8) -> Backend:
    if engine == "zarr-python":
        import zarr

        return ZarrBackend(zarr.open_array(path, mode="r"))
    if engine == "zarrista":
        import zarrista

        return ZarristaBackend(zarrista.Array.open(zarrista.store.FilesystemStore(str(path))))
    if engine == "tensorstore":
        import tensorstore as ts

        return TensorStoreBackend(
            ts.open(
                {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path) + "/"}},
                read=True,
                context=ts.Context(
                    {
                        "cache_pool": {"total_bytes_limit": 0},
                        "data_copy_concurrency": {"limit": read_threads},
                        "file_io_concurrency": {"limit": read_threads},
                    }
                ),
            ).result()
        )
    raise ValueError("Unknown engine")


def scan(
    root: Path,
    output: Path,
    *,
    engine: str,
    layout: str = "sharded",
    deadline: float | None = None,
    limit: int | None = None,
    workers: int = 1,
) -> dict:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if layout not in {"frame", "brick", "sharded"}:
        raise ValueError("Unknown layout")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError("deadline must be finite")
    progress_bytes = (root / "progress.json").read_bytes()
    progress = json.loads(progress_bytes)
    backend = open_backend(root / f"{layout}.zarr", engine)
    shape = backend.spec.shape
    if (
        len(shape) != 5
        or shape[0] < 4
        or shape[1] < 32
        or shape[1] % 32
        or shape[1:4] != (shape[1],) * 3
        or shape[4] != 3
        or backend.dtype != np.dtype("float32")
        or progress.get("frames_written") != shape[0]
        or progress.get("frames_total") != shape[0]
        or progress.get("logical_bytes_per_layout") != math.prod(shape) * 4
    ):
        raise ValueError("Fixture is incomplete or does not match the synthetic generator")
    per_worker_buffer = (256 << 20) // workers
    if math.prod(shape[1:]) * 4 >= per_worker_buffer:
        raise ValueError("Workers exceed the shared read-buffer budget for one frame each")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source-progress.json").write_bytes(progress_bytes)
    versions = {}
    for package in ("numpy", "zarr", "zarrista", "tensorstore", "arrayroute"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {
        "engine": engine,
        "layout": layout,
        "shape": shape,
        "generator": "arrayroute-broadband-83491-v1",
        "order": {"algorithm": Permutation.algorithm, "seed": 991, "epoch": 0},
        "state": "running",
        "complete": False,
        "frames_verified": 0,
        "verified_logical_bytes": 0,
        "read_seconds": 0.0,
        "reference_seconds": 0.0,
        "hash_seconds": 0.0,
        "started_at": time.time(),
        "deadline": deadline,
        "limit": limit,
        "versions": versions,
        "platform": platform.platform(),
        "cache_state": "OS uncontrolled; one traversal in seeded permuted order",
        "memory_policy": (
            "Shared read-buffer budget plus one reference frame per worker; "
            "native scratch/cache additional"
        ),
        "workers": workers,
        "max_in_flight_frames": workers,
        "read_threads_per_worker": max(1, 8 // workers),
        "per_worker_buffer_bytes": per_worker_buffer,
        "reference_bytes_upper_bound": workers * math.prod(shape[1:]) * 4,
        "timing_scope": "Component seconds are sums across workers; elapsed_seconds is wall time",
        "buffer_bytes": 256 << 20,
    }
    started = time.perf_counter()

    def save() -> None:
        report["elapsed_seconds"] = time.perf_counter() - started
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "report.json")

    def make_verifier():
        reader = ArrayReader(
            open_backend(root / f"{layout}.zarr", engine, read_threads=max(1, 8 // workers)),
            buffer_bytes=per_worker_buffer,
        )

        def verify(frame: int) -> dict:
            start = time.perf_counter()
            value = reader.read((frame,) + (slice(None),) * 4)
            read_s = time.perf_counter() - start
            start = time.perf_counter()
            expected = reference_frame(frame, shape[1])
            reference_s = time.perf_counter() - start
            start = time.perf_counter()
            expected_hash, actual_hash = digest(expected), digest(value)
            hash_s = time.perf_counter() - start
            valid = value.shape == expected.shape and value.dtype == expected.dtype
            valid = valid and actual_hash == expected_hash
            return {
                "frame": frame, "bytes": value.nbytes, "read_s": read_s,
                "reference_s": reference_s, "hash_s": hash_s,
                "expected_sha256": expected_hash, "actual_sha256": actual_hash,
                "verified": valid,
            }

        return verify

    def expired() -> bool:
        return deadline is not None and time.time() >= deadline

    save()
    try:
        with (output / "frames.jsonl").open("x") as log:
            stop = min(shape[0], limit) if limit is not None else shape[0]
            indices = Permutation(shape[0], seed=991).indices(stop=stop)
            try:
                with ordered_map(make_verifier, indices, workers=workers,
                                 cancelled=expired) as results:
                    for row in results:
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        if not row["verified"]:
                            raise ValueError(
                                f"Frame {row['frame']} differs from "
                                "independently regenerated values"
                            )
                        report["frames_verified"] += 1
                        report["verified_logical_bytes"] += row["bytes"]
                        report["read_seconds"] += row["read_s"]
                        report["reference_seconds"] += row["reference_s"]
                        report["hash_seconds"] += row["hash_s"]
                        if report["frames_verified"] % 32 == 0:
                            save()
            except ProcessingCancelledError:
                if not expired():
                    raise
                report["state"] = "deadline"
            else:
                report["complete"] = report["frames_verified"] == shape[0]
                report["state"] = "complete" if report["complete"] else "limited"
    except BaseException as exc:
        report["state"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_at"] = time.time()
        seconds = report["read_seconds"]
        report["aggregate_read_GB_s"] = (
            report["verified_logical_bytes"] / seconds / 1e9 if seconds else None
        )
        report["verified_wall_GB_s"] = (
            report["verified_logical_bytes"] / (time.perf_counter() - started) / 1e9
        )
        save()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--engine", choices=["zarr-python", "zarrista", "tensorstore"], required=True
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="Bounded concurrent independent frame verifiers")
    parser.add_argument("--layout", choices=["frame", "brick", "sharded"], default="sharded")
    parser.add_argument(
        "--deadline", type=float, help="Unix seconds; stop before starting a new frame"
    )
    parser.add_argument(
        "--limit", type=int, help="Pilot only; never marks an incomplete scan complete"
    )
    args = parser.parse_args()
    report = scan(**vars(args))
    print(json.dumps(report, indent=2))
    if report["state"] == "deadline":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
