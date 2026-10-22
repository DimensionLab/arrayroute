# SPDX-License-Identifier: Apache-2.0
"""Reproducible local layout/engine benchmark; each measured read is byte-verified."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .backends import TensorStoreBackend, ZarrBackend, ZarristaBackend
from .execution import ordered_map
from .plan import ArraySpec, plan
from .reader import ArrayReader, ReadStats


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def create_fixture(
    root: Path, *, frames: int, edge: int, writer: str = "zarr-python",
    layouts: Sequence[str] = ("frame", "brick", "sharded"), frame_workers: int = 1,
) -> dict:
    import zarr
    from zarr.codecs import ZstdCodec

    if frames < 4 or edge < 32 or edge % 32:
        raise ValueError("Use at least 4 frames and an edge divisible by 32")
    if not layouts or len(set(layouts)) != len(layouts) or set(layouts) - {
        "frame", "brick", "sharded"
    }:
        raise ValueError("Choose unique layouts from frame, brick and sharded")
    if writer not in {"zarr-python", "zarrista"}:
        raise ValueError("Unknown fixture writer")
    if isinstance(frame_workers, bool) or not isinstance(frame_workers, int) or frame_workers < 1:
        raise ValueError("frame_workers must be a positive integer")
    root.mkdir(parents=True, exist_ok=False)
    shape = (frames, edge, edge, edge, 3)
    arrays = {}
    for name, chunks, shards in (
        ("frame", (1, edge, edge, edge, 3), None),
        ("brick", (1, 32, 32, 32, 3), None),
        ("sharded", (1, 32, 32, 32, 3), (1, edge, edge, edge, 3)),
    ):
        if name not in layouts:
            continue
        arrays[name] = zarr.create_array(
            root / f"{name}.zarr",
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype="float32",
            compressors=[ZstdCodec(level=3)],
        )
    hashes = []

    def make_writer():
        if writer == "zarrista":
            import zarrista

            handles = {
                name: zarrista.Array.open(
                    zarrista.store.FilesystemStore(str(root / f"{name}.zarr"))
                )
                for name in arrays
            }
        else:
            handles = {name: zarr.open_array(root / f"{name}.zarr", mode="r+") for name in arrays}

        def write_frame(t: int) -> str:
            # Each task owns whole time shards. No concurrent read-modify-write of a shard.
            rng = np.random.default_rng(83491 + t)
            value = rng.standard_normal(shape[1:], dtype=np.float32)
            x = np.arange(edge, dtype=np.float32) / edge
            value *= np.float32(0.03)
            value += np.sin(x[:, None, None, None] * 12 + t * 0.1)
            sha = _digest(value)
            for array in handles.values():
                array[t : t + 1] = value.reshape((1, *value.shape))
            return sha

        return write_frame

    frame_bytes = edge**3 * 3 * 4
    start = time.perf_counter()
    with ordered_map(make_writer, range(frames), workers=frame_workers) as results:
        for t, sha in enumerate(results):
            hashes.append(sha)
            if t % 32 == 0 or t == frames - 1:
                progress = {
                    "frames_written": t + 1,
                    "frames_total": frames,
                    "logical_bytes_per_layout": (t + 1) * frame_bytes,
                    "selected_layouts": list(arrays),
                    "frame_workers": frame_workers,
                    "elapsed_s": time.perf_counter() - start,
                }
                temporary = root / "progress.tmp"
                temporary.write_text(json.dumps(progress))
                temporary.replace(root / "progress.json")
    receipt = {
        "shape": shape,
        "logical_bytes_per_layout": int(np.prod(shape)) * 4,
        "frame_sha256": hashes,
        "generation_seconds": time.perf_counter() - start,
        "kind": "deterministic broadband synthetic; not CFD; all values materialized",
        "writer": writer,
        "concurrent_layout_writers": 1,
        "frame_workers": frame_workers,
        "max_in_flight_frames": frame_workers,
        "generated_frame_bytes_upper_bound": frame_workers * frame_bytes,
        "memory_scope": "Generated frame arrays only; native codec scratch/cache additional",
        "selected_layouts": list(arrays),
        "layouts": {},
    }
    # Numeric verification must not wait for a multi-million-object inventory.
    (root / "materialization.json").write_text(json.dumps(receipt, indent=2))
    for name, array in arrays.items():
        file_count, file_bytes, allocated_bytes = 0, 0, 0
        for directory, _, names in os.walk(root / f"{name}.zarr"):
            for filename in names:
                file_count += 1
                stat = (Path(directory) / filename).stat()
                file_bytes += stat.st_size
                allocated_bytes += stat.st_blocks * 512
        receipt["layouts"][name] = {
            "chunks": array.chunks,
            "shards": array.shards,
            "physical_file_bytes": file_bytes,
            "allocated_file_bytes": allocated_bytes,
            "files": file_count,
        }
    (root / "fixture.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def scan(root: Path, *, engine: str, layout: str) -> dict:
    """Read and verify every unique frame; keep only one output frame resident."""
    import tensorstore as ts
    import zarr
    import zarrista

    from .reader import Backend

    fixture = json.loads((root / "fixture.json").read_text())
    if layout not in fixture["layouts"]:
        raise ValueError("Layout missing from the completed fixture")
    path = root / f"{layout}.zarr"
    backend: Backend
    if engine == "tensorstore":
        backend = TensorStoreBackend(
            ts.open(
                {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path) + "/"}},
                read=True,
                context=ts.Context(
                    {
                        "cache_pool": {"total_bytes_limit": 0},
                        "data_copy_concurrency": {"limit": 8},
                        "file_io_concurrency": {"limit": 8},
                    }
                ),
            ).result()
        )
    elif engine == "zarrista":
        backend = ZarristaBackend(zarrista.Array.open(zarrista.store.FilesystemStore(str(path))))
    elif engine == "zarr-python":
        backend = ZarrBackend(zarr.open_array(path, mode="r"))
    else:
        raise ValueError("Unknown engine")
    reader, stats = ArrayReader(backend, buffer_bytes=256 << 20), ReadStats()
    started, verified_bytes = time.perf_counter(), 0
    order = list(range(fixture["shape"][0]))
    random.Random(991).shuffle(order)
    record = root / f"scan-{layout}-{engine}.jsonl"
    with record.open("x") as log:
        for frame in order:
            start = time.perf_counter()
            value = reader.read((frame,) + (slice(None),) * 4, stats=stats)
            read_seconds = time.perf_counter() - start
            if _digest(value) != fixture["frame_sha256"][frame]:
                raise ValueError(f"Corrupt frame {frame}")
            verified_bytes += value.nbytes
            log.write(
                json.dumps({"frame": frame, "bytes": value.nbytes, "read_s": read_seconds}) + "\n"
            )
            log.flush()
            del value
    report = {
        "engine": engine,
        "layout": layout,
        "verified_logical_bytes": verified_bytes,
        "frames": len(order),
        "read_seconds": stats.seconds,
        "read_GB_s": verified_bytes / stats.seconds / 1e9,
        "including_sha256_seconds": time.perf_counter() - started,
        "cache_state": "OS uncontrolled; entire dataset traversed in seeded random order",
        "physical_file_bytes": fixture["layouts"][layout]["physical_file_bytes"],
    }
    (root / f"scan-{layout}-{engine}.json").write_text(json.dumps(report, indent=2))
    return report


def benchmark(root: Path, *, repeats: int = 7) -> dict:
    import tensorstore as ts
    import zarr
    import zarrista

    if repeats < 2:
        raise ValueError("At least two repetitions are required")
    fixture = json.loads((root / "fixture.json").read_text())
    frames, edge = fixture["shape"][:2]
    full = slice(None)
    queries = {
        "frame": (frames // 2, full, full, full, full),
        "x-plane": (frames // 2, edge // 2, full, full, full),
        "y-plane": (frames // 2, full, edge // 2, full, full),
        "z-plane": (frames // 2, full, full, edge // 2, full),
        "roi": (frames // 2, slice(5, 29), slice(9, 31), slice(7, 25), full),
        "training-window": (slice(0, 4), slice(0, 32), slice(0, 32), slice(0, 32), full),
    }
    reference = zarr.open_array(root / "frame.zarr", mode="r")
    expected = {
        name: _digest(np.asarray(reference[selection])) for name, selection in queries.items()
    }
    backends = {}
    context = ts.Context(
        {
            "cache_pool": {"total_bytes_limit": 0},
            "data_copy_concurrency": {"limit": 8},
            "file_io_concurrency": {"limit": 8},
        }
    )
    for name in fixture["layouts"]:
        path = root / f"{name}.zarr"
        backends[name, "zarr-python"] = ZarrBackend(zarr.open_array(path, mode="r"))
        backends[name, "tensorstore"] = TensorStoreBackend(
            ts.open(
                {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path) + "/"}},
                read=True,
                context=context,
            ).result()
        )
        backends[name, "zarrista"] = ZarristaBackend(
            zarrista.Array.open(zarrista.store.FilesystemStore(str(path)))
        )
    work = [(layout, engine, query) for layout, engine in backends for query in queries]
    random.Random(73).shuffle(work)
    results = []
    for layout, engine, name in work:
        reader = ArrayReader(backends[layout, engine], buffer_bytes=256 << 20)
        selection, elapsed = queries[name], []
        stats = ReadStats()
        for _ in range(repeats):
            stats = ReadStats()
            start = time.perf_counter()
            value = reader.read(selection, stats=stats)
            elapsed.append(time.perf_counter() - start)
            if _digest(value) != expected[name]:
                raise ValueError(f"Byte mismatch: {layout}/{engine}/{name}")
            del value
        cost = reader.plan(selection)
        results.append(
            {
                "layout": layout,
                "engine": engine,
                "query": name,
                "seconds": elapsed,
                "median_s": float(np.median(elapsed)),
                "p95_s": float(np.percentile(elapsed, 95)),
                "sha256": expected[name],
                "output_bytes": cost.output_bytes,
                "decoded_bytes_upper_bound": cost.decoded_bytes_upper_bound,
                "decode_amplification": cost.decode_amplification,
                "last_read": asdict(stats),
            }
        )
        print(json.dumps(results[-1]), flush=True)
    huge = ArraySpec((1 << 20, 1024, 1024, 256), (1, 64, 64, 64), 4)
    start = time.perf_counter()
    for _ in range(10000):
        plan(huge, (900000, slice(22, 86), slice(40, 104), 12))
    return {
        "schema_version": 1,
        "fixture": fixture,
        "results": results,
        "environment": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("numpy", "zarr", "zarrista", "tensorstore", "arrayroute")
            },
        },
        "cache_policy": "OS cache uncontrolled/warm; TensorStore data cache disabled",
        "transport_bytes": "not instrumented; decoded-byte figures are conservative estimates",
        "verification": "every timed output SHA-256 matches independent Zarr-Python reference",
        "planner_only": {
            "logical_bytes": huge.logical_bytes,
            "physical_bytes": 0,
            "queries": 10000,
            "seconds": time.perf_counter() - start,
        },
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--edge", type=int, default=128)
    parser.add_argument("--frame-workers", type=int, default=1,
                        help="Bounded independent frame writers; each owns its native handles")
    parser.add_argument("--writer", choices=["zarr-python", "zarrista"], default="zarr-python")
    parser.add_argument("--layouts", nargs="+", choices=["frame", "brick", "sharded"],
                        default=["frame", "brick", "sharded"],
                        help="Materialize only these layouts; one copy can be read by all engines")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--create-only", action="store_true")
    parser.add_argument("--scan-engine", choices=["zarr-python", "tensorstore", "zarrista"])
    parser.add_argument("--scan-layout", choices=["frame", "brick", "sharded"], default="sharded")
    args = parser.parse_args()
    if args.create:
        create_fixture(args.root, frames=args.frames, edge=args.edge, writer=args.writer,
                       layouts=args.layouts, frame_workers=args.frame_workers)
    if args.create_only:
        return
    if args.scan_engine:
        print(
            json.dumps(scan(args.root.resolve(), engine=args.scan_engine, layout=args.scan_layout))
        )
        return
    result = benchmark(args.root.resolve(), repeats=args.repeats)
    encoded = json.dumps(result, indent=2)
    (args.root / f"benchmark-{time.time_ns()}.json").write_text(encoded)
    (args.root / "benchmark.json").write_text(encoded)


if __name__ == "__main__":
    main()
