# SPDX-License-Identifier: Apache-2.0
"""Isolated codec scheduling comparison on a byte-verified snapshot fixture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
from pathlib import Path


def worker(fixture: Path, mode: str, pipeline: str, batch: int, workers: int) -> None:
    import zarr

    from .benchmark_snapshots import worker as read_worker

    # Configuration is process-global. Each measurement runs in its own process;
    # importing this benchmark never changes a caller's arrays or thread settings.
    zarr.config.set(
        {
            "codec_pipeline.path": f"zarr.core.codec_pipeline.{pipeline}CodecPipeline",
            "codec_pipeline.batch_size": batch,
            "codec_pipeline.max_workers": workers,
        }
    )
    read_worker(fixture, mode, concurrency=8)


def experiment(fixture: Path, output: Path, repeats: int) -> None:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    receipt = fixture / "fixture.json"
    metadata = json.loads(receipt.read_text())
    if not metadata.get("hashes") or not metadata.get("snapshot_id"):
        raise ValueError("Expected a fixture made by benchmark-snapshots")
    output.mkdir(parents=True, exist_ok=False)
    (output / "fixture.json").write_bytes(receipt.read_bytes())
    source = Path(__file__)
    (output / source.name).write_bytes(source.read_bytes())
    read_source = source.with_name("benchmark_snapshots.py")
    (output / read_source.name).write_bytes(read_source.read_bytes())
    (output / "snapshot.py").write_bytes(source.with_name("snapshot.py").read_bytes())
    configurations = [
        (mode, pipeline, batch, workers)
        for mode in ("zarr-python", "snapshot")
        for pipeline, batch, workers in (
            ("Batched", 1, 8),
            ("Batched", 8, 8),
            ("Batched", 32, 8),
            ("Batched", 128, 8),
            ("Fused", 1, 1),
            ("Fused", 1, 8),
        )
    ] + [("tensorstore", "Batched", 1, 8)]
    jobs = [(config, repeat) for config in configurations for repeat in range(repeats)]
    random.Random(537).shuffle(jobs)
    provenance = {
        "fixture_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("arrayroute", "zarr", "icechunk", "numpy", "tensorstore")
        },
        "cache": "fresh processes; warm/uncontrolled OS cache",
        "timing": "frame reads exclude hashes; whole-process peak RSS; open excludes Zarr import",
        "open_timing_limit": "other optional imports remain mode-dependent; compare read_s",
        "settings": "pipeline settings are process-global and isolated per measurement",
        "repeats": repeats,
        "jobs": len(jobs),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for (mode, pipeline, batch, workers), repeat in jobs:
        label = f"{mode}-{pipeline}-{batch}-{workers}-{repeat}"
        result = subprocess.run(
            [
                sys.executable, "-m", "arrayroute.benchmark_snapshot_pipelines",
                "--fixture", str(fixture), "--worker", "--mode", mode,
                "--pipeline", pipeline, "--batch", str(batch), "--workers", str(workers),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        (output / f"{label}.stderr").write_text(result.stderr)
        (output / f"{label}.stdout").write_text(result.stdout)
        result.check_returncode()
        value = json.loads(result.stdout)
        value.update(pipeline=pipeline, batch_size=batch, workers=workers, repeat=repeat)
        with (output / "measurements.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=["zarr-python", "snapshot", "tensorstore"])
    parser.add_argument("--pipeline", choices=["Batched", "Fused"], default="Batched")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.worker:
        if args.mode is None or min(args.batch, args.workers) < 1:
            parser.error("Worker requires a mode and positive batch/worker limits")
        worker(args.fixture, args.mode, args.pipeline, args.batch, args.workers)
    else:
        if args.output is None:
            parser.error("Specify a new --output directory")
        experiment(args.fixture, args.output, args.repeats)


if __name__ == "__main__":
    main()
