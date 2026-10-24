# SPDX-License-Identifier: Apache-2.0
"""Fresh-process metadata lookup benchmark; no numerical-throughput or PB-scale claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sqlite3
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .catalog import Catalog, write_catalog
from .memory import peak_process_rss_bytes


def record(index: int) -> tuple[str, dict[str, Any]]:
    key = f"case-{index:09d}"
    return key, {
        "id": key,
        "path": f"samples/{key}",
        "shape": [128, 128, 128, 3],
        "fields": [{"name": "velocity", "units": "m/s", "location": "cell", "channels": 3}],
        "time": {"units": "s", "steps": 1001, "dt": 0.002},
        "parameters": {"inlet_velocity_m_s": float(index % 20 + 1)},
        "source_revision": hashlib.sha256(key.encode()).hexdigest(),
    }


def digest(values: list) -> str:
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def worker(root: Path, engine: str, count: int) -> None:
    start = time.perf_counter()
    key = record(count - 1)[0]
    if engine == "json":
        with (root / "manifest.json").open() as stream:
            manifest = json.load(stream)
        page = manifest["samples"][:20]
        selected = next(value for value in manifest["samples"] if value["id"] == key)
    else:
        with Catalog(root / "catalog.sqlite") as catalog:
            page = [entry.value for entry in catalog.page(limit=20).entries]
            selected = catalog.get(key).value
    elapsed = time.perf_counter() - start
    checksum = digest([page, selected])
    print(
        json.dumps(
            {
                "engine": engine,
                "operation_seconds": elapsed,
                "peak_rss_bytes": peak_process_rss_bytes(),
                "result_sha256": checksum,
            }
        )
    )


def run(root: Path, count: int, repeats: int) -> None:
    if count < 1 or repeats < 1:
        raise ValueError("count and repeats must be positive")
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with (root / "manifest.json").open("x") as stream:
        stream.write('{"samples":[')
        for i in range(count):
            if i:
                stream.write(",")
            json.dump(record(i)[1], stream, separators=(",", ":"))
        stream.write("]}")
    json_build = time.perf_counter() - started
    started = time.perf_counter()
    receipt = write_catalog(
        root / "catalog.sqlite",
        (record(i) for i in range(count)),
        metadata={"generator": "arrayroute-catalog-v1", "records": count},
    )
    catalog_build = time.perf_counter() - started
    expected = digest([[record(i)[1] for i in range(min(20, count))], record(count - 1)[1]])
    jobs = [(engine, repeat) for engine in ("json", "sqlite") for repeat in range(repeats)]
    random.Random(231).shuffle(jobs)
    results = []
    with (root / "trials.jsonl").open("x") as log:
        for engine, repeat in jobs:
            started = time.perf_counter()
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "arrayroute.benchmark_catalog",
                    "--out",
                    str(root),
                    "--records",
                    str(count),
                    "--worker",
                    engine,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            wall = time.perf_counter() - started
            value = json.loads(completed.stdout)
            if value["result_sha256"] != expected:
                raise ValueError(f"{engine} returned different metadata")
            value.update(repeat=repeat, subprocess_wall_seconds=wall, stderr=completed.stderr)
            log.write(json.dumps(value) + "\n")
            log.flush()
            results.append(value)
    report = {
        "records": count,
        "repeats": repeats,
        "operation": "first 20 records and last-key lookup",
        "scope": "local metadata only; no numerical payload throughput",
        "cache_policy": "fresh processes, OS cache uncontrolled; shuffled trials seed 231",
        "python": sys.version,
        "sqlite": sqlite3.sqlite_version,
        "platform": platform.platform(),
        "json_bytes": (root / "manifest.json").stat().st_size,
        "json_build_seconds": json_build,
        "catalog_build_seconds": catalog_build,
        "catalog": asdict(receipt),
        "result_sha256": expected,
        "median": {
            engine: {
                metric: statistics.median(v[metric] for v in results if v["engine"] == engine)
                for metric in ("operation_seconds", "subprocess_wall_seconds", "peak_rss_bytes")
            }
            for engine in ("json", "sqlite")
        },
        "trials": results,
    }
    (root / "report.json").write_text(json.dumps(report, default=str, indent=2) + "\n")
    print(json.dumps(report, default=str, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--records", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--worker", choices=["json", "sqlite"])
    args = parser.parse_args()
    if args.worker:
        worker(args.out, args.worker, args.records)
    else:
        run(args.out, args.records, args.repeats)


if __name__ == "__main__":
    main()
