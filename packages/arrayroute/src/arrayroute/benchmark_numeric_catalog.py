# SPDX-License-Identifier: Apache-2.0
"""Reproducible scalar-index publication, ordering, paging and quantile measurements."""

from __future__ import annotations

import argparse
import itertools
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

from .memory import peak_process_rss_bytes
from .numeric_catalog import NumericCursor, NumericIndex, write_numeric_index


def value(i: int) -> float | None:
    return None if i % 7 == 0 else float(i % 51)


def expected_order(count: int, descending: bool):
    for v in sorted(range(51), reverse=descending):
        for i in range(v, count, 51):
            if i % 7:
                yield i, float(v)
    for i in range(0, count, 7):
        yield i, None


def key(ordinal: int, number: float | None, descending: bool):
    return (
        int(number is None),
        0.0 if number is None else (-number if descending else number),
        ordinal,
    )


def worker(root: Path, mode: str) -> dict:
    receipt = json.loads((root / "publication.json").read_text())
    count = receipt["records"]
    descending = mode.endswith("desc")
    deep = mode.startswith("deep")
    start = time.perf_counter()
    with NumericIndex(
        root / "numeric.sqlite",
        source_revision="numeric-benchmark-v1",
        expected_revision=receipt["revision"],
    ) as index:
        if mode == "quantiles":
            result = index.quantiles()
        else:
            cursor = None
            if deep:
                ordinal = count // 2
                k = key(ordinal, value(ordinal), descending)
                cursor = NumericCursor(index.revision, descending, *k)
            page = index.page(descending=descending, cursor=cursor, limit=64)
            result = tuple((e.ordinal, e.value) for e in page.entries)
    elapsed = time.perf_counter() - start
    if mode == "quantiles":
        counts = [sum(i % 7 != 0 for i in range(v, count, 51)) for v in range(51)]
        finite = sum(counts)
        expected = []
        for b in range(1, 5):
            target = (b * finite + 3) // 4
            accumulated = 0
            for v, n in enumerate(counts):
                accumulated += n
                if accumulated >= target:
                    expected.append(float(v))
                    break
        assert result == tuple(expected)
    else:
        rows = expected_order(count, descending)
        if deep:
            threshold = key(count // 2, value(count // 2), descending)
            rows = (r for r in rows if key(*r, descending) > threshold)
        assert result == tuple(itertools.islice(rows, 64))
    import hashlib

    return dict(
        mode=mode,
        seconds=elapsed,
        verified=True,
        result_sha256=hashlib.sha256(json.dumps(result).encode()).hexdigest(),
        peak_rss_bytes=peak_process_rss_bytes(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--records", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--worker", choices=["top-asc", "top-desc", "deep-asc", "deep-desc", "quantiles"]
    )
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.out, args.worker)))
        return
    if args.records < 1000 or not 1 <= args.repeats <= 100:
        parser.error("at least 1000 records and 1–100 repeats required")
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    receipt = write_numeric_index(
        args.out / "numeric.sqlite",
        (value(i) for i in range(args.records)),
        source_revision="numeric-benchmark-v1",
        field="inlet_velocity",
        units="m/s",
    )
    publication = asdict(receipt)
    publication["path"] = str(receipt.path)
    publication["seconds"] = time.perf_counter() - start
    with NumericIndex(receipt.path, source_revision="numeric-benchmark-v1") as index:
        start = time.perf_counter()
        assert index.verify() == receipt.revision
        publication["full_verification_seconds"] = time.perf_counter() - start
    (args.out / "publication.json").write_text(json.dumps(publication, indent=2) + "\n")
    schedule = [
        (m, r)
        for m in ["top-asc", "top-desc", "deep-asc", "deep-desc", "quantiles"]
        for r in range(args.repeats)
    ]
    random.Random(377).shuffle(schedule)
    report: dict[str, Any] = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        sqlite=sqlite3.sqlite_version,
        publication=publication,
        seed=377,
        scope=(
            "Local single scalar index, per-row integrity checks, missing values and ties; "
            "fresh processes, OS cache uncontrolled. No array I/O, network or rendering. "
            "RSS includes interpreter and result validation; operation time excludes validation."
        ),
        trials=[],
    )
    (args.out / "protocol.json").write_text(json.dumps(report, indent=2) + "\n")
    for mode, rep in schedule:
        start = time.perf_counter()
        response = subprocess.run(
            [
                sys.executable,
                "-m",
                "arrayroute.benchmark_numeric_catalog",
                "--out",
                str(args.out),
                "--worker",
                mode,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        wall = time.perf_counter() - start
        with (args.out / "raw.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        mode=mode,
                        rep=rep,
                        returncode=response.returncode,
                        stdout=response.stdout,
                        stderr=response.stderr,
                    )
                )
                + "\n"
            )
        response.check_returncode()
        trial = json.loads(response.stdout)
        trial.update(rep=rep, wall_s=wall)
        report["trials"].append(trial)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["medians"] = {}
    for mode, _ in schedule:
        rows = [t for t in report["trials"] if t["mode"] == mode]
        assert all(t["verified"] for t in rows)
        assert len({t["result_sha256"] for t in rows}) == 1
        report["medians"][mode] = {
            k: statistics.median(t[k] for t in rows)
            for k in ["seconds", "wall_s", "peak_rss_bytes"]
        }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["medians"], indent=2))


if __name__ == "__main__":
    main()
