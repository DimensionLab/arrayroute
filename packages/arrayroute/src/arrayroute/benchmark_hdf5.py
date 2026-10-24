# SPDX-License-Identifier: Apache-2.0
"""Fresh-process comparison of full HDF5 tables, projected reads and streamed inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .hdf5 import HDF5ReadSession
from .reader import ReadStats


def worker(path: Path, mode: str, operation: str) -> dict:
    import h5py

    selected = ["column_00", "column_05", "column_13"]
    stats = ReadStats()
    start = time.perf_counter()
    if mode == "full-table":
        with h5py.File(path, "r") as handle:
            table = {name: np.asarray(handle[name]) for name in sorted(handle)}
        if operation == "read":
            digest = hashlib.sha256()
            for name in selected:
                digest.update(memoryview(table[name]))
            result = digest.hexdigest()
        else:
            columns = []
            for name, values in table.items():
                valid = np.isfinite(values)
                finite = values[valid]
                columns.append(
                    dict(
                        name=name,
                        numeric=True,
                        invalid=int((~valid).sum()),
                        minimum=finite.min().item(),
                        maximum=finite.max().item(),
                    )
                )
            result = dict(rows=len(next(iter(table.values()))), columns=columns)
    else:
        with HDF5ReadSession(path, buffer_bytes=1 << 20) as session:
            if operation == "read":
                table = session.read_columns(columns=selected, stats=stats)
                digest = hashlib.sha256()
                for name in selected:
                    digest.update(memoryview(table[name]))
                result = digest.hexdigest()
            else:
                result = session.inspect_columns(stats=stats)
    elapsed = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(
        mode=mode,
        operation=operation,
        seconds=elapsed,
        peak_rss_bytes=rss if sys.platform == "darwin" else rss * 1024,
        result_sha256=hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest(),
        stats=vars(stats),
        pid=os.getpid(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", choices=["full-table", "arrayroute"])
    parser.add_argument("--operation", choices=["read", "inspect"])
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.file is None or args.operation is None:
            parser.error("Workers require --file and --operation")
        print(json.dumps(worker(args.file, args.worker, args.operation), sort_keys=True))
        return
    if not 1 <= args.rows <= 1_000_000 or not 1 <= args.repeats <= 10:
        parser.error("Rows must be 1..1000000 and repeats 1..10")
    import h5py

    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "table.h5"
    with h5py.File(path, "w") as handle:
        for column in range(24):
            rng = np.random.default_rng(20261024 + column)
            values = rng.standard_normal(args.rows)
            values[::10007] = np.nan
            handle.create_dataset(
                f"column_{column:02}",
                data=values,
                chunks=(min(8192, args.rows),),
                compression="gzip",
            )
    trials = [
        (mode, operation)
        for _ in range(args.repeats)
        for operation in ("read", "inspect")
        for mode in ("full-table", "arrayroute")
    ]
    random.Random(20261024).shuffle(trials)
    results = []
    started = time.time()
    for index, (mode, operation) in enumerate(trials):
        command = [
            sys.executable,
            "-m",
            "arrayroute.benchmark_hdf5",
            "--output",
            str(args.output),
            "--worker",
            mode,
            "--operation",
            operation,
            "--file",
            str(path),
        ]
        run = subprocess.run(command, capture_output=True, text=True, check=True)
        (args.output / f"trial-{index:02}.json").write_text(run.stdout)
        (args.output / f"trial-{index:02}.stderr").write_text(run.stderr)
        results.append(json.loads(run.stdout))
    for operation in ("read", "inspect"):
        assert len({r["result_sha256"] for r in results if r["operation"] == operation}) == 1
    metadata = dict(
        rows=args.rows,
        columns=24,
        selected_columns=3,
        logical_bytes=args.rows * 24 * 8,
        physical_bytes=path.stat().st_size,
        started_unix=started,
        finished_unix=time.time(),
        python=sys.version,
        numpy=np.__version__,
        h5py=h5py.__version__,
        platform=sys.platform,
        scope="Synthetic local HDF5 table; full-table baseline reads all columns. Projected "
        "read selects 3 columns; inspection scans all 24. OS caches uncontrolled. "
        "Fresh processes, deterministic shuffled order. Not equal I/O work or a cluster test.",
        results=results,
    )
    (args.output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    path.unlink()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
