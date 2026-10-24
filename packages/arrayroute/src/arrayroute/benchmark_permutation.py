# SPDX-License-Identifier: Apache-2.0
"""Measure index sampling, not physical data reads, in isolated worker processes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from arrayroute.permutation import Permutation

from .memory import peak_process_rss_bytes


def measure(engine: str, size: int, samples: int) -> dict:
    if engine == "torch":
        import torch
        from torch.utils.data import RandomSampler

        def torch_factory():
            return iter(RandomSampler(range(size), generator=torch.Generator().manual_seed(73)))

        factory = torch_factory
    elif engine == "arrayroute":

        def route_factory():
            return iter(Permutation(size, seed=73))

        factory = route_factory
    elif engine == "numpy":

        def numpy_factory():
            return iter(np.random.default_rng(73).permutation(size))

        factory = numpy_factory
    else:
        raise ValueError(engine)
    started = time.perf_counter()
    order = factory()
    first = int(next(order))
    first_seconds = time.perf_counter() - started
    values = [first, *map(int, itertools.islice(order, samples - 1))]
    seconds = time.perf_counter() - started
    assert len(values) == len(set(values)) == samples
    assert all(0 <= value < size for value in values)
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "arrayroute"]}
    if engine == "torch":
        versions["torch"] = importlib.metadata.version("torch")
    return {
        "engine": engine,
        "size": size,
        "sampled_indices": samples,
        "first_index_seconds": first_seconds,
        "seconds": seconds,
        "peak_rss_bytes": peak_process_rss_bytes(),
        "index_sha256": hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest(),
        "versions": versions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000_000, 10_000_000])
    parser.add_argument("--large-size", type=int, default=10**12)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--torch", action="store_true", help="Also test installed PyTorch sampler")
    parser.add_argument(
        "--engine", choices=["arrayroute", "numpy", "torch"], help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if min(*args.sizes, args.large_size, args.samples, args.repeats) < 1:
        parser.error("Sizes, samples and repeats must be positive")
    if args.engine:
        print(json.dumps(measure(args.engine, args.sizes[0], min(args.samples, args.sizes[0]))))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    jobs = [(size, engine) for size in args.sizes for engine in ["arrayroute", "numpy"]]
    if args.torch:
        jobs += [(size, "torch") for size in args.sizes]
    jobs.append((args.large_size, "arrayroute"))
    runs = []
    for repeat in range(args.repeats):
        for size, engine in jobs if repeat % 2 == 0 else list(reversed(jobs)):
            output = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "arrayroute.benchmark_permutation",
                    "--engine",
                    engine,
                    "--sizes",
                    str(size),
                    "--samples",
                    str(args.samples),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
                env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            )
            result = {"repeat": repeat, **json.loads(output.stdout)}
            (args.output / f"{repeat}-{size}-{engine}.json").write_text(
                json.dumps(result, indent=2)
            )
            (args.output / f"{repeat}-{size}-{engine}.stderr.txt").write_text(output.stderr)
            runs.append(result)
            print(json.dumps(result), flush=True)
    for size, engine in jobs:
        hashes = {r["index_sha256"] for r in runs if r["size"] == size and r["engine"] == engine}
        assert len(hashes) == 1, "Order changed between repeats"
    report = {
        "method": "Fresh process; imports outside clock; initialization + first indices timed. "
        "RSS includes imports and retained sampler state. No physical dataset I/O. "
        "Algorithms generate different permutations; uniform N! sampling is not claimed.",
        "platform": platform.platform(),
        "python": sys.version,
        "source_sha256": hashlib.sha256(
            Path(__file__).with_name("permutation.py").read_bytes()
        ).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": runs,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
