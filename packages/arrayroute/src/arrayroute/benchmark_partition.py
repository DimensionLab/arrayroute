# SPDX-License-Identifier: Apache-2.0
"""Measure distributed index planning only: no physical data or network I/O."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path

from arrayroute.permutation import BatchPartition, Permutation

from .memory import peak_process_rss_bytes


def measure(size: int, world_size: int, batch_size: int, batches: int, epoch: int) -> dict:
    first = BatchPartition(size, 0, world_size, batch_size, "drop", seed=73, epoch=epoch)
    start_batch = first.global_batches // 2
    observed_values: set[int] = set()
    observed_positions: set[int] = set()
    digest = hashlib.sha256()
    plan_bytes = 0
    started = time.perf_counter()
    timings = []
    order = Permutation(size, seed=73, epoch=epoch)
    for rank in range(world_size):
        began = time.perf_counter()
        plan = BatchPartition(size, rank, world_size, batch_size, "drop", seed=73,
                              epoch=epoch, start_batch=start_batch)
        iterator = iter(plan)
        selected = []
        for _ in range(min(batches, len(plan))):
            selected.extend(next(iterator))
        timings.append(time.perf_counter() - began)
        plan_bytes = max(plan_bytes, len(pickle.dumps(plan)))
        # Verification is outside each rank's planning timer, inside total wall time.
        positions = list(plan.positions()[:len(selected)])
        if any(order.inverse(value) != position for value, position in zip(
            selected, positions, strict=True
        )):
            raise ValueError("Partition values do not map to the declared positions")
        if observed_values.intersection(selected) or observed_positions.intersection(positions):
            raise ValueError("Rank selections overlap")
        observed_values.update(selected)
        observed_positions.update(positions)
        for value in selected:
            digest.update(value.to_bytes(8, "little"))
    count = min(batches, first.global_batches - start_batch) * batch_size * world_size
    if len(observed_values) != count:
        raise ValueError("Rank selection omitted or repeated values")
    elapsed = time.perf_counter() - started
    return {
        "size": size, "world_size": world_size, "batch_size": batch_size, "epoch": epoch,
        "start_batch": start_batch, "rank_batches_requested": batches,
        "sampled_indices": count, "dropped_global_tail": first.dropped_samples,
        "rank_planning_seconds": timings, "total_with_verification_seconds": elapsed,
        "max_pickled_plan_bytes": plan_bytes, "indices_sha256": digest.hexdigest(),
        "peak_process_rss_bytes": peak_process_rss_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10**6, 10**12 + 39])
    parser.add_argument("--world-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(*args.sizes, args.world_size, args.batch_size, args.batches, args.repeats) < 1:
        parser.error("All counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for repeat in range(args.repeats):
        for size in args.sizes:
            report = measure(size, args.world_size, args.batch_size, args.batches, epoch=3)
            reports.append(report)
            (args.output / f"repeat{repeat}-size{size}.json").write_text(
                json.dumps(report, indent=2)
            )
    for size in args.sizes:
        if len({r["indices_sha256"] for r in reports if r["size"] == size}) != 1:
            raise ValueError("Partition order changed across repeated measurements")
    source = Path(__file__).with_name("permutation.py")
    (args.output / "complete.json").write_text(json.dumps({
        "reports": len(reports), "platform": platform.platform(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Logical rank-index planning only; ranks run sequentially in one process. "
                 "No physical dataset reads, GPU work, cluster scaling or PB throughput measured. "
                 "Retained verification sets scale with sampled indices, not the logical domain.",
    }, indent=2))


if __name__ == "__main__":
    main()
