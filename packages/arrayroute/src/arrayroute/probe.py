# SPDX-License-Identifier: Apache-2.0
"""Bounded, verified read pilots over caller-selected immutable array regions."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np

from .concurrency import plan_concurrency
from .execution import ProcessingCancelledError, ordered_map
from .plan import ArraySpec, Selection
from .reader import ArrayReader, ReadStats


def probe_reads(
    make_reader: Callable[[int], ArrayReader],
    selections: Sequence[Selection],
    *,
    worker_counts: Sequence[int],
    total_buffer_bytes: int,
    max_output_bytes: int,
    time_budget_seconds: float,
    repeats: int = 3,
    seed: int = 917,
) -> dict:
    """Compare explicit worker counts; never change customer configuration automatically.

    The factory receives a per-worker buffer allowance and must construct independent
    read-only state over the same fixed source. Native caches and codec scratch are
    outside this allowance. At most 4096 selections and 256 workers are accepted.
    All payload reads, including a sequential reference pass, count toward the output
    cap. This is a decoded output cap, not a network-transfer or process-RSS cap.

    Trial order is shuffled, with a fresh worker pool per trial. Every result must match
    the reference SHA-256. This checks consistency, not scientific ground truth. Timed
    trials include reader setup, reading, hashing and thread teardown. OS/remote caches
    are uncontrolled; repeated selections may measure a warm working set.

    The time budget prevents starting more reads; an in-flight native call cannot be
    interrupted. Use process isolation for a hard wall-clock deadline. A partial result
    never recommends a worker count. Keep the source fixed throughout the pilot.
    """
    for value in (total_buffer_bytes, max_output_bytes, repeats, seed, *worker_counts):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                "Budgets, repetitions, seed and worker counts must be positive integers"
            )
    if not worker_counts or len(set(worker_counts)) != len(worker_counts):
        raise ValueError("Provide distinct worker counts")
    if not 1 <= len(selections) <= 4096 or max(worker_counts) > 256:
        raise ValueError("Use 1–4096 selections and at most 256 workers")
    if repeats > 100:
        raise ValueError("Use at most 100 repetitions")
    if (isinstance(time_budget_seconds, bool) or not math.isfinite(time_budget_seconds)
            or time_budget_seconds <= 0):
        raise ValueError("Time budget must be finite and positive")
    started = time.perf_counter()
    deadline = started + time_budget_seconds
    metadata_reader = make_reader(total_buffer_bytes)
    spec, dtype = metadata_reader.backend.spec, metadata_reader.backend.dtype
    if dtype.kind not in 'biufc' or dtype.itemsize > 16:
        raise TypeError("Read pilots require fixed-size numeric arrays")
    normalized = [metadata_reader.plan(selection).selection for selection in selections]
    output_bytes = sum(metadata_reader.plan(s).output_bytes for s in normalized)
    if not output_bytes:
        raise ValueError("Select at least one field value")
    allowed, skipped = [], []
    for count in sorted(worker_counts):
        # The requested count may not be a power of two in the bounded planner's list.
        choices = [plan_concurrency(spec, s, total_buffer_bytes=total_buffer_bytes,
                                    max_workers=count).candidates[-1] for s in normalized]
        if count > len(selections):
            skipped.append({'workers': count, 'reason': 'More workers than selected regions'})
        elif any(c.workers != count or c.materialization not in {'direct', 'tiled', 'empty'}
                 for c in choices):
            skipped.append({'workers': count, 'reason': 'A region cannot fit this worker budget'})
        else:
            allowed.append(count)
    if not allowed:
        raise MemoryError("No worker count can materialize the selected regions within budget")
    planned_bytes = output_bytes * (1 + repeats * len(allowed))
    if planned_bytes > max_output_bytes:
        raise ValueError(
            f"Pilot would read {planned_bytes} output bytes; cap is {max_output_bytes}"
        )
    del metadata_reader
    report = {
        'state': 'running', 'selections': len(selections), 'dtype': dtype.str,
        'source_shape': spec.shape, 'source_chunks': spec.chunks,
        'regions': [[{'start': s.start, 'stop': s.stop, 'step': s.step}
                     if isinstance(s, slice) else s for s in selection]
                    for selection in normalized],
        'output_bytes_per_pass': output_bytes, 'planned_output_bytes': planned_bytes,
        'total_buffer_bytes': total_buffer_bytes, 'skipped': skipped, 'seed': seed,
        'repeats': repeats, 'trials': [], 'fastest_observed_workers': None,
        'scope': 'Verified repeated-region wall time, including setup and SHA-256; '
                 'OS/remote caches uncontrolled; not raw I/O or training throughput',
        'memory_scope': 'Per-worker output, intermediates and decoded-chunk estimate; '
                        'excludes native caches, codec scratch and up to 64 KiB hash scratch '
                        'per worker',
        'deadline_scope': 'Cooperative between reads; in-flight native reads are joined',
    }

    def expired() -> bool:
        return time.perf_counter() >= deadline

    def run(count: int) -> list[dict]:
        return _run_reads(make_reader, normalized, total_buffer_bytes, count, spec, dtype, expired)

    try:
        reference = run(1)
        report['reference_sha256'] = [r['sha256'] for r in reference]
        order = [(rep, count) for rep in range(repeats) for count in allowed]
        random.Random(seed).shuffle(order)
        report['trial_order'] = order
        for repetition, count in order:
            start = time.perf_counter()
            rows = run(count)
            elapsed = time.perf_counter() - start
            if any((r['sha256'], r['bytes']) != (e['sha256'], e['bytes'])
                   for r, e in zip(rows, reference, strict=True)):
                raise ValueError("Pilot values changed from the sequential reference")
            strategies = Counter()
            for row in rows:
                strategies.update(row['read_strategies'])
            report['trials'].append({
                'repetition': repetition, 'workers': count, 'elapsed_seconds': elapsed,
                'output_bytes': sum(r['bytes'] for r in rows),
                'read_seconds_sum': sum(r['read_seconds'] for r in rows),
                'requests': sum(r['requests'] for r in rows), 'verified': True,
                'read_strategies': dict(strategies),
                'gather_workers_peak': max(r['gather_workers_peak'] for r in rows),
            })
        medians = {count: statistics.median(t['elapsed_seconds'] for t in report['trials']
                                           if t['workers'] == count) for count in allowed}
        report.update(state='completed', median_seconds_by_workers=medians,
                      fastest_observed_workers=min(medians, key=lambda c: medians[c]))
    except ProcessingCancelledError:
        report['state'] = 'time-budget-reached'
    report['elapsed_seconds'] = time.perf_counter() - started
    return report


def _run_reads(
    make_reader: Callable[[int], ArrayReader], selections: Sequence[Selection],
    total_buffer_bytes: int, count: int, spec: ArraySpec, dtype: np.dtype,
    cancelled: Callable[[], bool],
) -> list[dict]:
    """Shared worker-owned read/hash loop for concurrency and cross-policy pilots."""
    budget = total_buffer_bytes // count

    def make_worker():
        reader = make_reader(budget)
        if reader.backend.spec != spec or reader.backend.dtype != dtype:
            raise ValueError("Reader factory changed array metadata")
        if reader.buffer_bytes != budget:
            raise ValueError("Reader factory did not apply the assigned buffer budget")

        def read(index: int) -> dict:
            stats = ReadStats()
            value = reader.read(selections[index], stats=stats)
            digest = hashlib.sha256()
            for block in np.nditer(value, flags=['external_loop', 'buffered', 'zerosize_ok'],
                                   op_flags=['readonly'], order='C',
                                   buffersize=max(1, 32768 // dtype.itemsize)):
                digest.update(cast(np.ndarray, block).tobytes())
            return {'sha256': digest.hexdigest(), 'bytes': value.nbytes,
                    'requests': stats.requests, 'read_seconds': stats.seconds,
                    'read_strategies': stats.read_strategies,
                    'gather_workers_peak': stats.gather_workers_peak}

        return read

    with ordered_map(make_worker, range(len(selections)), workers=count,
                     max_pending=count, cancelled=cancelled) as results:
        return list(results)
