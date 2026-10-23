# SPDX-License-Identifier: Apache-2.0
"""Verified policy comparison without executing saved configuration or changing inputs."""
from __future__ import annotations

import math
import random
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .concurrency import plan_concurrency
from .errors import UnsupportedReadPolicyError
from .execution import ProcessingCancelledError
from .plan import Selection
from .probe import _run_reads
from .reader import ArrayReader


@dataclass(frozen=True)
class ReadPolicy:
    """A named factory of worker-owned readers over the same fixed source as the reference."""

    name: str
    make_reader: Callable[[int], ArrayReader]

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',
                                                            self.name):
            raise ValueError('Policy names need 1–64 ASCII letters, digits, dots, '
                             'dashes or underscores')
        if not callable(self.make_reader):
            raise TypeError('Policy reader factory must be callable')


def compare_reads(
    reference: Callable[[int], ArrayReader], policies: Sequence[ReadPolicy],
    selections: Sequence[Selection], *, worker_counts: Sequence[int],
    total_buffer_bytes: int, max_output_bytes: int, time_budget_seconds: float,
    repeats: int = 3, seed: int = 1739,
) -> dict:
    """Shuffle all policy/worker trials together and verify against one reference.

    Metadata-only preflight rejects oversized output before payload reads. Missing
    optional imports and explicit UnsupportedReadPolicyError are reported as skipped;
    other errors and changed values fail the comparison. All trials must complete
    before recommending the lowest observed median. This is workload-local evidence,
    not a prediction for other machines, data, transformations or cold caches.
    Factories must honor their assigned per-worker buffer. Compressed inputs, codec
    scratch, caches and caller-retained arrays are excluded, so this is not an RSS cap.
    The deadline is cooperative; in-flight native reads are always joined.
    """
    for value in (total_buffer_bytes, max_output_bytes, repeats, seed, *worker_counts):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Budgets, repetitions, seed and workers must be positive integers')
    if (not worker_counts or len(set(worker_counts)) != len(worker_counts)
            or max(worker_counts) > 256 or not 1 <= len(selections) <= 4096 or repeats > 100):
        raise ValueError('Use distinct workers up to 256, 1–4096 selections and 1–100 repetitions')
    if (not 1 <= len(policies) <= 64 or len(policies)*len(worker_counts) > 256
            or any(not isinstance(p, ReadPolicy) for p in policies)
            or len({p.name for p in policies}) != len(policies)):
        raise ValueError('Use distinct policies, at most 64 policies and 256 configurations')
    if (isinstance(time_budget_seconds, bool) or not math.isfinite(time_budget_seconds)
            or time_budget_seconds <= 0):
        raise ValueError('Time budget must be finite and positive')
    started = time.perf_counter()
    deadline = started + time_budget_seconds
    baseline = reference(total_buffer_bytes)
    if baseline.buffer_bytes != total_buffer_bytes:
        raise ValueError('Reference factory did not apply the assigned buffer budget')
    spec, dtype = baseline.backend.spec, baseline.backend.dtype
    if dtype.kind not in 'biufc' or dtype.itemsize > 16:
        raise TypeError('Read comparisons require fixed-size numeric arrays')
    normalized = [baseline.plan(s).selection for s in selections]
    output_bytes = sum(baseline.plan(s).output_bytes for s in normalized)
    if not output_bytes:
        raise ValueError('Select at least one field value')
    reference_plans = [plan_concurrency(spec, s, total_buffer_bytes=total_buffer_bytes,
                                       max_workers=1).candidates[-1] for s in normalized]
    if any(p.materialization not in {'direct', 'tiled', 'empty'} for p in reference_plans):
        raise MemoryError('The reference cannot materialize a selected region within budget')
    del baseline
    prepared, allowed, skipped = {}, [], []
    for policy in policies:
        try:
            reader = policy.make_reader(total_buffer_bytes)
        except (ImportError, UnsupportedReadPolicyError) as exc:
            skipped.append({'policy': policy.name, 'workers': None, 'reason': str(exc)})
            continue
        candidate_spec = reader.backend.spec
        if reader.buffer_bytes != total_buffer_bytes:
            raise ValueError(f'Policy {policy.name} did not apply the assigned buffer budget')
        if candidate_spec.shape != spec.shape or reader.backend.dtype != dtype:
            raise ValueError(f'Policy {policy.name} changed source shape or dtype')
        prepared[policy.name] = (policy.make_reader, candidate_spec)
        for count in sorted(worker_counts):
            choices = [plan_concurrency(candidate_spec, s, total_buffer_bytes=total_buffer_bytes,
                                        max_workers=count).candidates[-1] for s in normalized]
            if count > len(normalized):
                reason = 'More workers than selected regions'
            elif any(c.workers != count or c.materialization not in {'direct', 'tiled', 'empty'}
                     for c in choices):
                reason = 'A region cannot fit this worker budget'
            else:
                allowed.append((policy.name, count))
                continue
            skipped.append({'policy': policy.name, 'workers': count, 'reason': reason})
        del reader
    if not allowed:
        raise MemoryError('No supported policy/worker configuration fits the selected regions')
    planned = output_bytes*(1 + repeats*len(allowed))
    if planned > max_output_bytes:
        raise ValueError(f'Comparison would read {planned} output bytes; cap is {max_output_bytes}')
    report = {
        'state': 'running', 'selections': len(normalized), 'source_shape': spec.shape,
        'dtype': dtype.str, 'output_bytes_per_pass': output_bytes, 'planned_output_bytes': planned,
        'total_buffer_bytes': total_buffer_bytes, 'seed': seed, 'repeats': repeats,
        'skipped': skipped, 'trials': [], 'recommendation': None,
        'configurations': [{'policy': name, 'workers': count,
                            'buffer_bytes_per_worker': total_buffer_bytes//count,
                            'source_chunks': prepared[name][1].chunks} for name, count in allowed],
        'regions': [[{'start': s.start, 'stop': s.stop, 'step': s.step}
                     if isinstance(s, slice) else s for s in selection]
                    for selection in normalized],
        'scope': 'Verified repeated-region median wall time including setup, hashing and teardown; '
                 'OS/native caches uncontrolled; not raw I/O or end-to-end training throughput',
        'memory_scope': 'Read buffers exclude native caches, codec scratch, encoded inputs '
                        'and up to 64 KiB hash scratch per worker; not a process-RSS bound',
        'deadline_scope': 'Cooperative between reads; in-flight native calls are joined',
        'read_cost_scope': 'Completed tile plans, counting repeated chunk visits; decoded byte '
                           'bounds include chunk padding. Not observed disk/network traffic or '
                           'cache misses; cached backends may decode fewer bytes.',
    }

    def expired():
        return time.perf_counter() >= deadline

    try:
        expected = _run_reads(reference, normalized, total_buffer_bytes, 1, spec, dtype, expired)
        report['reference_sha256'] = [r['sha256'] for r in expected]
        order = [(r, name, workers) for r in range(repeats) for name, workers in allowed]
        random.Random(seed).shuffle(order)
        report['trial_order'] = order
        for repetition, name, workers in order:
            make_reader, candidate_spec = prepared[name]
            start, cpu = time.perf_counter(), time.process_time()
            rows = _run_reads(make_reader, normalized, total_buffer_bytes, workers,
                              candidate_spec, dtype, expired)
            elapsed, cpu_elapsed = time.perf_counter()-start, time.process_time()-cpu
            if any((r['sha256'], r['bytes']) != (e['sha256'], e['bytes'])
                   for r, e in zip(rows, expected, strict=True)):
                raise ValueError(f'Policy {name} values differ from the sequential reference')
            strategies = Counter()
            for row in rows:
                strategies.update(row['read_strategies'])
            report['trials'].append({
                'policy': name, 'workers': workers, 'repetition': repetition,
                'elapsed_seconds': elapsed, 'cpu_seconds': cpu_elapsed, 'verified': True,
                'output_bytes': sum(r['bytes'] for r in rows),
                'planned_chunk_visits': sum(r['planned_chunk_visits'] for r in rows),
                'decoded_bytes_upper_bound': sum(r['decoded_bytes_upper_bound'] for r in rows),
                'decode_amplification_upper_bound': (
                    sum(r['decoded_bytes_upper_bound'] for r in rows) / output_bytes),
                'requests': sum(r['requests'] for r in rows), 'read_strategies': dict(strategies),
                'gather_workers_peak': max(r['gather_workers_peak'] for r in rows),
            })
        summary = []
        for name, workers in allowed:
            rows = [t for t in report['trials'] if (t['policy'], t['workers']) == (name, workers)]
            summary.append({'policy': name, 'workers': workers,
                            'buffer_bytes_per_worker': total_buffer_bytes//workers,
                            'median_seconds': statistics.median(r['elapsed_seconds'] for r in rows),
                            'median_cpu_seconds': statistics.median(r['cpu_seconds'] for r in rows),
                            'min_seconds': min(r['elapsed_seconds'] for r in rows),
                            'max_seconds': max(r['elapsed_seconds'] for r in rows)})
        report.update(state='completed', summary=summary,
                      recommendation=min(summary, key=lambda r: (r['median_seconds'],
                                                                  r['median_cpu_seconds'])))
    except ProcessingCancelledError:
        report['state'] = 'time-budget-reached'
    report['elapsed_seconds'] = time.perf_counter()-started
    return report
