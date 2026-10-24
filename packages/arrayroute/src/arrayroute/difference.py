# SPDX-License-Identifier: Apache-2.0
"""Bounded full-array comparisons with canonical C-order numerical reductions."""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .ordered import OrderedSelection, read_flat
from .plan import Selection
from .reader import ArrayReader, ReadStats
from .statistics import CanonicalChannelMoments

BLOCK_ELEMENTS = 65536
ALGORITHM = 'c-order-difference-65536-v1'




class _Summary:
    def __init__(self, count: int, sample_count: int):
        self.count = count
        self.sample_count = min(count, sample_count)
        # Integer arithmetic keeps the same sample positions beyond 2**53 elements.
        i = np.arange(self.sample_count, dtype=np.int64)
        q, r = divmod(max(0, count - 1), max(1, self.sample_count - 1))
        self.indices = q*i + (r*i)//max(1, self.sample_count - 1)
        self.sample = np.empty(self.sample_count, dtype=np.float64)
        self.moments = CanonicalChannelMoments(1)
        self.finite_count = 0
        self.nan_count = 0
        self.minimum: int | float | None = None
        self.maximum: int | float | None = None

    def update(self, values: np.ndarray, start: int) -> None:
        finite = np.isfinite(values)
        self.finite_count += int(finite.sum())
        self.nan_count += int(np.isnan(values).sum())
        valid = values[finite]
        if valid.size:
            lo, hi = valid.min().item(), valid.max().item()
            self.minimum = lo if self.minimum is None else min(lo, self.minimum)
            self.maximum = hi if self.maximum is None else max(hi, self.maximum)
        self.moments.update(values.astype(np.float64).reshape(-1, 1), finite.reshape(-1, 1))
        first, end = np.searchsorted(self.indices, [start, start + values.size])
        self.sample[first:end] = values[self.indices[first:end] - start]

    def result(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            'n': self.count, 'finite_count': self.finite_count,
            'nan_frac': self.nan_count / self.count if self.count else 0.0,
            'nonfinite_frac': 1 - self.finite_count / self.count if self.count else 0.0,
            'statistics_scope': 'full_array',
            'percentile_scope': ('full_array' if self.sample_count == self.count
                                 else 'deterministic_sample'),
            'percentile_sample_count': self.sample_count,
            'percentile_sampling': 'floor(i*(n-1)/(k-1)); k=1 selects index 0',
        }
        if self.finite_count:
            snapshot = self.moments.snapshot()
            if not np.isfinite(snapshot.mean).all() or not np.isfinite(snapshot.variance).all():
                raise OverflowError('Array summary exceeds float64 range')
            result.update(min=self.minimum, max=self.maximum,
                          mean=float(snapshot.mean[0]), std=float(np.sqrt(snapshot.variance[0])))
        finite = self.sample[np.isfinite(self.sample)]
        if finite.size:
            result.update(zip(('p01', 'p50', 'p99'),
                              map(float, np.percentile(finite, [1, 50, 99])), strict=True))
        return result


def compare_array_readers(
    before: ArrayReader | None, after: ArrayReader | None, *,
    before_selection: Selection | None = None, after_selection: Selection | None = None,
    percentile_samples: int = 65536, max_sample_bytes: int = 128 << 20,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Scan all values; only percentile estimates are sampled.

    Optional selections compare all values of those regions, in selected C order. Source
    selections accept the reader's integer/positive-stride slice contract. Original chunk
    admission applies after mapping each bounded region; no full selection is materialized.
    Source metadata and payloads must stay immutable during the call. Exact counts and
    native min/max cover every element. Float64 moments and norms use fixed 65,536-value
    C-order blocks; replay pins NumPy/runtime. Integer counts do not round through floats.
    Metrics require matching shape and dtype; changes to either are reported separately.

    Read allowances apply independently to each reader. Additional bounded workspace is
    at most max_sample_bytes for percentile values/indices and selection scratch, plus
    two read batches (each at most its reader allowance) and O(BLOCK_ELEMENTS) numeric
    scratch and canonical moment state. Codec scratch, caches and encoded inputs are outside
    this bound. Cancellation is cooperative between native reads. Unsupported dtypes and
    corrupt reads fail; neither becomes an 'unreadable' success result.
    """
    if (isinstance(percentile_samples, bool) or not isinstance(percentile_samples, int)
            or not 1 <= percentile_samples <= 4_000_000):
        raise ValueError('percentile_samples must be an integer in [1, 4000000]')
    if (isinstance(max_sample_bytes, bool) or not isinstance(max_sample_bytes, int)
            or max_sample_bytes < 1):
        raise ValueError('max_sample_bytes must be a positive integer')
    if before is None and after is None:
        raise ValueError('At least one array is required')
    if ((before is None and before_selection is not None)
            or (after is None and after_selection is not None)):
        raise ValueError('A selection requires its source array')
    sources = [OrderedSelection.create(r, s) for r, s in
               ((before, before_selection), (after, after_selection)) if r is not None]
    readers = [s.reader for s in sources]
    counts = [math.prod(s.shape) for s in sources]
    for reader, count in zip(readers, counts, strict=True):
        if reader.backend.dtype.kind not in 'biuf' or reader.backend.dtype.itemsize > 8:
            raise TypeError('Comparison requires fixed-size real numeric arrays of at most 64 bits')
        if count > np.iinfo(np.int64).max:
            raise OverflowError('Array element count exceeds int64')
    if sum(min(c, percentile_samples)*64 for c in counts) > max_sample_bytes:
        raise MemoryError(
            'Percentile workspace exceeds max_sample_bytes; reduce percentile_samples')
    if cancelled is not None and cancelled():
        raise InterruptedError('Array comparison cancelled')
    summaries = [_Summary(c, percentile_samples) for c in counts]
    stats = [ReadStats() for _ in readers]
    same = (before is not None and after is not None
            and sources[0].shape == sources[1].shape
            and before.backend.dtype == after.backend.dtype)
    changed = bits_changed = 0
    delta_max = norm_a = norm_delta = 0.0
    batch_elements = BLOCK_ELEMENTS * max(1, min(16, min(
        r.buffer_bytes // (2*r.backend.dtype.itemsize*BLOCK_ELEMENTS) for r in readers
    )))
    for batch_start in range(0, max(counts), batch_elements):
        batches = [(read_flat(s, batch_start, min(c, batch_start + batch_elements),
                           stats[i], cancelled)
                    if batch_start < c else None)
                   for i, (s, c) in enumerate(zip(sources, counts, strict=True))]
        for start in range(batch_start, min(max(counts), batch_start + batch_elements),
                           BLOCK_ELEMENTS):
            if cancelled is not None and cancelled():
                raise InterruptedError('Array comparison cancelled')
            blocks = []
            for index, batch in enumerate(batches):
                if batch is None or start >= counts[index]:
                    blocks.append(None)
                    continue
                offset = start - batch_start
                values = batch[offset:offset + BLOCK_ELEMENTS]
                summaries[index].update(values, start)
                blocks.append(values)
            if same:
                a, b = blocks
                assert a is not None and b is not None
                equal = (a == b) | (np.isnan(a) & np.isnan(b))
                changed += int((~equal).sum())
                bits_changed += int(np.any(a.view(np.uint8).reshape(len(a), -1)
                                          != b.view(np.uint8).reshape(len(b), -1), axis=1).sum())
                fa, fb = a.astype(np.float64), b.astype(np.float64)
                finite = np.isfinite(fa) & np.isfinite(fb)
                with np.errstate(over='raise', invalid='raise'):
                    delta = fb[finite] - fa[finite]
                if delta.size:
                    delta_max = max(delta_max, float(np.abs(delta).max()))
                    scale = float(np.abs(delta).max())
                    if scale:
                        norm_delta = math.hypot(
                            norm_delta, scale*float(np.linalg.norm(delta/scale)))
                aa = fa[np.isfinite(fa)]
                scale = float(np.abs(aa).max()) if aa.size else 0.0
                if scale:
                    norm_a = math.hypot(norm_a, scale*float(np.linalg.norm(aa/scale)))
    scope = 'full_selection' if any(s.explicit for s in sources) else 'full_array'
    result: dict[str, Any] = {'algorithm': ALGORITHM, 'comparison_scope': scope,
                              'read_batch_elements': batch_elements}
    index = 0
    for name, reader in (('before', before), ('after', after)):
        result['n_' + name] = None if reader is None else counts[index]
        if reader is not None:
            source = sources[index]
            result['shape_' + name] = list(source.shape)
            result['source_shape_' + name] = list(reader.backend.spec.shape)
            result['selection_' + name] = [v if isinstance(v, int) else
                                           [v.start, v.stop, v.step] for v in source.selection]
            result['dtype_' + name] = reader.backend.dtype.str
            result[name] = summaries[index].result()
            if source.explicit:
                result[name]['statistics_scope'] = 'full_selection'
                if result[name]['percentile_scope'] == 'full_array':
                    result[name]['percentile_scope'] = 'full_selection'
            result['read_' + name] = vars(stats[index])
            index += 1
    if same:
        if not math.isfinite(norm_a) or not math.isfinite(norm_delta):
            raise OverflowError('Difference norm exceeds float64 range')
        result.update(n_changed=changed, n_bitwise_changed=bits_changed,
                      frac_changed=changed/counts[0] if counts[0] else 0.0,
                      max_abs_diff=delta_max, rel_l2_diff=norm_delta/norm_a if norm_a else None,
                      metric_precision='float64; nonfinite pairs excluded from differences')
    else:
        result['note'] = ('added' if before is None else 'removed' if after is None else
                          'shape changed' if sources[0].shape != sources[1].shape
                          else 'dtype changed')
    return result
