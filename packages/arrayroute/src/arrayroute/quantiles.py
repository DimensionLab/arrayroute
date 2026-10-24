# SPDX-License-Identifier: Apache-2.0
"""Exact finite quantiles and canonical moments without whole-array materialization."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .ordered import OrderedSelection, read_flat
from .plan import Selection
from .reader import ArrayReader, ReadStats
from .statistics import CanonicalChannelMoments

ALGORITHM = "finite-f64-radix-quantiles-canonical-moments-v1"
_SIGN = 1 << 63
_MASK = (1 << 64) - 1


def count_nonfinite(
    reader: ArrayReader,
    selection: Selection | None = None,
    *,
    block_elements: int = 65536,
    stats: ReadStats | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[int, int]:
    """Count NaNs/infinities in one bounded pass, without conversion or sampling.

    Explicit scratch is one ordered result and one boolean mask, in addition to the
    reader allowance and backend caches. Native calls cannot be cancelled mid-read.
    """
    if (
        isinstance(block_elements, bool)
        or not isinstance(block_elements, int)
        or not (1 <= block_elements <= 1_000_000)
    ):
        raise ValueError("block_elements must be an integer in [1, 1000000]")
    if reader.backend.dtype.kind not in "biuf" or reader.backend.dtype.itemsize > 8:
        raise TypeError("Nonfinite counts require real numeric values of at most 64 bits")
    source = OrderedSelection.create(reader, selection)
    block = min(block_elements, reader.buffer_bytes // reader.backend.dtype.itemsize)
    if block < 1:
        raise MemoryError("Reader allowance cannot fit one element")
    stats = stats if stats is not None else ReadStats()
    n_nan = n_inf = 0
    if cancelled is not None and cancelled():
        raise InterruptedError("Nonfinite count cancelled")
    for start in range(0, math.prod(source.shape), block):
        values = read_flat(
            source, start, min(math.prod(source.shape), start + block), stats, cancelled
        )
        n_nan += int(np.count_nonzero(np.isnan(values)))
        n_inf += int(np.count_nonzero(np.isinf(values)))
        del values
    if cancelled is not None and cancelled():
        raise InterruptedError("Nonfinite count cancelled")
    return n_nan, n_inf


def _keys(values: np.ndarray) -> np.ndarray:
    values[values == 0] = 0.0  # One numerical ordering for signed zero.
    bits = values.view(np.uint64)
    return np.where(bits >> 63, ~bits, bits ^ np.uint64(_SIGN))


def _value(key: int) -> float:
    bits = key ^ _SIGN if key & _SIGN else ~key & _MASK
    return float(np.array([bits], dtype=np.uint64).view(np.float64)[0])


@dataclass(frozen=True)
class FiniteStatistics:
    count: int
    n_nan: int
    n_inf: int
    mean: float
    std: float
    minimum: float
    maximum: float
    quantiles: tuple[float, ...]
    passes: int
    source_sha256: str
    algorithm: str = ALGORITHM


class _Stream:
    def __init__(
        self,
        reader: ArrayReader,
        selection: Selection | None,
        block: int,
        io_elements: int,
        stats: ReadStats,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        self.source = OrderedSelection.create(reader, selection)
        self.count = math.prod(self.source.shape)
        if self.count > np.iinfo(np.int64).max:
            raise OverflowError("Selected element count exceeds int64")
        self.block = block
        self.io_elements = io_elements
        self.stats = stats
        self.cancelled = cancelled
        self.passes = 0
        self.signature = None

    def values(self) -> Iterator[np.ndarray]:
        signature = hashlib.sha256()
        self.check_cancelled()
        for start in range(0, self.count, self.io_elements):
            raw = read_flat(
                self.source,
                start,
                min(self.count, start + self.io_elements),
                self.stats,
                self.cancelled,
            )
            signature.update(memoryview(raw).cast("B"))
            for offset in range(0, raw.size, self.block):
                self.check_cancelled()
                values = raw[offset : offset + self.block].astype(np.float64)
                yield values
                del values
            del raw
        self.check_cancelled()
        current = signature.hexdigest()
        if self.signature is not None and current != self.signature:
            raise ValueError("Array changed between exact-statistics passes")
        self.signature = current
        self.passes += 1

    def check_cancelled(self) -> None:
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("Exact statistics cancelled")


@dataclass
class _Bucket:
    prefix: int
    shift: int
    count: int
    ranks: dict[int, int]  # global rank -> rank within this bucket

    def select(self, keys: np.ndarray) -> np.ndarray:
        return keys if self.shift == 64 else keys[keys >> self.shift == self.prefix >> self.shift]

    def split(self, counts: np.ndarray) -> list[_Bucket]:
        cumulative = np.cumsum(counts)
        if int(cumulative[-1]) != self.count:
            raise ValueError("Finite bucket count changed between statistics passes")
        children: dict[int, dict[int, int]] = {}
        for global_rank, local in self.ranks.items():
            bin_id = int(np.searchsorted(cumulative, local, side="right"))
            before = int(cumulative[bin_id - 1]) if bin_id else 0
            children.setdefault(bin_id, {})[global_rank] = local - before
        shift = self.shift - (counts.size.bit_length() - 1)
        return [
            _Bucket(self.prefix | (bin_id << shift), shift, int(counts[bin_id]), ranks)
            for bin_id, ranks in sorted(children.items())
        ]


def finite_statistics(
    reader: ArrayReader,
    selection: Selection | None = None,
    *,
    quantiles: Sequence[float] = (0.01, 0.99),
    workspace_bytes: int = 32 << 20,
    stats: ReadStats | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> FiniteStatistics:
    """Full-selection finite statistics after float64 conversion (including integer inputs).

    Quantiles use exact order statistics and linear interpolation, never sampling. Signed
    zero is canonicalized numerically; source integrity hashes retain original bytes.
    Moments use a fixed-leaf, origin/scale-normalized reduction. Bitwise replay requires
    the same NumPy/runtime, input order and values; it is independent of read/workspace
    budgets. NumPy's whole-array reduction can differ in last bits.

    Up to 16 quantiles are supported. A workspace of at least 4 MiB bounds explicit
    candidates, working arrays and moment state; reader/native scratch and caches are
    additional. No disk spill. Small selections use one pass, larger selections at most
    nine; source SHA-256 must match on every completed pass. Pin an immutable source for
    the operation. Cancellation is checked between reads/passes, not inside native calls.
    """
    if (
        isinstance(workspace_bytes, bool)
        or not isinstance(workspace_bytes, int)
        or workspace_bytes < 4 << 20
    ):
        raise ValueError("workspace_bytes must be an integer of at least 4 MiB")
    if reader.backend.dtype.kind not in "biuf" or reader.backend.dtype.itemsize > 8:
        raise TypeError("Statistics require real numeric values of at most 64 bits")
    if len(quantiles) > 16:
        raise ValueError("At most 16 quantiles are supported")
    q = tuple(float(v) for v in quantiles)
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in q):
        raise ValueError("Quantiles must be finite values in [0, 1]")
    io_elements = min(
        (workspace_bytes // 8) // reader.backend.dtype.itemsize,
        reader.buffer_bytes // (2 * reader.backend.dtype.itemsize),
    )
    capacity = workspace_bytes // 32
    scratch = (
        workspace_bytes - (1 << 20) - capacity * 8 - io_elements * reader.backend.dtype.itemsize
    )
    block = min(65536, scratch // 512, io_elements)
    if block < 1:
        raise MemoryError("Reader allowance cannot fit an ordered statistics element")
    stream = _Stream(
        reader,
        selection,
        block,
        io_elements,
        stats if stats is not None else ReadStats(),
        cancelled,
    )
    collected = np.empty(stream.count, dtype=np.float64) if stream.count <= capacity else None
    count = n_nan = n_inf = below = equal = 0
    minimum, maximum, anchor = math.inf, -math.inf, math.nan
    histogram = np.zeros(65536, dtype=np.int64)
    for a in stream.values():
        finite = a[np.isfinite(a)]
        finite[finite == 0] = 0.0
        nan = int(np.count_nonzero(np.isnan(a)))
        n_nan += nan
        n_inf += a.size - finite.size - nan
        if finite.size:
            if math.isnan(anchor):
                anchor = float(finite[0])
            minimum = min(minimum, float(finite.min()))
            maximum = max(maximum, float(finite.max()))
            below += int(np.count_nonzero(finite < anchor))
            equal += int(np.count_nonzero(finite == anchor))
            if collected is None:
                histogram += np.bincount((_keys(finite) >> 48).astype(np.int64), minlength=65536)
            else:
                collected[count : count + finite.size] = finite
            count += finite.size
    if not count or minimum == maximum:
        value = minimum if count else math.nan
        return FiniteStatistics(
            count,
            n_nan,
            n_inf,
            value,
            0.0 if count else math.nan,
            value,
            value,
            tuple(value for _ in q),
            stream.passes,
            str(stream.signature),
        )
    origin = anchor
    scale = max(abs(minimum - origin), abs(maximum - origin))
    if not math.isfinite(scale):
        origin, scale = 0.0, max(abs(minimum), abs(maximum))
    moments = CanonicalChannelMoments(1)
    indices = [(count - 1) * v for v in q]
    ranks = sorted({rank for h in indices for rank in (math.floor(h), math.ceil(h))})
    answers = {r: anchor for r in ranks if below <= r < below + equal}
    answers.update({r: minimum if r == 0 else maximum for r in ranks if r in (0, count - 1)})
    pending = [r for r in ranks if r not in answers]
    if collected is not None:
        for start in range(0, count, block):
            stream.check_cancelled()
            moments.update(
                ((collected[start : min(count, start + block)] - origin) / scale).reshape(-1, 1)
            )
        if pending:
            collected[:count].partition(pending)
        answers.update({r: float(collected[r]) for r in pending})
    else:
        buckets = _Bucket(0, 64, count, {r: r for r in pending}).split(histogram) if pending else []
        need_moments = True
        while buckets or need_moments:
            stream.check_cancelled()
            for bucket in buckets:
                if bucket.shift == 0:
                    answers.update({r: _value(bucket.prefix) for r in bucket.ranks})
            buckets = [b for b in buckets if b.shift]
            if not buckets and not need_moments:
                break
            gather = sum(b.count for b in buckets) <= capacity
            buffers = [np.empty(b.count, dtype=np.uint64) for b in buckets] if gather else []
            offsets = [0] * len(buckets)
            counts = [np.zeros(256, dtype=np.int64) for b in buckets] if not gather else []
            for a in stream.values():
                finite = a[np.isfinite(a)]
                if need_moments and finite.size:
                    moments.update(((finite - origin) / scale).reshape(-1, 1))
                if not buckets:
                    continue
                keys = _keys(finite)
                for i, bucket in enumerate(buckets):
                    selected = bucket.select(keys)
                    if gather:
                        end = offsets[i] + selected.size
                        if end > bucket.count:
                            raise ValueError("Finite bucket grew between statistics passes")
                        buffers[i][offsets[i] : end] = selected
                        offsets[i] = end
                    else:
                        bins = ((selected >> (bucket.shift - 8)) & 255).astype(np.int64)
                        counts[i] += np.bincount(bins, minlength=256)
            need_moments = False
            if gather:
                for i, bucket in enumerate(buckets):
                    if offsets[i] != bucket.count:
                        raise ValueError("Finite bucket shrank between statistics passes")
                    buffers[i].partition(list(bucket.ranks.values()))
                    answers.update({r: _value(int(buffers[i][k])) for r, k in bucket.ranks.items()})
                buckets = []
            else:
                buckets = [
                    child for b, c in zip(buckets, counts, strict=True) for child in b.split(c)
                ]
    stream.check_cancelled()
    summary = moments.snapshot()
    values = []
    for h in indices:
        lo, hi = answers[math.floor(h)], answers[math.ceil(h)]
        fraction = h - math.floor(h)
        if lo == hi or fraction == 0:
            values.append(lo)
        elif math.isfinite(hi - lo):
            values.append(float(np.quantile([lo, hi], fraction)))
        else:
            values.append(lo * (1 - fraction) + hi * fraction)
    return FiniteStatistics(
        count,
        n_nan,
        n_inf,
        origin + scale * float(summary.mean[0]),
        scale * math.sqrt(float(summary.variance[0])),
        minimum,
        maximum,
        tuple(values),
        stream.passes,
        str(stream.signature),
    )
