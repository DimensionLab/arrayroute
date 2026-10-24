# SPDX-License-Identifier: Apache-2.0
"""Fitted, lazy standardization of multiple aligned channel matrices."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .ordered import OrderedSelection, read_flat
from .plan import ArraySpec, Selection, plan
from .reader import ArrayReader, ReadStats
from .statistics import CanonicalChannelMoments

ALGORITHM = "finite-channel-affine-canonical-4096-v1"


@dataclass(frozen=True)
class ChannelStandardization:
    offset: tuple[float, ...]
    scale: tuple[float, ...]
    center: tuple[float, ...]
    spread: tuple[float, ...]
    count: tuple[int, ...]
    source_sha256: str
    algorithm: str = ALGORITHM


def _admit(source: ArrayReader, workspace_bytes: int, *, fitting: bool = True) -> int:
    shape, dtype = source.backend.spec.shape, source.backend.dtype
    if len(shape) != 2 or shape[1] < 1:
        raise ValueError("Standardization requires a (rows, channels) matrix")
    if shape[0] > np.iinfo(np.int64).max:
        raise OverflowError("Row count exceeds int64")
    if dtype.kind not in "biuf" or dtype.itemsize > 8:
        raise TypeError("Standardization requires real values of at most 64 bits")
    if isinstance(workspace_bytes, bool) or not isinstance(workspace_bytes, int):
        raise ValueError("workspace_bytes must be an integer")
    minimum = shape[1] * (4096 * 96 + 256 if fitting else dtype.itemsize + 24)
    if workspace_bytes < minimum:
        raise MemoryError(f"Channel statistics require at least {minimum} workspace bytes")
    rows = min(4096, source.buffer_bytes // (2 * dtype.itemsize * shape[1]))
    if rows < 1:
        raise MemoryError("Source read allowance cannot fit one channel row")
    return rows


def _cancel(check: Callable[[], bool] | None) -> None:
    if check is not None and check():
        raise InterruptedError("Channel standardization cancelled")


def fit_channel_standardization(
    source: ArrayReader, *, workspace_bytes: int = 32 << 20,
    source_stats: ReadStats | None = None, cancelled: Callable[[], bool] | None = None,
) -> ChannelStandardization:
    """Two full ordered passes: finite ranges, then fixed-leaf normalized moments.

    Scratch admission includes one 4096-row canonical leaf and numeric working arrays;
    source reader buffers/native caches are additional. Counts exclude NaN and infinity.
    SHA-256 over native C-order bytes must match between passes. Pin an immutable source
    revision through fitting and later reads; this is not snapshot isolation. Channel
    count/order, source dtype and runtime are part of the replay contract.
    """
    rows = _admit(source, workspace_bytes)
    _cancel(cancelled)
    n, channels = source.backend.spec.shape
    stats = source_stats if source_stats is not None else ReadStats()
    ordered = OrderedSelection.create(source, None)
    lo, hi = np.full(channels, np.inf), np.full(channels, -np.inf)
    signature = hashlib.sha256()
    for first in range(0, n, rows):
        raw = read_flat(ordered, first * channels, min(n, first + rows) * channels,
                        stats, cancelled)
        signature.update(memoryview(raw).cast("B"))
        values = raw.reshape(-1, channels).astype("f8")
        finite = np.isfinite(values)
        lo = np.minimum(lo, np.min(np.where(finite, values, np.inf), axis=0))
        hi = np.maximum(hi, np.max(np.where(finite, values, -np.inf), axis=0))
        del raw, values, finite
    present = np.isfinite(lo)
    lo, hi = np.where(present, lo, 0), np.where(present, hi, 0)
    offset = lo / 2 + hi / 2
    scale = np.maximum(np.abs(lo - offset), np.abs(hi - offset))
    scale = np.where(scale > 0, scale, 1)
    moments = CanonicalChannelMoments(channels)
    repeated = hashlib.sha256()
    for first in range(0, n, rows):
        raw = read_flat(ordered, first * channels, min(n, first + rows) * channels,
                        stats, cancelled)
        repeated.update(memoryview(raw).cast("B"))
        values = raw.reshape(-1, channels).astype("f8")
        finite = np.isfinite(values)
        with np.errstate(invalid="ignore", over="ignore"):
            values -= offset
            values /= scale
        moments.update(values, finite)
        del raw, values, finite
    _cancel(cancelled)
    if repeated.digest() != signature.digest():
        raise ValueError("Source changed between channel-statistics passes")
    state = moments.snapshot()
    spread = np.sqrt(np.maximum(state.m2 / np.maximum(state.count, 1), 0))
    return ChannelStandardization(
        tuple(offset), tuple(scale), tuple(state.mean), tuple(np.where(spread > 0, spread, 1)),
        tuple(int(n) for n in state.count), signature.hexdigest(),
    )


class StandardizedChannelsBackend:
    """Borrow aligned matrices and expose their standardized concatenated columns lazily.

    Construction fits each source sequentially without a whole-history stack. Workspace
    admission is per source; O(total channels) immutable parameter metadata is additional.
    The source readers, transform workspace and outer ArrayReader output have separate
    allowances. Source I/O is recorded in source_stats, outer counters are derived reads.
    Do not attach the same shared pool to both source and outer readers.

    Output is ((x - offset) / scale - center) / spread in float64. Constant finite
    channels become zero; all-nonfinite channels become NaN. NaN/inf inputs otherwise
    remain nonfinite, rather than being replaced by invented measurements. Fitting ignores
    them. Sources must remain pinned after fitting; partial reads cannot rehash histories.
    Pass recorded ``parameters`` to apply training statistics to other rows without
    fitting on held-out data. Their hashes identify the fitting data, not new input.
    """

    def __init__(
        self, sources: Sequence[ArrayReader], *, workspace_bytes: int = 32 << 20,
        source_stats: ReadStats | None = None, cancelled: Callable[[], bool] | None = None,
        parameters: Sequence[ChannelStandardization] | None = None,
    ) -> None:
        if not sources:
            raise ValueError("At least one channel source is required")
        sources = tuple(sources)
        for source in sources:
            _admit(source, workspace_bytes, fitting=parameters is None)
        rows = sources[0].backend.spec.shape[0]
        if any(s.backend.spec.shape[0] != rows for s in sources):
            raise ValueError("Channel sources must have the same row count")
        self.source_stats = source_stats if source_stats is not None else ReadStats()
        self.workspace_bytes, self.cancelled = workspace_bytes, cancelled
        self.sources = sources
        if parameters is None:
            self.parameters = tuple(fit_channel_standardization(
                s, workspace_bytes=workspace_bytes, source_stats=self.source_stats,
                cancelled=cancelled,
            ) for s in sources)
        else:
            if len(parameters) != len(sources):
                raise ValueError("Parameters must match the ordered sources")
            copied = []
            for source, fitted in zip(sources, parameters, strict=True):
                width = source.backend.spec.shape[1]
                vectors = (fitted.offset, fitted.scale, fitted.center, fitted.spread)
                if (fitted.algorithm != ALGORITHM or len(fitted.count) != width
                    or any(len(v) != width or not all(math.isfinite(x) for x in v)
                           for v in vectors)
                    or any(v <= 0 for v in (*fitted.scale, *fitted.spread))
                    or any(isinstance(n, bool) or not isinstance(n, int) or n < 0
                           for n in fitted.count)
                    or len(fitted.source_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in fitted.source_sha256)):
                    raise ValueError("Invalid recorded channel statistics")
                copied.append(ChannelStandardization(
                    tuple(float(x) for x in fitted.offset),
                    tuple(float(x) for x in fitted.scale),
                    tuple(float(x) for x in fitted.center),
                    tuple(float(x) for x in fitted.spread), tuple(fitted.count),
                    fitted.source_sha256,
                ))
            self.parameters = tuple(copied)
        self.dtype = np.dtype("f8")
        channels = sum(s.backend.spec.shape[1] for s in sources)
        self.spec = ArraySpec((rows, channels), (max(rows, 1), channels), 8, contiguous=True)

    def read(self, selection: Selection) -> np.ndarray:
        _cancel(self.cancelled)
        request = plan(self.spec, selection)
        time, columns = request.selection
        indices = range(columns, columns + 1) if isinstance(columns, int) else range(
            columns.start, columns.stop, columns.step,
        )
        nrows = 1 if isinstance(time, int) else len(range(time.start, time.stop, time.step))
        output = np.empty(request.output_shape, dtype="f8")
        matrix = output.reshape(nrows, len(indices))
        base = 0
        for source, params in zip(self.sources, self.parameters, strict=True):
            channels = source.backend.spec.shape[1]
            first = max(base, indices.start)
            first += (indices.start - first) % indices.step
            stop = min(base + channels, indices.stop)
            count = len(range(first, stop, indices.step))
            if count and nrows:
                local = slice(first - base, stop - base, indices.step)
                target = (first - indices.start) // indices.step
                offset, scale, center, spread = (
                    np.asarray(v)[local] for v in (params.offset, params.scale,
                                                  params.center, params.spread)
                )
                missing = np.asarray(params.count)[local] == 0
                rows = min(
                    self.workspace_bytes // (count * (source.backend.dtype.itemsize + 24)),
                    source.buffer_bytes // (2 * count * source.backend.dtype.itemsize),
                )
                ordered = OrderedSelection.create(source, (time, local))
                for start in range(0, nrows, rows):
                    _cancel(self.cancelled)
                    end = min(nrows, start + rows)
                    raw = read_flat(ordered, start * count, end * count,
                                    self.source_stats, self.cancelled)
                    values = raw.reshape(-1, count).astype("f8")
                    with np.errstate(invalid="ignore", over="ignore"):
                        values -= offset
                        values /= scale
                        values -= center
                        values /= spread
                    values[:, missing] = np.nan
                    matrix[start:end, target:target + count] = values
                    del raw, values
            base += channels
        _cancel(self.cancelled)
        return output
