# SPDX-License-Identifier: Apache-2.0
"""Bounded selected values in logical C order, grouped by physical storage chunk."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Sequence
from contextlib import nullcontext

import numpy as np

from .ordered import OrderedSelection
from .plan import Selection
from .reader import ArrayReader, ReadStats


def take_flat(
    reader: ArrayReader, indices: Sequence[int] | np.ndarray, *, selection: Selection | None = None,
    stats: ReadStats | None = None, cancelled: Callable[[], bool] | None = None,
    max_chunk_amplification: float = 1.0,
) -> np.ndarray:
    """Gather selected logical values, preserving caller order, duplicates and dtype.

    Indices address the C-flattened selected view (negative indices count from its end).
    Requests are grouped by physical source chunk. Within a group, bounded rectangles
    may include unrequested values; groups are split until output plus decoded data fit.
    max_chunk_amplification bounds touched chunks relative to occupied chunks in each
    joined region. The default reads no extra chunks. A whole frame is not required.

    The reader allowance covers the result, one gathered subset and one bounded read.
    Python index metadata
    is O(number of requested values * rank), additional to this allowance, as are native
    caches and codec scratch. Inputs remain immutable. Shared budgets are reserved once
    for the complete call. Cancellation is checked between groups/reads, not inside I/O.
    """
    if (isinstance(max_chunk_amplification, bool)
            or not math.isfinite(max_chunk_amplification) or max_chunk_amplification < 1):
        raise ValueError("max_chunk_amplification must be finite and at least one")
    if reader.backend.dtype.hasobject:
        raise TypeError("Indexed reads require a fixed-size non-object dtype")
    if isinstance(indices, np.ndarray) and indices.ndim != 1:
        raise ValueError("Indices must be a one-dimensional sequence")
    source = OrderedSelection.create(reader, selection)
    size = math.prod(source.shape)
    output_bytes = len(indices) * reader.backend.dtype.itemsize
    if 2 * output_bytes >= reader.buffer_bytes:
        raise MemoryError("Indexed output leaves no read buffer; split the request")

    def check() -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError("Indexed read cancelled")

    normalized = []
    for value in indices:
        check()
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("Indices must be integers, not booleans")
        flat = operator.index(value)
        if flat < 0:
            flat += size
        if not 0 <= flat < size:
            raise IndexError("Flat index outside selected view")
        normalized.append(flat)
    check()
    reservation = (reader.shared_budget.reserve(reader.buffer_bytes, cancelled=cancelled)
                   if reader.shared_budget is not None else nullcontext())
    with reservation:
        worker = ArrayReader(reader.backend, buffer_bytes=reader.buffer_bytes - 2 * output_bytes)
        combined = _coalesced_read(
            worker, source, normalized, stats, check, max_chunk_amplification)
        if combined is not None:
            return combined
        return _chunk_reads(worker, source, normalized, stats, check)


def _coalesced_read(worker, source, indices, stats, check, amplification):
    if (not indices or not source.shape
            or math.prod(source.shape) > np.iinfo(np.intp).max
            or max(worker.backend.spec.shape) > np.iinfo(np.intp).max):
        return None
    coordinates = np.unravel_index(np.asarray(indices, dtype=np.intp), source.shape)
    columns = iter(coordinates)
    chunk_columns = []
    for s, chunk in zip(source.selection, worker.backend.spec.chunks, strict=True):
        if not isinstance(s, int):
            chunk_columns.append((s.start + next(columns) * s.step) // chunk)
    chunks = np.column_stack(chunk_columns)
    output = np.empty(len(indices), dtype=worker.backend.dtype)
    pending = [np.arange(len(indices), dtype=np.intp)]
    while pending:
        check()
        positions = pending.pop()
        points = tuple(c[positions] for c in coordinates)
        lower = tuple(int(c.min()) for c in points)
        upper = tuple(int(c.max()) + 1 for c in points)
        region = tuple(slice(lo, hi) for lo, hi in zip(lower, upper, strict=True))
        selected = source.map_region(region)
        plan = worker.plan(selected)
        occupied = np.unique(chunks[positions], axis=0)
        fits = (plan.buffer_bytes_upper_bound <= worker.buffer_bytes
                and plan.chunks_touched <= len(occupied) * amplification)
        if not fits and len(occupied) > 1:
            # Split only on physical chunk boundaries, so separate branches cannot
            # decode the same chunk. Sparse empty regions are never enumerated.
            axis = int(np.argmax(np.ptp(occupied, axis=0)))
            planes = np.unique(occupied[:, axis])
            cut = planes[len(planes) // 2]
            before = chunks[positions, axis] < cut
            pending.extend((positions[~before], positions[before]))
            continue
        if not fits:
            # Within one chunk, split the output rectangle if it exceeds the allowance.
            # The decoder's indivisible chunk is still admitted by ArrayReader.
            if len(positions) > 1:
                axis = max(range(len(lower)), key=lambda a: upper[a] - lower[a])
                if upper[axis] - lower[axis] > 1:
                    order = np.argsort(points[axis], kind="stable")
                    middle = len(order) // 2
                    pending.extend((positions[order[middle:]], positions[order[:middle]]))
                    continue
        values = worker.read(selected, stats=stats)
        check()
        output[positions] = values[tuple(c - lo for c, lo in zip(points, lower, strict=True))]
        del values
    return output


def _chunk_reads(worker, source, indices, stats, check):
    groups: dict[tuple[int, ...], dict[tuple[int, ...], list[int]]] = {}
    chunks = worker.backend.spec.chunks
    for position, flat in enumerate(indices):
        check()
        point = []
        for extent in reversed(source.shape):
            flat, coordinate = divmod(flat, extent)
            point.append(coordinate)
        logical = tuple(reversed(point))
        coordinates = iter(logical)
        physical = tuple(s if isinstance(s, int) else s.start + next(coordinates) * s.step
                         for s in source.selection)
        chunk = tuple(i // c for i, c in zip(physical, chunks, strict=True))
        groups.setdefault(chunk, {}).setdefault(logical, []).append(position)
    check()
    output = np.empty(len(indices), dtype=worker.backend.dtype)
    for chunk in sorted(groups):
        destinations = groups[chunk]
        pending = [list(destinations)]
        while pending:
            check()
            points = pending.pop()
            lower = tuple(min(p[axis] for p in points) for axis in range(len(source.shape)))
            upper = tuple(max(p[axis] for p in points) + 1
                          for axis in range(len(source.shape)))
            region = tuple(slice(lo, hi) for lo, hi in zip(lower, upper, strict=True))
            selected = source.map_region(region)
            plan = worker.plan(selected)
            if plan.buffer_bytes_upper_bound > worker.buffer_bytes and len(points) > 1:
                axis = max(range(len(lower)), key=lambda a: upper[a] - lower[a])
                ordered = sorted(points, key=lambda p: p[axis])
                middle = len(ordered) // 2
                pending.extend((ordered[middle:], ordered[:middle]))
                continue
            values = worker.read(selected, stats=stats)
            check()
            for point in points:
                local = tuple(i - lo for i, lo in zip(point, lower, strict=True))
                for destination in destinations[point]:
                    output[destination] = values[local]
            del values
    return output
