# SPDX-License-Identifier: Apache-2.0
"""Bounded selection costs; lazy subdivision without a global chunk inventory."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Literal

Selection = tuple[int | slice, ...]


@dataclass(frozen=True)
class DecodeWindow:
    """Backend guarantee about live decoded chunks, excluding codec scratch/cache.

    Use only when the reader releases decoded chunks before requesting successors.
    A stride-only guarantee does not apply to contiguous native bulk reads.
    """

    max_chunks: int
    strided_only: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_chunks, bool)
            or not isinstance(self.max_chunks, int)
            or self.max_chunks < 1
        ):
            raise ValueError("max_chunks must be a positive integer")
        if not isinstance(self.strided_only, bool):
            raise TypeError("strided_only must be a boolean")


@dataclass(frozen=True)
class GatherWindow:
    """Independent regions with single-chunk partial or two-region bulk decoding."""

    region_shape: tuple[int, ...]
    workers: int
    bulk: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.bulk, bool):
            raise TypeError("bulk must be a boolean")
        if (isinstance(self.workers, bool) or not isinstance(self.workers, int)
                or not 1 <= self.workers <= 256):
            raise ValueError("Gather workers must be an integer from 1 to 256")
        if (not self.region_shape or any(isinstance(v, bool) or not isinstance(v, int)
                                        or v < 1 for v in self.region_shape)):
            raise ValueError("Gather regions require positive integer dimensions")


@dataclass(frozen=True)
class ArraySpec:
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    itemsize: int
    decode_window: DecodeWindow | None = None
    dense_gather: bool = False
    gather_window: GatherWindow | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dense_gather, bool):
            raise TypeError("dense_gather must be a boolean")
        if self.decode_window is not None and not isinstance(self.decode_window, DecodeWindow):
            raise TypeError("decode_window must be a DecodeWindow or None")
        if not self.shape or len(self.shape) != len(self.chunks):
            raise ValueError("shape and chunks must have the same nonzero rank")
        for values, minimum in ((self.shape, 0), (self.chunks, 1), ((self.itemsize,), 1)):
            if any(isinstance(v, bool) or not isinstance(v, int) or v < minimum for v in values):
                raise ValueError("shape, chunks and itemsize require valid integer dimensions")

        gather = self.gather_window
        if gather is not None:
            if not isinstance(gather, GatherWindow):
                raise TypeError("gather_window must be a GatherWindow or None")
            if (self.dense_gather or self.decode_window is None
                    or self.decode_window.max_chunks != 1):
                raise ValueError("Parallel gathering requires a single-chunk native decode window")
            if (len(gather.region_shape) != len(self.chunks)
                    or any(r % c for r, c in zip(gather.region_shape, self.chunks, strict=True))):
                raise ValueError("Gather regions must be a whole number of chunks on each axis")

    @property
    def logical_bytes(self) -> int:
        return math.prod(self.shape) * self.itemsize


@dataclass(frozen=True)
class ReadPlan:
    selection: Selection
    output_shape: tuple[int, ...]
    output_bytes: int
    chunks_touched: int
    decoded_bytes_upper_bound: int
    peak_decoded_bytes_upper_bound: int
    intermediate_bytes_upper_bound: int = 0
    strategy: Literal["native", "dense-gather", "parallel-gather", "bulk-gather"] = "native"
    read_selection: Selection | None = None
    gather_workers: int = 1

    @property
    def decode_amplification(self) -> float:
        return self.decoded_bytes_upper_bound / self.output_bytes if self.output_bytes else 0.0

    @property
    def buffer_bytes_upper_bound(self) -> int:
        # Encoded transport buffers and backend caches are separate from this budget.
        return (self.output_bytes + self.peak_decoded_bytes_upper_bound
                + self.intermediate_bytes_upper_bound)


def _length(s: slice) -> int:
    return max(0, (s.stop - s.start + s.step - 1) // s.step)


def plan(spec: ArraySpec, selection: Selection, *, buffer_bytes: int | None = None) -> ReadPlan:
    if buffer_bytes is not None and (
        isinstance(buffer_bytes, bool) or not isinstance(buffer_bytes, int) or buffer_bytes < 1
    ):
        raise ValueError("buffer_bytes must be a positive integer or None")
    if len(selection) != len(spec.shape):
        raise ValueError("Specify one integer or slice for every array dimension")
    normalized: list[int | slice] = []
    shape, counts = [], []
    for sel, size, chunk in zip(selection, spec.shape, spec.chunks, strict=True):
        if isinstance(sel, slice):
            if sel.step is not None and sel.step <= 0:
                raise ValueError("Only positive slice strides are supported")
            start, stop, step = sel.indices(size)
            s = slice(start, stop, step)
            length = _length(s)
            normalized.append(s)
            shape.append(length)
            last = start + (length - 1) * step
            counts.append(min(length, last // chunk - start // chunk + 1) if length else 0)
        else:
            if isinstance(sel, bool):
                raise TypeError("Boolean indexing is not a region selection")
            index = operator.index(sel)
            index = index + size if index < 0 else index
            if not 0 <= index < size:
                raise IndexError(f"Index {sel} outside dimension of size {size}")
            normalized.append(index)
            counts.append(1)
    touched = math.prod(counts)
    live_chunks = touched
    window = spec.decode_window
    if window is not None and (
        not window.strided_only or any(isinstance(s, slice) and s.step > 1 for s in normalized)
    ):
        live_chunks = min(touched, window.max_chunks)
    request = ReadPlan(
        tuple(normalized),
        tuple(shape),
        math.prod(shape) * spec.itemsize,
        touched,
        touched * math.prod(spec.chunks) * spec.itemsize,
        live_chunks * math.prod(spec.chunks) * spec.itemsize,
    )
    if (buffer_bytes is None or not request.output_bytes
            or not any(isinstance(s, slice) and s.step > 1 for s in normalized)):
        return request
    gather = spec.gather_window
    if gather is not None:
        regions = plan(ArraySpec(spec.shape, gather.region_shape, spec.itemsize),
                       request.selection).chunks_touched
        region_output = spec.itemsize * math.prod(
            min(_length(s), (r + s.step - 1) // s.step)
            for s, r in zip(request.selection, gather.region_shape, strict=True)
            if isinstance(s, slice)
        )
        if gather.bulk:
            bulk = _bulk_plan(spec, request, regions, region_output, buffer_bytes)
            if bulk is not None:
                return bulk
        for workers in range(min(gather.workers, regions), 1, -1):
            candidate = replace(
                request, strategy="parallel-gather", gather_workers=workers,
                peak_decoded_bytes_upper_bound=workers * request.peak_decoded_bytes_upper_bound,
                intermediate_bytes_upper_bound=min(request.output_bytes, workers * region_output),
            )
            if candidate.buffer_bytes_upper_bound <= buffer_bytes:
                return candidate
        return request
    if not spec.dense_gather:
        return request
    bounds = tuple(
        s if isinstance(s, int) else slice(s.start, s.start + (_length(s) - 1) * s.step + 1, 1)
        for s in normalized
    )
    dense = plan(replace(spec, dense_gather=False), bounds)
    candidate = replace(
        request, chunks_touched=dense.chunks_touched,
        decoded_bytes_upper_bound=dense.decoded_bytes_upper_bound,
        peak_decoded_bytes_upper_bound=dense.peak_decoded_bytes_upper_bound,
        intermediate_bytes_upper_bound=dense.output_bytes,
        strategy="dense-gather", read_selection=bounds,
    )
    if (candidate.chunks_touched == request.chunks_touched
            and candidate.buffer_bytes_upper_bound <= buffer_bytes):
        return candidate
    return request


def _bulk_plan(
    spec: ArraySpec, request: ReadPlan, regions: int, region_output: int, buffer_bytes: int,
) -> ReadPlan | None:
    gather = spec.gather_window
    assert gather is not None
    bounds = []
    for s, r, size in zip(request.selection, gather.region_shape, spec.shape, strict=True):
        start = s if isinstance(s, int) else s.start
        last = s if isinstance(s, int) else s.start + (_length(s) - 1) * s.step
        bounds.append(slice(start // r * r, min(size, (last // r + 1) * r)))
    expanded = plan(replace(spec, gather_window=None), tuple(bounds))
    # Bulk reads must not decode skipped in-domain chunks. Padded boundary chunks
    # still count toward the physical decode bound below.
    if expanded.chunks_touched != request.chunks_touched:
        return None
    region_bytes = math.prod(gather.region_shape) * spec.itemsize
    for workers in range(min(gather.workers, regions), 0, -1):
        candidate = replace(
            request, strategy="bulk-gather", gather_workers=workers,
            decoded_bytes_upper_bound=regions * region_bytes,
            peak_decoded_bytes_upper_bound=2 * workers * region_bytes,
            intermediate_bytes_upper_bound=min(request.output_bytes, workers * region_output),
        )
        if candidate.buffer_bytes_upper_bound <= buffer_bytes:
            return candidate
    return None


@dataclass(frozen=True)
class Tile:
    plan: ReadPlan
    destination: tuple[slice, ...]


def tiles(spec: ArraySpec, selection: Selection, *, buffer_bytes: int) -> Iterator[Tile]:
    """Yield disjoint output tiles, preferring cuts at existing chunk boundaries."""
    if buffer_bytes < 1:
        raise ValueError("buffer_bytes must be positive")
    initial = plan(spec, selection, buffer_bytes=buffer_bytes)
    stack = [Tile(initial, tuple(slice(0, n) for n in initial.output_shape))]
    while stack:
        tile = stack.pop()
        if tile.plan.output_bytes == 0:
            continue
        if tile.plan.buffer_bytes_upper_bound <= buffer_bytes:
            yield tile
            continue
        options = [
            (i, s)
            for i, s in enumerate(tile.plan.selection)
            if isinstance(s, slice) and _length(s) > 1
        ]
        if not options:
            raise MemoryError("One decoded chunk exceeds the buffer budget; rechunk the array")
        def split_priority(pair: tuple[int, slice]) -> tuple[int, int]:
            axis, region = pair
            count = _length(region)
            last = region.start + (count - 1) * region.step
            touched = min(count, last // spec.chunks[axis] - region.start // spec.chunks[axis] + 1)
            # Cutting within one chunk cannot reduce decoded bytes. Split across chunks first.
            return touched, count

        axis, sel = max(options, key=split_priority)
        count = _length(sel)
        chunk = spec.chunks[axis]
        midpoint = sel.start + (count // 2) * sel.step
        boundary = (midpoint // chunk) * chunk
        cut = min(count - 1, max(1, (boundary - sel.start + sel.step - 1) // sel.step))
        source_cut = sel.start + cut * sel.step
        out_axis = sum(isinstance(s, slice) for s in tile.plan.selection[:axis])
        dest = tile.destination[out_axis]
        children = []
        for source, target in (
            (slice(sel.start, source_cut, sel.step), slice(dest.start, dest.start + cut)),
            (slice(source_cut, sel.stop, sel.step), slice(dest.start + cut, dest.stop)),
        ):
            selectors, destinations = list(tile.plan.selection), list(tile.destination)
            selectors[axis], destinations[out_axis] = source, target
            children.append(Tile(
                plan(spec, tuple(selectors), buffer_bytes=buffer_bytes), tuple(destinations)))
        stack.extend(reversed(children))
