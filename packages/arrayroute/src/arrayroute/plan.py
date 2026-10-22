# SPDX-License-Identifier: Apache-2.0
"""O(rank) selection costs; lazy subdivision without a global chunk inventory."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterator
from dataclasses import dataclass

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
class ArraySpec:
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    itemsize: int
    decode_window: DecodeWindow | None = None

    def __post_init__(self) -> None:
        if self.decode_window is not None and not isinstance(self.decode_window, DecodeWindow):
            raise TypeError("decode_window must be a DecodeWindow or None")
        if not self.shape or len(self.shape) != len(self.chunks):
            raise ValueError("shape and chunks must have the same nonzero rank")
        for values, minimum in ((self.shape, 0), (self.chunks, 1), ((self.itemsize,), 1)):
            if any(isinstance(v, bool) or not isinstance(v, int) or v < minimum for v in values):
                raise ValueError("shape, chunks and itemsize require valid integer dimensions")

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

    @property
    def decode_amplification(self) -> float:
        return self.decoded_bytes_upper_bound / self.output_bytes if self.output_bytes else 0.0

    @property
    def buffer_bytes_upper_bound(self) -> int:
        # Encoded transport buffers and backend caches are separate from this budget.
        return self.output_bytes + self.peak_decoded_bytes_upper_bound


def _length(s: slice) -> int:
    return max(0, (s.stop - s.start + s.step - 1) // s.step)


def plan(spec: ArraySpec, selection: Selection) -> ReadPlan:
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
    return ReadPlan(
        tuple(normalized),
        tuple(shape),
        math.prod(shape) * spec.itemsize,
        touched,
        touched * math.prod(spec.chunks) * spec.itemsize,
        live_chunks * math.prod(spec.chunks) * spec.itemsize,
    )


@dataclass(frozen=True)
class Tile:
    plan: ReadPlan
    destination: tuple[slice, ...]


def tiles(spec: ArraySpec, selection: Selection, *, buffer_bytes: int) -> Iterator[Tile]:
    """Yield disjoint output tiles, preferring cuts at existing chunk boundaries."""
    if buffer_bytes < 1:
        raise ValueError("buffer_bytes must be positive")
    initial = plan(spec, selection)
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
            children.append(Tile(plan(spec, tuple(selectors)), tuple(destinations)))
        stack.extend(reversed(children))
