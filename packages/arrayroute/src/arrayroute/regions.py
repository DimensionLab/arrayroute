# SPDX-License-Identifier: Apache-2.0
"""Bounded rectangular traversal that never straddles a preferred storage block."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from operator import index

from .ordered import flat_blocks, flat_regions


def block_regions(
    shape: Sequence[int], block_shape: Sequence[int], *, max_elements: int,
    coalesce: bool = False,
) -> Iterator[tuple[slice, ...]]:
    """Cover an array once, completing each preferred block before visiting the next.

    Visit blocks lexicographically and subdivide oversized blocks in local C order.
    A block that fits is yielded whole, avoiding partial sink chunk updates. Every region
    preserves rank, has unit stride, contains at most max_elements values and stays inside
    one block. With coalesce=True, combine adjacent whole blocks into larger rectangles
    when the limit permits, expanding trailing axes first. Output boundaries stay on the
    original block grid unless a single block itself needs subdivision. Ragged edge blocks
    are clipped to the array. Zero-sized arrays yield none.

    Traversal retains rank-bounded state, independent of the number of blocks. This is a
    geometry planner, not I/O or a memory reservation. Callers account for dtype, working
    buffers and codec/cache memory and own publication/cancellation. Global C-order
    reductions must not use this order: spatial blocks can change reduction grouping.
    """
    if isinstance(max_elements, bool):
        raise ValueError("max_elements must be a positive integer")
    if not isinstance(coalesce, bool):
        raise TypeError("coalesce must be a boolean")
    limit = index(max_elements)
    sizes, blocks = tuple(index(n) for n in shape), tuple(index(n) for n in block_shape)
    if limit < 1 or not sizes or len(sizes) != len(blocks):
        raise ValueError("A positive limit and matching nonempty shapes are required")
    if any(n < 0 for n in sizes) or any(b < 1 for b in blocks):
        raise ValueError("Array extents must be nonnegative and block extents positive")
    if any(n == 0 for n in sizes):
        return
    extent = [min(n, b) for n, b in zip(sizes, blocks, strict=True)]
    if coalesce and math.prod(extent) <= limit:
        for axis in reversed(range(len(sizes))):
            available = limit // (math.prod(extent) // extent[axis])
            extent[axis] = (sizes[axis] if sizes[axis] <= available
                            else available // blocks[axis] * blocks[axis])
        blocks = tuple(extent)

    def visit(axis: int, origin: tuple[int, ...]) -> Iterator[tuple[slice, ...]]:
        if axis < len(sizes):
            for start in range(0, sizes[axis], blocks[axis]):
                yield from visit(axis + 1, (*origin, start))
            return
        extent = tuple(min(b, n - i) for i, b, n in zip(origin, blocks, sizes, strict=True))
        if math.prod(extent) <= limit:
            yield tuple(slice(i, i + n) for i, n in zip(origin, extent, strict=True))
            return
        for start, stop in flat_blocks(extent, limit):
            for local in flat_regions(extent, start, stop):
                yield tuple(
                    slice(i + r, i + r + 1) if isinstance(r, int)
                    else slice(i + (r.start or 0), i + (n if r.stop is None else r.stop))
                    for i, n, r in zip(origin, extent, local, strict=True)
                )

    yield from visit(0, ())
