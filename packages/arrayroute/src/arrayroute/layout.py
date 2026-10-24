# SPDX-License-Identifier: Apache-2.0
"""Bounded spatial chunks and object shards for regular scientific grids."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import index


@dataclass(frozen=True)
class GridLayout:
    chunks: tuple[int, ...]
    shards: tuple[int, ...] | None
    decoded_chunk_bytes: int
    decoded_shard_bytes: int
    object_count_upper_bound: int


@dataclass(frozen=True)
class GridLayoutPolicy:
    """Balance spatial retrieval against object count; not a universal optimal layout.

    Spatial edges are upper targets. Byte limits include padded edge blocks. Time steps
    are kept in separate shards by default; requested steps_per_chunk is an upper bound.
    Component/other axes stay together unless explicit axis caps or the byte limit split them.
    No values are read and no numerical conversion is performed.
    """

    chunk_edge: int = 32
    shard_edge: int = 128
    chunk_bytes: int = 512 << 10
    shard_bytes: int = 32 << 20

    def __post_init__(self) -> None:
        for name in ("chunk_edge", "shard_edge", "chunk_bytes", "shard_bytes"):
            value = index(getattr(self, name))
            if value < 1:
                raise ValueError("Layout edges and byte limits must be positive")
            object.__setattr__(self, name, value)
        if self.shard_edge < self.chunk_edge or self.shard_bytes < self.chunk_bytes:
            raise ValueError("Shard limits must be at least the chunk limits")

    def plan(
        self,
        shape: Sequence[int],
        *,
        itemsize: int,
        spatial_axes: Sequence[int],
        time_axis: int | None = None,
        steps_per_chunk: int = 1,
        axis_chunk_limits: Mapping[int, int] | None = None,
    ) -> GridLayout:
        shape = tuple(index(n) for n in shape)
        axes = tuple(index(a) for a in spatial_axes)
        itemsize, steps = index(itemsize), index(steps_per_chunk)
        time_axis = None if time_axis is None else index(time_axis)
        if self.chunk_bytes < itemsize:
            raise ValueError("Chunk byte limit must fit at least one element")
        if not shape or min(shape) < 1 or itemsize < 1 or steps < 1:
            raise ValueError("Grid extents, item size and steps per chunk must be positive")
        if not 1 <= len(axes) <= 3 or len(set(axes)) != len(axes):
            raise ValueError("Choose one to three distinct spatial axes")
        if any(not 0 <= axis < len(shape) for axis in axes):
            raise ValueError("Spatial axis outside array rank")
        if time_axis is not None and (not 0 <= time_axis < len(shape) or time_axis in axes):
            raise ValueError("Time axis must be distinct from spatial axes and within array rank")
        if itemsize > self.chunk_bytes:
            raise ValueError("A single value exceeds the chunk byte limit")
        chunks = list(shape)
        for axis in axes:
            chunks[axis] = min(shape[axis], self.chunk_edge)
        if time_axis is not None:
            chunks[time_axis] = min(shape[time_axis], steps)
        for raw_axis, raw_limit in (axis_chunk_limits or {}).items():
            if isinstance(raw_axis, bool) or isinstance(raw_limit, bool):
                raise TypeError("Axis chunk limits require integer axes and extents, not booleans")
            axis, limit = index(raw_axis), index(raw_limit)
            if not 0 <= axis < len(shape) or limit < 1:
                raise ValueError(
                    "Axis chunk limits require an axis within array rank and extent >= 1"
                )
            chunks[axis] = min(chunks[axis], limit)
        while math.prod(chunks) * itemsize > self.chunk_bytes:
            axis = max(range(len(shape)), key=lambda i: chunks[i])
            chunks[axis] = max(1, chunks[axis] // 2)
        multiples = [1] * len(shape)
        for axis in axes:
            extent = min(self.shard_edge, shape[axis])
            multiples[axis] = (extent + chunks[axis] - 1) // chunks[axis]
        chunk_size = math.prod(chunks) * itemsize
        while math.prod(multiples) * chunk_size > self.shard_bytes:
            axis = max(axes, key=lambda i: multiples[i])
            multiples[axis] = max(1, multiples[axis] // 2)
        shards = tuple(c * m for c, m in zip(chunks, multiples, strict=True))
        object_count = math.prod((n + s - 1) // s for n, s in zip(shape, shards, strict=True))
        return GridLayout(
            tuple(chunks),
            shards if any(m > 1 for m in multiples) else None,
            chunk_size,
            math.prod(shards) * itemsize,
            object_count,
        )
