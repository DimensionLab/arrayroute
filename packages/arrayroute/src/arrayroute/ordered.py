# SPDX-License-Identifier: Apache-2.0
"""Bounded C-order selections and exact byte-stream hashing over scientific arrays."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .plan import Selection
from .reader import ArrayReader, ReadStats


def flat_regions(
    shape: tuple[int, ...], start: int, stop: int, prefix: tuple = ()
) -> Iterator[tuple]:
    """Cover a flat C-order interval with disjoint rectangular selections."""
    if not shape:
        if stop > start:
            yield prefix
        return
    width = math.prod(shape[1:])
    while start < stop:
        row, offset = divmod(start, width)
        if offset == 0 and stop - start >= width:
            rows = (stop - start) // width
            yield (*prefix, slice(row, row + rows), *(slice(None) for _ in shape[1:]))
            start += rows * width
        else:
            end = min(width, offset + stop - start)
            yield from flat_regions(shape[1:], offset, end, (*prefix, row))
            start += end - offset


@dataclass(frozen=True)
class OrderedSelection:
    reader: ArrayReader
    selection: Selection
    shape: tuple[int, ...]
    explicit: bool

    @classmethod
    def create(cls, reader: ArrayReader, selection: Selection | None) -> OrderedSelection:
        request = reader.plan(
            selection if selection is not None else (slice(None),) * len(reader.backend.spec.shape)
        )
        return cls(reader, request.selection, request.output_shape, selection is not None)

    def map_region(self, region: Selection) -> Selection:
        indices = iter(region)
        mapped: list[int | slice] = []
        for source in self.selection:
            if isinstance(source, int):
                mapped.append(source)
                continue
            local = next(indices)
            if isinstance(local, int):
                mapped.append(source.start + local * source.step)
            else:
                length = len(range(source.start, source.stop, source.step))
                start, stop, step = local.indices(length)
                mapped.append(
                    slice(
                        source.start + start * source.step,
                        min(source.stop, source.start + stop * source.step),
                        source.step * step,
                    )
                )
        return tuple(mapped)


def read_flat(
    source: OrderedSelection,
    start: int,
    stop: int,
    stats: ReadStats,
    cancelled: Callable[[], bool] | None,
) -> np.ndarray:
    reader = source.reader
    if not 0 <= start <= stop <= math.prod(source.shape):
        raise IndexError("Flat interval outside selected array")
    if (stop - start) * reader.backend.dtype.itemsize > reader.buffer_bytes:
        raise MemoryError("Flat result exceeds the reader allowance; use smaller intervals")
    out = np.empty(stop - start, dtype=reader.backend.dtype)
    offset = 0
    for selection in flat_regions(source.shape, start, stop):
        if cancelled is not None and cancelled():
            raise InterruptedError("Ordered array read cancelled")
        values = reader.read(source.map_region(selection), stats=stats)
        out[offset : offset + values.size] = values.reshape(-1)
        offset += values.size
        del values
    return out


def update_digest(
    digest: Any,
    reader: ArrayReader,
    selection: Selection | None = None,
    *,
    block_elements: int = 65536,
    stats: ReadStats | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Append exact selected native-dtype C-order bytes to an existing hashlib digest.

    Layout, chunks and read boundaries do not change the byte stream. Read allowance plus
    one bounded output interval covers explicit buffers, excluding backend caches/scratch.
    Neither dtype conversion nor NaN/signed-zero canonicalization is performed.
    """
    if (
        isinstance(block_elements, bool)
        or not isinstance(block_elements, int)
        or block_elements < 1
    ):
        raise ValueError("block_elements must be a positive integer")
    source = OrderedSelection.create(reader, selection)
    size = math.prod(source.shape)
    count = min(block_elements, reader.buffer_bytes // reader.backend.dtype.itemsize)
    if count < 1:
        raise MemoryError("Reader allowance cannot fit one element")
    stats = stats if stats is not None else ReadStats()
    for start in range(0, size, count):
        values = read_flat(source, start, min(size, start + count), stats, cancelled)
        digest.update(memoryview(values).cast("B"))
        del values
