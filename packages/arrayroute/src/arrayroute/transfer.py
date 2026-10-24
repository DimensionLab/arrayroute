# SPDX-License-Identifier: Apache-2.0
"""Synchronous, bounded transfers from deferred array sources to slice sinks."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

Region = tuple[slice, ...]


@dataclass(frozen=True)
class BlockArray:
    """Borrow a source without loading it; the owner must outlive the transfer.

    ``read`` returns the exact requested unit-stride region with the declared dtype.
    Backend scratch/caches are additional to the transfer's returned-block allowance.
    No implicit NumPy coercion is provided: consumers must explicitly request regions.
    """

    shape: tuple[int, ...]
    dtype: np.dtype
    read: Callable[[Region], np.ndarray]

    def __post_init__(self) -> None:
        shape = tuple(operator.index(n) for n in self.shape)
        dtype = np.dtype(self.dtype)
        if not shape or any(n < 1 for n in shape):
            raise ValueError("Block arrays require nonempty positive dimensions")
        if dtype.hasobject or dtype.itemsize < 1:
            raise TypeError("Block arrays require fixed-size non-object dtypes")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @classmethod
    def from_array(cls, value: Any) -> BlockArray:
        if isinstance(value, cls):
            return value
        a = np.asarray(value)
        return cls(a.shape, a.dtype, lambda region: a[region])

    def with_component_axis(self) -> BlockArray:
        def read(region: Region) -> np.ndarray:
            return self.read(region[:-1])[..., None]

        return BlockArray((*self.shape, 1), self.dtype, read)


def copy_blocks(
    source: BlockArray,
    write: Callable[[Region, np.ndarray], None],
    *,
    block_shape: Sequence[int],
    dtype: Any,
    buffer_bytes: int,
    cancelled: Callable[[], bool] | None = None,
) -> int:
    """Visit blocks lexicographically and wait for each write before reading the next.

    The budget covers one returned source block plus one contiguous converted block, not
    source-owned arrays, reader scratch, sink codec buffers or OS caches. ``write`` must
    finish consuming the block before returning. Block boundaries are never subdivided:
    callers can align them with storage shards without repeated read/modify/write cycles.
    The caller owns transaction/publication. Returns the number of output payload bytes.
    """
    block = tuple(operator.index(n) for n in block_shape)
    target = np.dtype(dtype)
    budget = operator.index(buffer_bytes)
    if len(block) != len(source.shape) or any(n < 1 for n in block):
        raise ValueError("Block shape must contain one positive extent per source axis")
    if target.hasobject or target.itemsize < 1:
        raise TypeError("Transfer dtype must be fixed-size and non-object")
    count = math.prod(min(n, b) for n, b in zip(source.shape, block, strict=True))
    needed = count * (source.dtype.itemsize + target.itemsize)
    if budget < needed:
        raise MemoryError(f"Transfer needs {needed} bytes per block; buffer_bytes is {budget}")
    written = 0
    for start in product(*(range(0, n, b) for n, b in zip(source.shape, block, strict=True))):
        if cancelled is not None and cancelled():
            raise InterruptedError("Array transfer cancelled")
        region = tuple(
            slice(i, min(i + b, n)) for i, b, n in zip(start, block, source.shape, strict=True)
        )
        value = source.read(region)
        expected = tuple(s.stop - s.start for s in region)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != expected
            or value.dtype != source.dtype
        ):
            raise ValueError("Source returned a block inconsistent with its shape/dtype contract")
        converted = np.ascontiguousarray(value, dtype=target)
        if cancelled is not None and cancelled():
            raise InterruptedError("Array transfer cancelled")
        write(region, converted)
        written += converted.nbytes
        del value, converted
    return written
