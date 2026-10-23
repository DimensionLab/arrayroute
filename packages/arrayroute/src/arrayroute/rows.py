# SPDX-License-Identifier: Apache-2.0
"""Ordered, duplicate-aware batches over the leading axis of a chunked array."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Sequence
from contextlib import nullcontext

import numpy as np

from .plan import Selection
from .reader import ArrayReader, ReadStats


def read_rows(
    reader: ArrayReader,
    indices: Sequence[int],
    *,
    tail: Selection | None = None,
    coalesce: bool = True,
    stats: ReadStats | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Gather leading-axis rows in caller order, including repeated and negative indices.

    Each distinct row is read once. Adjacent requested rows may share a bounded read;
    gaps are never filled with unrequested rows. ``coalesce=False`` keeps deduplication
    but disables adjacency merging for layouts where it is slower. This does not
    change the sampler or optimizer order. ``tail`` selects the remaining axes.

    The reader's allowance covers the complete output plus one read's planned
    decoded/output buffers. A shared pool admits that whole allowance atomically,
    avoiding nested reservations. Native caches, codec scratch, encoded bytes and
    Python index metadata (O(batch rows)) are additional, as for ordinary reads.
    Split a batch if its output and a row read cannot fit. No data cache survives
    this call. The source must remain immutable for its duration.

    Cancellation is checked before reads and copies; it cannot interrupt native I/O.
    Stats count physical reads, so duplicate output rows do not increase them.
    """
    if not isinstance(coalesce, bool):
        raise TypeError("coalesce must be a boolean")
    spec = reader.backend.spec
    tail = (slice(None),) * (len(spec.shape) - 1) if tail is None else tail
    empty = reader.plan((slice(0, 0), *tail))
    row_shape = empty.output_shape[1:]
    output_shape = (len(indices), *row_shape)
    output_bytes = math.prod(output_shape) * spec.itemsize
    if output_bytes >= reader.buffer_bytes:
        raise MemoryError("Row batch leaves no read buffer; split the batch")

    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError("Row batch cancelled")

    destinations: dict[int, list[int]] = {}
    for position, value in enumerate(indices):
        check_cancelled()
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("Row indices must be integers, not booleans")
        row = operator.index(value)
        if row < 0:
            row += spec.shape[0]
        if not 0 <= row < spec.shape[0]:
            raise IndexError("Row index out of range")
        if output_bytes:
            destinations.setdefault(row, []).append(position)
    check_cancelled()
    if not output_bytes:
        return np.empty(output_shape, dtype=reader.backend.dtype)

    reservation = (
        reader.shared_budget.reserve(reader.buffer_bytes, cancelled=cancelled)
        if reader.shared_budget is not None else nullcontext()
    )
    with reservation:
        output = np.empty(output_shape, dtype=reader.backend.dtype)
        worker = ArrayReader(reader.backend, buffer_bytes=reader.buffer_bytes - output_bytes)
        rows = sorted(destinations)
        position = 0
        while position < len(rows):
            check_cancelled()
            begin = rows[position]
            end_position = position + 1
            if coalesce:
                while end_position < len(rows) and rows[end_position] == rows[end_position - 1] + 1:
                    candidate = worker.plan((slice(begin, rows[end_position] + 1), *tail))
                    if candidate.buffer_bytes_upper_bound > worker.buffer_bytes:
                        break
                    end_position += 1
            values = worker.read((slice(begin, rows[end_position - 1] + 1), *tail), stats=stats)
            check_cancelled()
            for row in rows[position:end_position]:
                for target in destinations[row]:
                    output[target] = values[row - begin]
            del values
            position = end_position
        return output
