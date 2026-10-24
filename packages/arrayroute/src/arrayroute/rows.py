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


def _fitting_end(
    reader: ArrayReader, rows: list[int], start: int, stop: int, tail: Selection, stride: int = 1,
) -> int:
    """Largest fitting prefix of an arithmetic run; one row may require tiling."""
    if stop == start + 1:
        return stop

    def fits(end: int) -> bool:
        request = reader.plan((slice(rows[start], rows[end - 1] + 1, stride), *tail))
        return request.buffer_bytes_upper_bound <= reader.buffer_bytes

    if fits(stop):
        return stop
    low, high = start + 1, min(start + 2, stop)
    # With fixed tail selections, native output and live chunk counts only grow.
    # Optional gather strategies are selected only when they fit the same allowance.
    # Bound the search by the fitting prefix, not the remaining run: tiny budgets
    # must not trigger a logarithmic search over a huge suffix for every row.
    while high < stop and fits(high):
        low, high = high, min(start + 2 * (high - start), stop)
    high -= 1
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    return low


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

    Each distinct row is read once. Regularly spaced rows may share a bounded read;
    gaps are never filled with unrequested rows. ``coalesce=False`` keeps deduplication
    but disables merging for layouts where it is slower. This does not
    change the sampler or optimizer order. ``tail`` selects the remaining axes.

    The reader's allowance covers the complete output plus one read's planned
    decoded/output buffers. A shared pool admits the output plus the largest planned
    read atomically, avoiding nested reservations without reserving unused capacity.
    Native caches, codec scratch, encoded bytes and
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

    worker = ArrayReader(reader.backend, buffer_bytes=reader.buffer_bytes - output_bytes)
    rows = sorted(destinations)

    def requests():
        position, run_end, stride = 0, 0, 1
        while position < len(rows):
            check_cancelled()
            end_position = position + 1
            if coalesce:
                if position == run_end:
                    run_end = end_position
                    if run_end < len(rows):
                        stride = rows[run_end] - rows[position]
                        run_end += 1
                        while run_end < len(rows) and rows[run_end] - rows[run_end - 1] == stride:
                            run_end += 1
                end_position = _fitting_end(worker, rows, position, run_end, tail, stride)
            yield position, end_position, stride
            position = end_position

    # Only pooled reads need a preflight pass. Keep request metadata O(batch rows),
    # and retain the exact plan boundaries for dispatch after admission.
    batches = list(requests()) if reader.shared_budget is not None else requests()
    amount = output_bytes
    if reader.shared_budget is not None:
        for start, stop, stride in batches:
            request = worker.plan((slice(rows[start], rows[stop - 1] + 1, stride), *tail))
            amount = max(amount, output_bytes + min(
                request.buffer_bytes_upper_bound, worker.buffer_bytes,
            ))
    reservation = (
        reader.shared_budget.reserve(amount, cancelled=cancelled)
        if reader.shared_budget is not None else nullcontext()
    )
    with reservation:
        output = np.empty(output_shape, dtype=reader.backend.dtype)
        for position, end_position, stride in batches:
            check_cancelled()
            begin = rows[position]
            values = worker.read(
                (slice(begin, rows[end_position - 1] + 1, stride), *tail), stats=stats,
            )
            check_cancelled()
            for row in rows[position:end_position]:
                for target in destinations[row]:
                    output[target] = values[(row - begin) // stride]
            del values
        return output
