# SPDX-License-Identifier: Apache-2.0
"""Bounded aligned reads across arrays with different sequence-axis positions."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .reader import ArrayReader, ReadStats


@dataclass(frozen=True)
class AxisBatch:
    start: int
    stop: int
    values: tuple[np.ndarray, ...]


def aligned_axis_batches(
    sources: Sequence[tuple[ArrayReader, int]],
    *,
    buffer_bytes: int,
    workspace_bytes_per_step: int,
    steps: int | None = None,
    cancelled: Callable[[], bool] | None = None,
    stats: ReadStats | None = None,
) -> Iterator[AxisBatch]:
    """Read matching consecutive steps, retaining all other axes in each source.

    The allowance covers retained input outputs plus caller-declared workspace per step.
    Reader I/O allowances, backend caches, codecs and retained consumer results are extra.
    All sources must have the same nonempty sequence extent. ``steps`` selects a prefix.
    Sources are caller-owned and must remain immutable. Release each batch before advancing;
    retained batches are outside this allowance. Shared read pools cover I/O, not consumers.
    Arithmetic is the consumer's responsibility; batching alone cannot promise reproducible
    reductions. Cancellation is checked between reads, not within native backend calls.
    """
    for name, value, minimum in (
        ("buffer_bytes", buffer_bytes, 1),
        ("workspace_bytes_per_step", workspace_bytes_per_step, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not sources:
        raise ValueError("At least one source is required")
    normalized = []
    extent = None
    input_bytes = []
    for reader, axis in sources:
        shape = reader.backend.spec.shape
        if (
            isinstance(axis, bool)
            or not isinstance(axis, int)
            or not -len(shape) <= axis < len(shape)
        ):
            raise ValueError("Sequence axis is outside the source shape")
        axis %= len(shape)
        if min(shape) < 1 or (extent is not None and shape[axis] != extent):
            raise ValueError("Sources require matching nonempty sequence axes")
        extent = shape[axis]
        normalized.append((reader, axis))
        input_bytes.append(
            math.prod(shape[:axis] + shape[axis + 1 :]) * reader.backend.dtype.itemsize
        )
    assert extent is not None
    if steps is None:
        steps = extent
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= extent:
        raise ValueError("steps must select a nonempty prefix")
    per_step = sum(input_bytes) + workspace_bytes_per_step
    count = min(
        steps,
        buffer_bytes // per_step,
        *(
            reader.buffer_bytes // size
            for (reader, _), size in zip(normalized, input_bytes, strict=True)
        ),
    )
    if count < 1:
        raise MemoryError("One aligned step exceeds the input/workspace or reader allowance")
    # Keep room for read_plan's decoded chunks or tiled output assembly when possible.
    count = min(
        count,
        *(
            max(1, r.buffer_bytes // (2 * size))
            for (r, _), size in zip(normalized, input_bytes, strict=True)
        ),
    )
    for start in range(0, steps, count):
        stop = min(steps, start + count)
        values = []
        for reader, axis in normalized:
            if cancelled is not None and cancelled():
                raise InterruptedError("Aligned batch read cancelled")
            selection = tuple(
                slice(start, stop) if i == axis else slice(None)
                for i in range(len(reader.backend.spec.shape))
            )
            values.append(reader.read(selection, stats=stats))
        yield AxisBatch(start, stop, tuple(values))
        del values
