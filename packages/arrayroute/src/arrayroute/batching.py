# SPDX-License-Identifier: Apache-2.0
"""Deterministic overlap screening for batches of leading-axis selections."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RowBatchPlan:
    requested_rows: int
    unique_rows: tuple[int, ...]
    combine: bool
    min_reuse_fraction: float

    @property
    def reuse_fraction(self) -> float:
        return (1 - len(self.unique_rows) / self.requested_rows
                if self.requested_rows else 0.0)


def plan_row_batch(
    selections: Sequence[Sequence[int]], *, row_count: int, min_reuse_fraction: float = 0.25,
) -> RowBatchPlan:
    """Screen combined reads by reusable rows, without I/O or sampler reordering.

    ``unique_rows`` retains first-occurrence order, normalizing negative indices.
    Combine only if multiple selections request repeated rows and the saved fraction meets
    the caller's threshold. This is an overlap heuristic, not a latency prediction:
    decode cost, layout, copy overhead and backend caches still require measurement.
    Metadata size is O(requested batch rows), independent of the array's row count.
    """
    if isinstance(row_count, (bool, np.bool_)) or operator.index(row_count) < 0:
        raise ValueError("row_count must be a nonnegative integer")
    row_count = operator.index(row_count)
    if (isinstance(min_reuse_fraction, (bool, np.bool_))
            or not math.isfinite(min_reuse_fraction) or not 0 <= min_reuse_fraction <= 1):
        raise ValueError("min_reuse_fraction must be finite and between zero and one")
    unique: dict[int, None] = {}
    total = 0
    for selection in selections:
        for value in selection:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError("Row indices must be integers, not booleans")
            row = operator.index(value)
            if row < 0:
                row += row_count
            if not 0 <= row < row_count:
                raise IndexError("Row index out of range")
            unique[row] = None
            total += 1
    saved = total - len(unique)
    combine = len(selections) > 1 and saved > 0 and saved >= min_reuse_fraction * total
    return RowBatchPlan(total, tuple(unique), combine, float(min_reuse_fraction))
