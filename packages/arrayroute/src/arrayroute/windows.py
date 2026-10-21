# SPDX-License-Identifier: Apache-2.0
"""Random access to time windows without materializing every window's index."""

from __future__ import annotations

import operator
import sys
from bisect import bisect_right
from collections.abc import Iterable, Iterator


class WindowIndex:
    """Map a window number to (trajectory number, last input frame).

    Metadata is O(trajectories), lookup O(log trajectories). Targets at multiples of
    ``lead`` never cross trajectory boundaries. Sampling/shuffling remains the caller's
    responsibility; this index preserves the same canonical order on every worker.
    """

    def __init__(
        self, lengths: Iterable[int], *, history: int, unroll: int, lead: int, step: int = 1
    ):
        if any(isinstance(v, bool) or operator.index(v) < 1 for v in (history, unroll, lead, step)):
            raise ValueError("history, unroll, lead and step must be positive integers")
        self.history, self.step = history, step
        self._ends: list[int] = []
        total = 0
        for length in lengths:
            if isinstance(length, bool) or operator.index(length) < 0:
                raise ValueError("Trajectory lengths must be nonnegative integers")
            available = length - lead * unroll - history
            total += max(0, available // step + 1)
            if total > sys.maxsize:
                raise OverflowError("Window count exceeds platform index capacity")
            self._ends.append(total)

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int) -> tuple[int, int]:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Window index out of range")
        trajectory = bisect_right(self._ends, index)
        start = self._ends[trajectory - 1] if trajectory else 0
        return trajectory, self.history - 1 + (index - start) * self.step

    def __iter__(self) -> Iterator[tuple[int, int]]:
        previous = 0
        for trajectory, end in enumerate(self._ends):
            for offset in range(end - previous):
                yield trajectory, self.history - 1 + offset * self.step
            previous = end
