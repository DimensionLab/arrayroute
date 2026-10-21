# SPDX-License-Identifier: Apache-2.0
"""A streaming execution contract; backends own their I/O and cache policy."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .plan import ArraySpec, ReadPlan, Selection, Tile, plan, tiles


class Backend(Protocol):
    spec: ArraySpec
    dtype: np.dtype

    def read(self, selection: Selection) -> np.ndarray: ...


@dataclass
class ReadStats:
    requests: int = 0
    output_bytes: int = 0
    decoded_bytes_upper_bound: int = 0
    seconds: float = 0.0


class ArrayReader:
    def __init__(self, backend: Backend, *, buffer_bytes: int = 128 << 20):
        if buffer_bytes < 1:
            raise ValueError("buffer_bytes must be positive")
        self.backend = backend
        self.buffer_bytes = buffer_bytes

    def plan(self, selection: Selection) -> ReadPlan:
        return plan(self.backend.spec, selection)

    def stream(
        self,
        selection: Selection,
        *,
        stats: ReadStats | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[Tile, np.ndarray]]:
        for tile in tiles(self.backend.spec, selection, buffer_bytes=self.buffer_bytes):
            if cancelled is not None and cancelled():
                raise InterruptedError("Array read cancelled")
            start = time.perf_counter()
            value = self.backend.read(tile.plan.selection)
            if value.shape != tile.plan.output_shape or value.dtype != self.backend.dtype:
                raise ValueError("Backend returned an unexpected shape or dtype")
            if stats is not None:
                stats.requests += 1
                stats.seconds += time.perf_counter() - start
                stats.output_bytes += value.nbytes
                stats.decoded_bytes_upper_bound += tile.plan.decoded_bytes_upper_bound
            yield tile, value
            # Do not retain the previous buffer while requesting the next one.
            del value

    def read(self, selection: Selection, *, stats: ReadStats | None = None) -> np.ndarray:
        request = self.plan(selection)
        if request.buffer_bytes_upper_bound <= self.buffer_bytes:
            chunks = self.stream(selection, stats=stats)
            if request.output_bytes:
                return next(chunks)[1]
            return np.empty(request.output_shape, dtype=self.backend.dtype)
        if request.output_bytes >= self.buffer_bytes:
            raise MemoryError("Selection exceeds materialization budget; use stream() or a region")
        result = np.empty(request.output_shape, dtype=self.backend.dtype)
        reader = ArrayReader(self.backend, buffer_bytes=self.buffer_bytes - result.nbytes)
        for tile, value in reader.stream(selection, stats=stats):
            result[tile.destination] = value
            del value
        return result
