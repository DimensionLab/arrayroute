# SPDX-License-Identifier: Apache-2.0
"""A streaming execution contract; backends own their I/O and cache policy."""

from __future__ import annotations

import time
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .budget import ReadBudget
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
    def __init__(
        self, backend: Backend, *, buffer_bytes: int = 128 << 20,
        shared_budget: ReadBudget | None = None,
    ):
        if buffer_bytes < 1:
            raise ValueError("buffer_bytes must be positive")
        self.backend = backend
        self.buffer_bytes = (min(buffer_bytes, shared_budget.max_bytes)
                             if shared_budget is not None else buffer_bytes)
        self.shared_budget = shared_budget
        if backend.spec.dense_gather and not callable(getattr(backend, "read_plan", None)):
            raise TypeError("Dense gather metadata requires a backend with read_plan")
        self._execute: Callable[[ReadPlan], np.ndarray] = getattr(
            backend, "read_plan", lambda request: backend.read(request.selection))
        if not callable(self._execute):
            raise TypeError("Backend read_plan must be callable")

    def plan(self, selection: Selection) -> ReadPlan:
        return plan(self.backend.spec, selection, buffer_bytes=self.buffer_bytes)

    def stream(
        self,
        selection: Selection,
        *,
        stats: ReadStats | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Generator[tuple[Tile, np.ndarray], None, None]:
        for tile in tiles(self.backend.spec, selection, buffer_bytes=self.buffer_bytes):
            if cancelled is not None and cancelled():
                raise InterruptedError("Array read cancelled")
            reservation = (
                self.shared_budget.reserve(tile.plan.buffer_bytes_upper_bound, cancelled=cancelled)
                if self.shared_budget is not None else nullcontext()
            )
            with reservation:
                start = time.perf_counter()
                value = self._execute(tile.plan)
                if value.shape != tile.plan.output_shape or value.dtype != self.backend.dtype:
                    raise ValueError("Backend returned an unexpected shape or dtype")
                if stats is not None:
                    stats.requests += 1
                    stats.seconds += time.perf_counter() - start
                    stats.output_bytes += value.nbytes
                    stats.decoded_bytes_upper_bound += tile.plan.decoded_bytes_upper_bound
                yield tile, value
                # Release the previous buffer before returning its reservation.
                del value

    def read(self, selection: Selection, *, stats: ReadStats | None = None) -> np.ndarray:
        request = self.plan(selection)
        if self.shared_budget is not None:
            # Admit the entire materialization before allocating its output. Reserving
            # outputs separately from tiles can deadlock concurrent materializations.
            amount = min(request.buffer_bytes_upper_bound, self.buffer_bytes)
            with self.shared_budget.reserve(amount):
                return ArrayReader(self.backend, buffer_bytes=self.buffer_bytes).read(
                    selection, stats=stats)
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

    @contextmanager
    def prefetch(
        self,
        selection: Selection,
        *,
        stats: ReadStats | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[Iterator[tuple[Tile, np.ndarray]]]:
        """Overlap one ordered read with the consumer using two budgeted slots.

        Each slot receives half of the effective buffer for output, intermediates and
        estimated decoded chunks. A shared pool caps the effective reader buffer.
        Backend caches/codec scratch and arrays retained by the consumer
        beyond the current tile are additional. A chunk must fit in one slot.
        One worker owns all backend reads; do not otherwise read that backend while
        this context is active. Closing waits for an in-flight read: cancellation
        prevents further reads but cannot interrupt a backend's blocking I/O.
        Statistics include prefetched reads even if the consumer stops early.
        The cancellation callback must be thread-safe. Release each value before
        advancing if retaining a previous tile would exceed the consumer's budget.
        """
        if self.shared_budget is not None:
            # Both slots remain owned until context exit, even when the worker has
            # advanced its generator while the consumer still holds the previous tile.
            with self.shared_budget.reserve(self.buffer_bytes, cancelled=cancelled):
                reader = ArrayReader(self.backend, buffer_bytes=self.buffer_bytes)
                with reader.prefetch(selection, stats=stats, cancelled=cancelled) as output:
                    yield output
            return
        budget = self.buffer_bytes // 2
        if budget < 1:
            raise ValueError("Prefetch requires a budget for two read slots")
        reader = ArrayReader(self.backend, buffer_bytes=budget)
        source = reader.stream(selection, stats=stats, cancelled=cancelled)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arrayroute-read")

        def advance() -> tuple[Tile, np.ndarray] | None:
            return next(source, None)

        def ordered() -> Generator[tuple[Tile, np.ndarray], None, None]:
            pending = executor.submit(advance)
            while True:
                if cancelled is not None and cancelled():
                    raise InterruptedError("Array read cancelled")
                current = pending.result()
                if current is None:
                    return
                pending = executor.submit(advance)
                yield current
                del current

        output = ordered()
        try:
            yield output
        finally:
            # A running generator cannot be closed until its worker has returned.
            executor.shutdown(wait=True, cancel_futures=True)
            output.close()
            source.close()
