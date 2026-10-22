# SPDX-License-Identifier: Apache-2.0
"""Ordered, bounded execution with worker-owned readers or transforms."""

from __future__ import annotations

import operator
import threading
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ProcessingCancelledError(InterruptedError):
    """The caller's cancellation callback stopped bounded execution."""


class OrderedPool(Generic[T, R]):
    """Reuse thread-owned readers across exclusive, bounded passes.

    Each active thread constructs one callable and keeps it until this pool closes.
    A pass owns at most ``max_pending`` tasks/results, including its yielded result.
    Caller-retained values, native caches and scratch are additional memory. This is
    a task-count bound, not an RSS cap. Factories must not return shared mutable readers.

    Map contexts are exclusive, including before consumption starts. Task failures
    surface in input order; no task is retried or skipped. Leaving a map context
    cancels queued work and waits for active calls before permitting another pass.
    Closing the pool also joins its threads. Neither operation can interrupt a blocked
    native call; use an external process deadline when a hard time limit is required.
    """

    def __init__(
        self,
        make_worker: Callable[[], Callable[[T], R]],
        *,
        workers: int,
        max_pending: int | None = None,
    ) -> None:
        if isinstance(workers, bool) or isinstance(max_pending, bool):
            raise ValueError("workers and max_pending must be positive integers")
        workers = operator.index(workers)
        capacity = workers if max_pending is None else operator.index(max_pending)
        if workers < 1 or capacity < 1:
            raise ValueError("workers and max_pending must be positive integers")
        self._make_worker = make_worker
        self._capacity = capacity
        self._local = threading.local()
        self._state_lock = threading.Lock()
        self._active = False
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="arrayroute-map"
        )

    def _execute(self, item: T, check: Callable[[], None]) -> R:
        check()
        if not hasattr(self._local, "process"):
            self._local.process = self._make_worker()
        return self._local.process(item)

    @contextmanager
    def map(
        self, items: Iterable[T], *, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[Iterator[R]]:
        """Consume in input order; the caller retains ownership of the input iterable."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Ordered pool is closed")
            if self._active:
                raise RuntimeError("Concurrent map contexts are not supported")
            self._active = True
        pending: deque[Future[R]] = deque()

        def check() -> None:
            if self._closed:
                raise RuntimeError("Ordered pool is closed")
            if cancelled is not None and cancelled():
                raise ProcessingCancelledError("Parallel processing cancelled")

        def iterate() -> Generator[R, None, None]:
            source = iter(items)
            exhausted = False
            while True:
                check()
                while not exhausted and len(pending) < self._capacity:
                    check()
                    try:
                        item = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.append(self._executor.submit(self._execute, item, check))
                    del item
                if not pending:
                    return
                # Keep the awaited future in the deque so cleanup also waits for it
                # if the consumer is interrupted while future.result() is waiting.
                future = pending[0]
                result = future.result()
                pending.popleft()
                del future
                check()
                yield result
                del result

        output = iterate()
        try:
            yield output
        finally:
            try:
                for future in pending:
                    future.cancel()
                if pending:
                    wait(pending)
                output.close()
                pending.clear()
            finally:
                with self._state_lock:
                    self._active = False

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
        # Drain the queue: shutdown(cancel_futures=True) removes cancelled futures
        # before workers notify waiters. Queued calls fail check() without doing I/O.
        self._executor.shutdown(wait=True)

    def __enter__(self) -> OrderedPool[T, R]:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Ordered pool is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@contextmanager
def ordered_map(
    make_worker: Callable[[], Callable[[T], R]],
    items: Iterable[T],
    *,
    workers: int,
    max_pending: int | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Iterator[Iterator[R]]:
    """Run one bounded pass and release its thread-owned workers on context exit.

    See ``OrderedPool`` for ordering, memory and cancellation guarantees. Use a pool
    explicitly when repeated passes should retain the same native readers and caches.
    """
    with OrderedPool(make_worker, workers=workers, max_pending=max_pending) as pool:
        with pool.map(items, cancelled=cancelled) as output:
            yield output
