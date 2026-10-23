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
    a task-count bound, not an RSS cap. Optional byte admission also bounds the sum
    of caller-supplied task estimates until each result is consumed. Estimates must
    cover the memory they claim; caches, scratch and caller-retained values remain
    additional. One not-yet-admitted input may be retained for lookahead. Factories
    must not return shared mutable readers.

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
        max_pending_bytes: int | None = None,
        estimate_bytes: Callable[[T], int] | None = None,
    ) -> None:
        if isinstance(workers, bool) or isinstance(max_pending, bool):
            raise ValueError("workers and max_pending must be positive integers")
        workers = operator.index(workers)
        capacity = workers if max_pending is None else operator.index(max_pending)
        if workers < 1 or capacity < 1:
            raise ValueError("workers and max_pending must be positive integers")
        if (max_pending_bytes is None) != (estimate_bytes is None):
            raise ValueError("Byte admission requires both max_pending_bytes and estimate_bytes")
        if max_pending_bytes is not None:
            if isinstance(max_pending_bytes, bool) or operator.index(max_pending_bytes) < 1:
                raise ValueError("max_pending_bytes must be a positive integer")
            max_pending_bytes = operator.index(max_pending_bytes)
        self._make_worker = make_worker
        self._capacity = capacity
        self._byte_capacity = max_pending_bytes
        self._estimate_bytes = estimate_bytes
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
        pending: deque[tuple[Future[R], int]] = deque()

        def check() -> None:
            if self._closed:
                raise RuntimeError("Ordered pool is closed")
            if cancelled is not None and cancelled():
                raise ProcessingCancelledError("Parallel processing cancelled")

        def iterate() -> Generator[R, None, None]:
            source = iter(items)
            exhausted = False
            deferred: tuple[T, int] | None = None
            admitted_bytes = 0
            while True:
                check()
                while not exhausted and len(pending) < self._capacity:
                    check()
                    if deferred is None:
                        try:
                            item = next(source)
                        except StopIteration:
                            exhausted = True
                            break
                        amount = 0 if self._estimate_bytes is None else self._estimate_bytes(item)
                        if isinstance(amount, bool) or operator.index(amount) < 0:
                            raise ValueError("Task byte estimate must be a nonnegative integer")
                        amount = operator.index(amount)
                        if self._byte_capacity is not None and amount > self._byte_capacity:
                            raise MemoryError(
                                f"Task requires {amount} planned bytes; pending-byte allowance "
                                f"is {self._byte_capacity}. "
                                "Reduce the task size or raise the allowance."
                            )
                    else:
                        item, amount = deferred
                        deferred = None
                    if (self._byte_capacity is not None
                            and admitted_bytes + amount > self._byte_capacity):
                        deferred = (item, amount)
                        del item
                        break
                    pending.append((self._executor.submit(self._execute, item, check), amount))
                    admitted_bytes += amount
                    del item
                if not pending:
                    return
                # Keep the awaited future in the deque so cleanup also waits for it
                # if the consumer is interrupted while future.result() is waiting.
                future, amount = pending[0]
                result = future.result()
                pending.popleft()
                del future
                check()
                yield result
                del result
                admitted_bytes -= amount

        output = iterate()
        try:
            yield output
        finally:
            try:
                for future, _ in pending:
                    future.cancel()
                if pending:
                    wait(future for future, _ in pending)
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
    max_pending_bytes: int | None = None,
    estimate_bytes: Callable[[T], int] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Iterator[Iterator[R]]:
    """Run one bounded pass and release its thread-owned workers on context exit.

    See ``OrderedPool`` for ordering, memory and cancellation guarantees. Use a pool
    explicitly when repeated passes should retain the same native readers and caches.
    """
    with OrderedPool(make_worker, workers=workers, max_pending=max_pending,
                     max_pending_bytes=max_pending_bytes, estimate_bytes=estimate_bytes) as pool:
        with pool.map(items, cancelled=cancelled) as output:
            yield output
