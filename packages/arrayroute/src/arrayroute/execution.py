# SPDX-License-Identifier: Apache-2.0
"""Ordered, bounded execution with worker-owned readers or transforms."""

from __future__ import annotations

import operator
import threading
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ProcessingCancelledError(InterruptedError):
    """The caller's cancellation callback stopped bounded execution."""


@contextmanager
def ordered_map(
    make_worker: Callable[[], Callable[[T], R]],
    items: Iterable[T],
    *,
    workers: int,
    max_pending: int | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Iterator[Iterator[R]]:
    """Map lazily in input order, constructing one callable on each active thread.

    At most ``max_pending`` tasks/results are owned by this context, including its
    currently yielded result. Caller-retained results, native caches and scratch are
    additional. Tasks must have bounded memory; this is a count bound, not an RSS cap.
    Factories must not return shared mutable readers. A worker is reused on its owning
    thread and released when that thread exits. The caller owns the input iterable.

    Task errors are raised at their input position; source errors stop iteration.
    Early close/cancellation cancels queued
    work and joins active threads; it cannot interrupt a blocking native call. Use an
    external process deadline when a hard time limit is required. No results are skipped
    or replaced on failure, and completed results cannot overtake an earlier task.
    """
    if isinstance(workers, bool) or isinstance(max_pending, bool):
        raise ValueError("workers and max_pending must be positive integers")
    workers = operator.index(workers)
    capacity = workers if max_pending is None else operator.index(max_pending)
    if workers < 1 or capacity < 1:
        raise ValueError("workers and max_pending must be positive integers")
    local = threading.local()
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="arrayroute-map")
    pending: deque[Future[R]] = deque()

    def check() -> None:
        if cancelled is not None and cancelled():
            raise ProcessingCancelledError("Parallel processing cancelled")

    def execute(item: T) -> R:
        check()
        if not hasattr(local, "process"):
            local.process = make_worker()
        return local.process(item)

    def iterate() -> Generator[R, None, None]:
        source = iter(items)
        exhausted = False
        while True:
            check()
            while not exhausted and len(pending) < capacity:
                check()
                try:
                    item = next(source)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(executor.submit(execute, item))
                del item
            if not pending:
                return
            future = pending.popleft()
            result = future.result()
            del future
            check()
            yield result
            del result

    output = iterate()
    try:
        yield output
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        output.close()
        pending.clear()
