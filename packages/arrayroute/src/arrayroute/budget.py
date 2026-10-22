# SPDX-License-Identifier: Apache-2.0
"""Process-local admission control for planned decoded and output buffers."""
from __future__ import annotations

import os
import threading
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from operator import index


class ReadBudget:
    """Share a planned-buffer allowance across independent reader threads.

    Accounts for planned output, intermediates and live decoded chunks, not RSS.
    Encoded transport, native caches, codec scratch and caller-retained outputs are
    additional. FIFO admission avoids indefinitely bypassing a larger queued read.
    Construct one pool per process after spawn. Nested reservations in one thread
    are rejected: holding an output while requesting another can otherwise deadlock.
    """

    def __init__(self, max_bytes: int):
        if isinstance(max_bytes, bool) or index(max_bytes) < 1:
            raise ValueError('Read budget must be a positive integer byte count')
        self._max_bytes = index(max_bytes)
        self._pid = os.getpid()
        self._condition = threading.Condition()
        self._waiters: deque[object] = deque()
        self._owners: set[int] = set()
        self._used = self._peak = self._peak_active = 0

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def _check_process(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError('Create a new ReadBudget in each process after spawn')

    def __getstate__(self):
        raise TypeError('ReadBudget is process-local and cannot be serialized')

    def snapshot(self) -> dict[str, int]:
        self._check_process()
        with self._condition:
            return dict(max_bytes=self.max_bytes, used_bytes=self._used,
                        peak_bytes=self._peak, active=len(self._owners),
                        peak_active=self._peak_active, waiting=len(self._waiters))

    @contextmanager
    def reserve(
        self, amount: int, *, cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[None]:
        self._check_process()
        if isinstance(amount, bool) or index(amount) < 0:
            raise ValueError('Reservation must be a nonnegative integer byte count')
        amount = index(amount)
        if amount > self.max_bytes:
            raise MemoryError('Read exceeds shared budget; lower the reader buffer or use stream')
        if amount == 0:
            yield
            return
        owner = threading.get_ident()
        token = object()
        with self._condition:
            if owner in self._owners:
                raise RuntimeError('Release the current read before nesting another reservation')
            self._waiters.append(token)
            try:
                while True:
                    if cancelled is not None and cancelled():
                        raise InterruptedError('Read cancelled while awaiting shared buffers')
                    if self._waiters[0] is token and self._used + amount <= self.max_bytes:
                        self._waiters.popleft()
                        self._owners.add(owner)
                        self._used += amount
                        self._peak = max(self._peak, self._used)
                        self._peak_active = max(self._peak_active, len(self._owners))
                        self._condition.notify_all()
                        break
                    self._condition.wait(timeout=.05 if cancelled is not None else None)
            except BaseException:
                self._waiters.remove(token)
                self._condition.notify_all()
                raise
        try:
            yield
        finally:
            with self._condition:
                self._used -= amount
                self._owners.remove(owner)
                self._condition.notify_all()
