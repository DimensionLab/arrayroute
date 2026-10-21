# SPDX-License-Identifier: Apache-2.0
"""Byte-accounted LRU admission independent of storage and array frameworks."""

from collections import OrderedDict
from operator import index
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class ByteLRU(Generic[K, V]):
    """Bound retained values by caller-reported bytes and entry count.

    The bound excludes Python bookkeeping and values still held by callers after eviction.
    Values must not grow after insertion. Use a cache per worker; operations are not locked.
    """

    def __init__(self, capacity_bytes: int, *, max_entries: int = 256):
        capacity_bytes, max_entries = index(capacity_bytes), index(max_entries)
        if capacity_bytes < 0 or max_entries < 0:
            raise ValueError("Cache limits must be nonnegative")
        self.capacity_bytes = capacity_bytes
        self.max_entries = max_entries
        self.used_bytes = 0
        self._entries: OrderedDict[K, tuple[V, int]] = OrderedDict()

    def get(self, key: K) -> V | None:
        found = self._entries.get(key)
        if found is None:
            return None
        self._entries.move_to_end(key)
        return found[0]

    def put(self, key: K, value: V, *, size_bytes: int) -> bool:
        size_bytes = index(size_bytes)
        if size_bytes < 0:
            raise ValueError("Entry size must be nonnegative")
        old = self._entries.pop(key, None)
        if old is not None:
            self.used_bytes -= old[1]
        if not self.capacity_bytes or not self.max_entries or size_bytes > self.capacity_bytes:
            return False
        while self._entries and (
            self.used_bytes + size_bytes > self.capacity_bytes
            or len(self._entries) >= self.max_entries
        ):
            _, (_, removed_bytes) = self._entries.popitem(last=False)
            self.used_bytes -= removed_bytes
        self._entries[key] = (value, size_bytes)
        self.used_bytes += size_bytes
        return True

    def __len__(self) -> int:
        return len(self._entries)
