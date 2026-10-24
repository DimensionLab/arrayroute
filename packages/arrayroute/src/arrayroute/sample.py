# SPDX-License-Identifier: Apache-2.0
"""Seeded, bounded row samples with replay independent of input batch boundaries."""

from __future__ import annotations

import operator

import numpy as np

ALGORITHM = "pcg64-priority-rows-v1"


class PrioritySample:
    """Keep the smallest seeded uint64 priorities, with input position breaking ties.

    Every row consumes one PCG64 raw draw. The same seed and ordered input produce the
    same selected rows across update boundaries, on the same RNG/numerical implementation.
    This is a reproducible pseudorandom sample, not a cryptographic primitive or a promise
    of cross-version RNG identity. Record ALGORITHM, seed, capacity and input identity.

    Retained arrays occupy capacity * (row bytes + 16), excluding Python overhead.
    An update additionally holds its input and O(capacity + batch rows) merge/partition
    buffers; callers must bound update batches. No full-stream history is retained.
    """

    def __init__(self, capacity: int, *, seed: int = 0):
        if isinstance(capacity, bool) or operator.index(capacity) < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = operator.index(capacity)
        self.seed = operator.index(seed)
        self.seen = 0
        self._rng = np.random.PCG64(self.seed)
        self._values: np.ndarray | None = None
        self._keys = np.empty(0, dtype=np.uint64)
        self._positions = np.empty(0, dtype=np.uint64)

    def update(self, values: np.ndarray) -> None:
        a = np.asarray(values)
        if a.ndim < 1 or a.dtype.hasobject or a.dtype.itemsize < 1:
            raise TypeError("Samples require rows of fixed-size, non-object values")
        if self._values is not None and (
            a.shape[1:] != self._values.shape[1:] or a.dtype != self._values.dtype
        ):
            raise ValueError("Row shape and dtype must remain fixed")
        stop = self.seen + len(a)
        if stop >= 1 << 64:
            raise OverflowError("Sample input positions exceed uint64")
        next_rng = np.random.PCG64(0)
        next_rng.state = self._rng.state
        keys = np.concatenate((self._keys, next_rng.random_raw(len(a))))
        positions = np.concatenate((self._positions,
                                    np.arange(self.seen, stop, dtype=np.uint64)))
        merged = a.copy() if self._values is None else np.concatenate((self._values, a))
        if len(keys) > self.capacity:
            threshold = np.partition(keys, self.capacity - 1)[self.capacity - 1]
            lower = np.flatnonzero(keys < threshold)
            tied = np.flatnonzero(keys == threshold)
            tied = tied[np.argsort(positions[tied], kind="stable")]
            selected = np.concatenate((lower, tied[:self.capacity - len(lower)]))
            keys, positions, merged = keys[selected], positions[selected], merged[selected]
        self._keys, self._positions, self._values = keys, positions, merged
        self._rng = next_rng
        self.seen = stop

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """Return owned rows and uint64 input positions, in original stream order."""
        if self._values is None:
            raise ValueError("No input row schema has been supplied")
        order = np.argsort(self._positions, kind="stable")
        return self._values[order], self._positions[order]
