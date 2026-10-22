# SPDX-License-Identifier: Apache-2.0
"""Reproducible random-access permutations without a dataset-sized index array.

Uses balanced Feistel rounds and cycle walking (Black and Rogaway, 2002). This is a
sampling primitive, not an encryption API or a uniform draw from all N! permutations.
"""

from __future__ import annotations

import hashlib
import operator
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from itertools import islice
from typing import ClassVar, Literal


class Permutation:
    """Map epoch positions bijectively onto ``range(size)`` with constant retained state.

    Eight keyed BLAKE2b Feistel rounds permute the next enclosing even-bit domain.
    Cycle walking restricts it to ``size``. The enclosing domain is less than four
    times the requested size for size > 1; individual lookups have variable latency.
    No per-sample index, padding, or duplicated tail is allocated. Partition positions
    across workers, not seeds: ``indices(start=rank, step=world_size)``. Such shards
    can have unequal lengths; synchronized training must handle that explicitly.

    Keep size, seed and epoch with a saved position to resume the same order. Input
    files and the canonical ordering of the underlying dataset must stay fixed.
    """

    __slots__ = ("size", "seed", "epoch", "_bits", "_mask", "_keys")
    algorithm = "feistel-blake2b-8-cw-v1"

    def __init__(self, size: int, *, seed: int = 0, epoch: int = 0):
        size, seed, epoch = (operator.index(v) for v in (size, seed, epoch))
        if not 0 <= size <= sys.maxsize:
            raise ValueError("size must be within the platform's nonnegative index range")
        if not 0 <= seed < 1 << 64 or not 0 <= epoch < 1 << 64:
            raise ValueError("seed and epoch must be unsigned 64-bit integers")
        self.size, self.seed, self.epoch = size, seed, epoch
        self._bits = max(1, ((max(1, size) - 1).bit_length() + 1) // 2)
        self._mask = (1 << self._bits) - 1
        key = seed.to_bytes(8, "little") + epoch.to_bytes(8, "little")
        self._keys = tuple(
            hashlib.blake2b(key + bytes([r]), digest_size=32, person=b"arrayroute-prm1").digest()
            for r in range(8)
        )

    def __len__(self) -> int:
        return self.size

    def _round(self, value: int, key: bytes) -> int:
        return int.from_bytes(
            hashlib.blake2b(value.to_bytes(8, "little"), key=key, digest_size=8).digest(),
            "little",
        ) & self._mask

    def _walk(self, value: int, *, inverse: bool = False) -> int:
        while True:
            left, right = value >> self._bits, value & self._mask
            if inverse:
                for key in reversed(self._keys):
                    left, right = right ^ self._round(left, key), left
            else:
                for key in self._keys:
                    left, right = right, left ^ self._round(right, key)
            value = (left << self._bits) | right
            if value < self.size:
                return value

    def _index(self, position: int) -> int:
        position = operator.index(position)
        if position < 0:
            position += self.size
        if not 0 <= position < self.size:
            raise IndexError("Permutation index out of range")
        return position

    def __getitem__(self, position: int) -> int:
        return self._walk(self._index(position)) if self.size > 1 else self._index(position)

    def inverse(self, value: int) -> int:
        """The position that maps to value; useful for locating/resuming an example."""
        return self._walk(self._index(value), inverse=True) if self.size > 1 else self._index(value)

    def indices(self, *, start: int = 0, stop: int | None = None, step: int = 1) -> Iterator[int]:
        start, stop, step = (
            operator.index(start),
            self.size if stop is None else operator.index(stop),
            operator.index(step),
        )
        if start < 0 or not 0 <= stop <= self.size or step < 1:
            raise ValueError("Require nonnegative start, 0 <= stop <= size and a positive step")
        return (self[position] for position in range(start, stop, step))

    def __iter__(self) -> Iterator[int]:
        return self.indices()


@dataclass(frozen=True, slots=True)
class BatchPartition:
    """Rank-local batches of one shared epoch permutation, without padding duplicates.

    ``tail='drop'`` removes the incomplete *global* batch, so all ranks have the
    same number of full batches. ``tail='uneven'`` preserves every sample exactly
    once; consumers must explicitly handle unequal collective/optimizer counts.

    ``start_batch`` is the number of globally consumed batches per rank, not the
    number prefetched. A checkpoint must separately bind this plan to an immutable
    source revision and retain optimizer/RNG state. Changing world size mid-epoch is
    not a supported replay. Iteration retains O(batch_size) indices, not O(size).
    """

    size: int
    rank: int
    world_size: int
    batch_size: int
    tail: Literal["drop", "uneven"]
    seed: int = 0
    epoch: int = 0
    start_batch: int = 0
    _order: Permutation = field(init=False, repr=False, compare=False)
    algorithm: ClassVar[str] = "strided-rank-batches-v1"

    def __post_init__(self) -> None:
        for name in ("size", "rank", "world_size", "batch_size", "seed", "epoch", "start_batch"):
            object.__setattr__(self, name, operator.index(getattr(self, name)))
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("Require world_size >= 1 and 0 <= rank < world_size")
        if not 1 <= self.batch_size <= sys.maxsize:
            raise ValueError("batch_size must be within the platform's positive index range")
        if self.tail not in ("drop", "uneven"):
            raise ValueError("tail must be explicitly 'drop' or 'uneven'")
        object.__setattr__(self, "_order", Permutation(self.size, seed=self.seed, epoch=self.epoch))
        if not 0 <= self.start_batch <= self.global_batches:
            raise ValueError("start_batch must be within the global epoch batch range")

    @property
    def global_batches(self) -> int:
        """Maximum rank batch count before resume; equal on every rank in drop mode."""
        width = self.world_size * self.batch_size
        return self.size // width if self.tail == "drop" else -(-self.size // width)

    @property
    def dropped_samples(self) -> int:
        """Global incomplete-tail count, independent of resume position."""
        return self.size % (self.world_size * self.batch_size) if self.tail == "drop" else 0

    def positions(self) -> range:
        """Unpermuted global positions owned by this rank after the consumed prefix."""
        start = self.start_batch * self.batch_size * self.world_size + self.rank
        return range(start, self.size - self.dropped_samples, self.world_size)

    @property
    def sample_count(self) -> int:
        return len(self.positions())

    def __len__(self) -> int:
        return -(-self.sample_count // self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        indices = (self._order[position] for position in self.positions())
        while batch := list(islice(indices, self.batch_size)):
            yield batch

    def state_dict(self) -> dict[str, int | str]:
        """Portable partition identity. Never advances a cursor on prefetch."""
        return {
            "algorithm": self.algorithm,
            "permutation": Permutation.algorithm,
            **{f.name: getattr(self, f.name) for f in fields(self) if f.init},
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> BatchPartition:
        required = {f.name for f in fields(cls) if f.init} | {"algorithm", "permutation"}
        if set(state) != required:
            raise ValueError("Partition state has missing or unknown fields")
        if state["algorithm"] != cls.algorithm or state["permutation"] != Permutation.algorithm:
            raise ValueError("Partition state uses an unsupported ordering algorithm")
        return cls(**{name: state[name] for name in required - {"algorithm", "permutation"}})
