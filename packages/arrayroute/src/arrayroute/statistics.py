# SPDX-License-Identifier: Apache-2.0
"""Mergeable population moments for channel-last scientific buffers."""

from __future__ import annotations

import numpy as np


class ChannelMoments:
    """Accumulate finite observations without retaining past buffers.

    Masks exclude missing observations independently per channel. Counts remain integer;
    means and centred sums use float64 and the parallel variance merge formula. Pass
    bounded buffers: conversion and centred temporary arrays scale with the input block.
    Floating-point results depend on update/merge grouping. Use CanonicalChannelMoments
    when storage block boundaries must not affect the result.
    """

    def __init__(self, channels: int):
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer")
        self.count = np.zeros(channels, dtype=np.int64)
        self.mean = np.zeros(channels, dtype=np.float64)
        self.m2 = np.zeros(channels, dtype=np.float64)

    def update(self, values: np.ndarray, valid: np.ndarray | None = None) -> None:
        if values.ndim < 1 or values.shape[-1] != len(self.count):
            raise ValueError("Expected channel-last values matching the accumulator")
        x = np.asarray(values, dtype=np.float64).reshape(-1, len(self.count))
        if valid is None:
            mask = np.ones(x.shape, dtype=bool)
        else:
            if valid.dtype != np.bool_ or valid.shape != values.shape:
                raise ValueError("Validity must be a boolean array with the same shape")
            mask = valid.reshape(x.shape)
        if not np.isfinite(x[mask]).all():
            raise ValueError("Nonfinite observations inside the validity mask")
        count = mask.sum(axis=0, dtype=np.int64)
        if not count.any():
            return
        # Subtract an observed origin before summing, avoiding a large DC offset in means.
        origin = x[np.argmax(mask, axis=0), np.arange(x.shape[1])]
        origin = np.where(count > 0, origin, 0)
        shifted = np.where(mask, x, origin) - origin
        mean = origin + shifted.sum(axis=0) / np.maximum(count, 1)
        centred = np.where(mask, x, mean) - mean
        other = ChannelMoments(len(self.count))
        other.count, other.mean, other.m2 = count, mean, (centred * centred).sum(axis=0)
        self.merge(other)

    def merge(self, other: ChannelMoments) -> None:
        if other.count.shape != self.count.shape:
            raise ValueError("Cannot merge accumulators with different channel counts")
        if np.any(other.count > np.iinfo(np.int64).max - self.count):
            raise OverflowError("Observation count exceeds int64 capacity")
        total = self.count + other.count
        delta = other.mean - self.mean
        denominator = np.maximum(total, 1)
        self.m2 += other.m2 + delta**2 * (self.count / denominator) * other.count
        self.mean += delta * (other.count / denominator)
        self.count = total

    @property
    def variance(self) -> np.ndarray:
        if np.any(self.count == 0):
            raise ValueError("No valid observations for one or more channels")
        return np.maximum(self.m2 / self.count, 0)


class CanonicalChannelMoments:
    """Versioned, fixed-leaf reduction of an ordered channel-last stream.

    Read boundaries, array strides and intermediate snapshots do not affect the result.
    Callers must preserve sample/row order and pin NumPy/runtime for bitwise replay;
    arbitrary worker completion order and cross-platform equivalence are not promised.
    Each leaf owns C-contiguous float64 values and boolean validity, bounded to 4096
    rows. Partial leaves persist across updates; snapshots never commit a partial leaf.
    """

    ALGORITHM = "canonical-origin-moments-4096-v1"
    LEAF_ROWS = 4096

    def __init__(self, channels: int):
        self._moments = ChannelMoments(channels)
        self._values = np.empty((self.LEAF_ROWS, channels), dtype=np.float64)
        self._valid = np.empty((self.LEAF_ROWS, channels), dtype=bool)
        self._pending = 0
        self._rows_seen = 0
        self._origin = np.zeros(channels, dtype=np.float64)
        self._origin_set = np.zeros(channels, dtype=bool)

    def update(self, values: np.ndarray, valid: np.ndarray | None = None) -> None:
        channels = len(self._moments.count)
        if values.ndim < 1 or values.shape[-1] != channels:
            raise ValueError("Expected channel-last values matching the accumulator")
        if values.dtype.kind not in "fiu" or values.dtype.itemsize > 8:
            raise ValueError("Expected real numeric values of at most 64 bits")
        if valid is not None and (valid.dtype != np.bool_ or valid.shape != values.shape):
            raise ValueError("Validity must be a boolean array with the same shape")
        if not np.all(np.isfinite(values) | (False if valid is None else ~valid)):
            raise ValueError("Nonfinite observations inside the validity mask")
        rows = values.reshape(-1, channels)
        if len(rows) > np.iinfo(np.int64).max - self._rows_seen:
            raise OverflowError("Observation row count exceeds int64 capacity")
        mask = None if valid is None else valid.reshape(-1, channels)
        first = 0
        while first < len(rows):
            if self._pending == 0 and len(rows) - first >= self.LEAF_ROWS:
                # Batch independent leaves, never the order of their final fold.
                capacity = max(1, min(16, (8 << 20) // (self.LEAF_ROWS * channels * 8)))
                leaves = min(capacity, (len(rows) - first) // self.LEAF_ROWS)
                stop = first + leaves * self.LEAF_ROWS
                self._accumulate_leaves(
                    rows[first:stop], None if mask is None else mask[first:stop]
                )
                self._rows_seen += stop - first
                first = stop
                continue
            count = min(len(rows) - first, self.LEAF_ROWS - self._pending)
            target = slice(self._pending, self._pending + count)
            self._values[target] = rows[first : first + count]
            self._valid[target] = True if mask is None else mask[first : first + count]
            self._pending += count
            first += count
            if self._pending == self.LEAF_ROWS:
                self._accumulate(
                    self._moments, self._values, self._valid, self._origin, self._origin_set
                )
                self._pending = 0
            self._rows_seen += count

    @property
    def rows_seen(self) -> int:
        """Consumed rows, including masked rows; excludes prefetched/unsubmitted data."""
        return self._rows_seen

    def checkpoint(self, *, contract_digest: str) -> bytes:
        """Serialize exact state bound to a SHA-256 pipeline contract.

        The caller's contract must pin immutable inputs, row order, transformations,
        masks and numerical environment. Persist these bytes atomically; restart the
        same stream at ``rows_seen``. A checksum detects corruption, not tampering.
        """
        from ._moments_checkpoint import encode

        return encode(self, contract_digest)

    @classmethod
    def from_checkpoint(
        cls, data: bytes, *, contract_digest: str, max_state_bytes: int = 64 << 20,
    ) -> CanonicalChannelMoments:
        """Restore after integrity, contract, runtime and allocation-limit checks."""
        from ._moments_checkpoint import decode

        return decode(cls, data, contract_digest, max_state_bytes)

    def _accumulate_leaves(self, rows: np.ndarray, mask: np.ndarray | None) -> None:
        channels = len(self._origin)
        values = np.asarray(rows, dtype=np.float64, order="C").reshape(
            -1, self.LEAF_ROWS, channels
        )
        if mask is None:
            self._origin[:] = np.where(self._origin_set, self._origin, values[0, 0])
            self._origin_set[:] = True
            x = values - self._origin
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite observations inside the validity mask")
            count = np.full((len(values), channels), self.LEAF_ROWS, dtype=np.int64)
            origin = x[:, 0, :]
            shifted = x - origin[:, None, :]
            mean = origin + shifted.sum(axis=1) / count
            centred = x - mean[:, None, :]
        else:
            valid = np.ascontiguousarray(mask).reshape(values.shape)
            count = valid.sum(axis=1, dtype=np.int64)
            active = count.any(axis=1)
            if not active.any():
                return
            if not active.all():
                values, valid, count = values[active], valid[active], count[active]
            if (count == self.LEAF_ROWS).all():
                self._accumulate_leaves(values.reshape(-1, channels), None)
                return
            if not self._origin_set.all():
                flat = valid.reshape(-1, channels)
                observed = flat.any(axis=0)
                first = values.reshape(-1, channels)[np.argmax(flat, axis=0), np.arange(channels)]
                self._origin[:] = np.where(~self._origin_set & observed, first, self._origin)
                self._origin_set |= observed
            x = np.where(valid, values, self._origin) - self._origin
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite observations inside the validity mask")
            origin = x[
                np.arange(len(values))[:, None], np.argmax(valid, axis=1),
                np.arange(channels)[None, :],
            ]
            origin = np.where(count > 0, origin, 0)
            shifted = np.where(valid, x, origin[:, None, :]) - origin[:, None, :]
            mean = origin + shifted.sum(axis=1) / np.maximum(count, 1)
            centred = np.where(valid, x, mean[:, None, :]) - mean[:, None, :]
        m2 = (centred * centred).sum(axis=1)
        other = ChannelMoments(channels)
        for index in range(len(values)):
            if count[index].any():
                other.count, other.mean, other.m2 = count[index], mean[index], m2[index]
                self._moments.merge(other)

    @staticmethod
    def _accumulate(
        moments: ChannelMoments, values: np.ndarray, valid: np.ndarray,
        origin: np.ndarray, initialized: np.ndarray,
    ) -> None:
        # Keep leaf means near zero: adding a large DC offset before merging loses
        # low bits of between-leaf differences even when each leaf is centred.
        if not initialized.all():
            observed = valid.any(axis=0)
            first = values[np.argmax(valid, axis=0), np.arange(values.shape[1])]
            origin[:] = np.where(~initialized & observed, first, origin)
            initialized |= observed
        moments.update(np.where(valid, values, origin) - origin, valid)

    def snapshot(self) -> ChannelMoments:
        """Return independent moments without changing future reduction boundaries."""
        result = ChannelMoments(len(self._moments.count))
        result.count = self._moments.count.copy()
        result.mean = self._moments.mean.copy()
        result.m2 = self._moments.m2.copy()
        origin, initialized = self._origin.copy(), self._origin_set.copy()
        if self._pending:
            self._accumulate(
                result, self._values[: self._pending], self._valid[: self._pending],
                origin, initialized,
            )
        result.mean += origin
        return result

    @property
    def count(self) -> np.ndarray:
        return self.snapshot().count

    @property
    def mean(self) -> np.ndarray:
        return self.snapshot().mean

    @property
    def variance(self) -> np.ndarray:
        return self.snapshot().variance

    @property
    def mean_square(self) -> np.ndarray:
        moments = self.snapshot()
        return moments.variance + moments.mean**2
