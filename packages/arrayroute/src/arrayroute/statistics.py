# SPDX-License-Identifier: Apache-2.0
"""Mergeable population moments for channel-last scientific buffers."""

from __future__ import annotations

import numpy as np


class ChannelMoments:
    """Accumulate finite observations without retaining past buffers.

    Masks exclude missing observations independently per channel. Counts remain integer;
    means and centred sums use float64 and the parallel variance merge formula. Pass
    bounded buffers: conversion and centred temporary arrays scale with the input block.
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
