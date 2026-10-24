# SPDX-License-Identifier: Apache-2.0
"""Fixed-block temporal moments and probes accompanying paired error curves."""

from __future__ import annotations

import math

import numpy as np

DIAGNOSTIC_ALGORITHM = "paired-temporal-moments-256-64-v1"


class TemporalDiagnostics:
    def __init__(self, shape: tuple[int, ...], probes: int):
        self.steps, self.channels = shape[0], shape[-1]
        self.points = math.prod(shape[1:-1])
        self.indices = np.unique(
            np.linspace(0, self.points - 1, min(self.points, probes)).astype(int)
        )
        self.probes = [np.empty((self.steps, len(self.indices), self.channels)) for _ in range(2)]
        self.totals = np.zeros(6)
        self.origin: list[np.ndarray] = []
        self.mean: list[np.ndarray] = []
        self.m2: list[np.ndarray] = []
        self.window_offset = 0

    def begin(self, shape: tuple[int, ...], offset: int) -> None:
        self.window_offset = offset
        size = math.prod(shape[1:-1])
        self.origin, self.mean, self.m2 = (
            [np.zeros((size, self.channels)) for _ in range(2)] for _ in range(3)
        )

    def update(self, start: int, offset: int, pred: np.ndarray, truth: np.ndarray) -> None:
        count = pred.shape[1]
        region = slice(offset, offset + count)
        absolute = self.window_offset + offset
        which = np.flatnonzero((self.indices >= absolute) & (self.indices < absolute + count))
        for role, values in enumerate((pred, truth)):
            if start == 0:
                self.origin[role][region] = values[0]
            shifted = values - self.origin[role][region]
            mean = shifted.mean(0)
            delta = mean - self.mean[role][region]
            centred = shifted - mean
            k = len(values)
            self.m2[role][region] += np.einsum("tmc,tmc->mc", centred, centred)
            self.m2[role][region] += delta * delta * (start * k / (start + k))
            self.mean[role][region] += delta * (k / (start + k))
            if len(which):
                self.probes[role][start : start + k, which] = values[
                    :, self.indices[which] - absolute
                ]

    def reduce(self, offset: int, count: int) -> None:
        region = slice(offset, offset + count)
        pm, tm = (self.mean[i][region] + self.origin[i][region] for i in range(2))
        ps, ts = (np.sqrt(np.maximum(self.m2[i][region] / self.steps, 0)) for i in range(2))
        self.totals += np.array(
            [
                np.square(pm - tm).sum(),
                np.square(tm).sum(),
                np.square(ps - ts).sum(),
                np.square(ts).sum(),
                ps.sum(),
                ts.sum(),
            ]
        )

    def release(self) -> None:
        self.origin, self.mean, self.m2 = [], [], []

    def result(
        self, pred_mean: np.ndarray, true_mean: np.ndarray, pred_std: float
    ) -> dict[str, np.ndarray]:
        names = (
            "time_mean_difference_sq",
            "time_mean_reference_sq",
            "time_std_difference_sq",
            "time_std_reference_sq",
            "time_std_prediction_mean",
            "time_std_reference_mean",
        )
        totals = self.totals.copy()
        totals[4:] /= self.points * self.channels
        return {
            **{key: np.array(v) for key, v in zip(names, totals, strict=True)},
            "global_std_prediction": np.array(pred_std),
            "probes_prediction": np.concatenate(
                [self.probes[0].reshape(self.steps, -1), pred_mean], axis=1
            ),
            "probes_reference": np.concatenate(
                [self.probes[1].reshape(self.steps, -1), true_mean], axis=1
            ),
        }
