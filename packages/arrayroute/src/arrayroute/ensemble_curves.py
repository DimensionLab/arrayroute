# SPDX-License-Identifier: Apache-2.0
"""Deterministic space/time tiled predictive ensemble diagnostics."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .error_curves import _regions
from .reader import ArrayReader, ReadStats

ALGORITHM = "ensemble-time-space-curves-v1-256-8"
_TIME_BLOCK = 8


def gaussian_crps(mu: np.ndarray, sigma: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Gaussian CRPS; zero spread gives absolute error.

    The normal CDF uses Abramowitz & Stegun 7.1.26 (erf absolute error <= 1.5e-7).
    This is a Gaussian approximation to the predictive distribution, not empirical CRPS.
    """
    mu, sigma, truth = (np.asarray(v, dtype=np.float64) for v in (mu, sigma, truth))
    safe = np.where(sigma > 0, sigma, 1.0)
    z = (truth - mu) / safe
    x = z / math.sqrt(2.0)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (
        0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    erf = np.sign(x) * (1.0 - poly * np.exp(-a * a))
    cdf = 0.5 * (1.0 + erf)
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    out = safe * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    return np.where(sigma > 0, out, np.abs(truth - mu))


def ensemble_curves(
    ensemble: ArrayReader,
    reference: ArrayReader,
    *,
    z_nominal: float,
    multipliers: np.ndarray | None = None,
    n_scores: int = 0,
    steps: int | None = None,
    buffer_bytes: int = 128 << 20,
    epsilon: float = 1e-12,
    cancelled: Callable[[], bool] | None = None,
    stats: ReadStats | None = None,
) -> dict[str, np.ndarray]:
    """Score (member, time, spatial..., component) against (time, spatial..., component).

    Fixed eight-frame / <=256-point numerical groups preserve arithmetic across admitted
    I/O windows and storage layouts on the same numerical stack. Both space and time are
    tiled; members and channels within a canonical tile must fit. Sources are immutable,
    caller-owned and must use matching coordinates/units (validated by the consumer).
    Reference values must be finite. Nonfinite spread, RMSE and CRPS become infinity;
    undefined spread/skill ratios remain NaN. No validity masks are accepted.

    The allowance admits output/state, retained inputs and conservative numerical scratch.
    Backend-owned derived workspace, caches, codec scratch and metadata are additional.
    Reader shared pools admit I/O only; independent scorers need independent workspace
    admission. Prefixes are read without later steps. Cancellation is checked between
    operations, not inside native I/O.
    This changes grouping from monolithic NumPy reductions; record ALGORITHM with results.
    """
    if isinstance(buffer_bytes, bool) or not isinstance(buffer_bytes, int) or buffer_bytes < 1:
        raise ValueError("buffer_bytes must be a positive integer")
    if not math.isfinite(z_nominal) or z_nominal < 0:
        raise ValueError("z_nominal must be finite and nonnegative")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if isinstance(n_scores, bool) or not isinstance(n_scores, int) or n_scores < 0:
        raise ValueError("n_scores must be a nonnegative integer")
    shape, eshape = reference.backend.spec.shape, ensemble.backend.spec.shape
    if (
        len(shape) < 3
        or min(shape) < 1
        or len(eshape) != len(shape) + 1
        or eshape[1:] != shape
        or eshape[0] < 2
    ):
        raise ValueError("Expected matching nonempty fields and at least two ensemble members")
    for reader in (ensemble, reference):
        if reader.backend.dtype.kind not in "biuf" or reader.backend.dtype.itemsize > 8:
            raise TypeError("Ensemble diagnostics require real values of at most 64 bits")
    if steps is not None:
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= shape[0]:
            raise ValueError("steps must select a nonempty prefix")
        shape = (steps, *shape[1:])
    factors = None
    if multipliers is not None:
        factors = np.asarray(multipliers)
        if (
            factors.ndim != 1
            or not factors.size
            or not np.isfinite(factors).all()
            or np.any(factors < 0)
        ):
            raise ValueError("multipliers must be a nonempty finite nonnegative vector")
    n, channels, members = shape[0], shape[-1], eshape[0]
    points = math.prod(shape[1:-1])
    scores = min(n_scores, points * channels)
    frames = min(n, _TIME_BLOCK)
    state_bytes = n * (128 + 16 * scores) + scores * 16
    scratch = frames * min(points, 256) * channels * (32 * members + 256)
    remaining = buffer_bytes - state_bytes - scratch
    if remaining <= 0:
        raise MemoryError("Ensemble state and canonical tile exceed numerical allowance")
    allowance = remaining // 2
    readers = [
        ArrayReader(
            r.backend, buffer_bytes=min(r.buffer_bytes, allowance), shared_budget=r.shared_budget
        )
        for r in (ensemble, reference)
    ]
    sizes = [members * ensemble.backend.dtype.itemsize, reference.backend.dtype.itemsize]
    max_points = min(
        allowance // (frames * channels * sum(sizes)),
        *(
            r.buffer_bytes // (2 * frames * channels * size)
            for r, size in zip(readers, sizes, strict=True)
        ),
    )
    if max_points < min(points, 256):
        raise MemoryError("Canonical ensemble group exceeds read allowance")

    def check() -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError("Ensemble reduction cancelled")

    check()
    variance, squared_error, crps, covered, calibrated = (np.zeros(n) for _ in range(5))
    selected = np.unique(np.linspace(0, points * channels - 1, scores).astype(np.int64))
    score_values = np.empty((n, len(selected))) if scores else None
    infl = math.sqrt(1.0 + 1.0 / members)
    window_offset = 0
    for window in _regions(shape, max_points):
        window_points = 0
        for start in range(0, n, frames):
            check()
            stop = min(n, start + frames)
            time = slice(start, stop)
            selection = (time, *window[1:])
            raw_e = readers[0].read((slice(None), *selection), stats=stats)
            raw_t = readers[1].read(selection, stats=stats)
            k = len(raw_t)
            window_points = math.prod(raw_t.shape[1:-1])
            offset = window_offset
            for region in _regions(raw_t.shape):
                check()
                e = np.array(raw_e[(slice(None), *region)], dtype=np.float64, order="C")
                e = e.reshape(members, k, -1, channels)
                t = np.array(raw_t[region], dtype=np.float64, order="C").reshape(k, -1, channels)
                if not np.isfinite(t).all():
                    raise ValueError("Ensemble reference contains nonfinite values")
                count = t.shape[1]
                with np.errstate(all="ignore"):
                    mu = e.mean(axis=0)
                    sd = e.std(axis=0, ddof=1)
                    error = t - mu
                    spread = sd * infl
                    absolute = np.abs(error)
                    variance[time] += np.einsum("tmc,tmc->t", sd, sd)
                    squared_error[time] += np.einsum("tmc,tmc->t", error, error)
                    covered[time] += (absolute <= z_nominal * spread).sum(axis=(1, 2))
                    crps[time] += gaussian_crps(mu, spread, t).sum(axis=(1, 2))
                    if factors is not None:
                        m = factors[np.minimum(np.arange(start, stop), len(factors) - 1)]
                        calibrated[time] += (absolute <= m[:, None, None] * spread).sum(axis=(1, 2))
                    if score_values is not None:
                        chosen = (selected >= offset * channels) & (
                            selected < (offset + count) * channels
                        )
                        local = selected[chosen] - offset * channels
                        sp = spread.reshape(k, -1)[:, local]
                        score = absolute.reshape(k, -1)[:, local] / np.where(sp > 0, sp, np.nan)
                        score_values[time, chosen] = np.where(np.isfinite(score), score, np.inf)
                offset += count
                del e, t, mu, sd, error, spread, absolute
            del raw_e, raw_t
        window_offset += window_points
    with np.errstate(all="ignore"):
        spread = np.sqrt(variance / points)
        rmse = np.sqrt(squared_error / points)
        ratio = np.where(rmse > 0, infl * spread / np.maximum(rmse, epsilon), np.nan)
        crps /= points * channels
    output = {
        "spread": np.where(np.isfinite(spread), spread, np.inf),
        "rmse": np.where(np.isfinite(rmse), rmse, np.inf),
        "ratio": ratio,
        "coverage": covered / (points * channels),
        "crps": np.where(np.isfinite(crps), crps, np.inf),
    }
    if factors is not None:
        output["coverage_conformal"] = calibrated / (points * channels)
    if score_values is not None:
        output["scores"] = score_values
    return output
