# SPDX-License-Identifier: Apache-2.0
"""Deterministic space/time tiled rollout reductions."""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator

import numpy as np

from .plan import Selection
from .reader import ArrayReader, ReadStats

ALGORITHM = "paired-time-space-curves-v2-256-64"
_TIME_BLOCK = 64


def _regions(shape: tuple[int, ...], max_points: int = 256) -> Iterator[Selection]:
    # Window boundaries align with canonical spatial groups, including ragged rows.
    spatial = shape[1:-1]
    axis, suffix = len(spatial) - 1, 1
    while axis > 0 and suffix * spatial[axis] <= max_points:
        suffix *= spatial[axis]
        axis -= 1
    canonical = min(spatial[axis], max(1, 256 // suffix)) if suffix <= 256 else 1
    width = min(spatial[axis], max_points // suffix // canonical * canonical)
    if width < 1:
        raise MemoryError("Canonical spatial group exceeds read-window allowance")
    for prefix in itertools.product(*(range(n) for n in spatial[:axis])):
        for start in range(0, spatial[axis], width):
            yield (slice(None), *prefix, slice(start, min(start + width, spatial[axis])),
                   *(slice(None) for _ in spatial[axis + 1:]), slice(None))


def error_curves(
    prediction: ArrayReader, reference: ArrayReader, *, validity: ArrayReader | None = None,
    buffer_bytes: int = 128 << 20, epsilon: float = 1e-12,
    cancelled: Callable[[], bool] | None = None, stats: ReadStats | None = None,
) -> dict[str, np.ndarray]:
    """Score immutable (time, spatial..., component) arrays without whole-time columns.

    Fixed 64-frame blocks and at most 256-point spatial groups preserve arithmetic
    order across I/O budgets, chunks and C/Fortran layouts on the same numerical stack.
    Temporal climatology is retrospective. Output/state is O(time * components);
    payload windows are bounded in both time and space. Binary validity masks must
    be common across vector components. Missing observations remain unscored.

    The allowance covers retained windows, reduction state and conservative NumPy
    workspace; backend caches, codec scratch and metadata are additional. Shared
    reader budgets cover I/O only; concurrent scorers need separate numerical budgets.
    Cancellation is polled between operations, not inside a blocking backend read.
    No cross-platform bit identity or identity to monolithic reductions is promised.
    """
    if isinstance(buffer_bytes, bool) or not isinstance(buffer_bytes, int) or buffer_bytes < 1:
        raise ValueError("buffer_bytes must be a positive integer")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    readers = [prediction, reference] + ([validity] if validity is not None else [])
    shape = prediction.backend.spec.shape
    if len(shape) < 3 or min(shape) < 1:
        raise ValueError("Expected nonempty time/spatial/component axes")
    if any(r.backend.spec.shape != shape or r.backend.dtype.kind not in "fiub"
           for r in readers):
        raise ValueError("Readers must have matching real numeric shapes")
    n, channels = shape[0], shape[-1]
    frames = min(n, _TIME_BLOCK)
    state_bytes = n * (12 * channels + 40) * 8
    workspace_bytes = frames * min(math.prod(shape[1:-1]), 256) * channels * 160
    remaining = buffer_bytes - state_bytes - workspace_bytes
    if remaining <= 0:
        raise MemoryError("Error-curve state and canonical tile exceed numerical allowance")
    read_allowance = remaining // 2
    readers = [ArrayReader(r.backend, buffer_bytes=min(r.buffer_bytes, read_allowance),
                           shared_budget=r.shared_budget) for r in readers]
    per_point = channels * (frames * sum(r.backend.dtype.itemsize for r in readers) + 64)
    max_points = min(read_allowance // per_point, *(
        r.buffer_bytes // (2 * frames * channels * r.backend.dtype.itemsize) for r in readers))
    if max_points < min(math.prod(shape[1:-1]), 256):
        raise MemoryError("Canonical spatial group exceeds read-window allowance")

    def check() -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError("Error-curve reduction cancelled")

    def read_window(reader: ArrayReader, region: Selection, start: int) -> np.ndarray:
        check()
        return reader.read((slice(start, min(start + frames, n)), *region[1:]), stats=stats)

    def mask_for(raw: np.ndarray | None, block_shape: tuple[int, ...]) -> np.ndarray:
        if raw is None:
            return np.ones(block_shape, bool)
        if not np.isin(raw, [0, 1]).all():
            raise ValueError("Validity must contain only zero/one")
        if not np.all(raw == raw[..., :1]):
            raise ValueError("A vector field needs one common validity mask")
        return raw.astype(bool, order="C")

    check()
    count = np.zeros(n, np.float64)
    mean_p, mean_t = np.zeros((n, channels)), np.zeros((n, channels))
    anchor_p, anchor_t = np.zeros((n, channels)), np.zeros((n, channels))
    se, tt, pp, clim, pers, pers_tt, common_count = (np.zeros(n) for _ in range(7))
    var_t, var_p, cov = np.zeros(n), np.zeros(n), np.zeros(n)
    finite = np.ones(n, bool)
    for window in _regions(shape, max_points):
        climate_sum = climate_count = first_t = first_mask = None
        for start in range(0, n, frames):
            raw_p = read_window(readers[0], window, start)
            raw_t = read_window(readers[1], window, start)
            raw_mask = read_window(readers[2], window, start) if validity is not None else None
            if climate_sum is None:
                climate_sum = np.zeros(raw_t.shape[1:], np.float64)
                climate_count = np.zeros(raw_t.shape[1:], np.float64)
                first_t = np.zeros(raw_t.shape[1:], np.float64)
                first_mask = np.zeros(raw_t.shape[1:], bool)
            assert climate_count is not None and first_t is not None and first_mask is not None
            k = len(raw_t)
            frame = slice(start, start + k)
            for region in _regions(raw_t.shape):
                check()
                spatial = region[1:]
                p = np.array(raw_p[region], dtype=np.float64, order="C").reshape(k, -1, channels)
                t = np.array(raw_t[region], dtype=np.float64, order="C").reshape(k, -1, channels)
                mask = mask_for(None if raw_mask is None else raw_mask[region].reshape(t.shape),
                                t.shape)
                if not np.isfinite(np.where(mask, t, 0)).all():
                    raise ValueError("Nonfinite reference inside the validity mask")
                finite[frame] &= np.isfinite(np.where(mask, p, 0)).all(axis=(1, 2))
                p = np.where(mask & np.isfinite(p), p, 0)
                t = np.where(mask, t, 0)
                bc = mask[..., 0].sum(1).astype(np.float64)
                total = count[frame] + bc
                first = mask[..., 0].argmax(axis=1)
                rows = np.arange(k)
                initialize = (count[frame] == 0) & (bc > 0)
                # Subtract a fixed observed anchor before Chan's central-moment merge.
                # This preserves small fluctuations on top of large physical offsets.
                anchor_p[frame][initialize] = p[rows, first][initialize]
                anchor_t[frame][initialize] = t[rows, first][initialize]
                ps = np.where(mask, p - anchor_p[frame, None], 0)
                ts = np.where(mask, t - anchor_t[frame, None], 0)
                mp = ps.sum(1) / np.maximum(bc[:, None], 1)
                mt = ts.sum(1) / np.maximum(bc[:, None], 1)
                dp, dt = mp - mean_p[frame], mt - mean_t[frame]
                pa = np.where(mask, ps - mp[:, None], 0)
                ta = np.where(mask, ts - mt[:, None], 0)
                weight = count[frame] * bc / np.maximum(total, 1)
                var_p[frame] += np.einsum("tmc,tmc->t", pa, pa) + weight * (dp * dp).sum(1)
                var_t[frame] += np.einsum("tmc,tmc->t", ta, ta) + weight * (dt * dt).sum(1)
                cov[frame] += np.einsum("tmc,tmc->t", pa, ta) + weight * (dp * dt).sum(1)
                mean_p[frame] += dp * (bc / np.maximum(total, 1))[:, None]
                mean_t[frame] += dt * (bc / np.maximum(total, 1))[:, None]
                count[frame] = total
                d = p - t
                se[frame] += np.einsum("tmc,tmc->t", d, d)
                tt[frame] += np.einsum("tmc,tmc->t", t, t)
                pp[frame] += np.einsum("tmc,tmc->t", p, p)
                local_shape = climate_sum[spatial].shape
                climate_sum[spatial] += t.sum(0).reshape(local_shape)
                climate_count[spatial] += mask.sum(0).reshape(local_shape)
                if start == 0:
                    first_t[spatial] = t[0].reshape(local_shape)
                    first_mask[spatial] = mask[0].reshape(local_shape)
                common = mask & first_mask[spatial].reshape(1, -1, channels)
                diff = np.where(common, first_t[spatial].reshape(1, -1, channels) - t, 0)
                observed = np.where(common, t, 0)
                pers[frame] += np.einsum("tmc,tmc->t", diff, diff)
                pers_tt[frame] += np.einsum("tmc,tmc->t", observed, observed)
                common_count[frame] += common[..., 0].sum(1)
                del p, t, mask, ps, ts, pa, ta, d, diff, observed
            del raw_p, raw_t, raw_mask
        assert climate_sum is not None and climate_count is not None
        climate_sum /= np.maximum(climate_count, 1)
        del climate_count, first_t, first_mask
        for start in range(0, n, frames):
            raw_t = read_window(readers[1], window, start)
            raw_mask = read_window(readers[2], window, start) if validity is not None else None
            k = len(raw_t)
            for region in _regions(raw_t.shape):
                check()
                t = np.array(raw_t[region], dtype=np.float64, order="C").reshape(k, -1, channels)
                mask = mask_for(None if raw_mask is None else raw_mask[region].reshape(t.shape),
                                t.shape)
                dc = np.where(mask, climate_sum[region[1:]].reshape(1, -1, channels) - t, 0)
                clim[start:start + k] += np.einsum("tmc,tmc->t", dc, dc)
                del t, mask, dc
            del raw_t, raw_mask
        del climate_sum
    check()
    denominator = np.maximum(count, 1)
    total = max(float(count.sum()), 1)
    mean_t += anchor_t
    origin = mean_t[int(np.argmax(count > 0))]
    global_mean = origin + ((mean_t - origin) * count[:, None]).sum(0) / total
    sigma = math.sqrt(float(var_t.sum() + (
        (mean_t - global_mean) ** 2 * count[:, None]).sum()) / total)
    var_t, var_p, cov = (x / denominator for x in (var_t, var_p, cov))
    finite &= count > 0
    with np.errstate(all="ignore"):
        rmse = np.sqrt(se / denominator)
        corr = cov / np.sqrt(np.maximum(var_p * var_t, epsilon**4))
        if validity is None:
            corr = np.where(var_p > epsilon**2, corr, 0)
        corr = np.where(var_t > epsilon**2, corr, np.nan)
        persistence = np.sqrt(pers) / np.maximum(np.sqrt(pers_tt), epsilon)
        persistence[common_count == 0] = np.inf
        return {
            "l2_rel": np.where(finite, np.sqrt(se) / np.maximum(np.sqrt(tt), epsilon), np.inf),
            "rmse": np.where(finite, rmse, np.inf),
            "nrmse": np.where(finite, rmse / max(sigma, epsilon), np.inf),
            "vrmse": np.where(finite, rmse / np.sqrt(np.maximum(var_t, epsilon)), np.inf),
            "corr": np.where(finite, corr, np.nan),
            "persistence": persistence,
            "climatology": np.where(count > 0,
                                    np.sqrt(clim) / np.maximum(np.sqrt(tt), epsilon), np.inf),
            "ms_pred": np.where(finite, pp / denominator, np.inf),
            "ms_true": np.where(count > 0, tt / denominator, np.nan),
            "finite": finite,
            "sigma": np.array(sigma),
        }
