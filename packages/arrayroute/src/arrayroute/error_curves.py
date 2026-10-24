# SPDX-License-Identifier: Apache-2.0
"""Canonical, two-pass error curves for time/space/channel arrays."""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator

import numpy as np

from .plan import Selection
from .reader import ArrayReader, ReadStats

ALGORITHM = "paired-time-space-curves-v1-256"


def _regions(shape: tuple[int, ...]) -> Iterator[Selection]:
    # Fixed spatial grouping makes reduction order independent of read allowances,
    # storage chunks and backend completion order. Each region includes all times.
    spatial = shape[1:-1]
    axis, suffix = len(spatial) - 1, 1
    while axis > 0 and suffix * spatial[axis] <= 256:
        suffix *= spatial[axis]
        axis -= 1
    width = max(1, 256 // suffix)
    for prefix in itertools.product(*(range(n) for n in spatial[:axis])):
        for start in range(0, spatial[axis], width):
            yield (slice(None), *prefix, slice(start, min(start + width, spatial[axis])),
                   *(slice(None) for _ in spatial[axis + 1:]), slice(None))


class _WindowRead:
    """Keep one bounded spatial window while reducing fixed, smaller subregions."""

    def __init__(self, reader: ArrayReader, capacity: int, stats: ReadStats | None):
        self.reader, self.stats = reader, stats
        shape = reader.backend.spec.shape
        max_points = capacity // (shape[0] * shape[-1] * reader.backend.dtype.itemsize)
        spatial = shape[1:-1]
        axis, suffix = len(spatial) - 1, 1
        while axis > 0 and suffix * spatial[axis] <= max_points:
            suffix *= spatial[axis]
            axis -= 1
        self.axis = axis + 1
        width = max_points // suffix
        # A window boundary must never split a canonical reduction group.
        canonical = min(spatial[axis], max(1, 256 // suffix)) if suffix <= 256 else 1
        self.width = min(spatial[axis], width // canonical * canonical)
        if self.width < 1:
            raise MemoryError("Canonical time column exceeds read-window allowance")
        self.bounds: tuple[tuple[int, int], ...] | None = None
        self.value: np.ndarray | None = None

    def read(self, region: Selection) -> np.ndarray:
        shape = self.reader.backend.spec.shape
        starts = [v if isinstance(v, int) else (v.start or 0) for v in region]
        bounds = tuple(
            (starts[i], starts[i] + 1) if 0 < i < self.axis else
            (starts[i] // self.width * self.width,
             min((starts[i] // self.width + 1) * self.width, size)) if i == self.axis else
            (0, size) for i, size in enumerate(shape)
        )
        if bounds != self.bounds:
            self.value = None
            self.value = self.reader.read(tuple(slice(a, b) for a, b in bounds), stats=self.stats)
            self.bounds = bounds
        selected = tuple(
            v - bounds[i][0] if isinstance(v, int) else
            slice((v.start or 0) - bounds[i][0],
                  (shape[i] if v.stop is None else v.stop) - bounds[i][0], v.step)
            for i, v in enumerate(region)
        )
        if self.value is None:
            raise RuntimeError("Read window is unavailable")
        return self.value[selected]


def error_curves(
    prediction: ArrayReader, reference: ArrayReader, *, validity: ArrayReader | None = None,
    buffer_bytes: int = 128 << 20, epsilon: float = 1e-12,
    cancelled: Callable[[], bool] | None = None, stats: ReadStats | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-time error, correlation, energy and persistence/climatology baselines.

    Arrays have shape (time, spatial..., component). A binary validity array, when
    supplied, has the same shape and a common mask across vector components. Nonfinite
    observed reference values fail; nonfinite observed predictions score infinite error.
    Reductions use float64 and fixed spatial groups, in two passes over immutable inputs.
    Outputs scale with time; a 256-point time column must fit the numerical allowance.
    Results are deterministic across I/O budgets/chunking on the same numerical stack,
    but are not bit-identical to a monolithic NumPy reduction or promised across platforms.

    buffer_bytes admits reduction state, conservative NumPy workspace and each planned
    read, including retained coalesced read windows. It excludes backend caches/codec
    scratch and metadata. Reader shared pools cover I/O reservations, not the caller-owned
    reduction workspace. Concurrent calls need separate numerical allowances. Cancellation
    is polled between reads/reductions; a blocking backend call cannot be interrupted.
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
    state_bytes = n * (6 * channels + 32) * 8
    workspace_bytes = n * min(math.prod(shape[1:-1]), 256) * channels * 160
    if state_bytes + workspace_bytes >= buffer_bytes:
        raise MemoryError("Error-curve state and canonical time column exceed numerical allowance")
    read_allowance = (buffer_bytes - state_bytes - workspace_bytes) // 2
    readers = [ArrayReader(r.backend, buffer_bytes=min(r.buffer_bytes, read_allowance),
                           shared_budget=r.shared_budget) for r in readers]

    def check() -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError("Error-curve reduction cancelled")

    def blocks():
        windows = [_WindowRead(r, min(read_allowance // len(readers), r.buffer_bytes // 2),
                               stats) for r in readers]
        for region in _regions(shape):
            check()
            for r in readers:
                plan = r.plan(region)
                if plan.output_bytes >= r.buffer_bytes:
                    raise MemoryError("Error-curve region exceeds the remaining read allowance")
            raw_p = windows[0].read(region)
            check()
            raw_t = windows[1].read(region)
            check()
            p = np.array(raw_p, dtype=np.float64, order="C").reshape(n, -1, channels)
            t = np.array(raw_t, dtype=np.float64, order="C").reshape(n, -1, channels)
            del raw_p, raw_t
            if validity is None:
                mask = np.ones(t.shape, bool)
            else:
                raw_mask = windows[2].read(region).reshape(t.shape)
                if not np.isin(raw_mask, [0, 1]).all():
                    raise ValueError("Validity must contain only zero/one")
                if not np.all(raw_mask == raw_mask[..., :1]):
                    raise ValueError("A vector field needs one common validity mask")
                mask = raw_mask.astype(bool, order="C")
                del raw_mask
            if not np.isfinite(np.where(mask, t, 0)).all():
                raise ValueError("Nonfinite reference inside the validity mask")
            good = np.isfinite(np.where(mask, p, 0)).all(axis=(1, 2))
            p = np.where(mask & np.isfinite(p), p, 0)
            t = np.where(mask, t, 0)
            check()
            yield p, t, mask, good
            del p, t, mask, good

    check()
    count = np.zeros(n, np.float64)
    mean_p, mean_t = np.zeros((n, channels)), np.zeros((n, channels))
    se, tt, pp, clim, pers, pers_tt, common_count = (np.zeros(n) for _ in range(7))
    finite = np.ones(n, bool)
    for p, t, mask, good in blocks():
        finite &= good
        bc = mask[..., 0].sum(1).astype(np.float64)
        total = count + bc
        first = mask[..., 0].argmax(axis=1)
        rows = np.arange(n)
        for values, mean in ((p, mean_p), (t, mean_t)):
            origin = values[rows, first]
            shifted = np.where(mask, values - origin[:, None], 0)
            local_mean = origin + shifted.sum(1) / np.maximum(bc[:, None], 1)
            mean += (local_mean - mean) * (bc / np.maximum(total, 1))[:, None]
        count = total
        d = p - t
        se += np.einsum("tmc,tmc->t", d, d)
        tt += np.einsum("tmc,tmc->t", t, t)
        pp += np.einsum("tmc,tmc->t", p, p)
        common = mask & mask[:1]
        diff = np.where(common, t[:1] - t, 0)
        pers += np.einsum("tmc,tmc->t", diff, diff)
        observed = np.where(common, t, 0)
        pers_tt += np.einsum("tmc,tmc->t", observed, observed)
        common_count += common[..., 0].sum(1)
        climate = t.sum(0, keepdims=True) / np.maximum(mask.sum(0, keepdims=True), 1)
        dc = np.where(mask, climate - t, 0)
        clim += np.einsum("tmc,tmc->t", dc, dc)
        del p, t, mask, good, d, diff, observed, climate, dc, shifted, values, origin

    var_t, var_p, cov = np.zeros(n), np.zeros(n), np.zeros(n)
    for p, t, mask, _good in blocks():
        ta = np.where(mask, t - mean_t[:, None], 0)
        pa = np.where(mask, p - mean_p[:, None], 0)
        var_t += np.einsum("tmc,tmc->t", ta, ta)
        var_p += np.einsum("tmc,tmc->t", pa, pa)
        cov += np.einsum("tmc,tmc->t", pa, ta)
        del p, t, mask, ta, pa
    check()
    denominator = np.maximum(count, 1)
    total = max(float(count.sum()), 1)
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
