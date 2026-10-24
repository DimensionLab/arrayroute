# SPDX-License-Identifier: Apache-2.0
"""Lazy numerical arrays with explicit source and transform memory ownership."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .ordered import OrderedSelection, flat_blocks, read_flat
from .plan import ArraySpec, Selection
from .reader import ArrayReader, ReadStats

VECTOR_MAGNITUDE_ALGORITHM = "float64-source-range-vector-magnitude-v1"


class VectorMagnitudeBackend:
    """Drop a channel-last axis by computing its Euclidean norm in bounded blocks.

    Wrap this backend in an ArrayReader to bound the derived output separately. Its
    source reader bounds native reads; ``workspace_bytes`` bounds transform scratch.
    Source/native caches are additional. This adapter borrows the source reader and
    does not close it or acquire its shared budget a second time. Source I/O statistics
    go to ``source_stats``; outer reader statistics describe derived output requests.

    Float64 sources use channel-order hypot to avoid spurious square overflow/underflow.
    Smaller real floats and integer/bool sources use float64 square/sum/sqrt: their entire
    finite range, with admitted component counts, is safe in float64. Reductions may
    differ in last bits across these source dtypes. Source bytes/units are not modified.
    A fingerprint of this derived array covers magnitudes, not the original vectors.
    Pin the source revision for repeatable multi-pass consumers. Cancellation is checked
    between reads and blocks; native calls cannot be interrupted halfway through.
    """

    algorithm = VECTOR_MAGNITUDE_ALGORITHM

    def __init__(
        self,
        source: ArrayReader,
        *,
        workspace_bytes: int = 8 << 20,
        source_stats: ReadStats | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        spec, dtype = source.backend.spec, source.backend.dtype
        if len(spec.shape) < 2 or spec.shape[-1] < 1:
            raise ValueError("Vector magnitudes require a nonempty final component axis")
        if dtype.kind not in "biuf" or dtype.itemsize > 8:
            raise TypeError("Vector magnitudes require real numeric values of at most 64 bits")
        if (
            isinstance(workspace_bytes, bool)
            or not isinstance(workspace_bytes, int)
            or workspace_bytes < 1
        ):
            raise ValueError("workspace_bytes must be a positive integer")
        components = spec.shape[-1]
        self.reduction = "hypot" if dtype.kind == "f" and dtype.itemsize == 8 else "square-sum"
        self.rows_per_block = min(
            workspace_bytes // (components * (dtype.itemsize + 8) + 9),
            source.buffer_bytes // (2 * components * dtype.itemsize),
        )
        if self.rows_per_block < 1:
            raise MemoryError("Source/transform allowance cannot fit one complete vector")
        self.source = source
        self.source_budgets = tuple(
            pool for pool in ((source.shared_budget,) +
                              tuple(getattr(source.backend, "source_budgets", ())))
            if pool is not None
        )
        self.workspace_bytes = workspace_bytes
        self.source_stats = source_stats if source_stats is not None else ReadStats()
        self.cancelled = cancelled
        self.dtype = np.dtype("f8")
        shape = spec.shape[:-1]
        # The derived output has no storage chunks; the source reader admits actual I/O.
        self.spec = ArraySpec(shape, tuple(max(n, 1) for n in shape), 8, contiguous=True)

    def _check_cancelled(self) -> None:
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("Vector magnitude read cancelled")

    def read(self, selection: Selection) -> np.ndarray:
        self._check_cancelled()
        if len(selection) != len(self.spec.shape):
            raise ValueError("Selection must specify every derived axis")
        ordered = OrderedSelection.create(self.source, (*selection, slice(None)))
        components = self.source.backend.spec.shape[-1]
        shape = ordered.shape[:-1]
        output = np.empty(shape, dtype=self.dtype)
        flat = output.reshape(-1)
        for start, stop in flat_blocks(shape, self.rows_per_block):
            self._check_cancelled()
            raw = read_flat(
                ordered, start * components, stop * components, self.source_stats, self.cancelled,
            )
            values = raw.reshape(-1, components).astype(np.float64)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                if self.reduction == "square-sum":
                    np.square(values, out=values)
                    magnitude = values.sum(axis=-1)
                    np.sqrt(magnitude, out=magnitude)
                else:
                    magnitude = np.hypot.reduce(values, axis=-1)
                    for index in range(components):
                        np.copyto(magnitude, np.nan, where=np.isnan(values[:, index]))
                flat[start:stop] = magnitude
            del raw, values, magnitude
        self._check_cancelled()
        return output
