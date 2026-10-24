# SPDX-License-Identifier: Apache-2.0
"""Lazy ordered stacks and means over immutable array readers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .ordered import OrderedSelection, flat_blocks, read_flat
from .plan import ArraySpec, Selection, plan
from .reader import ArrayReader, ReadStats

MEAN_ALGORITHM = "ordered-scaled-compensated-mean-f64-v1"


class _Sources:
    def __init__(
        self,
        sources: Sequence[ArrayReader],
        *,
        workspace_bytes: int,
        cancelled: Callable[[], bool] | None,
        source_stats: ReadStats | None,
    ):
        if not sources:
            raise ValueError("At least one source is required")
        self.sources = tuple(sources)
        self.source_budgets = tuple(
            pool
            for r in self.sources
            for pool in ((r.shared_budget,) + tuple(getattr(r.backend, "source_budgets", ())))
            if pool is not None
        )
        first = self.sources[0].backend
        if any(
            r.backend.spec.shape != first.spec.shape or r.backend.dtype != first.dtype
            for r in self.sources
        ):
            raise ValueError("Sources must have identical shapes and dtypes")
        if first.dtype.kind not in "biuf" or first.dtype.itemsize > 8:
            raise TypeError("Composite arrays require real numeric values of at most 64 bits")
        if (
            isinstance(workspace_bytes, bool)
            or not isinstance(workspace_bytes, int)
            or workspace_bytes < 192
        ):
            raise ValueError("workspace_bytes must be an integer of at least 192")
        self.block_elements = min(
            workspace_bytes // 192,
            *(r.buffer_bytes // (2 * r.backend.dtype.itemsize) for r in self.sources),
        )
        if self.block_elements < 1:
            raise MemoryError("Source allowance cannot fit one element")
        self.workspace_bytes = workspace_bytes
        self.cancelled = cancelled
        self.source_stats = source_stats if source_stats is not None else ReadStats()

    def check(self) -> None:
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("Composite array read cancelled")


class StackBackend(_Sources):
    """Add a leading source axis without materializing complete inputs.

    Source order is the supplied order. An enclosing ArrayReader budgets derived output;
    workspace_bytes budgets flat-copy scratch. One source reader is active at a time and
    its read allowance is additional, as are codecs/caches. Do not wrap the derived reader
    in the same shared pool as its sources: that would recursively reserve the pool.
    Source statistics count underlying reads; outer statistics count derived output.
    The caller keeps sources immutable and owns their lifetime. This backend never closes
    them. Integer and positive-stride selections preserve native dtype and exact bytes.
    """

    def __init__(
        self,
        sources: Sequence[ArrayReader],
        *,
        workspace_bytes: int = 8 << 20,
        cancelled: Callable[[], bool] | None = None,
        source_stats: ReadStats | None = None,
    ):
        super().__init__(
            sources, workspace_bytes=workspace_bytes, cancelled=cancelled, source_stats=source_stats
        )
        first = self.sources[0].backend
        self.dtype = first.dtype
        # Stacking copies native values; it needs no compensated-mean numerical scratch.
        self.block_elements = min(
            workspace_bytes // self.dtype.itemsize,
            *(r.buffer_bytes // (2 * self.dtype.itemsize) for r in self.sources),
        )
        shape = (len(sources), *first.spec.shape)
        self.spec = ArraySpec(
            shape, tuple(max(1, n) for n in shape), self.dtype.itemsize, contiguous=True
        )

    def read(self, selection: Selection) -> np.ndarray:
        self.check()
        request = plan(self.spec, selection)
        member, *spatial = request.selection
        indices = (
            [member] if isinstance(member, int) else range(member.start, member.stop, member.step)
        )
        output = np.empty(request.output_shape, dtype=self.dtype)
        for out_index, source_index in enumerate(indices):
            source = OrderedSelection.create(self.sources[source_index], tuple(spatial))
            flat = (
                output.reshape(-1)
                if isinstance(member, int)
                else output.reshape(len(indices), -1)[out_index]
            )
            for start, stop in flat_blocks(source.shape, self.block_elements):
                self.check()
                flat[start:stop] = read_flat(source, start, stop, self.source_stats, self.cancelled)
        return output


class MeanBackend(_Sources):
    """Float64 mean with fixed source order and compensated, pre-scaled addition.

    Ownership and budget contracts are the same as StackBackend. Numerical workspace is
    bounded independently of source count. Scaling before summation avoids ordinary sum
    overflow. Any remaining finite-input arithmetic overflow raises; nonfinite inputs
    propagate IEEE values. Same source order/values/runtime reproduce bits across read
    budgets and storage layouts. Reordering members or changing platforms is not covered.
    This policy improves on float32 accumulation but need not equal NumPy mean bitwise.
    """

    algorithm = MEAN_ALGORITHM

    def __init__(
        self,
        sources: Sequence[ArrayReader],
        *,
        workspace_bytes: int = 8 << 20,
        cancelled: Callable[[], bool] | None = None,
        source_stats: ReadStats | None = None,
    ):
        super().__init__(
            sources, workspace_bytes=workspace_bytes, cancelled=cancelled, source_stats=source_stats
        )
        self.dtype = np.dtype("f8")
        shape = self.sources[0].backend.spec.shape
        self.spec = ArraySpec(shape, tuple(max(1, n) for n in shape), 8, contiguous=True)

    def read(self, selection: Selection) -> np.ndarray:
        self.check()
        ordered = [OrderedSelection.create(r, selection) for r in self.sources]
        output = np.empty(ordered[0].shape, self.dtype)
        flat = output.reshape(-1)
        for start, stop in flat_blocks(output.shape, self.block_elements):
            total, correction = np.zeros(stop - start), np.zeros(stop - start)
            with np.errstate(over="ignore", invalid="ignore"):
                for source in ordered:
                    self.check()
                    value = read_flat(source, start, stop, self.source_stats, self.cancelled)
                    value = value.astype(np.float64) / len(ordered)
                    updated = total + value
                    finite = np.isfinite(total) & np.isfinite(value)
                    if np.any(finite & ~np.isfinite(updated)):
                        raise OverflowError("Finite mean accumulation exceeded float64 range")
                    delta = np.where(
                        np.abs(total) >= np.abs(value),
                        (total - updated) + value,
                        (value - updated) + total,
                    )
                    correction = np.where(finite, correction + delta, 0)
                    total = updated
                result = total + correction
                if np.any(np.isfinite(total) & ~np.isfinite(result)):
                    raise OverflowError("Finite compensated mean exceeded float64 range")
            flat[start:stop] = result
        return output
