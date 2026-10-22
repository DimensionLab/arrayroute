# SPDX-License-Identifier: Apache-2.0
"""Bounded partial and bulk gathering over independent region readers."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from .backends import _grid_regions
from .execution import ordered_map
from .plan import GatherWindow, ReadPlan, Selection
from .reader import Backend


class ParallelGatherBackend:
    """Read disjoint chunk-aligned regions with bounded, worker-owned intermediates.

    Each factory result must have the same metadata and single-chunk decode guarantee
    as base. It must own its mutable state; read-only native array handles may be shared.
    Results must not retain arrays larger than their selected region or one decoded
    chunk. Workers must not reserve from the parent reader's shared ReadBudget.
    Only planned strided calls parallelize. No thread pool survives a call, including
    failures. Native reads are joined before the output reservation is released.

    An optional bulk factory returns worker-local callables that gather from a full
    region. They must return compact arrays with the selected shape/dtype, retain no
    decoder state between calls, and use at most two decoded region buffers plus the
    compact result. Encoded inputs and codec scratch are outside that bound. Bulk
    plans never add skipped in-domain chunks; edge padding is included in their cost.
    """

    def __init__(
        self, base: Backend, factory: Callable[[], Backend], window: GatherWindow,
        *, bulk_factory: Callable[[], Callable[[Selection], np.ndarray]] | None = None,
    ):
        if base.spec.gather_window is not None:
            raise ValueError("Nested gather backends are not supported")
        self.base, self.factory, self.window = base, factory, window
        if window.bulk != (bulk_factory is not None):
            raise ValueError("Bulk gathering requires both a capability and a reader factory")
        self.bulk_factory = bulk_factory
        self.dtype = base.dtype
        self.spec = replace(base.spec, gather_window=window)

    def read(self, selection: Selection) -> np.ndarray:
        return self.base.read(selection)

    def read_plan(self, request: ReadPlan) -> np.ndarray:
        if request.strategy == 'native':
            return self.base.read(request.selection)
        bulk = request.strategy == 'bulk-gather'
        if (request.strategy not in {'parallel-gather', 'bulk-gather'}
                or not (1 if bulk else 2) <= request.gather_workers <= self.window.workers
                or (bulk and self.bulk_factory is None)):
            raise ValueError('Parallel gather requires its planned worker count')
        output = np.empty(request.output_shape, dtype=self.dtype)

        def make_worker():
            if bulk:
                assert self.bulk_factory is not None
                read_bulk = self.bulk_factory()
                return lambda region: (region, read_bulk(region))
            backend = self.factory()
            if backend.spec != self.base.spec or backend.dtype != self.dtype:
                raise ValueError('Gather factory changed array metadata')

            def read(region):
                return region, backend.read(region)

            return read

        regions = _grid_regions(request.selection, self.window.region_shape)
        with ordered_map(make_worker, regions, workers=request.gather_workers,
                         max_pending=request.gather_workers) as rows:
            for region, value in rows:
                targets = []
                shape = []
                for part, original in zip(region, request.selection, strict=True):
                    if isinstance(original, slice):
                        assert isinstance(part, slice)
                        count = len(range(part.start, part.stop, part.step))
                        begin = (part.start - original.start) // original.step
                        targets.append(slice(begin, begin + count))
                        shape.append(count)
                if value.shape != tuple(shape) or value.dtype != self.dtype:
                    raise ValueError('Gather worker returned an unexpected shape or dtype')
                output[tuple(targets)] = value
                del value
        return output
