# SPDX-License-Identifier: Apache-2.0
"""Metadata-only concurrency choices under one aggregate read-buffer allowance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .plan import ArraySpec, Selection, plan, tiles


@dataclass(frozen=True)
class ReadConcurrency:
    workers: int
    buffer_bytes_per_worker: int
    materialization: Literal["empty", "direct", "tiled", "stream-required", "chunk-too-large"]
    requests_per_value_lower_bound: int
    request_count_exact: bool


@dataclass(frozen=True)
class ConcurrencyPlan:
    total_buffer_bytes: int
    requested_max_workers: int
    output_bytes_per_value: int
    untiled_buffer_bytes_per_value: int
    max_workers_without_tiling: int
    candidates: tuple[ReadConcurrency, ...]
    scope: str = (
        "Output plus peak live decoded chunks for one value per reader; excludes "
        "queued batches, caller-retained outputs, native caches, transport and codec scratch. "
        "Candidates are memory plans, not throughput predictions."
    )


def plan_concurrency(
    spec: ArraySpec,
    selection: Selection,
    *,
    total_buffer_bytes: int,
    max_workers: int,
    count_limit: int = 256,
) -> ConcurrencyPlan:
    """Suggest bounded benchmark candidates without reading values or indexing the dataset.

    Include powers of two, the requested ceiling and the largest untiled worker count.
    Budgets use ArrayReader.read's materialization rules. Each reader has the same
    selection cost. For variable requests, plan the largest requirement, including
    chunk alignment and layout as well as shape.
    The tile count stops after count_limit + 1 tiles and is then a lower bound. Thus
    metadata planning does not enumerate all tiles of a TB/PB selection. Worker counts
    are capped by the number of bytes available, since each reader needs a positive budget.
    """
    for value in (total_buffer_bytes, max_workers, count_limit):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                "Buffer bytes, maximum workers and count limit must be positive integers"
            )
    request = plan(spec, selection)
    ceiling = min(max_workers, total_buffer_bytes)
    untiled = min(ceiling, total_buffer_bytes // max(1, request.buffer_bytes_upper_bound))
    counts = {1, ceiling}
    if untiled:
        counts.add(untiled)
    power = 2
    while power < ceiling:
        counts.add(power)
        power *= 2
    candidates = []
    for workers in sorted(counts):
        budget = total_buffer_bytes // workers
        requests, exact = 0, True
        mode: Literal["empty", "direct", "tiled", "stream-required", "chunk-too-large"]
        if request.output_bytes == 0:
            mode = "empty"
        elif request.buffer_bytes_upper_bound <= budget:
            mode, requests = "direct", 1
        elif request.output_bytes >= budget:
            mode = "stream-required"
        else:
            mode = "tiled"
            try:
                for _ in tiles(spec, selection, buffer_bytes=budget - request.output_bytes):
                    requests += 1
                    if requests > count_limit:
                        exact = False
                        break
            except MemoryError:
                mode, requests, exact = "chunk-too-large", 0, True
        candidates.append(ReadConcurrency(workers, budget, mode, requests, exact))
    return ConcurrencyPlan(
        total_buffer_bytes, max_workers, request.output_bytes,
        request.buffer_bytes_upper_bound, untiled, tuple(candidates),
    )
