# SPDX-License-Identifier: Apache-2.0
"""Optional Zarr-Python adapter for bounded, explicit-range batches."""
from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
from zarr.abc.store import RangeByteRequest, Store
from zarr.core.buffer.core import ArrayLike
from zarr.storage import WrapperStore

from .ranges import RangePolicy, plan_ranges, read_ranges


class BoundedRangeStore(WrapperStore[Store]):
    """Apply a per-store policy to partial-shard range batches, without global settings.

    The allowance is per get_ranges call, not across arrays, shards, or consumers.
    Ordinary get calls (metadata and whole shards) use the underlying store unchanged.
    Zarr's decoder may retain yielded encoded buffers; those consumer buffers are not
    bounded by this scheduler. Explicit range batches are required; unknown-length
    ranges are rejected rather than pretending to enforce a byte bound.
    """

    def __init__(self, store: Store, *, policy: RangePolicy):
        if not hasattr(Store, 'get_ranges'):
            raise ImportError('BoundedRangeStore requires Zarr-Python >= 3.3 with get_ranges')
        super().__init__(store)
        self.policy = policy

    def _with_store(self, store):
        return type(self)(store, policy=self.policy)

    def _plan_inputs(self, byte_ranges, max_gap_bytes, max_coalesced_bytes):
        if len(byte_ranges) > self.policy.max_ranges:
            raise ValueError('Too many ranges; partition the metadata batch')
        ranges = []
        for request in byte_ranges:
            if not isinstance(request, RangeByteRequest):
                raise ValueError('Bounded range batches require explicit start/stop offsets')
            ranges.append((request.start, request.end))
        policy = replace(
            self.policy,
            max_gap_bytes=(self.policy.max_gap_bytes if max_gap_bytes is None
                           else min(self.policy.max_gap_bytes, max_gap_bytes)),
            max_request_bytes=(self.policy.max_request_bytes if max_coalesced_bytes is None
                               else min(self.policy.max_request_bytes, max_coalesced_bytes)),
        )
        return ranges, policy

    def get_ranges_sync(
        self, key, byte_ranges, *, prototype, max_gap_bytes=1 << 20,
        max_coalesced_bytes=16 << 20,
    ):
        """The synchronous API retains all results, so its entire plan must fit."""
        ranges, policy = self._plan_inputs(byte_ranges, max_gap_bytes, max_coalesced_bytes)
        plan = plan_ranges(ranges, policy)
        if plan.fetched_bytes > policy.max_inflight_bytes:
            raise ValueError('Synchronous range results exceed the byte allowance; use async reads')
        results = []
        for group in plan.groups:
            value = self.get_sync(key, prototype=prototype,
                                  byte_range=RangeByteRequest(group.start, group.stop))
            if value is None:
                raise FileNotFoundError(key)
            if len(value) != group.size:
                raise ValueError('Range source returned an unexpected byte count')
            results.extend((i, value[a - group.start:b - group.start])
                           for i, a, b in group.members)
        return results

    async def get_ranges(
        self, key, byte_ranges, *, prototype, max_concurrency=10,
        max_gap_bytes=1 << 20, max_coalesced_bytes=16 << 20,
    ):
        ranges, policy = self._plan_inputs(byte_ranges, max_gap_bytes, max_coalesced_bytes)
        policy = replace(
            policy,
            concurrency=(self.policy.concurrency if max_concurrency is None
                         else min(self.policy.concurrency, max_concurrency)),
        )

        async def fetch(start, stop):
            value = await self._store.get(key, prototype, RangeByteRequest(start, stop))
            return None if value is None else memoryview(value.as_numpy_array()).cast('B')

        stream = read_ranges(fetch, ranges, policy=policy)
        try:
            async for batch in stream:
                yield tuple((i, prototype.buffer.from_array_like(
                    cast(ArrayLike, np.frombuffer(part, dtype='u1'))))
                            for i, part in batch.parts)
                del batch
        finally:
            await stream.aclose()
