# SPDX-License-Identifier: Apache-2.0
"""Bounded range planning and backpressure for caller-owned immutable objects."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from operator import index


def _integer(value: int, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError('Byte limits and offsets must be integers, not booleans')
    value = index(value)
    if value < minimum:
        raise ValueError(f'Expected an integer >= {minimum}')
    return value


@dataclass(frozen=True)
class RangePolicy:
    max_request_bytes: int
    max_inflight_bytes: int
    max_extra_bytes: int = 0
    max_gap_bytes: int = 0
    concurrency: int = 4
    max_ranges: int = 65536

    def __post_init__(self) -> None:
        for name in ('max_request_bytes', 'max_inflight_bytes', 'concurrency', 'max_ranges'):
            object.__setattr__(self, name, _integer(getattr(self, name), minimum=1))
        for name in ('max_extra_bytes', 'max_gap_bytes'):
            object.__setattr__(self, name, _integer(getattr(self, name)))
        if self.max_request_bytes > self.max_inflight_bytes:
            raise ValueError('A request must fit the in-flight byte allowance')


@dataclass(frozen=True)
class RangeGroup:
    start: int
    stop: int
    members: tuple[tuple[int, int, int], ...]

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class RangePlan:
    groups: tuple[RangeGroup, ...]
    unique_requested_bytes: int
    fetched_bytes: int
    extra_bytes: int


def plan_ranges(ranges: Sequence[tuple[int, int]], policy: RangePolicy) -> RangePlan:
    """Plan one bounded batch. Offsets are half-open; inputs must be nonempty ranges.

    Extra bytes include bridged gaps and repeated overlap across separate groups.
    This is a greedy plan, not a globally optimal partition. An individual range larger
    than max_request_bytes is rejected before I/O; callers must split that request.
    Metadata is O(batch size), independent of the containing object's logical size.
    """
    if len(ranges) > policy.max_ranges:
        raise ValueError('Too many ranges; partition the metadata batch')
    entries = []
    for i, (start, stop) in enumerate(ranges):
        start, stop = _integer(start), _integer(stop)
        if stop <= start or stop - start > policy.max_request_bytes:
            raise ValueError('Each range must be nonempty and fit max_request_bytes')
        entries.append((i, start, stop))
    entries.sort(key=lambda item: (item[1], item[2]))
    unique, covered_end = 0, 0
    groups: list[RangeGroup] = []
    members: list[tuple[int, int, int]] = []
    start = end = spent_gaps = 0
    for entry in entries:
        _, a, b = entry
        unique += max(0, b - max(a, covered_end))
        covered_end = max(covered_end, b)
        gap = max(0, a - end)
        if (members and gap <= policy.max_gap_bytes
                and max(end, b) - start <= policy.max_request_bytes
                and spent_gaps + gap <= policy.max_extra_bytes):
            members.append(entry)
            end = max(end, b)
            spent_gaps += gap
        else:
            if members:
                groups.append(RangeGroup(start, end, tuple(members)))
            start, end, members = a, b, [entry]
    if members:
        groups.append(RangeGroup(start, end, tuple(members)))
    fetched = sum(group.size for group in groups)
    if fetched - unique > policy.max_extra_bytes:
        raise ValueError('Overlapping inputs exceed the extra-byte allowance in this plan')
    return RangePlan(tuple(groups), unique, fetched, fetched - unique)


@dataclass(frozen=True)
class RangeBatch:
    start: int
    stop: int
    parts: tuple[tuple[int, memoryview], ...]


Fetch = Callable[[int, int], Awaitable[bytes | bytearray | memoryview | None]]


async def _fetch_group(fetch: Fetch, group: RangeGroup) -> RangeBatch:
    value = await fetch(group.start, group.stop)
    if value is None:
        raise FileNotFoundError('Range source is absent')
    view = memoryview(value).cast('B').toreadonly()
    if len(view) != group.size:
        raise ValueError('Range source returned an unexpected byte count')
    return RangeBatch(group.start, group.stop, tuple(
        (i, view[a - group.start:b - group.start]) for i, a, b in group.members))


async def read_ranges(
    fetch: Fetch, ranges: Sequence[tuple[int, int]], *, policy: RangePolicy,
) -> AsyncGenerator[RangeBatch, None]:
    """Yield completion-order, zero-copy views; input indices identify the fragments.

    In-flight, completed-but-not-yielded and the current yielded request share the
    byte allowance. Tasks are created only when a slot is available. The caller must
    release each batch before advancing to keep total live payload within that bound.
    Retained consumer views, callback transport buffers and metadata are additional.
    fetch must not mutate returned buffers and must honour cancellation/cleanup itself.
    On failure or aclose, outstanding tasks are cancelled and awaited. Blocking native
    I/O hidden by a callback cannot be forcibly stopped by this scheduler.
    """
    plan = plan_ranges(ranges, policy)
    pending: dict[asyncio.Task[RangeBatch], int] = {}
    position = reserved = 0
    try:
        while position < len(plan.groups) or pending:
            while position < len(plan.groups) and len(pending) < policy.concurrency:
                group = plan.groups[position]
                if reserved + group.size > policy.max_inflight_bytes:
                    break
                task = asyncio.create_task(_fetch_group(fetch, group))
                pending[task] = group.size
                reserved += group.size
                position += 1
                del task
            ready, unfinished = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            del unfinished
            task = ready.pop()
            del ready
            size = pending.pop(task)
            batch = task.result()
            del task
            yield batch
            del batch
            reserved -= size
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        pending.clear()
