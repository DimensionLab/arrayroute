# SPDX-License-Identifier: Apache-2.0
import asyncio
import gc
import random
import weakref

import numpy as np
import pytest

from arrayroute.ranges import RangePolicy, plan_ranges, read_ranges


def test_coalescing_budget_counts_extra_bytes_and_preserves_duplicate_indices():
    policy = RangePolicy(64, 128, max_extra_bytes=3, max_gap_bytes=4)
    plan = plan_ranges([(10, 14), (0, 4), (0, 4), (7, 9)], policy)
    assert plan.unique_requested_bytes == 10
    assert plan.fetched_bytes == 13 and plan.extra_bytes == 3
    assert [(g.start, g.stop) for g in plan.groups] == [(0, 9), (10, 14)]
    assert sorted(i for g in plan.groups for i, _, _ in g.members) == [0, 1, 2, 3]


def test_random_plans_cover_every_request_without_exceeding_budget():
    rng = random.Random(1849)
    for _ in range(250):
        ranges = [(a, a + rng.randrange(1, 9)) for a in rng.sample(range(100), 15)]
        policy = RangePolicy(32, 64, max_extra_bytes=10, max_gap_bytes=3)
        try:
            plan = plan_ranges(ranges, policy)
        except ValueError as error:
            assert 'Overlapping inputs' in str(error)
            continue
        wanted = set().union(*(set(range(a, b)) for a, b in ranges))
        assert plan.unique_requested_bytes == len(wanted)
        assert plan.fetched_bytes == sum(g.size for g in plan.groups)
        assert plan.extra_bytes == plan.fetched_bytes - len(wanted) <= 10
        assert len([i for g in plan.groups for i, _, _ in g.members]) == len(ranges)
        for group in plan.groups:
            assert group.size <= policy.max_request_bytes
            for i, a, b in group.members:
                assert (a, b) == ranges[i] and group.start <= a < b <= group.stop


def test_petabyte_offsets_are_metadata_only():
    plan = plan_ranges([(10**15, 10**15 + 16), (10**15 + 24, 10**15 + 32)],
                       RangePolicy(32, 32, max_gap_bytes=8, max_extra_bytes=8))
    assert plan.fetched_bytes == 32 and plan.extra_bytes == 8


@pytest.mark.parametrize('ranges', [[(0, 0)], [(-1, 3)], [(0, 65)], [(False, 1)]])
def test_invalid_ranges_fail_before_fetch(ranges):
    async def run():
        async def fetch(a, b):
            pytest.fail('Invalid planning must not perform I/O')
        with pytest.raises((TypeError, ValueError)):
            async for _ in read_ranges(fetch, ranges, policy=RangePolicy(64, 64)):
                pass
    asyncio.run(run())


@pytest.mark.parametrize('kwargs', [
    {'max_request_bytes': 0}, {'max_inflight_bytes': 8}, {'concurrency': 0},
    {'max_gap_bytes': -1}, {'max_extra_bytes': True}, {'max_ranges': 0},
])
def test_invalid_policy(kwargs):
    with pytest.raises((TypeError, ValueError)):
        RangePolicy(**({'max_request_bytes': 16, 'max_inflight_bytes': 32} | kwargs))


def test_slow_consumer_keeps_completed_payloads_and_tasks_bounded():
    async def run():
        refs, seen, alive_tasks = [], {}, []
        ranges = [(i * 64, i * 64 + (16 if i % 2 else 32)) for i in range(100)]
        policy = RangePolicy(32, 64, concurrency=20)

        async def fetch(a, b):
            await asyncio.sleep(0)
            value = np.arange(a, b, dtype=np.int64).astype('u1')
            refs.append((weakref.ref(value), value.nbytes))
            alive_tasks.append(len(asyncio.all_tasks()))
            return memoryview(value)

        async for batch in read_ranges(fetch, ranges, policy=policy):
            await asyncio.sleep(0.001)
            assert sum(n for ref, n in refs if ref() is not None) <= 64
            for i, view in batch.parts:
                assert view.readonly
                seen[i] = view.tobytes()
            del view, batch
        assert len(seen) == 100
        assert max(alive_tasks) <= 5  # Main task plus at most four 16-byte reads.
        assert sum(ref() is not None for ref, _ in refs) == 0
        for i, (a, b) in enumerate(ranges):
            assert seen[i] == bytes(x % 256 for x in range(a, b))

    enabled = gc.isenabled()
    gc.disable()
    try:
        asyncio.run(run())
    finally:
        if enabled:
            gc.enable()
        gc.collect()


def test_merged_views_preserve_input_order_mapping():
    async def run():
        async def fetch(a, b):
            return bytes(range(a, b))
        outputs = {}
        async for batch in read_ranges(fetch, [(10, 14), (0, 4), (0, 4), (7, 9)],
                policy=RangePolicy(64, 128, max_gap_bytes=4, max_extra_bytes=3)):
            outputs.update((i, value.tobytes()) for i, value in batch.parts)
        assert outputs == {0: bytes(range(10, 14)), 1: bytes(range(4)),
                           2: bytes(range(4)), 3: bytes(range(7, 9))}
    asyncio.run(run())


@pytest.mark.parametrize('returned,error', [(None, FileNotFoundError), (b'x', ValueError)])
def test_failed_read_cancels_and_awaits_other_callbacks(returned, error):
    async def run():
        live = set()
        async def fetch(a, b):
            live.add(a)
            try:
                if a == 0:
                    await asyncio.sleep(0.01)
                    return returned
                await asyncio.sleep(100)
                return bytes(b - a)
            finally:
                live.remove(a)
        with pytest.raises(error):
            async for _ in read_ranges(fetch, [(0, 8), (16, 24)],
                                       policy=RangePolicy(8, 16)):
                pass
        assert not live
    asyncio.run(run())


def test_early_close_and_empty_plan_do_not_leave_tasks():
    async def run():
        active = set()
        async def fetch(a, b):
            active.add(a)
            try:
                await asyncio.sleep(0 if a == 0 else 100)
                return bytes(b - a)
            finally:
                active.remove(a)
        stream = read_ranges(fetch, [(0, 8), (16, 24)], policy=RangePolicy(8, 16))
        await anext(stream)
        await stream.aclose()
        assert not active
        assert [x async for x in read_ranges(fetch, [], policy=RangePolicy(8, 8))] == []
    asyncio.run(run())
