# SPDX-License-Identifier: Apache-2.0
import pickle
import sys
from dataclasses import replace

import pytest

from arrayroute.permutation import BatchPartition, Permutation


def test_versioned_order_can_be_replayed():
    assert Permutation.algorithm == "feistel-blake2b-8-cw-v1"
    assert list(Permutation(17, seed=73, epoch=4)) == [
        3,
        12,
        11,
        9,
        13,
        10,
        14,
        6,
        7,
        15,
        2,
        4,
        8,
        1,
        0,
        5,
        16,
    ]


def test_complete_permutations_and_inverses_across_domain_boundaries():
    sizes = [*range(130), 255, 256, 257, 1023, 1024, 1025]
    for size in sizes:
        order = Permutation(size, seed=413, epoch=6)
        values = list(order)
        assert sorted(values) == list(range(size))
        assert [order.inverse(value) for value in values] == list(range(size))


@pytest.mark.parametrize("size,workers", [(1003, 7), (3, 8), (0, 3)])
def test_partitioned_positions_are_disjoint_and_cover_epoch_without_padding(size, workers):
    order = Permutation(size, seed=88, epoch=3)
    parts = [list(order.indices(start=rank, step=workers)) for rank in range(workers)]
    combined = [value for part in parts for value in part]
    assert len(combined) == len(set(combined)) == len(order)
    assert sorted(combined) == list(range(len(order)))
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_reproducible_resume_and_large_domain_random_access():
    for size in [1, 10**12, sys.maxsize]:
        order = Permutation(size, seed=2**64 - 1)
        clone = pickle.loads(pickle.dumps(order))
        for position in {0, size // 2, size - 1}:
            value = order[position]
            assert 0 <= value < size
            assert order.inverse(value) == position
            assert clone[position] == value
        assert len(pickle.dumps(order)) < 1024
    order = Permutation(200, seed=8, epoch=5)
    assert list(order.indices(start=73)) == list(order)[73:]
    assert order[-1] == order[199]
    assert list(order) != list(Permutation(200, seed=8, epoch=6))
    assert list(order) != list(Permutation(200, seed=9, epoch=5))


def test_invalid_domains_and_positions_fail():
    for kwargs in [{"size": -1}, {"size": sys.maxsize + 1}, {"size": 2, "epoch": -1}]:
        with pytest.raises(ValueError):
            Permutation(**kwargs)
    with pytest.raises(TypeError):
        Permutation(2.5)
    with pytest.raises(IndexError):
        _ = Permutation(0)[0]
    with pytest.raises(IndexError):
        _ = Permutation(3)[3]
    with pytest.raises(ValueError):
        Permutation(3).indices(step=0)
    assert list(Permutation(0)) == []


@pytest.mark.parametrize("tail", ["drop", "uneven"])
def test_batched_rank_partitions_reconstruct_shared_epoch_and_resume(tail):
    for size in (0, 1, 3, 16, 17, 65):
        for world in (1, 2, 5, 19):
            for batch_size in (1, 4, 11):
                plans = [BatchPartition(size, rank, world, batch_size, tail, seed=19, epoch=7)
                         for rank in range(world)]
                batches = [list(plan) for plan in plans]
                positions = [p for plan in plans for p in plan.positions()]
                used = size - plans[0].dropped_samples
                assert sorted(positions) == list(range(used))
                assert len(positions) == len(set(positions))
                values = {p: v for plan, rows in zip(plans, batches, strict=True)
                          for p, v in zip(plan.positions(), sum(rows, []), strict=True)}
                assert [values[p] for p in range(used)] == list(Permutation(
                    size, seed=19, epoch=7).indices(stop=used))
                assert len(set(values.values())) == used
                if tail == "drop":
                    assert len({len(rows) for rows in batches}) == 1
                    assert all(len(row) == batch_size for rows in batches for row in rows)
                else:
                    assert used == size
                for plan, rows in zip(plans, batches, strict=True):
                    assert len(plan) == len(rows)
                    assert plan.sample_count == sum(map(len, rows))
                    for consumed in {0, plan.global_batches // 2, plan.global_batches}:
                        resumed = replace(plan, start_batch=consumed)
                        assert list(resumed) == rows[consumed:]
                        assert list(BatchPartition.from_state_dict(resumed.state_dict())) == (
                            rows[consumed:]
                        )


def test_large_partition_resume_does_not_materialize_epoch_or_consume_prefetch():
    plan = BatchPartition(10**12 + 39, 71, 128, 8, "drop", seed=3, epoch=5,
                          start_batch=100_000_000)
    before = plan.state_dict()
    first = next(iter(plan))
    assert before == plan.state_dict()
    assert len(pickle.dumps(plan)) < 2048
    assert first == [Permutation(plan.size, seed=3, epoch=5)[position]
                     for position in plan.positions()[:8]]
    assert next(iter(BatchPartition.from_state_dict(before))) == first
    assert plan.dropped_samples == plan.size % (128 * 8)


@pytest.mark.parametrize("changes", [
    {"rank": -1}, {"rank": 2}, {"world_size": 0}, {"batch_size": 0},
    {"batch_size": sys.maxsize + 1}, {"size": -1}, {"tail": "pad"},
    {"start_batch": -1}, {"start_batch": 99}, {"seed": -1},
])
def test_partition_rejects_invalid_contract(changes):
    options = dict(size=7, rank=0, world_size=2, batch_size=2, tail="drop")
    with pytest.raises(ValueError):
        BatchPartition(**(options | changes))


def test_partition_state_requires_complete_versioned_identity():
    state = BatchPartition(7, 0, 2, 2, "uneven").state_dict()
    for bad in [state | {"algorithm": "future"}, state | {"permutation": "future"},
                state | {"unexpected": 4}, {k: v for k, v in state.items() if k != "rank"}]:
        with pytest.raises(ValueError):
            BatchPartition.from_state_dict(bad)


def test_rank_planning_benchmark_reports_sampled_indices_not_full_domain():
    from arrayroute.benchmark_partition import measure

    report = measure(2049, 8, 4, 2, epoch=3)
    assert report["sampled_indices"] == 64
    assert report["dropped_global_tail"] == 1
    assert report["start_batch"] == 32
    assert report["max_pickled_plan_bytes"] < 2048
    assert len(report["rank_planning_seconds"]) == 8
    assert report["indices_sha256"] == measure(2049, 8, 4, 2, epoch=3)["indices_sha256"]
