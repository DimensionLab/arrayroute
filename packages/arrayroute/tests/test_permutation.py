# SPDX-License-Identifier: Apache-2.0
import pickle
import sys

import pytest

from arrayroute.permutation import Permutation


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
