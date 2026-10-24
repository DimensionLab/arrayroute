# SPDX-License-Identifier: Apache-2.0
import itertools
import math

import numpy as np
import pytest

from arrayroute.regions import block_regions


@pytest.mark.parametrize("shape,blocks", [
    ((17,), (6,)), ((9, 17), (4, 7)), ((7, 9, 13, 3), (3, 4, 5, 3)),
    ((1, 3, 1), (9, 8, 4)), ((2, 1, 3, 1, 4), (1, 1, 2, 1, 3)),
])
@pytest.mark.parametrize("limit", [1, 7, 32, 1000000])
@pytest.mark.parametrize("coalesce", [False, True])
def test_exact_once_bounded_coverage_and_chunk_containment(shape, blocks, limit, coalesce):
    seen = np.zeros(shape, dtype="uint8")
    previous = None
    finished = set()
    block_size = math.prod(min(n, b) for n, b in zip(shape, blocks, strict=True))
    combined = coalesce and block_size <= limit
    for region in block_regions(shape, blocks, max_elements=limit, coalesce=coalesce):
        assert len(region) == len(shape)
        assert math.prod(r.stop - r.start for r in region) <= limit
        chunk = tuple(r.start // b for r, b in zip(region, blocks, strict=True))
        if combined:
            assert all(r.start % b == 0 and (r.stop % b == 0 or r.stop == n)
                       for r, b, n in zip(region, blocks, shape, strict=True))
        else:
            assert all(r.start // b == (r.stop - 1) // b
                       for r, b in zip(region, blocks, strict=True))
        if chunk != previous:
            assert chunk not in finished
            if previous is not None:
                assert chunk > previous
                finished.add(previous)
            previous = chunk
        seen[region] += 1
    np.testing.assert_array_equal(seen, np.ones(shape, dtype="uint8"))


def test_one_write_per_chunk_when_chunk_fits():
    result = list(block_regions((33, 24, 24, 12, 3), (6, 24, 24, 12, 3),
                                max_elements=6 * 24 * 24 * 12 * 3))
    assert len(result) == 6
    assert result[-1] == (slice(30, 33), slice(0, 24), slice(0, 24),
                          slice(0, 12), slice(0, 3))


def test_lazy_extreme_scale_grid_does_not_materialize_axis_pools():
    result = list(itertools.islice(block_regions((10**15, 10**15), (1, 1),
                                                 max_elements=1), 3))
    assert result == [(slice(0, 1), slice(i, i + 1)) for i in range(3)]


def test_zero_extent():
    assert list(block_regions((3, 0, 7), (2, 2, 2), max_elements=9)) == []


def test_coalesced_batches_preserve_chunk_boundaries():
    regions = list(block_regions((65, 32, 32, 16, 3), (2, 32, 32, 16, 3),
                                 max_elements=20 * 32 * 32 * 16 * 3, coalesce=True))
    assert len(regions) == 4
    assert [(r[0].start, r[0].stop) for r in regions] == [(0, 20), (20, 40), (40, 60), (60, 65)]


def test_coalescing_must_be_explicit_boolean():
    with pytest.raises(TypeError, match="boolean"):
        list(block_regions((2,), (1,), max_elements=2, coalesce="yes"))


@pytest.mark.parametrize("shape,blocks,limit", [
    ((), (), 1), ((3,), (), 1), ((3,), (0,), 1), ((-1,), (1,), 1),
    ((3,), (1,), 0), ((3,), (1,), True),
])
def test_invalid_contract(shape, blocks, limit):
    with pytest.raises(ValueError):
        list(block_regions(shape, blocks, max_elements=limit))
