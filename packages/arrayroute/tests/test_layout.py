# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import pytest

from arrayroute.layout import GridLayoutPolicy


def test_velocity_layout_keeps_frames_addressable_and_bricks_small():
    layout = GridLayoutPolicy().plan(
        (20, 128, 128, 128, 3), itemsize=4, spatial_axes=(1, 2, 3), time_axis=0
    )
    assert layout.chunks == (1, 32, 32, 32, 3)
    assert layout.shards == (1, 128, 128, 128, 3)
    assert layout.decoded_chunk_bytes == 393216
    assert layout.decoded_shard_bytes == 25165824
    assert layout.object_count_upper_bound == 20


def test_small_steady_arrays_do_not_get_unnecessary_sharding():
    layout = GridLayoutPolicy().plan((4, 7, 2), itemsize=8, spatial_axes=(0, 1))
    assert layout.chunks == (4, 7, 2) and layout.shards is None
    assert layout.object_count_upper_bound == 1


def test_padded_irregular_shapes_and_large_components_obey_byte_limits():
    rng = np.random.default_rng(71)
    for _ in range(150):
        ndim = int(rng.integers(1, 4))
        spatial = tuple(map(int, rng.integers(1, 300, size=ndim)))
        shape = (17, *spatial, int(rng.integers(1, 300)))
        policy = GridLayoutPolicy(chunk_bytes=8192, shard_bytes=65536)
        layout = policy.plan(
            shape,
            itemsize=8,
            spatial_axes=tuple(range(1, ndim + 1)),
            time_axis=0,
            steps_per_chunk=4,
        )
        shard = layout.shards or layout.chunks
        assert layout.decoded_chunk_bytes <= 8192
        assert layout.decoded_shard_bytes <= 65536
        assert layout.chunks[0] <= 4 and shard[0] == layout.chunks[0]
        assert all(s % c == 0 for s, c in zip(shard, layout.chunks, strict=True))
        assert all(1 <= c <= n for c, n in zip(layout.chunks, shape, strict=True))
        assert layout.object_count_upper_bound == math.prod(
            (n + s - 1) // s for n, s in zip(shape, shard, strict=True)
        )


def test_large_logical_shape_is_planned_without_enumeration_or_fixed_width_overflow():
    layout = GridLayoutPolicy().plan((2**40, 2**40, 2**40, 3), itemsize=4, spatial_axes=(0, 1, 2))
    assert layout.object_count_upper_bound == (2**33) ** 3


@pytest.mark.parametrize(
    "options", [{"chunk_bytes": 0}, {"shard_edge": 16}, {"shard_bytes": 1024}, {"chunk_edge": 1.5}]
)
def test_invalid_policy(options):
    with pytest.raises((ValueError, TypeError)):
        GridLayoutPolicy(**options)


@pytest.mark.parametrize(
    "shape,axes,time",
    [((0, 2), (0,), None), ((2, 2), (0, 0), None), ((2, 2), (2,), None), ((2, 2), (0,), 0)],
)
def test_invalid_grid(shape, axes, time):
    with pytest.raises(ValueError):
        GridLayoutPolicy().plan(shape, itemsize=4, spatial_axes=axes, time_axis=time)
