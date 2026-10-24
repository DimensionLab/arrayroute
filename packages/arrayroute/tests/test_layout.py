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


def test_station_columns_split_variables_and_preserve_bounded_objects():
    from arrayroute import ArraySpec, plan

    shape = (192, 13, 33, 81, 5)
    layout = GridLayoutPolicy(chunk_edge=1, shard_edge=32).plan(
        shape,
        itemsize=4,
        spatial_axes=(2, 3),
        time_axis=0,
        steps_per_chunk=16,
        axis_chunk_limits={4: 1},
    )
    assert layout.chunks == (16, 13, 1, 1, 1)
    assert layout.shards == (16, 13, 32, 32, 1)
    assert layout.object_count_upper_bound == 360
    request = plan(
        ArraySpec(shape, layout.chunks, 4),
        (slice(None), slice(None), slice(16, 17), slice(40, 41), slice(0, 1)),
    )
    assert request.output_bytes == request.decoded_bytes_upper_bound == 9984
    assert request.chunks_touched == 12


def test_axis_caps_compose_with_time_spatial_and_byte_limits_deterministically():
    policy = GridLayoutPolicy(chunk_bytes=2048, shard_bytes=8192)
    options = dict(itemsize=4, spatial_axes=(1, 2), time_axis=0, steps_per_chunk=8)
    first = policy.plan((100, 129, 131, 50), axis_chunk_limits={0: 3, 1: 7, 3: 2}, **options)
    second = policy.plan((100, 129, 131, 50), axis_chunk_limits={3: 2, 1: 7, 0: 3}, **options)
    assert first == second
    assert first.chunks[0] <= 3 and first.chunks[1] <= 7 and first.chunks[3] <= 2
    assert first.decoded_chunk_bytes <= 2048 and first.decoded_shard_bytes <= 8192


@pytest.mark.parametrize(
    "limits", [{-1: 1}, {3: 1}, {1: 0}, {1: -1}, {1: 1.5}, {True: 1}, {1: True}]
)
def test_invalid_axis_caps(limits):
    with pytest.raises((TypeError, ValueError)):
        GridLayoutPolicy().plan(
            (20, 30, 4), itemsize=4, spatial_axes=(0, 1), axis_chunk_limits=limits
        )


def test_tiny_histories_pack_time_without_enumerating_rows():
    policy = GridLayoutPolicy()
    layout = policy.plan((10**12, 1, 1, 1), itemsize=4, spatial_axes=(1, 2), time_axis=0)
    assert layout.chunks == (4096, 1, 1, 1)
    assert layout.shards is None
    assert layout.decoded_chunk_bytes == 16384
    assert layout.object_count_upper_bound == (10**12 + 4095) // 4096
    assert layout == policy.plan((10**12, 1, 1, 1), itemsize=4, spatial_axes=(1, 2), time_axis=0)


def test_auto_time_uses_full_frame_not_spatial_brick_and_respects_overrides():
    policy = GridLayoutPolicy(
        chunk_edge=4, shard_edge=8, chunk_bytes=4096, shard_bytes=16384, max_time_steps=32
    )
    small = policy.plan((1000, 8, 8, 2), itemsize=4, spatial_axes=(1, 2), time_axis=0)
    assert small.chunks[0] == 8
    large = policy.plan((1000, 64, 64, 2), itemsize=4, spatial_axes=(1, 2), time_axis=0)
    assert large.chunks[0] == 1
    explicit = policy.plan(
        (1000, 1, 1, 1), itemsize=4, spatial_axes=(1, 2), time_axis=0, steps_per_chunk=64
    )
    assert explicit.chunks[0] == 64
    isolated = policy.plan(
        (1000, 1, 1, 1), itemsize=4, spatial_axes=(1, 2), time_axis=0, steps_per_chunk=1
    )
    assert isolated.chunks == (1, 1, 1, 1)
    capped = policy.plan(
        (1000, 1, 1, 1), itemsize=4, spatial_axes=(1, 2), time_axis=0, axis_chunk_limits={0: 3}
    )
    assert capped.chunks[0] == 3


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_automatic_time_limit(limit):
    with pytest.raises((ValueError, TypeError)):
        GridLayoutPolicy(max_time_steps=limit)


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_invalid_explicit_time_limit(steps):
    with pytest.raises((ValueError, TypeError)):
        GridLayoutPolicy().plan(
            (20, 2, 2), itemsize=4, spatial_axes=(1,), time_axis=0, steps_per_chunk=steps
        )


def test_time_axis_can_be_in_the_middle():
    policy = GridLayoutPolicy(chunk_bytes=8192, shard_bytes=32768)
    layout = policy.plan((7, 10000, 9, 2), itemsize=8, spatial_axes=(0, 2), time_axis=1)
    assert layout.chunks == (7, 8, 9, 2)
    assert layout.decoded_chunk_bytes <= 8192
