# SPDX-License-Identifier: Apache-2.0
import threading

import numpy as np
import pytest

from arrayroute.execution import ordered_map
from arrayroute.statistics import CanonicalChannelMoments, ChannelMoments


def test_partitioned_masked_moments_match_dense_reference():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(197, 13, 4)).astype(np.float32)
    mask = rng.random(x.shape) > 0.3
    x[~mask] = np.nan
    full = ChannelMoments(4)
    partitions = []
    for block, valid in zip(np.array_split(x, 7), np.array_split(mask, 7), strict=True):
        part = ChannelMoments(4)
        part.update(block, valid)
        partitions.append(part)
    for part in reversed(partitions):
        full.merge(part)
    for c in range(4):
        reference = x[..., c][mask[..., c]].astype(np.float64)
        assert full.count[c] == reference.size
        assert full.mean[c] == pytest.approx(reference.mean(), abs=1e-14)
        assert full.variance[c] == pytest.approx(reference.var(), abs=1e-14)


@pytest.mark.parametrize("accumulator", [ChannelMoments, CanonicalChannelMoments])
def test_large_offset_preserves_small_fluctuations(accumulator):
    x = (1e12 + np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], 1000))[:, None]
    moments = accumulator(1)
    for block in np.array_split(x, 25):
        moments.update(block)
    assert moments.mean[0] == 1e12
    assert moments.variance[0] == pytest.approx(2.0, abs=1e-12)


def test_empty_mask_and_nonfinite_validation():
    moments = ChannelMoments(2)
    moments.update(np.empty((0, 2)))
    moments.update(np.full((2, 2), np.nan), np.zeros((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="No valid observations"):
        _ = moments.variance
    with pytest.raises(ValueError, match="Nonfinite"):
        moments.update(np.array([[0, np.inf]]))
    with pytest.raises(ValueError, match="Validity"):
        moments.update(np.ones((2, 2)), np.ones((2, 1), dtype=bool))
    moments.update(np.array([[1, np.nan]]), np.array([[True, False]]))
    moments.update(np.array([[3, 9]]))
    np.testing.assert_array_equal(moments.count, [2, 1])
    np.testing.assert_allclose(moments.mean, [2, 9])
    np.testing.assert_allclose(moments.variance, [1, 0])


def _bits(moments):
    return tuple(a.tobytes() for a in (moments.count, moments.mean, moments.variance))


@pytest.mark.parametrize("channels", [1, 4])
@pytest.mark.parametrize("masked", [False, True])
def test_canonical_moments_are_bitwise_independent_of_read_boundaries(channels, masked):
    rng = np.random.default_rng(20261023)
    values = rng.normal(size=(131071, channels)).astype(np.float32)
    valid = rng.random(values.shape) > 0.25 if masked else None
    if valid is not None:
        valid[:4100, 0] = False
        values[~valid] = np.nan
    reference = CanonicalChannelMoments(channels)
    reference.update(values, valid)
    for partitions in (7, 64, 257):
        result = CanonicalChannelMoments(channels)
        for indices in np.array_split(np.arange(len(values)), partitions):
            result.update(values[indices], None if valid is None else valid[indices])
            snapshot = result.snapshot()
            snapshot.mean[:] = 123  # A consumer cannot mutate the accumulator.
        assert _bits(result) == _bits(reference)
        assert result.mean_square.tobytes() == reference.mean_square.tobytes()
    for channel in range(channels):
        x = values[:, channel] if valid is None else values[valid[:, channel], channel]
        x = x.astype(np.float64)
        assert reference.mean[channel] == pytest.approx(x.mean(), abs=1e-14)
        assert reference.variance[channel] == pytest.approx(x.var(), abs=1e-14)
        assert reference.mean_square[channel] == pytest.approx((x * x).mean(), abs=1e-14)


def test_canonical_memory_layout_and_single_row_updates():
    x = np.random.default_rng(5).normal(size=(137, 59, 4))
    reference = CanonicalChannelMoments(4)
    reference.update(x)
    for layout in (np.asfortranarray(x), x[..., ::-1].copy()[..., ::-1]):
        candidate = CanonicalChannelMoments(4)
        for row in layout.reshape(-1, 4):
            candidate.update(row)
        assert _bits(candidate) == _bits(reference)


def test_canonical_invalid_update_is_atomic_and_empty_channels_fail():
    moments = CanonicalChannelMoments(2)
    moments.update(np.ones((7, 2)))
    before = _bits(moments)
    invalid = np.ones((10000, 2))
    invalid[-1, -1] = np.inf
    with pytest.raises(ValueError, match="Nonfinite"):
        moments.update(invalid)
    assert _bits(moments) == before
    with pytest.raises(ValueError, match="Validity"):
        moments.update(np.ones((7, 2)), np.ones((7, 1), bool))
    with pytest.raises(ValueError, match="real numeric"):
        moments.update(np.ones((7, 2), complex))
    moments.update(np.empty((0, 2)))
    assert _bits(moments) == before
    missing = CanonicalChannelMoments(2)
    missing.update(np.full((10000, 2), np.nan), np.zeros((10000, 2), bool))
    with pytest.raises(ValueError, match="No valid observations"):
        _ = missing.variance


@pytest.mark.parametrize("workers", [1, 4])
def test_canonical_reduction_with_out_of_order_reader_completion(workers):
    values = np.random.default_rng(42).normal(size=(20173, 3))
    reference = CanonicalChannelMoments(3)
    reference.update(values)
    parts = np.array_split(values, 9)
    second_done = threading.Event()
    completed = []

    def factory():
        def read(index):
            if workers > 1 and index == 0:
                assert second_done.wait(5)
            completed.append(index)
            if index == 1:
                second_done.set()
            return parts[index]
        return read

    actual = CanonicalChannelMoments(3)
    with ordered_map(factory, range(len(parts)), workers=workers) as buffers:
        for buffer in buffers:
            actual.update(buffer)
    if workers > 1:
        assert completed[0] != 0
    assert _bits(actual) == _bits(reference)
