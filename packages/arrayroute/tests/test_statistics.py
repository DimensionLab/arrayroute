# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.statistics import ChannelMoments


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


def test_large_offset_preserves_small_fluctuations():
    x = (1e12 + np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], 1000))[:, None]
    moments = ChannelMoments(1)
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
