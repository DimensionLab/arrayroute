# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.normalization import normalize_channels
from arrayroute.torch_transforms import TorchChannelNormalizer

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("axis", [-1, 0, 1])
def test_normalization_preserves_input_and_matches_cpu(axis):
    source = np.random.default_rng(812).normal(size=(3, 5, 2)).astype("float32")
    mean, scale = [0.2, -1.7], [0.35, 8.0]
    expected = normalize_channels(source, mean, scale)
    value = torch.from_numpy(np.moveaxis(source.copy(), -1, axis))
    original = value.clone()
    transform = TorchChannelNormalizer(mean, scale)
    mean[0] = 200  # Caller mutation cannot change a recorded transform.
    actual = transform(value, channel_axis=axis)
    np.testing.assert_array_equal(np.moveaxis(actual.numpy(), axis, -1), expected)
    assert torch.equal(value, original)
    assert actual.data_ptr() != value.data_ptr()


@pytest.mark.parametrize("mean,scale", [([], []), ([0], [0]), ([float('nan')], [1]),
                                         ([0], [float('inf')]), ([0, 1], [1])])
def test_invalid_statistics(mean, scale):
    with pytest.raises(ValueError):
        TorchChannelNormalizer(mean, scale)


def test_invalid_inputs():
    transform = TorchChannelNormalizer([0, 0], [1, 1])
    for value, axis in [(torch.zeros(3), -1), (torch.zeros(2).double(), -1),
                        (torch.zeros(2), 1), (torch.zeros(2), True), (torch.tensor(1.), -1)]:
        with pytest.raises(ValueError):
            transform(value, channel_axis=axis)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is required")
def test_cuda_replay_and_stream_ownership():
    source = np.random.default_rng(121).normal(size=(8, 13, 3)).astype("float32")
    transform = TorchChannelNormalizer([0.25, -0.75, 1.3], [0.4, 2.1, 9.7], device="cuda")
    stream = torch.cuda.Stream()
    value = torch.tensor(source, device="cuda")
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        first = transform(value)
        second = transform(value)
        value.record_stream(stream)
    torch.cuda.current_stream().wait_stream(stream)
    first.record_stream(torch.cuda.current_stream())
    second.record_stream(torch.cuda.current_stream())
    assert torch.equal(first, second)
    expected = normalize_channels(source, [0.25, -0.75, 1.3], [0.4, 2.1, 9.7])
    np.testing.assert_allclose(first.cpu().numpy(), expected, rtol=2e-7, atol=1e-7)
    np.testing.assert_array_equal(value.cpu().numpy(), source)
