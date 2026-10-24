# SPDX-License-Identifier: Apache-2.0
import hashlib
from dataclasses import asdict, replace

import numpy as np
import pytest

from arrayroute.backends import NumpyBackend
from arrayroute.reader import ArrayReader, ReadStats
from arrayroute.standardized import StandardizedChannelsBackend, fit_channel_standardization


def reader(a, budget=1 << 20):
    return ArrayReader(NumpyBackend(a), buffer_bytes=budget)


def expected(a):
    return (a - np.nanmean(a, axis=0)) / np.where(np.nanstd(a, axis=0) > 0,
                                                np.nanstd(a, axis=0), 1)


@pytest.mark.parametrize('dtype', ['f4', '>f4', 'f8', '>f8', 'i4', 'u8', 'bool'])
@pytest.mark.parametrize('selection', [
    (slice(None), slice(None)), (slice(2, 16, 3), slice(1, 8, 2)), (-1, -2),
    (slice(0, 0), slice(None)), (slice(None), slice(0, 0)),
])
def test_aligned_sources_selection_and_source_identity(dtype, selection):
    rng = np.random.default_rng(5)
    arrays = [rng.integers(0, 100, (17, c)).astype(dtype) for c in [2, 3, 4]]
    signatures = [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]
    backend = StandardizedChannelsBackend([reader(a, 256) for a in arrays])
    result = ArrayReader(backend).read(selection)
    baseline = np.concatenate([expected(a.astype('f8')) for a in arrays], axis=1)[selection]
    np.testing.assert_allclose(result, baseline, rtol=2e-14, atol=1e-14)
    assert [p.source_sha256 for p in backend.parameters] == signatures
    assert [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays] == signatures


def test_replay_across_read_and_workspace_budgets():
    a = np.random.default_rng(81).normal(size=(12001, 4)) + 1e8
    a[5:200:3, 0] = np.nan
    fits = []
    results = []
    for budget, workspace in [(256, 4 << 20), (1 << 20, 32 << 20)]:
        stats = ReadStats()
        backend = StandardizedChannelsBackend([reader(a, budget)], workspace_bytes=workspace,
                                             source_stats=stats)
        assert stats.output_bytes == a.nbytes * 2
        fits.append(backend.parameters)
        results.append(ArrayReader(backend).read((slice(None), slice(None))))
    assert fits[0] == fits[1]
    np.testing.assert_array_equal(results[0], results[1])
    # Offset subtraction is deliberately more accurate than whole-array raw mean summation.
    centered = a - 1e8
    np.testing.assert_allclose(results[0], expected(centered), rtol=2e-12, atol=2e-12)


def test_extremes_constant_and_missing_channels():
    a = np.array([[-1e308, -1e-300, 3, np.nan], [1e308, 1e-300, 3, np.inf]])
    backend = StandardizedChannelsBackend([reader(a)])
    result = ArrayReader(backend).read((slice(None), slice(None)))
    np.testing.assert_array_equal(result[:, :3], [[-1, -1, 0], [1, 1, 0]])
    assert np.isnan(result[:, 3]).all()
    assert backend.parameters[0].count == (2, 2, 2, 0)


def test_nonfinite_values_do_not_poison_valid_channel_measurements():
    a = np.array([[1., np.nan], [2., np.inf], [3., -np.inf]])
    backend = StandardizedChannelsBackend([reader(a)])
    result = ArrayReader(backend).read((slice(None), slice(None)))
    np.testing.assert_allclose(result[:, 0], [-np.sqrt(1.5), 0, np.sqrt(1.5)])
    assert np.isnan(result[:, 1]).all()


def test_empty_history():
    backend = StandardizedChannelsBackend([reader(np.empty((0, 3)))])
    assert ArrayReader(backend).read((slice(None), slice(None))).shape == (0, 3)


def test_invalid_sources_and_admission_precede_reads():
    backend = NumpyBackend(np.ones((10, 2)))
    def fail(selection):
        pytest.fail('Unexpected read before admission')
    backend.read = fail
    with pytest.raises(MemoryError):
        StandardizedChannelsBackend([ArrayReader(backend), reader(np.ones((10, 100)))])
    with pytest.raises(ValueError, match='same row count'):
        StandardizedChannelsBackend([ArrayReader(backend), reader(np.ones((9, 2)))])
    with pytest.raises(InterruptedError):
        StandardizedChannelsBackend([ArrayReader(backend)], cancelled=lambda: True)
    with pytest.raises(MemoryError):
        StandardizedChannelsBackend([ArrayReader(backend)], workspace_bytes=1)
    with pytest.raises(ValueError):
        StandardizedChannelsBackend([])


def test_mutation_between_fit_passes_fails():
    a = np.ones((10, 2))
    backend = NumpyBackend(a)
    original = backend.read
    calls = 0
    def read(selection):
        nonlocal calls
        calls += 1
        if calls == 2:
            a[0, 0] = 2
        return original(selection)
    backend.read = read
    with pytest.raises(ValueError, match='changed between'):
        fit_channel_standardization(ArrayReader(backend))


def test_cancellation_after_fit_stops_payload_reads():
    cancelled = False
    backend = StandardizedChannelsBackend([reader(np.ones((10, 2)))],
                                         cancelled=lambda: cancelled)
    cancelled = True
    with pytest.raises(InterruptedError):
        ArrayReader(backend).read((slice(None), slice(None)))


def test_apply_training_statistics_without_refitting_held_out_data():
    from arrayroute.standardized import ChannelStandardization
    train = np.array([[1., 10.], [3., 30.]])
    fitted = fit_channel_standardization(reader(train))
    restored = ChannelStandardization(**asdict(fitted))
    stats = ReadStats()
    backend = StandardizedChannelsBackend([reader(np.array([[5., 50.]]))],
                                         parameters=[restored], source_stats=stats,
                                         workspace_bytes=64)
    assert stats.requests == 0
    np.testing.assert_array_equal(ArrayReader(backend).read((slice(None), slice(None))), [[3., 3.]])
    assert backend.parameters[0].source_sha256 == hashlib.sha256(train.tobytes()).hexdigest()
    with pytest.raises(ValueError, match='Invalid recorded'):
        StandardizedChannelsBackend([reader(train)], parameters=[replace(fitted, spread=(0., 1.))])
