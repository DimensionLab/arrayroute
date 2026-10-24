# SPDX-License-Identifier: Apache-2.0
import hashlib
import weakref

import numpy as np
import pytest

from arrayroute.backends import NumpyBackend
from arrayroute.budget import ReadBudget
from arrayroute.derived import VectorMagnitudeBackend
from arrayroute.quantiles import finite_statistics
from arrayroute.reader import ArrayReader, ReadStats


@pytest.mark.parametrize("dtype", ["f4", ">f4", "f8", ">f8", "i4", "u8", "bool"])
@pytest.mark.parametrize("selection", [
    (slice(None), slice(None)), (slice(1, 17, 3), slice(2, 18, 4)), (-1, 3),
    (slice(0, 0), slice(None)),
])
def test_magnitudes_preserve_selection_order_and_dtype(dtype, selection):
    a = np.random.default_rng(113).integers(0, 100, (17, 19, 3)).astype(dtype)
    original = a.tobytes()
    backend = VectorMagnitudeBackend(ArrayReader(NumpyBackend(a), buffer_bytes=256),
                                     workspace_bytes=128)
    result = ArrayReader(backend, buffer_bytes=4096).read(selection)
    expected = np.hypot.reduce(a.astype("f8"), axis=-1)[selection]
    assert result.dtype == np.dtype("f8")
    np.testing.assert_allclose(result, expected, rtol=3e-15, atol=0)
    assert a.tobytes() == original


def test_extreme_vectors_and_nonfinite_values():
    a = np.array([[1e308, 1e308], [1e-300, 1e-300], [np.nan, 1], [np.inf, 1], [0., -0.]])
    backend = VectorMagnitudeBackend(ArrayReader(NumpyBackend(a)))
    result = ArrayReader(backend).read((slice(None),))
    assert np.isfinite(result[0]) and result[0] == pytest.approx(np.sqrt(2) * 1e308)
    assert result[1] > 0 and result[1] / 1e-300 == pytest.approx(np.sqrt(2))
    assert np.isnan(result[2]) and np.isinf(result[3]) and result[4] == 0


def test_bounded_derived_quantiles_are_reproducible_and_record_source_io():
    a = np.random.default_rng(53).normal(size=(300_013, 3)).astype("f4")
    expected = np.sqrt(np.sum(a.astype("f8")**2, axis=-1))
    results = []
    for source_budget, transform_budget, stats_budget in [
        (64 << 10, 16 << 10, 4 << 20), (2 << 20, 1 << 20, 32 << 20),
    ]:
        stats = ReadStats()
        source = ArrayReader(NumpyBackend(a), buffer_bytes=source_budget)
        with pytest.raises(MemoryError):
            source.read((slice(None), slice(None)))
        backend = VectorMagnitudeBackend(source, workspace_bytes=transform_budget,
                                         source_stats=stats)
        result = finite_statistics(ArrayReader(backend, buffer_bytes=2 << 20),
                                   workspace_bytes=stats_budget)
        assert stats.requests > 1 and stats.output_bytes == a.nbytes * result.passes
        assert result.source_sha256 == hashlib.sha256(expected.tobytes()).hexdigest()
        np.testing.assert_array_equal(result.quantiles, np.quantile(expected, [.01, .99]))
        results.append(result)
    assert results[0].quantiles == results[1].quantiles
    assert results[0].mean.hex() == results[1].mean.hex()
    assert results[0].std.hex() == results[1].std.hex()


def test_source_pool_released_after_success_failure_and_cancellation():
    pool = ReadBudget(1024)
    source_backend = NumpyBackend(np.ones((100, 3)))
    source = ArrayReader(source_backend, shared_budget=pool)
    derived = VectorMagnitudeBackend(source, workspace_bytes=256)
    ArrayReader(derived).read((slice(None),))
    assert pool.snapshot()["used_bytes"] == 0
    def fail(selection):
        raise OSError("broken source")
    source_backend.read = fail
    with pytest.raises(OSError, match="broken source"):
        ArrayReader(derived).read((slice(None),))
    assert pool.snapshot()["used_bytes"] == 0
    derived.cancelled = lambda: True
    with pytest.raises(InterruptedError):
        ArrayReader(derived).read((slice(None),))


def test_invalid_workspace_and_layout_precede_payload_reads():
    source = ArrayReader(NumpyBackend(np.ones((4, 3))))
    for workspace in [0, True, 1.5]:
        with pytest.raises(ValueError):
            VectorMagnitudeBackend(source, workspace_bytes=workspace)
    with pytest.raises(MemoryError):
        VectorMagnitudeBackend(source, workspace_bytes=1)
    with pytest.raises(ValueError):
        VectorMagnitudeBackend(ArrayReader(NumpyBackend(np.ones(4))))
    with pytest.raises(TypeError):
        VectorMagnitudeBackend(ArrayReader(NumpyBackend(np.ones((4, 3), dtype='c8'))))


def test_mid_read_cancellation_stops_source_requests():
    source = ArrayReader(NumpyBackend(np.ones((100, 3))))
    stats = ReadStats()
    backend = VectorMagnitudeBackend(source, workspace_bytes=128, source_stats=stats,
                                     cancelled=lambda: stats.requests >= 2)
    with pytest.raises(InterruptedError):
        ArrayReader(backend).read((slice(None),))
    assert stats.requests == 2


def test_float32_full_finite_range_is_safe_for_square_sum():
    tiny = np.nextafter(np.float32(0), np.float32(1))
    largest = np.finfo('f4').max
    a = np.array([[tiny, tiny, tiny], [largest, largest, largest]], dtype='f4')
    backend = VectorMagnitudeBackend(ArrayReader(NumpyBackend(a)))
    result = ArrayReader(backend).read((slice(None),))
    assert backend.reduction == 'square-sum'
    assert np.all(np.isfinite(result)) and np.all(result > 0)
    np.testing.assert_allclose(result, np.hypot.reduce(a.astype('f8'), axis=-1), rtol=3e-15)


def test_integer_extremes_are_converted_before_squaring():
    a = np.array([[np.iinfo('u8').max]*3], dtype='u8')
    backend = VectorMagnitudeBackend(ArrayReader(NumpyBackend(a)))
    result = ArrayReader(backend).read((slice(None),))
    np.testing.assert_allclose(result, np.hypot.reduce(a.astype('f8'), axis=-1), rtol=3e-15)


@pytest.mark.parametrize('dtype', ['f4', 'f8'])
def test_nan_components_propagate_even_with_infinity(dtype):
    a = np.array([[np.inf, np.nan], [np.nan, np.inf]], dtype=dtype)
    backend = VectorMagnitudeBackend(ArrayReader(NumpyBackend(a)))
    assert np.all(np.isnan(ArrayReader(backend).read((slice(None),))))


def test_transform_does_not_retain_previous_converted_block(monkeypatch):
    import arrayroute.derived as module
    source = ArrayReader(NumpyBackend(np.ones((100, 3), dtype='f8')))
    original_read, original_isnan = module.read_flat, np.isnan
    references = []
    def record_isnan(values):
        if values.base is not None:
            references.append(weakref.ref(values.base))
        return original_isnan(values)
    def read(*args, **kwargs):
        assert all(ref() is None for ref in references)
        return original_read(*args, **kwargs)
    monkeypatch.setattr(module, 'read_flat', read)
    monkeypatch.setattr(np, 'isnan', record_isnan)
    ArrayReader(VectorMagnitudeBackend(source, workspace_bytes=256)).read((slice(None),))
    assert references and all(ref() is None for ref in references)
