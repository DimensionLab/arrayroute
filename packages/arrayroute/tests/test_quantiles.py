# SPDX-License-Identifier: Apache-2.0
import hashlib

import numpy as np
import pytest

from arrayroute.backends import NumpyBackend
from arrayroute.quantiles import finite_statistics
from arrayroute.reader import ArrayReader


def reader(values, budget=1 << 20):
    return ArrayReader(NumpyBackend(values), buffer_bytes=budget)


@pytest.mark.parametrize("dtype", ["f4", ">f4", "f8", ">f8", "i4", "u8", "bool"])
def test_small_exact_finite_quantiles_and_moments(dtype):
    values = np.random.default_rng(212).integers(0, 600, size=(17, 19)).astype(dtype)
    if values.dtype.kind == "f":
        values.flat[:4] = [np.nan, np.inf, -np.inf, -0.0]
    q = [0, 0.01, 0.5, 0.99, 1]
    result = finite_statistics(reader(values, 256), quantiles=q)
    a = values.astype("f8").ravel()
    a = a[np.isfinite(a)]
    np.testing.assert_array_equal(result.quantiles, np.quantile(a, q))
    assert result.mean == pytest.approx(a.mean(), rel=1e-14)
    assert result.std == pytest.approx(a.std(), rel=1e-14)
    assert result.passes == 1
    assert result.count == a.size
    assert result.source_sha256 == hashlib.sha256(values.tobytes()).hexdigest()


@pytest.mark.parametrize("distribution", ["normal", "repeated", "offset", "skew"])
def test_large_exact_selection_and_reproducible_moments(distribution):
    rng = np.random.default_rng(71)
    values = rng.normal(size=300_013)
    if distribution == "repeated":
        values = rng.choice([-0.0001, 0, 1, 200], size=values.size)
    elif distribution == "offset":
        values = 1e12 + values
    elif distribution == "skew":
        values = np.exp(values * 8)
    values[:5] = [np.nan, np.inf, -np.inf, -0.0, 0.0]
    q = [0, 0.01, 0.23, 0.5, 0.777, 0.99, 1]
    finite = values[np.isfinite(values)]
    a = finite_statistics(reader(values, 64 << 10), quantiles=q, workspace_bytes=4 << 20)
    b = finite_statistics(reader(values, 2 << 20), quantiles=q, workspace_bytes=32 << 20)
    assert a.passes > 1 and b.passes == 1
    np.testing.assert_array_equal(a.quantiles, np.quantile(finite, q))
    assert a.quantiles == b.quantiles
    assert a.mean.hex() == b.mean.hex()
    assert a.std.hex() == b.std.hex()
    assert a.mean == pytest.approx(finite.mean(), rel=3e-14, abs=1e-14)
    assert a.std == pytest.approx(finite.std(), rel=3e-14)
    assert a.source_sha256 == b.source_sha256 == hashlib.sha256(values.tobytes()).hexdigest()
    assert a.count == values.size - 3 and (a.n_nan, a.n_inf) == (1, 2)


def test_selection_and_native_bytes():
    values = np.arange(41 * 53, dtype=">f8").reshape(41, 53)
    selection = (slice(1, 38, 3), slice(2, 50, 5))
    selected = values[selection]
    result = finite_statistics(reader(values, 256), selection)
    np.testing.assert_array_equal(result.quantiles, np.quantile(selected, [0.01, 0.99]))
    assert result.source_sha256 == hashlib.sha256(selected.tobytes()).hexdigest()


@pytest.mark.parametrize(
    "values",
    [np.array([]), np.array([np.nan, np.inf, -np.inf]), np.zeros(300_001), np.full(300_001, -0.0)],
)
def test_empty_nonfinite_and_constant(values):
    result = finite_statistics(reader(values), workspace_bytes=4 << 20)
    assert result.passes == 1
    if not np.isfinite(values).any():
        assert np.isnan(result.mean) and all(np.isnan(result.quantiles))
    else:
        assert result.mean == 0 and result.std == 0
        assert all(v.hex() == "0x0.0p+0" for v in result.quantiles)


def test_extreme_finite_values_do_not_overflow_moments_or_interpolation():
    values = np.array([-1e308, 1e308])
    result = finite_statistics(reader(values), quantiles=[0, 0.25, 0.5, 0.75, 1])
    assert result.mean == 0 and result.std == 1e308
    assert result.quantiles == (-1e308, -5e307, 0, 5e307, 1e308)


def test_empty_quantiles_still_compute_moments():
    result = finite_statistics(
        reader(np.arange(300_001, dtype="f8")), quantiles=[], workspace_bytes=4 << 20
    )
    assert result.quantiles == () and result.mean == pytest.approx(150000, rel=2e-14)


def test_changed_source_fails_instead_of_mixing_passes():
    values = np.arange(300_001, dtype="f8")
    backend = NumpyBackend(values)
    original = backend.read
    reads = 0

    def read(selection):
        nonlocal reads
        if selection[0].start == 0:
            reads += 1
            if reads == 2:
                values[-1] += 1
        return original(selection)

    backend.read = read
    with pytest.raises(ValueError, match="changed|grew|shrank"):
        finite_statistics(ArrayReader(backend, buffer_bytes=1 << 20), workspace_bytes=4 << 20)
    assert reads >= 2


def test_cancellation_and_admission_precede_payload_reads():
    backend = NumpyBackend(np.arange(10, dtype="f4"))

    def reject(selection):
        pytest.fail("unexpected payload read")

    backend.read = reject
    r = ArrayReader(backend)
    with pytest.raises(InterruptedError):
        finite_statistics(r, cancelled=lambda: True)
    for workspace in [True, 0, 3 << 20, 1.5]:
        with pytest.raises(ValueError):
            finite_statistics(r, workspace_bytes=workspace)
    for q in [[np.nan], [-0.01], [1.01], [0] * 17]:
        with pytest.raises(ValueError):
            finite_statistics(r, quantiles=q)


def test_mid_pass_cancellation():
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls > 12

    with pytest.raises(InterruptedError):
        finite_statistics(reader(np.arange(300_001), 256), cancelled=cancel)


def test_repeated_neighbouring_values_resolve_without_unbounded_candidates():
    values = np.resize(np.array([1.0, np.nextafter(1.0, 2.0)]), 300_001)
    values[0] = 2.0
    q = [0.01, 0.25, 0.5, 0.75, 0.99]
    result = finite_statistics(reader(values), quantiles=q, workspace_bytes=4 << 20)
    np.testing.assert_array_equal(result.quantiles, np.quantile(values, q))
    assert 2 <= result.passes <= 9


def test_nonfinite_counts_are_bounded_exact_and_cancellable():
    from arrayroute.quantiles import count_nonfinite

    values = np.array([np.nan, np.inf, -np.inf, 1, -0.0] * 1001, dtype=">f8")
    assert count_nonfinite(reader(values, 128), block_elements=17) == (1001, 2002)
    assert count_nonfinite(reader(values, 128), (slice(0, 10),)) == (2, 4)
    with pytest.raises(InterruptedError):
        count_nonfinite(reader(values), cancelled=lambda: True)
    with pytest.raises(ValueError):
        count_nonfinite(reader(values), block_elements=1_000_001)


def test_uint64_conversion_is_explicit_and_native_hash_remains_exact():
    values = np.array([0, 2**53 + 1, 2**63 + 3, 2**64 - 1], dtype="u8")
    result = finite_statistics(reader(values), quantiles=[0.25, 0.5, 0.75])
    np.testing.assert_array_equal(
        result.quantiles, np.quantile(values.astype("f8"), [0.25, 0.5, 0.75])
    )
    assert result.source_sha256 == hashlib.sha256(values.tobytes()).hexdigest()


def test_nan_payload_change_is_detected_even_when_numeric_statistics_match():
    values = np.arange(300_001, dtype="f8")
    values[1] = np.nan
    backend = NumpyBackend(values)
    original = backend.read
    passes = 0

    def read(selection):
        nonlocal passes
        if selection[0].start == 0:
            passes += 1
            if passes == 2:
                values.view("u8")[1] ^= np.uint64(1)
        return original(selection)

    backend.read = read
    with pytest.raises(ValueError, match="Array changed"):
        finite_statistics(ArrayReader(backend), workspace_bytes=4 << 20)
