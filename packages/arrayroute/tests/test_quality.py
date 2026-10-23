# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.quality import ErrorAccumulator, assess_error


def test_tolerance_is_not_silently_relaxed():
    original = np.zeros(3, dtype=np.float32)
    actual = np.array([0, 0.5, 1.005], dtype=np.float32)
    assessment = assess_error(original, actual, absolute_tolerance=1)
    assert not assessment.passed
    assert assessment.max_absolute_error > 1
    assert assessment.relative_l2 is None
    assert assessment.finite_values == 3
    assert assessment.rmse == pytest.approx(np.sqrt(np.mean(actual.astype(np.float64) ** 2)))


def test_special_values_and_shape_cannot_hide_corruption():
    values = np.array([np.nan, np.inf, -np.inf, 0, 1], dtype=np.float32)
    assert assess_error(values, values.copy(), absolute_tolerance=0).passed
    changed = np.array([0, -np.inf, np.nan, np.nan, 1], dtype=np.float32)
    assessment = assess_error(values, changed, absolute_tolerance=1e10)
    assert not assessment.passed and assessment.nonfinite_mismatches == 4
    with pytest.raises(ValueError, match="shape and dtype"):
        assess_error(values, values.astype(np.float64), absolute_tolerance=1)
    with pytest.raises(ValueError, match="shape and dtype"):
        assess_error(values, values.reshape(1, -1), absolute_tolerance=1)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_norms_do_not_square_unscaled_large_or_small_values(scale):
    values = np.array([1, 2, 3], dtype=np.float64) * scale
    actual = values * 1.5
    result = assess_error(values, actual, absolute_tolerance=2 * scale)
    assert result.passed
    assert result.relative_l2 == pytest.approx(0.5)
    assert result.rmse / scale == pytest.approx(np.sqrt(14 / 3) / 2)


def test_invalid_tolerances_and_overflow_fail_closed():
    values = np.ones(3, dtype=np.float32)
    for invalid in [float("nan"), float("inf"), -1]:
        with pytest.raises(ValueError):
            assess_error(values, values, absolute_tolerance=invalid)
    with pytest.raises(TypeError):
        assess_error(values.astype(int), values.astype(int), absolute_tolerance=1)
    huge = np.array([1e308])
    with pytest.raises(ArithmeticError):
        assess_error(huge, -huge, absolute_tolerance=1)
    empty = assess_error(np.array([]), np.array([]), absolute_tolerance=0)
    assert empty.passed and empty.finite_values == 0


def test_streaming_and_distributed_error_reduction_matches_whole_array():
    source = np.linspace(-3, 8, 201, dtype=np.float32)
    source[2] = np.nan
    actual = source + np.sin(source) * 0.03
    actual[3] = np.inf
    direct = assess_error(source, actual, absolute_tolerance=0.02)
    left, right = (ErrorAccumulator(absolute_tolerance=0.02) for _ in range(2))
    for start in range(0, 201, 17):
        worker = left if start % 2 else right
        worker.update(source[start : start + 17], actual[start : start + 17])
    left.merge(right.result)
    merged = left.result
    assert merged.max_absolute_error == direct.max_absolute_error
    assert merged.rmse == pytest.approx(direct.rmse)
    assert merged.reference_rmse == pytest.approx(direct.reference_rmse)
    assert merged.relative_l2 == pytest.approx(direct.relative_l2)
    assert merged.finite_values == direct.finite_values
    assert merged.nonfinite_mismatches == direct.nonfinite_mismatches == 1
    assert not merged.passed
    with pytest.raises(ValueError, match="different tolerances"):
        left.merge(ErrorAccumulator(absolute_tolerance=1).result)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, ">f4", ">f8"])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "reversed", "scalar"])
def test_finite_path_matches_explicit_comparable_selection(dtype, layout):
    source = np.random.default_rng(172).normal(size=(13, 7)).astype(dtype)
    actual = (source.astype(np.float64) * 1.1).astype(dtype)
    if layout == "fortran":
        source, actual = np.asfortranarray(source), np.asfortranarray(actual)
    elif layout == "strided":
        source, actual = source[::2, ::2], actual[::2, ::2]
    elif layout == "reversed":
        source, actual = source[::-1, ::-1], actual[::-1, ::-1]
    elif layout == "scalar":
        source, actual = source[0, 0:1].reshape(()), actual[0, 0:1].reshape(())
    before = source.tobytes(), actual.tobytes()
    source.setflags(write=False)
    actual.setflags(write=False)
    result = assess_error(source, actual, absolute_tolerance=0.2)
    # A matching NaN forces exceptional-value handling without adding a finite
    # reduction term. Its finite selection must equal the fast path bit for bit.
    expected = assess_error(
        np.append(source.reshape(-1), np.nan).astype(dtype),
        np.append(actual.reshape(-1), np.nan).astype(dtype),
        absolute_tolerance=0.2,
    )
    assert result == expected
    assert (source.tobytes(), actual.tobytes()) == before
