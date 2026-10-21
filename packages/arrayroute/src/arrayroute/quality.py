# SPDX-License-Identifier: Apache-2.0
"""Independent reconstruction checks; tolerances use the source field's units."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ErrorAssessment:
    absolute_tolerance: float
    max_absolute_error: float
    rmse: float
    reference_rmse: float
    relative_l2: float | None
    finite_values: int
    nonfinite_mismatches: int

    @property
    def passed(self) -> bool:
        return self.nonfinite_mismatches == 0 and self.max_absolute_error <= self.absolute_tolerance


def assess_error(
    reference: np.ndarray, reconstructed: np.ndarray, *, absolute_tolerance: float
) -> ErrorAssessment:
    """Measure a supplied block, without silently enlarging its absolute tolerance.

    Shapes and dtypes must agree. NaN positions and signs of infinity must survive;
    NaN payloads and signed zero are not lossless requirements in this numerical check.
    Measurements use float64 and may allocate several block-sized working arrays.
    Use bounded blocks for large data. relative_l2 is None for an all-zero reference
    with nonzero error, or if the ratio exceeds float64's representable range.
    """
    bound = float(absolute_tolerance)
    if not math.isfinite(bound) or bound < 0:
        raise ValueError("Absolute tolerance must be finite and nonnegative")
    if reference.shape != reconstructed.shape or reference.dtype != reconstructed.dtype:
        raise ValueError("Reconstruction shape and dtype must match the reference")
    if reference.dtype.kind != "f" or reference.dtype.itemsize > 8:
        raise TypeError("Reconstruction assessment requires float16, float32 or float64")
    expected_finite, actual_finite = np.isfinite(reference), np.isfinite(reconstructed)
    comparable = expected_finite & actual_finite
    matching_special = (
        (np.isnan(reference) & np.isnan(reconstructed))
        | (np.isposinf(reference) & np.isposinf(reconstructed))
        | (np.isneginf(reference) & np.isneginf(reconstructed))
    )
    mismatch = int(np.count_nonzero(~(comparable | matching_special)))
    original = reference[comparable].astype(np.float64)
    actual = reconstructed[comparable].astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(actual - original)
    if not np.all(np.isfinite(difference)):
        raise ArithmeticError("Reconstruction error exceeds float64's measurement range")
    largest = float(np.max(difference, initial=0))
    scale = float(np.max(np.abs(original), initial=0))
    rmse = largest * math.sqrt(float(np.mean((difference / largest) ** 2))) if largest else 0.0
    reference_rmse = scale * math.sqrt(float(np.mean((original / scale) ** 2))) if scale else 0.0
    return ErrorAssessment(
        bound,
        largest,
        rmse,
        reference_rmse,
        _relative(rmse, reference_rmse),
        int(comparable.sum()),
        mismatch,
    )


def _relative(rmse: float, reference_rmse: float) -> float | None:
    if not rmse:
        return 0.0
    if not reference_rmse:
        return None
    ratio = rmse / reference_rmse
    return ratio if math.isfinite(ratio) else None


class ErrorAccumulator:
    """Accumulate independent, non-overlapping blocks without retaining their arrays."""

    def __init__(self, *, absolute_tolerance: float):
        bound = float(absolute_tolerance)
        if not math.isfinite(bound) or bound < 0:
            raise ValueError("Absolute tolerance must be finite and nonnegative")
        self.result = ErrorAssessment(bound, 0, 0, 0, 0, 0, 0)

    def update(self, reference: np.ndarray, reconstructed: np.ndarray) -> ErrorAssessment:
        block = assess_error(
            reference, reconstructed, absolute_tolerance=self.result.absolute_tolerance
        )
        self.merge(block)
        return block

    def merge(self, block: ErrorAssessment) -> None:
        previous = self.result
        if block.absolute_tolerance != previous.absolute_tolerance:
            raise ValueError("Cannot merge assessments with different tolerances")
        count = previous.finite_values + block.finite_values
        left = math.sqrt(previous.finite_values / count) if count else 0
        right = math.sqrt(block.finite_values / count) if count else 0
        rmse = math.hypot(previous.rmse * left, block.rmse * right)
        baseline = math.hypot(previous.reference_rmse * left, block.reference_rmse * right)
        self.result = ErrorAssessment(
            previous.absolute_tolerance,
            max(previous.max_absolute_error, block.max_absolute_error),
            rmse,
            baseline,
            _relative(rmse, baseline),
            count,
            previous.nonfinite_mismatches + block.nonfinite_mismatches,
        )
