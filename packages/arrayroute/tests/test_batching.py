# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import pytest

from arrayroute.batching import plan_row_batch


def test_overlap_threshold_keeps_original_first_occurrence_order():
    plan = plan_row_batch([[5, 6, 7], [7, 5, 8]], row_count=10, min_reuse_fraction=1/3)
    assert plan.combine
    assert plan.unique_rows == (5, 6, 7, 8)
    assert plan.requested_rows == 6
    assert plan.reuse_fraction == pytest.approx(1/3)
    assert not plan_row_batch([[5, 6, 7], [7, 5, 8]], row_count=10,
                             min_reuse_fraction=.34).combine


def test_negative_indices_are_the_same_physical_rows():
    plan = plan_row_batch([[-1, 0], [4, -5]], row_count=5)
    assert plan.combine and plan.unique_rows == (4, 0) and plan.reuse_fraction == .5


def test_sparse_petabyte_scale_domain_does_not_expand_indices():
    plan = plan_row_batch([[2**48-1], [2**48-1]], row_count=2**48)
    assert plan.combine and plan.unique_rows == (2**48-1,)


@pytest.mark.parametrize('selections', [[], [[]], [[], []], [[0, 1]], [[0], [1]]])
def test_zero_threshold_still_requires_reuse_between_selections(selections):
    assert not plan_row_batch(selections, row_count=10, min_reuse_fraction=0).combine


@pytest.mark.parametrize('threshold', [True, np.bool_(False), math.nan, math.inf, -.1, 1.01])
def test_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        plan_row_batch([[0], [0]], row_count=1, min_reuse_fraction=threshold)


@pytest.mark.parametrize('value', [True, np.bool_(True), .5])
def test_rejects_noninteger_or_boolean_rows(value):
    with pytest.raises(TypeError):
        plan_row_batch([[value]], row_count=2)


@pytest.mark.parametrize('value', [-3, 2])
def test_rejects_out_of_bounds_rows(value):
    with pytest.raises(IndexError):
        plan_row_batch([[value]], row_count=2)


def test_threshold_one_selects_separate_reads():
    assert not plan_row_batch([[0]*100, [0]*100], row_count=1,
                             min_reuse_fraction=1).combine
