# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from arrayroute.difference import BLOCK_ELEMENTS, _regions, compare_array_readers
from arrayroute.plan import ArraySpec
from arrayroute.reader import ArrayReader


class Backend:
    def __init__(self, values):
        self.values = values
        self.dtype = values.dtype
        self.spec = ArraySpec(values.shape, tuple(max(1, min(s, 16)) for s in values.shape),
                              values.dtype.itemsize)
        self.reads = []

    def read(self, selection):
        self.reads.append(selection)
        return self.values[selection].copy()


def reader(values, budget=2 << 20):
    return ArrayReader(Backend(np.asarray(values)), buffer_bytes=budget)


@pytest.mark.parametrize('shape', [(1,), (19, 97, 53), (3, 5, 7, 11)])
def test_flat_regions_preserve_exact_c_order(shape):
    values = np.arange(np.prod(shape)).reshape(shape)
    for start, stop in [(0, values.size), (1, values.size - 1), (3, 9)]:
        if stop < start or stop > values.size:
            continue
        pieces = [values[s].reshape(-1) for s in _regions(shape, start, stop)]
        actual = np.concatenate(pieces) if pieces else np.array([], dtype=values.dtype)
        np.testing.assert_array_equal(actual, values.reshape(-1)[start:stop])


def test_full_counts_detect_change_outside_percentile_sample():
    a = np.arange(2*BLOCK_ELEMENTS + 17, dtype=np.float32)
    b = a.copy()
    b[137] += 1000
    before, after = reader(a), reader(b)
    result = compare_array_readers(before, after, percentile_samples=2)
    assert result['n_changed'] == 1
    assert result['n_bitwise_changed'] == 1
    assert result['frac_changed'] == 1/len(a)
    assert result['max_abs_diff'] == 1000
    assert result['rel_l2_diff'] == pytest.approx(1000/np.linalg.norm(a.astype(np.float64)))
    assert result['before']['mean'] == pytest.approx(a.astype(np.float64).mean())
    assert result['after']['std'] == pytest.approx(b.astype(np.float64).std())
    assert result['before']['percentile_scope'] == 'deterministic_sample'
    assert len(before.backend.reads) >= 1
    assert all(np.asarray(a[s]).size <= result['read_batch_elements']
               for s in before.backend.reads)


def test_deterministic_metrics_across_read_budgets_and_same_reader():
    rng = np.random.default_rng(19)
    a = rng.normal(size=(117, 97, 53)).astype(np.float32)
    b = a + np.float32(.01)
    results = []
    for budget in (2 << 20, 4 << 20):
        result = compare_array_readers(reader(a, budget), reader(b, budget), percentile_samples=101)
        results.append({k: v for k, v in result.items() if not k.startswith('read_')})
    assert results[0] == results[1]
    r = reader(a)
    same = compare_array_readers(r, r)
    assert same['n_changed'] == same['n_bitwise_changed'] == 0
    assert same['before'] == same['after']
    assert same['before']['finite_count'] == a.size


def test_nonfinite_and_bitwise_changes_are_distinct():
    a = np.array([0., np.nan, np.inf, -np.inf, 3], dtype=np.float64)
    b = a.copy()
    b[0] = -0.
    b.view(np.uint64)[1] ^= np.uint64(1)
    b[2] = 7
    result = compare_array_readers(reader(a), reader(b))
    assert result['n_changed'] == 1
    assert result['n_bitwise_changed'] == 3
    assert result['max_abs_diff'] == 0
    assert result['before']['nan_frac'] == .2
    assert result['before']['nonfinite_frac'] == .6


def test_int64_counts_do_not_lose_precision():
    a = np.array([2**63-2, 2**63-1], dtype=np.int64)
    b = a.copy()
    b[0] += 1
    result = compare_array_readers(reader(a), reader(b))
    assert result['n_changed'] == 1
    assert result['before']['min'] == 2**63-2
    assert 'float64' in result['metric_precision']


@pytest.mark.parametrize('shape', [(0, 3), (1,)])
def test_empty_added_removed_and_changed_shape(shape):
    r = reader(np.zeros(shape, dtype=np.float32))
    assert compare_array_readers(None, r)['note'] == 'added'
    assert compare_array_readers(r, None)['note'] == 'removed'
    assert compare_array_readers(r, r)['n_changed'] == 0
    assert compare_array_readers(r, reader(np.zeros((2, 3))))['note'] == 'shape changed'
    if shape == (1,):
        changed = compare_array_readers(r, reader(np.zeros(shape, np.int64)))
        assert changed['note'] == 'dtype changed'


def test_invalid_budgets_cancellation_and_corruption_fail_before_or_during_io():
    r = reader(np.arange(100, dtype=np.float64))
    with pytest.raises(MemoryError, match='workspace'):
        compare_array_readers(r, r, max_sample_bytes=1)
    with pytest.raises(ValueError):
        compare_array_readers(r, r, percentile_samples=0)
    with pytest.raises(InterruptedError):
        compare_array_readers(r, r, cancelled=lambda: True)
    assert r.backend.reads == []
    with pytest.raises(TypeError):
        compare_array_readers(reader(np.array(['abc'])), r)
    def broken(_selection):
        raise OSError('corrupt chunk')
    r.backend.read = broken
    with pytest.raises(OSError, match='corrupt chunk'):
        compare_array_readers(r, r)


def test_bool_and_all_nonfinite_results_are_json_safe():
    import json

    a = reader(np.array([True, False, True]))
    b = reader(np.array([False, False, True]))
    assert compare_array_readers(a, b)['n_changed'] == 1
    r = reader(np.array([np.nan, np.inf, -np.inf]))
    result = compare_array_readers(r, r)
    assert result['before']['finite_count'] == 0
    assert 'p50' not in result['before']
    json.dumps(result, allow_nan=False)


def test_cancellation_during_comparison_stops_without_partial_result():
    r = reader(np.ones(BLOCK_ELEMENTS*20, dtype=np.float32))
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 4
    with pytest.raises(InterruptedError):
        compare_array_readers(r, r, cancelled=cancelled)
    assert len(r.backend.reads) < 20
