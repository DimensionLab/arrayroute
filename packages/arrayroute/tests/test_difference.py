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


@pytest.mark.parametrize('selection', [
    (slice(2, 23, 3), slice(1, 97, 2), slice(5, 53, 3)),
    (-1, slice(None, None, 3), slice(2, 9)),
    (3, -2, 7),
    (slice(2, 2), slice(None), slice(None)),
    (slice(None), slice(None), slice(None)),
])
def test_selected_comparison_matches_materialized_reference(selection):
    a = np.arange(23*97*53, dtype=np.int64).reshape(23, 97, 53)
    b = a.copy()
    b[::2, 1::3, ::4] += 100
    ra, rb = reader(a), reader(b)
    result = compare_array_readers(ra, rb, before_selection=selection,
                                   after_selection=selection, percentile_samples=19)
    aa, bb = np.asarray(a[selection]), np.asarray(b[selection])
    assert result['comparison_scope'] == 'full_selection'
    assert result['n_before'] == aa.size
    assert result['shape_after'] == list(bb.shape)
    assert result['source_shape_after'] == list(b.shape)
    assert result['n_changed'] == int((aa != bb).sum())
    assert result['max_abs_diff'] == (float(np.max(np.abs(bb-aa))) if aa.size else 0)
    assert result['before']['statistics_scope'] == 'full_selection'
    if aa.size:
        assert result['before']['mean'] == pytest.approx(aa.mean())
    else:
        assert not ra.backend.reads and not rb.backend.reads


def test_regions_can_have_different_origins_and_source_shapes():
    a = np.arange(2*BLOCK_ELEMENTS + 17, dtype=np.float32)
    b = np.pad(a, (7, 9), constant_values=-999)
    result = compare_array_readers(reader(a), reader(b),
                                   after_selection=(slice(7, -9),))
    assert result['n_changed'] == 0
    assert result['shape_before'] == result['shape_after']
    assert result['source_shape_before'] != result['source_shape_after']
    assert result['before']['statistics_scope'] == 'full_array'
    assert result['after']['statistics_scope'] == 'full_selection'


@pytest.mark.parametrize('selection', [(slice(None, None, -1),), (True,), (999,), ()])
def test_invalid_region_fails_before_payload_reads(selection):
    r = reader(np.ones(5))
    with pytest.raises((ValueError, IndexError, TypeError)):
        compare_array_readers(r, r, before_selection=selection)
    assert not r.backend.reads


def test_selection_without_source_is_rejected():
    with pytest.raises(ValueError, match='selection requires'):
        compare_array_readers(None, reader(np.ones(5)), before_selection=(slice(None),))
