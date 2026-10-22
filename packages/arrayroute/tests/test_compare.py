# SPDX-License-Identifier: Apache-2.0
import threading
import time

import numpy as np
import pytest

from arrayroute import ArrayReader, ArraySpec
from arrayroute.compare import ReadPolicy, compare_reads
from arrayroute.errors import UnsupportedReadPolicyError
from arrayroute.settings import ReadSettings


def factory(calls, *, delay=0, offset=0, chunk=4):
    class Backend:
        dtype = np.dtype('f4')
        spec = ArraySpec((4, 32), (1, chunk), 4)

        def __init__(self):
            self.owner = threading.get_ident()

        def read(self, selection):
            assert threading.get_ident() == self.owner
            time.sleep(delay)
            calls.append(selection)
            return (np.arange(128, dtype='f4').reshape(4, 32)+offset)[selection]
    return lambda budget: ArrayReader(Backend(), buffer_bytes=budget)


def compare(reference, policies, **overrides):
    options = dict(worker_counts=[1, 2], total_buffer_bytes=4096, max_output_bytes=1 << 20,
                   repeats=2, time_budget_seconds=30)
    options.update(overrides)
    return compare_reads(reference, policies, [(i, slice(None, None, 2)) for i in range(4)],
                         **options)


def test_policies_share_reference_and_shuffle_globally_with_different_chunk_layouts():
    calls = []
    reference = factory(calls)
    policies = [ReadPolicy('slow', factory(calls, delay=.012)),
                ReadPolicy('fast', factory(calls, chunk=8))]
    report = compare(reference, policies)
    assert report['state'] == 'completed' and report['recommendation']['policy'] == 'fast'
    assert len(report['trials']) == 8 and len(calls) == 36
    assert report['planned_output_bytes'] == 9*256
    assert set(map(tuple, report['trial_order'])) == {
        (r, p, w) for r in [0, 1] for p in ['slow', 'fast'] for w in [1, 2]}
    assert all(t['verified'] and t['requests'] == 4 and t['cpu_seconds'] >= 0
               for t in report['trials'])
    assert all(s['min_seconds'] <= s['median_seconds'] <= s['max_seconds']
               for s in report['summary'])


def test_payload_cap_is_checked_across_all_policies_before_reading():
    calls = []
    source = factory(calls)
    with pytest.raises(ValueError, match='Comparison would read'):
        compare(source, [ReadPolicy('one', source), ReadPolicy('two', source)],
                max_output_bytes=256)
    assert not calls


@pytest.mark.parametrize('error', [ImportError('optional module missing'),
                                  UnsupportedReadPolicyError('unsharded source')])
def test_explicit_unavailable_policy_is_reported_but_not_timed(error):
    calls = []
    source = factory(calls)

    def unavailable(budget):
        raise error

    report = compare(source, [ReadPolicy('missing', unavailable), ReadPolicy('usable', source)])
    assert report['recommendation']['policy'] == 'usable'
    assert report['skipped'] == [{'policy': 'missing', 'workers': None, 'reason': str(error)}]
    assert len(report['trials']) == 4


def test_other_preflight_errors_are_not_hidden():
    source = factory([])

    def invalid(budget):
        raise ValueError('bad metadata')

    with pytest.raises(ValueError, match='bad metadata'):
        compare(source, [ReadPolicy('broken', invalid)])


def test_changed_values_abort_without_a_recommendation():
    with pytest.raises(ValueError, match='differ from the sequential reference'):
        compare(factory([]), [ReadPolicy('wrong', factory([], offset=1))])


def test_timeout_joins_workers_and_has_no_recommendation():
    calls = []
    source = factory(calls, delay=.02)
    report = compare(source, [ReadPolicy('slow', source)], time_budget_seconds=.01)
    assert report['state'] == 'time-budget-reached' and report['recommendation'] is None
    count = len(calls)
    time.sleep(.03)
    assert len(calls) == count


def test_preflight_rejects_wrong_buffer_without_reading():
    calls = []
    source = factory(calls)
    with pytest.raises(ValueError, match='assigned buffer'):
        compare(source, [ReadPolicy('oversized', lambda budget: source(budget*2))])
    assert not calls


def test_unsupported_worker_counts_and_small_budgets_are_skipped():
    source = factory([])
    report = compare(source, [ReadPolicy('a', source)], worker_counts=[1, 2, 8],
                     total_buffer_bytes=128)
    assert report['recommendation']['workers'] == 1
    assert [r['workers'] for r in report['skipped']] == [2, 8]


@pytest.mark.parametrize('overrides', [
    {'repeats': 0}, {'repeats': 101}, {'worker_counts': []}, {'worker_counts': [True]},
    {'worker_counts': [1, 1]}, {'total_buffer_bytes': True}, {'time_budget_seconds': float('nan')},
])
def test_invalid_controls_do_not_read(overrides):
    calls = []
    source = factory(calls)
    with pytest.raises(ValueError):
        compare(source, [ReadPolicy('valid', source)], **overrides)
    assert not calls


@pytest.mark.parametrize('name', ['', 'x y', '../x', 'x'*65, 1])
def test_policy_names_are_bounded_ids(name):
    with pytest.raises(ValueError):
        ReadPolicy(name, factory([]))


@pytest.mark.parametrize('threads', [True, 0, 257, 1.5])
def test_saved_native_thread_limits_are_validated(threads):
    with pytest.raises(ValueError):
        ReadSettings(tensorstore_threads=threads)
