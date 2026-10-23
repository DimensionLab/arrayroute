# SPDX-License-Identifier: Apache-2.0
import threading
import time

import numpy as np
import pytest

from arrayroute import ArrayReader, ArraySpec
from arrayroute.probe import probe_reads


class Backend:
    def __init__(self, values, calls):
        self.values = values
        self.spec = ArraySpec(values.shape, (1, 4), values.dtype.itemsize)
        self.dtype = values.dtype
        self.calls = calls
        self.owner = threading.get_ident()

    def read(self, selection):
        assert self.owner == threading.get_ident()
        self.calls.append(selection)
        return self.values[selection]


def setup():
    values = np.arange(128, dtype=np.float32).reshape(4, 32)[:, ::2]
    calls = []
    def factory(budget):
        return ArrayReader(Backend(values, calls), buffer_bytes=budget)

    options = dict(worker_counts=[1, 2, 4], total_buffer_bytes=2048,
                   max_output_bytes=4096, repeats=2, time_budget_seconds=30)
    selections = [(i, slice(None)) for i in range(4)]
    return factory, selections, options, calls


def test_shuffled_pilot_checks_every_result_and_accounts_all_passes():
    factory, selections, options, calls = setup()
    report = probe_reads(factory, selections, **options)
    assert report['state'] == 'completed'
    assert len(report['trials']) == 6 and len(report['reference_sha256']) == 4
    assert report['planned_output_bytes'] == 7 * 4 * 16 * 4
    assert report['fastest_observed_workers'] in [1, 2, 4]
    assert set(map(tuple, report['trial_order'])) == {(r, w) for r in range(2) for w in [1, 2, 4]}
    assert len(calls) == 28
    assert all(t['verified'] and t['requests'] == 4 and t['output_bytes'] == 256
               for t in report['trials'])
    assert all(t['planned_chunk_visits'] == 16 and t['decoded_bytes_upper_bound'] == 256
               and t['decode_amplification_upper_bound'] == 1 for t in report['trials'])


def test_preflight_rejects_payload_cost_before_reading():
    factory, selections, options, calls = setup()
    options['max_output_bytes'] = 1
    with pytest.raises(ValueError, match='Pilot would read'):
        probe_reads(factory, selections, **options)
    assert calls == []


def test_too_many_workers_are_reported_without_being_run():
    factory, selections, options, calls = setup()
    options['worker_counts'] = [1, 8]
    report = probe_reads(factory, selections, **options)
    assert report['skipped'] == [{'workers': 8, 'reason': 'More workers than selected regions'}]
    assert report['fastest_observed_workers'] == 1 and len(calls) == 12


def test_changed_values_are_rejected_instead_of_ranked():
    factory, selections, options, _ = setup()
    generation = 0

    def changed(budget):
        nonlocal generation
        reader = factory(budget)
        generation += 1
        if generation > 2:  # Metadata and reference readers precede timed trials.
            reader.backend.values = reader.backend.values + 1
        return reader

    with pytest.raises(ValueError, match='changed from the sequential reference'):
        probe_reads(changed, selections, **options)


def test_deadline_returns_partial_without_recommendation_and_joins_reader():
    factory, selections, options, calls = setup()
    options['time_budget_seconds'] = 0.01

    def slow(budget):
        reader = factory(budget)
        original = reader.backend.read

        def read(selection):
            time.sleep(0.02)
            return original(selection)

        reader.backend.read = read
        return reader

    report = probe_reads(slow, selections, **options)
    assert report['state'] == 'time-budget-reached'
    assert report['fastest_observed_workers'] is None
    count = len(calls)
    time.sleep(0.03)
    assert len(calls) == count <= 1


def test_factory_must_honor_memory_and_identity():
    factory, selections, options, calls = setup()

    def oversized(budget):
        return factory(budget * 2)

    with pytest.raises(ValueError, match='assigned buffer budget'):
        probe_reads(oversized, selections, **options)
    assert not calls


def test_insufficient_per_worker_buffers_are_skipped():
    factory, selections, options, calls = setup()
    options['total_buffer_bytes'] = 128
    result = probe_reads(factory, selections, **options)
    assert result['fastest_observed_workers'] == 1
    assert [item['workers'] for item in result['skipped']] == [2, 4]
    assert len(calls) == 12


def test_hashing_noncontiguous_output_uses_bounded_scratch(monkeypatch):
    import arrayroute.probe as module

    original = module.hashlib.sha256
    sizes = []

    class Digest:
        def __init__(self):
            self.inner = original()

        def update(self, block):
            sizes.append(len(block))
            self.inner.update(block)

        def hexdigest(self):
            return self.inner.hexdigest()

    monkeypatch.setattr(module.hashlib, 'sha256', Digest)
    values = np.arange(262144, dtype=np.float32).reshape(1, -1)[:, ::2]

    def factory(budget):
        return ArrayReader(Backend(values, []), buffer_bytes=budget)

    result = probe_reads(factory, [(0, slice(None))], worker_counts=[1],
                         total_buffer_bytes=2 << 20, max_output_bytes=2 << 20,
                         time_budget_seconds=30, repeats=1)
    assert result['state'] == 'completed'
    assert result['reference_sha256'] == [original(values[0].tobytes()).hexdigest()]
    assert max(sizes) <= 32768 and sum(sizes) == values.nbytes * 2


@pytest.mark.parametrize('key,value', [
    ('repeats', 0), ('repeats', 101), ('total_buffer_bytes', True),
    ('max_output_bytes', -1), ('worker_counts', []), ('worker_counts', [1, 1]),
    ('worker_counts', [257]), ('time_budget_seconds', float('inf')),
    ('time_budget_seconds', False),
])
def test_invalid_controls(key, value):
    factory, selections, options, calls = setup()
    options[key] = value
    with pytest.raises(ValueError):
        probe_reads(factory, selections, **options)
    assert not calls
