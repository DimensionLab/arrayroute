# SPDX-License-Identifier: Apache-2.0
import threading

import pytest

import arrayroute.execution as execution
from arrayroute.execution import OrderedPool


def test_trace_preserves_delivery_when_completion_is_out_of_order():
    second = threading.Event()
    events = []

    def process(value):
        if value == 0:
            assert second.wait(5)
        return value

    def observe(event):
        events.append(event)
        if event.phase == 'finished' and event.task_index == 1:
            second.set()

    with OrderedPool(lambda: process, workers=2) as pool:
        with pool.map(range(8), observer=observe) as values:
            assert list(values) == list(range(8))
    assert [e.task_index for e in events if e.phase == 'delivered'] == list(range(8))
    assert [e.task_index for e in events if e.phase == 'finished'][0] == 1
    for index in range(8):
        task = [e for e in events if e.task_index == index]
        assert [e.phase for e in task] == [
            'admitted', 'started', 'finished', 'delivered', 'released',
        ]
        assert [e.timestamp_ns for e in task] == sorted(e.timestamp_ns for e in task)
        assert not any(e.failed for e in task)
        assert task[1].thread_id == task[2].thread_id != threading.get_ident()


def test_no_observer_does_not_allocate_events_or_read_clock(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('Disabled tracing must not sample clocks or allocate events')

    monkeypatch.setattr(execution.time, 'perf_counter_ns', unexpected)
    monkeypatch.setattr(execution, 'PoolEvent', unexpected)
    with OrderedPool(lambda: lambda value: value, workers=2) as pool:
        with pool.map(range(6)) as values:
            assert list(values) == list(range(6))


def test_failed_transform_is_reported_without_delivering_bad_result():
    events = []

    def fail(_):
        raise OSError('corrupt payload')

    with OrderedPool(lambda: fail, workers=1) as pool:
        with pool.map([1], observer=events.append) as values:
            with pytest.raises(OSError, match='corrupt payload'):
                next(values)
    assert [e.phase for e in events] == ['admitted', 'started', 'finished']
    assert events[-1].failed


@pytest.mark.parametrize('phase', ['admitted', 'started', 'finished', 'delivered', 'released'])
def test_observer_failure_fails_pass_and_pool_can_be_reused(phase):
    def observe(event):
        if event.phase == phase:
            raise ValueError('observer failed')

    with OrderedPool(lambda: lambda value: value, workers=2) as pool:
        with pytest.raises(ValueError, match='observer failed'), pool.map(
            range(4), observer=observe,
        ) as values:
            list(values)
        with pool.map([8]) as values:
            assert list(values) == [8]


def test_early_consumer_exit_releases_delivered_result_and_indices_restart():
    with OrderedPool(lambda: lambda value: value, workers=1) as pool:
        for _ in range(2):
            events = []
            with pool.map(range(4), observer=events.append) as values:
                assert next(values) == 0
            assert [(e.phase, e.task_index) for e in events][-1] == ('released', 0)
