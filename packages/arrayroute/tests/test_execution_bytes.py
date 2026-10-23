# SPDX-License-Identifier: Apache-2.0
import threading

import pytest

from arrayroute.execution import OrderedPool, ProcessingCancelledError, ordered_map


def test_bytes_limit_submission_without_reordering_or_reestimating():
    started, estimated = [], []
    second = threading.Event()
    weights = [3, 3, 5, 1, 0]

    def estimate(i):
        estimated.append(i)
        return weights[i]

    def factory():
        def process(i):
            started.append(i)
            if i == 0:
                assert second.wait(5)
            if i == 1:
                second.set()
            return i
        return process

    with ordered_map(factory, range(5), workers=3, max_pending=8,
                     max_pending_bytes=6, estimate_bytes=estimate) as values:
        assert next(values) == 0
        assert set(started) == {0, 1}
        assert estimated == [0, 1, 2]
        assert next(values) == 1
        assert set(started) == {0, 1}
        assert list(values) == [2, 3, 4]
    assert estimated == list(range(5))


def test_one_full_budget_task_progresses_and_pool_can_be_reused():
    with OrderedPool(lambda: lambda x: x, workers=8, max_pending=16,
                     max_pending_bytes=10, estimate_bytes=lambda x: 10) as pool:
        for _ in range(3):
            with pool.map(range(100)) as values:
                assert list(values) == list(range(100))


def test_oversized_task_is_rejected_before_worker_creation():
    with ordered_map(lambda: pytest.fail('No worker should be created'), [11], workers=2,
                     max_pending_bytes=10, estimate_bytes=lambda x: x) as values:
        with pytest.raises(MemoryError, match='11 planned bytes.*10'):
            next(values)


@pytest.mark.parametrize('budget,estimate', [(None, lambda x: 1), (1, None),
                                            (0, lambda x: 1), (True, lambda x: 1)])
def test_invalid_byte_admission(budget, estimate):
    with pytest.raises(ValueError):
        OrderedPool(lambda: lambda x: x, workers=1,
                    max_pending_bytes=budget, estimate_bytes=estimate)


@pytest.mark.parametrize('estimate', [-1, True, 1.5])
def test_invalid_task_estimate_does_not_execute(estimate):
    with ordered_map(lambda: pytest.fail('Invalid estimate must not execute'), [0], workers=1,
                     max_pending_bytes=10, estimate_bytes=lambda x: estimate) as values:
        with pytest.raises((ValueError, TypeError)):
            next(values)


def test_cancelled_deferred_task_is_not_submitted_and_next_pass_works():
    cancelled = False
    started = []
    with OrderedPool(lambda: lambda x: started.append(x) or x, workers=2,
                     max_pending_bytes=1, estimate_bytes=lambda x: 1) as pool:
        with pool.map(range(5), cancelled=lambda: cancelled) as values:
            assert next(values) == 0
            cancelled = True
            with pytest.raises(ProcessingCancelledError):
                next(values)
        assert started == [0]
        with pool.map([9]) as values:
            assert list(values) == [9]


def test_failed_task_retains_order_and_releases_pass():
    def process(i):
        if i == 1:
            raise OSError('decoder failed')
        return i
    with OrderedPool(lambda: process, workers=3, max_pending_bytes=5,
                     estimate_bytes=lambda x: 3) as pool:
        with pool.map(range(4)) as values:
            assert next(values) == 0
            with pytest.raises(OSError, match='decoder failed'):
                next(values)
        with pool.map([8]) as values:
            assert list(values) == [8]
