# SPDX-License-Identifier: Apache-2.0
import threading
from itertools import count

import pytest

from arrayroute.execution import ProcessingCancelledError, ordered_map


def test_order_capacity_and_worker_ownership():
    consumed = []
    owners = []
    completed = []
    second_done = threading.Event()

    def source():
        for item in count():
            consumed.append(item)
            yield item

    def factory():
        owner = threading.get_ident()
        owners.append(owner)

        def process(item):
            assert threading.get_ident() == owner
            if item == 0:
                assert second_done.wait(5)
            completed.append(item)
            if item == 1:
                second_done.set()
            return item * 2

        return process

    with ordered_map(factory, source(), workers=2, max_pending=3) as values:
        for i in range(12):
            assert next(values) == i * 2
            assert len(consumed) == i + 3
    assert completed[0] != 0
    assert len(owners) == len(set(owners)) == 2
    assert len(consumed) == 14


def test_read_or_factory_failure_never_skips_a_position():
    def factory():
        def process(i):
            if i == 2:
                raise OSError("corrupt frame")
            return i
        return process

    with ordered_map(factory, range(20), workers=4) as values:
        assert next(values) == 0
        assert next(values) == 1
        with pytest.raises(OSError, match="corrupt frame"):
            next(values)

    def broken_factory():
        raise OSError("cannot open reader")

    with ordered_map(broken_factory, range(5), workers=2) as values:
        with pytest.raises(OSError, match="cannot open reader"):
            next(values)


def test_close_joins_in_flight_work():
    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    owned = []

    def factory():
        owned.append(threading.current_thread())

        def process(i):
            if i == 0:
                assert started.wait(5)
            else:
                started.set()
                assert release.wait(5)
            return i
        return process

    def consume():
        with ordered_map(factory, range(100), workers=2) as values:
            assert next(values) == 0
        closed.set()

    thread = threading.Thread(target=consume)
    thread.start()
    try:
        assert started.wait(5)
        assert not closed.wait(.05)
    finally:
        release.set()
        thread.join(5)
    assert closed.is_set()
    assert all(not t.is_alive() for t in owned)


def test_cancellation_before_submission_does_not_consume_input():
    def items():
        pytest.fail("Cancelled source must not be advanced")
        yield 1

    with ordered_map(lambda: lambda x: x, items(), workers=2, cancelled=lambda: True) as values:
        with pytest.raises(ProcessingCancelledError):
            next(values)


def test_source_failure_and_empty_source():
    def source():
        yield 0
        raise ValueError("source failed")

    with ordered_map(lambda: lambda x: x, source(), workers=2) as values:
        with pytest.raises(ValueError, match="source failed"):
            list(values)
    with ordered_map(lambda: pytest.fail("Empty input needs no worker"), [], workers=2) as values:
        assert list(values) == []


@pytest.mark.parametrize("workers,pending", [(0, None), (-1, 2), (True, 2), (1, 0), (1, True)])
def test_invalid_capacity_is_rejected(workers, pending):
    with pytest.raises(ValueError):
        with ordered_map(lambda: lambda x: x, [], workers=workers, max_pending=pending):
            pass


def test_pool_reuses_thread_owned_workers_and_caches_across_passes():
    from arrayroute.execution import OrderedPool

    instances = []
    owners = []
    rendezvous = threading.Barrier(3)

    def factory():
        owner = threading.get_ident()
        owners.append(threading.current_thread())
        seen = []
        instances.append(seen)

        def process(item):
            assert owner == threading.get_ident()
            rendezvous.wait(timeout=5)
            seen.append(item)
            return item
        return process

    with OrderedPool(factory, workers=3) as pool:
        for epoch in range(8):
            with pool.map([(epoch, i) for i in range(3)]) as values:
                assert list(values) == [(epoch, i) for i in range(3)]
        assert len(instances) == 3
        assert all(len(seen) == 8 for seen in instances)
    assert all(not thread.is_alive() for thread in owners)
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        with pool.map([0]):
            pass


def test_pool_pass_is_exclusive_even_before_consumption_and_after_failure():
    from arrayroute.execution import OrderedPool

    def factory():
        def process(item):
            if item == 1:
                raise OSError("broken input")
            return item
        return process

    with OrderedPool(factory, workers=2) as pool:
        with pool.map(range(5)) as values:
            with pytest.raises(RuntimeError, match="Concurrent"):
                with pool.map([]):
                    pass
            assert next(values) == 0
            with pytest.raises(OSError, match="broken input"):
                next(values)
        with pool.map([7, 8]) as values:
            assert list(values) == [7, 8]


def test_pool_early_pass_exit_joins_before_starting_another_pass():
    from arrayroute.execution import OrderedPool

    started, release, exited = threading.Event(), threading.Event(), threading.Event()

    def factory():
        def process(item):
            if item == 0:
                assert started.wait(5)
            if item == 1:
                started.set()
                assert release.wait(5)
            return item
        return process

    with OrderedPool(factory, workers=2) as pool:
        def consume():
            with pool.map(range(20)) as values:
                assert next(values) == 0
            exited.set()

        consumer = threading.Thread(target=consume)
        consumer.start()
        try:
            assert started.wait(5)
            assert not exited.wait(.05)
            with pytest.raises(RuntimeError, match="Concurrent"):
                with pool.map([]):
                    pass
        finally:
            release.set()
            consumer.join(5)
        assert exited.is_set()
        with pool.map([99]) as values:
            assert list(values) == [99]


def test_closed_pool_rejects_an_active_iterator_without_submitting_more():
    from arrayroute.execution import OrderedPool

    consumed = []

    def source():
        for item in range(100):
            consumed.append(item)
            yield item

    pool = OrderedPool(lambda: lambda i: i, workers=2)
    with pool.map(source()) as values:
        assert next(values) == 0
        pool.close()
        with pytest.raises(RuntimeError, match="closed"):
            next(values)
    assert consumed == [0, 1]


def test_close_with_queued_tasks_does_not_strand_cleanup_waiters():
    from arrayroute.execution import OrderedPool

    blocked, release, finished = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def factory():
        def process(item):
            if item == 1:
                blocked.set()
                assert release.wait(5)
            return item
        return process

    pool = OrderedPool(factory, workers=1, max_pending=8)

    def consume():
        try:
            with pool.map(range(100)) as values:
                assert next(values) == 0
                assert blocked.wait(5)
                closer = threading.Thread(target=pool.close)
                closer.start()
                release.set()
                closer.join(5)
                assert not closer.is_alive()
            finished.set()
        except BaseException as error:
            errors.append(error)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    consumer.join(10)
    release.set()
    assert not errors, errors
    assert finished.is_set(), "Cleanup stranded a cancelled future after pool shutdown"
