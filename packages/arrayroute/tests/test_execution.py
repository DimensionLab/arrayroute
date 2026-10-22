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
