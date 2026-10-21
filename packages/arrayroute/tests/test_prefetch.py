# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident

import numpy as np
import pytest

from arrayroute import ArrayReader, ArraySpec, ReadStats


class Backend:
    def __init__(self):
        self.values = np.arange(64, dtype=np.float32)
        self.dtype = self.values.dtype
        self.spec = ArraySpec(self.values.shape, (4,), 4)
        self.calls = []
        self.second = Event()

    def read(self, selection):
        self.calls.append((selection, get_ident()))
        if len(self.calls) == 2:
            self.second.set()
        return self.values[selection].copy()


def test_order_bytes_stats_and_two_slot_planning():
    backend = Backend()
    reader = ArrayReader(backend, buffer_bytes=64)
    result = np.empty_like(backend.values)
    stats = ReadStats()
    with reader.prefetch((slice(None),), stats=stats) as blocks:
        for tile, value in blocks:
            assert tile.plan.buffer_bytes_upper_bound <= 32
            result[tile.destination] = value
    np.testing.assert_array_equal(result, backend.values)
    assert stats.requests == 16
    assert stats.output_bytes == backend.values.nbytes
    assert len({ident for _, ident in backend.calls}) == 1
    assert backend.calls[0][1] != get_ident()


def test_only_one_read_ahead_and_early_exit_cleans_up():
    backend = Backend()
    with ArrayReader(backend, buffer_bytes=64).prefetch((slice(None),)) as blocks:
        _, first = next(blocks)
        assert backend.second.wait(3)
        # The second read may finish; a third cannot be scheduled until consumption.
        assert len(backend.calls) == 2
        np.testing.assert_array_equal(first, backend.values[:4])
    assert len(backend.calls) == 2
    assert list(blocks) == []


def test_no_read_when_context_not_consumed():
    backend = Backend()
    with ArrayReader(backend, buffer_bytes=64).prefetch((slice(None),)):
        pass
    assert backend.calls == []


def test_cancellation_and_consumer_error_stop_scheduling():
    backend = Backend()
    cancelled = Event()
    with ArrayReader(backend, buffer_bytes=64).prefetch(
        (slice(None),), cancelled=cancelled.is_set
    ) as blocks:
        next(blocks)
        cancelled.set()
        with pytest.raises(InterruptedError):
            next(blocks)
    assert len(backend.calls) <= 2
    backend = Backend()
    with pytest.raises(RuntimeError, match='consumer'):
        with ArrayReader(backend, buffer_bytes=64).prefetch((slice(None),)) as blocks:
            next(blocks)
            raise RuntimeError('consumer')
    assert len(backend.calls) <= 2


def test_backend_error_is_not_lost():
    class Broken(Backend):
        def read(self, selection):
            if self.calls:
                raise OSError('storage unavailable')
            return super().read(selection)

    with ArrayReader(Broken(), buffer_bytes=64).prefetch((slice(None),)) as blocks:
        next(blocks)
        with pytest.raises(OSError, match='storage unavailable'):
            next(blocks)


def test_close_waits_for_native_read_before_releasing_backend():
    entered, release, closed = Event(), Event(), Event()

    class Blocking(Backend):
        def read(self, selection):
            if self.calls:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('test failed to release native I/O')
            return super().read(selection)

    def consume():
        with ArrayReader(Blocking(), buffer_bytes=64).prefetch((slice(None),)) as blocks:
            next(blocks)
            assert entered.wait(3)
        closed.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(consume)
        try:
            assert entered.wait(3)
            assert not closed.is_set()
        finally:
            release.set()
        future.result(timeout=5)
    assert closed.is_set()


@pytest.mark.parametrize('selection', [(slice(0, 0),), (slice(3, 30, 3),), (12,)])
def test_empty_strided_and_scalar_selections(selection):
    backend = Backend()
    reader = ArrayReader(backend, buffer_bytes=128)
    expected = backend.values[selection]
    result = np.empty_like(expected)
    with reader.prefetch(selection) as blocks:
        for tile, value in blocks:
            result[tile.destination] = value
    np.testing.assert_array_equal(result, expected)


def test_insufficient_budget_fails_before_backend_io():
    backend = Backend()
    with pytest.raises(ValueError, match='two read slots'):
        with ArrayReader(backend, buffer_bytes=1).prefetch((slice(None),)):
            pass
    with pytest.raises((ValueError, MemoryError)):
        with ArrayReader(backend, buffer_bytes=16).prefetch((slice(None),)) as blocks:
            next(blocks)
    assert not backend.calls
