# SPDX-License-Identifier: Apache-2.0
import gc
import hashlib
import pickle
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from arrayroute.budget import ReadBudget
from arrayroute.plan import ArraySpec
from arrayroute.reader import ArrayReader
from arrayroute.session import LocalReadSession


class Backend:
    dtype = np.dtype('f4')
    spec = ArraySpec((64, 64), (8, 8), 4)

    def read(self, selection):
        return np.arange(64**2, dtype='f4').reshape(64, 64)[selection].copy()


def test_concurrent_materialization_does_not_hold_outputs_while_waiting_for_tiles():
    pool = ReadBudget(4096)
    selection = (slice(None, None, 8), slice(None, None, 8))
    barrier = threading.Barrier(8)

    def run(_):
        reader = ArrayReader(Backend(), buffer_bytes=2048, shared_budget=pool)
        barrier.wait(timeout=5)
        return reader.read(selection).tobytes()

    with ThreadPoolExecutor(8) as executor:
        results = list(executor.map(run, range(8)))
    assert all(r == Backend().read(selection).tobytes() for r in results)
    assert pool.snapshot()['used_bytes'] == 0
    assert pool.snapshot()['peak_bytes'] <= 4096


def test_slow_stream_consumers_share_planned_budget_without_retaining_outputs():
    pool = ReadBudget(4 << 20)
    refs, lock = [], threading.Lock()
    barrier = threading.Barrier(8)

    class LargeBackend:
        dtype = np.dtype('f4')
        spec = ArraySpec((512, 512), (512, 512), 4)

        def read(self, selection):
            value = np.ones((512, 512), dtype='f4')
            with lock:
                refs.append(weakref.ref(value))
                assert sum(v.nbytes for r in refs if (v := r()) is not None) <= 2 << 20
            return value

    def run(_):
        barrier.wait(timeout=5)
        for tile, value in ArrayReader(LargeBackend(), shared_budget=pool).stream(
                (slice(None), slice(None))):
            digest = hashlib.sha256(value).hexdigest()
            time.sleep(.01)
            del value, tile
        return digest

    with ThreadPoolExecutor(8) as executor:
        hashes = list(executor.map(run, range(8)))
    assert len(set(hashes)) == 1
    assert pool.snapshot()['peak_active'] == 2
    assert pool.snapshot()['peak_bytes'] == 4 << 20
    assert pool.snapshot()['used_bytes'] == 0
    assert all(ref() is None for ref in refs)


def test_stream_close_and_backend_error_return_reservations():
    pool = ReadBudget(4096)
    reader = ArrayReader(Backend(), buffer_bytes=2048, shared_budget=pool)
    stream = reader.stream((slice(None), slice(None)))
    next(stream)
    assert pool.snapshot()['used_bytes'] > 0
    stream.close()
    assert pool.snapshot()['used_bytes'] == 0

    class Broken(Backend):
        def read(self, selection):
            raise OSError('storage unavailable')

    with pytest.raises(OSError):
        ArrayReader(Broken(), buffer_bytes=2048, shared_budget=pool).read((0, 0))
    assert pool.snapshot()['used_bytes'] == 0


def test_prefetch_charges_both_slots_until_context_exit():
    pool = ReadBudget(4096)
    reader = ArrayReader(Backend(), buffer_bytes=2048, shared_budget=pool)
    with reader.prefetch((slice(None), slice(None))) as source:
        next(source)
        assert pool.snapshot()['used_bytes'] == 2048
    assert pool.snapshot()['used_bytes'] == 0


def test_waiting_cancellation_and_fifo_admission():
    pool = ReadBudget(10)
    cancel = threading.Event()
    order = []

    def run(amount, name, stop=None):
        with pool.reserve(amount, cancelled=stop):
            order.append(name)

    def wait_for(count):
        deadline = time.monotonic() + 3
        while pool.snapshot()['waiting'] != count:
            assert time.monotonic() < deadline
            time.sleep(.001)

    with ThreadPoolExecutor(2) as executor:
        with pool.reserve(6):
            first = executor.submit(run, 8, 'large', cancel.is_set)
            wait_for(1)
            second = executor.submit(run, 2, 'small')
            wait_for(2)
            assert order == []
            cancel.set()
            with pytest.raises(InterruptedError):
                first.result(timeout=3)
            second.result(timeout=3)
        assert order == ['small']
    assert pool.snapshot()['waiting'] == pool.snapshot()['used_bytes'] == 0


def test_nested_oversized_and_cross_process_requests_fail_without_io(monkeypatch):
    pool = ReadBudget(8)
    with pool.reserve(4), pytest.raises(RuntimeError, match='nesting'):
        with pool.reserve(1):
            pass
    with pytest.raises(MemoryError):
        with pool.reserve(9):
            pass
    with pytest.raises(TypeError):
        pickle.dumps(pool)
    with pytest.raises(TypeError):
        pickle.dumps(LocalReadSession(read_budget=pool))
    monkeypatch.setattr('arrayroute.budget.os.getpid', lambda: pool._pid + 1)
    with pytest.raises(RuntimeError, match='process'):
        pool.snapshot()


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_budget(value):
    with pytest.raises((ValueError, TypeError)):
        ReadBudget(value)


def test_no_gc_required_for_budget_release():
    enabled = gc.isenabled()
    gc.disable()
    try:
        pool = ReadBudget(1024)
        reader = ArrayReader(Backend(), buffer_bytes=1024, shared_budget=pool)
        for _ in range(20):
            assert reader.read((0, 0)) == 0
        assert pool.snapshot()['used_bytes'] == 0
    finally:
        if enabled:
            gc.enable()


def test_native_cache_group_and_read_admission_compose(tmp_path):
    import zarr

    from arrayroute.session import TensorStoreReadGroup

    values = np.arange(256, dtype='f4').reshape(16, 16)
    zarr.create_array(tmp_path, data=values, chunks=(8, 8))
    pool = ReadBudget(4096)
    group = TensorStoreReadGroup(cache_bytes=1 << 20, threads=2, immutable=True)
    sessions = [group.session(buffer_bytes=4096, read_budget=pool) for _ in range(2)]
    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(lambda s: s.read(tmp_path, '', (slice(None), slice(None))),
                                    sessions))
    assert all(r.tobytes() == values.tobytes() for r in results)
    assert all(s.open(tmp_path, '').shared_budget is pool for s in sessions)
    assert sessions[0]._context is sessions[1]._context
    assert pool.snapshot()['used_bytes'] == 0
