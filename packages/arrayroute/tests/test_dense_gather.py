# SPDX-License-Identifier: Apache-2.0
import gc
import weakref
from dataclasses import replace

import numpy as np
import pytest

from arrayroute import ArrayReader, ArraySpec, DecodeWindow, ReadStats, plan
from arrayroute.backends import DenseGatherBackend
from arrayroute.budget import ReadBudget
from arrayroute.concurrency import plan_concurrency
from arrayroute.session import LocalReadSession, TensorStoreReadGroup


class ObservedArray:
    def __init__(self):
        self.values = np.arange(32 * 32, dtype='f4').reshape(32, 32)
        self.dtype = self.values.dtype
        self.spec = ArraySpec((32, 32), (4, 4), 4, DecodeWindow(1, strided_only=True))
        self.calls, self.refs = [], []

    def read(self, selection):
        self.calls.append(selection)
        result = self.values[selection].copy()
        self.refs.append(weakref.ref(result))
        return result


SELECTION = (slice(None, None, 2),) * 2


def test_plan_charges_intermediate_and_native_decoding_without_extra_chunks():
    spec = DenseGatherBackend(ObservedArray()).spec
    native = plan(spec, SELECTION)
    assert native.strategy == 'native'
    bulk = plan(spec, SELECTION, buffer_bytes=9000)
    assert bulk.strategy == 'dense-gather'
    assert bulk.read_selection == (slice(0, 31, 1),) * 2
    assert bulk.output_bytes == 1024
    assert bulk.intermediate_bytes_upper_bound == 31 * 31 * 4
    assert bulk.peak_decoded_bytes_upper_bound == bulk.decoded_bytes_upper_bound == 4096
    assert bulk.buffer_bytes_upper_bound == 8964
    assert bulk.chunks_touched == native.chunks_touched
    assert plan(spec, SELECTION, buffer_bytes=8964).strategy == 'dense-gather'
    assert plan(spec, SELECTION, buffer_bytes=8963).strategy == 'native'
    sparse = plan(spec, (slice(None, None, 9),) * 2, buffer_bytes=1 << 20)
    assert sparse.strategy == 'native' and sparse.intermediate_bytes_upper_bound == 0
    for selection in [(slice(0, 0, 2), slice(None)), (2, 3), (slice(None),) * 2]:
        assert plan(spec, selection, buffer_bytes=1 << 20).strategy == 'native'


@pytest.mark.parametrize('budget', [0, -1, True, 1.5])
def test_invalid_plan_budgets(budget):
    with pytest.raises(ValueError):
        plan(ObservedArray().spec, SELECTION, buffer_bytes=budget)


def test_capability_requires_planned_execution_and_valid_strategy():
    with pytest.raises(TypeError):
        ArraySpec((1,), (1,), 4, dense_gather=1)
    source = ObservedArray()
    source.spec = replace(source.spec, dense_gather=True)
    with pytest.raises(TypeError, match='read_plan'):
        ArrayReader(source)
    with pytest.raises(ValueError, match='read_strategy'):
        LocalReadSession(read_strategy='typo')


def test_intermediate_is_released_before_result_leaves_reserved_read():
    source = ObservedArray()
    pool = ReadBudget(9000)
    reader = ArrayReader(DenseGatherBackend(source), buffer_bytes=9000, shared_budget=pool)
    enabled = gc.isenabled()
    gc.disable()
    try:
        value = reader.read(SELECTION)
        assert value.tobytes() == source.values[SELECTION].tobytes()
        assert value.flags.c_contiguous and value.base is None
        assert all(ref() is None for ref in source.refs)
    finally:
        if enabled:
            gc.enable()
    assert source.calls == [(slice(0, 31, 1),) * 2]
    assert pool.snapshot()['peak_bytes'] == 8964
    assert pool.snapshot()['used_bytes'] == 0


def test_prefetch_replans_against_each_half_budget():
    source = ObservedArray()
    pool = ReadBudget(12000)
    reader = ArrayReader(DenseGatherBackend(source), buffer_bytes=12000, shared_budget=pool)
    assert reader.plan(SELECTION).strategy == 'dense-gather'
    with reader.prefetch(SELECTION) as stream:
        tile, value = next(stream)
        assert tile.plan.strategy == 'native'
        assert tile.plan.buffer_bytes_upper_bound <= 6000
        assert value.tobytes() == source.values[SELECTION].tobytes()
    assert all(s.step == 2 for s in source.calls[0])
    assert pool.snapshot()['used_bytes'] == 0


@pytest.mark.parametrize('budget', [256, 2000, 9000])
def test_stream_and_worker_choices_obey_effective_budget(budget):
    source = ObservedArray()
    backend = DenseGatherBackend(source)
    actual, coverage = np.empty((16, 16), dtype='f4'), np.zeros((16, 16), dtype='u1')
    for tile, value in ArrayReader(backend, buffer_bytes=budget).stream(SELECTION):
        assert tile.plan.buffer_bytes_upper_bound <= budget
        actual[tile.destination] = value
        coverage[tile.destination] += 1
    assert actual.tobytes() == source.values[SELECTION].tobytes()
    assert np.all(coverage == 1)
    choices = plan_concurrency(backend.spec, SELECTION, total_buffer_bytes=budget, max_workers=4)
    for c in choices.candidates:
        reader = ArrayReader(backend, buffer_bytes=c.buffer_bytes_per_worker)
        if c.materialization in {'stream-required', 'chunk-too-large'}:
            with pytest.raises(MemoryError):
                reader.read(SELECTION)
        else:
            assert reader.plan(SELECTION).strategy == c.strategy
            assert reader.read(SELECTION).tobytes() == source.values[SELECTION].tobytes()


@pytest.mark.parametrize('engine', ['zarr-python', 'tensorstore', 'zarrista'])
def test_native_engines_preserve_exceptional_bits_and_integer_axes(tmp_path, engine):
    import zarr

    rng = np.random.default_rng(642)
    values = rng.integers(0, 2**32, size=(7, 13, 11, 3), dtype='u4').view('f4')
    path = tmp_path / 'values.zarr'
    zarr.create_array(path, data=values, chunks=(2, 3, 2, 3), shards=(4, 6, 4, 3))
    reader = LocalReadSession(engine, read_strategy='dense').open(path, '')
    for selection in [(3, slice(1, 13, 2), slice(None), 1),
                      (slice(None, None, 2), slice(0, 0), slice(None), 2),
                      (slice(1, 7, 2), 3, slice(None), slice(None)),
                      (1, 2, 3, 2), (slice(None),) * 4]:
        stats = ReadStats()
        assert reader.read(selection, stats=stats).tobytes() == values[selection].tobytes()
        if reader.plan(selection).output_bytes:
            assert (stats.decoded_bytes_upper_bound
                    == reader.plan(selection).decoded_bytes_upper_bound)


def test_cached_miss_receives_plan_and_hits_do_not_retain_dense_intermediate(tmp_path):
    from contextlib import closing

    from arrayroute.disk_cache import ArrayIdentity, TileCache

    source = ObservedArray()
    with closing(TileCache(tmp_path, max_entry_bytes=1 << 20, capacity_bytes=4 << 20)) as cache:
        backend = cache.backend(
            DenseGatherBackend(source), ArrayIdentity('source', 'revision', 'f4'))
        reader = ArrayReader(backend, buffer_bytes=9000)
        first = reader.read(SELECTION)
        second = reader.read(SELECTION)
        assert first.tobytes() == second.tobytes() == source.values[SELECTION].tobytes()
        assert len(source.calls) == 1 and source.calls[0] == (slice(0, 31, 1),) * 2
        assert all(ref() is None for ref in source.refs)
        assert backend.stats.hits == 1


def test_shared_native_group_composes_with_dense_gather(tmp_path):
    import zarr

    values = np.arange(32 * 32, dtype='f4').reshape(32, 32)
    path = tmp_path / 'values.zarr'
    zarr.create_array(path, data=values, chunks=(4, 4), shards=(16, 16))
    group = TensorStoreReadGroup(threads=1, cache_bytes=0)
    reader = group.session(buffer_bytes=9000, read_strategy='dense').open(path, '')
    assert reader.plan(SELECTION).strategy == 'dense-gather'
    assert reader.read(SELECTION).tobytes() == values[SELECTION].tobytes()


def test_shared_reservations_cover_intermediates_across_concurrent_readers():
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    pool = ReadBudget(2 * 8964)
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    active = [0, 0]

    class SlowSource(ObservedArray):
        def read(self, selection):
            with lock:
                active[0] += 1
                active[1] = max(active)
                assert active[0] * 8964 <= pool.snapshot()['used_bytes']
            try:
                time.sleep(.01)
                return super().read(selection)
            finally:
                with lock:
                    active[0] -= 1

    def read(_):
        reader = ArrayReader(
            DenseGatherBackend(SlowSource()), buffer_bytes=9000, shared_budget=pool)
        barrier.wait(timeout=5)
        return reader.read(SELECTION).tobytes()

    with ThreadPoolExecutor(4) as executor:
        results = list(executor.map(read, range(4)))
    assert len(set(results)) == 1 and active[1] == 2
    assert pool.snapshot()['peak_bytes'] == 2 * 8964
    assert pool.snapshot()['used_bytes'] == 0


@pytest.mark.parametrize('method', ['read', 'stream', 'prefetch'])
def test_shared_pool_smaller_than_reader_replans_before_allocation(method):
    source = ObservedArray()
    pool = ReadBudget(2500)
    reader = ArrayReader(DenseGatherBackend(source), buffer_bytes=9000, shared_budget=pool)
    assert reader.buffer_bytes == 2500
    assert reader.plan(SELECTION).strategy == 'native'
    if method == 'read':
        actual = reader.read(SELECTION)
    else:
        actual = np.empty((16, 16), dtype='f4')
        if method == 'prefetch':
            with reader.prefetch(SELECTION) as stream:
                for tile, value in stream:
                    actual[tile.destination] = value
                    del value
        else:
            for tile, value in reader.stream(SELECTION):
                actual[tile.destination] = value
                del value
    assert actual.tobytes() == source.values[SELECTION].tobytes()
    assert pool.snapshot()['peak_bytes'] <= 2500
    assert pool.snapshot()['used_bytes'] == 0
