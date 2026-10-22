# SPDX-License-Identifier: Apache-2.0
import gc
import threading
import time
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import replace

import numpy as np
import pytest

from arrayroute import ArrayReader, ArraySpec, DecodeWindow, GatherWindow, plan
from arrayroute.budget import ReadBudget
from arrayroute.gather import ParallelGatherBackend
from arrayroute.session import LocalReadSession


class Source:
    def __init__(self, values=None):
        self.values = (np.arange(32*32, dtype='f4').reshape(32, 32)
                       if values is None else values)
        self.dtype = self.values.dtype
        self.spec = ArraySpec(self.values.shape, (4,)*self.values.ndim,
                              self.dtype.itemsize, DecodeWindow(1, strided_only=True))

    def read(self, selection):
        return self.values[selection].copy()


def backend(workers=4):
    return ParallelGatherBackend(Source(), Source, GatherWindow((8, 8), workers))


SELECTION = (slice(None, None, 2),)*2


def test_planner_reduces_workers_before_using_native_path():
    spec = backend().spec
    assert plan(spec, SELECTION).strategy == 'native'
    request = plan(spec, SELECTION, buffer_bytes=1536)
    assert request.strategy == 'parallel-gather' and request.gather_workers == 4
    assert request.output_bytes == 1024
    assert request.intermediate_bytes_upper_bound == 256
    assert request.peak_decoded_bytes_upper_bound == 256
    assert request.decoded_bytes_upper_bound == 4096
    for budget, workers in [(1535, 3), (1407, 2), (1279, 1)]:
        request = plan(spec, SELECTION, buffer_bytes=budget)
        assert request.gather_workers == workers
        assert request.buffer_bytes_upper_bound <= budget
    for selection in [(slice(0, 4, 2),)*2, (slice(0, 0, 2), slice(None)),
                      (3, 2), (slice(None),)*2]:
        assert plan(spec, selection, buffer_bytes=999999).strategy == 'native'


@pytest.mark.parametrize('workers', [0, -1, True, 257, 2.5])
def test_invalid_worker_count(workers):
    with pytest.raises(ValueError):
        GatherWindow((8, 8), workers)
    with pytest.raises(ValueError):
        LocalReadSession('zarrista', read_strategy='parallel', gather_workers=workers)


def test_invalid_capabilities_and_strategy_combinations():
    spec = Source().spec
    for changes in [dict(dense_gather=True), dict(decode_window=None),
                    dict(decode_window=DecodeWindow(2))]:
        with pytest.raises(ValueError):
            replace(spec, gather_window=GatherWindow((8, 8), 4), **changes)
    for shape in [(6, 8), (8,), (8, 8, 8)]:
        with pytest.raises(ValueError):
            replace(spec, gather_window=GatherWindow(shape, 4))
    with pytest.raises(TypeError):
        replace(spec, gather_window=True)
    for shape in [(), (0, 8), (True, 8), (1.5, 8)]:
        with pytest.raises(ValueError):
            GatherWindow(shape, 4)
    with pytest.raises(ValueError, match='Zarrista'):
        LocalReadSession('tensorstore', read_strategy='parallel')
    with pytest.raises(ValueError, match='gather_workers'):
        LocalReadSession('zarrista', gather_workers=8)
    with pytest.raises(ValueError, match='Nested'):
        ParallelGatherBackend(backend(), Source, GatherWindow((8, 8), 4))
    source = Source()
    source.spec = backend().spec
    with pytest.raises(TypeError, match='read_plan'):
        ArrayReader(source)


@pytest.mark.parametrize('budget', [96, 1100, 1280, 1536, 8192])
def test_stream_materialization_and_prefetch_are_exact_under_budget(budget):
    reader = ArrayReader(backend(), buffer_bytes=budget)
    expected = Source().values[SELECTION]
    with pytest.raises(MemoryError) if budget <= 1024 else nullcontext():
        actual = reader.read(SELECTION)
        assert actual.tobytes() == expected.tobytes()
    for prefetched in [False, True]:
        if prefetched and budget < 2*(64+4):
            continue
        actual, coverage = np.empty_like(expected), np.zeros(expected.shape, dtype='u1')
        with reader.prefetch(SELECTION) if prefetched else _stream(reader) as rows:
            for tile, value in rows:
                assert tile.plan.buffer_bytes_upper_bound <= budget//(2 if prefetched else 1)
                actual[tile.destination] = value
                coverage[tile.destination] += 1
                del value
        assert actual.tobytes() == expected.tobytes() and np.all(coverage == 1)


@contextmanager
def _stream(reader):
    rows = reader.stream(SELECTION)
    try:
        yield rows
    finally:
        rows.close()


def test_workers_are_bounded_and_release_results_without_cyclic_gc():
    lock = threading.Lock()
    counts = {'active': 0, 'peak': 0, 'calls': 0, 'live_outputs': 0}
    refs = []
    pool = ReadBudget(1536)

    class Observed(Source):
        def read(self, selection):
            with lock:
                counts['active'] += 1
                counts['peak'] = max(counts['peak'], counts['active'])
                assert pool.snapshot()['used_bytes'] >= 1536
            try:
                time.sleep(.005)
                value = super().read(selection)
                with lock:
                    refs.append(weakref.ref(value))
                    counts['calls'] += 1
                    counts['live_outputs'] = max(counts['live_outputs'],
                                                sum(r() is not None for r in refs))
                return value
            finally:
                with lock:
                    counts['active'] -= 1

    reader = ArrayReader(ParallelGatherBackend(Source(), Observed, GatherWindow((8, 8), 4)),
                         buffer_bytes=9000, shared_budget=pool)
    enabled = gc.isenabled()
    gc.disable()
    try:
        actual = reader.read(SELECTION)
        assert actual.tobytes() == Source().values[SELECTION].tobytes()
        assert all(ref() is None for ref in refs)
    finally:
        if enabled:
            gc.enable()
    assert counts['calls'] == 16 and counts['peak'] == 4
    assert counts['live_outputs'] <= 4 and counts['active'] == 0
    assert pool.snapshot()['used_bytes'] == 0 and pool.snapshot()['peak_bytes'] == 1536


@pytest.mark.parametrize('failure', ['factory', 'read', 'shape', 'dtype'])
def test_worker_failures_join_threads_and_return_shared_reservation(failure):
    before = {t.ident for t in threading.enumerate()}
    pool = ReadBudget(1536)

    class Broken(Source):
        def read(self, selection):
            time.sleep(.001)
            if failure == 'read':
                raise OSError('read failure')
            result = super().read(selection)
            return result[:1] if failure == 'shape' else result.astype('f8')

    def factory():
        value = Broken()
        if failure == 'factory':
            value.spec = replace(value.spec, shape=(33, 32))
        return value

    reader = ArrayReader(ParallelGatherBackend(Source(), factory, GatherWindow((8, 8), 4)),
                         buffer_bytes=1536, shared_budget=pool)
    with pytest.raises((OSError, ValueError)):
        reader.read(SELECTION)
    assert pool.snapshot()['used_bytes'] == 0
    assert not [t for t in threading.enumerate()
                if t.ident not in before and t.name.startswith('arrayroute-map')]


def test_zarrista_random_regions_integer_axes_and_exceptional_bits(tmp_path):
    import zarr

    rng = np.random.default_rng(624)
    values = rng.integers(0, 2**32, size=(9, 17, 15, 3), dtype='u4').view('f4')
    path = tmp_path/'field.zarr'
    zarr.create_array(path, data=values, chunks=(2, 3, 2, 3), shards=(4, 6, 4, 3))
    reader = LocalReadSession('zarrista', read_strategy='parallel', gather_workers=4).open(path, '')
    saw_parallel = False
    for _ in range(100):
        selection = tuple(int(rng.integers(n)) if rng.random() < .25 else
            slice(int(rng.integers(n)), n, int(rng.integers(1, 7))) for n in values.shape)
        saw_parallel |= reader.plan(selection).strategy == 'parallel-gather'
        assert reader.read(selection).tobytes() == values[selection].tobytes()
    assert saw_parallel


def test_unsharded_array_requires_explicit_other_strategy(tmp_path):
    import zarr

    path = tmp_path/'field.zarr'
    zarr.create_array(path, data=np.zeros((8, 8), dtype='f4'), chunks=(4, 4))
    with pytest.raises(ValueError, match='independently decodable'):
        LocalReadSession('zarrista', read_strategy='parallel').open(path, '')


def test_native_subchunks_are_not_repeated_or_retained_across_workers(tmp_path):
    from collections import Counter

    import zarr
    import zarrista

    from arrayroute.backends import ZarristaBackend

    path = tmp_path/'field.zarr'
    values = np.random.default_rng(812).normal(size=(1, 64, 64, 32, 3)).astype('f4')
    zarr.create_array(path, data=values, chunks=(1, 8, 8, 8, 3), shards=(1, 16, 16, 16, 3))
    native = zarrista.Array.open(zarrista.store.FilesystemStore(str(path)))
    refs, calls = [], Counter()
    peak = [0]
    lock = threading.Lock()

    class Observed:
        def __getattr__(self, name):
            return getattr(native, name)

        def __getitem__(self, selection):
            return native[selection]

        def retrieve_subchunk(self, coordinates, *, shard_cache):
            value = np.asarray(native.retrieve_subchunk(coordinates, shard_cache=shard_cache))
            with lock:
                refs.append(weakref.ref(value))
                calls[tuple(coordinates)] += 1
                peak[0] = max(peak[0], sum(ref() is not None for ref in refs))
            time.sleep(.001)
            return value

    wrapped = Observed()
    base = ZarristaBackend(wrapped)
    reader = ArrayReader(ParallelGatherBackend(base, lambda: ZarristaBackend(wrapped),
                         GatherWindow((1, 16, 16, 16, 3), 4)), buffer_bytes=1 << 20)
    selection = (0, slice(1, 64, 2), slice(2, 63, 3), slice(None, None, 7), slice(None))
    request = reader.plan(selection)
    assert request.strategy == 'parallel-gather'
    enabled = gc.isenabled()
    gc.disable()
    try:
        assert reader.read(selection).tobytes() == values[selection].tobytes()
        assert all(ref() is None for ref in refs)
    finally:
        if enabled:
            gc.enable()
    assert 2 <= peak[0] <= 4
    assert len(calls) == request.chunks_touched and set(calls.values()) == {1}


def test_cache_forwards_parallel_plan_and_hits_do_not_launch_workers(tmp_path):
    from contextlib import closing

    from arrayroute.disk_cache import ArrayIdentity, TileCache

    factories = []

    def factory():
        factories.append(1)
        return Source()

    source = ParallelGatherBackend(Source(), factory, GatherWindow((8, 8), 4))
    with closing(TileCache(tmp_path, capacity_bytes=1 << 20, max_entry_bytes=4096)) as cache:
        reader = ArrayReader(cache.backend(source, ArrayIdentity('source', 'rev', 'f4')),
                             buffer_bytes=1536)
        expected = reader.read(SELECTION).tobytes()
        count = len(factories)
        assert 1 <= count <= 4
        assert reader.read(SELECTION).tobytes() == expected
        assert len(factories) == count


def test_petabyte_planning_does_not_enumerate_region_grid():
    spec = ArraySpec((1000000,)*3, (32,)*3, 4, DecodeWindow(1, strided_only=True),
                     gather_window=GatherWindow((128,)*3, 256))
    request = plan(spec, (slice(None, None, 64),)*3, buffer_bytes=256 << 20)
    assert spec.logical_bytes == 4 * 10**18
    assert request.output_bytes > 256 << 20
    assert request.strategy == 'native'
