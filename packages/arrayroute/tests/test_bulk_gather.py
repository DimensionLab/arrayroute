# SPDX-License-Identifier: Apache-2.0
import gc
import threading
import time
import weakref
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from arrayroute import ArrayReader, ArraySpec, DecodeWindow, GatherWindow, ReadStats, plan
from arrayroute.backends import ZarristaBackend
from arrayroute.budget import ReadBudget
from arrayroute.gather import ParallelGatherBackend
from arrayroute.session import LocalReadSession


def spec():
    return ArraySpec((32, 32), (4, 4), 4, DecodeWindow(1, strided_only=True),
                     gather_window=GatherWindow((8, 8), 4, bulk=True))


@pytest.mark.parametrize('budget,workers,strategy', [
    (3328, 4, 'bulk-gather'), (3327, 3, 'bulk-gather'),
    (2751, 2, 'bulk-gather'), (2175, 1, 'bulk-gather'),
    (1599, 4, 'parallel-gather'), (1279, 1, 'native'),
])
def test_bulk_budget_selects_worker_count_and_partial_decode(budget, workers, strategy):
    request = plan(spec(), (slice(None, None, 2),)*2, buffer_bytes=budget)
    assert request.strategy == strategy and request.gather_workers == workers
    assert request.buffer_bytes_upper_bound <= budget
    assert request.decoded_bytes_upper_bound == 4096


@pytest.mark.parametrize('selection', [
    (slice(None, None, 9),)*2, (slice(4, 32, 2), slice(None, None, 2)),
    (0, slice(None, None, 2)), (slice(0, 0, 2), slice(None)),
])
def test_bulk_does_not_read_skipped_chunks(selection):
    assert plan(spec(), selection, buffer_bytes=100000).strategy != 'bulk-gather'


def test_padded_region_bytes_are_counted():
    request = plan(replace(spec(), shape=(29, 29)), (slice(None, None, 2),)*2,
                   buffer_bytes=10000)
    assert request.strategy == 'bulk-gather'
    assert request.decoded_bytes_upper_bound == 16*8*8*4
    assert request.peak_decoded_bytes_upper_bound == 2*4*8*8*4


def test_bulk_requires_explicit_factory_and_boolean():
    with pytest.raises(TypeError, match='boolean'):
        GatherWindow((8, 8), 4, bulk=1)
    with pytest.raises(ValueError, match='Zarrista'):
        LocalReadSession('tensorstore', read_strategy='bulk')


@pytest.fixture
def field(tmp_path):
    import zarr
    import zarrista

    values = np.random.default_rng(732).integers(
        0, 2**32, size=(3, 29, 27, 3), dtype='u4').view('f4')
    path = tmp_path/'f.zarr'
    zarr.create_array(path, data=values, chunks=(1, 4, 3, 3), shards=(1, 8, 6, 3))
    return path, values, zarrista.Array.open(zarrista.store.FilesystemStore(str(path)))


def test_randomized_bulk_partial_and_integer_axes_preserve_bits(field):
    path, values, _ = field
    reader = LocalReadSession('zarrista', read_strategy='bulk', gather_workers=4).open(path, '')
    rng = np.random.default_rng(843)
    selections = [(1, slice(None, None, 2), slice(None, None, 2), slice(None))]
    selections += [tuple(int(rng.integers(n)) if rng.random() < .2 else
        slice(int(rng.integers(n)), n, int(rng.integers(1, 8))) for n in values.shape)
        for _ in range(100)]
    strategies = set()
    for selection in selections:
        strategies.add(reader.plan(selection).strategy)
        assert reader.read(selection).tobytes() == values[selection].tobytes()
    assert 'bulk-gather' in strategies and 'parallel-gather' in strategies


@pytest.mark.parametrize('budget', [6000, 16000, 1 << 20])
def test_stream_prefetch_shared_and_materialization(field, budget):
    path, values, _ = field
    pool = ReadBudget(budget)
    reader = LocalReadSession('zarrista', read_strategy='bulk', buffer_bytes=1 << 20,
                              read_budget=pool).open(path, '')
    selection = (1, slice(None, None, 2), slice(None, None, 2), slice(None))
    expected = values[selection]
    assert reader.read(selection).tobytes() == expected.tobytes()
    actual = np.empty_like(expected)
    with reader.prefetch(selection) as tiles:
        for tile, value in tiles:
            assert tile.plan.buffer_bytes_upper_bound <= budget//2
            actual[tile.destination] = value
            del value
    assert actual.tobytes() == expected.tobytes()
    assert pool.snapshot()['used_bytes'] == 0 and pool.snapshot()['peak_bytes'] <= budget


def test_bulk_lifetimes_chunk_counts_and_cache_plan(field):
    path, values, native = field
    calls, refs, lock, peak = Counter(), [], threading.Lock(), [0]

    class Observed:
        def __getattr__(self, name):
            return getattr(native, name)

        def retrieve_chunk(self, coordinates, *, concurrent_target):
            assert concurrent_target == 1
            value = np.asarray(native.retrieve_chunk(coordinates, concurrent_target=1))
            with lock:
                calls[tuple(coordinates)] += 1
                refs.append(weakref.ref(value))
                peak[0] = max(peak[0], sum(r() is not None for r in refs))
            time.sleep(.001)
            return value

    observed = Observed()
    window = GatherWindow((1, 8, 6, 3), 4, bulk=True)
    backend = ParallelGatherBackend(ZarristaBackend(observed), lambda: ZarristaBackend(observed),
        window, bulk_factory=lambda: ZarristaBackend(observed).read_shard_region)
    reader = ArrayReader(backend, buffer_bytes=1 << 20)
    selection = (1, slice(None, None, 2), slice(None, None, 2), slice(None))
    stats = ReadStats()
    enabled = gc.isenabled()
    gc.disable()
    try:
        assert reader.read(selection, stats=stats).tobytes() == values[selection].tobytes()
        assert all(r() is None for r in refs)
    finally:
        if enabled:
            gc.enable()
    assert 2 <= peak[0] <= 4 and set(calls.values()) == {1} and len(calls) == 20
    assert stats.read_strategies == {'bulk-gather': 1} and stats.gather_workers_peak == 4

    from contextlib import closing

    from arrayroute.disk_cache import ArrayIdentity, TileCache
    with closing(TileCache(path.parent/'cache', capacity_bytes=1 << 20,
                           max_entry_bytes=1 << 14)) as cache:
        cached = ArrayReader(cache.backend(backend, ArrayIdentity('source', 'rev', 'f4')),
                             buffer_bytes=1 << 20)
        cached.read(selection)
        count = calls.copy()
        assert cached.read(selection).tobytes() == values[selection].tobytes()
        assert calls == count


@pytest.mark.parametrize('failure', ['factory', 'read', 'shape', 'dtype'])
def test_bulk_failure_joins_workers_and_releases_pool(field, failure):
    _, _, native = field
    active, lock = [0], threading.Lock()

    def factory():
        if failure == 'factory':
            raise RuntimeError('factory')

        def read(region):
            with lock:
                active[0] += 1
            try:
                time.sleep(.005)
                if failure == 'read':
                    raise RuntimeError('read')
                value = ZarristaBackend(native).read_shard_region(region)
                return value.ravel() if failure == 'shape' else value.astype('f8')
            finally:
                with lock:
                    active[0] -= 1
        return read

    base = ZarristaBackend(native)
    window = GatherWindow((1, 8, 6, 3), 4, bulk=True)
    with pytest.raises(ValueError, match='factory'):
        ParallelGatherBackend(base, lambda: base, window)
    pool = ReadBudget(1 << 20)
    reader = ArrayReader(ParallelGatherBackend(base, lambda: base, window, bulk_factory=factory),
                         buffer_bytes=1 << 20, shared_budget=pool)
    with pytest.raises((ValueError, RuntimeError)):
        reader.read((1, slice(None, None, 2), slice(None, None, 2), slice(None)))
    assert active[0] == 0 and pool.snapshot()['used_bytes'] == 0


def test_metadata_scale_bulk_planning_has_no_region_inventory():
    source = ArraySpec((10**6,)*3, (32,)*3, 4, DecodeWindow(1, strided_only=True),
                       gather_window=GatherWindow((128,)*3, 256, bulk=True))
    assert plan(source, (slice(None, None, 8),)*3, buffer_bytes=256 << 20).strategy == 'native'
