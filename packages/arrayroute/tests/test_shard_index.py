# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import zarr

from arrayroute import ArrayReader
from arrayroute.backends import ZarristaBackend

zarrista = pytest.importorskip('zarrista')


@pytest.mark.parametrize('sharded', [False, True])
def test_strides_match_bits_across_partial_shards_and_missing_chunks(tmp_path, sharded):
    rng = np.random.default_rng(289)
    values = rng.integers(0, 2**32, size=(9, 13, 7), dtype=np.uint32).view(np.float32)
    path = tmp_path / 'field.zarr'
    options = {'shards': (6, 8, 4)} if sharded else {}
    stored = zarr.create_array(path, shape=values.shape, dtype='f4', chunks=(3, 4, 2),
                               fill_value=-7, **options)
    stored[:6] = values[:6]
    values[6:] = -7
    backend = ZarristaBackend(zarrista.Array.open(zarrista.store.FilesystemStore(str(path))))
    for budget in [400, 1 << 20]:
        reader = ArrayReader(backend, buffer_bytes=budget)
        for selection in [
            (slice(1, 9, 2), slice(2, 13, 3), 3),
            (7, slice(None, None, 2), slice(1, 7, 3)),
            (slice(None, None, 4), 0, slice(None, None, 2)),
            (slice(0, 0, 2), slice(None), slice(None)),
        ]:
            assert reader.read(selection).tobytes() == values[selection].tobytes()


def test_shard_index_does_not_outlive_read_or_accumulate(tmp_path):
    path = tmp_path / 'field.zarr'
    values = np.arange(16 * 24, dtype='f4').reshape(16, 24)
    stored = zarr.create_array(path, data=values, chunks=(2, 3), shards=(4, 6))
    native = zarrista.Array.open(zarrista.store.FilesystemStore(str(path)))
    caches, sizes = [], []

    class Observed:
        def __getattr__(self, name):
            return getattr(native, name)

        def shard_cache(self):
            cache = native.shard_cache()
            caches.append(cache)
            return cache

        def retrieve_subchunk(self, coordinates, *, shard_cache):
            value = native.retrieve_subchunk(coordinates, shard_cache=shard_cache)
            sizes.append(shard_cache.size())
            return value

    reader = ArrayReader(ZarristaBackend(Observed()), buffer_bytes=1 << 20)
    selection = (slice(1, 16, 2), slice(0, 24, 2))
    assert reader.read(selection).tobytes() == values[selection].tobytes()
    stored[:] = values + 7
    assert reader.read(selection).tobytes() == (values + 7)[selection].tobytes()
    assert len(caches) == 2 and caches[0] is not caches[1]
    assert sizes and max(sizes) == 1


def test_outer_shard_codec_strides_use_whole_shard_budget(tmp_path):
    from zarr.codecs import BytesCodec, ShardingCodec, ZstdCodec

    path = tmp_path / 'field.zarr'
    values = np.arange(32 * 32, dtype='f4').reshape(32, 32)
    with pytest.warns(UserWarning, match='disables partial'):
        stored = zarr.create_array(path, shape=values.shape, chunks=(16, 16), dtype='f4',
            serializer=ShardingCodec(chunk_shape=(8, 8), codecs=[BytesCodec()]),
            compressors=[ZstdCodec()])
    stored[:] = values
    backend = ZarristaBackend(zarrista.Array.open(zarrista.store.FilesystemStore(str(path))))
    selection = (slice(1, 32, 3), slice(2, 32, 4))
    reader = ArrayReader(backend, buffer_bytes=2048)
    assert reader.read(selection).tobytes() == values[selection].tobytes()


@pytest.mark.parametrize('sharded', [False, True])
def test_strided_output_is_released_without_cyclic_gc(tmp_path, sharded):
    import gc
    import weakref

    path = tmp_path / 'field.zarr'
    options = {'shards': (8, 8)} if sharded else {}
    zarr.create_array(path, data=np.arange(256, dtype='f4').reshape(16, 16),
                      chunks=(4, 4), **options)
    native = zarrista.Array.open(zarrista.store.FilesystemStore(str(path)))
    reader = ArrayReader(ZarristaBackend(native), buffer_bytes=1 << 20)
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(3):
            output = reader.read((slice(None, None, 2), slice(None, None, 2)))
            reference = weakref.ref(output)
            del output
            assert reference() is None
    finally:
        if enabled:
            gc.enable()
        gc.collect()


@pytest.mark.parametrize('selection', [
    (slice(None, None, 2),) * 3,
    (slice(1, 29, 3), slice(2, 31, 5), slice(3, 27, 7)),
    (3, slice(None, None, 17), slice(2, 30, 19)),
    (slice(1, 32, 31), 13, slice(2, 32, 11)),
])
def test_each_intersected_shard_is_visited_once_without_reordering_values(tmp_path, selection):
    import itertools

    values = np.arange(32**3, dtype='f4').reshape((32,) * 3)
    path = tmp_path / 'field.zarr'
    zarr.create_array(path, data=values, chunks=(4,) * 3, shards=(8,) * 3)
    native = zarrista.Array.open(zarrista.store.FilesystemStore(str(path)))
    visits, misses = [], []

    class Observed:
        def __getattr__(self, name):
            return getattr(native, name)

        def retrieve_subchunk(self, coordinates, *, shard_cache):
            visits.append(tuple(coordinates))
            if shard_cache.is_empty():
                misses.append(tuple(c // 2 for c in coordinates))
            return native.retrieve_subchunk(coordinates, shard_cache=shard_cache)

    result = ArrayReader(ZarristaBackend(Observed())).read(selection)
    assert result.tobytes() == values[selection].tobytes()
    points = itertools.product(*(
        [s] if isinstance(s, int) else range(*s.indices(32)) for s in selection))
    expected_chunks = {tuple(p // 4 for p in point) for point in points}
    expected_shards = {tuple(c // 2 for c in chunk) for chunk in expected_chunks}
    assert len(visits) == len(expected_chunks) and set(visits) == expected_chunks
    assert len(misses) == len(expected_shards) and set(misses) == expected_shards
    assert misses == sorted(misses)


def test_random_stride_phases_and_edge_shards_match_numpy(tmp_path):
    shape = (7, 13, 11, 3)
    rng = np.random.default_rng(817)
    values = rng.integers(0, 2**32, size=shape, dtype='u4').view('f4')
    path = tmp_path / 'field.zarr'
    zarr.create_array(path, data=values, chunks=(2, 3, 2, 1), shards=(4, 6, 4, 3))
    reader = ArrayReader(ZarristaBackend(
        zarrista.Array.open(zarrista.store.FilesystemStore(str(path)))))
    for _ in range(100):
        selection = tuple(
            int(rng.integers(-n, n)) if rng.random() < .25 else
            slice(int(rng.integers(-n, n)), int(rng.integers(0, n + 1)),
                  int(rng.integers(2, n + 3))) for n in shape)
        assert reader.read(selection).tobytes() == values[selection].tobytes()


def test_shard_region_iteration_stays_lazy_for_petabyte_shape():
    import itertools

    from arrayroute import ArraySpec, plan
    from arrayroute.backends import _grid_regions

    spec = ArraySpec((1 << 20, 1024, 1024, 256), (1, 32, 32, 32), 4)
    assert spec.logical_bytes == 1 << 50
    selection = plan(spec, (slice(None, None, 3),) * 4).selection
    first = list(itertools.islice(_grid_regions(selection, (1, 128, 128, 128)), 3))
    assert len(first) == 3
    assert first[0] == (slice(0, 1, 3), slice(0, 128, 3), slice(0, 128, 3), slice(0, 128, 3))
    assert first[1][-1] == slice(129, 256, 3)
    assert first[2][-2] == slice(129, 256, 3)
