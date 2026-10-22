# SPDX-License-Identifier: Apache-2.0
import gc
import math
import weakref
from dataclasses import replace

import numpy as np
import pytest

from arrayroute import ArrayReader, ArraySpec, DecodeWindow, ReadStats, plan, tiles
from arrayroute.budget import ReadBudget
from arrayroute.concurrency import plan_concurrency
from arrayroute.session import LocalReadSession


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1"])
def test_invalid_decode_window(value):
    with pytest.raises(ValueError):
        DecodeWindow(value)


def test_decode_window_requires_explicit_types():
    with pytest.raises(TypeError):
        DecodeWindow(1, strided_only=1)
    with pytest.raises(TypeError):
        ArraySpec((10,), (2,), 4, decode_window=1)


@pytest.mark.parametrize("max_chunks", [1, 3, 1000])
def test_total_decode_work_is_independent_of_peak_live_memory(max_chunks):
    spec = ArraySpec((31, 45), (4, 8), 4, DecodeWindow(max_chunks, strided_only=True))
    selection = (slice(1, 31, 3), slice(0, 45, 5))
    capped = plan(spec, selection)
    conservative = plan(replace(spec, decode_window=None), selection)
    assert capped.decoded_bytes_upper_bound == conservative.decoded_bytes_upper_bound
    assert capped.decode_amplification == conservative.decode_amplification
    assert capped.peak_decoded_bytes_upper_bound == min(
        capped.chunks_touched, max_chunks) * 4 * 8 * 4
    assert capped.buffer_bytes_upper_bound == (
        capped.output_bytes + capped.peak_decoded_bytes_upper_bound)
    contiguous = plan(spec, (slice(None), slice(None)))
    assert contiguous.peak_decoded_bytes_upper_bound == contiguous.decoded_bytes_upper_bound
    scalar = plan(spec, (2, 3))
    assert scalar.peak_decoded_bytes_upper_bound == 128
    empty = plan(spec, (slice(0, 0, 2), slice(None)))
    assert empty.buffer_bytes_upper_bound == empty.decoded_bytes_upper_bound == 0


def test_petabyte_metadata_and_concurrency_use_live_window_without_enumeration():
    spec = ArraySpec((1 << 20, 1024, 1024, 256), (1, 64, 64, 64), 4,
                     DecodeWindow(1, strided_only=True))
    assert spec.logical_bytes == 1 << 50
    selection = (900_000, slice(None, None, 16), slice(None, None, 16),
                 slice(None, None, 16))
    request = plan(spec, selection)
    assert request.decoded_bytes_upper_bound == 1 << 30
    assert request.peak_decoded_bytes_upper_bound == 1 << 20
    assert request.output_bytes == 1 << 18
    choices = plan_concurrency(spec, selection, total_buffer_bytes=16 << 20, max_workers=8)
    assert choices.max_workers_without_tiling == 8
    assert all(c.materialization == "direct" for c in choices.candidates)
    conservative = plan_concurrency(replace(spec, decode_window=None), selection,
                                    total_buffer_bytes=16 << 20, max_workers=8)
    assert conservative.max_workers_without_tiling == 0
    assert len(list(tiles(spec, selection, buffer_bytes=2 << 20))) == 1


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("selection", [
    (slice(1, 13, 2), slice(0, 17, 3), slice(2, 11, 4)),
    (3, slice(1, 17, 2), slice(0, 11, 3)),
    (slice(None, None, 3), 6, slice(None)),
])
def test_native_partial_shards_release_decoded_chunk_before_successor(tmp_path, missing, selection):
    import zarr

    shape, chunks = (13, 17, 11), (2, 3, 2)
    original = np.arange(math.prod(shape), dtype="f4").reshape(shape)
    path = tmp_path / "fields.zarr"
    array = zarr.create_array(path, shape=shape, dtype="f4", chunks=chunks,
                              shards=(4, 6, 4), fill_value=-2)
    if missing:
        original.fill(-2)
    else:
        array[:] = original
    reader = LocalReadSession("zarrista").open(path, "")
    backend, refs = reader.backend, []
    native = backend.array

    class ObservedArray:
        def __getattr__(self, name):
            return getattr(native, name)

        def retrieve_subchunk(self, coordinates, *, shard_cache):
            assert all(ref() is None for ref in refs), "Decoded predecessor remains alive"
            value = np.asarray(native.retrieve_subchunk(coordinates, shard_cache=shard_cache))
            assert value.nbytes <= math.prod(chunks) * 4
            assert shard_cache.size() <= 1
            refs.append(weakref.ref(value))
            return value

    backend.array = ObservedArray()
    request = reader.plan(selection)
    pool = ReadBudget(request.buffer_bytes_upper_bound)
    reader = ArrayReader(backend, buffer_bytes=pool.max_bytes, shared_budget=pool)
    stats = ReadStats()
    enabled = gc.isenabled()
    gc.disable()
    try:
        actual = reader.read(selection, stats=stats)
        assert actual.tobytes() == original[selection].tobytes()
        assert len(refs) == request.chunks_touched
        assert all(ref() is None for ref in refs)
    finally:
        if enabled:
            gc.enable()
    assert stats.requests == 1
    assert stats.decoded_bytes_upper_bound == request.decoded_bytes_upper_bound
    assert pool.snapshot()["peak_bytes"] == request.buffer_bytes_upper_bound
    assert pool.snapshot()["used_bytes"] == 0


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore", "zarrista"])
@pytest.mark.parametrize("sharded", [False, True])
def test_only_native_partial_shards_have_a_decode_window(tmp_path, engine, sharded):
    import zarr

    path = tmp_path / "data.zarr"
    zarr.create_array(path, shape=(32, 32), dtype="f4", chunks=(4, 4),
                      shards=(16, 16) if sharded else None)
    reader = LocalReadSession(engine).open(path, "")
    assert (reader.backend.spec.decode_window is not None) == (sharded and engine == "zarrista")
    request = reader.plan((slice(None), slice(None)))
    assert request.peak_decoded_bytes_upper_bound == request.decoded_bytes_upper_bound


@pytest.mark.parametrize("prefetch", [False, True])
def test_live_window_tiles_reconstruct_large_selection_under_small_budget(tmp_path, prefetch):
    from contextlib import nullcontext

    import zarr

    path = tmp_path / "data.zarr"
    values = np.arange(64 * 64, dtype="f4").reshape(64, 64)
    zarr.create_array(path, data=values, chunks=(4, 4), shards=(16, 16))
    pool = ReadBudget(512)
    reader = LocalReadSession("zarrista", buffer_bytes=512, read_budget=pool).open(path, "")
    selection = (slice(None, None, 3), slice(None, None, 2))
    expected = values[selection]
    actual, coverage = np.empty_like(expected), np.zeros(expected.shape, dtype="u1")
    source = reader.prefetch(selection) if prefetch else nullcontext(reader.stream(selection))
    with source as stream:
        for tile, value in stream:
            assert tile.plan.buffer_bytes_upper_bound <= (256 if prefetch else 512)
            assert tile.plan.peak_decoded_bytes_upper_bound == 4 * 4 * 4
            actual[tile.destination] = value
            coverage[tile.destination] += 1
            del value
    assert actual.tobytes() == expected.tobytes()
    assert np.all(coverage == 1)
    assert pool.snapshot()["used_bytes"] == 0
    assert pool.snapshot()["peak_bytes"] <= 512
