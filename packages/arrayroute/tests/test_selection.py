# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
from dataclasses import asdict

import numpy as np
import pytest

from arrayroute import ArrayReader, ArraySpec, ReadStats, plan, tiles
from arrayroute.backends import ZarrBackend


class MemoryBackend:
    def __init__(self, array, chunks):
        self.array, self.dtype = array, array.dtype
        self.spec = ArraySpec(array.shape, chunks, array.dtype.itemsize)
        self.calls = 0

    def read(self, selection):
        self.calls += 1
        return np.asarray(self.array[selection]).copy()


def test_costs_match_enumerated_chunks_for_strided_selections():
    rng = np.random.default_rng(73)
    for _ in range(1500):
        size, chunk = int(rng.integers(1, 200)), int(rng.integers(1, 25))
        a, b, stride = (
            int(rng.integers(-200, 200)),
            int(rng.integers(-200, 200)),
            int(rng.integers(1, 40)),
        )
        selection = slice(a, b, stride)
        indices = range(*selection.indices(size))
        result = plan(ArraySpec((size,), (chunk,), 4), (selection,))
        assert result.output_shape == (len(indices),)
        assert result.chunks_touched == len({i // chunk for i in indices})


def test_petabyte_plan_is_constant_size():
    spec = ArraySpec((1 << 20, 1024, 1024, 256), (1, 64, 64, 64), 4)
    assert spec.logical_bytes == 1 << 50
    p = plan(spec, (900_000, slice(22, 86), slice(40, 104), 12))
    assert p.output_bytes == 64 * 64 * 4
    assert p.chunks_touched == 4
    assert len(str(asdict(p))) < 1000
    # Laziness: only the first tile of a full-PB traversal is requested.
    first = next(tiles(spec, (slice(None),) * 4, buffer_bytes=8 << 20))
    assert first.plan.buffer_bytes_upper_bound <= 8 << 20


def test_splits_reduce_decoded_chunks_before_subdividing_spatial_cells():
    spec = ArraySpec((20, 32, 32, 32, 1), (1, 32, 32, 32, 1), 4)
    regions = list(itertools.islice(tiles(spec, (slice(None),) * 5, buffer_bytes=1 << 20), 20))
    assert len(regions) == 8
    assert sum(t.plan.output_bytes for t in regions) == spec.logical_bytes
    assert sum(t.plan.decoded_bytes_upper_bound for t in regions) == spec.logical_bytes
    assert all(t.plan.buffer_bytes_upper_bound <= 1 << 20 for t in regions)


@pytest.mark.parametrize(
    "selection",
    [
        (slice(None), slice(None), slice(None)),
        (slice(1, 9, 2), slice(None, None, 3), 2),
        (-1, slice(2, 8), slice(None, None, 2)),
        (slice(0, 0), slice(None), slice(None)),
        (1, 2, 3),
    ],
)
def test_stream_reconstructs_numpy_exactly(selection):
    original = np.arange(10 * 9 * 7, dtype=np.float32).reshape(10, 9, 7)
    backend = MemoryBackend(original, (2, 3, 2))
    reader = ArrayReader(backend, buffer_bytes=160)
    expected = original[selection]
    actual, coverage = np.empty_like(expected), np.zeros(expected.shape, dtype=np.uint8)
    stats = ReadStats()
    for tile, value in reader.stream(selection, stats=stats):
        actual[tile.destination] = value
        coverage[tile.destination] += 1
        assert tile.plan.buffer_bytes_upper_bound <= 160
    assert actual.tobytes() == expected.tobytes()
    assert np.all(coverage == 1)
    assert stats.output_bytes == expected.nbytes


def test_materialize_guard_and_cancellation_do_not_issue_io():
    backend = MemoryBackend(np.zeros((100, 100), dtype=np.float32), (10, 10))
    reader = ArrayReader(backend, buffer_bytes=1000)
    with pytest.raises(MemoryError):
        reader.read((slice(None), slice(None)))
    with pytest.raises(InterruptedError):
        next(reader.stream((slice(None), slice(None)), cancelled=lambda: True))
    assert backend.calls == 0
    with pytest.raises(MemoryError, match="decoded chunk"):
        next(ArrayReader(backend, buffer_bytes=10).stream((0, 0)))


def test_materialization_subdivides_decode_amplification():
    original = np.arange(10000, dtype=np.int32).reshape(100, 100)
    reader = ArrayReader(MemoryBackend(original, (10, 10)), buffer_bytes=2000)
    stats = ReadStats()
    assert np.array_equal(reader.read((slice(None), 2), stats=stats), original[:, 2])
    assert stats.requests > 1


@pytest.mark.parametrize(
    "shape,chunks,itemsize",
    [((2,), (0,), 4), ((-1,), (2,), 4), ((2,), (2, 3), 4), ((2,), (2,), 0), ((True,), (2,), 4)],
)
def test_invalid_spec(shape, chunks, itemsize):
    with pytest.raises(ValueError):
        ArraySpec(shape, chunks, itemsize)


def test_invalid_selection():
    spec = ArraySpec((4, 5), (2, 2), 4)
    for selection, error in [
        ((slice(None),), ValueError),
        ((4, 0), IndexError),
        ((True, 0), TypeError),
        ((slice(None, None, -1), 0), ValueError),
    ]:
        with pytest.raises(error):
            plan(spec, selection)


def test_zarr_shards_and_exceptional_float_bits(tmp_path):
    import zarr

    bits = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7FC12345], dtype="u4")
    original = np.tile(bits.view("f4"), (8, 1))
    array = zarr.create_array(tmp_path / "test.zarr", data=original, chunks=(2, 2), shards=(4, 6))
    reader = ArrayReader(ZarrBackend(array), buffer_bytes=80)
    result = np.empty_like(original)
    for tile, value in reader.stream((slice(None), slice(None))):
        result[tile.destination] = value
    assert result.tobytes() == original.tobytes()


def test_multidimensional_chunk_count():
    spec = ArraySpec((13, 17, 9), (3, 4, 2), 4)
    selection = (slice(1, 13, 4), slice(2, 17, 7), slice(0, 9, 2))
    expected = {
        tuple(x // c for x, c in zip(point, spec.chunks, strict=True))
        for point in itertools.product(
            *(range(*s.indices(n)) for s, n in zip(selection, spec.shape, strict=True))
        )
    }
    assert plan(spec, selection).chunks_touched == len(expected)


@pytest.mark.parametrize("engine", ["tensorstore", "zarrista"])
def test_native_backends_match_numpy_strides_and_exceptional_bits(tmp_path, engine):
    pytest.importorskip(engine)
    import zarr

    from arrayroute.backends import TensorStoreBackend, ZarristaBackend

    rng = np.random.default_rng(991)
    original = rng.integers(0, 2**32, size=(7, 9, 11), dtype=np.uint32).view(np.float32)
    path = tmp_path / "native.zarr"
    zarr.create_array(path, data=original, chunks=(2, 3, 2), shards=(4, 6, 4))
    if engine == "tensorstore":
        import tensorstore as ts

        backend = TensorStoreBackend(
            ts.open(
                {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path) + "/"}}
            ).result()
        )
    else:
        import zarrista

        backend = ZarristaBackend(zarrista.Array.open(zarrista.store.FilesystemStore(str(path))))
    reader = ArrayReader(backend, buffer_bytes=300)
    for selection in [(2, slice(1, 8, 2), slice(0, 10, 3)), (slice(1, 6, 2), 4, slice(2, 11, 4))]:
        assert reader.read(selection).tobytes() == original[selection].tobytes()


@pytest.mark.parametrize('engine', ['zarr-python', 'zarrista'])
def test_outer_shard_codec_is_charged_as_whole_shard(tmp_path, engine):
    import zarr
    from zarr.codecs import BytesCodec, ShardingCodec, ZstdCodec

    from arrayroute.session import LocalReadSession

    path = tmp_path / "outer.zarr"
    with pytest.warns(UserWarning, match="disables partial"):
        array = zarr.create_array(
            path,
            shape=(32, 32),
            chunks=(16, 16),
            dtype="f4",
            serializer=ShardingCodec(chunk_shape=(8, 8), codecs=[BytesCodec()]),
            compressors=[ZstdCodec()],
        )
    array[:] = 1
    backend = LocalReadSession(engine).open(path, '').backend
    assert backend.spec.chunks == (16, 16)
    assert backend.spec.decode_window is None
    with pytest.raises(MemoryError):
        ArrayReader(backend, buffer_bytes=512).read((0, 0))
    assert ArrayReader(backend, buffer_bytes=2048).read((0, 0)) == np.float32(1)


@pytest.mark.parametrize('engine', ['zarr-python', 'zarrista'])
def test_outer_transpose_preserves_axes_and_charges_whole_shard(tmp_path, engine):
    import zarr
    from zarr.codecs import BytesCodec, ShardingCodec, TransposeCodec

    from arrayroute.session import LocalReadSession

    path = tmp_path / 'transposed.zarr'
    with pytest.warns(UserWarning, match='disables partial'):
        array = zarr.create_array(path, shape=(32, 64), chunks=(16, 32), dtype='f4',
            filters=[TransposeCodec(order=(1, 0))], compressors=[],
            serializer=ShardingCodec(chunk_shape=(8, 4), codecs=[BytesCodec()]))
    values = np.arange(32 * 64, dtype='f4').reshape(32, 64)
    array[:] = values
    backend = LocalReadSession(engine).open(path, '').backend
    assert backend.spec.chunks == (16, 32)
    assert backend.spec.decode_window is None
    with pytest.raises(MemoryError):
        ArrayReader(backend, buffer_bytes=1024).read((7, 25))
    assert ArrayReader(backend, buffer_bytes=4096).read((7, 25)) == values[7, 25]
