# SPDX-License-Identifier: Apache-2.0
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from arrayroute import ArrayReader, ArraySpec, ReadStats
from arrayroute.backends import NumpyBackend
from arrayroute.budget import ReadBudget
from arrayroute.indexed import take_flat
from arrayroute.session import LocalReadSession


class Observed:
    def __init__(self, values, chunks):
        self.values = values
        self.dtype = values.dtype
        self.spec = ArraySpec(values.shape, chunks, values.dtype.itemsize)
        self.calls = []

    def read(self, selection):
        self.calls.append(selection)
        return self.values[selection].copy()


@pytest.mark.parametrize("dtype", ["<f4", ">f4", "<f8", "i2"])
@pytest.mark.parametrize("amplification", [1.0, 2.0])
@pytest.mark.parametrize("selection", [None, (2, slice(1, 9, 2), slice(None)),
                                       (slice(0, 7, 3), 4, slice(2, 11, 3))])
def test_bit_identity_with_reordered_duplicates_and_selected_views(dtype, selection, amplification):
    rng = np.random.default_rng(19)
    values = np.frombuffer(rng.bytes(7 * 9 * 11 * np.dtype(dtype).itemsize), dtype=dtype)
    values = values.reshape(7, 9, 11)
    view = values if selection is None else values[selection]
    ids = [0, -1, 2, 0, -3, 5]
    backend = Observed(values, (2, 3, 4))
    actual = take_flat(ArrayReader(backend, buffer_bytes=4096), ids, selection=selection,
                       max_chunk_amplification=amplification)
    assert actual.tobytes() == view.reshape(-1)[ids].tobytes()
    assert actual.dtype == values.dtype


def test_sparse_contiguous_source_does_not_scan_empty_space():
    values = np.arange(100000, dtype="float32")
    backend = NumpyBackend(values)
    stats = ReadStats()
    actual = take_flat(ArrayReader(backend, buffer_bytes=64), [99999, 0, 99999], stats=stats)
    np.testing.assert_array_equal(actual, [99999, 0, 99999])
    assert stats.requests == 2
    assert stats.output_bytes == 8


def test_duplicate_point_is_fetched_once_and_scalar_view_works():
    backend = Observed(np.arange(12, dtype="int32").reshape(3, 4), (1, 2))
    actual = take_flat(ArrayReader(backend), [0, -1, 0], selection=(1, 2))
    np.testing.assert_array_equal(actual, [6, 6, 6])
    assert len(backend.calls) == 1


def test_coalescing_preserves_bits_without_reading_unoccupied_chunks():
    values = np.arange(64, dtype="float32").reshape(8, 8)
    backend = Observed(values, (4, 4))
    ids = [0, 4, 32, 36, 0]
    assert take_flat(ArrayReader(backend), ids).tobytes() == values.ravel()[ids].tobytes()
    assert len(backend.calls) == 1
    backend.calls.clear()
    ids = [0, 36]
    np.testing.assert_array_equal(take_flat(ArrayReader(backend), ids), values.ravel()[ids])
    assert len(backend.calls) == 2


@pytest.mark.parametrize("factor", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("budget", [4096, 65536])
def test_grouped_reads_bound_decode_amplification_and_memory(factor, budget):
    rng = np.random.default_rng(111)
    values = rng.normal(size=(19, 27, 23)).astype("float32")
    backend = Observed(values, (4, 5, 6))
    ids = rng.choice(values.size, 93, replace=False)
    reader = ArrayReader(backend, buffer_bytes=budget)
    stats = ReadStats()
    result = take_flat(reader, ids, stats=stats, max_chunk_amplification=factor)
    assert result.tobytes() == values.ravel()[ids].tobytes()
    coordinates = np.array(np.unravel_index(ids, values.shape)).T
    occupied = len(np.unique(coordinates // np.array(backend.spec.chunks), axis=0))
    assert stats.planned_chunk_visits <= factor * occupied
    for call in backend.calls:
        assert reader.plan(call).buffer_bytes_upper_bound + 2 * result.nbytes <= budget


def test_tight_budget_coalesces_subregions_instead_of_individual_chunks():
    values = np.arange(32 * 32, dtype="float32").reshape(32, 32)
    backend = Observed(values, (4, 4))
    ids = np.array([i * 32 + j for i in range(0, 32, 4) for j in range(0, 32, 4)])
    result = take_flat(ArrayReader(backend, buffer_bytes=2048), ids)
    assert result.tobytes() == values.ravel()[ids].tobytes()
    assert 1 < len(backend.calls) < 64


@pytest.mark.parametrize("factor", [0, 0.9, True, float("inf"), float("nan")])
def test_invalid_amplification_fails_before_read(factor):
    backend = Observed(np.arange(16), (4,))
    with pytest.raises(ValueError, match="amplification"):
        take_flat(ArrayReader(backend), [0], max_chunk_amplification=factor)
    assert backend.calls == []


def test_logical_shapes_larger_than_numpy_index_space_remain_addressable():
    class Procedural:
        dtype = np.dtype("uint64")
        spec = ArraySpec((1 << 32, 1 << 32), (1, 1), 8)

        def read(self, selection):
            rows = np.arange(*selection[0].indices(self.spec.shape[0]), dtype="uint64")
            cols = np.arange(*selection[1].indices(self.spec.shape[1]), dtype="uint64")
            return rows[:, None] + cols[None, :]

    result = take_flat(ArrayReader(Procedural(), buffer_bytes=256), [0, (1 << 64) - 1, 0])
    np.testing.assert_array_equal(result, np.array([0, (1 << 33) - 2, 0], dtype="uint64"))


@pytest.mark.parametrize("indices,error", [([True], TypeError), ([np.bool_(0)], TypeError),
                                          ([1.2], TypeError), ([100], IndexError),
                                          ([-101], IndexError), (np.ones((2, 2)), ValueError)])
def test_bad_indices_fail_before_payload(indices, error):
    backend = Observed(np.zeros((10, 10), dtype="float32"), (2, 2))
    with pytest.raises(error):
        take_flat(ArrayReader(backend), indices)
    assert backend.calls == []


def test_chunk_admission_and_empty_selection():
    backend = Observed(np.zeros((100, 100), dtype="float32"), (100, 100))
    assert take_flat(ArrayReader(backend, buffer_bytes=64), []).shape == (0,)
    with pytest.raises(MemoryError, match="chunk"):
        take_flat(ArrayReader(backend, buffer_bytes=64), [0])
    assert backend.calls == []


def test_shared_budget_concurrency_and_cancellation_release():
    values = np.arange(100, dtype="float32").reshape(10, 10)
    pool = ReadBudget(256)
    barrier = threading.Barrier(2)

    def run(ids):
        barrier.wait(timeout=3)
        return take_flat(ArrayReader(Observed(values, (2, 2)), shared_budget=pool), ids)

    with ThreadPoolExecutor(2) as executor:
        a, b = executor.submit(run, [0, 12, 99]), executor.submit(run, [99, 0, 99])
        np.testing.assert_array_equal(a.result(timeout=3), [0, 12, 99])
        np.testing.assert_array_equal(b.result(timeout=3), [99, 0, 99])
    backend = Observed(values, (2, 2))
    with pytest.raises(InterruptedError):
        take_flat(ArrayReader(backend, shared_budget=pool), [0, 99],
                  cancelled=lambda: len(backend.calls) > 0)
    assert pool.snapshot()["used_bytes"] == 0
    assert pool.snapshot()["peak_bytes"] <= 256


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore"])
def test_real_chunked_engines(tmp_path, engine):
    import zarr

    values = np.arange(7 * 9 * 11, dtype="float32").reshape(7, 9, 11)
    group = zarr.open_group(str(tmp_path / "data.zarr"), mode="w")
    group.create_array("u", data=values, chunks=(2, 3, 4))
    reader = LocalReadSession(engine, buffer_bytes=1024, immutable=True).open(
        tmp_path / "data.zarr", "u")
    ids = np.array([4, 7, -1, 0, 4])
    result = take_flat(reader, ids, selection=(slice(1, 7, 2), slice(None), 2))
    np.testing.assert_array_equal(result, values[1:7:2, :, 2].reshape(-1)[ids])
