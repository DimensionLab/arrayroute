# SPDX-License-Identifier: Apache-2.0
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from arrayroute import ArrayReader, ArraySpec, ReadStats
from arrayroute.budget import ReadBudget
from arrayroute.rows import read_rows


class Observed:
    def __init__(self, values, chunks):
        self.values = values
        self.dtype = values.dtype
        self.spec = ArraySpec(values.shape, chunks, values.dtype.itemsize)
        self.calls = []

    def read(self, selection):
        self.calls.append(selection)
        return self.values[selection].copy()


@pytest.mark.parametrize('coalesce', [False, True])
@pytest.mark.parametrize('dtype', ['<f4', '>f4', '<f8', 'i2'])
def test_reordered_duplicates_negative_indices_and_nan_bits(dtype, coalesce):
    rng = np.random.default_rng(171)
    values = np.frombuffer(rng.bytes(12 * 9 * np.dtype(dtype).itemsize), dtype=dtype)
    values = values.reshape(12, 9)
    source = Observed(values, (2, 3))
    ids = [7, 2, -10, 1, 6, 7, 9]
    actual = read_rows(ArrayReader(source), ids, tail=(slice(1, 9, 2),), coalesce=coalesce)
    assert actual.tobytes() == values[ids, 1:9:2].tobytes()
    assert actual.flags.c_contiguous
    # Only selected unique rows, even when an interval contains multiple rows.
    touched = [i for call in source.calls for i in range(call[0].start, call[0].stop)]
    assert sorted(touched) == [1, 2, 6, 7, 9]
    assert len(source.calls) == (3 if coalesce else 5)


def test_scalar_rows_and_dropped_tail_axes():
    values = np.array([0x7f800001, 0x80000000, 0x7fc01234, 0xff800000], dtype='u4').view('f4')
    actual = read_rows(ArrayReader(Observed(values, (1,))), [2, 0, 1, 2])
    assert actual.tobytes() == values[[2, 0, 1, 2]].tobytes()
    cube = np.arange(120, dtype='f4').reshape(5, 4, 6)
    actual = read_rows(ArrayReader(Observed(cube, (1, 2, 3))), [4, 1], tail=(2, slice(None)))
    assert np.array_equal(actual, cube[[4, 1], 2])


def test_shared_budget_counts_full_output_and_splits_merged_reads():
    values = np.arange(12 * 4, dtype='f4').reshape(12, 4)
    source = Observed(values, (1, 4))
    pool = ReadBudget(144)
    stats = ReadStats()
    result = read_rows(ArrayReader(source, shared_budget=pool), [4, 3, 2, 1], stats=stats)
    assert np.array_equal(result, values[[4, 3, 2, 1]])
    assert stats.requests == 2  # 64-byte output leaves 80 bytes for planned reads.
    assert pool.snapshot()['peak_bytes'] <= 144
    assert pool.snapshot()['used_bytes'] == 0


def test_concurrent_materializations_do_not_nest_or_deadlock():
    values = np.arange(32 * 4, dtype='f4').reshape(32, 4)
    pool = ReadBudget(256)
    barrier = threading.Barrier(2)

    def run(ids):
        barrier.wait(timeout=3)
        return read_rows(ArrayReader(Observed(values, (1, 4)), shared_budget=pool), ids)

    with ThreadPoolExecutor(2) as executor:
        futures = [executor.submit(run, ids) for ids in [[3, 2, 1], [9, 7, 9]]]
        for future, ids in zip(futures, [[3, 2, 1], [9, 7, 9]], strict=True):
            assert np.array_equal(future.result(timeout=3), values[ids])
    assert pool.snapshot()['peak_bytes'] <= 256
    assert pool.snapshot()['used_bytes'] == 0


def test_small_batches_share_capacity_while_native_reads_are_active():
    values = np.arange(32 * 4, dtype='f4').reshape(32, 4)
    pool = ReadBudget(1024)
    barrier = threading.Barrier(2)

    class Concurrent(Observed):
        def read(self, selection):
            barrier.wait(timeout=5)
            assert pool.snapshot()['active'] == 2
            barrier.wait(timeout=5)
            return super().read(selection)

    def run(ids):
        return read_rows(ArrayReader(Concurrent(values, (1, 4)), shared_budget=pool), ids)

    ids = [[3, 2, 1], [9, 7, 9]]
    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(run, ids))
    for result, selected in zip(results, ids, strict=True):
        assert result.tobytes() == values[selected].tobytes()
    # Outputs: 48 + 48; native output + decoded chunks: 96 + 64.
    assert pool.snapshot()['peak_bytes'] == 256
    assert pool.snapshot()['peak_active'] == 2
    assert pool.snapshot()['used_bytes'] == 0


@pytest.mark.parametrize('ids,error', [([12], IndexError), ([-13], IndexError),
                                    ([1.5], TypeError), ([True], TypeError),
                                    ([np.bool_(False)], TypeError)])
def test_invalid_rows_do_not_issue_reads(ids, error):
    source = Observed(np.zeros((12, 4), dtype='f4'), (1, 4))
    with pytest.raises(error):
        read_rows(ArrayReader(source), ids)
    assert not source.calls


def test_empty_outputs_validate_indices_but_do_not_read():
    source = Observed(np.zeros((12, 0), dtype='f4'), (1, 1))
    assert read_rows(ArrayReader(source), [4, 3]).shape == (2, 0)
    with pytest.raises(IndexError):
        read_rows(ArrayReader(source), [12])
    assert read_rows(ArrayReader(source), []).shape == (0, 0)
    assert not source.calls
    empty = Observed(np.zeros((0, 4), dtype='f4'), (1, 4))
    assert read_rows(ArrayReader(empty), []).shape == (0, 4)


def test_oversized_batch_fails_before_index_materialization_or_io():
    source = Observed(np.zeros((12, 4), dtype='f4'), (1, 4))
    with pytest.raises(MemoryError, match='split'):
        read_rows(ArrayReader(source, buffer_bytes=128), range(10**12))
    assert not source.calls


def test_failure_and_cancellation_release_shared_allowance():
    source = Observed(np.zeros((12, 4), dtype='f4'), (1, 4))
    pool = ReadBudget(256)
    reader = ArrayReader(source, shared_budget=pool)
    with pytest.raises(InterruptedError):
        read_rows(reader, [1, 2], cancelled=lambda: True)
    assert not source.calls
    with pytest.raises(InterruptedError):
        read_rows(reader, [1, 2], cancelled=lambda: bool(source.calls))
    assert pool.snapshot()['used_bytes'] == 0
    source.read = lambda selection: np.zeros((3,), dtype='u1')
    with pytest.raises(ValueError, match='shape or dtype'):
        read_rows(reader, [1, 2])
    assert pool.snapshot()['used_bytes'] == 0


@pytest.mark.parametrize('sharded', [False, True])
def test_real_tensorstore_zarr_with_partial_chunks(tmp_path, sharded):
    zarr = pytest.importorskip('zarr')
    ts = pytest.importorskip('tensorstore')
    from arrayroute.backends import TensorStoreBackend

    values = np.random.default_rng(3).normal(size=(19, 3, 5, 7)).astype('f4')
    options = {'shards': (8, 3, 5, 7)} if sharded else {}
    zarr.create_array(tmp_path / 'data', data=values, chunks=(2, 1, 5, 7), **options)
    native = ts.open({'driver': 'zarr3', 'kvstore': {'driver': 'file',
                     'path': str(tmp_path / 'data') + '/'}}, read=True, write=False).result()
    reader = ArrayReader(TensorStoreBackend(native), buffer_bytes=8000)
    ids = [18, 0, 3, 2, 18, 4]
    actual = read_rows(reader, ids, tail=(1, slice(None), slice(None, None, 2)))
    assert actual.tobytes() == values[ids, 1, :, ::2].tobytes()


def test_long_contiguous_batch_does_not_plan_every_growing_prefix(monkeypatch):
    values = np.arange(8192, dtype='f4')
    source = Observed(values, (32,))
    calls = []
    original = ArrayReader.plan

    def observed(self, selection):
        calls.append(selection)
        return original(self, selection)

    monkeypatch.setattr(ArrayReader, 'plan', observed)
    result = read_rows(ArrayReader(source, buffer_bytes=1 << 20), range(len(values)))
    assert result.tobytes() == values.tobytes()
    assert len(source.calls) == 1
    assert len(calls) < 10


@pytest.mark.parametrize('mode', ['native', 'window', 'dense', 'parallel', 'bulk'])
@pytest.mark.parametrize('tail', [(slice(None),), (slice(1, 12, 3),), (3,)])
@pytest.mark.parametrize('stride', [1, 3, 7])
def test_coalescing_prefix_search_matches_exhaustive_budget_boundary(mode, tail, stride):
    from arrayroute.plan import DecodeWindow, GatherWindow
    from arrayroute.rows import _fitting_end

    source = Observed(np.zeros((61, 13), dtype='f4'), (4, 4))
    options = {}
    if mode != 'native':
        options['decode_window'] = DecodeWindow(1, strided_only=mode == 'dense')
    if mode == 'dense':
        options['dense_gather'] = True
    if mode in ('parallel', 'bulk'):
        options['gather_window'] = GatherWindow((8, 8), 4, bulk=mode == 'bulk')
    source.spec = ArraySpec((61, 13), (4, 4), 4, **options)
    source.read_plan = lambda request: source.read(request.selection)
    for begin in (0, 1, 17):
        rows = list(range(begin, 61, stride))
        for allowance in (1, 64, 129, 256, 511, 1024, 4096, 10000):
            reader = ArrayReader(source, buffer_bytes=allowance)
            fits = [reader.plan((slice(begin, row + 1, stride), *tail)).buffer_bytes_upper_bound
                    <= allowance for row in rows]
            assert fits == sorted(fits, reverse=True)
            expected = max([1, *[i + 1 for i, fit in enumerate(fits) if fit]])
            assert _fitting_end(reader, rows, 0, len(rows), tail, stride) == expected


def test_sparse_arithmetic_rows_preserve_order_and_skip_unrequested_payload():
    values = np.arange(1000 * 8, dtype='f4').reshape(1000, 8)
    source = Observed(values, (1, 8))
    ids = [*range(7, 997, 17), 999, 7, 24]
    stats = ReadStats()
    result = read_rows(ArrayReader(source), ids[::-1], stats=stats)
    assert result.tobytes() == values[ids[::-1]].tobytes()
    selected = [i for call in source.calls for i in range(
        call[0].start, call[0].stop, call[0].step,
    )]
    assert selected == sorted(set(ids))
    assert stats.requests == 2


def test_sparse_runs_split_inside_a_small_shared_budget():
    values = np.arange(2000 * 4, dtype='f4').reshape(2000, 4)
    source = Observed(values, (1, 4))
    ids = list(range(3, 1999, 13)) + [16, 3]
    pool = ReadBudget(len(ids) * 16 + 128)
    result = read_rows(ArrayReader(source, shared_budget=pool), ids)
    assert result.tobytes() == values[ids].tobytes()
    assert pool.snapshot()['used_bytes'] == 0
    selected = [i for call in source.calls for i in range(
        call[0].start, call[0].stop, call[0].step,
    )]
    assert selected == sorted(set(ids))


def test_long_runs_split_at_budget_and_gaps_without_reading_unrequested_rows():
    values = np.arange(250 * 4, dtype='f4').reshape(250, 4)
    source = Observed(values, (1, 4))
    ids = [*range(200, 230), *range(1, 96), 9, 211]
    output_bytes = len(ids) * 16
    result = read_rows(ArrayReader(source, buffer_bytes=output_bytes + 128), ids)
    assert result.tobytes() == values[ids].tobytes()
    touched = [i for call in source.calls for i in range(call[0].start, call[0].stop)]
    assert touched == sorted(set(ids))
    assert all(call[0].stop - call[0].start <= 4 for call in source.calls)


def test_single_row_allowance_keeps_planning_linear(monkeypatch):
    values = np.arange(2048, dtype='f4')
    source = Observed(values, (1,))
    calls = 0
    original = ArrayReader.plan

    def observed(self, selection):
        nonlocal calls
        calls += 1
        return original(self, selection)

    monkeypatch.setattr(ArrayReader, 'plan', observed)
    result = read_rows(ArrayReader(source, buffer_bytes=values.nbytes + 8), range(len(values)))
    assert result.tobytes() == values.tobytes()
    assert len(source.calls) == len(values)
    assert calls <= 4 * len(values) + 1
