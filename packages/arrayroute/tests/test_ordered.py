# SPDX-License-Identifier: Apache-2.0
import hashlib

import numpy as np
import pytest

from arrayroute.backends import NumpyBackend
from arrayroute.ordered import OrderedSelection, read_flat, update_digest
from arrayroute.reader import ArrayReader, ReadStats
from arrayroute.statistics import CompensatedSum


@pytest.mark.parametrize("dtype", ["<f4", ">f8", "i8", "?"])
@pytest.mark.parametrize("size", [1, 17, 65536])
def test_digest_matches_selected_c_order_bytes(dtype, size):
    a = np.arange(7 * 19 * 13).reshape(7, 19, 13).astype(dtype)
    if a.dtype.kind == "f":
        a[1, 1, 1] = -0.0
        a[1, 1, 4] = np.nan
    a = a[:, ::-1]
    selection = (slice(1, 7, 2), slice(1, 19, 3), slice(None, None, 2))
    expected = hashlib.sha256(np.ascontiguousarray(a[selection]).tobytes()).hexdigest()
    reader = ArrayReader(NumpyBackend(a), buffer_bytes=4096)
    digest = hashlib.sha256()
    update_digest(digest, reader, selection, block_elements=size)
    assert digest.hexdigest() == expected


def test_flat_admission_and_cancellation_before_io():
    reader = ArrayReader(NumpyBackend(np.zeros((50, 50), dtype="f8")), buffer_bytes=512)
    source = OrderedSelection.create(reader, None)
    with pytest.raises(MemoryError):
        read_flat(source, 0, 100, ReadStats(), None)
    with pytest.raises(InterruptedError):
        update_digest(hashlib.sha256(), reader, cancelled=lambda: True)
    with pytest.raises(IndexError):
        read_flat(source, -1, 2, ReadStats(), None)


def test_compensation_and_ieee_nonfinite_partials():
    total = CompensatedSum()
    for value in [1e16, 1.0, -1e16]:
        total.add(value)
    assert total.value == 1
    total.add(float("inf"))
    total.add(1.0)
    assert total.value == float("inf")
    total.add(float("-inf"))
    assert np.isnan(total.value)


@pytest.mark.parametrize("byteorder", ["<", ">"])
@pytest.mark.parametrize("selection", [None, (slice(0, 5, 2),), (1,), (-1,)])
def test_digest_preserves_nan_payloads_and_scalar_endianness(byteorder, selection):
    bits = np.array(
        [0x7FC00001, 0x7FA00007, 0xFFC00019, 0x80000000, 0], dtype=f"{byteorder}u4"
    )
    values = bits.view(f"{byteorder}f4")
    # Slice the integer representation so the reference never converts floating NaNs.
    selected_bits = bits if selection is None else bits[selection]
    expected = np.asarray(selected_bits, dtype=bits.dtype).tobytes()
    digest = hashlib.sha256()
    update_digest(
        digest, ArrayReader(NumpyBackend(values), buffer_bytes=128), selection, block_elements=1
    )
    assert digest.hexdigest() == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize("shape", [(), (0,), (5, 0, 3), (129, 32, 16, 3), (5, 7, 11), (1, 1)])
@pytest.mark.parametrize("limit", [1, 7, 64, 1000, 65536])
def test_aligned_blocks_preserve_bounded_flat_coverage(shape, limit):
    import math

    from arrayroute.ordered import flat_blocks, flat_regions

    blocks = list(flat_blocks(shape, limit))
    position = 0
    for start, stop in blocks:
        assert start == position
        assert 0 < stop - start <= limit
        # Alignment turns every block into one rectangle, including a ragged last block.
        assert len(list(flat_regions(shape, start, stop))) == 1
        position = stop
    assert position == math.prod(shape)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_aligned_blocks_reject_invalid_allowance(limit):
    from arrayroute.ordered import flat_blocks

    with pytest.raises(ValueError):
        list(flat_blocks((10,), limit))


def test_default_hash_batches_rows_without_changing_bytes():
    values = np.arange(129 * 32 * 16 * 3, dtype="f4").reshape(129, 32, 16, 3)
    reader = ArrayReader(NumpyBackend(values), buffer_bytes=128 << 10)
    digest, stats = hashlib.sha256(), ReadStats()
    update_digest(digest, reader, stats=stats)
    assert digest.hexdigest() == hashlib.sha256(values.tobytes()).hexdigest()
    assert stats.requests == 13  # ten complete rows per bounded batch
    assert stats.output_bytes == values.nbytes
    scalar = ArrayReader(NumpyBackend(np.array([3.], dtype="f8")), buffer_bytes=8)
    digest = hashlib.sha256()
    update_digest(digest, scalar, (0,))
    assert digest.hexdigest() == hashlib.sha256(np.array(3.).tobytes()).hexdigest()
