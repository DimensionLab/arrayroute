# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.transfer import BlockArray, copy_blocks


def test_strided_source_edges_order_and_bitwise_parity():
    values = np.arange(5 * 7 * 3, dtype=np.float64).reshape(5, 7, 3)[:, ::-1]
    values[0, 0, 0] = -0.0
    values[0, 0, 1] = np.nan
    seen = []
    target = np.empty(values.shape, dtype=np.float32)

    def read(region):
        seen.append(tuple(s.start for s in region))
        return values[region]

    source = BlockArray(values.shape, values.dtype, read)
    count = copy_blocks(
        source,
        lambda r, x: target.__setitem__(r, x),
        block_shape=(2, 3, 3),
        dtype="f4",
        buffer_bytes=216,
    )
    assert count == target.nbytes
    assert seen == sorted(seen) and len(seen) == 9
    assert target.tobytes() == values.astype("f4").tobytes()


def test_insufficient_budget_rejects_before_source_or_sink():
    source = BlockArray((9, 7), np.dtype("f8"), lambda _: pytest.fail("read"))
    with pytest.raises(MemoryError, match="buffer_bytes"):
        copy_blocks(
            source,
            lambda *a: pytest.fail("write"),
            block_shape=(3, 4),
            dtype="f4",
            buffer_bytes=143,
        )


@pytest.mark.parametrize("wrong", ["shape", "dtype"])
def test_source_contract_violation_never_reaches_sink(wrong):
    source = BlockArray(
        (2, 3),
        np.dtype("f8"),
        lambda _: np.ones(
            (3, 2) if wrong == "shape" else (2, 3), dtype="f4" if wrong == "dtype" else "f8"
        ),
    )
    with pytest.raises(ValueError, match="contract"):
        copy_blocks(
            source, lambda *a: pytest.fail("write"), block_shape=(2, 3), dtype="f4", buffer_bytes=72
        )


def test_cancellation_after_read_prevents_write():
    cancelled = False

    def read(region):
        nonlocal cancelled
        cancelled = True
        return np.ones((2, 3), dtype="f4")

    with pytest.raises(InterruptedError):
        copy_blocks(
            BlockArray((2, 3), np.dtype("f4"), read),
            lambda *a: pytest.fail("write"),
            block_shape=(2, 3),
            dtype="f4",
            buffer_bytes=48,
            cancelled=lambda: cancelled,
        )


def test_source_and_sink_exceptions_propagate_without_next_read():
    reads = []

    def read(region):
        reads.append(region)
        return np.ones((1, 3), dtype="f4")

    def fail(*args):
        raise OSError("sink failed")

    with pytest.raises(OSError, match="sink failed"):
        copy_blocks(
            BlockArray((8, 3), np.dtype("f4"), read),
            fail,
            block_shape=(1, 3),
            dtype="f4",
            buffer_bytes=24,
        )
    assert len(reads) == 1


def test_grid_layout_rejects_sub_element_budget_instead_of_looping():
    from arrayroute.layout import GridLayoutPolicy

    with pytest.raises(ValueError, match="one element"):
        GridLayoutPolicy(chunk_bytes=1).plan((2, 3), itemsize=4, spatial_axes=(0, 1))
