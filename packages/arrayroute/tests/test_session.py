# SPDX-License-Identifier: Apache-2.0
import pickle

import numpy as np
import pytest
import zarr

from arrayroute.session import LocalReadSession


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore", "zarrista"])
def test_session_reuses_evicts_and_serializes_without_native_handles(tmp_path, engine):
    root = zarr.open_group(tmp_path, mode="w")
    values = np.arange(8 * 6 * 3, dtype=np.float32).reshape(8, 6, 3)
    for name, offset in [("a", 0), ("b", 1000)]:
        root.create_array(name, data=values + offset, chunks=(2, 3, 3))
    session = LocalReadSession(engine, max_open_arrays=1, buffer_bytes=1 << 20)
    selection = (slice(1, 8, 3), slice(1, 5), 2)
    a = session.open(tmp_path, "a")
    np.testing.assert_array_equal(session.read(tmp_path, "a", selection), values[selection])
    assert session.open(tmp_path, "a") is a
    np.testing.assert_array_equal(
        session.read(tmp_path, "b", selection), (values + 1000)[selection]
    )
    assert len(session._readers) == 1
    assert session.open(tmp_path, "a") is not a
    clone = pickle.loads(pickle.dumps(session))
    assert not clone._readers and clone._context is None
    np.testing.assert_array_equal(clone.read(tmp_path, "a", selection), values[selection])
    session.clear()
    assert not session._readers and session._context is None


def test_session_shares_native_context_and_does_not_silently_change_engine(tmp_path):
    root = zarr.open_group(tmp_path, mode="w")
    for name in ["a", "b"]:
        root.create_array(name, data=np.arange(5), chunks=(3,))
    session = LocalReadSession("tensorstore", tensorstore_cache_bytes=1024)
    session.open(tmp_path, "a")
    context = session._context
    session.open(tmp_path, "b")
    assert session._context is context
    with pytest.raises(ValueError, match="Unknown"):
        LocalReadSession("typo")
    with pytest.raises(ValueError, match="requires"):
        LocalReadSession("zarrista", tensorstore_cache_bytes=1)
    with pytest.raises(ValueError):
        LocalReadSession(buffer_bytes=0)
    with pytest.raises(TypeError):
        LocalReadSession(max_open_arrays=1.5)
    with pytest.raises((FileNotFoundError, KeyError, ValueError)):
        session.open(tmp_path, "absent")


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore", "zarrista"])
@pytest.mark.parametrize("coalesce", [False, True])
def test_session_row_batches_reuse_handles_and_preserve_order(tmp_path, engine, coalesce):
    from arrayroute import ReadStats
    from arrayroute.budget import ReadBudget

    values = np.arange(12 * 6 * 3, dtype=np.float32).reshape(12, 6, 3)
    root = zarr.open_group(tmp_path, mode="w")
    root.create_array("a", data=values, chunks=(2, 3, 3))
    budget = ReadBudget(1 << 20)
    session = LocalReadSession(engine, buffer_bytes=1 << 20, read_budget=budget)
    handle = session.open(tmp_path, "a")
    indices = [8, 2, 3, 2, -1, 4]
    stats = ReadStats()
    result = session.read_rows(
        tmp_path, "a", indices, tail=(slice(1, 6, 2), 1), coalesce=coalesce, stats=stats,
    )
    assert result.tobytes() == values[indices, 1:6:2, 1].tobytes()
    assert session.open(tmp_path, "a") is handle
    assert stats.requests > 0
    assert budget.snapshot()["used_bytes"] == 0
    with pytest.raises(InterruptedError):
        session.read_rows(tmp_path, "a", indices, cancelled=lambda: True)
    assert budget.snapshot()["used_bytes"] == 0


@pytest.mark.parametrize('engine', ['zarr-python', 'tensorstore', 'zarrista'])
def test_boolean_masks_with_parameterless_codec_metadata(tmp_path, engine):
    root = zarr.open_group(tmp_path, mode='w')
    values = (np.arange(9 * 7).reshape(9, 7) % 3) == 0
    root.create_array('mask', data=values, chunks=(3, 4))
    session = LocalReadSession(engine, tensorstore_threads=2)
    selection = (slice(1, 8, 2), slice(2, 6))
    result = session.read(tmp_path, 'mask', selection)
    assert result.dtype == values.dtype
    assert result.tobytes() == values[selection].tobytes()
    gathered = session.read_rows(tmp_path, 'mask', [7, 1, 7, -1])
    assert gathered.tobytes() == values[[7, 1, 7, -1]].tobytes()
