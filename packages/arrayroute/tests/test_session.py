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
