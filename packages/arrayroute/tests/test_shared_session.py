# SPDX-License-Identifier: Apache-2.0
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
import zarr

from arrayroute.session import TensorStoreReadGroup


def test_native_cache_is_shared_but_handle_caches_are_private(tmp_path):
    values = np.arange(32, dtype=np.float32)
    source = zarr.create_array(tmp_path / "a", data=values, chunks=(32,))
    group = TensorStoreReadGroup(cache_bytes=1 << 20, threads=2, immutable=True)
    first, second = group.session(), group.session()
    np.testing.assert_array_equal(first.read(tmp_path, "a", (slice(None),)), values)
    # Deliberately violate immutability to distinguish cached data from a fresh read.
    source[:] = values + 100
    np.testing.assert_array_equal(second.read(tmp_path, "a", (slice(None),)), values)
    assert first._context is second._context
    assert first._readers is not second._readers
    assert first.open(tmp_path, "a") is not second.open(tmp_path, "a")
    first.clear()
    np.testing.assert_array_equal(first.read(tmp_path, "a", (slice(None),)), values)
    fresh = TensorStoreReadGroup(cache_bytes=1 << 20, immutable=True).session()
    np.testing.assert_array_equal(fresh.read(tmp_path, "a", (slice(None),)), values + 100)


def test_concurrent_sessions_keep_exact_order_and_independent_outputs(tmp_path):
    values = np.arange(64 * 16, dtype=np.float32).reshape(64, 16)
    source = zarr.open_group(tmp_path, mode="w")
    source.create_array("a", data=values, chunks=(4, 16))
    source.create_array("b", data=values + 10000, chunks=(4, 16))
    group = TensorStoreReadGroup(cache_bytes=16384, threads=4, immutable=True)
    barrier = Barrier(4)

    def worker(number):
        session = group.session(buffer_bytes=2048, max_open_arrays=1)
        barrier.wait(timeout=10)
        for iteration in range(20):
            name = "a" if iteration % 2 == 0 else "b"
            selection = (slice(number, 64, 7), slice(1, 15, 3))
            output = session.read(tmp_path, name, selection)
            expected = (values + (0 if name == "a" else 10000))[selection]
            np.testing.assert_array_equal(output, expected)
            output[:] = -1
            assert len(session._readers) == 1
        return session

    with ThreadPoolExecutor(max_workers=4) as executor:
        sessions = list(executor.map(worker, range(4)))
    assert len({id(s._readers) for s in sessions}) == 4
    assert len({id(s._context) for s in sessions}) == 1


def test_mutable_source_rechecks_and_missing_reads_do_not_poison_group(tmp_path):
    source = zarr.create_array(tmp_path / "a", data=np.arange(8), chunks=(8,))
    group = TensorStoreReadGroup(cache_bytes=1024)
    first, second = group.session(), group.session()
    first.read(tmp_path, "a", (slice(None),))
    with pytest.raises(ValueError):
        second.read(tmp_path, "missing", (slice(None),))
    source[:] = np.arange(8) + 30
    np.testing.assert_array_equal(second.read(tmp_path, "a", (slice(None),)), np.arange(8) + 30)


def test_group_rejects_cross_process_reuse_and_serialization(monkeypatch):
    group = TensorStoreReadGroup()
    session = group.session()
    for value in (group, session):
        with pytest.raises(TypeError, match="in-process"):
            pickle.dumps(value)
    monkeypatch.setattr("arrayroute.session.os.getpid", lambda: group._pid + 1)
    with pytest.raises(RuntimeError, match="consuming process"):
        group.session()
    with pytest.raises(RuntimeError, match="consuming process"):
        session.open(".", "missing")


@pytest.mark.parametrize("options,error", [
    ({"cache_bytes": -1}, ValueError), ({"threads": 0}, ValueError),
    ({"threads": 1.5}, TypeError), ({"threads": True}, TypeError),
    ({"immutable": "false"}, TypeError),
])
def test_group_validates_resource_policy(options, error):
    with pytest.raises(error):
        TensorStoreReadGroup(**options)
