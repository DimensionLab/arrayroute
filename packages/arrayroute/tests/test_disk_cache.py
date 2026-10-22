# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import os
import pickle
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from arrayroute.disk_cache import ArrayIdentity, TileCache
from arrayroute.plan import ArraySpec
from arrayroute.reader import ArrayReader


class Source:
    def __init__(self, value):
        self.value = value
        self.dtype = value.dtype
        self.spec = ArraySpec(value.shape, tuple(min(16, n) for n in value.shape), value.itemsize)
        self.calls = 0
        self.fail = False
        self.delay = 0.0

    def read(self, selection):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Origin unavailable")
        time.sleep(self.delay)
        return self.value[selection].copy()


def identity(revision="snapshot-a", representation="SI-v1", source="velocity"):
    return ArrayIdentity(source, revision, representation)


def open_cache(path, capacity=4 << 20, entry=1 << 20):
    return TileCache(path, capacity_bytes=capacity, max_entry_bytes=entry)


def test_warm_restart_normalized_selection_and_mutation(tmp_path):
    values = np.arange(64 * 64, dtype=">f4").reshape(64, 64)
    source = Source(values)
    cache = open_cache(tmp_path)
    reader = ArrayReader(cache.backend(source, identity()))
    selection = (slice(1, 24, 2), -1)
    np.testing.assert_array_equal(reader.read(selection), values[selection])
    reader.read(selection)[:] = -4
    cache.close()
    cache = open_cache(tmp_path)
    backend = cache.backend(source, identity())
    source.fail = True
    np.testing.assert_array_equal(backend.read((slice(1, 24, 2), 63)), values[selection])
    assert source.calls == 1 and backend.stats.hits == 1
    clone = pickle.loads(pickle.dumps(cache))
    np.testing.assert_array_equal(
        clone.backend(source, identity()).read(selection), values[selection])
    clone.close()
    cache.close()


@pytest.mark.parametrize("change", ["revision", "representation", "source"])
def test_identity_change_cannot_reuse_stale_data(tmp_path, change):
    source = Source(np.arange(256, dtype=np.float32))
    cache = open_cache(tmp_path)
    cache.backend(source, identity()).read((slice(None),))
    changed = {change: "different"}
    source.value[:] = 500
    backend = cache.backend(source, identity(**changed))
    np.testing.assert_array_equal(backend.read((slice(None),)), source.value)
    assert source.calls == 2 and backend.stats.misses == 1
    cache.close()


@pytest.mark.parametrize("corruption", ["truncated", "flipped", "oversize", "pickle-mode"])
def test_corrupt_entries_are_reloaded_without_pickle(tmp_path, corruption, monkeypatch):
    source = Source(np.arange(256, dtype=np.float32))
    cache = open_cache(tmp_path)
    backend = cache.backend(source, identity())
    backend.read((slice(None),))
    with sqlite3.connect(tmp_path / "data" / "cache.db") as db:
        key, filename = db.execute("SELECT key, filename FROM Cache").fetchone()
        if corruption == "pickle-mode":
            db.execute("UPDATE Cache SET mode=4 WHERE key=?", (key,))
        else:
            file = tmp_path / "data" / filename
            payload = file.read_bytes()
            if corruption == "truncated":
                file.write_bytes(payload[:10])
            elif corruption == "flipped":
                file.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
            else:
                file.write_bytes(b"x" * ((1 << 20) + 100))
    monkeypatch.setattr(pickle, "load", lambda *a, **kw: pytest.fail("pickle executed"))
    monkeypatch.setattr(pickle, "loads", lambda *a, **kw: pytest.fail("pickle executed"))
    np.testing.assert_array_equal(backend.read((slice(None),)), source.value)
    assert backend.stats.corruptions == 1 and source.calls == 2
    cache.close()


def test_source_failure_not_cached_and_oversize_bypassed(tmp_path):
    source = Source(np.zeros((128, 128), np.float32))
    cache = open_cache(tmp_path, entry=4096)
    backend = cache.backend(source, identity())
    backend.read((slice(None), slice(None)))
    assert backend.stats.bypasses == 1 and len(cache._cache) == 0
    source.fail = True
    with pytest.raises(RuntimeError, match="Origin"):
        backend.read((0, slice(None)))
    assert len(cache._cache) == 0
    source.fail = False
    source.value = np.zeros((128, 128), np.float64)
    with pytest.raises(ValueError, match="unexpected"):
        backend.read((0, slice(None)))
    assert len(cache._cache) == 0
    cache.close()


def test_lru_eviction_with_source_larger_than_cache(tmp_path):
    values = np.arange(16 * 512 * 512, dtype=np.float32).reshape(16, 512, 512)
    source = Source(values)
    cache = open_cache(tmp_path)
    backend = cache.backend(source, identity())
    for frame in range(3):
        backend.read((frame, slice(None), slice(None)))
    backend.read((0, slice(None), slice(None)))
    backend.read((3, slice(None), slice(None)))
    assert cache.volume() < cache.capacity_bytes
    hits = backend.stats.hits
    backend.read((0, slice(None), slice(None)))
    assert backend.stats.hits == hits + 1
    misses = backend.stats.misses
    backend.read((1, slice(None), slice(None)))
    assert backend.stats.misses == misses + 1
    for frame in range(4, 16):
        np.testing.assert_array_equal(
            backend.read((frame, slice(None), slice(None))), values[frame])
        assert cache.volume() < cache.capacity_bytes
    assert len(cache._cache) <= 3
    cache.close()


def _process_read(args):
    path, cold = args
    source = Source(np.arange(64 * 64, dtype=np.float32).reshape(64, 64))
    source.fail = not cold
    source.delay = 0.1 if cold else 0
    cache = open_cache(path)
    backend = cache.backend(source, identity())
    hashes = []
    for _ in range(20):
        value = backend.read((slice(None), slice(None)))
        hashes.append(hashlib.sha256(value.tobytes()).hexdigest())
    result = (hashes, backend.stats.hits, backend.stats.misses, os.getpid())
    cache.close()
    return result


def _subprocess_read(args):
    result = subprocess.run([sys.executable, __file__, "--cache-worker", json.dumps(args)],
                            capture_output=True, text=True, check=True, timeout=30)
    return json.loads(result.stdout)


def test_independent_workers_share_cache_and_reopen_on_restart(tmp_path):
    # File entry points work under pytest importlib mode in a standalone export too.
    with ThreadPoolExecutor(4) as pool:
        cold = list(pool.map(_subprocess_read, [(str(tmp_path), True)] * 4))
    with ThreadPoolExecutor(4) as pool:
        warm = list(pool.map(_subprocess_read, [(str(tmp_path), False)] * 4))
    expected = hashlib.sha256(np.arange(4096, dtype=np.float32).tobytes()).hexdigest()
    assert all(set(hashes) == {expected} for hashes, _, _, _ in cold + warm)
    assert all(hits == 20 and misses == 0 for _, hits, misses, _ in warm)
    assert sum(misses for _, _, misses, _ in cold) == 1
    assert len({pid for _, _, _, pid in cold + warm}) == 8
    cache = open_cache(tmp_path)
    assert cache._cache.check() == []
    cache.close()


def test_fill_timeout_and_killed_holder_release_the_os_lock(tmp_path):
    from filelock import Timeout

    cache = TileCache(tmp_path, capacity_bytes=4 << 20, max_entry_bytes=1 << 20,
                      fill_timeout_seconds=0.03)
    source = Source(np.arange(256, dtype=np.float32))
    backend = cache.backend(source, identity())
    selection = (slice(None),)
    backend.read(selection)
    key = next(iter(cache._cache))
    cache._cache.delete(key)
    ready = tmp_path / "ready"
    child = subprocess.Popen([sys.executable, __file__, "--hold-lock",
                              json.dumps([str(tmp_path), key, str(ready)])])
    try:
        deadline = time.monotonic() + 10
        while not ready.exists():
            assert child.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        with pytest.raises(Timeout):
            backend.read(selection)
        assert source.calls == 1
    finally:
        child.kill()
        child.wait(timeout=10)
    np.testing.assert_array_equal(backend.read(selection), source.value)
    assert source.calls == 2
    assert len(list((tmp_path / "locks").glob("*.lock"))) == 1
    cache.close()


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan"), True])
def test_invalid_fill_timeout(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        TileCache(tmp_path, capacity_bytes=4 << 20, max_entry_bytes=1 << 20,
                  fill_timeout_seconds=timeout)


def test_invalid_configuration_and_non_numeric_arrays(tmp_path):
    with pytest.raises(ValueError, match="identifiers"):
        identity(revision="")
    with pytest.raises(ValueError, match="capacity"):
        TileCache(tmp_path, capacity_bytes=10, max_entry_bytes=5)
    with pytest.raises(ValueError):
        TileCache(tmp_path, capacity_bytes=True, max_entry_bytes=1)
    cache = open_cache(tmp_path)
    with pytest.raises(ValueError, match="different"):
        open_cache(tmp_path, capacity=8 << 20)
    with pytest.raises(ValueError, match="numeric"):
        cache.backend(Source(np.array(["a", "b"], dtype=object)), identity())
    with pytest.raises(TypeError, match="identity|Identity"):
        cache.backend(Source(np.ones(2)), "mutable")
    with pytest.raises(ValueError, match="source backend directly"):
        cache.backend(cache.backend(Source(np.ones(2)), identity()), identity())
    cache.close()


def test_cache_metadata_cannot_delete_an_external_file(tmp_path):
    from arrayroute.disk_cache import CacheCorruptionError

    source = Source(np.arange(256, dtype=np.float32))
    cache = open_cache(tmp_path / "cache")
    backend = cache.backend(source, identity())
    backend.read((slice(None),))
    external = tmp_path / "keep.bin"
    external.write_bytes(b"keep this file")
    with sqlite3.connect(tmp_path / "cache" / "data" / "cache.db") as db:
        db.execute("UPDATE Cache SET filename=?", (str(external),))
    with pytest.raises(CacheCorruptionError, match="outside"):
        backend.read((slice(None),))
    assert external.read_bytes() == b"keep this file"
    cache.close()


if __name__ == "__main__":
    if sys.argv[1] == "--cache-worker":
        print(json.dumps(_process_read(json.loads(sys.argv[2]))))
    else:
        assert sys.argv[1] == "--hold-lock"
        path, key, ready = json.loads(sys.argv[2])
        held_cache = open_cache(path)
        with held_cache._fill_lock(key):
            Path(ready).write_text("locked")
            time.sleep(30)
