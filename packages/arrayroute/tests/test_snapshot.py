# SPDX-License-Identifier: Apache-2.0
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
import zarr

ic = pytest.importorskip("icechunk")

from arrayroute.remote import S3Store  # noqa: E402
from arrayroute.snapshot import (  # noqa: E402
    FilesystemRepository,
    S3Repository,
    Snapshot,
    SnapshotReadSession,
)


@pytest.fixture
def repository(tmp_path):
    location = FilesystemRepository(str(tmp_path / "repository"))
    repo = ic.Repository.create(ic.local_filesystem_storage(location.path))
    session = repo.writable_session("main")
    group = zarr.group(store=session.store)
    values = np.random.default_rng(46).standard_normal((4, 17, 13, 9, 3), dtype=np.float32)
    for name in ["velocity", "reference"]:
        group.create_array(name, data=values, chunks=(1, 8, 8, 8, 3))
    group.attrs.update({"units": "m/s", "time_units": "s"})
    snapshot_id = session.commit("original simulation outputs")
    return location, repo, values, snapshot_id


def test_revision_survives_new_commits_eviction_and_spawn(repository):
    location, repo, values, sid = repository
    pinned = Snapshot.resolve(location)
    assert pinned.snapshot_id == sid
    session = SnapshotReadSession(pinned, max_open_arrays=1)
    selection = (1, slice(None), 3, slice(None), 2)
    np.testing.assert_array_equal(session.read("velocity", selection), values[selection])
    newer = repo.writable_session("main")
    zarr.open_group(newer.store, mode="a")["velocity"][:] = 99
    next_id = newer.commit("replace field")
    assert next_id != sid
    session.read("reference", selection)
    assert list(session._readers) == ["reference"]
    session.clear()
    np.testing.assert_array_equal(session.read("velocity", selection), values[selection])
    clone = pickle.loads(pickle.dumps(session))
    assert clone._session is None and not clone._readers
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(session.read, "velocity", selection) for _ in range(2)]
        for result in futures:
            np.testing.assert_array_equal(result.result(), values[selection])
    latest = SnapshotReadSession(Snapshot.resolve(location))
    np.testing.assert_array_equal(latest.read("velocity", selection), 99)


def test_uncommitted_updates_are_invisible_and_readers_cannot_write(repository):
    location, repo, values, _ = repository
    writer = repo.writable_session("main")
    zarr.open_group(writer.store, mode="a")["velocity"][:] = 7
    session = SnapshotReadSession(Snapshot.resolve(location))
    reader = session.open("velocity")
    np.testing.assert_array_equal(reader.read((0,) + (slice(None),) * 4), values[0])
    with pytest.raises(ic.ReadOnlyError):
        reader.backend.array[:] = 5
    writer.discard_changes()
    np.testing.assert_array_equal(session.read("velocity", (0,) + (slice(None),) * 4), values[0])


def test_budget_and_sparse_selection_preserve_bits(repository):
    location, _, values, _ = repository
    session = SnapshotReadSession(Snapshot.resolve(location), buffer_bytes=24 << 10)
    selection = (slice(None), slice(None, None, 7), 3, slice(None, None, 2), slice(None))
    assert session.read("velocity", selection).tobytes() == values[selection].tobytes()
    with pytest.raises(MemoryError):
        session.read("velocity", (slice(None),) * 5)
    result = np.empty_like(values)
    for tile, block in session.open("velocity").stream((slice(None),) * 5):
        result[tile.destination] = block
    assert result.tobytes() == values.tobytes()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"buffer_bytes": 0},
        {"cache_bytes": -1},
        {"max_open_arrays": 0},
        {"metadata_entries": 0},
        {"concurrency": 0},
    ],
)
def test_limits_fail_before_io(kwargs):
    with pytest.raises(ValueError):
        SnapshotReadSession(Snapshot(FilesystemRepository("missing"), "abc"), **kwargs)


def test_reject_untrusted_paths_and_missing_revisions(repository):
    location, _, _, _ = repository
    session = SnapshotReadSession(Snapshot.resolve(location))
    for name in ["../velocity", "/velocity", "a//b", "a\\b", "a\x00b"]:
        with pytest.raises(ValueError):
            session.open(name)
    with pytest.raises(ic.SnapshotNotFoundError):
        SnapshotReadSession(Snapshot(location, "00000000000000000000")).open("velocity")
    with pytest.raises(ValueError):
        S3Repository(S3Store("test-bucket", endpoint="http://localhost:9000"))
    remote = S3Repository(S3Store("test-bucket", credentials="environment"))
    assert pickle.loads(pickle.dumps(remote)) == remote


def test_virtual_chunk_access_is_not_implicitly_authorized(tmp_path):
    from icechunk.config import RepositoryConfig
    from icechunk.storage import local_filesystem_store
    from icechunk.virtual import VirtualChunkContainer

    target = tmp_path / "external.bin"
    target.write_bytes(np.arange(4, dtype=np.float32).tobytes())
    location = FilesystemRepository(str(tmp_path / "repository"))
    config = RepositoryConfig.default()
    config.set_virtual_chunk_container(
        VirtualChunkContainer(tmp_path.as_uri() + "/", local_filesystem_store(str(tmp_path)))
    )
    repo = ic.Repository.create(ic.local_filesystem_storage(location.path), config=config)
    writer = repo.writable_session("main")
    zarr.group(writer.store).create_array(
        "field", shape=(4,), dtype="float32", chunks=(4,), compressors=None
    )
    writer.store.set_virtual_ref("field/c/0", target.as_uri(), offset=0, length=16)
    sid = writer.commit("external object reference")
    with pytest.raises(ic.IcechunkError):
        SnapshotReadSession(Snapshot(location, sid)).read("field", (slice(None),))
