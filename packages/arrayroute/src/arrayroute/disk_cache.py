# SPDX-License-Identifier: Apache-2.0
"""Verified decoded tiles shared by workers on one machine's local filesystem."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from operator import index
from pathlib import Path

import numpy as np
from diskcache import Cache, Disk
from diskcache.core import MODE_BINARY, UNKNOWN
from filelock import FileLock, SoftFileLock

from .plan import ReadPlan, Selection, plan
from .reader import Backend

_MAGIC = b"ARTILE01"
_HEADER = len(_MAGIC) + 32


def _digest(key: str, payload: bytes | memoryview) -> bytes:
    digest = hashlib.sha256(key.encode())
    digest.update(payload)
    return digest.digest()


class CacheCorruptionError(ValueError):
    """A cache entry cannot be trusted; reload it from the immutable source."""


class _TilePayload:
    """Synchronous bounded buffer views for DiskCache's binary stream writer."""

    def __init__(self, key: str, value: np.ndarray):
        self.payload = memoryview(np.ascontiguousarray(value)).cast("B")
        self.header = _MAGIC + _digest(key, self.payload)
        self.size = len(self.header) + len(self.payload)
        self.offset = 0

    def read(self, size: int) -> bytes | memoryview:
        if size <= 0:
            raise ValueError("Tile stream reads require a positive size")
        if self.offset < _HEADER:
            end = min(_HEADER, self.offset + size)
            chunk = self.header[self.offset:end]
        else:
            end = min(self.size, self.offset + size)
            chunk = self.payload[self.offset - _HEADER:end - _HEADER]
        self.offset = end
        return chunk


class _BytesDisk(Disk):
    """Never serialize or deserialize Python objects in the tile cache."""

    def __init__(self, directory, min_file_size=0, pickle_protocol=0, max_payload_bytes=1):
        super().__init__(directory, min_file_size=0, pickle_protocol=pickle_protocol)
        self.max_payload_bytes = max_payload_bytes

    def put(self, key):
        if not isinstance(key, str):
            raise TypeError("Tile cache keys must be strings")
        return key, True

    def get(self, key, raw):
        if not raw or not isinstance(key, str):
            raise CacheCorruptionError("Invalid tile cache key encoding")
        return key

    def store(self, value, read, key=UNKNOWN):
        if (type(value) is not _TilePayload or not read or value.offset != 0
                or value.size > self.max_payload_bytes):
            raise ValueError("Tile cache accepts fresh bounded tile streams only")
        return super().store(value, True, key)

    def remove(self, file_path):
        root = Path(self._directory).resolve()
        if not (root / file_path).resolve().is_relative_to(root):
            raise CacheCorruptionError("Refusing to remove a path outside the tile cache")
        return super().remove(file_path)

    def fetch(self, mode, filename, value, read):
        if mode != MODE_BINARY or not isinstance(filename, str):
            raise CacheCorruptionError("Invalid tile cache value encoding")
        root = Path(self._directory).resolve()
        path = (root / filename).resolve()
        if not path.is_relative_to(root):
            raise CacheCorruptionError("Invalid tile cache path")
        stream = path.open("rb")
        try:
            if os.fstat(stream.fileno()).st_size > self.max_payload_bytes:
                raise CacheCorruptionError("Tile cache value exceeds its size bound")
            if read:
                return stream
            return stream.read(self.max_payload_bytes + 1)
        except BaseException:
            stream.close()
            raise
        finally:
            if not read:
                stream.close()


@dataclass(frozen=True)
class ArrayIdentity:
    """Non-secret source/array name, immutable revision, and representation version.

    Revision is a caller assertion, not a snapshot or source checksum. Representation
    must change with decoding, physical units or preprocessing semantics. Cache paths
    must be scoped to the authorized user/tenant; this library is not an auth boundary.
    """

    source: str
    revision: str
    representation: str

    def __post_init__(self):
        for value in (self.source, self.revision, self.representation):
            if not isinstance(value, str) or not value.strip() or len(value) > 4096:
                raise ValueError("Source, revision and representation must be nonempty identifiers")


@dataclass
class TileCacheStats:
    hits: int = 0
    misses: int = 0
    corruptions: int = 0
    bypasses: int = 0
    hit_bytes: int = 0
    source_bytes: int = 0
    coalesced_hits: int = 0
    fill_wait_seconds: float = 0.0


class TileCache:
    """Process-shared LRU cache on local disk, never NFS or object storage.

    The size limit is DiskCache's approximate database-plus-payload volume after cull.
    WAL, filesystem bookkeeping, concurrent writes and temporary files are additional;
    use a filesystem quota for a strict disk limit. All workers must use the same policy.
    Source errors propagate. Cache corruption is discarded and read again from source.
    Cache/database I/O errors propagate instead of silently changing the execution policy.
    Misses sharing a key coordinate through 256 stable lock stripes. Waiting workers
    recheck the cache after the active fill, avoiding duplicate source reads while
    the entry remains retained. Unrelated keys on the same stripe can briefly wait.
    The OS releases a crashed process's hard lock; waits have an explicit timeout.
    """

    def __init__(self, directory: str | Path, *, capacity_bytes: int, max_entry_bytes: int,
                 fill_timeout_seconds: float = 60):
        if any(isinstance(v, bool) for v in (capacity_bytes, max_entry_bytes)):
            raise ValueError("Cache limits must be positive integers")
        capacity_bytes, max_entry_bytes = index(capacity_bytes), index(max_entry_bytes)
        if capacity_bytes < 1 << 20 or not 0 < max_entry_bytes <= capacity_bytes // 2:
            raise ValueError("Cache capacity must be >= 1 MiB; entries must be <= half capacity")
        if (isinstance(fill_timeout_seconds, bool) or not math.isfinite(fill_timeout_seconds)
                or fill_timeout_seconds < 0):
            raise ValueError("Fill timeout must be finite and nonnegative")
        if FileLock is SoftFileLock:
            raise RuntimeError("Tile cache coordination requires OS hard file locks")
        self.directory = str(Path(directory).absolute())
        self.capacity_bytes = capacity_bytes
        self.max_entry_bytes = max_entry_bytes
        self.fill_timeout_seconds = fill_timeout_seconds
        root = Path(self.directory)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        policy = json.dumps({"schema": 2, "capacity_bytes": capacity_bytes,
                             "max_entry_bytes": max_entry_bytes}, sort_keys=True).encode()
        fd, temporary = tempfile.mkstemp(prefix="policy-", dir=root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(policy)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, root / "policy.json")
            except FileExistsError:
                pass
            if (root / "policy.json").read_bytes() != policy:
                raise ValueError("Existing tile cache has a different size policy or schema")
        finally:
            os.unlink(temporary)
        self._cache = Cache(
            str(root / "data"), disk=_BytesDisk, timeout=30,
            size_limit=capacity_bytes, eviction_policy="least-recently-used",
            cull_limit=1,
            disk_min_file_size=0, disk_max_payload_bytes=max_entry_bytes + _HEADER,
        )
        (root / "locks").mkdir(exist_ok=True, mode=0o700)

    def __getstate__(self):
        return {"directory": self.directory, "capacity_bytes": self.capacity_bytes,
                "max_entry_bytes": self.max_entry_bytes,
                "fill_timeout_seconds": self.fill_timeout_seconds}

    def __setstate__(self, state):
        self.__init__(**state)

    def close(self) -> None:
        self._cache.close()

    def volume(self) -> int:
        return self._cache.volume()

    def backend(self, source: Backend, identity: ArrayIdentity) -> CachedBackend:
        return CachedBackend(source, self, identity)

    def _fill_lock(self, key: str):
        return FileLock(Path(self.directory) / "locks" / f"{key[:2]}.lock",
                        timeout=self.fill_timeout_seconds, poll_interval=0.005,
                        mode=0o600, thread_local=True, is_singleton=False,
                        fallback_to_soft=False, preserve_lock_file=True)


class CachedBackend:
    """Cache exact normalized selections, preserving the ArrayReader backend contract.

    Own the source backend and stats in one worker. Tiles returned to callers are
    writable arrays; caller mutations never alter retained values. Hits read directly
    into their output array and hash it in place. Misses stream source arrays without
    serialized payload copies; non-contiguous arrays need one contiguous copy. The max-entry limit
    bounds each cached payload, not overall process RSS.
    Overlapping selections with different bounds are different entries.
    """

    def __init__(self, source: Backend, cache: TileCache, identity: ArrayIdentity):
        if isinstance(source, CachedBackend):
            raise ValueError("Wrap the source backend directly, not another cached backend")
        if not isinstance(identity, ArrayIdentity):
            raise TypeError("Provide an explicit immutable ArrayIdentity")
        self.source, self.cache = source, cache
        self.spec, self.dtype = source.spec, np.dtype(source.dtype)
        if self.dtype.kind not in "biufc" or self.dtype.itemsize != self.spec.itemsize:
            raise ValueError("Tile cache requires a fixed-width numeric or boolean dtype")
        self.stats = TileCacheStats()
        self._identity = {"identity": asdict(identity), "spec": asdict(self.spec),
                          "dtype": self.dtype.str, "schema": 1}

    def read(self, selection: Selection) -> np.ndarray:
        return self.read_plan(plan(self.spec, selection))

    def read_plan(self, request: ReadPlan) -> np.ndarray:
        encoded = [s if isinstance(s, int) else [s.start, s.stop, s.step]
                   for s in request.selection]
        key = hashlib.sha256(json.dumps({**self._identity, "selection": encoded},
                                       sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        eligible = 0 < request.output_bytes <= self.cache.max_entry_bytes
        if not eligible:
            self.stats.bypasses += 1
            return self._load(key, request, cache_value=False)
        try:
            value = self._cached(key, request)
            if value is not None:
                return value
        except CacheCorruptionError:
            pass  # Recheck under the fill lock before removing a possibly repaired entry.
        started = time.perf_counter()
        with self.cache._fill_lock(key):
            self.stats.fill_wait_seconds += time.perf_counter() - started
            try:
                value = self._cached(key, request)
                if value is not None:
                    self.stats.coalesced_hits += 1
                    return value
            except CacheCorruptionError:
                self.cache._cache.delete(key)
                self.stats.corruptions += 1
            self.stats.misses += 1
            return self._load(key, request, cache_value=True)

    def _cached(self, key: str, request: ReadPlan) -> np.ndarray | None:
        stream = self.cache._cache.get(key, default=None, read=True)
        if stream is None:
            return None
        if not isinstance(stream, io.BufferedReader):
            raise CacheCorruptionError("Tile cache value is not a binary stream")
        with stream:
            header = stream.read(_HEADER)
            value = np.empty(request.output_shape, dtype=self.dtype)
            payload = memoryview(value).cast("B")
            size = stream.readinto(payload)
            if (size != request.output_bytes or header[:8] != _MAGIC
                    or stream.read(1) or _digest(key, payload) != header[8:_HEADER]):
                raise CacheCorruptionError("Tile cache checksum or size differs")
        self.stats.hits += 1
        self.stats.hit_bytes += value.nbytes
        return value

    def _load(self, key: str, request: ReadPlan, *, cache_value: bool) -> np.ndarray:
        execute = getattr(self.source, "read_plan", None)
        value = execute(request) if execute is not None else self.source.read(request.selection)
        if value.shape != request.output_shape or value.dtype != self.dtype:
            raise ValueError("Source returned an unexpected tile shape or dtype")
        self.stats.source_bytes += value.nbytes
        if cache_value:
            self.cache._cache.set(key, _TilePayload(key, value), read=True)
            self.cache._cache.cull()
        return value
