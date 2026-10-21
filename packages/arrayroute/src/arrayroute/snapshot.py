# SPDX-License-Identifier: Apache-2.0
"""Optional Icechunk adapter: immutable revisions, reopened inside each read worker."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from operator import index
from pathlib import Path
from typing import Any, cast

from .backends import ZarrBackend
from .plan import Selection
from .reader import ArrayReader, ReadStats
from .remote import S3Store, _path


@dataclass(frozen=True)
class FilesystemRepository:
    """Trusted local/mounted repository, with an absolute path stable across worker cwd changes."""

    path: str

    def __post_init__(self) -> None:
        if not self.path or any(ord(c) < 32 for c in self.path):
            raise ValueError("Repository path must be nonempty and contain no control characters")
        object.__setattr__(self, "path", str(Path(self.path).absolute()))


@dataclass(frozen=True)
class S3Repository:
    """Credential-provider reference; no credentials are serialized into worker tasks."""

    store: S3Store
    allow_http: bool = False
    force_path_style: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.store, S3Store):
            raise TypeError("Expected an S3Store")
        if self.store.endpoint and self.store.endpoint.startswith("http:") and not self.allow_http:
            raise ValueError("Plain HTTP requires explicit allow_http=True")


RepositoryLocation = FilesystemRepository | S3Repository


def _storage(location: RepositoryLocation):
    import icechunk as ic

    if isinstance(location, FilesystemRepository):
        return ic.local_filesystem_storage(location.path)
    if not isinstance(location, S3Repository):
        raise TypeError("Expected a FilesystemRepository or S3Repository")
    store = location.store
    return ic.s3_storage(
        bucket=store.bucket,
        prefix=store.prefix,
        region=store.region,
        endpoint_url=store.endpoint,
        allow_http=location.allow_http,
        force_path_style=location.force_path_style,
        anonymous=store.credentials == "anonymous",
        from_env=store.credentials != "anonymous",
    )


def _verify_storage(location: RepositoryLocation) -> None:
    if isinstance(location, S3Repository):
        from .s3_writes import verify_s3_writes

        verify_s3_writes(
            location.store,
            allow_http=location.allow_http,
            force_path_style=location.force_path_style,
        )


def _repository(location: RepositoryLocation):
    import icechunk as ic

    return ic.Repository.open(_storage(location), authorize_virtual_chunk_access={})


def _writable_repository(location: RepositoryLocation):
    repo = _repository(location)
    if isinstance(location, S3Repository):
        from .s3_writes import UnsafeObjectStoreError

        settings = repo.config.storage
        # None inherits the S3 backend's safe defaults. An explicit False disables
        # conditional writes even when the endpoint itself passed the capability probe.
        if settings is not None and (
            settings.unsafe_use_conditional_create is False
            or settings.unsafe_use_conditional_update is False
        ):
            raise UnsafeObjectStoreError(
                "Repository configuration disables conditional writes; restore safe settings "
                "before preparing data"
            )
    return repo


@dataclass(frozen=True)
class Snapshot:
    """Serializable revision reference. It is not a backup or a garbage-collection lease.

    The owner must retain this revision while jobs/reports reference it. External virtual
    chunks are deliberately unauthorized by this adapter: a snapshot cannot make mutable
    external objects immutable. Store locations must be authorized by the calling app.
    """

    location: RepositoryLocation
    snapshot_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.location, (FilesystemRepository, S3Repository)):
            raise TypeError("Expected a FilesystemRepository or S3Repository")
        if not self.snapshot_id or not self.snapshot_id.isalnum():
            raise ValueError("Expected a snapshot ID, not a branch name or URL")

    @classmethod
    def resolve(cls, location: RepositoryLocation, *, branch: str = "main") -> Snapshot:
        """Resolve a mutable branch exactly once, before distributing work."""
        import icechunk as ic

        repository = ic.Repository.open(_storage(location), authorize_virtual_chunk_access={})
        return cls(location, repository.readonly_session(branch=branch).snapshot_id)


class SnapshotReadSession:
    """Read one fixed revision with bounded handles and explicitly configured caches.

    Use one session per worker and spawn rather than fork. Pickling drops all native
    objects; a worker reopens the same snapshot, never the current branch. Chunk/attribute
    caches have byte budgets, metadata caches have entry budgets, and selected arrays use
    ArrayReader's buffer budget. These are not a complete process RSS bound.

    Local reads use a 1 MiB range-splitting target to avoid subdividing small files.
    Remote repositories keep the storage backend's target. range_request_bytes
    overrides either policy without changing the persisted repository configuration.
    """

    def __init__(
        self,
        snapshot: Snapshot,
        *,
        buffer_bytes: int = 128 << 20,
        max_open_arrays: int = 64,
        cache_bytes: int = 0,
        metadata_entries: int = 1024,
        concurrency: int = 8,
        range_request_bytes: int | None = None,
    ):
        if not isinstance(snapshot, Snapshot):
            raise TypeError("Resolve and pin a Snapshot before creating a read session")
        buffer_bytes, max_open_arrays, cache_bytes, metadata_entries, concurrency = (
            index(v)
            for v in (buffer_bytes, max_open_arrays, cache_bytes, metadata_entries, concurrency)
        )
        if min(buffer_bytes, max_open_arrays, metadata_entries, concurrency) < 1 or cache_bytes < 0:
            raise ValueError(
                "Positive buffer/handle/metadata/concurrency limits and cache >= 0 required"
            )
        if range_request_bytes is not None:
            range_request_bytes = index(range_request_bytes)
            if not 0 < range_request_bytes <= (1 << 64) - 1:
                raise ValueError("range_request_bytes must be a positive uint64 byte count")
        self.snapshot = snapshot
        self.buffer_bytes = buffer_bytes
        self.max_open_arrays = max_open_arrays
        self.cache_bytes = cache_bytes
        self.metadata_entries = metadata_entries
        self.concurrency = concurrency
        self.range_request_bytes = range_request_bytes
        self._session: Any = None
        self._readers: OrderedDict[str, ArrayReader] = OrderedDict()

    def clear(self) -> None:
        self._readers.clear()
        self._session = None

    def __getstate__(self) -> dict:
        return {**vars(self), "_session": None, "_readers": OrderedDict()}

    @property
    def effective_range_request_bytes(self) -> int | None:
        if self.range_request_bytes is not None:
            return self.range_request_bytes
        return 1 << 20 if isinstance(self.snapshot.location, FilesystemRepository) else None

    def open(self, array_path: str) -> ArrayReader:
        import icechunk as ic
        import zarr
        from icechunk.config import CachingConfig, RepositoryConfig

        path = _path(array_path)
        if path in self._readers:
            self._readers.move_to_end(path)
            return self._readers[path]
        if self._session is None:
            repository = ic.Repository.open(
                _storage(self.snapshot.location),
                config=RepositoryConfig(
                    max_concurrent_requests=self.concurrency,
                    storage=(
                        ic.StorageSettings(
                            concurrency=ic.StorageConcurrencySettings(
                                ideal_concurrent_request_size=self.effective_range_request_bytes
                            )
                        )
                        if self.effective_range_request_bytes is not None
                        else None
                    ),
                    caching=CachingConfig(
                        num_snapshot_nodes=self.metadata_entries,
                        num_chunk_refs=self.metadata_entries,
                        num_transaction_changes=0,
                        num_bytes_attributes=self.cache_bytes,
                        num_bytes_chunks=self.cache_bytes,
                    ),
                ),
                authorize_virtual_chunk_access={},
            )
            self._session = repository.readonly_session(snapshot_id=self.snapshot.snapshot_id)
        array = zarr.open_array(store=self._session.store, path=path, mode="r")
        reader = ArrayReader(ZarrBackend(array), buffer_bytes=self.buffer_bytes)
        while len(self._readers) >= self.max_open_arrays:
            self._readers.popitem(last=False)
        self._readers[path] = reader
        return reader

    def read(self, array_path: str, selection: Selection, *, stats: ReadStats | None = None):
        return self.open(array_path).read(selection, stats=stats)

    def attributes(self, array_path: str) -> dict[str, Any]:
        """Copy array attributes from the same revision as its numerical data."""
        return dict(cast(ZarrBackend, self.open(array_path).backend).array.attrs)
