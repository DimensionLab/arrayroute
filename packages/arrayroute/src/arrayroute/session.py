# SPDX-License-Identifier: Apache-2.0
"""Worker-local, bounded metadata reuse for fixed Zarr stores on a filesystem."""

from __future__ import annotations

import os
from collections import OrderedDict
from operator import index
from pathlib import Path
from typing import Any

from .backends import TensorStoreBackend, ZarrBackend, ZarristaBackend
from .plan import Selection
from .reader import ArrayReader, Backend, ReadStats


class LocalReadSession:
    """Reuse read-only array handles and one native context within a worker.

    Sessions are worker-local and not thread-safe. Handles are bounded by count, not
    metadata bytes. Decoded buffers are not retained
    by this class. TensorStore's shared cache soft limit and thread limits apply
    across its handles; other engines use their own process-level threading policies.
    Input metadata must remain fixed for the session's lifetime. Call clear() after
    changing inputs. Serialized copies reopen handles in their destination process;
    use spawn, not fork, with native storage engines and multiprocessing workers.

    With immutable=True, TensorStore skips data freshness checks and can reuse cached
    chunks directly. This is a caller assertion, not a snapshot or a write lock: values
    must remain unchanged too. clear() starts a fresh context after input changes.
    Sessions created by TensorStoreReadGroup keep that group's context across clear();
    create a new group after changing inputs. Group-owned sessions cannot be serialized.

    This adapter supports local or mounted filesystems. ArrayReader's Backend protocol
    remains the extension point for object stores and other storage systems.
    """

    def __init__(
        self,
        engine: str = "zarr-python",
        *,
        buffer_bytes: int = 128 << 20,
        max_open_arrays: int = 64,
        tensorstore_cache_bytes: int = 0,
        tensorstore_threads: int = 8,
        immutable: bool = False,
    ):
        if not isinstance(immutable, bool):
            raise TypeError("immutable must be a boolean")
        buffer_bytes, max_open_arrays, tensorstore_cache_bytes, tensorstore_threads = (
            index(v)
            for v in (buffer_bytes, max_open_arrays, tensorstore_cache_bytes, tensorstore_threads)
        )
        if engine not in {"zarr-python", "tensorstore", "zarrista"}:
            raise ValueError(f"Unknown array engine {engine!r}")
        if (
            min(buffer_bytes, max_open_arrays, tensorstore_threads) < 1
            or tensorstore_cache_bytes < 0
        ):
            raise ValueError("Positive buffer, handle and thread limits are required; cache >= 0")
        if engine != "tensorstore" and tensorstore_cache_bytes:
            raise ValueError("tensorstore_cache_bytes requires the tensorstore engine")
        self.engine = engine
        self.buffer_bytes = buffer_bytes
        self.max_open_arrays = max_open_arrays
        self.tensorstore_cache_bytes = tensorstore_cache_bytes
        self.tensorstore_threads = tensorstore_threads
        self.immutable = immutable
        self._context: Any = None
        self._group: TensorStoreReadGroup | None = None
        self._readers: OrderedDict[tuple[str, str], ArrayReader] = OrderedDict()

    def clear(self) -> None:
        """Drop this session's handles; an explicit shared group's cache stays alive."""
        self._readers.clear()
        self._context = None

    def __getstate__(self) -> dict:
        if self._group is not None:
            raise TypeError("Shared read sessions are in-process only; create a group after spawn")
        return {**vars(self), "_readers": OrderedDict(), "_context": None}

    def open(self, store: str | Path, array_path: str) -> ArrayReader:
        if self._group is not None:
            self._group._check_process()
        key = (str(Path(store).absolute()), array_path.strip("/"))
        found = self._readers.get(key)
        if found is not None:
            self._readers.move_to_end(key)
            return found
        backend: Backend
        if self.engine == "zarr-python":
            import zarr

            backend = ZarrBackend(zarr.open_array(str(Path(key[0]) / key[1]), mode="r"))
        elif self.engine == "tensorstore":
            import tensorstore as ts

            if self._context is None:
                self._context = self._group._context if self._group is not None else ts.Context(
                    {
                        "cache_pool": {"total_bytes_limit": self.tensorstore_cache_bytes},
                        "data_copy_concurrency": {"limit": self.tensorstore_threads},
                        "file_io_concurrency": {"limit": self.tensorstore_threads},
                    }
                )
            backend = TensorStoreBackend(
                ts.open(
                    {
                        "driver": "zarr3",
                        "kvstore": {"driver": "file", "path": str(Path(key[0]) / key[1]) + "/"},
                    },
                    read=True,
                    recheck_cached_data=not self.immutable,
                    context=self._context,
                ).result()
            )
        else:
            import zarrista

            backend = ZarristaBackend(
                zarrista.Array.open(zarrista.store.FilesystemStore(key[0]), path="/" + key[1])
            )
        reader = ArrayReader(backend, buffer_bytes=self.buffer_bytes)
        while len(self._readers) >= self.max_open_arrays:
            self._readers.popitem(last=False)
        self._readers[key] = reader
        return reader

    def read(
        self,
        store: str | Path,
        array_path: str,
        selection: Selection,
        *,
        stats: ReadStats | None = None,
    ):
        return self.open(store, array_path).read(selection, stats=stats)


class TensorStoreReadGroup:
    """Share native cache and thread budgets across independent in-process sessions.

    A group can create one session per worker thread. Python handle caches, readers,
    output arrays and statistics remain worker-owned; only TensorStore's native
    context is shared. ``cache_bytes`` is a soft limit for the whole group, not per
    session, and excludes outputs, decoder scratch and Python metadata.

    Metadata must stay fixed. ``immutable=True`` additionally asserts that values
    stay fixed; this is not a storage snapshot. ``session.clear()`` drops local
    handles but cannot invalidate shared cache entries. After changing source data,
    release all old sessions and create a new group. Groups are deliberately not
    serializable and must be constructed after spawning processes, never before fork.
    """

    def __init__(self, *, cache_bytes: int = 0, threads: int = 8, immutable: bool = False):
        if isinstance(cache_bytes, bool) or isinstance(threads, bool):
            raise TypeError("Cache bytes and thread count must be integers, not booleans")
        cache_bytes, threads = index(cache_bytes), index(threads)
        if cache_bytes < 0 or threads < 1:
            raise ValueError("Cache bytes must be nonnegative and thread count positive")
        if not isinstance(immutable, bool):
            raise TypeError("immutable must be a boolean")
        import tensorstore as ts

        self._cache_bytes, self._threads, self._immutable = cache_bytes, threads, immutable
        self._pid = os.getpid()
        self._context = ts.Context({
            "cache_pool": {"total_bytes_limit": cache_bytes},
            "data_copy_concurrency": {"limit": threads},
            "file_io_concurrency": {"limit": threads},
        })

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes

    @property
    def threads(self) -> int:
        return self._threads

    @property
    def immutable(self) -> bool:
        return self._immutable

    def _check_process(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError("Shared read groups must be constructed in the consuming process")

    def __getstate__(self) -> dict:
        raise TypeError("Shared read groups are in-process only; create a group after spawn")

    def session(
        self, *, buffer_bytes: int = 128 << 20, max_open_arrays: int = 64,
    ) -> LocalReadSession:
        """Create a worker-owned session using this group's aggregate native budget."""
        self._check_process()
        session = LocalReadSession(
            "tensorstore", buffer_bytes=buffer_bytes, max_open_arrays=max_open_arrays,
            tensorstore_cache_bytes=self.cache_bytes, tensorstore_threads=self.threads,
            immutable=self.immutable,
        )
        session._group = self
        return session
