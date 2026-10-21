# SPDX-License-Identifier: Apache-2.0
"""Worker-local, bounded metadata reuse for fixed Zarr stores on a filesystem."""

from __future__ import annotations

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
        self._readers: OrderedDict[tuple[str, str], ArrayReader] = OrderedDict()

    def clear(self) -> None:
        self._readers.clear()
        self._context = None

    def __getstate__(self) -> dict:
        return {**vars(self), "_readers": OrderedDict(), "_context": None}

    def open(self, store: str | Path, array_path: str) -> ArrayReader:
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
                self._context = ts.Context(
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
