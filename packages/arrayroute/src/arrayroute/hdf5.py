# SPDX-License-Identifier: Apache-2.0
"""Local, read-only HDF5 adapters with explicit file ownership and embedded-data policy."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .plan import ArraySpec, Selection, plan, tiles
from .reader import ArrayReader, ReadStats


class HDF5Backend:
    """Borrow a session-owned dataset; payload reads fail after close or across fork.

    Fixed-size dtypes only. Chunk costs conservatively cover intersected decoded chunks.
    Contiguous data has no decoded chunk cost; native conversion/codec scratch and file
    metadata are additional. HDF5's raw chunk cache is disabled by the owning session.
    """

    def __init__(self, dataset: Any, check_open: Callable[[], None]):
        import h5py

        self._dataset = dataset
        self._check_open = check_open
        if dataset.is_virtual or dataset.external:
            raise ValueError("HDF5 arrays must store data in this file")
        self.dtype = np.dtype(dataset.dtype)
        if self.dtype.hasobject or not self.dtype.itemsize:
            raise TypeError(
                "HDF5 reads require fixed-size dtypes; variable-length/references rejected"
            )
        if self.dtype.subdtype is not None:
            raise TypeError("HDF5 array-valued dtypes require explicit dimension expansion")
        if dataset.shape is None or not dataset.shape:
            raise ValueError("HDF5 reads require a non-null array with at least one dimension")
        layout = dataset.id.get_create_plist().get_layout()
        if layout not in (h5py.h5d.CONTIGUOUS, h5py.h5d.CHUNKED):
            raise ValueError("Only contiguous and chunked HDF5 storage layouts are supported")
        shape = tuple(int(n) for n in dataset.shape)
        contiguous = layout == h5py.h5d.CONTIGUOUS
        chunks = tuple(max(1, n) for n in shape) if contiguous else tuple(dataset.chunks)
        self.spec = ArraySpec(shape, chunks, self.dtype.itemsize, contiguous=contiguous)

    def read(self, selection: Selection) -> np.ndarray:
        self._check_open()
        request = plan(self.spec, selection)
        result = np.empty(request.output_shape, dtype=self.dtype)
        if result.size:
            self._dataset.read_direct(result, source_sel=request.selection)
        return result


class HDF5ReadSession:
    """Own one immutable local file and a bounded metadata-handle cache.

    No external/soft links, virtual datasets, external raw storage or variable-length
    payloads are followed. This policy is not a sandbox for libhdf5 or codec plugins.
    Source metadata and values must remain immutable until close. Create an independent
    session inside each spawned process; h5py's global lock serializes calls in threads.
    Returned arrays are owned, writable NumPy values and survive session close. Readers
    borrow the file and cannot outlive it. Cache limits cover this session's references;
    callers retaining reader handles retain their dataset metadata too.
    """

    def __init__(
        self, path: str | Path, *, buffer_bytes: int = 128 << 20, max_open_arrays: int = 64
    ):
        for name, value in [("buffer_bytes", buffer_bytes), ("max_open_arrays", max_open_arrays)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        import h5py

        self.path = Path(path)
        self.buffer_bytes = buffer_bytes
        self.max_open_arrays = max_open_arrays
        self._pid = os.getpid()
        self._file = h5py.File(self.path, "r", rdcc_nbytes=0)
        self._readers: OrderedDict[str, ArrayReader] = OrderedDict()
        self._closed = False

    def _check_open(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError(
                "Open HDF5 sessions inside each spawned worker; never reuse after fork"
            )
        if self._closed:
            raise RuntimeError("HDF5 session is closed")

    def _resolve(self, path: str):
        import h5py

        self._check_open()
        if not isinstance(path, str) or "\x00" in path:
            raise ValueError("HDF5 paths must be strings without NUL characters")
        parts = path.strip("/").split("/") if path.strip("/") else []
        if any(p in ("", ".", "..") for p in parts):
            raise ValueError("HDF5 paths must not contain empty, dot or parent components")
        current = self._file
        for part in parts:
            if not isinstance(current, h5py.Group):
                raise ValueError("HDF5 path traverses a dataset")
            link = current.get(part, getlink=True)
            if link is None:
                raise KeyError(path)
            if not isinstance(link, h5py.HardLink):
                raise ValueError("HDF5 paths must be embedded, not external or soft links")
            current = current[part]
        return current

    def open(self, path: str) -> ArrayReader:
        import h5py

        self._check_open()
        if not isinstance(path, str):
            raise ValueError("HDF5 paths must be strings")
        key = path.strip("/")
        if key in self._readers:
            self._readers.move_to_end(key)
            return self._readers[key]
        dataset = self._resolve(path)
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError("Choose an HDF5 dataset, not a group")
        reader = ArrayReader(HDF5Backend(dataset, self._check_open), buffer_bytes=self.buffer_bytes)
        self._readers[key] = reader
        while len(self._readers) > self.max_open_arrays:
            self._readers.popitem(last=False)
        return reader

    @staticmethod
    def _cancel(cancelled: Callable[[], bool] | None) -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError("HDF5 table read cancelled")

    def column_names(self, group: str = "/") -> list[str]:
        """Validate immediate embedded children and list rank-one arrays without reading values."""
        import h5py

        node = self._resolve(group)
        if not isinstance(node, h5py.Group):
            raise ValueError("Choose an HDF5 group containing one-dimensional columns")
        names = []
        for name in sorted(node):
            if not isinstance(node.get(name, getlink=True), h5py.HardLink):
                raise ValueError("HDF5 columns must be embedded, not external or soft links")
            child = node[name]
            if isinstance(child, h5py.Dataset):
                if child.is_virtual or child.external:
                    raise ValueError("HDF5 columns must store data in this file")
                if child.ndim == 1:
                    names.append(name)
        return names

    def _columns(
        self, group: str, columns: Sequence[str] | None, cancelled: Callable[[], bool] | None
    ) -> tuple[list[str], int, int]:
        self._cancel(cancelled)
        if isinstance(columns, (str, bytes)):
            raise ValueError("Columns must be a sequence of names, not a string")
        available = self.column_names(group)
        names = available if columns is None else list(columns)
        if (
            not names
            or any(not isinstance(n, str) or not n or "/" in n for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Columns must be unique non-empty immediate dataset names")
        if set(names) - set(available):
            raise ValueError("Requested column is absent or not a one-dimensional array")
        rows, total = None, 0
        for name in names:
            self._cancel(cancelled)
            reader = self.open(group.rstrip("/") + "/" + name)
            count = reader.backend.spec.shape[0]
            if rows is not None and count != rows:
                raise ValueError("HDF5 columns must have the same number of rows")
            rows = count
            total += reader.backend.spec.logical_bytes
            # Admit a tile before allocating any output. Fixed chunk shapes make this
            # sufficient to reject a single chunk that can never fit the read allowance.
            next(tiles(reader.backend.spec, (slice(None),), buffer_bytes=reader.buffer_bytes), None)
        assert rows is not None
        return names, rows, total

    def read_columns(
        self,
        group: str = "/",
        *,
        columns: Sequence[str] | None = None,
        max_output_bytes: int = 256 << 20,
        stats: ReadStats | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, np.ndarray]:
        """Materialize selected columns after total output admission, with bounded read scratch.

        The output cap is separate from the per-read allowance, native codec scratch and
        metadata. Caller owns all returned arrays. A cancelled/failed call returns no table.
        """
        if (
            isinstance(max_output_bytes, bool)
            or not isinstance(max_output_bytes, int)
            or max_output_bytes < 1
        ):
            raise ValueError("max_output_bytes must be a positive integer")
        names, _, total = self._columns(group, columns, cancelled)
        if total > max_output_bytes:
            raise MemoryError("HDF5 table exceeds the decoded output size limit")
        result = {}
        for name in names:
            self._cancel(cancelled)
            reader = self.open(group.rstrip("/") + "/" + name)
            output = np.empty(reader.backend.spec.shape, dtype=reader.backend.dtype)
            for tile, values in reader.stream((slice(None),), stats=stats, cancelled=cancelled):
                output[tile.destination] = values
                del values
            result[name] = output
        return result

    def inspect_columns(
        self,
        group: str = "/",
        *,
        columns: Sequence[str] | None = None,
        stats: ReadStats | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Scan real-valued column ranges without retaining full columns.

        Native integer extrema retain their precision. Fixed strings are parsed as float64;
        a nonnumeric string marks the whole column nonnumeric. Complex/compound values are
        not treated as real numbers. One tile plus bounded conversion/validity scratch is
        live; this is not a process RSS cap. Row/column metadata are retained separately.
        """
        names, rows, _ = self._columns(group, columns, cancelled)
        result = []
        for name in names:
            self._cancel(cancelled)
            reader = self.open(group.rstrip("/") + "/" + name)
            numeric = reader.backend.dtype.kind in "biufSU"
            invalid = 0
            minimum = maximum = None
            if numeric:
                for _, values in reader.stream((slice(None),), stats=stats, cancelled=cancelled):
                    try:
                        numbers = values.astype(np.float64) if values.dtype.kind in "SU" else values
                    except (ValueError, TypeError):
                        numeric = False
                        break
                    valid = np.isfinite(numbers)
                    invalid += int((~valid).sum())
                    finite = numbers[valid]
                    if finite.size:
                        lo, hi = finite.min().item(), finite.max().item()
                        minimum = lo if minimum is None else min(minimum, lo)
                        maximum = hi if maximum is None else max(maximum, hi)
                    del finite, valid, numbers, values
            result.append(
                dict(name=name, numeric=numeric, invalid=invalid, minimum=minimum, maximum=maximum)
                if numeric
                else dict(name=name, numeric=False, invalid=0)
            )
        return dict(rows=rows, columns=result)

    def close(self) -> None:
        if not self._closed:
            self._check_open()
            self._readers.clear()
            self._file.close()
            self._closed = True

    def __enter__(self) -> HDF5ReadSession:
        self._check_open()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def __getstate__(self):
        raise TypeError("HDF5 sessions own native handles; reopen in each spawned worker")
