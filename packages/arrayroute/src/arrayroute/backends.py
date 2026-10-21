# SPDX-License-Identifier: Apache-2.0
"""Optional adapters. Importing arrayroute does not load a storage engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from .plan import ArraySpec, Selection, plan


class ZarrBackend:
    def __init__(self, array: Any):
        self.array = array
        self.dtype = np.dtype(array.dtype)
        if self.dtype.hasobject or self.dtype.itemsize == 0:
            raise TypeError("A fixed-size, non-object dtype is required")
        self.spec = ArraySpec(tuple(array.shape), tuple(array.chunks), self.dtype.itemsize)

    def read(self, selection: Selection) -> np.ndarray:
        return np.asarray(self.array[selection])


class TensorStoreBackend:
    def __init__(self, array: Any):
        self.array = array
        self.dtype = np.dtype(array.dtype.numpy_dtype)
        self.spec = ArraySpec(
            tuple(array.shape), tuple(array.chunk_layout.read_chunk.shape), self.dtype.itemsize
        )

    def read(self, selection: Selection) -> np.ndarray:
        return np.asarray(self.array[selection].read().result())


class ZarristaBackend:
    def __init__(self, array: Any):
        self.array = array
        self.dtype = np.dtype(array.metadata["data_type"])
        chunks = array.effective_subchunk_shape or array.chunk_shape([0] * array.ndim)
        codecs = array.metadata["codecs"]
        if any(c["name"] == "sharding_indexed" for c in codecs[:-1]):
            # A codec outside the shard requires decoding the complete shard.
            chunks = array.chunk_shape([0] * array.ndim)
        self.spec = ArraySpec(tuple(array.shape), tuple(chunks), self.dtype.itemsize)

    def read(self, selection: Selection) -> np.ndarray:
        if any(isinstance(s, slice) and s.step not in (None, 1) for s in selection):
            return self._strided(selection)
        return self._contiguous(selection)

    def _contiguous(self, selection: Selection) -> np.ndarray:
        # Zarrista retains integer-indexed dimensions, unlike NumPy and Zarr-Python.
        axes = tuple(i for i, s in enumerate(selection) if isinstance(s, int))
        selectors = tuple(slice(s, s + 1) if isinstance(s, int) else s for s in selection)
        return np.squeeze(np.asarray(self.array[selectors]), axis=axes)

    def _strided(self, selection: Selection) -> np.ndarray:
        request = plan(self.spec, selection)
        result = np.empty(request.output_shape, dtype=self.dtype)

        def visit(axis, sources, targets, strides):
            if axis == len(selection):
                result[tuple(targets)] = self._contiguous(tuple(sources))[tuple(strides)]
                return
            sel, chunk = request.selection[axis], self.spec.chunks[axis]
            if isinstance(sel, int):
                visit(axis + 1, [*sources, sel], targets, strides)
                return
            pos, output = sel.start, 0
            while pos < sel.stop:
                end = min(sel.stop, (pos // chunk + 1) * chunk)
                count = (end - pos + sel.step - 1) // sel.step
                last = pos + (count - 1) * sel.step
                visit(
                    axis + 1,
                    [*sources, slice(pos, last + 1)],
                    [*targets, slice(output, output + count)],
                    [*strides, slice(None, None, sel.step)],
                )
                pos += count * sel.step
                output += count

        # The native reader supports unit strides only. Read each touched chunk once,
        # then gather within it; never read the skipped chunks of a bounding box.
        visit(0, [], [], [])
        return result
