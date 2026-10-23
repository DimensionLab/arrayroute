# SPDX-License-Identifier: Apache-2.0
"""Optional adapters. Importing arrayroute does not load a storage engine."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .plan import ArraySpec, DecodeWindow, ReadPlan, Selection, plan

if TYPE_CHECKING:
    from .reader import Backend


class DenseGatherBackend:
    """Opt into native bounding-box reads when the plan permits their live buffers."""

    def __init__(self, backend: Backend):
        self.backend = backend
        self.dtype = backend.dtype
        self.spec = replace(backend.spec, dense_gather=True)

    def read(self, selection: Selection) -> np.ndarray:
        # Unplanned calls cannot authorize an additional intermediate allocation.
        return self.backend.read(selection)

    def read_plan(self, request: ReadPlan) -> np.ndarray:
        if request.strategy == "native":
            return self.backend.read(request.selection)
        if request.strategy != "dense-gather" or request.read_selection is None:
            raise ValueError("Dense gather requires its planned bounding selection")
        dense = self.backend.read(request.read_selection)
        gather = tuple(slice(None, None, s.step) for s in request.selection if isinstance(s, slice))
        # A view would retain the full bounding array beyond the read reservation.
        return dense[gather].copy(order="C")


def _grid_regions(
    selection: Selection, shape: tuple[int, ...], axis: int = 0, prefix: Selection = (),
) -> Iterator[Selection]:
    """Visit only intersected grid blocks, retaining the original stride phase."""
    if axis == len(selection):
        yield prefix
        return
    sel = selection[axis]
    if isinstance(sel, int):
        yield from _grid_regions(selection, shape, axis + 1, (*prefix, sel))
        return
    pos = sel.start
    while pos < sel.stop:
        end = min(sel.stop, (pos // shape[axis] + 1) * shape[axis])
        yield from _grid_regions(
            selection, shape, axis + 1, (*prefix, slice(pos, end, sel.step)),
        )
        pos += ((end - pos + sel.step - 1) // sel.step) * sel.step


class ZarrBackend:
    def __init__(self, array: Any):
        self.array = array
        self.dtype = np.dtype(array.dtype)
        if self.dtype.hasobject or self.dtype.itemsize == 0:
            raise TypeError("A fixed-size, non-object dtype is required")
        chunks = tuple(array.chunks)
        metadata = getattr(array, 'metadata', None)
        codecs = getattr(metadata, 'codecs', ())
        if (metadata is not None and getattr(metadata, 'sharding_codec', None) is not None
                and len(codecs) != 1):
            # Outer transforms/compressors disable partial shard decoding in Zarr.
            chunks = tuple(metadata.chunk_grid.chunk_shape)
        self.spec = ArraySpec(tuple(array.shape), chunks, self.dtype.itemsize)

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
        outer_codec = len(codecs) != 1 and any(
            (c if isinstance(c, str) else c["name"]) == "sharding_indexed" for c in codecs
        )
        self._partial_shards = array.is_sharded and not outer_codec
        if outer_codec:
            # A codec outside the shard requires decoding the complete shard.
            chunks = array.chunk_shape([0] * array.ndim)
        window = DecodeWindow(1, strided_only=True) if self._partial_shards else None
        self.spec = ArraySpec(tuple(array.shape), tuple(chunks), self.dtype.itemsize, window)

    def read(self, selection: Selection) -> np.ndarray:
        if any(isinstance(s, slice) and s.step not in (None, 1) for s in selection):
            return self._strided(selection)
        return self._contiguous(selection)

    def _contiguous(self, selection: Selection) -> np.ndarray:
        # Zarrista retains integer-indexed dimensions, unlike NumPy and Zarr-Python.
        axes = tuple(i for i, s in enumerate(selection) if isinstance(s, int))
        selectors = tuple(slice(s, s + 1) if isinstance(s, int) else s for s in selection)
        return np.squeeze(np.asarray(self.array[selectors]), axis=axes)

    def read_shard_region(self, selection: Selection) -> np.ndarray:
        """Gather a single normalized region from a full shard, under an admitted plan."""
        shape = tuple(self.array.chunk_shape([0] * self.array.ndim))
        starts = [s if isinstance(s, int) else s.start for s in selection]
        coordinates = [s // r for s, r in zip(starts, shape, strict=True)]
        for s, r in zip(selection, shape, strict=True):
            if isinstance(s, slice) and (s.stop <= s.start or s.start // r != (s.stop - 1) // r):
                raise ValueError("Bulk selection must stay within one shard")
        dense = np.asarray(self.array.retrieve_chunk(coordinates, concurrent_target=1))
        local = tuple(
            s % r if isinstance(s, int) else
            slice(s.start % r, s.start % r + s.stop - s.start, s.step)
            for s, r in zip(selection, shape, strict=True)
        )
        return dense[local].copy(order="C")

    def _strided(self, selection: Selection) -> np.ndarray:
        request = plan(self.spec, selection)
        result = np.empty(request.output_shape, dtype=self.dtype)
        cache = self.array.shard_cache() if self._partial_shards else None
        shard_shape = (
            tuple(self.array.chunk_shape([0] * self.array.ndim)) if cache is not None else ()
        )
        shard_count = (
            plan(ArraySpec(self.spec.shape, shard_shape, self.spec.itemsize),
                 request.selection).chunks_touched if cache is not None else 0
        )
        one_chunk_per_shard = shard_count > 1 and shard_count == request.chunks_touched

        def visit(recur, region, axis, sources, targets, strides):
            if axis == len(selection):
                if cache is None:
                    value = self._contiguous(tuple(sources))
                else:
                    if one_chunk_per_shard:
                        cache.clear()
                    starts = [s if isinstance(s, int) else s.start for s in sources]
                    chunks = self.spec.chunks
                    coordinates = [p // c for p, c in zip(starts, chunks, strict=True)]
                    local = tuple(
                        s % c if isinstance(s, int)
                        else slice(s.start % c, s.start % c + s.stop - s.start)
                        for s, c in zip(sources, chunks, strict=True)
                    )
                    value = np.asarray(self.array.retrieve_subchunk(
                        coordinates, shard_cache=cache))[local]
                result[tuple(targets)] = value[tuple(strides)]
                return
            sel, chunk = region[axis], self.spec.chunks[axis]
            if isinstance(sel, int):
                recur(recur, region, axis + 1, [*sources, sel], targets, strides)
                return
            pos = sel.start
            original = request.selection[axis]
            assert isinstance(original, slice)
            output = (pos - original.start) // sel.step
            while pos < sel.stop:
                end = min(sel.stop, (pos // chunk + 1) * chunk)
                count = (end - pos + sel.step - 1) // sel.step
                last = pos + (count - 1) * sel.step
                recur(
                    recur,
                    region,
                    axis + 1,
                    [*sources, slice(pos, last + 1)],
                    [*targets, slice(output, output + count)],
                    [*strides, slice(None, None, sel.step)],
                )
                pos += count * sel.step
                output += count

        # The native reader supports unit strides only. Read each touched chunk once,
        # then gather within it; never read the skipped chunks of a bounding box.
        # Passing the callback avoids a closure cycle retaining result until cyclic GC.
        if cache is None or shard_count <= 1 or one_chunk_per_shard:
            visit(visit, request.selection, 0, [], [], [])
        else:
            for region in _grid_regions(request.selection, shard_shape):
                # Finish each shard before discarding its index; retain no global inventory.
                cache.clear()
                visit(visit, region, 0, [], [], [])
        return result
