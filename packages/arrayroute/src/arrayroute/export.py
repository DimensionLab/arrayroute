# SPDX-License-Identifier: Apache-2.0
"""Stream a selection into an ordinary NumPy file without exposing a partial output."""

from __future__ import annotations

import math
import operator
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from .plan import Selection
from .reader import ArrayReader, ReadStats


def _write_block(output: BinaryIO, block: np.ndarray, buffer: int) -> None:
    with np.nditer(
        block,
        flags=["external_loop", "buffered", "zerosize_ok"],
        # NumPy's stub misspells the runtime flag "contig" as "config".
        op_flags=cast(Any, ["readonly", "contig"]),
        order="C",
        buffersize=max(1, buffer // block.dtype.itemsize),
    ) as iterator:
        for chunk in iterator:
            pending = memoryview(np.asarray(chunk)).cast("B")
            while pending:
                written = output.write(pending)
                if not written:
                    raise OSError("Could not write array output")
                pending = pending[written:]


def save_npy(
    reader: ArrayReader,
    selection: Selection,
    path: str | Path,
    *,
    max_output_bytes: int,
    write_buffer_bytes: int = 1 << 20,
    stats: ReadStats | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Publish once, with no replacement; the payload limit excludes the small NPY header.

    The reader's budget plus write_buffer_bytes bounds explicit array buffers, not OS
    caches or backend codec scratch. Same-directory hard links must be supported. A
    failure/cancellation removes the temporary output; the destination stays absent.
    """
    limit, buffer = operator.index(max_output_bytes), operator.index(write_buffer_bytes)
    if limit < 0 or buffer < reader.backend.dtype.itemsize:
        raise ValueError("Output limit must be nonnegative and write buffer fit one element")
    request = reader.plan(selection)
    if request.output_bytes > limit:
        raise ValueError(f"Selection needs {request.output_bytes} bytes; output limit is {limit}")
    if reader.backend.dtype.hasobject:
        raise ValueError("Object arrays cannot be exported without pickle")
    target = Path(path)
    if os.path.lexists(target):
        raise FileExistsError(target)
    if cancelled is not None and cancelled():
        raise InterruptedError("Array export cancelled")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w+b", buffering=0) as output:
            np.lib.format.write_array_header_2_0(
                output,
                {
                    "descr": np.lib.format.dtype_to_descr(reader.backend.dtype),
                    "fortran_order": False,
                    "shape": request.output_shape,
                },
            )
            header_size = output.tell()
            output.truncate(header_size + request.output_bytes)
            strides = tuple(
                math.prod(request.output_shape[i + 1 :]) for i in range(len(request.output_shape))
            )
            for tile, values in reader.stream(selection, stats=stats, cancelled=cancelled):
                # Coalesce complete trailing dimensions into contiguous file extents.
                axis = max(0, values.ndim - 1)
                while axis > 0 and (
                    tile.destination[axis].start == 0
                    and tile.destination[axis].stop == request.output_shape[axis]
                ):
                    axis -= 1
                for prefix in np.ndindex(values.shape[:axis]):
                    if cancelled is not None and cancelled():
                        raise InterruptedError("Array export cancelled")
                    starts = tuple(s.start for s in tile.destination)
                    offset = sum(s * stride for s, stride in zip(starts, strides, strict=True))
                    offset += sum(i * strides[k] for k, i in enumerate(prefix))
                    output.seek(header_size + offset * values.dtype.itemsize)
                    _write_block(output, values[prefix] if prefix else values, buffer)
                del values
            os.fsync(output.fileno())
        if cancelled is not None and cancelled():
            raise InterruptedError("Array export cancelled")
        # Atomic no-clobber publication, including a competing creator or dangling symlink.
        os.link(temporary, target)
    finally:
        os.unlink(temporary)
