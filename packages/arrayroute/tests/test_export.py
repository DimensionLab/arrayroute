# SPDX-License-Identifier: Apache-2.0
import weakref

import numpy as np
import pytest

from arrayroute import ArrayReader, ArraySpec, ReadStats
from arrayroute.export import save_npy


class Backend:
    def __init__(self, values):
        self.values, self.dtype = values, values.dtype
        self.spec = ArraySpec(values.shape, (2, 3, 2), values.dtype.itemsize)
        self.references = []

    def read(self, selection):
        assert all(reference() is None for reference in self.references)
        result = np.array(self.values[selection], dtype=self.dtype, copy=True, order="F")
        self.references.append(weakref.ref(result))
        return result


@pytest.mark.parametrize("dtype", ["<f4", ">f4", "<u8"])
@pytest.mark.parametrize(
    "selection",
    [
        (slice(None), slice(None), slice(None)),
        (slice(1, 9, 2), slice(1, 8, 3), 2),
        (1, 2, 3),
        (slice(0, 0), slice(None), slice(None)),
    ],
)
def test_stream_export_matches_numpy_without_retaining_tiles(tmp_path, dtype, selection):
    source = np.arange(10 * 9 * 7).reshape(10, 9, 7).astype(dtype)
    reader = ArrayReader(Backend(source), buffer_bytes=256)
    output = tmp_path / "selection.npy"
    expected = np.asarray(source[selection], dtype=source.dtype)
    stats = ReadStats()
    save_npy(
        reader,
        selection,
        output,
        max_output_bytes=expected.nbytes,
        write_buffer_bytes=32,
        stats=stats,
    )
    actual = np.load(output, allow_pickle=False)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    assert stats.output_bytes == expected.nbytes
    assert list(tmp_path.iterdir()) == [output]


def test_limits_cancellation_and_failures_never_publish_partial_data(tmp_path):
    source = np.arange(10 * 9 * 7, dtype=np.float32).reshape(10, 9, 7)
    reader = ArrayReader(Backend(source), buffer_bytes=256)
    output = tmp_path / "selection.npy"
    selection = (slice(None),) * 3
    with pytest.raises(ValueError, match="output limit"):
        save_npy(reader, selection, output, max_output_bytes=10)
    assert not reader.backend.references
    with pytest.raises(InterruptedError):
        save_npy(
            reader,
            selection,
            output,
            max_output_bytes=source.nbytes,
            cancelled=lambda: len(reader.backend.references) > 2,
        )
    assert len(reader.backend.references) == 3
    assert not list(tmp_path.iterdir())
    reader.buffer_bytes = 1
    with pytest.raises(MemoryError):
        save_npy(reader, selection, output, max_output_bytes=source.nbytes)
    assert not list(tmp_path.iterdir())


def test_competing_destination_is_preserved(tmp_path, monkeypatch):
    import os

    source = np.arange(2 * 3 * 2, dtype=np.int32).reshape(2, 3, 2)
    reader = ArrayReader(Backend(source), buffer_bytes=256)
    output = tmp_path / "selection.npy"
    link = os.link

    def competing_writer(temp, target):
        output.write_bytes(b"another writer")
        link(temp, target)

    monkeypatch.setattr(os, "link", competing_writer)
    with pytest.raises(FileExistsError):
        save_npy(reader, (slice(None),) * 3, output, max_output_bytes=source.nbytes)
    assert output.read_bytes() == b"another writer"
    assert list(tmp_path.iterdir()) == [output]


def test_dangling_symlinks_and_object_arrays_are_rejected(tmp_path):
    output = tmp_path / "selection.npy"
    output.symlink_to(tmp_path / "missing")
    source = np.ones((2, 3, 2), dtype=np.float32)
    with pytest.raises(FileExistsError):
        save_npy(
            ArrayReader(Backend(source)), (slice(None),) * 3, output, max_output_bytes=source.nbytes
        )
    objects = source.astype(object)
    with pytest.raises(ValueError, match="pickle"):
        save_npy(
            ArrayReader(Backend(objects)),
            (slice(None),) * 3,
            tmp_path / "object.npy",
            max_output_bytes=objects.nbytes,
        )
