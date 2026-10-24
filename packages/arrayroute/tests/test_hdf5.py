# SPDX-License-Identifier: Apache-2.0
"""Embedded HDF5 reads preserve values and admit storage costs before payload I/O."""

import pickle

import numpy as np
import pytest

from arrayroute.hdf5 import HDF5Backend, HDF5ReadSession
from arrayroute.plan import ArraySpec, DecodeWindow, plan, tiles
from arrayroute.reader import ReadStats

h5py = pytest.importorskip("h5py")


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize(
    "selection",
    [
        (slice(None), slice(None), slice(None)),
        (slice(1, 9, 2), -1, slice(0, 7, 3)),
        (2, 3, 4),
        (slice(0, 0), slice(None), slice(None)),
    ],
)
def test_hyperslabs_match_numpy_and_stream_replay(tmp_path, compressed, selection):
    path = tmp_path / "fields.h5"
    values = np.arange(9 * 8 * 7, dtype="<f4").reshape(9, 8, 7)
    values[1, 2, 3] = -0.0
    values[4, 3, 2] = np.nan
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "pressure",
            data=values,
            **({"chunks": (3, 2, 7), "compression": "gzip"} if compressed else {}),
        )
    expected = values[selection]
    with HDF5ReadSession(path, buffer_bytes=512) as s:
        reader = s.open("pressure")
        outputs = []
        for _ in range(2):
            output = np.empty(expected.shape, expected.dtype)
            stats = ReadStats()
            for tile, data in reader.stream(selection, stats=stats):
                assert tile.plan.buffer_bytes_upper_bound <= 512
                output[tile.destination] = data
            assert output.tobytes() == expected.tobytes()
            assert stats.output_bytes == expected.nbytes
            if not compressed:
                assert stats.planned_chunk_visits == 0
            outputs.append(output.tobytes())
        assert outputs[0] == outputs[1]
        if expected.nbytes < 256:
            assert reader.read(selection).tobytes() == expected.tobytes()


def test_contiguous_plan_does_not_invent_decoded_chunks():
    spec = ArraySpec((1000000,), (1000000,), 8, contiguous=True)
    request = plan(spec, (slice(None),))
    assert request.chunks_touched == request.decoded_bytes_upper_bound == 0
    chunks = list(tiles(spec, (slice(None),), buffer_bytes=65536))
    assert 100 < len(chunks) < 256
    assert sum(t.plan.output_bytes for t in chunks) == 8000000
    assert all(t.plan.buffer_bytes_upper_bound <= 65536 for t in chunks)
    with pytest.raises(ValueError, match="Contiguous"):
        ArraySpec((10,), (10,), 4, contiguous=True, decode_window=DecodeWindow(1))
    with pytest.raises(TypeError, match="boolean"):
        ArraySpec((10,), (10,), 4, contiguous=1)


def test_projection_admission_and_cancel_before_payload(tmp_path, monkeypatch):
    path = tmp_path / "columns.h5"
    with h5py.File(path, "w") as f:
        f["x"] = np.arange(100)
        f["label"] = np.array([b"train"] * 100)
        f.create_dataset("unused_volume", shape=(100000, 100000), dtype="f8")
        f.create_dataset("unused_strings", data=["huge"] * 100, dtype=h5py.string_dtype())
    calls = []
    original = HDF5Backend.read

    def read(self, selection):
        calls.append(self._dataset.name)
        return original(self, selection)

    monkeypatch.setattr(HDF5Backend, "read", read)
    with HDF5ReadSession(path, buffer_bytes=128, max_open_arrays=1) as s:
        with pytest.raises(MemoryError, match="output"):
            s.read_columns(columns=["x"], max_output_bytes=799)
        with pytest.raises(InterruptedError):
            s.read_columns(columns=["x"], cancelled=lambda: True)
        assert calls == []
        table = s.read_columns(columns=["label", "x"])
        assert list(table) == ["label", "x"]
        assert table["x"].tobytes() == np.arange(100).tobytes()
        assert set(calls) == {"/label", "/x"}
        with pytest.raises(TypeError, match="fixed-size"):
            s.open("unused_strings")
        for names in (["x", "x"], ["missing"], "x"):
            with pytest.raises(ValueError):
                s.read_columns(columns=names)


def test_chunk_too_large_rejected_before_payload(tmp_path, monkeypatch):
    path = tmp_path / "chunk.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.arange(100), chunks=(100,), compression="gzip")

    def forbidden(*args):
        pytest.fail("Payload read before admission")

    monkeypatch.setattr(HDF5Backend, "read", forbidden)
    with HDF5ReadSession(path, buffer_bytes=128) as s:
        with pytest.raises(MemoryError):
            s.read_columns()


def test_streamed_inspection_keeps_integer_precision_and_missing_counts(tmp_path):
    path = tmp_path / "stats.h5"
    integers = np.arange(100, dtype=np.int64) + 2**60
    with h5py.File(path, "w") as f:
        f["integer"] = integers
        f["real"] = np.array([np.nan, np.inf, -np.inf, -2, 3] * 20)
        f["text"] = np.array([b"1.5", b"nan"] * 50)
        f["labels"] = np.array([b"train"] * 100)
        f["complex"] = np.ones(100, dtype=complex)
    with HDF5ReadSession(path, buffer_bytes=128) as s:
        stats = ReadStats()
        report = s.inspect_columns(stats=stats)
    by_name = {c["name"]: c for c in report["columns"]}
    assert report["rows"] == 100
    assert by_name["integer"]["minimum"] == int(integers[0])
    assert by_name["integer"]["maximum"] == int(integers[-1])
    assert by_name["real"]["invalid"] == 60
    assert by_name["real"]["minimum"] == -2
    assert by_name["text"]["invalid"] == 50
    assert not by_name["labels"]["numeric"]
    assert not by_name["complex"]["numeric"]
    assert stats.requests > 10


@pytest.mark.parametrize("kind", ["soft", "external", "virtual", "raw"])
def test_external_data_rejected(tmp_path, kind):
    path = tmp_path / "unsafe.h5"
    with h5py.File(path, "w") as f:
        if kind == "soft":
            f["x"] = h5py.SoftLink("/other")
        elif kind == "external":
            f["x"] = h5py.ExternalLink("elsewhere.h5", "/x")
        elif kind == "virtual":
            layout = h5py.VirtualLayout((20,), "f4")
            layout[:] = h5py.VirtualSource("elsewhere.h5", "x", (20,))
            f.create_virtual_dataset("x", layout)
        else:
            f.create_dataset("x", shape=(20,), dtype="f4", external=[("raw.bin", 0, 80)])
    with HDF5ReadSession(path) as s:
        with pytest.raises(ValueError):
            s.open("x")
        with pytest.raises(ValueError):
            s.read_columns()


def test_nested_link_lifetime_and_process_ownership(tmp_path, monkeypatch):
    path = tmp_path / "lifetime.h5"
    with h5py.File(path, "w") as f:
        f.create_group("table")["x"] = np.arange(20)
        f["alias"] = h5py.SoftLink("/table")
    with HDF5ReadSession(path) as s:
        with pytest.raises(ValueError, match="soft"):
            s.open("alias/x")
        with pytest.raises(TypeError, match="native handles"):
            pickle.dumps(s)
        reader = s.open("table/x")
        array = reader.read((slice(None),))
        with monkeypatch.context() as m:
            m.setattr("arrayroute.hdf5.os.getpid", lambda: -1)
            with pytest.raises(RuntimeError, match="fork"):
                reader.read((slice(None),))
    assert array.tobytes() == np.arange(20).tobytes()
    with pytest.raises(RuntimeError, match="closed"):
        reader.read((slice(None),))
    s.close()


def test_unsupported_shapes_and_types(tmp_path):
    path = tmp_path / "types.h5"
    with h5py.File(path, "w") as f:
        f["scalar"] = 1
        f.create_dataset("null", dtype="f4")
        f.create_dataset("references", (20,), dtype=h5py.ref_dtype)
        f["short"] = np.arange(19)
        f["long"] = np.arange(20)
    with HDF5ReadSession(path) as s:
        for name in ["scalar", "null"]:
            with pytest.raises(ValueError, match="non-null"):
                s.open(name)
        with pytest.raises(TypeError, match="fixed-size"):
            s.open("references")
        with pytest.raises(ValueError, match="same number"):
            s.read_columns(columns=["short", "long"])


def test_array_valued_dtype_rejected_before_payload(tmp_path, monkeypatch):
    path = tmp_path / "array-dtype.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", shape=(20,), dtype=np.dtype(("f4", (3,))))

    def forbidden(*args):
        pytest.fail("Array-valued dtype reached payload reading")

    monkeypatch.setattr(HDF5Backend, "read", forbidden)
    with HDF5ReadSession(path) as s:
        with pytest.raises(TypeError, match="dimension expansion"):
            s.read_columns()


def test_scalar_and_metadata_admission(tmp_path):
    path = tmp_path / "scalar.h5"
    with h5py.File(path, "w") as f:
        f["scalar"] = np.float64(-0.0)
        f["text"] = "variable length"
        f["link"] = h5py.SoftLink("/scalar")
    with HDF5ReadSession(path, buffer_bytes=8) as s:
        assert s.metadata("scalar").shape == ()
        assert s.read_scalar("scalar").tobytes() == np.float64(-0.0).tobytes()
        with pytest.raises(TypeError, match="fixed-size"):
            s.read_scalar("text")
        with pytest.raises(ValueError, match="soft"):
            s.metadata("link")
    with HDF5ReadSession(path, buffer_bytes=4) as s:
        with pytest.raises(MemoryError):
            s.read_scalar("scalar")


@pytest.mark.parametrize("time_chunk", [None, 1, 2])
def test_large_strided_frames_match_native_without_repeated_chunk_decode(
    tmp_path, monkeypatch, time_chunk
):
    path = tmp_path / "frames.h5"
    values = np.arange(8 * 256 * 1024, dtype=np.float32).reshape(8, 256, 1024)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "frames",
            data=values,
            **({"chunks": (time_chunk, 256, 1024), "compression": "gzip"} if time_chunk else {}),
        )
    calls = []
    original = h5py.Dataset.read_direct

    def read(self, *args, **kwargs):
        calls.append(kwargs["source_sel"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "read_direct", read)
    for mode in ["native", "auto"]:
        calls.clear()
        with HDF5ReadSession(path, strided_reads=mode) as session:
            result = session.open("frames").read((slice(None, None, 2), slice(None), slice(None)))
        assert result.tobytes() == values[::2].tobytes()
        assert len(calls) == (4 if mode == "auto" and time_chunk in (None, 1) else 1)
    with pytest.raises(ValueError, match="strided_reads"):
        HDF5ReadSession(path, strided_reads="unknown")
