# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.benchmark_codecs import CPU_CODECS, PATTERNS, digest, fixture, measure, verify


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("codec", CPU_CODECS)
def test_codec_fixture_roundtrips_and_records_actual_size(pattern, codec):
    pytest.importorskip("numcodecs")
    result = measure(codec, pattern, side=8, repeats=2, threads=1)
    source = fixture(pattern, 8)
    assert result["source_bytes"] == source.nbytes
    assert result["source_sha256"] == digest(source) == digest(fixture(pattern, 8))
    assert result["byte_identical"] and result["verified_roundtrips"] == 3
    assert len(result["measurements"]) == 2
    assert all(item["encoded_bytes"] > 0 for item in result["measurements"])
    assert all(item["host_encode_s"] > 0 for item in result["measurements"])
    assert not result["hardware"]


def test_lossless_comparison_includes_nan_payload_and_signed_zero():
    values = np.array([0, 0x80000000, 0x7FC00001, 0x7FC00002, 0x7F800000], dtype="<u4")
    verify(values.view("<f4"), values.tobytes())
    changed = values.copy()
    changed[2] = changed[3]
    with pytest.raises(ValueError, match="source bytes"):
        verify(values.view("<f4"), changed.tobytes())
    with pytest.raises(ValueError, match="source bytes"):
        verify(values, values.tobytes()[:-1])


def test_fixture_rejects_unknown_patterns_and_invalid_sizes():
    with pytest.raises(ValueError):
        fixture("other", 8)
    with pytest.raises(ValueError):
        fixture("velocity", 1)


@pytest.mark.parametrize("codec", CPU_CODECS)
@pytest.mark.parametrize("order", ["C", "F"])
def test_supplied_array_preserves_bits_and_source_file(tmp_path, codec, order):
    pytest.importorskip("numcodecs")
    source = (
        np.array([0, 0x80000000, 0x7FC00001, 0x7FC00002, 0x7F800000, 0xFF800000], dtype="<u4")
        .view("<f4")
        .reshape(2, 3)
    )
    source = np.array(source, order=order)
    path = tmp_path / "field.npy"
    np.save(path, source)
    before = path.read_bytes()
    result = measure(codec, "unused", 8, 1, 1, input_path=path, max_input_bytes=24)
    assert result["pattern"] == "input" and result["side"] is None
    assert result["shape"] == (2, 3) and result["source_sha256"] == digest(source)
    assert result["byte_identical"] and path.read_bytes() == before


@pytest.mark.parametrize(
    "source",
    [
        np.array(1),
        np.array([]),
        np.array([1j]),
        np.array(["a"]),
        np.array([object()], dtype=object),
    ],
)
def test_supplied_array_rejects_unsupported_input(tmp_path, source):
    from arrayroute.benchmark_codecs import load_input

    path = tmp_path / "field.npy"
    np.save(path, source)
    with pytest.raises(ValueError):
        load_input(path, max_bytes=1024)


def test_input_limit_checked_before_contiguous_copy(tmp_path, monkeypatch):
    from arrayroute.benchmark_codecs import load_input

    path = tmp_path / "field.npy"
    np.save(path, np.asfortranarray(np.ones((4, 4), dtype="f4")))
    monkeypatch.setattr(np, "ascontiguousarray", lambda _: pytest.fail("Oversized input copied"))
    with pytest.raises(MemoryError, match="allowance"):
        load_input(path, max_bytes=63)


def test_archive_is_not_a_single_array(tmp_path):
    from arrayroute.benchmark_codecs import load_input

    path = tmp_path / "archive.npz"
    np.savez(path, a=np.ones(4))
    with pytest.raises(ValueError, match="not an archive"):
        load_input(path, max_bytes=1024)


def test_cli_runs_supplied_array_in_fresh_processes(tmp_path, monkeypatch):
    import json
    import sys

    from arrayroute.benchmark_codecs import main

    pytest.importorskip("numcodecs")
    path = tmp_path / "field.npy"
    source = np.arange(100, dtype="f4").reshape(10, 10)
    np.save(path, source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark-codecs",
            "--input",
            str(path),
            "--output",
            str(tmp_path / "report"),
            "--codecs",
            "zstd",
            "blosc-lz4-bitshuffle",
            "--repeats",
            "1",
            "--process-repeats",
            "2",
            "--threads",
            "1",
        ],
    )
    main()
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert not report["failures"] and len(report["runs"]) == 4
    assert report["input_identity"] == {
        "source_sha256": digest(source),
        "shape": [10, 10],
        "dtype": "<f4",
    }
    assert all(row["source_sha256"] == digest(source) for row in report["runs"])
    assert report["method"].startswith("Supplied immutable NPY input")


def test_cli_rejects_changed_shape_even_when_payload_bytes_match(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys

    from arrayroute.benchmark_codecs import main

    pytest.importorskip("numcodecs")
    path = tmp_path / "field.npy"
    source = np.arange(100, dtype="f4").reshape(10, 10)
    np.save(path, source)
    real_run = subprocess.run

    def changed(*args, **kwargs):
        np.save(path, source.reshape(2, 50))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", changed)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark-codecs",
            "--input",
            str(path),
            "--output",
            str(tmp_path / "report"),
            "--codecs",
            "zstd",
            "--repeats",
            "1",
            "--process-repeats",
            "1",
            "--threads",
            "1",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert not report["runs"] and len(report["failures"]) == 1
    assert "changed" in report["failures"][0]["error"]


@pytest.mark.parametrize("codec", CPU_CODECS)
@pytest.mark.parametrize("dtype", ["<u4", ">u4"])
def test_axis_layout_preserves_original_bits_and_shape(tmp_path, codec, dtype):
    pytest.importorskip("numcodecs")
    bits = np.array([0, 0x80000000, 0x7FC00001, 0x7FC00002, 0x7F800000, 0xFF800000], dtype=dtype)
    source = np.tile(bits, 4).view(dtype.replace("u", "f")).reshape(2, 3, 4)
    path = tmp_path / "field.npy"
    np.save(path, source)
    result = measure(codec, "unused", 8, 1, 1, input_path=path, axis_order=(2, 0, 1))
    assert result["shape"] == (2, 3, 4)
    assert result["packed_shape"] == (4, 2, 3)
    assert result["source_sha256"] == digest(source)
    assert result["byte_identical"]


@pytest.mark.parametrize("axes", [(0, 0), (0,), (1, 2), (True, 0)])
def test_invalid_axis_order_rejected(tmp_path, axes):
    path = tmp_path / "field.npy"
    np.save(path, np.ones((2, 3), dtype="f4"))
    with pytest.raises(ValueError, match="permutation"):
        measure("zstd", "unused", 8, 1, 1, input_path=path, axis_order=axes)


def test_permutation_gpu_not_claimed_or_imported(tmp_path):
    path = tmp_path / "field.npy"
    np.save(path, np.ones((2, 3), dtype="f4"))
    with pytest.raises(ValueError, match="only for CPU"):
        measure("gpu-lz4", "unused", 8, 1, 1, input_path=path, axis_order=(1, 0))
