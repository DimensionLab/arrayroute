# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.benchmark import _digest, create_fixture


@pytest.mark.parametrize("frame_workers", [1, 4])
@pytest.mark.parametrize("engine", ["zarr-python", "zarrista"])
def test_fixture_native_write_and_receipt_are_independently_readable(
    tmp_path, engine, frame_workers
):
    if engine == "zarrista":
        pytest.importorskip(engine)
    import zarr

    root = tmp_path / "fixture"
    receipt = create_fixture(root, frames=4, edge=32, writer=engine, frame_workers=frame_workers)
    assert receipt["frame_workers"] == frame_workers
    assert receipt["max_in_flight_frames"] == frame_workers
    assert receipt["logical_bytes_per_layout"] == 4 * 32**3 * 3 * 4
    for name in receipt["layouts"]:
        a = zarr.open_array(root / f"{name}.zarr", mode="r")
        for t in range(4):
            assert _digest(np.asarray(a[t])) == receipt["frame_sha256"][t]
    with pytest.raises(FileExistsError):
        create_fixture(root, frames=4, edge=32)


@pytest.mark.parametrize("writer", ["zarr-python", "zarrista"])
def test_one_materialized_layout_is_readable_by_all_engines(tmp_path, writer):
    import json

    from arrayroute.benchmark_materialized import scan

    if writer == "zarrista":
        pytest.importorskip(writer)
    root = tmp_path / "fixture"
    receipt = create_fixture(root, frames=4, edge=32, writer=writer, layouts=["sharded"])
    assert set(receipt["layouts"]) == {"sharded"}
    assert receipt["selected_layouts"] == ["sharded"]
    assert receipt["concurrent_layout_writers"] == 1
    assert not (root / "frame.zarr").exists()
    assert not (root / "brick.zarr").exists()
    assert json.loads((root / "progress.json").read_text())["selected_layouts"] == ["sharded"]
    for engine in ["zarr-python", "tensorstore", "zarrista"]:
        pytest.importorskip("zarr" if engine == "zarr-python" else engine)
        report = scan(root, tmp_path / f"scan-{engine}", engine=engine)
        assert report["complete"]
        assert report["verified_logical_bytes"] == receipt["logical_bytes_per_layout"]


@pytest.mark.parametrize("options", [
    {"layouts": []}, {"layouts": ["unknown"]}, {"layouts": ["sharded", "sharded"]},
    {"writer": "unknown"}, {"frame_workers": 0}, {"frame_workers": True},
    {"frame_workers": 1.5},
])
def test_invalid_fixture_choices_do_not_create_partial_output(tmp_path, options):
    root = tmp_path / "fixture"
    with pytest.raises(ValueError):
        create_fixture(root, frames=4, edge=32, **options)
    assert not root.exists()


def test_parallel_write_failure_never_publishes_completion(tmp_path, monkeypatch):
    import json

    import arrayroute.benchmark as benchmark
    from arrayroute.benchmark_materialized import reference_frame

    original = benchmark._digest
    broken = original(reference_frame(1, 32))

    def fail_frame(value):
        sha = original(value)
        if sha == broken:
            raise OSError("injected frame failure")
        return sha

    monkeypatch.setattr(benchmark, "_digest", fail_frame)
    root = tmp_path / "partial"
    with pytest.raises(OSError, match="injected frame failure"):
        create_fixture(root, frames=8, edge=32, layouts=["sharded"], frame_workers=4)
    assert not (root / "fixture.json").exists()
    assert not (root / "materialization.json").exists()
    progress = root / "progress.json"
    assert not progress.exists() or json.loads(progress.read_text())["frames_written"] < 8
