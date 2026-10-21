# SPDX-License-Identifier: Apache-2.0
import json
import time

import numpy as np
import pytest
import zarr

from arrayroute.benchmark import create_fixture
from arrayroute.benchmark_materialized import digest, reference_frame, scan
from arrayroute.permutation import Permutation


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "data"
    create_fixture(root, frames=4, edge=32)
    # The expensive physical inventory is not a precondition for numeric verification.
    (root / "fixture.json").unlink()
    return root


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore", "zarrista"])
@pytest.mark.parametrize("layout", ["frame", "brick", "sharded"])
def test_complete_scan_independently_verifies_every_frame(fixture, tmp_path, engine, layout):
    pytest.importorskip("zarr" if engine == "zarr-python" else engine)
    out = tmp_path / "report"
    report = scan(fixture, out, engine=engine, layout=layout)
    assert report["complete"] and report["state"] == "complete"
    assert report["frames_verified"] == 4
    assert report["verified_logical_bytes"] == 4 * 32**3 * 3 * 4
    lines = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
    assert sorted(line["frame"] for line in lines) == list(range(4))
    assert all(line["expected_sha256"] == line["actual_sha256"] for line in lines)
    with pytest.raises(FileExistsError):
        scan(fixture, out, engine=engine, layout=layout)


def test_regenerator_matches_previously_verified_cloud_frame():
    assert digest(reference_frame(0, 128)) == (
        "e57096032394db1696cfe8484c0f2a25d13b129d791f186740b2732b949d2b53"
    )


def test_corruption_is_recorded_and_never_marked_complete(fixture, tmp_path):
    array = zarr.open_array(fixture / "sharded.zarr", mode="r+")
    frame = Permutation(4, seed=991)[0]
    array[frame] = np.zeros((32, 32, 32, 3), dtype=np.float32)
    output = tmp_path / "corrupt"
    with pytest.raises(ValueError, match="differs"):
        scan(fixture, output, engine="zarr-python")
    report = json.loads((output / "report.json").read_text())
    assert not report["complete"] and report["state"] == "failed"
    assert report["frames_verified"] == 0
    assert json.loads((output / "frames.jsonl").read_text())["verified"] is False


def test_deadline_and_pilot_are_partial(fixture, tmp_path):
    report = scan(fixture, tmp_path / "expired", engine="zarr-python", deadline=time.time() - 1)
    assert not report["complete"] and report["state"] == "deadline"
    assert report["frames_verified"] == 0
    report = scan(fixture, tmp_path / "pilot", engine="zarr-python", limit=2)
    assert not report["complete"] and report["state"] == "limited"
    assert report["frames_verified"] == 2


def test_incomplete_materialization_refused(fixture, tmp_path):
    progress = fixture / "progress.json"
    info = json.loads(progress.read_text())
    info["frames_written"] -= 1
    progress.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="incomplete"):
        scan(fixture, tmp_path / "report", engine="zarr-python")
    assert not (tmp_path / "report").exists()


def test_materialization_receipt_precedes_inventory(tmp_path, monkeypatch):
    import arrayroute.benchmark as benchmark

    def broken_inventory(*args, **kwargs):
        raise OSError("inventory failed")

    monkeypatch.setattr(benchmark.os, "walk", broken_inventory)
    root = tmp_path / "fixture"
    with pytest.raises(OSError, match="inventory failed"):
        create_fixture(root, frames=4, edge=32)
    receipt = json.loads((root / "materialization.json").read_text())
    assert len(receipt["frame_sha256"]) == 4
    assert not (root / "fixture.json").exists()
    assert scan(root, tmp_path / "verified", engine="zarr-python")["complete"]
