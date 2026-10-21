# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from arrayroute.benchmark import create_fixture
from arrayroute.benchmark_parallel import scan_parallel


@pytest.mark.parametrize("engine", ["zarr-python", "tensorstore", "zarrista"])
def test_spawned_scan_checks_every_frame_once_with_uneven_partitions(tmp_path, engine):
    pytest.importorskip({"zarr-python": "zarr"}.get(engine, engine))
    root, output = tmp_path / "fixture", tmp_path / "scan"
    fixture = create_fixture(root, frames=5, edge=32)
    report = scan_parallel(root, output, engine=engine, workers=2, buffer_bytes=2 << 20)
    assert report["frames"] == 5
    assert report["verified_logical_bytes"] == fixture["logical_bytes_per_layout"]
    assert [worker["frames"] for worker in report["workers"]] == [3, 2]
    rows = [
        json.loads(line)
        for path in output.glob("rank-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert sorted(row["position"] for row in rows) == list(range(5))
    assert sorted(row["frame"] for row in rows) == list(range(5))
    for row in rows:
        assert row["sha256"] == fixture["frame_sha256"][row["frame"]]
    assert json.loads((output / "report.json").read_text()) == report
    with pytest.raises(FileExistsError):
        scan_parallel(root, output, engine=engine)


def test_corrupt_frame_fails_without_publishing_a_success_report(tmp_path):
    import zarr

    root, output = tmp_path / "fixture", tmp_path / "scan"
    create_fixture(root, frames=4, edge=32)
    array = zarr.open_array(root / "sharded.zarr", mode="r+")
    array[:] = 42
    with pytest.raises(ValueError, match="Corrupt frame"):
        scan_parallel(root, output, engine="zarr-python", workers=2)
    assert (output / "CANCELLED").is_file()
    assert not (output / "report.json").exists()


@pytest.mark.parametrize(
    "settings", [{"workers": 0}, {"workers": 1.5}, {"threads_per_worker": 0}, {"buffer_bytes": -1}]
)
def test_invalid_limits_do_not_create_output(tmp_path, settings):
    with pytest.raises(ValueError, match="positive integers"):
        scan_parallel(tmp_path / "missing", tmp_path / "output", **settings)
    assert not (tmp_path / "output").exists()
