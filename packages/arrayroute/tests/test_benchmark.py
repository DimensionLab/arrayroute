# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arrayroute.benchmark import _digest, create_fixture


@pytest.mark.parametrize("engine", ["zarr-python", "zarrista"])
def test_fixture_native_write_and_receipt_are_independently_readable(tmp_path, engine):
    if engine == "zarrista":
        pytest.importorskip(engine)
    import zarr

    root = tmp_path / "fixture"
    receipt = create_fixture(root, frames=4, edge=32, writer=engine)
    assert receipt["logical_bytes_per_layout"] == 4 * 32**3 * 3 * 4
    for name in receipt["layouts"]:
        a = zarr.open_array(root / f"{name}.zarr", mode="r")
        for t in range(4):
            assert _digest(np.asarray(a[t])) == receipt["frame_sha256"][t]
    with pytest.raises(FileExistsError):
        create_fixture(root, frames=4, edge=32)
