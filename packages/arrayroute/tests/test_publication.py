# SPDX-License-Identifier: Apache-2.0
import errno
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from arrayroute.publication import staged_directory


def test_complete_tree_visible_only_at_commit(tmp_path):
    target = tmp_path / "dataset"
    with staged_directory(target) as stage:
        assert stage.name == target.name
        (stage / "values").write_bytes(bytes(range(256)))
        assert not target.exists()
    assert (target / "values").read_bytes() == bytes(range(256))
    assert list(tmp_path.iterdir()) == [target]


def test_unsupported_filesystem_fails_before_dataset_work(tmp_path, monkeypatch):
    from arrayroute import publication

    def unsupported(*args):
        raise OSError(errno.ENOTSUP, "exclusive rename unavailable")

    monkeypatch.setattr(publication, "_exclusive_rename", unsupported)
    with pytest.raises(OSError, match="unavailable"), staged_directory(tmp_path / "dataset"):
        pytest.fail("expensive dataset work must not start")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["empty", "populated", "file", "symlink"])
def test_never_replace_existing_target(tmp_path, kind):
    target = tmp_path / "dataset"
    if kind in ("empty", "populated"):
        target.mkdir()
        if kind == "populated":
            (target / "keep").write_bytes(b"original")
    elif kind == "file":
        target.write_bytes(b"original")
    else:
        target.symlink_to(tmp_path / "missing")
    before = target.lstat()
    with pytest.raises(FileExistsError), staged_directory(target):
        pytest.fail("existing target admitted")
    assert target.lstat() == before


def test_failure_and_late_cancellation_remove_unpublished_tree(tmp_path):
    target = tmp_path / "dataset"
    with pytest.raises(RuntimeError), staged_directory(target) as stage:
        (stage / "manifest").write_text("complete-looking but not ready")
        raise RuntimeError("quality check failed")
    assert list(tmp_path.iterdir()) == []
    cancelled = False
    with pytest.raises(InterruptedError), staged_directory(target, cancelled=lambda: cancelled):
        cancelled = True
    assert list(tmp_path.iterdir()) == []


def test_competing_publishers_cannot_clobber_each_other(tmp_path):
    target = tmp_path / "dataset"
    barrier = Barrier(2)

    def publish(value):
        try:
            with staged_directory(target) as stage:
                (stage / "values").write_bytes(value)
                barrier.wait(timeout=10)
            return value
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, [b"first", b"second"]))
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    assert (target / "values").read_bytes() == winners[0]
    assert list(tmp_path.iterdir()) == [target]


def test_competing_empty_directory_is_preserved(tmp_path):
    target = tmp_path / "dataset"
    with pytest.raises(FileExistsError), staged_directory(target) as stage:
        (stage / "values").write_bytes(b"new")
        target.mkdir()
        before = target.stat()
    assert target.stat() == before
    assert list(target.iterdir()) == []


def test_abrupt_process_exit_never_publishes(tmp_path):
    target = tmp_path / "dataset"
    code = """
import os, sys
from arrayroute.publication import staged_directory
with staged_directory(sys.argv[1]) as stage:
    (stage / "manifest.json").write_text("partial")
    os._exit(17)
"""
    result = subprocess.run([sys.executable, "-c", code, str(target)], check=False)
    assert result.returncode == 17
    assert not os.path.lexists(target)
    partials = list(tmp_path.glob(".arrayroute-*.partial"))
    assert len(partials) == 1
    assert (partials[0] / "dataset/manifest.json").read_text() == "partial"
    shutil.rmtree(partials[0])
