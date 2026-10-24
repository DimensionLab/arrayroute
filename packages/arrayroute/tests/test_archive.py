# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
import shutil
import tarfile
import threading
from contextlib import contextmanager

import pytest

from arrayroute.archive import pack_directory


def tree(root):
    root.mkdir()
    (root / "empty").mkdir()
    (root / "b").mkdir()
    (root / "b" / "values").write_bytes(bytes(range(256)) * 19)
    (root / "zero").touch()
    (root / ("ž" * 60)).write_bytes(b"unicode payload")
    return root


def test_canonical_bytes_across_paths_metadata_and_buffers(tmp_path):
    source = tree(tmp_path / "source")
    clone = tmp_path / "clone"
    shutil.copytree(source, clone)
    for p in [clone, *clone.rglob("*")]:
        os.utime(p, (123456, 123456))
        p.chmod(0o700 if p.is_dir() else 0o600)
    outputs = []
    for i, (root, buffer) in enumerate([(source, 1), (clone, 1537), (source, 1 << 20)]):
        out = tmp_path / f"{i}.tar"
        receipt = pack_directory(
            root, out, archive_name="sample.zarr", buffer_bytes=buffer, max_output_bytes=1 << 20,
        )
        data = out.read_bytes()
        assert receipt.sha256 == hashlib.sha256(data).hexdigest()
        assert receipt.archive_bytes == len(data) and len(data) % 10240 == 0
        assert receipt.payload_bytes == 256 * 19 + len(b"unicode payload")
        assert receipt.members == 6
        outputs.append(data)
    assert outputs[0] == outputs[1] == outputs[2]
    with tarfile.open(tmp_path / "0.tar") as archive:
        assert archive.getnames() == [
            "sample.zarr", "sample.zarr/b", "sample.zarr/b/values", "sample.zarr/empty",
            "sample.zarr/zero", "sample.zarr/" + "ž" * 60,
        ]
        for member in archive:
            assert member.mtime == member.uid == member.gid == 0
            assert member.uname == member.gname == ""
            assert member.mode == (0o755 if member.isdir() else 0o644)
        archive.extractall(tmp_path / "unpacked", filter="data")
    for p in source.rglob("*"):
        if p.is_file():
            restored = tmp_path / "unpacked/sample.zarr" / p.relative_to(source)
            assert p.read_bytes() == restored.read_bytes()


def test_matches_standard_library_pax_encoding(tmp_path):
    source = tree(tmp_path / "source")
    actual, expected = tmp_path / "actual.tar", tmp_path / "expected.tar"
    pack_directory(source, actual, archive_name="s", max_output_bytes=1 << 20)
    with tarfile.open(expected, "w", format=tarfile.PAX_FORMAT) as archive:
        def add(path, name):
            info = tarfile.TarInfo(name)
            info.mode = 0o755 if path.is_dir() else 0o644
            info.type = tarfile.DIRTYPE if path.is_dir() else tarfile.REGTYPE
            if path.is_dir():
                archive.addfile(info)
                for child in sorted(path.iterdir()):
                    add(child, name + "/" + child.name)
            else:
                info.size = path.stat().st_size
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
        add(source, "s")
    assert actual.read_bytes() == expected.read_bytes()


def test_large_payload_preserves_exact_stream(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    values = bytes(range(256)) * 8193
    (source / "values").write_bytes(values)
    receipts = []
    for index, buffer in enumerate([257, 1 << 20]):
        path = tmp_path / f"{index}.tar"
        receipts.append(pack_directory(source, path, buffer_bytes=buffer, max_output_bytes=4 << 20))
        assert receipts[-1].sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipts[0] == receipts[1]
    assert (tmp_path / "0.tar").read_bytes() == (tmp_path / "1.tar").read_bytes()


def test_oversized_member_is_rejected_before_payload_open(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    with (source / "large").open("wb") as stream:
        stream.truncate(1 << 30)
    fdopen = os.fdopen

    def reject_payload(fd, mode, **kwargs):
        assert mode != "rb", "oversized payload must not be opened"
        return fdopen(fd, mode, **kwargs)

    monkeypatch.setattr(os, "fdopen", reject_payload)
    with pytest.raises(MemoryError, match="member"):
        pack_directory(source, tmp_path / "out.tar", max_output_bytes=1 << 20)
    assert not (tmp_path / "out.tar").exists()


@pytest.mark.parametrize("option,value", [
    ("max_pending_names", 1), ("max_members", 1), ("max_path_bytes", 4),
    ("max_output_bytes", 1024),
])
def test_resource_failure_preserves_destination_and_removes_staging(tmp_path, option, value):
    source = tree(tmp_path / "source")
    out = tmp_path / "out.tar"
    out.write_bytes(b"old archive")
    options = {"max_output_bytes": 1 << 20, option: value}
    with pytest.raises(MemoryError):
        pack_directory(source, out, replace=True, **options)
    assert out.read_bytes() == b"old archive"
    assert not list(tmp_path.glob(".arrayroute-*.partial"))


def test_depth_and_pending_names_cover_all_traversal_levels(tmp_path):
    source = tmp_path / "source"
    (source / "a" / "deeper").mkdir(parents=True)
    (source / "z").touch()
    (source / "a" / "one").touch()
    with pytest.raises(MemoryError, match="depth"):
        pack_directory(source, tmp_path / "depth.tar", max_output_bytes=1 << 20, max_depth=1)
    # Root retains z while a adds deeper and one: three pending names, not two.
    with pytest.raises(MemoryError, match="pending_names"):
        pack_directory(source, tmp_path / "names.tar", max_output_bytes=1 << 20,
                       max_pending_names=2)
    assert not list(tmp_path.glob("*.tar"))


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "a/b", "a\\b", "a\x00b"])
def test_invalid_archive_names(tmp_path, name):
    source = tree(tmp_path / "source")
    with pytest.raises(ValueError, match="component"):
        pack_directory(source, tmp_path / "out.tar", archive_name=name, max_output_bytes=1 << 20)


def test_symlinks_special_files_and_nested_destination_are_rejected(tmp_path):
    source = tree(tmp_path / "source")
    with pytest.raises(ValueError, match="outside"):
        pack_directory(source, source / "out.tar", max_output_bytes=1 << 20)
    (source / "link").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="symlinks"):
        pack_directory(source, tmp_path / "out.tar", max_output_bytes=1 << 20)
    (source / "link").unlink()
    if hasattr(os, "mkfifo"):
        os.mkfifo(source / "pipe")
        with pytest.raises(ValueError, match="special"):
            pack_directory(source, tmp_path / "out.tar", max_output_bytes=1 << 20)
    assert not (tmp_path / "out.tar").exists()


def test_no_clobber_even_if_target_is_created_during_publication(tmp_path, monkeypatch):
    source = tree(tmp_path / "source")
    out = tmp_path / "out.tar"
    link = os.link

    def race(src, dest):
        out.write_bytes(b"competing output")
        link(src, dest)

    monkeypatch.setattr(os, "link", race)
    with pytest.raises(FileExistsError):
        pack_directory(source, out, max_output_bytes=1 << 20)
    assert out.read_bytes() == b"competing output"
    with pytest.raises(FileExistsError):
        pack_directory(source, out, max_output_bytes=1 << 20)
    assert not list(tmp_path.glob(".arrayroute-*.partial"))


@pytest.mark.parametrize("action", ["cancel", "append", "truncate"])
def test_bounded_reads_cancel_and_changed_source(tmp_path, monkeypatch, action):
    source = tmp_path / "source"
    source.mkdir()
    data = source / "values"
    data.write_bytes(b"x" * 16384)
    out = tmp_path / "out.tar"
    out.write_bytes(b"previous")
    calls = []
    cancelled = threading.Event()
    fdopen = os.fdopen

    @contextmanager
    def observe(fd, mode, **kwargs):
        with fdopen(fd, mode, **kwargs) as stream:
            class Reader:
                def fileno(self):
                    return stream.fileno()

                def readinto(self, buffer):
                    calls.append(len(buffer))
                    result = stream.readinto(buffer)
                    if len(calls) == 1:
                        if action == "cancel":
                            cancelled.set()
                        elif action == "append":
                            with data.open("ab") as changed:
                                changed.write(b"new")
                        else:
                            data.write_bytes(b"short")
                    return result

            yield Reader() if mode == "rb" else stream

    monkeypatch.setattr(os, "fdopen", observe)
    with pytest.raises(InterruptedError if action == "cancel" else OSError):
        pack_directory(source, out, max_output_bytes=1 << 20, buffer_bytes=257,
                       replace=True, cancelled=cancelled.is_set)
    assert calls and max(calls) <= 257
    assert out.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".arrayroute-*.partial"))


def test_cancellation_after_fsync_still_preserves_previous_output(tmp_path, monkeypatch):
    source = tree(tmp_path / "source")
    out = tmp_path / "out.tar"
    out.write_bytes(b"previous")
    cancelled = threading.Event()
    fsync = os.fsync

    def stop(fd):
        fsync(fd)
        cancelled.set()

    monkeypatch.setattr(os, "fsync", stop)
    with pytest.raises(InterruptedError):
        pack_directory(source, out, max_output_bytes=1 << 20, replace=True,
                       cancelled=cancelled.is_set)
    assert out.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".arrayroute-*.partial"))
