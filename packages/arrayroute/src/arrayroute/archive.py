# SPDX-License-Identifier: Apache-2.0
"""Deterministic, bounded packing of immutable local data trees."""

from __future__ import annotations

import hashlib
import operator
import os
import stat
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArchiveReceipt:
    algorithm: str
    sha256: str
    archive_bytes: int
    payload_bytes: int
    members: int


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _opened_matches_path(before: os.stat_result, opened: os.stat_result) -> bool:
    # CPython on Windows reports birth time via stat but change time via fstat.
    # Compare ctime only within the same API; retain both before/after checks.
    left, right = _identity(before), _identity(opened)
    return left[:-1] == right[:-1] if os.name == "nt" else left == right


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or operator.index(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return operator.index(value)


def pack_directory(
    source: str | Path,
    destination: str | Path,
    *,
    max_output_bytes: int,
    archive_name: str | None = None,
    buffer_bytes: int = 1 << 20,
    max_pending_names: int = 65536,
    max_members: int = 10_000_000,
    max_depth: int = 64,
    max_path_bytes: int = 4096,
    replace: bool = False,
    cancelled: Callable[[], bool] | None = None,
) -> ArchiveReceipt:
    """Pack directories/regular files as a canonical uncompressed POSIX pax tar.

    Fixed owner/time/mode metadata and lexical depth-first ordering make identical
    path/payload trees produce identical bytes. Source permissions, hardlink identity,
    xattrs and sparse extents are not preserved; symlinks/special files are rejected.
    Copy and buffered output memory together fit buffer_bytes. Pending traversal names,
    path lengths, depth, member count and complete archive bytes have separate bounds.
    Python objects, tar headers and OS caches are additional to the copy buffer.

    Requires an immutable trusted source tree and stable destination parent. Observed
    file/directory changes fail, but this is not a filesystem snapshot or protection
    against hostile concurrent directory replacement. Cancellation is checked per
    entry/copy block. Handled failures leave the prior destination untouched and remove
    staging; a killed process may leave a hidden partial file. Publication is atomic,
    using same-directory hard links without replacement, or os.replace when requested.
    The file is fsynced; power-loss durability of directory entries is not guaranteed.
    """
    limit = _positive(max_output_bytes, "max_output_bytes")
    buffer_bytes = _positive(buffer_bytes, "buffer_bytes")
    max_pending_names = _positive(max_pending_names, "max_pending_names")
    max_members = _positive(max_members, "max_members")
    max_depth = _positive(max_depth, "max_depth")
    max_path_bytes = _positive(max_path_bytes, "max_path_bytes")
    source, target = Path(source).absolute(), Path(destination).absolute()
    name = source.name if archive_name is None else archive_name
    if not name or name in {".", ".."} or any(c in name for c in "/\\\x00"):
        raise ValueError("archive_name must be a single safe path component")
    if not stat.S_ISDIR(source.lstat().st_mode):
        raise ValueError("Archive source must be a directory, not a symlink")
    if target.resolve().is_relative_to(source.resolve()):
        raise ValueError("Archive destination must be outside the source tree")
    if not replace and os.path.lexists(target):
        raise FileExistsError(target)

    def check() -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError("Archive packing cancelled")

    check()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".arrayroute-", suffix=".partial", dir=target.parent)
    digest = hashlib.sha256()
    written = payload = members = pending_names = 0
    output_buffer = min(64 << 10, buffer_bytes // 4)
    if output_buffer < 2:
        output_buffer = 0
    try:
        with os.fdopen(fd, "wb", buffering=output_buffer) as output:
            buffer = bytearray(min(buffer_bytes - output_buffer, limit))

            def emit(data: bytes | memoryview) -> None:
                nonlocal written
                check()
                if written + len(data) > limit:
                    raise MemoryError("Archive exceeds max_output_bytes")
                view = memoryview(data)
                digest.update(view)
                while view:
                    count = output.write(view)
                    if not count:
                        raise OSError("Archive write made no progress")
                    written += count
                    view = view[count:]

            def visit(path: Path, arcname: str, depth: int) -> None:
                nonlocal payload, members, pending_names
                check()
                if depth > max_depth or len(arcname.encode("utf-8")) > max_path_bytes:
                    raise MemoryError("Archive path exceeds depth/path allowance")
                before = path.lstat()
                directory = stat.S_ISDIR(before.st_mode)
                if not directory and not stat.S_ISREG(before.st_mode):
                    raise ValueError(f"Archive refuses symlinks/special files: {path}")
                members += 1
                if members > max_members:
                    raise MemoryError("Archive exceeds max_members")
                info = tarfile.TarInfo(arcname)
                info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
                info.mode = 0o755 if directory else 0o644
                info.size = 0 if directory else before.st_size
                header = info.tobuf(format=tarfile.PAX_FORMAT, encoding="utf-8", errors="strict")
                padded_size = ((info.size + 511) // 512) * 512
                if written + len(header) + padded_size > limit:
                    raise MemoryError("Archive member exceeds max_output_bytes")
                emit(header)
                if directory:
                    names = []
                    with os.scandir(path) as entries:
                        for entry in entries:
                            check()
                            if pending_names >= max_pending_names:
                                raise MemoryError("Archive exceeds max_pending_names")
                            if len(entry.name.encode("utf-8")) > max_path_bytes:
                                raise MemoryError("Archive name exceeds path allowance")
                            names.append(entry.name)
                            pending_names += 1
                    names.sort(reverse=True)
                    while names:
                        child = names.pop()
                        pending_names -= 1
                        if any(c in child for c in "\\\x00"):
                            raise ValueError("Archive names must be portable POSIX paths")
                        visit(path / child, arcname + "/" + child, depth + 1)
                else:
                    flags = (
                        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                        | getattr(os, "O_BINARY", 0)
                    )
                    with os.fdopen(os.open(path, flags), "rb", buffering=0) as stream:
                        opened = os.fstat(stream.fileno())
                        if not _opened_matches_path(before, opened):
                            raise OSError("Archive source changed before reading")
                        remaining = before.st_size
                        while remaining:
                            check()
                            view = memoryview(buffer)[:min(len(buffer), remaining)]
                            count = stream.readinto(view)
                            if not count:
                                raise OSError("Archive source was truncated")
                            emit(view[:count])
                            remaining -= count
                        if _identity(os.fstat(stream.fileno())) != _identity(opened):
                            raise OSError("Archive source changed while reading")
                    payload += before.st_size
                    emit(b"\0" * (padded_size - before.st_size))
                if _identity(path.lstat()) != _identity(before):
                    raise OSError("Archive source changed during traversal")

            visit(source, name, 0)
            emit(b"\0" * 1024)
            emit(b"\0" * ((-written) % 10240))
            output.flush()
            os.fsync(output.fileno())
        check()
        if replace:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
        return ArchiveReceipt(
            "canonical-pax-tree-v1", digest.hexdigest(), written, payload, members,
        )
    finally:
        Path(temporary).unlink(missing_ok=True)
