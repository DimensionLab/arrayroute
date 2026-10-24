# SPDX-License-Identifier: Apache-2.0
"""Publish complete local directory trees without replacing an existing destination."""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


def _exclusive_rename(source: Path, target: Path) -> None:
    if sys.platform == "win32":
        # Windows rename rejects an existing destination, including an empty directory.
        os.rename(source, target)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args = (os.fsencode(source), os.fsencode(target), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        args = (-100, os.fsencode(source), -100, os.fsencode(target), 1)  # AT_FDCWD, NOREPLACE
    else:
        raise OSError(errno.ENOTSUP, "Exclusive directory publication is unavailable", str(target))
    rename.restype = ctypes.c_int
    if rename(*args) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


@contextmanager
def staged_directory(
    target: str | Path, *, cancelled: Callable[[], bool] | None = None
) -> Iterator[Path]:
    """Build privately, then atomically publish on normal context exit.

    Requires a local filesystem supporting exclusive rename. Existing destinations are
    never replaced, even if created concurrently. The caller must close all writers before
    exit. Cancellation is checked before allocation and immediately before publication;
    callbacks are not polled during caller work. Handled failures remove staging. A killed
    process may leave its hidden `.arrayroute-*.partial` directory for explicit cleanup.
    This provides atomic visibility, not power-loss durability, recovery or source snapshots.
    The parent directory must be trusted and remain unchanged during the operation.
    """
    destination = Path(target).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    if cancelled is not None and cancelled():
        raise InterruptedError("Directory publication cancelled")
    destination.parent.mkdir(parents=True, exist_ok=True)
    container = Path(
        tempfile.mkdtemp(prefix=".arrayroute-", suffix=".partial", dir=destination.parent)
    )
    stage = container / destination.name
    try:
        stage.mkdir()
        yield stage
        if cancelled is not None and cancelled():
            raise InterruptedError("Directory publication cancelled")
        if stage.is_symlink() or not stage.is_dir():
            raise ValueError("Publication staging must remain a directory")
        _exclusive_rename(stage, destination)
    finally:
        shutil.rmtree(container)
