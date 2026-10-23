# SPDX-License-Identifier: Apache-2.0
"""Bounded, checksummed canonical-moment state; no executable serialization."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import struct
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .statistics import CanonicalChannelMoments

_MAGIC = b"ARCM\x00\x01\r\n"
_HEADER_LIMIT = 8192


def _runtime() -> dict[str, str]:
    # These are early mismatch checks. The caller's contract must pin the full
    # numerical environment, including builds and CPU/kernel dispatch settings.
    return {
        "numpy": np.__version__, "python": platform.python_version(),
        "implementation": platform.python_implementation(), "machine": platform.machine(),
        "system": platform.system(), "byteorder": sys.byteorder,
    }


def _contract(digest: str) -> None:
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("contract_digest must be a lowercase SHA-256 digest")


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _validate(state: CanonicalChannelMoments) -> None:
    n, pending, count = state._rows_seen, state._pending, state._moments.count
    if not 0 <= n <= np.iinfo(np.int64).max or pending != n % state.LEAF_ROWS:
        raise ValueError("Invalid checkpoint row position")
    if (count < 0).any() or (count > n - pending).any():
        raise ValueError("Invalid checkpoint observation counts")
    if not np.array_equal(state._origin_set, count > 0):
        raise ValueError("Invalid checkpoint origins")
    for values in (state._moments.mean, state._moments.m2, state._origin):
        if not np.isfinite(values).all() or (values[count == 0] != 0).any():
            raise ValueError("Invalid checkpoint moments")
    if (state._moments.m2 < 0).any():
        raise ValueError("Invalid checkpoint variance")
    if not np.isfinite(state._values[:pending][state._valid[:pending]]).all():
        raise ValueError("Nonfinite checkpoint observations inside validity mask")


def encode(state: CanonicalChannelMoments, contract_digest: str) -> bytes:
    _contract(contract_digest)
    _validate(state)
    metadata = _json({
        "algorithm": state.ALGORITHM, "channels": len(state._origin),
        "rows_seen": state._rows_seen, "contract_digest": contract_digest, "runtime": _runtime(),
    })
    arrays = (
        (state._moments.count, "<i8"), (state._moments.mean, "<f8"),
        (state._moments.m2, "<f8"), (state._origin, "<f8"),
        (state._values[:state._pending], "<f8"), (state._valid[:state._pending], "u1"),
    )
    payload = b"".join(np.asarray(a, dtype=dtype, order="C").tobytes() for a, dtype in arrays)
    body = _MAGIC + struct.pack("<I", len(metadata)) + metadata + payload
    return body + hashlib.sha256(body).digest()


def decode(
    cls: type[CanonicalChannelMoments], data: bytes, contract_digest: str, max_state_bytes: int,
) -> CanonicalChannelMoments:
    _contract(contract_digest)
    if type(max_state_bytes) is not int or max_state_bytes < 1:
        raise ValueError("max_state_bytes must be a positive integer")
    if not isinstance(data, bytes) or not 44 <= len(data) <= max_state_bytes:
        raise ValueError("Checkpoint size exceeds limit or is truncated")
    if data[:8] != _MAGIC:
        raise ValueError("Unsupported checkpoint format")
    if hashlib.sha256(memoryview(data)[:-32]).digest() != data[-32:]:
        raise ValueError("Checkpoint checksum mismatch")
    size = struct.unpack_from("<I", data, 8)[0]
    if not 1 <= size <= _HEADER_LIMIT or 12 + size > len(data) - 32:
        raise ValueError("Invalid checkpoint metadata length")
    header = data[12:12 + size]
    try:
        meta = json.loads(header)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid checkpoint metadata") from exc
    if not isinstance(meta, dict) or set(meta) != {
        "algorithm", "channels", "rows_seen", "contract_digest", "runtime",
    }:
        raise ValueError("Invalid checkpoint metadata fields")
    if _json(meta) != header:
        raise ValueError("Checkpoint metadata must be canonical JSON")
    if meta["algorithm"] != cls.ALGORITHM:
        raise ValueError("Checkpoint numerical algorithm mismatch")
    if meta["contract_digest"] != contract_digest:
        raise ValueError("Checkpoint pipeline contract mismatch")
    if meta["runtime"] != _runtime():
        raise ValueError("Checkpoint numerical runtime mismatch")
    channels, rows = meta["channels"], meta["rows_seen"]
    if type(channels) is not int or channels < 1 or type(rows) is not int or rows < 0:
        raise ValueError("Invalid checkpoint dimensions")
    # Validate both serialized bytes and the full pending-leaf allocation before
    # constructing an accumulator, even when the serialized pending leaf is empty.
    if channels * (cls.LEAF_ROWS * 9 + 33) > max_state_bytes:
        raise ValueError("Checkpoint state allocation exceeds limit")
    pending = rows % cls.LEAF_ROWS
    if len(data) != 12 + size + channels * (32 + pending * 9) + 32:
        raise ValueError("Checkpoint payload length mismatch")
    state = cls(channels)
    position = 12 + size
    for target, dtype in (
        (state._moments.count, "<i8"), (state._moments.mean, "<f8"),
        (state._moments.m2, "<f8"), (state._origin, "<f8"),
        (state._values[:pending], "<f8"), (state._valid[:pending], "u1"),
    ):
        source = np.frombuffer(data, dtype=dtype, count=target.size, offset=position)
        if dtype == "u1" and (source > 1).any():
            raise ValueError("Invalid checkpoint boolean mask")
        target[:] = source.reshape(target.shape)
        position += source.nbytes
    state._rows_seen, state._pending = rows, pending
    state._origin_set[:] = state._moments.count > 0
    _validate(state)
    return state
