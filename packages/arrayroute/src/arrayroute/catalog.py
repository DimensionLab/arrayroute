# SPDX-License-Identifier: Apache-2.0
"""Indexed, atomically published local metadata snapshots; array payloads stay external."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_APPLICATION_ID = 0x41524331
_VERSION = 1
_MAX_RECORD = 1024 * 1024
_MAX_KEY = 4096
_SCHEMA = (
    "CREATE TABLE catalog (id INTEGER PRIMARY KEY CHECK(id=1), metadata TEXT NOT NULL, "
    "records INTEGER NOT NULL, revision TEXT NOT NULL)",
    "CREATE TABLE entries (ordinal INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE, "
    "payload TEXT NOT NULL, digest TEXT NOT NULL)",
)


def _json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    if len(encoded.encode("utf-8")) > _MAX_RECORD:
        raise ValueError("catalog JSON record exceeds 1 MiB")
    return encoded


def _digest(ordinal: int, key: str, payload: str) -> str:
    digest = hashlib.sha256(b"arrayroute-catalog-record-v1\0")
    for item in (str(ordinal), key, payload):
        raw = item.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _revision(metadata: str):
    return hashlib.sha256(b"arrayroute-catalog-v1\0" + metadata.encode("utf-8") + b"\0")


def _positive(value: int, name: str, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")


@dataclass(frozen=True)
class CatalogReceipt:
    path: Path
    records: int
    revision: str
    file_bytes: int
    file_sha256: str


@dataclass(frozen=True)
class CatalogEntry:
    ordinal: int
    key: str
    value: Any


@dataclass(frozen=True)
class CatalogPage:
    entries: tuple[CatalogEntry, ...]
    next_ordinal: int | None
    encoded_bytes: int


def _publish_snapshot(temporary: Path, target: Path, count: int, identity: str) -> CatalogReceipt:
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
        file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    size = temporary.stat().st_size
    os.link(temporary, target)
    # Persist the publication as well as the SQLite contents on POSIX filesystems.
    if os.name == "posix":
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return CatalogReceipt(target, count, identity, size, file_hash)


def write_catalog(
    path: str | Path,
    records: Iterable[tuple[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
    cache_mib: int = 8,
) -> CatalogReceipt:
    """Consume records once and publish without replacing any existing destination.

    Keys are unique and insertion order defines pagination. Publication requires a local
    filesystem supporting hard links. A caller wanting cancellation can raise from its
    record iterator; no partial snapshot is published, including on KeyboardInterrupt.
    """
    _positive(cache_mib, "cache_mib", 1024)
    target = Path(path).absolute()
    if os.path.lexists(target):
        raise FileExistsError(target)
    root = _json(dict(metadata or {}))
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    connection = None
    try:
        connection = sqlite3.connect(temporary)
        connection.execute(f"PRAGMA cache_size=-{cache_mib * 1024}")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(f"PRAGMA application_id={_APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version={_VERSION}")
        for statement in _SCHEMA:
            connection.execute(statement)
        revision = _revision(root)
        count = 0
        for ordinal, (key, value) in enumerate(records):
            if not isinstance(key, str) or not key or len(key.encode("utf-8")) > _MAX_KEY:
                raise ValueError("catalog key must contain 1–4096 UTF-8 bytes")
            payload = _json(value)
            digest = _digest(ordinal, key, payload)
            connection.execute(
                "INSERT INTO entries VALUES (?, ?, ?, ?)", (ordinal, key, payload, digest)
            )
            revision.update(bytes.fromhex(digest))
            count += 1
        identity = revision.hexdigest()
        connection.execute("INSERT INTO catalog VALUES (1, ?, ?, ?)", (root, count, identity))
        connection.commit()
        connection.close()
        connection = None
        return _publish_snapshot(temporary, target, count, identity)
    finally:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)


class Catalog:
    """Read one snapshot with bounded SQLite cache and key/range indexes.

    Keep the context short: a read transaction pins its revision and prevents cooperating
    SQLite writers from committing in place. Published snapshots should never be edited.
    Hashes detect accidental corruption, not authenticity; authenticate receipts separately.
    Each thread/process needs its own instance. Decoded Python objects and OS file cache are
    outside ``cache_mib``; ``max_bytes`` bounds encoded payloads in a returned page.
    """

    def __init__(
        self, path: str | Path, *, cache_mib: int = 8, expected_revision: str | None = None
    ) -> None:
        _positive(cache_mib, "cache_mib", 1024)
        self.path = Path(path).absolute()
        self._connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        connection = self._connection
        try:
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, _MAX_RECORD + 16384)
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute(f"PRAGMA cache_size=-{cache_mib * 1024}")
            connection.execute("BEGIN")
            if (
                connection.execute("PRAGMA application_id").fetchone()[0] != _APPLICATION_ID
                or connection.execute("PRAGMA user_version").fetchone()[0] != _VERSION
            ):
                raise ValueError("not an ArrayRoute catalog version 1")
            schema = connection.execute("SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL")
            if sorted(row[0] for row in schema) != sorted(_SCHEMA):
                raise ValueError("unexpected catalog schema")
            rows = connection.execute("SELECT metadata, records, revision FROM catalog").fetchmany(
                2
            )
            if len(rows) != 1:
                raise ValueError("catalog must have one root record")
            self._root, self.count, self.revision = rows[0]
            if (
                type(self.count) is not int
                or self.count < 0
                or not isinstance(self.revision, str)
                or len(self.revision) != 64
                or any(c not in "0123456789abcdef" for c in self.revision)
            ):
                raise ValueError("invalid catalog identity")
            self.metadata = json.loads(self._root)
            if not isinstance(self.metadata, dict) or _json(self.metadata) != self._root:
                raise ValueError("invalid catalog metadata")
            if expected_revision is not None and self.revision != expected_revision:
                raise ValueError("catalog revision does not match expected revision")
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _entry(row: tuple) -> CatalogEntry:
        ordinal, key, payload, digest = row
        if (
            type(ordinal) is not int
            or ordinal < 0
            or not isinstance(key, str)
            or not key
            or len(key.encode("utf-8")) > _MAX_KEY
            or not isinstance(payload, str)
            or len(payload.encode("utf-8")) > _MAX_RECORD
            or _digest(ordinal, key, payload) != digest
        ):
            raise ValueError("catalog record failed integrity check")
        value = json.loads(payload)
        if _json(value) != payload:
            raise ValueError("invalid catalog JSON")
        return CatalogEntry(ordinal, key, value)

    def get(self, key: str) -> CatalogEntry:
        row = self._connection.execute(
            "SELECT ordinal, key, payload, digest FROM entries WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return self._entry(row)

    def page(
        self, *, start: int = 0, limit: int = 100, max_bytes: int = 8 * 1024 * 1024
    ) -> CatalogPage:
        if type(start) is not int or start < 0:
            raise ValueError("start must be a nonnegative integer")
        _positive(limit, "limit", 4096)
        _positive(max_bytes, "max_bytes", 64 * 1024 * 1024)
        result = []
        size = 0
        cursor = self._connection.execute(
            "SELECT ordinal, key, payload, digest FROM entries "
            "WHERE ordinal>=? ORDER BY ordinal LIMIT ?",
            (min(start, self.count), limit),
        )
        try:
            for row in cursor:
                if not isinstance(row[1], str) or not isinstance(row[2], str):
                    raise ValueError("catalog record must contain text keys and JSON payloads")
                encoded = len(row[1].encode("utf-8")) + len(row[2].encode("utf-8"))
                if size + encoded > max_bytes:
                    if not result:
                        raise ValueError("max_bytes is too small for the next record")
                    break
                result.append(self._entry(row))
                size += encoded
        finally:
            cursor.close()
        next_ordinal = result[-1].ordinal + 1 if result else self.count
        return CatalogPage(tuple(result), next_ordinal if next_ordinal < self.count else None, size)

    def verify(self) -> str:
        """Explicit full scan; compare the returned digest to a separately trusted receipt."""
        revision = _revision(self._root)
        count = 0
        for row in self._connection.execute(
            "SELECT ordinal, key, payload, digest FROM entries ORDER BY ordinal"
        ):
            entry = self._entry(row)
            if entry.ordinal != count:
                raise ValueError("catalog ordinals are not contiguous")
            revision.update(bytes.fromhex(row[3]))
            count += 1
        if count != self.count or revision.hexdigest() != self.revision:
            raise ValueError("catalog revision failed integrity check")
        return revision.hexdigest()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
