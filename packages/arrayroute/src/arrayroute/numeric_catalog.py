# SPDX-License-Identifier: Apache-2.0
"""Immutable scalar indexes for globally ordered, bounded metadata selections."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import CatalogReceipt, _positive, _publish_snapshot

_APPLICATION_ID = 0x41524E31
_SCHEMA = (
    "CREATE TABLE root (id INTEGER PRIMARY KEY CHECK(id=1), metadata TEXT NOT NULL, "
    "records INTEGER NOT NULL, numeric_records INTEGER NOT NULL, revision TEXT NOT NULL)",
    "CREATE TABLE entries (ordinal INTEGER PRIMARY KEY, value REAL, "
    "digest BLOB NOT NULL CHECK(length(digest)=32), "
    "missing INTEGER GENERATED ALWAYS AS (value IS NULL) VIRTUAL, "
    "forward REAL GENERATED ALWAYS AS (coalesce(value,0.0)) VIRTUAL, "
    "backward REAL GENERATED ALWAYS AS (-coalesce(value,0.0)) VIRTUAL)",
    "CREATE INDEX ascending_values ON entries(missing,forward,ordinal)",
    "CREATE INDEX descending_values ON entries(missing,backward,ordinal)",
)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if type(value) not in (float, int) or (type(value) is int and abs(value) > 2**53):
        raise ValueError("values must be finite floats, exact integers up to 2**53, or None")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite values cannot be indexed")
    return 0.0 if number == 0 else number


def _text(value: str, label: str, *, empty: bool = False) -> None:
    if not isinstance(value, str) or (not empty and not value) or len(value.encode()) > 4096:
        raise ValueError(f"{label} must contain {'0' if empty else '1'}–4096 UTF-8 bytes")


def _digest(ordinal: int, value: float | None) -> bytes:
    scalar = "missing" if value is None else value.hex()
    return hashlib.sha256(f"arrayroute-scalar-v1\0{ordinal}\0{scalar}".encode()).digest()


def _revision(metadata: str):
    return hashlib.sha256(b"arrayroute-numeric-v1\0" + metadata.encode() + b"\0")


@dataclass(frozen=True)
class NumericEntry:
    ordinal: int
    value: float | None


@dataclass(frozen=True)
class NumericCursor:
    revision: str
    descending: bool
    missing: int
    sort_value: float
    ordinal: int


@dataclass(frozen=True)
class NumericPage:
    entries: tuple[NumericEntry, ...]
    next_cursor: NumericCursor | None


def write_numeric_index(
    path: str | Path,
    values: Iterable[float | int | None],
    *,
    source_revision: str,
    field: str,
    units: str = "",
    cache_mib: int = 8,
) -> CatalogReceipt:
    """Index one scalar column in source-record order; missing values sort last.

    Publish a separate immutable file per queried scalar column. Array data and source
    records remain external. Zero signs compare equally; equal values retain source order.
    """
    _positive(cache_mib, "cache_mib", 1024)
    _text(source_revision, "source_revision")
    _text(field, "field")
    _text(units, "units", empty=True)
    target = Path(path).absolute()
    if os.path.lexists(target):
        raise FileExistsError(target)
    metadata = json.dumps(
        dict(source_revision=source_revision, field=field, units=units),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    connection = None
    try:
        connection = sqlite3.connect(temporary)
        connection.execute(f"PRAGMA cache_size=-{cache_mib * 1024}")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(f"PRAGMA application_id={_APPLICATION_ID}")
        connection.execute("PRAGMA user_version=1")
        for statement in _SCHEMA[:2]:
            connection.execute(statement)
        revision = _revision(metadata)
        count = numeric = 0
        for ordinal, raw in enumerate(values):
            value = _number(raw)
            digest = _digest(ordinal, value)
            connection.execute(
                "INSERT INTO entries(ordinal,value,digest) VALUES(?,?,?)", (ordinal, value, digest)
            )
            revision.update(digest)
            count += 1
            numeric += value is not None
        for statement in _SCHEMA[2:]:
            connection.execute(statement)
        revision.update(f"{count}\0{numeric}".encode())
        identity = revision.hexdigest()
        connection.execute(
            "INSERT INTO root VALUES(1,?,?,?,?)", (metadata, count, numeric, identity)
        )
        connection.commit()
        connection.close()
        connection = None
        return _publish_snapshot(temporary, target, count, identity)
    finally:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)


class NumericIndex:
    """Pin one read-only index revision; authenticate file receipts outside this library.

    Each thread/process needs its own reader. The source revision is mandatory to avoid
    silently applying ordinals to a different source snapshot. Row hashes detect accidental
    damage; use ``verify()`` or a trusted file hash to check the entire index before use.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        source_revision: str,
        expected_revision: str | None = None,
        cache_mib: int = 8,
    ):
        _positive(cache_mib, "cache_mib", 1024)
        _text(source_revision, "source_revision")
        self._connection = sqlite3.connect(Path(path).absolute().as_uri() + "?mode=ro", uri=True)
        c = self._connection
        try:
            c.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 128 * 1024)
            c.execute("PRAGMA trusted_schema=OFF")
            c.execute("PRAGMA query_only=ON")
            c.execute("PRAGMA mmap_size=0")
            c.execute(f"PRAGMA cache_size=-{cache_mib * 1024}")
            c.execute("BEGIN")
            if (
                c.execute("PRAGMA application_id").fetchone()[0] != _APPLICATION_ID
                or c.execute("PRAGMA user_version").fetchone()[0] != 1
            ):
                raise ValueError("not an ArrayRoute numeric index version 1")
            schema = c.execute("SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL")
            if sorted(row[0] for row in schema) != sorted(_SCHEMA):
                raise ValueError("unexpected numeric index schema")
            rows = c.execute(
                "SELECT metadata,records,numeric_records,revision FROM root"
            ).fetchmany(2)
            if len(rows) != 1:
                raise ValueError("numeric index must have one root")
            self._root, self.count, self.numeric_count, self.revision = rows[0]
            if (
                type(self.count) is not int
                or type(self.numeric_count) is not int
                or not 0 <= self.numeric_count <= self.count
                or not isinstance(self.revision, str)
                or len(self.revision) != 64
                or any(ch not in "0123456789abcdef" for ch in self.revision)
            ):
                raise ValueError("invalid numeric index identity")
            meta = json.loads(self._root)
            if not isinstance(meta, dict) or set(meta) != {"source_revision", "field", "units"}:
                raise ValueError("invalid numeric index metadata")
            for key in meta:
                _text(meta[key], key, empty=key == "units")
            self.source_revision, self.field, self.units = (
                meta["source_revision"],
                meta["field"],
                meta["units"],
            )
            if self.source_revision != source_revision:
                raise ValueError("numeric index belongs to a different source revision")
            if expected_revision is not None and self.revision != expected_revision:
                raise ValueError("numeric index does not match expected revision")
            self._quantiles: dict[int, tuple[float, ...]] = {}
        except BaseException:
            c.close()
            raise

    def _entry(self, row: tuple) -> NumericEntry:
        ordinal, raw, digest = row
        value = _number(raw)
        if (
            type(ordinal) is not int
            or not 0 <= ordinal < self.count
            or _digest(ordinal, value) != digest
        ):
            raise ValueError("numeric record failed integrity check")
        return NumericEntry(ordinal, value)

    def page(
        self, *, descending: bool = False, cursor: NumericCursor | None = None, limit: int = 100
    ) -> NumericPage:
        _positive(limit, "limit", 4096)
        if type(descending) is not bool:
            raise ValueError("descending must be boolean")
        column = "backward" if descending else "forward"
        args: tuple = ()
        clause = ""
        if cursor is not None:
            if (
                not isinstance(cursor, NumericCursor)
                or cursor.revision != self.revision
                or type(cursor.descending) is not bool
                or cursor.descending != descending
                or type(cursor.missing) is not int
                or cursor.missing not in (0, 1)
                or type(cursor.ordinal) is not int
                or not 0 <= cursor.ordinal < self.count
                or type(cursor.sort_value) not in (int, float)
                or (type(cursor.sort_value) is int and abs(cursor.sort_value) > 2**53)
                or not math.isfinite(cursor.sort_value)
                or (cursor.missing == 1 and cursor.sort_value != 0)
            ):
                raise ValueError("cursor does not match this numeric selection")
            clause = f" WHERE (missing,{column},ordinal)>(?,?,?)"
            args = (cursor.missing, cursor.sort_value, cursor.ordinal)
        rows = self._connection.execute(
            f"SELECT ordinal,value,digest FROM entries{clause} "
            f"ORDER BY missing,{column},ordinal LIMIT ?",
            (*args, limit + 1),
        ).fetchall()
        entries = tuple(self._entry(row) for row in rows[:limit])
        next_cursor = None
        if len(rows) > limit:
            last = entries[-1]
            sort_value = 0.0 if last.value is None else (-last.value if descending else last.value)
            next_cursor = NumericCursor(
                self.revision, descending, int(last.value is None), sort_value, last.ordinal
            )
        return NumericPage(entries, next_cursor)

    def quantiles(self, bins: int = 4) -> tuple[float, ...]:
        """Global inclusive upper edges; missing values are excluded, ties share a bin."""
        _positive(bins, "bins", 64)
        if bins not in self._quantiles:
            count = self.numeric_count
            actual = min(bins, count)
            edges = []
            for b in range(1, actual + 1):
                row = self._connection.execute(
                    "SELECT ordinal,value,digest FROM entries WHERE missing=0 "
                    "ORDER BY missing,forward,ordinal LIMIT 1 OFFSET ?",
                    ((b * count + actual - 1) // actual - 1,),
                ).fetchone()
                if row is None or (value := self._entry(row).value) is None:
                    raise ValueError("numeric summary disagrees with its records")
                edges.append(value)
            self._quantiles[bins] = tuple(edges)
        return self._quantiles[bins]

    def verify(self) -> str:
        revision = _revision(self._root)
        count = numeric = 0
        for row in self._connection.execute(
            "SELECT ordinal,value,digest FROM entries ORDER BY ordinal"
        ):
            entry = self._entry(row)
            if entry.ordinal != count:
                raise ValueError("numeric ordinals are not contiguous")
            revision.update(row[2])
            count += 1
            numeric += entry.value is not None
        revision.update(f"{count}\0{numeric}".encode())
        if (
            count != self.count
            or numeric != self.numeric_count
            or revision.hexdigest() != self.revision
        ):
            raise ValueError("numeric index revision failed integrity check")
        return revision.hexdigest()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> NumericIndex:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
