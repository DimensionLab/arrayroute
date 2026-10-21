# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import sqlite3

import pytest

from arrayroute.catalog import Catalog, write_catalog


def test_round_trip_page_and_lookup(tmp_path):
    target = tmp_path / "catalog #1.sqlite"
    values = [
        (f"case-{i}", {"pressure": {"units": "Pa", "location": "cell"}, "time_s": i / 10})
        for i in range(111)
    ]
    receipt = write_catalog(target, iter(values), metadata={"schema": "test", "unicode": "Košice"})
    assert receipt.file_sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    with Catalog(target, expected_revision=receipt.revision) as catalog:
        assert catalog.count == 111
        assert catalog.metadata["unicode"] == "Košice"
        assert catalog.verify() == receipt.revision
        first = catalog.page(limit=3)
        assert [e.key for e in first.entries] == ["case-0", "case-1", "case-2"]
        assert first.next_ordinal == 3
        assert catalog.page(start=110).next_ordinal is None
        assert catalog.page(start=200).entries == ()
        assert catalog.get("case-102").value == values[102][1]
        with pytest.raises(KeyError):
            catalog.get("unknown")
    with pytest.raises(sqlite3.ProgrammingError):
        catalog.get("case-0")


def test_byte_budget_and_empty(tmp_path):
    target = tmp_path / "catalog"
    write_catalog(target, [("a", "x" * 100), ("b", "y" * 100)])
    with Catalog(target) as catalog:
        page = catalog.page(max_bytes=110)
        assert len(page.entries) == 1
        assert page.next_ordinal == 1
        assert page.encoded_bytes == 103
        with pytest.raises(ValueError, match="too small"):
            catalog.page(max_bytes=100)
        for options in ({"start": -1}, {"limit": True}, {"limit": 0}, {"max_bytes": 0}):
            with pytest.raises(ValueError):
                catalog.page(**options)
    empty = tmp_path / "empty"
    receipt = write_catalog(empty, [])
    with Catalog(empty) as catalog:
        assert catalog.page().next_ordinal is None
        assert catalog.verify() == receipt.revision


@pytest.mark.parametrize(
    "records",
    [
        [("same", 1), ("same", 2)],
        [("", 1)],
        [("x" * 4097, 1)],
        [("x", float("nan"))],
        [("x", "y" * 1048576)],
        [(None, 1)],
    ],
)
def test_invalid_input_is_never_published(tmp_path, records):
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        write_catalog(tmp_path / "catalog", records)
    assert list(tmp_path.iterdir()) == []


def test_cancel_and_existing_destination(tmp_path):
    def interrupted():
        yield "a", {"ok": True}
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        write_catalog(tmp_path / "catalog", interrupted())
    assert list(tmp_path.iterdir()) == []
    target = tmp_path / "catalog"
    target.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        write_catalog(target, [])
    assert target.is_symlink()


def test_publish_race_does_not_replace_target(tmp_path):
    target = tmp_path / "catalog"

    def records():
        yield "a", 1
        target.write_bytes(b"other writer")

    with pytest.raises(FileExistsError):
        write_catalog(target, records())
    assert target.read_bytes() == b"other writer"
    assert list(tmp_path.iterdir()) == [target]


def test_corruption_schema_and_expected_identity(tmp_path):
    target = tmp_path / "catalog"
    write_catalog(target, [("a", {"value": 1})])
    with pytest.raises(ValueError, match="expected revision"):
        Catalog(target, expected_revision="0" * 64)
    with sqlite3.connect(target) as connection:
        connection.execute("UPDATE entries SET payload='{}'")
    with Catalog(target) as catalog:
        with pytest.raises(ValueError, match="integrity"):
            catalog.get("a")
        with pytest.raises(ValueError, match="integrity"):
            catalog.verify()
    with sqlite3.connect(target) as connection:
        connection.execute("CREATE VIEW surprise AS SELECT 1")
    with pytest.raises(ValueError, match="schema"):
        Catalog(target)


def test_missing_rows_detected_by_explicit_verification(tmp_path):
    target = tmp_path / "catalog"
    write_catalog(target, [("a", 1), ("b", 2)])
    with sqlite3.connect(target) as connection:
        connection.execute("DELETE FROM entries WHERE ordinal=0")
    with Catalog(target) as catalog:
        with pytest.raises(ValueError, match="contiguous"):
            catalog.verify()


def test_snapshot_read_transaction_blocks_in_place_writer(tmp_path):
    target = tmp_path / "catalog"
    write_catalog(target, [("a", 1)])
    with Catalog(target) as catalog:
        writer = sqlite3.connect(target, timeout=0.01)
        try:
            writer.execute("UPDATE entries SET payload='2'")
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                writer.commit()
            assert catalog.get("a").value == 1
        finally:
            writer.rollback()
            writer.close()


def test_revision_is_deterministic_and_changes_with_metadata_or_order(tmp_path):
    a = write_catalog(tmp_path / "a", [("a", 1), ("b", 2)], metadata={"u": "Pa"})
    b = write_catalog(tmp_path / "b", [("a", 1), ("b", 2)], metadata={"u": "Pa"})
    c = write_catalog(tmp_path / "c", [("b", 2), ("a", 1)], metadata={"u": "Pa"})
    d = write_catalog(tmp_path / "d", [("a", 1), ("b", 2)], metadata={"u": "bar"})
    assert a.revision == b.revision
    assert len({a.revision, c.revision, d.revision}) == 3


def test_benchmark_compares_identical_metadata_in_fresh_processes(tmp_path):
    import json

    from arrayroute.benchmark_catalog import run

    root = tmp_path / "benchmark"
    run(root, count=23, repeats=1)
    report = json.loads((root / "report.json").read_text())
    assert {r["engine"] for r in report["trials"]} == {"json", "sqlite"}
    assert all(r["result_sha256"] == report["result_sha256"] for r in report["trials"])
    assert report["records"] == 23
