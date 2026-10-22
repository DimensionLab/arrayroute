# SPDX-License-Identifier: Apache-2.0
import dataclasses
import hashlib
import random
import sqlite3

import pytest

from arrayroute.numeric_catalog import NumericCursor, NumericIndex, write_numeric_index


def read_all(index, descending=False, limit=3):
    found, cursor = [], None
    while True:
        page = index.page(descending=descending, cursor=cursor, limit=limit)
        found.extend((e.ordinal, e.value) for e in page.entries)
        cursor = page.next_cursor
        if cursor is None:
            return found


def test_ordered_pages_missing_values_ties_and_global_quantiles(tmp_path):
    values = [None, 4, -1, 4, 0, -0.0, None, 2, 9, -2]
    receipt = write_numeric_index(
        tmp_path / "velocity #1",
        iter(values),
        source_revision="source-1",
        field="velocity",
        units="m/s",
    )
    assert receipt.file_sha256 == hashlib.sha256(receipt.path.read_bytes()).hexdigest()
    with NumericIndex(
        receipt.path, source_revision="source-1", expected_revision=receipt.revision
    ) as index:
        assert index.count == 10 and index.numeric_count == 8
        assert (index.field, index.units) == ("velocity", "m/s")
        for descending in (False, True):
            expected = sorted(
                enumerate(values),
                key=lambda r: (
                    r[1] is None,
                    0 if r[1] is None else (-r[1] if descending else r[1]),
                    r[0],
                ),
            )
            assert read_all(index, descending) == expected
        assert index.quantiles() == (-1, 0, 4, 9)
        assert index.page(limit=1).entries[0].value == -2
        assert index.quantiles() == (-1, 0, 4, 9)
        assert index.verify() == receipt.revision
    with pytest.raises(sqlite3.ProgrammingError):
        index.page()


def test_randomized_ordering_matches_whole_dataset(tmp_path):
    rng = random.Random(172)
    values = [None if rng.random() < 0.2 else rng.randint(-20, 20) / 3 for _ in range(1203)]
    receipt = write_numeric_index(tmp_path / "index", values, source_revision="s", field="x")
    with NumericIndex(receipt.path, source_revision="s") as index:
        finite = sorted(v for v in values if v is not None)
        assert index.quantiles() == tuple(
            finite[(b * len(finite) + 3) // 4 - 1] for b in range(1, 5)
        )
        for descending in (False, True):
            expected = sorted(
                enumerate(values),
                key=lambda r: (
                    r[1] is None,
                    0 if r[1] is None else (-r[1] if descending else r[1]),
                    r[0],
                ),
            )
            for size in (1, 17, 500):
                assert read_all(index, descending, size) == expected


@pytest.mark.parametrize("values", [[], [None] * 3, [1], [1] * 20])
def test_empty_sparse_and_equal_columns(tmp_path, values):
    receipt = write_numeric_index(tmp_path / "index", values, source_revision="s", field="x")
    with NumericIndex(receipt.path, source_revision="s") as index:
        assert read_all(index) == list(enumerate(values))
        expected = tuple([1] * min(4, len(values))) if values and values[0] == 1 else ()
        assert index.quantiles() == expected
        assert index.verify() == receipt.revision


@pytest.mark.parametrize("bad", [True, "1", float("nan"), float("inf"), -float("inf"), 2**53 + 1])
def test_invalid_values_never_publish(tmp_path, bad):
    with pytest.raises(ValueError):
        write_numeric_index(tmp_path / "index", [1, bad], source_revision="s", field="x")
    assert list(tmp_path.iterdir()) == []


def test_interrupt_and_publication_race(tmp_path):
    target = tmp_path / "index"

    def interrupted():
        yield 1
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        write_numeric_index(target, interrupted(), source_revision="s", field="x")
    assert list(tmp_path.iterdir()) == []

    def raced():
        yield 1
        target.write_bytes(b"other writer")

    with pytest.raises(FileExistsError):
        write_numeric_index(target, raced(), source_revision="s", field="x")
    assert target.read_bytes() == b"other writer"
    assert list(tmp_path.iterdir()) == [target]


def test_source_revision_and_selection_bound_cursors(tmp_path):
    a = write_numeric_index(tmp_path / "a", [1, 2, None], source_revision="a", field="x")
    b = write_numeric_index(tmp_path / "b", [1, 2, None], source_revision="b", field="x")
    with pytest.raises(ValueError, match="source revision"):
        NumericIndex(a.path, source_revision="b")
    with pytest.raises(ValueError, match="expected revision"):
        NumericIndex(a.path, source_revision="a", expected_revision="0" * 64)
    with (
        NumericIndex(a.path, source_revision="a") as index,
        NumericIndex(b.path, source_revision="b") as other,
    ):
        cursor = index.page(limit=1).next_cursor
        assert cursor is not None
        with pytest.raises(ValueError):
            other.page(cursor=cursor)
        with pytest.raises(ValueError):
            index.page(cursor=cursor, descending=True)
        for field, value in [
            ("missing", True),
            ("ordinal", -1),
            ("sort_value", float("inf")),
            ("sort_value", 2**10000),
            ("descending", 1),
            ("ordinal", True),
        ]:
            with pytest.raises(ValueError):
                index.page(cursor=dataclasses.replace(cursor, **{field: value}))
        for limit in (False, 0, 4097):
            with pytest.raises(ValueError):
                index.page(limit=limit)
        for bins in (False, 0, 65):
            with pytest.raises(ValueError):
                index.quantiles(bins)
        with pytest.raises(ValueError):
            index.page(cursor=NumericCursor(index.revision, False, 1, 3.0, 0))


def test_payload_damage_deletion_and_schema_rejected(tmp_path):
    receipt = write_numeric_index(tmp_path / "index", [1, 2, 3], source_revision="s", field="x")
    with sqlite3.connect(receipt.path) as db:
        db.execute("UPDATE entries SET value=4 WHERE ordinal=0")
    with NumericIndex(receipt.path, source_revision="s") as index:
        with pytest.raises(ValueError, match="integrity"):
            index.page(limit=100)
        with pytest.raises(ValueError, match="integrity"):
            index.verify()
    removed = write_numeric_index(tmp_path / "removed", [1, 2, 3], source_revision="s", field="x")
    with sqlite3.connect(removed.path) as db:
        db.execute("DELETE FROM entries WHERE ordinal=1")
    with NumericIndex(removed.path, source_revision="s") as index:
        with pytest.raises(ValueError, match="contiguous"):
            index.verify()
    with sqlite3.connect(removed.path) as db:
        db.execute("CREATE TABLE unexpected(value TEXT)")
    with pytest.raises(ValueError, match="schema"):
        NumericIndex(removed.path, source_revision="s")


def test_deep_pages_use_composite_index(tmp_path):
    receipt = write_numeric_index(tmp_path / "index", range(2000), source_revision="s", field="x")
    with NumericIndex(receipt.path, source_revision="s") as index:
        for column in ("forward", "backward"):
            plan = index._connection.execute(
                f"EXPLAIN QUERY PLAN SELECT ordinal,value,digest FROM entries "
                f"WHERE (missing,{column},ordinal)>(?,?,?) "
                f"ORDER BY missing,{column},ordinal LIMIT 10",
                (0, 100, 100),
            ).fetchall()
            assert any("SEARCH entries USING INDEX" in r[-1] for r in plan), plan
            assert not any("TEMP B-TREE" in r[-1] for r in plan), plan
