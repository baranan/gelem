"""
tests/test_table_versions.py

Tests for P1.12b: a single monotonically increasing write-ticket version
per Dataset instance, bumped on every accepted commit, exposed as
Dataset.table_version(name) and Dataset.table_versions().

Written from the P1.12b work-item specification, not from the
implementation. Nothing consumes the counter yet (P1.12f will); these
tests only check that the number is minted, bumped, never reused, and
cleared -- but not reset -- by a fresh load.

No Qt, no widgets -- this file joins the combined non-widget group.

Run with:
    python -m pytest tests/test_table_versions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import pytest

from models.dataset import Dataset


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ds_with_frames(tmp_path) -> Dataset:
    """A Dataset whose 'frames' table has a repeated-string column ('grp')
    and a decimal float column ('score'), built through the real CSV path.
    Import-time schema messages are drained so a later assertion on them is
    clean."""
    csv = tmp_path / "d.csv"
    csv.write_text(
        "grp,score\n"
        "A,1.5\n"
        "B,2.5\n"
        "C,3.5\n"
        "D,4.5\n"
    )
    ds = Dataset()
    ds.load_csv_as_primary(csv)
    ds.take_schema_messages()
    return ds


def _first_row_id(ds: Dataset, table: str = "frames") -> str:
    return ds.get_table(table)["row_id"].iloc[0]


# ---------------------------------------------------------------------------
# case 1 -- a newly accepted table has a version
# ---------------------------------------------------------------------------

def test_a_newly_accepted_table_has_a_version():
    # Dataset.__init__ accepts an empty 'frames' table, so a bare Dataset
    # already has a version for it.
    ds = Dataset()
    version = ds.table_version("frames")
    assert isinstance(version, int)
    assert version >= 1


# ---------------------------------------------------------------------------
# case 2 -- a second accepted write to the same table strictly increases it
# ---------------------------------------------------------------------------

def test_a_second_accepted_write_to_the_same_table_strictly_increases_the_version(tmp_path):
    ds = _ds_with_frames(tmp_path)
    before = ds.table_version("frames")

    rid = _first_row_id(ds)
    ds.apply_row_updates("frames", {rid: {"note_count": 7}})

    after = ds.table_version("frames")
    assert after > before


# ---------------------------------------------------------------------------
# case 3 -- two different tables never share a number at the same time
# ---------------------------------------------------------------------------

def test_two_different_tables_never_share_a_number(tmp_path):
    ds = _ds_with_frames(tmp_path)
    # A second table, created from a pre-built frame.
    ds.create_table_from_df("built", pd.DataFrame({"x": [1, 2, 3]}))

    assert ds.table_version("frames") != ds.table_version("built")

    # And no two tables anywhere share a version right now.
    versions = list(ds.table_versions().values())
    assert len(versions) == len(set(versions))


# ---------------------------------------------------------------------------
# case 4 -- every accepted write path bumps its table's version (one each)
# ---------------------------------------------------------------------------

def test_add_column_bumps_the_version(tmp_path):
    ds = _ds_with_frames(tmp_path)
    before = ds.table_version("frames")

    row_ids = list(ds.get_table("frames")["row_id"])
    values = pd.Series({rid: 1.5 + i for i, rid in enumerate(row_ids)})
    ds.add_column("mapped", values, "numeric")

    assert ds.table_version("frames") > before


def test_add_computed_column_bumps_the_version(tmp_path):
    ds = _ds_with_frames(tmp_path)
    before = ds.table_version("frames")

    ds.add_computed_column("double_score", "score * 2")

    assert ds.table_version("frames") > before


def test_update_row_bumps_the_version(tmp_path):
    ds = _ds_with_frames(tmp_path)
    before = ds.table_version("frames")

    ds.update_row(_first_row_id(ds), {"note_count": 3})

    assert ds.table_version("frames") > before


def test_apply_row_updates_bumps_the_version(tmp_path):
    ds = _ds_with_frames(tmp_path)
    before = ds.table_version("frames")

    ds.apply_row_updates("frames", {_first_row_id(ds): {"note_count": 9}})

    assert ds.table_version("frames") > before


def test_create_table_from_rows_bumps_the_version(tmp_path):
    ds = _ds_with_frames(tmp_path)
    # The new table did not exist before, so "bump" here means "it gets a
    # version, and it is a fresh ticket -- higher than 'frames' currently
    # holds".
    frames_version = ds.table_version("frames")
    row_ids = list(ds.get_table("frames")["row_id"])[:2]
    ds.create_table_from_rows("subset", row_ids, "frames")

    assert ds.table_version("subset") > frames_version


def test_create_table_from_df_bumps_the_version(tmp_path):
    ds = _ds_with_frames(tmp_path)
    frames_version = ds.table_version("frames")

    ds.create_table_from_df("built", pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

    assert ds.table_version("built") > frames_version


def test_aggregate_bumps_the_version(tmp_path):
    ds = _ds_with_frames(tmp_path)
    frames_version = ds.table_version("frames")

    ds.aggregate("by_grp", "frames", "grp", {"score": "mean"})

    assert ds.table_version("by_grp") > frames_version


# ---------------------------------------------------------------------------
# case 5 -- THE IMPORTANT ONE: a rejected, rolled-back apply_row_updates does
#           NOT bump the version
# ---------------------------------------------------------------------------

def test_a_rejected_apply_row_updates_does_not_bump_the_version(tmp_path):
    ds = _ds_with_frames(tmp_path)
    # strict_schema False so the rejection degrades to "all unplaceable"
    # rather than re-raising -- this test is about the rollback path, which
    # returns WITHOUT reaching _accept_table.
    ds.strict_schema = False

    rid = _first_row_id(ds)
    before = ds.table_version("frames")

    # An operator emitting a Timestamp: the new column infers to datetime64,
    # which infer_schema refuses -> _accept_table raises SchemaRejection ->
    # apply_row_updates rolls back and reports the whole batch unplaceable.
    result = ds.apply_row_updates(
        "frames", {rid: {"emitted_at": pd.Timestamp("2024-01-01")}}
    )

    # The rollback really happened: the batch came back unplaceable AND the
    # column was not stored.
    assert result == [rid], "expected the rejected batch to be reported unplaceable"
    assert "emitted_at" not in ds.read_only_view("frames").columns

    # ...and no version was minted for a write that never landed.
    assert ds.table_version("frames") == before


# ---------------------------------------------------------------------------
# case 6 -- a fresh load clears the version map but not the counter
# ---------------------------------------------------------------------------

def test_a_fresh_load_mints_higher_numbers_and_drops_pre_reset_tables(tmp_path):
    # Project on disk: 'frames' plus one aggregated table.
    ds1 = _ds_with_frames(tmp_path)
    ds1.aggregate("by_grp", "frames", "grp", {"score": "mean"})
    proj = tmp_path / "proj"
    ds1.save(proj)

    # A second Dataset that first builds a table of its own ('scratch'), then
    # loads the project -- which goes through _reset_tables.
    ds2 = _ds_with_frames(tmp_path)
    ds2.create_table_from_df("scratch", pd.DataFrame({"z": [1, 2]}))
    highest_before_reset = max(ds2.table_versions().values())
    assert "scratch" in ds2.table_versions()

    ds2.load(proj)

    # Every number minted by the load is strictly greater than any seen
    # before the reset: the counter was NOT reset, only the map was cleared.
    for name, version in ds2.table_versions().items():
        assert version > highest_before_reset, (
            f"{name!r} got version {version}, not above pre-reset high "
            f"{highest_before_reset}"
        )

    # A table that existed only before the reset is gone from the map.
    assert "scratch" not in ds2.table_versions()
    with pytest.raises(KeyError):
        ds2.table_version("scratch")

    # The loaded tables are all present with a version.
    assert ds2.table_version("frames") > highest_before_reset
    assert ds2.table_version("by_grp") > highest_before_reset


# ---------------------------------------------------------------------------
# case 7 -- table_version() on an unknown table raises KeyError
# ---------------------------------------------------------------------------

def test_table_version_on_an_unknown_table_raises_key_error():
    ds = Dataset()
    with pytest.raises(KeyError):
        ds.table_version("no_such_table")


# ---------------------------------------------------------------------------
# case 8 -- a table assigned straight into _tables has no version
# ---------------------------------------------------------------------------

def test_a_table_assigned_straight_into_tables_has_no_version():
    # A frame put directly into Dataset._tables never went through
    # _commit_prepared, so it has no schema (schema_for() returns None) and,
    # by exactly the same trap, no version -- table_version() raises.
    ds = Dataset()
    ds._tables["sneaky"] = pd.DataFrame(
        {"row_id": ["1", "2"], "k": [10, 20]}
    )

    assert ds.schema_for("sneaky") is None
    with pytest.raises(KeyError):
        ds.table_version("sneaky")

    assert "sneaky" not in ds.table_versions()
