"""
tests/test_merge_expansion.py

P1.5b: behaviour tests for the merge-expansion offer -- Dataset.merge_csv()
and Dataset.confirm_merge() creating a NEW table (one row per CSV row)
instead of refusing outright when a CSV key would expand the target table,
plus the wording layer of ui/merge_report_dialog.py that describes the
offer.

Written from the work item's spec and docs/architecture.md §4.2 (lineage,
columns_to_carry), not from the implementation. Real Dataset + pandas, no
Qt for the Dataset-level tests. strict_schema is on for the whole suite
(tests/conftest.py).
"""

from __future__ import annotations

import pandas as pd
import pytest

from models.dataset import Dataset
from models.table_schema import ColumnRole, ColumnSpec, TableSchema


def _write_csv(tmp_path, name: str, frame: pd.DataFrame):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _build_target_table(ds: Dataset) -> None:
    """A target table with one row per participant and an explicit schema:
    participant_id (identifier), trial_id (index), reaction_time
    (measurement, carried), blendshape_score (measurement, NOT carried).
    Built with an explicit schema, like tests/test_column_roles.py's
    fixture, so the test controls every role directly.
    """
    df = pd.DataFrame(
        {
            "row_id": ["1", "2"],
            "participant_id": ["p07", "p08"],
            "trial_id": pd.Series([1, 2], dtype="int64"),
            "reaction_time": pd.Series([0.51, 0.62], dtype="float64"),
            "blendshape_score": pd.Series([0.1, 0.2], dtype="float64"),
        }
    )
    schema = TableSchema(
        columns=(
            ColumnSpec("participant_id", "text", "object", ColumnRole.identifier, True),
            ColumnSpec("trial_id", "numeric", "int64", ColumnRole.index, True),
            ColumnSpec("reaction_time", "numeric", "float64", ColumnRole.measurement, True),
            ColumnSpec("blendshape_score", "numeric", "float64", ColumnRole.measurement, False),
        )
    )
    ds._accept_table("source", df, schema=schema, source="test")


def _expanding_csv(tmp_path):
    """Three trials for p07 and two for p08 -- participant_id repeats on
    both sides of the join, which is exactly what triggers would_expand."""
    return _write_csv(
        tmp_path,
        "trials.csv",
        pd.DataFrame(
            {
                "participant_id": ["p07", "p07", "p07", "p08", "p08"],
                "trial_num": [1, 2, 3, 1, 2],
                "score": [10, 20, 30, 40, 50],
            }
        ),
    )


# ---------------------------------------------------------------------------
# 1. Dataset.merge_csv() / confirm_merge() -- the expansion offer
# ---------------------------------------------------------------------------


def test_expand_report_names_a_default_table_and_row_count(tmp_path):
    ds = Dataset()
    _build_target_table(ds)
    report = ds.merge_csv(
        _expanding_csv(tmp_path),
        target_table="source", csv_key="participant_id", target_key="participant_id",
    )
    assert report.would_expand
    assert report.expand_table_name == "source_expanded"
    assert report.expand_row_count == 5


def test_expand_carried_columns_are_columns_to_carry_minus_the_join_key(tmp_path):
    # csv_key and target_key are both "participant_id" here, and
    # participant_id is an identifier, so it IS one of
    # columns_to_carry("source")'s results -- but P1.5b-fix: the join key
    # is handled by its own unconditional rule (see _build_expand_frame),
    # never as a generic carried column, so it must not appear twice in
    # expand_carried_columns's accounting either. See
    # test_join_key_present_exactly_once_when_names_match for the
    # corresponding table-level check.
    ds = Dataset()
    _build_target_table(ds)
    report = ds.merge_csv(
        _expanding_csv(tmp_path),
        target_table="source", csv_key="participant_id", target_key="participant_id",
    )
    assert report.expand_carried_columns == ["trial_id", "reaction_time"]
    assert "participant_id" not in report.expand_carried_columns
    assert "blendshape_score" not in report.expand_carried_columns


def test_confirm_merge_does_not_touch_the_target_table(tmp_path):
    ds = Dataset()
    _build_target_table(ds)
    before = ds.get_table("source").copy()
    report = ds.merge_csv(
        _expanding_csv(tmp_path),
        target_table="source", csv_key="participant_id", target_key="participant_id",
    )
    ds.confirm_merge(report)
    after = ds.get_table("source")
    pd.testing.assert_frame_equal(before, after)


def test_confirm_merge_creates_one_row_per_csv_row(tmp_path):
    ds = Dataset()
    _build_target_table(ds)
    report = ds.merge_csv(
        _expanding_csv(tmp_path),
        target_table="source", csv_key="participant_id", target_key="participant_id",
    )
    ds.confirm_merge(report)
    new_table = ds.get_table("source_expanded")
    assert len(new_table) == 5


def test_confirm_merge_every_csv_column_comes_across(tmp_path):
    ds = Dataset()
    _build_target_table(ds)
    report = ds.merge_csv(
        _expanding_csv(tmp_path),
        target_table="source", csv_key="participant_id", target_key="participant_id",
    )
    ds.confirm_merge(report)
    new_table = ds.get_table("source_expanded")
    assert list(new_table["trial_num"]) == [1, 2, 3, 1, 2]
    assert list(new_table["score"]) == [10, 20, 30, 40, 50]


def test_confirm_merge_carried_columns_have_correct_per_row_values(tmp_path):
    ds = Dataset()
    _build_target_table(ds)
    report = ds.merge_csv(
        _expanding_csv(tmp_path),
        target_table="source", csv_key="participant_id", target_key="participant_id",
    )
    ds.confirm_merge(report)
    new_table = ds.get_table("source_expanded")

    # Identifier survives with the right value on every derived row.
    assert list(new_table["participant_id"]) == ["p07", "p07", "p07", "p08", "p08"]
    # Index survives too -- carried from the parent, distinct from the
    # CSV's own trial_num column.
    assert list(new_table["trial_id"]) == [1, 1, 1, 2, 2]
    # A carried, flagged measurement -- constant per participant.
    assert list(new_table["reaction_time"]) == [0.51, 0.51, 0.51, 0.62, 0.62]
    # The unflagged measurement must not have been carried at all.
    assert "blendshape_score" not in new_table.columns


def test_confirm_merge_new_rows_get_fresh_row_ids(tmp_path):
    ds = Dataset()
    _build_target_table(ds)
    report = ds.merge_csv(
        _expanding_csv(tmp_path),
        target_table="source", csv_key="participant_id", target_key="participant_id",
    )
    ds.confirm_merge(report)
    new_ids = list(ds.get_table("source_expanded")["row_id"])
    assert len(set(new_ids)) == 5
    assert not set(new_ids) & set(ds.get_table("source")["row_id"])


def test_confirm_merge_refuses_an_existing_table_name(tmp_path):
    ds = Dataset()
    _build_target_table(ds)
    # Pre-create a table under the default expansion name.
    ds.create_table_from_df("source_expanded", pd.DataFrame({"x": [1, 2]}))
    existing_before = ds.get_table("source_expanded").copy()

    report = ds.merge_csv(
        _expanding_csv(tmp_path),
        target_table="source", csv_key="participant_id", target_key="participant_id",
    )
    with pytest.raises(ValueError):
        ds.confirm_merge(report)

    # Refused, not silently overwritten.
    pd.testing.assert_frame_equal(ds.get_table("source_expanded"), existing_before)


def test_target_key_not_in_carry_columns_does_not_leak_into_the_new_table(tmp_path):
    # target_key ("match_key") is a measurement that is NOT carried; only
    # participant_id is. The join needs match_key's values to align rows,
    # but the result must not smuggle it in as an extra column just
    # because it was needed for the join.
    ds = Dataset()
    df = pd.DataFrame(
        {
            "row_id": ["1", "2"],
            "match_key": pd.Series([10, 20], dtype="int64"),
            "participant_id": ["p1", "p2"],
        }
    )
    schema = TableSchema(
        columns=(
            ColumnSpec("match_key", "numeric", "int64", ColumnRole.measurement, False),
            ColumnSpec("participant_id", "text", "object", ColumnRole.identifier, True),
        )
    )
    ds._accept_table("vids", df, schema=schema, source="test")

    csv = _write_csv(
        tmp_path,
        "ext.csv",
        pd.DataFrame({"ext_key": [10, 10, 20, 20], "val": [1, 2, 3, 4]}),
    )
    report = ds.merge_csv(
        csv, target_table="vids", csv_key="ext_key", target_key="match_key",
    )
    assert report.would_expand
    ds.confirm_merge(report)

    new_table = ds.get_table("vids_expanded")
    assert "match_key" not in new_table.columns
    assert list(new_table.columns).count("ext_key") == 1
    assert list(new_table["participant_id"]) == ["p1", "p1", "p2", "p2"]
    assert list(new_table["ext_key"]) == [10, 10, 20, 20]


# ---------------------------------------------------------------------------
# 1b. P1.5b-fix: the join key's presence is unconditional and exactly once.
#
# docs/architecture.md §4.2: the column a merge joined on is what says which
# source row a new row came from -- it is lineage, so it must survive no
# matter what accident of the data (matching names, a role, a
# carry_to_children flag) would otherwise make it vanish or double up.
# Before this fix, "different names AND target_key also happens to be a
# carried column" produced TWO columns for one value -- see
# test_join_key_present_exactly_once_when_names_differ_and_key_is_carried.
# ---------------------------------------------------------------------------


def _one_row_target(ds: Dataset, *, key_name: str, key_role, carry_to_children: bool) -> None:
    """A minimal one-row target table whose only column is the join key,
    under a controlled name/role/carry_to_children -- isolates exactly how
    the join key's own naming and role affect its survival, independent of
    any other carried column."""
    df = pd.DataFrame({"row_id": ["1"], key_name: ["only"]})
    schema = TableSchema(
        columns=(ColumnSpec(key_name, "text", "object", key_role, carry_to_children),)
    )
    ds._accept_table("t", df, schema=schema, source="test")


def _repeating_csv(tmp_path, csv_key_name: str):
    """Two CSV rows sharing one key value that matches the target's only
    row -- the minimal shape that triggers would_expand."""
    return _write_csv(
        tmp_path, "rep.csv",
        pd.DataFrame({csv_key_name: ["only", "only"], "extra": [1, 2]}),
    )


def test_join_key_present_exactly_once_when_names_match(tmp_path):
    ds = Dataset()
    _one_row_target(ds, key_name="k", key_role=ColumnRole.identifier, carry_to_children=True)
    report = ds.merge_csv(
        _repeating_csv(tmp_path, "k"), target_table="t", csv_key="k", target_key="k",
    )
    assert report.would_expand
    ds.confirm_merge(report)

    new_table = ds.get_table("t_expanded")
    assert list(new_table.columns).count("k") == 1
    assert list(new_table["k"]) == ["only", "only"]


def test_join_key_present_exactly_once_when_names_differ_and_key_is_carried(tmp_path):
    # The combination that used to duplicate: target_key ("video_id")
    # differs from csv_key ("vid") AND target_key is an identifier, so it
    # IS one of columns_to_carry()'s results. The old code only dropped
    # target_key's own column when it was NOT carried, so this exact case
    # kept both "vid" and "video_id" -- two columns for one value.
    ds = Dataset()
    _one_row_target(
        ds, key_name="video_id", key_role=ColumnRole.identifier, carry_to_children=True,
    )
    report = ds.merge_csv(
        _repeating_csv(tmp_path, "vid"),
        target_table="t", csv_key="vid", target_key="video_id",
    )
    assert report.would_expand
    ds.confirm_merge(report)

    new_table = ds.get_table("t_expanded")
    assert "video_id" not in new_table.columns
    assert list(new_table.columns).count("vid") == 1
    assert list(new_table["vid"]) == ["only", "only"]
    # The dialog's "carried from" line must not claim video_id either --
    # it is not a second column the table has, it IS "vid".
    assert report.expand_carried_columns == []


def test_join_key_present_exactly_once_when_key_is_marked_not_carried(tmp_path):
    ds = Dataset()
    _one_row_target(
        ds, key_name="video_id", key_role=ColumnRole.measurement, carry_to_children=False,
    )
    report = ds.merge_csv(
        _repeating_csv(tmp_path, "vid"),
        target_table="t", csv_key="vid", target_key="video_id",
    )
    assert report.would_expand
    ds.confirm_merge(report)

    new_table = ds.get_table("t_expanded")
    assert "video_id" not in new_table.columns
    assert list(new_table.columns).count("vid") == 1
    assert list(new_table["vid"]) == ["only", "only"]


def test_a_duplicate_matching_no_target_row_still_merges_in_place(tmp_path):
    # Sanity check that this item did not change the ordinary path: a
    # duplicate CSV key that matches nothing is harmless and merges
    # in place, exactly as before P1.5b.
    ds = Dataset()
    _build_target_table(ds)
    csv = _write_csv(
        tmp_path,
        "ghost.csv",
        pd.DataFrame(
            {
                "participant_id": ["ghost", "ghost", "p07"],
                "score": [1, 2, 5],
            }
        ),
    )
    report = ds.merge_csv(
        csv, target_table="source", csv_key="participant_id", target_key="participant_id",
    )
    assert report.would_expand == []
    ds.confirm_merge(report)
    assert "score" in ds.get_table("source").columns
    assert "source_expanded" not in ds.list_tables()


# ---------------------------------------------------------------------------
# 2. AppController.confirm_merge() -- signal wiring for the expand case
# ---------------------------------------------------------------------------


def test_controller_confirm_merge_expand_emits_table_created(tmp_path, make_controller):
    controller, dataset, _ = make_controller(tmp_path)

    file_name = dataset.get_table("frames")["file_name"].iloc[0]
    csv = _write_csv(
        tmp_path,
        "expand.csv",
        pd.DataFrame(
            {
                "file_name": [file_name, file_name],
                "trial": [1, 2],
            }
        ),
    )

    reports = []
    controller.merge_report_ready.connect(reports.append)
    controller.load_csv(
        csv, target_table="frames", csv_key="file_name", target_key="file_name",
    )
    assert len(reports) == 1
    report = reports[0]
    assert report.would_expand

    created = []
    tables_updates = []
    controller.table_created.connect(created.append)
    controller.tables_updated.connect(tables_updates.append)
    controller.confirm_merge(report)

    assert created == ["frames_expanded"]
    assert tables_updates and "frames_expanded" in tables_updates[-1]
    assert "frames_expanded" in dataset.list_tables()
    # The target table itself is unaffected.
    assert len(dataset.get_table("frames")) == report.total_target_rows


# ---------------------------------------------------------------------------
# 3. ui/merge_report_dialog.py Layer A -- wording, no Qt.
# ---------------------------------------------------------------------------


from ui.merge_report_dialog import (
    carried_columns_text,
    explain_text,
    header_text,
    is_expand_offer,
    issue_tab_sources,
    proceed_button_text,
)


class _FakeReport:
    """Duck-typed stand-in for MergeReport -- the dialog module never
    imports the real class (see its module docstring)."""

    def __init__(self, **kwargs):
        self.target_table = "frames"
        self.total_csv_rows = 0
        self.total_target_rows = 0
        self.matched_rows = 0
        self.unmatched_target_rows = []
        self.unmatched_csv_rows = []
        self.duplicate_keys_target = []
        self.duplicate_keys_csv = []
        self.would_expand = []
        self.float_key_warning = None
        self.expand_table_name = ""
        self.expand_row_count = 0
        self.expand_carried_columns = []
        self.__dict__.update(kwargs)


def test_is_expand_offer_true_only_when_would_expand_set():
    assert is_expand_offer(_FakeReport(would_expand=["p07"])) is True
    assert is_expand_offer(_FakeReport()) is False


def test_header_text_names_the_new_table_row_count_for_an_offer():
    report = _FakeReport(would_expand=["p07"], expand_row_count=5)
    assert "5" in header_text(report)
    assert "refused" not in header_text(report).lower()


def test_header_text_is_the_ordinary_review_header_otherwise():
    assert header_text(_FakeReport()) == "Review the merge before applying it"


def test_explain_text_is_empty_for_the_ordinary_path():
    assert explain_text(_FakeReport()) == ""


def test_explain_text_names_target_and_new_table_for_an_offer():
    report = _FakeReport(
        would_expand=["p07", "p08"],
        target_table="source",
        expand_table_name="source_expanded",
    )
    text = explain_text(report)
    assert "source_expanded" in text
    assert "source" in text


def test_carried_columns_text_lists_the_carried_columns():
    report = _FakeReport(expand_carried_columns=["participant_id", "trial_id"])
    text = carried_columns_text(report)
    assert "participant_id" in text
    assert "trial_id" in text


def test_carried_columns_text_says_so_when_nothing_is_carried():
    report = _FakeReport(expand_carried_columns=[])
    assert "No columns" in carried_columns_text(report)


def test_proceed_button_text_names_the_new_table_for_an_offer():
    report = _FakeReport(would_expand=["p07"], expand_table_name="source_expanded")
    assert "source_expanded" in proceed_button_text(report)


def test_proceed_button_text_is_generic_for_the_ordinary_path():
    assert proceed_button_text(_FakeReport()) == "Proceed with merge"


def test_issue_tab_sources_includes_the_would_expand_list():
    report = _FakeReport(would_expand=["p07", "p08"])
    sources = dict(issue_tab_sources(report))
    matching = [items for title, items in sources.items() if items == ["p07", "p08"]]
    assert matching, f"would_expand list not found in {sources!r}"
