"""
tests/test_column_roles.py

column-roles-1: behaviour tests for the role/carry_to_children defaults
docs/architecture.md §4.2 specifies, and for Dataset.columns_to_carry().

Written from §4.2 and the work item's ruling (a non-numeric column that is
not a merge's own join key defaults to identifier), not from the
implementation. Real Dataset + pandas, no Qt. strict_schema is on for the
whole suite (tests/conftest.py).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from models.dataset import Dataset
from models.table_schema import ColumnHint, ColumnRole, ColumnSpec, TableSchema


def _write_csv(tmp_path, name: str, frame: pd.DataFrame):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 1. Import-path defaults
# ---------------------------------------------------------------------------


def test_load_folder_gives_every_text_column_identifier(tmp_path):
    # load_folder only ever produces full_path and file_name, both text.
    # Neither is a merge join key, so both default to identifier.
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"\x00")
    (images / "b.jpg").write_bytes(b"\x00")

    ds = Dataset()
    ds.load_folder(images)

    schema = ds.schema_for("frames")
    assert schema.spec_for("full_path").role is ColumnRole.identifier
    assert schema.spec_for("file_name").role is ColumnRole.identifier
    assert schema.spec_for("full_path").carry_to_children is True
    assert schema.spec_for("file_name").carry_to_children is True


def test_load_csv_as_primary_splits_numeric_measurement_from_text_identifier(
    tmp_path,
):
    csv = _write_csv(
        tmp_path,
        "primary.csv",
        pd.DataFrame(
            {
                "condition": ["approach", "approach", "withdraw"],
                "reaction_time": [0.51, 0.62, 0.48],
            }
        ),
    )
    ds = Dataset()
    ds.load_csv_as_primary(csv)

    schema = ds.schema_for("frames")
    assert schema.spec_for("condition").role is ColumnRole.identifier
    assert schema.spec_for("reaction_time").role is ColumnRole.measurement
    # carry_to_children true either way, per §4.2.
    assert schema.spec_for("condition").carry_to_children is True
    assert schema.spec_for("reaction_time").carry_to_children is True


def test_merge_csv_gives_merged_in_text_identifier_and_numeric_measurement(
    tmp_path,
):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"\x00")
    ds = Dataset()
    ds.load_folder(images)
    file_name = ds.get_table("frames")["file_name"].iloc[0]

    csv = _write_csv(
        tmp_path,
        "meta.csv",
        pd.DataFrame(
            {
                "file_name": [file_name],
                "session_id": ["S03"],
                "timestamp": [22.247],
            }
        ),
    )
    report = ds.merge_csv(
        csv, target_table="frames", csv_key="file_name", target_key="file_name",
    )
    ds.confirm_merge(report)

    schema = ds.schema_for("frames")
    # A merged-in text column that is not the join key: identifier default.
    assert schema.spec_for("session_id").role is ColumnRole.identifier
    # A merged-in numeric column: measurement default.
    assert schema.spec_for("timestamp").role is ColumnRole.measurement


def test_merge_csv_own_key_column_is_identifier_even_when_numeric(tmp_path):
    # target_key and csv_key are DIFFERENT names, so pandas keeps both
    # columns in the joined frame (see models/dataset.py's confirm_merge
    # docstring). The CSV's own key column, subj_id, is numeric, so the
    # plain numeric->measurement default would misclassify it; the merge's
    # explicit "the join key is an identifier" hint must override that.
    ds = Dataset()
    ds.create_table_from_df(
        "t",
        pd.DataFrame(
            {
                "subj_num": pd.Series([1, 2, 3], dtype="int64"),
                "value": pd.Series([1.1, 2.2, 3.3], dtype="float64"),
            }
        ),
    )
    # subj_num, being numeric, defaulted to measurement on creation -- this
    # merge must not go back and change that (see next test).
    assert ds.schema_for("t").spec_for("subj_num").role is ColumnRole.measurement

    csv = _write_csv(
        tmp_path,
        "ids.csv",
        pd.DataFrame({"subj_id": [1, 2, 3], "score": [10, 20, 30]}),
    )
    report = ds.merge_csv(csv, target_table="t", csv_key="subj_id", target_key="subj_num")
    ds.confirm_merge(report)

    schema = ds.schema_for("t")
    assert schema.spec_for("subj_id").role is ColumnRole.identifier
    # A merged-in numeric column that is NOT the key still defaults measurement.
    assert schema.spec_for("score").role is ColumnRole.measurement


def test_merge_csv_never_retags_the_pre_existing_target_key(tmp_path):
    # target_key already named an existing column of target_table before
    # this merge ran; confirm_merge must not reach back and change a role
    # that table already had, even though that role came from the general
    # numeric default and not from being "the join key".
    ds = Dataset()
    ds.create_table_from_df(
        "t", pd.DataFrame({"subj_num": pd.Series([1, 2, 3], dtype="int64")})
    )
    csv = _write_csv(
        tmp_path, "ids2.csv", pd.DataFrame({"subj_id": [1, 2, 3], "score": [1, 2, 3]})
    )
    report = ds.merge_csv(csv, target_table="t", csv_key="subj_id", target_key="subj_num")
    ds.confirm_merge(report)

    assert ds.schema_for("t").spec_for("subj_num").role is ColumnRole.measurement


def test_add_computed_column_and_add_column_follow_the_same_dtype_default(tmp_path):
    ds = Dataset()
    ds.load_csv_as_primary(
        _write_csv(tmp_path, "p.csv", pd.DataFrame({"x": [1, 2, 3]}))
    )
    ds.add_computed_column("doubled", "x * 2", col_type="numeric")
    ds.add_column(
        "label",
        pd.Series(
            {rid: "hit" for rid in ds.get_table("frames")["row_id"]}
        ),
        col_type="text",
    )

    schema = ds.schema_for("frames")
    assert schema.spec_for("doubled").role is ColumnRole.measurement
    assert schema.spec_for("label").role is ColumnRole.identifier


def test_aggregate_and_create_table_from_rows_and_df_use_the_same_default(tmp_path):
    ds = Dataset()
    ds.load_csv_as_primary(
        _write_csv(
            tmp_path,
            "g.csv",
            pd.DataFrame(
                {
                    "subject": ["sA", "sA", "sB", "sB"],
                    "score": [1.0, 2.0, 3.0, 4.0],
                }
            ),
        )
    )

    ds.aggregate("agg", "frames", "subject", {"score": "mean"})
    agg_schema = ds.schema_for("agg")
    assert agg_schema.spec_for("subject").role is ColumnRole.identifier
    assert agg_schema.spec_for("score").role is ColumnRole.measurement

    row_ids = list(ds.get_table("frames")["row_id"])
    ds.create_table_from_rows("subset", row_ids[:2])
    subset_schema = ds.schema_for("subset")
    assert subset_schema.spec_for("subject").role is ColumnRole.identifier
    assert subset_schema.spec_for("score").role is ColumnRole.measurement

    ds.create_table_from_df(
        "built",
        pd.DataFrame(
            {"label": ["a", "b"], "amount": pd.Series([1, 2], dtype="int64")}
        ),
    )
    built_schema = ds.schema_for("built")
    assert built_schema.spec_for("label").role is ColumnRole.identifier
    assert built_schema.spec_for("amount").role is ColumnRole.measurement


def test_aggregate_group_by_is_identifier_even_when_numeric(tmp_path):
    # A group-by column names the entity each output row belongs to -- the
    # definition of identifier in §4.2 -- so a NUMERIC group-by key (e.g. a
    # numeric subject_id) must not fall through to the plain
    # numeric->measurement default the way an ordinary numeric column would.
    ds = Dataset()
    ds.load_csv_as_primary(
        _write_csv(
            tmp_path,
            "num_group.csv",
            pd.DataFrame(
                {
                    "subject_id": pd.Series([1, 1, 2, 2], dtype="int64"),
                    "score": [1.0, 2.0, 3.0, 4.0],
                }
            ),
        )
    )

    ds.aggregate("agg", "frames", "subject_id", {"score": "mean"})

    schema = ds.schema_for("agg")
    assert schema.spec_for("subject_id").role is ColumnRole.identifier
    assert schema.spec_for("score").role is ColumnRole.measurement


def test_aggregate_multi_column_group_by_marks_every_key_identifier(tmp_path):
    ds = Dataset()
    ds.load_csv_as_primary(
        _write_csv(
            tmp_path,
            "multi_group.csv",
            pd.DataFrame(
                {
                    "subject_id": pd.Series([1, 1, 2, 2], dtype="int64"),
                    "condition": ["a", "b", "a", "b"],
                    "score": [1.0, 2.0, 3.0, 4.0],
                }
            ),
        )
    )

    ds.aggregate("agg", "frames", ["subject_id", "condition"], {"score": "mean"})

    schema = ds.schema_for("agg")
    assert schema.spec_for("subject_id").role is ColumnRole.identifier
    assert schema.spec_for("condition").role is ColumnRole.identifier
    assert schema.spec_for("score").role is ColumnRole.measurement


# ---------------------------------------------------------------------------
# 2. Dataset.columns_to_carry()
# ---------------------------------------------------------------------------


def _table_with_explicit_roles(ds: Dataset) -> None:
    """A table with one of each role, an unflagged measurement, and a
    flagged one -- built with an explicit schema so the test controls every
    role directly rather than relying on any default."""
    df = pd.DataFrame(
        {
            "row_id": ["1", "2"],
            "participant_id": ["p07", "p08"],
            "trial_id": pd.Series([1, 2], dtype="int64"),
            "reaction_time": pd.Series([0.5, 0.6], dtype="float64"),
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


def test_columns_to_carry_with_no_narrowing_matches_carried_columns():
    ds = Dataset()
    _table_with_explicit_roles(ds)

    result = ds.columns_to_carry("source")

    assert set(result) == {"participant_id", "trial_id", "reaction_time"}
    assert "blendshape_score" not in result


def test_columns_to_carry_narrowing_still_keeps_identifier_and_index():
    ds = Dataset()
    _table_with_explicit_roles(ds)

    # Narrow to nothing from the measurement side -- identifier/index survive.
    result = ds.columns_to_carry("source", carry_columns=[])

    assert set(result) == {"participant_id", "trial_id"}


def test_columns_to_carry_narrowing_cannot_add_back_an_unflagged_measurement():
    ds = Dataset()
    _table_with_explicit_roles(ds)

    # Explicitly naming blendshape_score does not carry it: carry_columns
    # only narrows the default-carried set, it never overrides
    # carry_to_children=False.
    result = ds.columns_to_carry("source", carry_columns=["blendshape_score"])

    assert "blendshape_score" not in result
    assert set(result) == {"participant_id", "trial_id"}


def test_columns_to_carry_narrowing_selects_a_flagged_measurement():
    ds = Dataset()
    _table_with_explicit_roles(ds)

    result = ds.columns_to_carry("source", carry_columns=["reaction_time"])

    assert set(result) == {"participant_id", "trial_id", "reaction_time"}


def test_columns_to_carry_returns_source_schema_order():
    ds = Dataset()
    _table_with_explicit_roles(ds)

    result = ds.columns_to_carry("source")

    # Schema order is participant_id, trial_id, reaction_time, ... -- the
    # result must follow that, not carry_columns' order.
    assert result == ["participant_id", "trial_id", "reaction_time"]


def test_columns_to_carry_unknown_carry_column_raises():
    ds = Dataset()
    _table_with_explicit_roles(ds)

    with pytest.raises(ValueError):
        ds.columns_to_carry("source", carry_columns=["not_a_real_column"])


def test_columns_to_carry_unknown_table_raises():
    ds = Dataset()
    with pytest.raises(KeyError):
        ds.columns_to_carry("does_not_exist")


def test_columns_to_carry_table_with_no_schema_carries_nothing():
    ds = Dataset()
    # Bypass the schema accept path entirely, as tests elsewhere do to model
    # "a table with no marked columns" -- schema_for() then returns None.
    ds._tables["raw"] = pd.DataFrame({"row_id": ["1"], "x": [1]})

    assert ds.columns_to_carry("raw") == []


# ---------------------------------------------------------------------------
# 3. An old project (no schemas.json) opens with the new defaults, and a
#    project already carrying a schemas.json keeps exactly what it saved.
# ---------------------------------------------------------------------------


def test_a_pre_schemas_json_project_opens_with_the_new_role_defaults(tmp_path):
    csv = _write_csv(
        tmp_path,
        "old.csv",
        pd.DataFrame({"condition": ["hit", "miss"], "score": [1.0, 2.0]}),
    )
    ds = Dataset()
    ds.load_csv_as_primary(csv)
    proj = tmp_path / "proj"
    ds.save(proj)
    # Simulate a project saved before schemas.json existed at all.
    (proj / "schemas.json").unlink()

    ds2 = Dataset()
    ds2.load(proj)  # must not raise -- "an old project must open"

    schema = ds2.schema_for("frames")
    assert schema.spec_for("condition").role is ColumnRole.identifier
    assert schema.spec_for("score").role is ColumnRole.measurement


def test_a_schemas_json_written_under_the_old_blanket_default_is_honoured_as_is(
    tmp_path,
):
    # Before this item, infer_schema gave EVERY column role "measurement".
    # A schemas.json written by that code (the only kind that could exist
    # before this item shipped) is honoured exactly as saved -- Dataset
    # does not silently reclassify an already-decided role on load, the
    # same way it never re-derives a dtype a saved schema already pins.
    csv = _write_csv(
        tmp_path,
        "hand.csv",
        pd.DataFrame({"condition": ["hit", "miss"], "score": [1.0, 2.0]}),
    )
    ds = Dataset()
    ds.load_csv_as_primary(csv)
    proj = tmp_path / "proj"
    ds.save(proj)

    sidecar = json.loads((proj / "schemas.json").read_text())
    for entry in sidecar["schemas"]["frames"]["columns"]:
        if entry["name"] == "condition":
            entry["role"] = "measurement"  # the pre-item blanket default
    (proj / "schemas.json").write_text(json.dumps(sidecar))

    ds2 = Dataset()
    ds2.load(proj)

    assert ds2.schema_for("frames").spec_for("condition").role is ColumnRole.measurement


# ---------------------------------------------------------------------------
# 4. Round-trip through save/load
# ---------------------------------------------------------------------------


def test_roles_and_carry_flags_round_trip_through_save_and_load(tmp_path):
    ds = Dataset()
    ds.load_csv_as_primary(
        _write_csv(
            tmp_path,
            "rt.csv",
            pd.DataFrame(
                {"condition": ["hit", "miss"], "reaction_time": [0.5, 0.6]}
            ),
        )
    )
    before = ds.schema_for("frames")

    proj = tmp_path / "proj"
    ds.save(proj)
    ds2 = Dataset()
    ds2.load(proj)
    after = ds2.schema_for("frames")

    assert after == before
    assert after.spec_for("condition").role is ColumnRole.identifier
    assert after.spec_for("condition").carry_to_children is True
    assert after.spec_for("reaction_time").role is ColumnRole.measurement
    assert after.spec_for("reaction_time").carry_to_children is True
