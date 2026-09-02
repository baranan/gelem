"""
tests/test_dataset_schema.py

P1.8b-1: Dataset holds one TableSchema per stored table, decided only in
Dataset._accept_table, and every write path routes through it. Uses a real
Dataset and pandas -- no Qt, no controller.
"""

import ast
import pathlib

import pandas as pd
import pytest

from models.dataset import Dataset, SchemaRejection, _SCHEMA_EXEMPT_COLUMNS
from models.table_schema import ColumnHint

REPO = pathlib.Path(__file__).parent.parent
TEST_IMAGES = REPO / "test_images"
METADATA_CSV = TEST_IMAGES / "metadata.csv"
DATASET_SRC = REPO / "models" / "dataset.py"

_HAVE_IMAGES = TEST_IMAGES.is_dir() and any(TEST_IMAGES.glob("*.jpg"))
_HAVE_CSV = METADATA_CSV.exists()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _assert_schema_matches_frame(ds: Dataset, table: str) -> None:
    """schema_for(table) names exactly the stored frame's non-row_id columns,
    in the frame's order, and never names row_id."""
    frame = ds.read_only_view(table)
    expected = tuple(
        c for c in frame.columns if c not in _SCHEMA_EXEMPT_COLUMNS
    )
    schema = ds.schema_for(table)
    assert schema is not None, f"{table!r} was not accepted through _accept_table"
    assert schema.column_names() == expected, (
        f"{table!r}: schema {schema.column_names()} != "
        f"frame non-row_id {expected}"
    )
    assert "row_id" not in schema.column_names()


def _ds_with_frames(tmp_path) -> Dataset:
    """A Dataset whose 'frames' table has a repeated-string column ('subject')
    and a decimal float column ('score'), built through the real CSV path."""
    csv = tmp_path / "d.csv"
    csv.write_text(
        "subject,score\n"
        "sA,13.351833\n"
        "sA,7.902144\n"
        "sB,22.447181\n"
        "sB,4.019263\n"
    )
    ds = Dataset()
    ds.load_csv_as_primary(csv)
    return ds


# ---------------------------------------------------------------------------
# check 4 -- WIRING: one test per site (ten), plus __init__
# ---------------------------------------------------------------------------

def test_wiring_init_accepts_the_empty_frames_table():
    ds = Dataset()
    _assert_schema_matches_frame(ds, "frames")
    assert ds.schema_for("frames").column_names() == ("full_path", "file_name")


@pytest.mark.skipif(not _HAVE_IMAGES, reason="test_images/ not present")
def test_wiring_load_folder():
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    _assert_schema_matches_frame(ds, "frames")


def test_wiring_load_csv_as_primary(tmp_path):
    ds = _ds_with_frames(tmp_path)
    _assert_schema_matches_frame(ds, "frames")


@pytest.mark.skipif(
    not (_HAVE_IMAGES and _HAVE_CSV), reason="test_images/metadata.csv not present"
)
def test_wiring_confirm_merge():
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    ds.confirm_merge(ds.merge_csv(METADATA_CSV, join_on="file_name"))
    _assert_schema_matches_frame(ds, "frames")


def test_wiring_add_computed_column(tmp_path):
    ds = _ds_with_frames(tmp_path)
    ds.add_computed_column("double_score", "score * 2")
    _assert_schema_matches_frame(ds, "frames")
    assert "double_score" in ds.schema_for("frames").column_names()


def test_wiring_add_column(tmp_path):
    ds = _ds_with_frames(tmp_path)
    row_ids = list(ds.get_table("frames")["row_id"])
    values = pd.Series({rid: 1.5 + i for i, rid in enumerate(row_ids)})
    ds.add_column("mapped", values, "numeric")
    _assert_schema_matches_frame(ds, "frames")
    assert "mapped" in ds.schema_for("frames").column_names()


def test_wiring_apply_row_updates(tmp_path):
    ds = _ds_with_frames(tmp_path)
    rid = ds.get_table("frames")["row_id"].iloc[0]
    ds.apply_row_updates("frames", {rid: {"note_count": 7}})
    _assert_schema_matches_frame(ds, "frames")
    assert "note_count" in ds.schema_for("frames").column_names()


def test_wiring_aggregate(tmp_path):
    ds = _ds_with_frames(tmp_path)
    ds.aggregate("by_subject", "frames", "subject", {"score": "mean"})
    _assert_schema_matches_frame(ds, "by_subject")


def test_wiring_create_table_from_rows(tmp_path):
    ds = _ds_with_frames(tmp_path)
    row_ids = list(ds.get_table("frames")["row_id"])[:2]
    ds.create_table_from_rows("subset", row_ids, "frames")
    _assert_schema_matches_frame(ds, "subset")


def test_wiring_create_table_from_df():
    ds = Dataset()
    ds.create_table_from_df(
        "built",
        pd.DataFrame({"alpha": [1, 2, 3], "beta": [4.1, 5.2, 6.3]}),
    )
    _assert_schema_matches_frame(ds, "built")
    assert ds.schema_for("built").column_names() == ("alpha", "beta")


def test_wiring_load(tmp_path):
    ds = _ds_with_frames(tmp_path)
    proj = tmp_path / "proj"
    ds.save(proj)
    ds2 = Dataset()
    ds2.load(proj)
    _assert_schema_matches_frame(ds2, "frames")


# ---------------------------------------------------------------------------
# check 3 -- AST GUARDRAIL
# ---------------------------------------------------------------------------

def _callers_of(attr: str) -> set[str]:
    tree = ast.parse(DATASET_SRC.read_text(encoding="utf-8"))
    callers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == attr
            ):
                callers.add(node.name)
    return callers


def test_only_accept_table_calls_set_table():
    set_table_callers = _callers_of("_set_table")
    assert set_table_callers == {"_accept_table"}, (
        f"_set_table must be called only by _accept_table; "
        f"other callers: {set_table_callers - {'_accept_table'}}"
    )

    accept_callers = _callers_of("_accept_table")
    assert accept_callers, "scan found no _accept_table callers -- vacuous"
    ten_sites = {
        "load_folder",
        "load_csv_as_primary",
        "confirm_merge",
        "add_computed_column",
        "add_column",
        "apply_row_updates",
        "aggregate",
        "create_table_from_rows",
        "create_table_from_df",
        "load",
    }
    assert ten_sites <= accept_callers, (
        f"these step-3 sites do not call _accept_table: {ten_sites - accept_callers}"
    )
    assert "__init__" in accept_callers


# ---------------------------------------------------------------------------
# check 1 -- a named column keeps its stored spec (deletion target)
# ---------------------------------------------------------------------------

def test_named_column_keeps_its_stored_spec_across_accepts():
    # The central rule of P1.8b-1. Deletion check 1 re-infers a spec for EVERY
    # column instead of only new ones; then 'k' is re-typed float64 on the
    # second accept, the frame is stored, and this test's pytest.raises fails.
    ds = Dataset()
    df1 = pd.DataFrame(
        {"row_id": ["1", "2", "3"], "k": pd.Series([10, 20, 30], dtype="int64")}
    )
    ds._accept_table("t", df1, source="test")
    assert ds.schema_for("t").spec_for("k").dtype == "int64"

    # 'k' now arrives float64 with a fractional value. Because the stored
    # schema already names 'k' as int64, it is checked against int64 and the
    # 1.5 does not survive -- the spec is NOT re-derived from the new frame.
    df2 = pd.DataFrame(
        {"row_id": ["1", "2", "3"], "k": pd.Series([1.5, 2.0, 3.0], dtype="float64")}
    )
    with pytest.raises(SchemaRejection):
        ds._accept_table("t", df2, source="test")


# ---------------------------------------------------------------------------
# check 2 -- row_id is exempt from every schema (deletion target)
# ---------------------------------------------------------------------------

def test_row_id_is_never_named_by_a_schema(tmp_path):
    # Deletion check 2 removes the row_id exemption; then row_id is inferred
    # into every schema and this fails on all three tables. The wiring tests
    # above (_assert_schema_matches_frame) also each assert row_id absence.
    ds = _ds_with_frames(tmp_path)
    ds.aggregate("agg", "frames", "subject", {"score": "mean"})
    ds.create_table_from_df("built", pd.DataFrame({"z": [1, 2]}))
    for table in ("frames", "agg", "built"):
        assert "row_id" not in ds.schema_for(table).column_names()


# ---------------------------------------------------------------------------
# check 5 -- check_frame runs once per accept, not twice
# ---------------------------------------------------------------------------

def test_check_frame_runs_once_per_accept(monkeypatch):
    import models.dataset as dsmod
    import models.table_schema as tsmod

    calls: list = []
    real = tsmod.check_frame

    def recording(df, schema):
        result = real(df, schema)
        calls.append(result)
        return result

    monkeypatch.setattr(dsmod, "check_frame", recording)
    monkeypatch.setattr(tsmod, "check_frame", recording)

    ds = Dataset()  # accepts 'frames' once
    calls.clear()
    # A ColumnHint(dtype="int32") narrows the arrived int64 column, so the
    # accept produces a storage_policy adjustment and normalise_frame is
    # actually called. If normalise_frame re-ran check_frame instead of using
    # the check= it is handed, this accept would count two calls.
    ds._accept_table(
        "t",
        pd.DataFrame(
            {"row_id": ["1", "2", "3"], "n": pd.Series([1, 2, 3], dtype="int64")}
        ),
        hints={"n": ColumnHint(dtype="int32")},
        source="test",
    )
    assert len(calls) == 1, (
        f"check_frame ran {len(calls)} times for one accept"
    )
    assert calls[0].adjustments, (
        "this accept must produce an adjustment or the test is vacuous"
    )


# ---------------------------------------------------------------------------
# check 6 -- strict mode, positive and negative
# ---------------------------------------------------------------------------

def test_strict_schema_allows_a_storage_policy_adjustment():
    ds = Dataset()
    ds.strict_schema = True
    # With a ColumnHint(dtype="category") the schema stores 'label' as category;
    # str -> category is a storage_policy adjustment, which strict mode allows.
    df = pd.DataFrame(
        {
            "row_id": ["1", "2", "3"],
            "label": pd.Series(["approach", "approach", "withdraw"]),
        }
    )
    ds._accept_table(
        "t", df, source="test",
        hints={"label": ColumnHint(dtype="category")},
    )  # must not raise
    assert ds.schema_for("t").spec_for("label").dtype == "category"
    assert any("label" in m for m in ds.take_schema_messages())


def test_strict_schema_rejects_an_unexpected_adjustment():
    ds = Dataset()
    ds.strict_schema = True
    # A float32 column infers to float64; float32 -> float64 is an 'unexpected'
    # adjustment (the frame declared a width the schema did not expect).
    df = pd.DataFrame(
        {"row_id": ["1", "2"], "v": pd.Series([1.25, 2.5], dtype="float32")}
    )
    with pytest.raises(SchemaRejection):
        ds._accept_table("t", df, source="test")

    # With strict off the same frame stores, recording the adjustment.
    ds.strict_schema = False
    ds._accept_table("t", df, source="test")
    assert ds.schema_for("t").spec_for("v").dtype == "float64"


# ---------------------------------------------------------------------------
# reporting: three destinations for an adjustment
# ---------------------------------------------------------------------------

def test_adjustments_are_reported_to_messages_and_provenance():
    ds = Dataset()
    # A ColumnHint(dtype="int32") over an arrived int64 column: str/int64 ->
    # int32 is a storage_policy adjustment, recorded three ways.
    ds._accept_table(
        "t",
        pd.DataFrame(
            {"row_id": ["1", "2", "3"], "n": pd.Series([1, 2, 3], dtype="int64")}
        ),
        hints={"n": ColumnHint(dtype="int32")},
        source="an-import",
    )

    messages = ds.take_schema_messages()
    assert any("'n'" in m for m in messages)
    # draining clears the list
    assert ds.take_schema_messages() == []

    entries = [
        e for e in ds.provenance.to_list()
        if e["action"] == "schema_adjustments"
    ]
    assert entries, "no schema_adjustments provenance entry"
    flat = [a for e in entries for a in e["params"]["adjustments"]]
    assert any(
        a["column"] == "n" and a["kind"] == "storage_policy" for a in flat
    )
    assert all(
        set(a) == {"column", "arrived_as", "stored_as", "reason", "kind"}
        for a in flat
    )
    assert all("table" in e["params"] and "source" in e["params"] for e in entries)


# ---------------------------------------------------------------------------
# INVESTIGATION B -- a column apply_row_updates creates and then fills across
#                    several ticks
# ---------------------------------------------------------------------------

def test_b1_text_column_filled_across_two_ticks(tmp_path):
    ds = _frames_for_updates(tmp_path)
    rids = list(ds.get_table("frames")["row_id"])

    r1 = ds.apply_row_updates(
        "frames",
        {rids[0]: {"emotion": "joy"},
         rids[1]: {"emotion": "joy"},
         rids[2]: {"emotion": "anger"}},
    )
    r2 = ds.apply_row_updates(
        "frames", {rids[3]: {"emotion": "surprise"}}
    )

    assert r1 == [] and r2 == [], f"unplaceable: call1={r1} call2={r2}"
    stored = list(ds.get_table("frames")["emotion"])
    assert stored == ["joy", "joy", "anger", "surprise"], stored


def test_b2_numeric_column_filled_across_two_ticks(tmp_path):
    # Call 1 fills EVERY row with a whole number, so infer_objects would give
    # int64 without the pin; call 2 replaces one with a decimal. This is the
    # test that fails if the float64 pin is removed.
    csv = tmp_path / "b2.csv"
    csv.write_text("grp\nA\nB\n")
    ds = Dataset()
    ds.load_csv_as_primary(csv)
    rids = list(ds.get_table("frames")["row_id"])

    r1 = ds.apply_row_updates(
        "frames", {rids[0]: {"count": 3}, rids[1]: {"count": 7}}
    )
    r2 = ds.apply_row_updates("frames", {rids[0]: {"count": 4.5}})

    assert r1 == [] and r2 == [], f"unplaceable: call1={r1} call2={r2}"
    assert list(ds.get_table("frames")["count"]) == [4.5, 7.0]


def test_b3_no_path_closes_a_repeated_text_column_to_a_category(tmp_path):
    # NEGATIVE check for CHANGE 1. If any "repeated text -> category" inference
    # were reintroduced, a repeated-label column would become a closed set and
    # a later label would fail at write time. Both the apply_row_updates path
    # (a column still being filled) and the whole-column create_table_from_df
    # path must leave repeated labels OPEN; a category only appears when a
    # ColumnHint asks for it.
    ds = _frames_for_updates(tmp_path)
    rids = list(ds.get_table("frames")["row_id"])

    ds.apply_row_updates(
        "frames",
        {rids[0]: {"label": "a"}, rids[1]: {"label": "a"}, rids[2]: {"label": "b"}},
    )
    assert ds.schema_for("frames").spec_for("label").dtype != "category"

    ds.create_table_from_df("whole", pd.DataFrame({"label": ["a", "a", "b"]}))
    assert ds.schema_for("whole").spec_for("label").dtype != "category"


# ---------------------------------------------------------------------------
# SchemaRejection carries the table, the source and the reason
# ---------------------------------------------------------------------------

def test_schema_rejection_names_table_source_and_reason():
    ds = Dataset()
    ds._accept_table(
        "t",
        pd.DataFrame({"row_id": ["1"], "k": pd.Series([5], dtype="int64")}),
        source="first",
    )
    with pytest.raises(SchemaRejection) as exc:
        ds._accept_table(
            "t",
            pd.DataFrame(
                {"row_id": ["1"], "k": pd.Series([9.9], dtype="float64")}
            ),
            source="second-source",
        )
    msg = str(exc.value)
    assert "t" in msg
    assert "second-source" in msg
    assert "k" in msg


# CHANGE 3: a duplicate column name is one more refused-frame case, raised as
# SchemaRejection (not a bare ValueError) and naming table and source.
def test_duplicate_column_names_raise_schema_rejection(tmp_path):
    ds = Dataset()
    df = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6]], columns=["row_id", "dup", "dup"]
    )
    with pytest.raises(SchemaRejection) as exc:
        ds._accept_table("t", df, source="a-source")
    msg = str(exc.value)
    assert "t" in msg and "a-source" in msg


# The gap CHANGE 1 closes: an operator writes a label into a PRE-EXISTING
# imported repeated-text column that was NOT in the CSV -- it must land, not be
# dropped as unplaceable.
def test_operator_can_add_a_new_label_to_an_imported_repeated_text_column(tmp_path):
    csv = tmp_path / "labels.csv"
    csv.write_text(
        "file_name,condition\n"
        "a.jpg,positive\nb.jpg,positive\nc.jpg,negative\n"
    )
    ds = Dataset()
    ds.load_csv_as_primary(csv)
    assert ds.schema_for("frames").spec_for("condition").dtype != "category"

    rid = ds.get_table("frames")["row_id"].iloc[0]
    result = ds.apply_row_updates("frames", {rid: {"condition": "neutral"}})

    assert result == []
    assert ds.get_table("frames")["condition"].iloc[0] == "neutral"


# ---------------------------------------------------------------------------
# a rejected write leaves the stored table exactly as it was
# ---------------------------------------------------------------------------

def test_a_rejected_write_leaves_the_stored_table_untouched():
    ds = Dataset()
    ds._accept_table(
        "t",
        pd.DataFrame(
            {"row_id": ["1", "2"], "k": pd.Series([10, 20], dtype="int64")}
        ),
        source="setup",
    )
    before = ds.read_only_view("t").copy()

    # Overwriting 'k' (schema says int64) with text crosses a kind -> reject.
    bad = pd.Series({"1": "x", "2": "y"})
    with pytest.raises(SchemaRejection):
        ds.add_column("k", bad, "text", table_name="t")

    pd.testing.assert_frame_equal(ds.read_only_view("t"), before)


def test_accept_table_raises_schema_rejection_on_an_unsupported_dtype():
    ds = Dataset()
    df = pd.DataFrame(
        {
            "row_id": ["1", "2"],
            "when": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    with pytest.raises(SchemaRejection):
        ds._accept_table("t", df, source="test")


# ---------------------------------------------------------------------------
# FIX 2 -- apply_row_updates must not raise a SchemaRejection into the Qt
#          timer drain; it degrades to "none of this batch was placed" unless
#          strict_schema is on.
# ---------------------------------------------------------------------------

def _frames_for_updates(tmp_path):
    csv = tmp_path / "u.csv"
    csv.write_text("grp,score\nA,1.5\nB,2.5\nC,3.5\nD,4.5\n")
    ds = Dataset()
    ds.load_csv_as_primary(csv)
    ds.take_schema_messages()  # drain import-time adjustments
    return ds


def test_apply_row_updates_degrades_a_schema_rejection_when_strict_off(tmp_path):
    ds = _frames_for_updates(tmp_path)
    rid = ds.get_table("frames")["row_id"].iloc[0]
    before = ds.read_only_view("frames").copy()

    # An operator emitting a Timestamp: the new column infers to datetime64,
    # which infer_schema refuses -> _accept_table raises SchemaRejection.
    result = ds.apply_row_updates(
        "frames", {rid: {"emitted_at": pd.Timestamp("2024-01-01")}}
    )

    assert result == [rid], "every row_id in the batch must come back unplaceable"
    pd.testing.assert_frame_equal(ds.read_only_view("frames"), before)
    assert any("emitted_at" in m or "datetime" in m
               for m in ds.take_schema_messages())


def test_apply_row_updates_reraises_a_schema_rejection_when_strict_on(tmp_path):
    ds = _frames_for_updates(tmp_path)
    ds.strict_schema = True
    rid = ds.get_table("frames")["row_id"].iloc[0]

    # NEGATIVE CHECK: if the `if self.strict_schema: raise` line in
    # apply_row_updates were deleted, strict mode would silently degrade to
    # "all unplaceable" and return normally -- an operator's own test would
    # then never see the wrong dtype. This pytest.raises is what catches that
    # deletion.
    with pytest.raises(SchemaRejection):
        ds.apply_row_updates(
            "frames", {rid: {"emitted_at": pd.Timestamp("2024-01-01")}}
        )

    # and the stored frame is still intact even though it raised
    assert "emitted_at" not in ds.read_only_view("frames").columns


def test_apply_row_updates_degrades_a_wrong_type_for_an_existing_column(tmp_path):
    # FIX A: a wrong-typed value for an EXISTING column makes pandas' own iat
    # setter raise TypeError, from inside the write loop -- which is now inside
    # the try. Strict off: whole batch unplaceable, stored frame unchanged.
    ds = _frames_for_updates(tmp_path)
    rid = ds.get_table("frames")["row_id"].iloc[0]
    before = ds.read_only_view("frames").copy()

    result = ds.apply_row_updates("frames", {rid: {"score": "not a number"}})

    assert result == [rid]
    pd.testing.assert_frame_equal(ds.read_only_view("frames"), before)

    ds.strict_schema = True
    with pytest.raises((TypeError, ValueError)):
        ds.apply_row_updates("frames", {rid: {"score": "not a number"}})
    pd.testing.assert_frame_equal(ds.read_only_view("frames"), before)


# ---------------------------------------------------------------------------
# FIX 3 -- a successful apply_row_updates keeps the SAME DataFrame object, so
#          the row-id index is not rebuilt every timer tick.
# ---------------------------------------------------------------------------

def test_apply_row_updates_keeps_the_same_frame_object(tmp_path):
    # Exists to protect the row-id index from per-tick rebuilds: a future
    # change that reintroduces a whole-frame copy on this path fails here.
    ds = _frames_for_updates(tmp_path)
    rid = ds.get_table("frames")["row_id"].iloc[0]
    before = ds.read_only_view("frames")

    ds.apply_row_updates("frames", {rid: {"note_count": 4}})

    assert ds.read_only_view("frames") is before
    assert "note_count" in ds.read_only_view("frames").columns


def test_apply_row_updates_does_not_rebuild_the_row_id_index(monkeypatch, tmp_path):
    import models.dataset as dsmod

    rebuilds: list[str] = []
    real = dsmod.Dataset._row_index_for

    def recording(self, table_name):
        before_id = id(self._row_index.get(table_name))
        result = real(self, table_name)
        if id(self._row_index[table_name]) != before_id:
            rebuilds.append(table_name)
        return result

    monkeypatch.setattr(dsmod.Dataset, "_row_index_for", recording)

    ds = _frames_for_updates(tmp_path)
    rid = ds.get_table("frames")["row_id"].iloc[0]
    ds.get_row(rid)          # prime the index (one rebuild here)
    rebuilds.clear()

    ds.apply_row_updates("frames", {rid: {"note_count": 4}})
    ds.get_row(rid)          # would rebuild if the frame object had changed

    assert rebuilds == [], f"row-id index rebuilt after a plain update: {rebuilds}"


# ---------------------------------------------------------------------------
# a dropped column is not a rejection
# ---------------------------------------------------------------------------

def test_a_column_the_frame_no_longer_has_is_dropped_not_rejected():
    ds = Dataset()
    ds._accept_table(
        "t",
        pd.DataFrame({"row_id": ["1", "2"], "a": [1, 2], "b": [3, 4]}),
        source="first",
    )
    assert ds.schema_for("t").column_names() == ("a", "b")
    # Second accept drops 'b' -- allowed, and the schema follows the frame.
    ds._accept_table(
        "t",
        pd.DataFrame({"row_id": ["1", "2"], "a": [5, 6]}),
        source="second",
    )
    assert ds.schema_for("t").column_names() == ("a",)
