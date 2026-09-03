"""
tests/test_project_load.py

P1.8c-2b: Dataset.load() restores the schemas save() wrote, is atomic (a bad
project leaves the open one untouched), and clears the column registry's
name -> type map so a second project does not inherit the first's column tags.

Real Dataset + pandas, no Qt. strict_schema is on for the whole suite
(tests/conftest.py).
"""

import json

import pandas as pd
import pytest

from models.dataset import Dataset, SchemaRejection
from models.table_schema import ColumnHint, ColumnRole


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ds_with_frames(tmp_path) -> Dataset:
    """A Dataset whose 'frames' table has a repeated-string column ('subject')
    and a decimal float column ('score'), built through the real CSV path.
    Mirrors the helper of the same name in tests/test_dataset_schema.py."""
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


def _last_load_entry(ds: Dataset) -> dict:
    entries = [e for e in ds.provenance.to_list() if e["action"] == "load"]
    assert entries, "no 'load' provenance entry"
    return entries[-1]


# ---------------------------------------------------------------------------
# schema restoration
# ---------------------------------------------------------------------------

def test_load_restores_a_pinned_schema_spec_for_spec(tmp_path):
    # A schema carrying a deliberately non-default dtype (int32, narrower than
    # inference would ever pick) and a non-default ColumnRole (identifier).
    ds = Dataset()
    df = pd.DataFrame(
        {
            "row_id": ["1", "2", "3"],
            "pid": pd.Series([10, 20, 30], dtype="int64"),
            "score": pd.Series([1.5, 2.0, 3.5], dtype="float64"),
        }
    )
    ds._accept_table(
        "t", df, source="setup",
        hints={
            "pid": ColumnHint(
                dtype="int32", role=ColumnRole.identifier, carry_to_children=False
            )
        },
    )
    ds.take_schema_messages()  # drop the int64 -> int32 storage_policy note
    saved = ds.schema_for("t")
    assert saved.spec_for("pid").dtype == "int32"
    assert saved.spec_for("pid").role is ColumnRole.identifier
    assert saved.spec_for("pid").carry_to_children is False

    proj = tmp_path / "proj"
    ds.save(proj)

    ds2 = Dataset()
    ds2.load(proj)

    restored = ds2.schema_for("t")
    assert restored == saved
    for name in saved.column_names():
        assert restored.spec_for(name) == saved.spec_for(name), name


def test_save_load_round_trip_makes_no_schema_adjustment_under_strict(tmp_path):
    # strict_schema is on for the whole suite. save() writes each table's exact
    # declared schema; load() checks the parquet against THAT restored schema,
    # not against a fresh inference. Parquet stores every supported dtype
    # exactly and the restored schema declares those same dtypes, so check_frame
    # finds an exact match and records nothing. An adjustment here would mean
    # the saved schema and the reloaded frame genuinely disagree -- a real
    # round-trip bug -- so asserting zero adjustments is meaningful, not
    # paranoid. (A pure object <-> str dtype-name change from a parquet round
    # trip is not an adjustment: the three pandas text names are one storage
    # kind, per P1.8c-1.)
    ds = _ds_with_frames(tmp_path)
    ds.aggregate("agg", "frames", "subject", {"score": "mean"})
    ds.create_table_from_df(
        "built", pd.DataFrame({"z": pd.Series([1, 2, 3], dtype="int64")})
    )
    ds.take_schema_messages()

    proj = tmp_path / "proj"
    ds.save(proj)

    ds2 = Dataset()
    ds2.load(proj)

    assert ds2.take_schema_messages() == []
    adjustments = [
        e for e in ds2.provenance.to_list() if e["action"] == "schema_adjustments"
    ]
    assert adjustments == [], adjustments


# ---------------------------------------------------------------------------
# schemas.json: the three decided fallback cases
# ---------------------------------------------------------------------------

def test_a_project_with_no_schemas_json_loads_by_inference(tmp_path):
    ds = _ds_with_frames(tmp_path)
    proj = tmp_path / "proj"
    ds.save(proj)
    (proj / "schemas.json").unlink()

    ds2 = Dataset()
    ds2.load(proj)

    schema = ds2.schema_for("frames")
    assert schema is not None
    # Inference makes every column a measurement.
    assert all(s.role is ColumnRole.measurement for s in schema.columns)
    # No message: a project saved before P1.8c-2a is the common case.
    assert ds2.take_schema_messages() == []
    assert "no schemas.json" in _last_load_entry(ds2)["params"]["schemas"]


def test_unknown_format_version_loads_by_inference_and_says_so(tmp_path):
    ds = Dataset()
    ds._accept_table(
        "t",
        pd.DataFrame({"row_id": ["1", "2"], "k": pd.Series([1, 2], dtype="int64")}),
        source="setup",
        hints={"k": ColumnHint(role=ColumnRole.identifier)},
    )
    assert ds.schema_for("t").spec_for("k").role is ColumnRole.identifier

    proj = tmp_path / "proj"
    ds.save(proj)
    sidecar = json.loads((proj / "schemas.json").read_text())
    sidecar["format_version"] = 999
    (proj / "schemas.json").write_text(json.dumps(sidecar))

    ds2 = Dataset()
    ds2.load(proj)

    # The saved identifier role is gone -- the schema was re-inferred.
    assert ds2.schema_for("t").spec_for("k").role is ColumnRole.measurement
    assert any("format_version" in m for m in ds2.take_schema_messages())
    assert "format_version" in _last_load_entry(ds2)["params"]["schemas"]


def test_corrupt_schemas_json_loads_by_inference_and_says_so(tmp_path):
    ds = Dataset()
    ds._accept_table(
        "t",
        pd.DataFrame({"row_id": ["1", "2"], "k": pd.Series([1, 2], dtype="int64")}),
        source="setup",
        hints={"k": ColumnHint(role=ColumnRole.identifier)},
    )
    proj = tmp_path / "proj"
    ds.save(proj)
    # Not valid JSON at all.
    (proj / "schemas.json").write_text("{ this is not valid json ")

    ds2 = Dataset()
    ds2.load(proj)

    assert ds2.schema_for("t").spec_for("k").role is ColumnRole.measurement
    assert any("schemas.json" in m for m in ds2.take_schema_messages())
    note = _last_load_entry(ds2)["params"]["schemas"]
    assert "could not be read" in note or "malformed" in note


def test_schemas_json_with_a_rejected_schema_entry_loads_by_inference(tmp_path):
    # Valid JSON, format_version 1, but one column entry has a role name that
    # matches no ColumnRole -- schema_from_dict raises SchemaSerialisationError,
    # which _read_saved_schemas turns into the same "ignore, infer, say so"
    # outcome as an unknown version.
    ds = Dataset()
    ds._accept_table(
        "t",
        pd.DataFrame({"row_id": ["1", "2"], "k": pd.Series([1, 2], dtype="int64")}),
        source="setup",
    )
    proj = tmp_path / "proj"
    ds.save(proj)
    sidecar = json.loads((proj / "schemas.json").read_text())
    sidecar["schemas"]["t"]["columns"][0]["role"] = "not_a_real_role"
    (proj / "schemas.json").write_text(json.dumps(sidecar))

    ds2 = Dataset()
    ds2.load(proj)

    assert ds2.schema_for("t") is not None
    assert any("malformed" in m for m in ds2.take_schema_messages())


def test_schemas_json_with_an_unhashable_role_value_loads_by_inference(tmp_path):
    # A hand edit that puts a list where a role name belongs makes
    # schema_from_dict raise TypeError (not SchemaSerialisationError) from its
    # membership test. _read_saved_schemas must still fall back to inference --
    # a corrupt sidecar never makes a project unopenable.
    ds = Dataset()
    ds._accept_table(
        "t",
        pd.DataFrame({"row_id": ["1", "2"], "k": pd.Series([1, 2], dtype="int64")}),
        source="setup",
    )
    proj = tmp_path / "proj"
    ds.save(proj)
    sidecar = json.loads((proj / "schemas.json").read_text())
    sidecar["schemas"]["t"]["columns"][0]["role"] = []
    (proj / "schemas.json").write_text(json.dumps(sidecar))

    ds2 = Dataset()
    ds2.load(proj)  # must not raise

    assert ds2.schema_for("t") is not None
    assert any("malformed" in m for m in ds2.take_schema_messages())


# ---------------------------------------------------------------------------
# atomicity
# ---------------------------------------------------------------------------

def test_a_failing_load_leaves_the_previous_project_fully_intact(tmp_path):
    # Project A: a real saved project with two tables.
    src = _ds_with_frames(tmp_path)
    src.aggregate("by_subject", "frames", "subject", {"score": "mean"})
    proj_a = tmp_path / "A"
    src.save(proj_a)

    ds = Dataset()
    ds.load(proj_a)
    tables_before = ds.list_tables()
    frames_before = ds.get_table("frames")
    agg_before = ds.get_table("by_subject")
    frames_schema_before = ds.schema_for("frames")
    provenance_before = ds.provenance.to_list()
    id_counter_before = ds._id_counter

    # Project B is made to fail preparation: frames.parquet holds a datetime64
    # column, which infer_schema refuses; _prepare_table turns that into a
    # SchemaRejection, raised inside load()'s pre-point-of-no-return loop.
    proj_b = tmp_path / "B"
    proj_b.mkdir()
    pd.DataFrame(
        {
            "row_id": ["1", "2"],
            "when": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    ).to_parquet(proj_b / "frames.parquet")

    with pytest.raises(SchemaRejection):
        ds.load(proj_b)

    # Every observable piece of project A is exactly as it was.
    assert ds.list_tables() == tables_before
    pd.testing.assert_frame_equal(ds.get_table("frames"), frames_before)
    pd.testing.assert_frame_equal(ds.get_table("by_subject"), agg_before)
    assert ds.schema_for("frames") == frames_schema_before
    assert ds.provenance.to_list() == provenance_before
    assert ds._id_counter == id_counter_before


# ---------------------------------------------------------------------------
# no prior in-memory state is consulted
# ---------------------------------------------------------------------------

def test_a_table_not_named_by_saved_schemas_ignores_the_prior_projects_spec(tmp_path):
    # Project A pins column 'k' of table 'shared' to int32 + identifier.
    a = Dataset()
    a._accept_table(
        "shared",
        pd.DataFrame({"row_id": ["1", "2"], "k": pd.Series([1, 2], dtype="int32")}),
        source="A",
        hints={"k": ColumnHint(dtype="int32", role=ColumnRole.identifier)},
    )
    a.take_schema_messages()
    proj_a = tmp_path / "A"
    a.save(proj_a)

    # Project B also has a 'shared' table, 'k' holding fractional floats, and
    # NO schemas.json -- so its saved schemas name no schema for 'shared'.
    b = Dataset()
    b._accept_table(
        "shared",
        pd.DataFrame(
            {"row_id": ["1", "2"], "k": pd.Series([1.5, 2.5], dtype="float64")}
        ),
        source="B",
    )
    proj_b = tmp_path / "B"
    b.save(proj_b)
    (proj_b / "schemas.json").unlink()

    # One Dataset opens A, then B. B's 'shared' must be inferred fresh, NOT
    # resolved against A's still-in-memory int32/identifier spec -- which would
    # additionally reject the 1.5 as not surviving int32.
    ds = Dataset()
    ds.load(proj_a)
    assert ds.schema_for("shared").spec_for("k").dtype == "int32"

    ds.load(proj_b)  # must not raise

    spec = ds.schema_for("shared").spec_for("k")
    assert spec.dtype == "float64"
    assert spec.role is ColumnRole.measurement
    assert list(ds.get_table("shared")["k"]) == [1.5, 2.5]


# ---------------------------------------------------------------------------
# registry: a second project does not inherit the first project's column tags
# ---------------------------------------------------------------------------

def test_loading_a_second_project_does_not_keep_the_first_projects_column_tags(tmp_path):
    from artifacts.artifact_store import ArtifactStore
    from column_types.registry import ColumnTypeRegistry

    store = ArtifactStore(tmp_path / "art")

    # Project A: a CSV whose columns include 'mood'.
    csv_a = tmp_path / "a.csv"
    csv_a.write_text("mood,val\nhappy,1\nsad,2\n")
    reg_a = ColumnTypeRegistry()
    reg_a.setup_defaults(store)
    ds_a = Dataset()
    ds_a.set_registry(reg_a)
    ds_a.load_csv_as_primary(csv_a)
    proj_a = tmp_path / "A"
    ds_a.save(proj_a)

    # Project B: a different CSV, no 'mood' column.
    csv_b = tmp_path / "b.csv"
    csv_b.write_text("temperature,val\n20,1\n21,2\n")
    reg_b = ColumnTypeRegistry()
    reg_b.setup_defaults(store)
    ds_b = Dataset()
    ds_b.set_registry(reg_b)
    ds_b.load_csv_as_primary(csv_b)
    proj_b = tmp_path / "B"
    ds_b.save(proj_b)

    # One dataset + registry opens A, then B.
    reg = ColumnTypeRegistry()
    reg.setup_defaults(store)
    ds = Dataset()
    ds.set_registry(reg)

    ds.load(proj_a)
    assert reg.get("mood") is not None, "sanity: project A registered 'mood'"

    ds.load(proj_b)
    assert reg.get("mood") is None, "project B must not inherit A's 'mood' tag"
    assert reg.get("temperature") is not None
    # The built-in types survive the clear -- 'temperature' could be
    # re-registered against them.
    assert reg.get("val") is not None


def test_a_second_project_does_not_inherit_a_media_tag_via_inference(tmp_path):
    # P1.8c-2b follow-up (Fix 4). load()'s _prepare_table must not consult the
    # registry, which during a load still holds the OUTGOING project's tags.
    # Project A tags column 'clip' as media_path; project B has its own 'clip'
    # column of plain text and no schemas.json entry for it, so B's 'clip' spec
    # is inferred. Its type_tag must come from inference ('text'), not from A's
    # lingering registry tag.
    from artifacts.artifact_store import ArtifactStore
    from column_types.registry import ColumnTypeRegistry

    store = ArtifactStore(tmp_path / "art")

    csv_a = tmp_path / "a.csv"
    csv_a.write_text("clip,val\na.mp4,1\nb.mp4,2\n")
    reg_a = ColumnTypeRegistry()
    reg_a.setup_defaults(store)
    ds_a = Dataset()
    ds_a.set_registry(reg_a)
    ds_a.load_csv_as_primary(csv_a)
    reg_a.register_by_tag("clip", "media_path")  # force the media tag
    proj_a = tmp_path / "A"
    ds_a.save(proj_a)

    csv_b = tmp_path / "b.csv"
    csv_b.write_text("clip,val\nhello,1\nworld,2\n")
    reg_b = ColumnTypeRegistry()
    reg_b.setup_defaults(store)
    ds_b = Dataset()
    ds_b.set_registry(reg_b)
    ds_b.load_csv_as_primary(csv_b)
    proj_b = tmp_path / "B"
    ds_b.save(proj_b)
    (proj_b / "schemas.json").unlink()  # so 'clip' is inferred, not restored

    reg = ColumnTypeRegistry()
    reg.setup_defaults(store)
    ds = Dataset()
    ds.set_registry(reg)

    ds.load(proj_a)
    # After loading A the registry maps 'clip' -> media_path. That is exactly
    # the tag that must NOT leak into B's freshly inferred 'clip' spec.
    assert reg.get("clip") is not None and reg.get("clip").tag == "media_path"

    ds.load(proj_b)
    spec = ds.schema_for("frames").spec_for("clip")
    assert spec.type_tag == "text", (
        f"B's inferred 'clip' spec inherited A's media tag: {spec.type_tag}"
    )


# ---------------------------------------------------------------------------
# an empty table's saved schema is restored, not re-inferred to float64
# ---------------------------------------------------------------------------

def test_load_restores_the_text_dtype_of_an_empty_table(tmp_path):
    # P1.8c-2b follow-up (Fix 1). pyarrow returns an empty column as float64, so
    # a re-inferred empty text column would come back typed as a number and the
    # first text value written into it later would be rejected as crossing a
    # kind. load() must cast the empty frame to the saved schema's dtypes before
    # validating.
    ds = Dataset()
    ds._accept_table(
        "notes",
        pd.DataFrame(
            {
                "row_id": pd.Series([], dtype="object"),
                "body": pd.Series([], dtype="object"),
            }
        ),
        source="setup",
    )
    text_dtype = ds.schema_for("notes").spec_for("body").dtype
    assert text_dtype in ("object", "string", "str")

    proj = tmp_path / "proj"
    ds.save(proj)

    ds2 = Dataset()
    ds2.load(proj)

    # The restored schema still names the text dtype -- not float64.
    assert ds2.schema_for("notes").spec_for("body").dtype == text_dtype

    # And a text value can now be written into that column: an operator handing
    # 'notes' a one-row frame is validated against the restored schema.
    ds2._accept_table(
        "notes",
        pd.DataFrame(
            {"row_id": ["1"], "body": pd.Series(["hello"], dtype="object")}
        ),
        source="operator",
    )  # must not raise
    assert ds2.get_table("notes")["body"].iloc[0] == "hello"


def test_load_falls_back_for_one_empty_table_whose_schema_cannot_be_cast(
    tmp_path, monkeypatch
):
    # If the empty-frame cast raises, ONLY that table falls back to inference,
    # with a message, and the rest of the project still loads. Every dtype that
    # reaches restored_schemas has already passed schema_from_dict's dtype gate
    # and an empty column casts losslessly, so a genuine failure needs the cast
    # helper stubbed to raise.
    import models.dataset as dsmod

    ds = Dataset()
    ds._accept_table(
        "notes",
        pd.DataFrame(
            {"row_id": pd.Series([], dtype="object"),
             "body": pd.Series([], dtype="object")}
        ),
        source="setup",
    )
    ds._accept_table(
        "rows",
        pd.DataFrame({"row_id": ["1"], "n": pd.Series([5], dtype="int64")}),
        source="setup",
    )
    proj = tmp_path / "proj"
    ds.save(proj)

    def boom(df, schema):
        raise ValueError("stubbed cast failure")

    monkeypatch.setattr(dsmod, "_cast_empty_frame_to_schema", boom)

    ds2 = Dataset()
    ds2.load(proj)  # must not raise

    # 'notes' (empty) fell back to inference and said so.
    assert ds2.schema_for("notes") is not None
    assert any(
        "notes" in m and "loaded by inference" in m
        for m in ds2.take_schema_messages()
    )
    # 'rows' (non-empty) was untouched by the fallback and kept its schema.
    assert ds2.schema_for("rows").spec_for("n").dtype == "int64"


# ---------------------------------------------------------------------------
# controller: a failing Dataset.load leaves the controller root unchanged
# ---------------------------------------------------------------------------

def test_a_failing_load_project_leaves_the_controller_root_unchanged(tmp_path, make_controller):
    # P1.8c-2b follow-up (Fix 3). AppController.load_project must not mutate
    # _project_root (or drop live runs) until Dataset.load() has returned.
    controller, dataset, _ = make_controller(tmp_path)
    root_before = controller._project_root
    tables_before = dataset.list_tables()

    # A project folder whose frames.parquet cannot be prepared (datetime col).
    bad = tmp_path / "bad"
    bad.mkdir()
    pd.DataFrame(
        {"row_id": ["1"], "when": pd.to_datetime(["2024-01-01"])}
    ).to_parquet(bad / "frames.parquet")

    controller.load_project(bad)  # emits error_occurred, does not raise

    assert controller._project_root == root_before
    assert dataset.list_tables() == tables_before
