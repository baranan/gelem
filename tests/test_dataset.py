"""
tests/test_dataset.py

Tests for Student B's implementations of Dataset and QueryEngine.

Run with:
    python tests/test_dataset.py

Each test prints PASS or FAIL with a description.
All tests are independent — a failing test does not stop the others.

Setup:
    Make sure you have run create_test_csv.py first to generate
    test_images/metadata.csv before running tests that need CSV data.

    The test_images folder may contain images (.jpg, .png) and/or
    videos (.mp4, .mov) — load_folder now handles both.
"""

import sys
from pathlib import Path

# Add project root to Python path.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import tempfile


# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0

def passed(name: str) -> None:
    global _passed
    _passed += 1
    print(f"  PASS  {name}")

def failed(name: str, reason: str) -> None:
    global _failed
    _failed += 1
    print(f"  FAIL  {name}")
    print(f"        {reason}")

def section(title: str) -> None:
    print()
    print(f"── {title} {'─' * (60 - len(title))}")

def run_test(name: str, fn) -> None:
    try:
        fn()
        passed(name)
    except AssertionError as e:
        failed(name, str(e))
    except Exception as e:
        failed(name, f"Unexpected error: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_IMAGES  = project_root / "test_images"
METADATA_CSV = TEST_IMAGES / "metadata.csv"
TEMP_DIR     = Path(tempfile.gettempdir())


# ---------------------------------------------------------------------------
# Section 1: Dataset.load_folder()
# ---------------------------------------------------------------------------

section("1. Dataset.load_folder()")

def test_load_folder_creates_rows():
    from models.dataset import Dataset
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    df = ds.get_table("frames")
    assert len(df) > 0, "No rows created — check folder path and media extensions"

def test_load_folder_required_columns():
    from models.dataset import Dataset
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    df = ds.get_table("frames")
    for col in ["row_id", "full_path", "file_name"]:
        assert col in df.columns, f"Missing required column: {col}"

def test_load_folder_row_ids_are_padded_strings():
    from models.dataset import Dataset
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    df    = ds.get_table("frames")
    first = df["row_id"].iloc[0]
    assert isinstance(first, str), f"row_id should be a string, got {type(first)}"
    assert first == "000001", f"First row_id should be '000001', got '{first}'"

def test_load_folder_row_ids_are_unique():
    from models.dataset import Dataset
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    df = ds.get_table("frames")
    assert df["row_id"].nunique() == len(df), "row_ids are not unique"

def test_load_folder_full_paths_exist():
    from models.dataset import Dataset
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    df = ds.get_table("frames")
    for path in df["full_path"]:
        assert Path(path).exists(), f"full_path does not exist: {path}"

def test_load_folder_resets_on_second_call():
    from models.dataset import Dataset
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    count1 = len(ds.get_table("frames"))
    ds.load_folder(TEST_IMAGES)
    count2 = len(ds.get_table("frames"))
    assert count1 == count2, (
        f"Second load should replace, not append. "
        f"First: {count1}, second: {count2}"
    )
    first_id = ds.get_table("frames")["row_id"].iloc[0]
    assert first_id == "000001", (
        f"row_ids should reset on second load, got '{first_id}'"
    )

def test_load_folder_registers_media_path_type():
    """
    load_folder should register full_path as 'media_path' (not 'image_path').
    This ensures images and videos are both handled by the media renderer.
    """
    from models.dataset import Dataset
    from column_types.registry import ColumnTypeRegistry
    from artifacts.artifact_store import ArtifactStore

    store    = ArtifactStore(TEMP_DIR / "gelem_test_artifacts")
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)

    ds = Dataset()
    ds.set_registry(registry)
    ds.load_folder(TEST_IMAGES)

    col_type = registry.get("full_path")
    assert col_type is not None, "full_path not registered with registry"
    assert col_type.tag == "media_path", (
        f"full_path should be registered as 'media_path', got '{col_type.tag}'"
    )

def test_load_folder_only_loads_supported_extensions():
    """
    load_folder should only create rows for files whose extension is
    in MEDIA_EXTENSIONS. Files like .csv, .txt, .json should be skipped.
    """
    from models.dataset import Dataset, MEDIA_EXTENSIONS
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Create a fake image file and a non-media file.
        (tmp / "face.jpg").write_bytes(b"fake jpg")
        (tmp / "notes.txt").write_text("some notes")
        (tmp / "data.csv").write_text("a,b\n1,2")

        ds = Dataset()
        ds.load_folder(tmp)
        df = ds.get_table("frames")

        file_names = list(df["file_name"])
        assert "face.jpg" in file_names, "face.jpg should be loaded"
        assert "notes.txt" not in file_names, "notes.txt should not be loaded"
        assert "data.csv" not in file_names, "data.csv should not be loaded"

run_test("Creates rows for media files", test_load_folder_creates_rows)
run_test("Required columns present", test_load_folder_required_columns)
run_test("row_ids are zero-padded strings", test_load_folder_row_ids_are_padded_strings)
run_test("row_ids are unique", test_load_folder_row_ids_are_unique)
run_test("full_path values point to real files", test_load_folder_full_paths_exist)
run_test("Second call resets the table", test_load_folder_resets_on_second_call)
run_test("Registers full_path as media_path type", test_load_folder_registers_media_path_type)
run_test("Only loads supported media extensions", test_load_folder_only_loads_supported_extensions)


# ---------------------------------------------------------------------------
# Section 2: Dataset.merge_csv()
# ---------------------------------------------------------------------------

section("2. Dataset.merge_csv() and confirm_merge()")

if not METADATA_CSV.exists():
    print("  SKIP  All CSV tests — metadata.csv not found.")
    print("        Run create_test_csv.py first.")
else:
    def test_merge_returns_report():
        from models.dataset import Dataset, MergeReport
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        report = ds.merge_csv(METADATA_CSV, join_on="file_name")
        assert isinstance(report, MergeReport), (
            f"merge_csv should return a MergeReport, got {type(report)}"
        )

    def test_merge_report_counts():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        report = ds.merge_csv(METADATA_CSV, join_on="file_name")
        assert report.total_image_files == len(ds.get_table("frames")), (
            "total_image_files should match number of loaded media files"
        )
        assert report.total_csv_rows > 0, "total_csv_rows should be > 0"
        assert report.matched_rows > 0, (
            "matched_rows should be > 0 — check join_on column name"
        )

    def test_merge_does_not_commit_before_confirm():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        cols_before = list(ds.get_table("frames").columns)
        ds.merge_csv(METADATA_CSV, join_on="file_name")
        cols_after = list(ds.get_table("frames").columns)
        assert cols_before == cols_after, (
            "Table should not change until confirm_merge() is called"
        )

    def test_confirm_merge_adds_columns():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        report = ds.merge_csv(METADATA_CSV, join_on="file_name")
        ds.confirm_merge(report)
        df = ds.get_table("frames")
        assert "condition" in df.columns,  "condition column missing after merge"
        assert "session_id" in df.columns, "session_id column missing after merge"
        assert "trial_id" in df.columns,   "trial_id column missing after merge"

    def test_confirm_merge_registers_text_type():
        """
        After confirm_merge, CSV string columns should be registered
        as 'text' (not 'categorical').
        """
        from models.dataset import Dataset
        from column_types.registry import ColumnTypeRegistry
        from artifacts.artifact_store import ArtifactStore

        store    = ArtifactStore(TEMP_DIR / "gelem_test_artifacts")
        registry = ColumnTypeRegistry()
        registry.setup_defaults(store)

        ds = Dataset()
        ds.set_registry(registry)
        ds.load_folder(TEST_IMAGES)
        report = ds.merge_csv(METADATA_CSV, join_on="file_name")
        ds.confirm_merge(report)

        col_type = registry.get("condition")
        assert col_type is not None, "condition not registered"
        assert col_type.tag == "text", (
            f"condition should be registered as 'text', got '{col_type.tag}'"
        )

    def test_confirm_merge_row_count_unchanged():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        count_before = len(ds.get_table("frames"))
        report = ds.merge_csv(METADATA_CSV, join_on="file_name")
        ds.confirm_merge(report)
        count_after = len(ds.get_table("frames"))
        assert count_before == count_after, (
            f"Left join should not change row count. "
            f"Before: {count_before}, after: {count_after}"
        )

    run_test("merge_csv returns MergeReport", test_merge_returns_report)
    run_test("MergeReport counts are correct", test_merge_report_counts)
    run_test("Table unchanged before confirm", test_merge_does_not_commit_before_confirm)
    run_test("confirm_merge adds CSV columns", test_confirm_merge_adds_columns)
    run_test("confirm_merge registers text type", test_confirm_merge_registers_text_type)
    run_test("Row count unchanged after merge", test_confirm_merge_row_count_unchanged)


# ---------------------------------------------------------------------------
# Section 3: Dataset.update_row()
# ---------------------------------------------------------------------------

section("3. Dataset.update_row()")

def test_update_row_existing_column():
    from models.dataset import Dataset
    ds     = Dataset()
    ds.load_folder(TEST_IMAGES)
    df     = ds.get_table("frames")
    row_id = df["row_id"].iloc[0]
    ds.update_row(row_id, {"file_name": "updated.jpg"})
    updated = ds.get_row(row_id)
    assert updated["file_name"] == "updated.jpg", (
        f"Expected 'updated.jpg', got '{updated['file_name']}'"
    )

def test_update_row_new_column():
    from models.dataset import Dataset
    ds     = Dataset()
    ds.load_folder(TEST_IMAGES)
    df     = ds.get_table("frames")
    row_id = df["row_id"].iloc[0]
    ds.update_row(row_id, {"bs_jawOpen": 0.42})
    updated = ds.get_row(row_id)
    assert "bs_jawOpen" in updated, "New column not created"
    assert abs(updated["bs_jawOpen"] - 0.42) < 0.001, (
        f"Expected 0.42, got {updated['bs_jawOpen']}"
    )

def test_update_row_other_rows_get_none():
    from models.dataset import Dataset
    ds        = Dataset()
    ds.load_folder(TEST_IMAGES)
    df        = ds.get_table("frames")
    first_id  = df["row_id"].iloc[0]
    second_id = df["row_id"].iloc[1]
    ds.update_row(first_id, {"new_col": 99.0})
    second_row = ds.get_row(second_id)
    val = second_row.get("new_col")
    assert val is None or pd.isna(val), (
        f"Other rows should be None/NaN for new column, got {val}"
    )

run_test("Update existing column", test_update_row_existing_column)
run_test("Create new column via update", test_update_row_new_column)
run_test("Other rows get None for new column", test_update_row_other_rows_get_none)


# ---------------------------------------------------------------------------
# Section 4: Dataset.add_computed_column()
# ---------------------------------------------------------------------------

section("4. Dataset.add_computed_column()")

if not METADATA_CSV.exists():
    print("  SKIP  All computed column tests — metadata.csv not found.")
else:
    def test_add_computed_column_exists():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        report = ds.merge_csv(METADATA_CSV, join_on="file_name")
        ds.confirm_merge(report)
        ds.add_computed_column(
            "timestamp_doubled", "timestamp * 2", col_type="numeric"
        )
        df = ds.get_table("frames")
        assert "timestamp_doubled" in df.columns, "Computed column not added"

    def test_add_computed_column_correct_values():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        report = ds.merge_csv(METADATA_CSV, join_on="file_name")
        ds.confirm_merge(report)
        ds.add_computed_column(
            "timestamp_doubled", "timestamp * 2", col_type="numeric"
        )
        df = ds.get_table("frames")
        for _, row in df.iterrows():
            expected = row["timestamp"] * 2
            actual   = row["timestamp_doubled"]
            assert abs(expected - actual) < 0.001, (
                f"Expected {expected}, got {actual}"
            )

    run_test("Computed column added", test_add_computed_column_exists)
    run_test("Computed column values correct", test_add_computed_column_correct_values)


# ---------------------------------------------------------------------------
# Section 5: Dataset.aggregate()
# ---------------------------------------------------------------------------

section("5. Dataset.aggregate()")

if not METADATA_CSV.exists():
    print("  SKIP  All aggregation tests — metadata.csv not found.")
else:
    def test_aggregate_creates_table():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        report = ds.merge_csv(METADATA_CSV, join_on="file_name")
        ds.confirm_merge(report)
        ds.aggregate(
            name="by_condition",
            source_table="frames",
            group_by="condition",
            aggregations={"timestamp": "mean"},
        )
        assert "by_condition" in ds.list_tables(), (
            "Aggregated table not in list_tables()"
        )

    def test_aggregate_correct_group_count():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        report = ds.merge_csv(METADATA_CSV, join_on="file_name")
        ds.confirm_merge(report)
        ds.aggregate(
            name="by_condition",
            source_table="frames",
            group_by="condition",
            aggregations={"timestamp": "mean"},
        )
        agg = ds.get_table("by_condition")
        assert len(agg) == 3, f"Expected 3 condition groups, got {len(agg)}"

    def test_aggregate_has_row_id():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        report = ds.merge_csv(METADATA_CSV, join_on="file_name")
        ds.confirm_merge(report)
        ds.aggregate(
            name="by_condition",
            source_table="frames",
            group_by="condition",
            aggregations={"timestamp": "mean"},
        )
        agg = ds.get_table("by_condition")
        assert "row_id" in agg.columns, "Aggregated table missing row_id"

    def test_aggregate_row_ids_dont_overlap():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        report = ds.merge_csv(METADATA_CSV, join_on="file_name")
        ds.confirm_merge(report)
        ds.aggregate(
            name="by_condition",
            source_table="frames",
            group_by="condition",
            aggregations={"timestamp": "mean"},
        )
        frame_ids = set(ds.get_table("frames")["row_id"])
        agg_ids   = set(ds.get_table("by_condition")["row_id"])
        overlap   = frame_ids & agg_ids
        assert not overlap, (
            f"Aggregated row_ids overlap with frame row_ids: {overlap}"
        )

    def test_aggregate_from_aggregated_table():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        report = ds.merge_csv(METADATA_CSV, join_on="file_name")
        ds.confirm_merge(report)
        ds.aggregate(
            name="by_session",
            source_table="frames",
            group_by="session_id",
            aggregations={"timestamp": "mean"},
        )
        ds.aggregate(
            name="overall",
            source_table="by_session",
            group_by="session_id",
            aggregations={"timestamp": "mean"},
        )
        assert "overall" in ds.list_tables(), (
            "Should be able to aggregate an aggregated table"
        )

    run_test("Creates aggregated table", test_aggregate_creates_table)
    run_test("Correct number of groups", test_aggregate_correct_group_count)
    run_test("Aggregated table has row_id", test_aggregate_has_row_id)
    run_test("Aggregated row_ids do not overlap frames", test_aggregate_row_ids_dont_overlap)
    run_test("Can aggregate an aggregated table", test_aggregate_from_aggregated_table)


# ---------------------------------------------------------------------------
# Section 6: QueryEngine.apply()
# ---------------------------------------------------------------------------

section("6. QueryEngine.apply()")

if not METADATA_CSV.exists():
    print("  SKIP  All QueryEngine tests — metadata.csv not found.")
else:
    def _make_dataset_with_csv():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        report = ds.merge_csv(METADATA_CSV, join_on="file_name")
        ds.confirm_merge(report)
        return ds

    def test_apply_returns_list():
        from models.query_engine import QueryEngine
        ds     = _make_dataset_with_csv()
        qe     = QueryEngine()
        result = qe.apply(ds.get_table("frames"))
        assert isinstance(result, list), (
            f"apply() should return a list, got {type(result)}"
        )

    def test_apply_returns_all_ids_unfiltered():
        from models.query_engine import QueryEngine
        ds     = _make_dataset_with_csv()
        qe     = QueryEngine()
        df     = ds.get_table("frames")
        result = qe.apply(df)
        assert len(result) == len(df), (
            f"No filters should return all rows. "
            f"Expected {len(df)}, got {len(result)}"
        )

    def test_apply_filter_eq():
        from models.query_engine import QueryEngine, Filter
        ds     = _make_dataset_with_csv()
        qe     = QueryEngine()
        df     = ds.get_table("frames")
        result = qe.apply(df, filters=[Filter("condition", "eq", "positive")])
        assert len(result) > 0, "Filter returned no rows"
        assert len(result) < len(df), "Filter should reduce row count"
        returned_df = df[df["row_id"].isin(result)]
        assert (returned_df["condition"] == "positive").all(), (
            "Some returned rows do not have condition == positive"
        )

    def test_apply_filter_contains():
        """
        The 'contains' filter type should return rows where the column
        value contains the given substring (case-insensitive).
        """
        from models.query_engine import QueryEngine, Filter
        ds     = _make_dataset_with_csv()
        qe     = QueryEngine()
        df     = ds.get_table("frames")
        # 'pos' is a substring of 'positive'
        result = qe.apply(df, filters=[Filter("condition", "contains", "pos")])
        assert len(result) > 0, "contains filter returned no rows"
        returned_df = df[df["row_id"].isin(result)]
        assert returned_df["condition"].str.contains("pos", case=False).all(), (
            "Some returned rows do not contain 'pos' in condition"
        )

    def test_apply_does_not_modify_dataframe():
        from models.query_engine import QueryEngine, Filter
        ds           = _make_dataset_with_csv()
        qe           = QueryEngine()
        df           = ds.get_table("frames")
        count_before = len(df)
        qe.apply(df, filters=[Filter("condition", "eq", "positive")])
        assert len(df) == count_before, (
            "apply() modified the original DataFrame — it must not"
        )

    def test_apply_sort():
        from models.query_engine import QueryEngine
        ds        = _make_dataset_with_csv()
        qe        = QueryEngine()
        df        = ds.get_table("frames")
        result    = qe.apply(df, sort_by="condition", ascending=True)
        result_df = df.set_index("row_id").loc[result]
        conditions = list(result_df["condition"])
        assert conditions == sorted(conditions), (
            "Results not sorted by condition ascending"
        )

    def test_apply_randomise_reproducible():
        from models.query_engine import QueryEngine
        ds = _make_dataset_with_csv()
        qe = QueryEngine()
        df = ds.get_table("frames")
        r1 = qe.apply(df, randomise=True, seed=42)
        r2 = qe.apply(df, randomise=True, seed=42)
        r3 = qe.apply(df, randomise=True, seed=99)
        assert r1 == r2, "Same seed should produce same order"
        assert r1 != r3, "Different seeds should produce different order"

    def test_apply_multiple_filters():
        from models.query_engine import QueryEngine, Filter
        ds     = _make_dataset_with_csv()
        qe     = QueryEngine()
        df     = ds.get_table("frames")
        result = qe.apply(df, filters=[
            Filter("condition",  "eq", "positive"),
            Filter("session_id", "eq", "S01"),
        ])
        returned_df = df[df["row_id"].isin(result)]
        assert (returned_df["condition"]  == "positive").all()
        assert (returned_df["session_id"] == "S01").all()

    run_test("Returns a list", test_apply_returns_list)
    run_test("No filters returns all rows", test_apply_returns_all_ids_unfiltered)
    run_test("eq filter works correctly", test_apply_filter_eq)
    run_test("contains filter works correctly", test_apply_filter_contains)
    run_test("Does not modify input DataFrame", test_apply_does_not_modify_dataframe)
    run_test("Sort works correctly", test_apply_sort)
    run_test("Randomise is reproducible with seed", test_apply_randomise_reproducible)
    run_test("Multiple filters use AND logic", test_apply_multiple_filters)


# ---------------------------------------------------------------------------
# Section 7: QueryEngine.apply_grouped()
# ---------------------------------------------------------------------------

section("7. QueryEngine.apply_grouped()")

if not METADATA_CSV.exists():
    print("  SKIP  All apply_grouped tests — metadata.csv not found.")
else:
    def test_apply_grouped_returns_dict():
        from models.query_engine import QueryEngine
        ds     = _make_dataset_with_csv()
        qe     = QueryEngine()
        df     = ds.get_table("frames")
        result = qe.apply_grouped(df, group_by="condition")
        assert isinstance(result, dict), (
            f"apply_grouped() should return a dict, got {type(result)}"
        )

    def test_apply_grouped_correct_keys():
        from models.query_engine import QueryEngine
        ds     = _make_dataset_with_csv()
        qe     = QueryEngine()
        df     = ds.get_table("frames")
        result = qe.apply_grouped(df, group_by="condition")
        assert set(result.keys()) == {"positive", "negative", "neutral"}, (
            f"Expected keys positive/negative/neutral, got {set(result.keys())}"
        )

    def test_apply_grouped_all_rows_accounted():
        from models.query_engine import QueryEngine
        ds      = _make_dataset_with_csv()
        qe      = QueryEngine()
        df      = ds.get_table("frames")
        result  = qe.apply_grouped(df, group_by="condition")
        all_ids = [rid for ids in result.values() for rid in ids]
        assert sorted(all_ids) == sorted(df["row_id"].tolist()), (
            "Not all row_ids present across groups"
        )

    def test_apply_grouped_with_filter():
        from models.query_engine import QueryEngine, Filter
        ds     = _make_dataset_with_csv()
        qe     = QueryEngine()
        df     = ds.get_table("frames")
        result = qe.apply_grouped(
            df,
            group_by="condition",
            filters=[Filter("session_id", "eq", "S01")],
        )
        all_ids  = [rid for ids in result.values() for rid in ids]
        returned = df[df["row_id"].isin(all_ids)]
        assert (returned["session_id"] == "S01").all(), (
            "Grouped result contains rows from other sessions"
        )

    run_test("Returns a dict", test_apply_grouped_returns_dict)
    run_test("Correct group keys", test_apply_grouped_correct_keys)
    run_test("All row_ids accounted for", test_apply_grouped_all_rows_accounted)
    run_test("Filters applied before grouping", test_apply_grouped_with_filter)


# ---------------------------------------------------------------------------
# Section 8: Dataset.save() and load()
# ---------------------------------------------------------------------------

section("8. Dataset.save() and Dataset.load()")

def test_save_creates_parquet():
    from models.dataset import Dataset
    ds        = Dataset()
    ds.load_folder(TEST_IMAGES)
    save_path = TEMP_DIR / "gelem_test_save"
    ds.save(save_path)
    assert (save_path / "frames.parquet").exists(), (
        "frames.parquet not created in project folder"
    )

def test_save_load_roundtrip():
    from models.dataset import Dataset
    ds        = Dataset()
    ds.load_folder(TEST_IMAGES)
    save_path = TEMP_DIR / "gelem_test_roundtrip"
    ds.save(save_path)

    ds2 = Dataset()
    ds2.load(save_path)
    df1 = ds.get_table("frames")
    df2 = ds2.get_table("frames")
    assert len(df1) == len(df2), (
        f"Row count changed after save/load. Before: {len(df1)}, after: {len(df2)}"
    )
    assert list(df1.columns) == list(df2.columns), (
        "Column names changed after save/load"
    )

def test_save_load_preserves_values():
    from models.dataset import Dataset
    ds        = Dataset()
    ds.load_folder(TEST_IMAGES)
    ds.update_row("000001", {"test_value": 42.0})
    save_path = TEMP_DIR / "gelem_test_values"
    ds.save(save_path)

    ds2 = Dataset()
    ds2.load(save_path)
    row = ds2.get_row("000001")
    assert "test_value" in row, "Custom column not preserved after save/load"
    assert abs(row["test_value"] - 42.0) < 0.001, (
        f"Value changed after save/load. Expected 42.0, got {row['test_value']}"
    )

run_test("save() creates frames.parquet", test_save_creates_parquet)
run_test("save/load roundtrip preserves row count", test_save_load_roundtrip)
run_test("save/load preserves custom values", test_save_load_preserves_values)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("─" * 60)
print(f"Results: {_passed} passed, {_failed} failed")
if _failed == 0:
    print("All tests passed.")
else:
    print(f"{_failed} test(s) failed — see details above.")
print("─" * 60)
