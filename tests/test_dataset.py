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

    # --- MANUALLY ADDED (Student B): merge edge cases. Tests above unchanged.
    # CASE 1 — one-to-many: CSV has >1 row per image; must be rejected, not
    # expanded (20 rows must not become 40).
    def _write_csv(folder, name, frame):
        """Helper: write `frame` to <folder>/<name> and return the Path."""
        path = Path(folder) / name
        frame.to_csv(path, index=False)
        return path

    def test_merge_one_to_many_does_not_crash():
        from models.dataset import Dataset, MergeReport
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        names = list(ds.get_table("frames")["file_name"])[:2]
        # Each image name appears twice -> one-to-many.
        bad = pd.DataFrame({
            "file_name": [names[0], names[0], names[1], names[1]],
            "score":     [1, 2, 3, 4],
        })
        with tempfile.TemporaryDirectory() as d:
            report = ds.merge_csv(_write_csv(d, "dupes.csv", bad),
                                  join_on="file_name")
        assert isinstance(report, MergeReport), (
            "merge_csv should return a MergeReport for a one-to-many CSV, "
            "not crash"
        )

    def test_merge_one_to_many_flags_duplicates():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        names = list(ds.get_table("frames")["file_name"])[:2]
        bad = pd.DataFrame({
            "file_name": [names[0], names[0], names[1], names[1]],
            "score":     [1, 2, 3, 4],
        })
        with tempfile.TemporaryDirectory() as d:
            report = ds.merge_csv(_write_csv(d, "dupes.csv", bad),
                                  join_on="file_name")
        assert names[0] in report.one_to_many and names[1] in report.one_to_many, (
            f"Duplicated image names should be listed in report.one_to_many; "
            f"got {report.one_to_many}"
        )

    def test_merge_one_to_many_does_not_expand():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        count_before = len(ds.get_table("frames"))
        names = list(ds.get_table("frames")["file_name"])[:2]
        bad = pd.DataFrame({
            "file_name": [names[0], names[0], names[1], names[1]],
            "score":     [1, 2, 3, 4],
        })
        with tempfile.TemporaryDirectory() as d:
            report = ds.merge_csv(_write_csv(d, "dupes.csv", bad),
                                  join_on="file_name")
            ds.confirm_merge(report)   # rejected one-to-many -> no-op
        count_after = len(ds.get_table("frames"))
        assert count_after == count_before, (
            f"A one-to-many merge must be rejected, not expand rows. "
            f"Before: {count_before}, after: {count_after}"
        )

    run_test("One-to-many merge does not crash", test_merge_one_to_many_does_not_crash)
    run_test("One-to-many merge flags duplicates", test_merge_one_to_many_flags_duplicates)
    run_test("One-to-many merge does not expand rows", test_merge_one_to_many_does_not_expand)

    def test_merge_unmatched_duplicate_still_merges():
        # A duplicate key matching NO image is harmless (can't expand any
        # image), so the merge must NOT be blocked.
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        real = list(ds.get_table("frames")["file_name"])[0]
        bad = pd.DataFrame({
            "file_name": ["ghost.jpg", "ghost.jpg", real],
            "score":     [1, 2, 5],
        })
        with tempfile.TemporaryDirectory() as d:
            report = ds.merge_csv(_write_csv(d, "ghost.csv", bad),
                                  join_on="file_name")
            ds.confirm_merge(report)
        df = ds.get_table("frames")
        assert report.one_to_many == [], (
            f"A duplicate matching no image must not block the merge; "
            f"got one_to_many={report.one_to_many}"
        )
        assert "score" in df.columns, (
            "A normal merge should still attach CSV columns"
        )

    run_test("Unmatched duplicate still merges", test_merge_unmatched_duplicate_still_merges)

    # CASE 2 — CSV missing the join column. The join column is whatever
    # join_on names (not necessarily file_name); if the CSV lacks it there's
    # nothing to match on, so merge_csv should raise a clear ValueError.
    def test_merge_missing_join_column_raises_clear_error():
        # Use a distinctive absent name so the error must reference the
        # REQUESTED column, not a hardcoded "file_name".
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        csv = pd.DataFrame({"name": ["a", "b"], "score": [1, 2]})  # no 'photo_id'
        with tempfile.TemporaryDirectory() as d:
            path = _write_csv(d, "noname.csv", csv)
            try:
                ds.merge_csv(path, join_on="photo_id")
                assert False, "expected a ValueError for the missing join column"
            except ValueError as e:
                assert "photo_id" in str(e), (
                    f"error should name the requested join column 'photo_id'; "
                    f"got: {e}"
                )

    def test_merge_with_alternate_join_column():
        # Filenames living in a differently-named column should still merge
        # (merge_csv takes join_on as a parameter).
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        names = list(ds.get_table("frames")["file_name"])[:3]
        csv = pd.DataFrame({"image": names, "score": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as d:
            report = ds.merge_csv(_write_csv(d, "altcol.csv", csv),
                                  join_on="image")
            ds.confirm_merge(report)
        assert "score" in ds.get_table("frames").columns, (
            "merging on an alternate join column should attach its data"
        )

    run_test("Missing join column raises clear error", test_merge_missing_join_column_raises_clear_error)
    run_test("Merge on alternate join column works", test_merge_with_alternate_join_column)

    # CASE 3 — column collision: an incoming CSV column already exists in the
    # table; keep both as <name>_a (existing) / <name>_b (incoming).
    def test_merge_column_collision_does_not_crash():
        from models.dataset import Dataset, MergeReport
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        names = list(ds.get_table("frames")["file_name"])[:3]
        csv = pd.DataFrame({"file_name": names, "path": ["x", "y", "z"]})
        with tempfile.TemporaryDirectory() as d:
            p = _write_csv(d, "coll.csv", csv)
            ds.confirm_merge(ds.merge_csv(p, join_on="file_name"))  # adds 'path'
            report = ds.merge_csv(p, join_on="file_name")           # 'path' collides
        assert isinstance(report, MergeReport), (
            "a column-collision merge should not crash"
        )

    def test_merge_column_collision_renames_both():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        names = list(ds.get_table("frames")["file_name"])[:3]
        csv = pd.DataFrame({"file_name": names, "path": ["x", "y", "z"]})
        with tempfile.TemporaryDirectory() as d:
            p = _write_csv(d, "coll.csv", csv)
            ds.confirm_merge(ds.merge_csv(p, join_on="file_name"))
            ds.confirm_merge(ds.merge_csv(p, join_on="file_name"))
        cols = list(ds.get_table("frames").columns)
        assert "path_a" in cols and "path_b" in cols, (
            f"both columns should be preserved as path_a/path_b; got {cols}"
        )
        assert "path" not in cols, "the clashing 'path' should have been renamed"

    def test_merge_column_collision_reported():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        names = list(ds.get_table("frames")["file_name"])[:3]
        csv = pd.DataFrame({"file_name": names, "path": ["x", "y", "z"]})
        with tempfile.TemporaryDirectory() as d:
            p = _write_csv(d, "coll.csv", csv)
            ds.confirm_merge(ds.merge_csv(p, join_on="file_name"))
            report = ds.merge_csv(p, join_on="file_name")
        assert report.renamed_columns.get("path") == ("path_a", "path_b"), (
            f"the collision should be recorded; got {report.renamed_columns}"
        )

    run_test("Column collision does not crash", test_merge_column_collision_does_not_crash)
    run_test("Column collision renames both", test_merge_column_collision_renames_both)
    run_test("Column collision is reported", test_merge_column_collision_reported)

    # CASE 4 — a failed merge must leave the table unchanged (merge_csv only
    # builds a report; confirm_merge is the only writer).
    def test_failed_merge_leaves_dataset_unchanged():
        from models.dataset import Dataset
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        rows_before = len(ds.get_table("frames"))
        cols_before = list(ds.get_table("frames").columns)
        bad = pd.DataFrame({"name": ["a", "b"], "score": [1, 2]})  # no join column
        with tempfile.TemporaryDirectory() as d:
            try:
                ds.merge_csv(_write_csv(d, "bad.csv", bad), join_on="file_name")
            except ValueError:
                pass  # expected failure
        assert len(ds.get_table("frames")) == rows_before, (
            "row count must not change after a failed merge"
        )
        assert list(ds.get_table("frames").columns) == cols_before, (
            "columns must not change after a failed merge"
        )

    def test_failed_merge_filters_still_work():
        from models.dataset import Dataset
        from models.query_engine import QueryEngine, Filter
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        df = ds.get_table("frames")
        target, target_id = df["file_name"].iloc[0], df["row_id"].iloc[0]
        bad = pd.DataFrame({"name": ["a"], "score": [1]})  # no join column
        with tempfile.TemporaryDirectory() as d:
            try:
                ds.merge_csv(_write_csv(d, "bad.csv", bad), join_on="file_name")
            except ValueError:
                pass
        rows = QueryEngine().apply(ds.get_table("frames"),
                                   filters=[Filter("file_name", "eq", target)])
        assert rows == [target_id], (
            f"filter should still return the one matching row; got {rows}"
        )

    run_test("Failed merge leaves dataset unchanged", test_failed_merge_leaves_dataset_unchanged)
    run_test("Failed merge: filters still work", test_failed_merge_filters_still_work)


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

# --- MANUALLY ADDED (Student B): Stage 10 thorough edge-case tests. -----
# Existing tests above are unchanged.

def test_load_missing_project_raises_clear_error():
    # A path that doesn't exist must fail loudly (no silent empty load).
    from models.dataset import Dataset
    ds = Dataset()
    try:
        ds.load(Path("definitely_does_not_exist_xyz_42"))
        assert False, "expected FileNotFoundError on missing project folder"
    except FileNotFoundError as e:
        assert "definitely_does_not_exist_xyz_42" in str(e), (
            f"error should name the missing folder; got: {e}"
        )

def test_load_empty_folder_raises_clear_error():
    # Folder exists but has no parquet files -> still loud fail.
    from models.dataset import Dataset
    ds = Dataset()
    with tempfile.TemporaryDirectory() as d:
        try:
            ds.load(Path(d))
            assert False, "expected FileNotFoundError on empty folder"
        except FileNotFoundError as e:
            msg = str(e).lower()
            assert "parquet" in msg or "tables" in msg, (
                f"error should mention missing tables/parquet; got: {e}"
            )

def test_load_replaces_existing_state():
    # "Open project" semantics: loading a project clears any prior in-memory
    # state. Guards the reset step that load() shares with load_folder /
    # load_csv_as_primary.
    from models.dataset import Dataset
    with tempfile.TemporaryDirectory() as d:
        project = Path(d)
        ds_a = Dataset()
        ds_a.load_folder(TEST_IMAGES)
        ds_a.save(project)
        ds_b = Dataset()
        ds_b.load_folder(TEST_IMAGES)
        ds_b._tables["extra"] = pd.DataFrame({"row_id": ["999999"], "x": [1]})
        ds_b.load(project)
        assert "extra" not in ds_b._tables, (
            "load should clear stale tables not in the saved project"
        )
        assert "frames" in ds_b._tables, "loaded project should restore frames"
        assert len(ds_b.get_table("frames")) == len(ds_a.get_table("frames")), (
            "frames row count after load should match the saved project"
        )

def test_save_load_preserves_id_counter():
    # _id_counter must be restored so the next _next_id() doesn't collide
    # with already-existing row_ids.
    from models.dataset import Dataset
    with tempfile.TemporaryDirectory() as d:
        project = Path(d)
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        ds.save(project)
        ds2 = Dataset()
        ds2.load(project)
        assert ds2._id_counter == ds._id_counter, (
            f"_id_counter should be restored; before={ds._id_counter}, "
            f"after={ds2._id_counter}"
        )
        existing_ids = set(ds2.get_table("frames")["row_id"])
        new_id = ds2._next_id()
        assert new_id not in existing_ids, (
            f"newly generated id '{new_id}' collides with an existing row_id"
        )

def test_save_load_preserves_multiple_tables():
    # A project with more than one table should round-trip both.
    from models.dataset import Dataset
    with tempfile.TemporaryDirectory() as d:
        project = Path(d)
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        ds._tables["extra"] = pd.DataFrame({
            "row_id": ["100001", "100002"],
            "value":  [10.0, 20.0],
        })
        ds.save(project)
        ds2 = Dataset()
        ds2.load(project)
        assert "frames" in ds2._tables, "frames must be restored"
        assert "extra"  in ds2._tables, "secondary table must be restored"
        assert len(ds2.get_table("extra")) == 2, (
            f"extra table should have 2 rows, got {len(ds2.get_table('extra'))}"
        )

def test_save_load_full_path_roundtrips():
    # full_path values must come back identical (external-images case).
    from models.dataset import Dataset
    with tempfile.TemporaryDirectory() as d:
        project = Path(d)
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        before = list(ds.get_table("frames")["full_path"])
        ds.save(project)
        ds2 = Dataset()
        ds2.load(project)
        after = list(ds2.get_table("frames")["full_path"])
        assert before == after, (
            f"full_path values should round-trip identically; "
            f"before sample={before[:2]}, after sample={after[:2]}"
        )

def test_save_uses_relative_path_when_images_inside_project():
    # When images live INSIDE project_path, save stores relative paths and
    # load restores absolute ones. Proves the "where possible" transform
    # actually fires; current external-images tests above never exercise it.
    from models.dataset import Dataset
    with tempfile.TemporaryDirectory() as d:
        project     = Path(d) / "proj"
        images_dir  = project / "images"
        images_dir.mkdir(parents=True)
        img1 = images_dir / "001.jpg"; img1.touch()
        img2 = images_dir / "002.jpg"; img2.touch()
        ds = Dataset()
        ds._tables["frames"] = pd.DataFrame({
            "row_id":    ["000001", "000002"],
            "full_path": [str(img1), str(img2)],
            "file_name": ["001.jpg", "002.jpg"],
        })
        ds.save(project)
        # The stored parquet should contain RELATIVE paths.
        stored = list(pd.read_parquet(project / "frames.parquet")["full_path"])
        assert all(not Path(p).is_absolute() for p in stored), (
            f"paths inside project should be stored relative; got {stored}"
        )
        assert all("\\" not in p for p in stored), (
            f"stored relative paths should use POSIX separators ('/'); got {stored}"
        )
        # And load should restore them to the originals.
        ds2 = Dataset()
        ds2.load(project)
        loaded = list(ds2.get_table("frames")["full_path"])
        assert loaded == [str(img1), str(img2)], (
            f"loaded paths should match the originals; got {loaded}"
        )

def test_save_load_preserves_provenance():
    # The history log survives save/load; load() records itself as the
    # last entry on the way back in.
    from models.dataset import Dataset
    with tempfile.TemporaryDirectory() as d:
        project = Path(d)
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        actions_before = [e["action"] for e in ds.provenance.to_list()]
        assert "load_folder" in actions_before, (
            "sanity: load_folder should be in the log before save"
        )
        ds.save(project)
        ds2 = Dataset()
        ds2.load(project)
        actions_after = [e["action"] for e in ds2.provenance.to_list()]
        assert "load_folder" in actions_after, (
            f"provenance should preserve load_folder entry; got {actions_after}"
        )
        assert actions_after[-1] == "load", (
            f"load() should record itself as the last entry; got {actions_after}"
        )

def test_save_load_re_registers_column_types():
    # After load, the registry knows about the loaded columns so the gallery
    # can render them (per the guide's "re-register column types" step).
    from models.dataset import Dataset
    from column_types.registry import ColumnTypeRegistry
    from artifacts.artifact_store import ArtifactStore

    with tempfile.TemporaryDirectory() as d:
        root      = Path(d)
        store     = ArtifactStore(root / "artifacts")
        registry  = ColumnTypeRegistry()
        registry.setup_defaults(store)
        ds = Dataset()
        ds.set_registry(registry)
        ds.load_folder(TEST_IMAGES)
        project = root / "project"
        ds.save(project)

        registry2 = ColumnTypeRegistry()
        registry2.setup_defaults(store)
        ds2 = Dataset()
        ds2.set_registry(registry2)
        ds2.load(project)
        full_path_type = registry2.get("full_path")
        assert full_path_type is not None, (
            "full_path should be re-registered after load"
        )
        assert full_path_type.tag == "media_path", (
            f"full_path should be registered as 'media_path'; "
            f"got '{full_path_type.tag}'"
        )
        file_name_type = registry2.get("file_name")
        assert file_name_type is None or file_name_type.tag != "media_path", (
            f"file_name should not be tagged 'media_path'; got "
            f"'{file_name_type.tag if file_name_type else None}'"
        )

run_test("load(): missing project folder raises clear error", test_load_missing_project_raises_clear_error)
run_test("load(): empty folder raises clear error",           test_load_empty_folder_raises_clear_error)
run_test("load(): replaces existing in-memory state",         test_load_replaces_existing_state)
run_test("save/load preserves _id_counter (no collision)",    test_save_load_preserves_id_counter)
run_test("save/load preserves multiple tables",               test_save_load_preserves_multiple_tables)
run_test("save/load full_path round-trips identically",       test_save_load_full_path_roundtrips)
run_test("save uses relative path when images inside project", test_save_uses_relative_path_when_images_inside_project)
run_test("save/load preserves provenance log",                test_save_load_preserves_provenance)
run_test("save/load re-registers column types",               test_save_load_re_registers_column_types)

def test_save_load_relative_source_path_resolves_correctly():
    # Locks finding #1: a relative full_path (e.g. from load_folder called
    # with a relative folder) must still resolve to the same file on disk
    # after save+load — not get re-rooted inside project_path.
    from models.dataset import Dataset
    with tempfile.TemporaryDirectory() as d:
        project = Path(d)
        rel_path = "test_images/001.jpg"
        ds = Dataset()
        ds._tables["frames"] = pd.DataFrame({
            "row_id":    ["000001"],
            "full_path": [rel_path],
            "file_name": ["001.jpg"],
        })
        ds.save(project)
        ds2 = Dataset()
        ds2.load(project)
        loaded = ds2.get_table("frames")["full_path"].iloc[0]
        assert Path(loaded).resolve() == Path(rel_path).resolve(), (
            f"round-trip should preserve target file; got {loaded}, "
            f"expected to resolve to {Path(rel_path).resolve()}"
        )

run_test("save/load: relative source path resolves correctly", test_save_load_relative_source_path_resolves_correctly)

def test_save_load_relativizes_all_media_path_columns():
    # Locks finding #4: any column the registry tags as media_path gets
    # relativized on save and restored on load — not only full_path.
    from models.dataset import Dataset
    from column_types.registry import ColumnTypeRegistry
    from artifacts.artifact_store import ArtifactStore
    with tempfile.TemporaryDirectory() as d:
        project = Path(d) / "proj"
        avatars = project / "avatars"
        avatars.mkdir(parents=True)
        avatar_file = avatars / "av1.jpg"
        avatar_file.touch()

        store    = ArtifactStore(Path(d) / "artifacts")
        registry = ColumnTypeRegistry()
        registry.setup_defaults(store)
        ds = Dataset()
        ds.set_registry(registry)
        ds._tables["frames"] = pd.DataFrame({
            "row_id":      ["000001"],
            "avatar_path": [str(avatar_file)],
        })
        registry.register_by_tag("avatar_path", "media_path")
        ds.save(project)

        # Stored avatar_path must be relative (registry tagged it media_path).
        stored = pd.read_parquet(project / "frames.parquet")["avatar_path"].iloc[0]
        assert not Path(stored).is_absolute(), (
            f"avatar_path (registered as media_path) should be stored relative; got {stored}"
        )

        # Load must restore avatar_path back to absolute.
        registry2 = ColumnTypeRegistry()
        registry2.setup_defaults(store)
        ds2 = Dataset()
        ds2.set_registry(registry2)
        ds2.load(project)
        loaded = ds2.get_table("frames")["avatar_path"].iloc[0]
        assert loaded == str(avatar_file), (
            f"avatar_path should resolve back to original absolute; got {loaded}"
        )

run_test("save/load relativizes all media_path columns", test_save_load_relativizes_all_media_path_columns)

def test_load_without_sidecar_still_registers_full_path():
    # Locks round-2 finding #1: a project saved without a registry has no
    # column_types.json. Load must still register full_path as media_path
    # (mirror load_folder's default) so the gallery renders thumbnails.
    from models.dataset import Dataset
    from column_types.registry import ColumnTypeRegistry
    from artifacts.artifact_store import ArtifactStore
    with tempfile.TemporaryDirectory() as d:
        project = Path(d) / "proj"
        # Save WITHOUT a registry -> no sidecar gets written.
        ds = Dataset()
        ds.load_folder(TEST_IMAGES)
        ds.save(project)
        assert not (project / "column_types.json").exists(), (
            "sanity: this test relies on no sidecar being written"
        )
        # Now load WITH a registry attached.
        store    = ArtifactStore(Path(d) / "artifacts")
        registry = ColumnTypeRegistry()
        registry.setup_defaults(store)
        ds2 = Dataset()
        ds2.set_registry(registry)
        ds2.load(project)
        ct = registry.get("full_path")
        assert ct is not None and ct.tag == "media_path", (
            f"full_path should fall back to media_path when sidecar is missing; "
            f"got tag={ct.tag if ct else None}"
        )

def test_load_unknown_tag_does_not_crash():
    # Locks round-2 finding #2: an unknown tag in column_types.json (e.g.
    # a custom type from an operator not available in this build) must
    # not sink the load — that one column stays unregistered, the rest
    # still register normally.
    import json as _json
    from models.dataset import Dataset
    from column_types.registry import ColumnTypeRegistry
    from artifacts.artifact_store import ArtifactStore
    with tempfile.TemporaryDirectory() as d:
        project = Path(d) / "proj"
        store    = ArtifactStore(Path(d) / "artifacts")
        registry = ColumnTypeRegistry()
        registry.setup_defaults(store)
        ds = Dataset()
        ds.set_registry(registry)
        ds.load_folder(TEST_IMAGES)
        ds.save(project)
        # Hand-edit the sidecar to inject an unknown tag.
        ct_path = project / "column_types.json"
        column_types = _json.loads(ct_path.read_text())
        column_types["mystery_col"] = "totally_unknown_tag"
        ct_path.write_text(_json.dumps(column_types))
        # Load should not crash.
        registry2 = ColumnTypeRegistry()
        registry2.setup_defaults(store)
        ds2 = Dataset()
        ds2.set_registry(registry2)
        ds2.load(project)
        assert registry2.get("full_path") is not None, (
            "known tags should still register despite an unknown tag in the sidecar"
        )

def test_save_load_realistic_merge_roundtrip():
    # End-to-end real-research scenario: load_folder + merge_csv + save +
    # load, with a registry attached. Asserts non-media column types
    # (numeric, text) also round-trip via the sidecar.
    from models.dataset import Dataset
    from column_types.registry import ColumnTypeRegistry
    from artifacts.artifact_store import ArtifactStore
    if not METADATA_CSV.exists():
        return  # SKIP — metadata.csv not available
    with tempfile.TemporaryDirectory() as d:
        project = Path(d) / "proj"
        store    = ArtifactStore(Path(d) / "artifacts")
        registry = ColumnTypeRegistry()
        registry.setup_defaults(store)
        ds = Dataset()
        ds.set_registry(registry)
        ds.load_folder(TEST_IMAGES)
        ds.confirm_merge(ds.merge_csv(METADATA_CSV, join_on="file_name"))
        # Sanity: registry should have tagged the merged columns.
        cond_before = registry.get("condition")
        ts_before   = registry.get("timestamp")
        assert cond_before is not None and cond_before.tag == "text", (
            f"sanity: condition should be 'text' before save; got {cond_before.tag if cond_before else None}"
        )
        assert ts_before is not None and ts_before.tag == "numeric", (
            f"sanity: timestamp should be 'numeric' before save; got {ts_before.tag if ts_before else None}"
        )
        ds.save(project)
        # Reload into fresh dataset + registry.
        registry2 = ColumnTypeRegistry()
        registry2.setup_defaults(store)
        ds2 = Dataset()
        ds2.set_registry(registry2)
        ds2.load(project)
        cond_after = registry2.get("condition")
        ts_after   = registry2.get("timestamp")
        assert cond_after is not None and cond_after.tag == "text", (
            f"condition should be 'text' after round-trip; got {cond_after.tag if cond_after else None}"
        )
        assert ts_after is not None and ts_after.tag == "numeric", (
            f"timestamp should be 'numeric' after round-trip; got {ts_after.tag if ts_after else None}"
        )

run_test("load(): no sidecar still registers full_path",      test_load_without_sidecar_still_registers_full_path)
run_test("load(): unknown tag in sidecar does not crash",     test_load_unknown_tag_does_not_crash)
run_test("save/load realistic merge round-trip preserves types", test_save_load_realistic_merge_roundtrip)


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
