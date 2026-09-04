"""
tests/test_dataset_access_paths.py

Tests for P0.2a (Dataset access paths): the row-id index, get_row()
without a full-table copy, snapshot_rows(), read_only_view(), and
apply_row_updates() as the primary batch write path.

Written from the work-item specification (docs/media_architecture.md
§6.1, item "P0.2 Dataset access paths and result delivery", sub-items
1-5), not from the implementation -- each test is designed to fail
against a version of models/dataset.py that predates P0.2a.

Kept separate from tests/test_dataset.py, which is a hybrid script: it
executes its own tests at import time (via module-level run_test() calls)
and is also collected by pytest, so every test in that file runs twice.
Noted in the P0.2a handoff as a follow-up, not fixed here.

Run with:
    python -m pytest tests/test_dataset_access_paths.py
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import pytest

TEST_IMAGES  = project_root / "test_images"
METADATA_CSV = TEST_IMAGES / "metadata.csv"

NEEDS_CSV = {"add_computed_column", "confirm_merge", "aggregate", "load_csv_as_primary"}


def _rows_equal(a: dict, b: dict) -> bool:
    """Dict equality that treats NaN/None as equal to itself, the way a
    row read twice from the same table should compare."""
    if set(a) != set(b):
        return False
    for key in a:
        va, vb = a[key], b[key]
        if pd.isna(va) and pd.isna(vb):
            continue
        if va != vb:
            return False
    return True


# ---------------------------------------------------------------------------
# 1. get_row() does not go through get_table()
# ---------------------------------------------------------------------------

def test_get_row_does_not_call_get_table(monkeypatch):
    from models.dataset import Dataset

    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    row_id = ds.get_table("frames")["row_id"].iloc[0]

    def _raise(self, *args, **kwargs):
        raise AssertionError("get_row() must not call get_table()")

    monkeypatch.setattr(Dataset, "get_table", _raise)

    row = ds.get_row(row_id)
    assert row["row_id"] == row_id


# ---------------------------------------------------------------------------
# 2. Launching an operator does not copy the table per row
# ---------------------------------------------------------------------------

def test_run_create_columns_does_not_copy_table_per_row(monkeypatch, tmp_path):
    from PySide6.QtWidgets import QApplication
    if QApplication.instance() is None:
        QApplication(sys.argv)

    from models.dataset import Dataset
    from models.query_engine import QueryEngine
    from artifacts.artifact_store import ArtifactStore
    from column_types.registry import ColumnTypeRegistry
    from operators.operator_registry import OperatorRegistry
    from operators.base import BaseOperator
    from controller import AppController

    class _DummyOperator(BaseOperator):
        name                  = "dummy_op"
        create_columns_label  = "Dummy"
        output_columns        = [("dummy_score", "numeric")]
        requires_image        = False

        def create_columns(self, row_id, image, metadata):
            return {"dummy_score": 1.0}

    store    = ArtifactStore(tmp_path / "artifacts")
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)

    dataset = Dataset()
    dataset.load_folder(TEST_IMAGES)

    op_registry = OperatorRegistry()
    op_registry.register(_DummyOperator())

    controller = AppController(dataset, QueryEngine(), store, registry, op_registry)
    row_ids    = list(dataset.get_table("frames")["row_id"])

    counts = {"get_row": 0, "get_table": 0}
    original_get_row   = Dataset.get_row
    original_get_table = Dataset.get_table

    def _spy_get_row(self, *args, **kwargs):
        counts["get_row"] += 1
        return original_get_row(self, *args, **kwargs)

    def _spy_get_table(self, *args, **kwargs):
        counts["get_table"] += 1
        return original_get_table(self, *args, **kwargs)

    monkeypatch.setattr(Dataset, "get_row", _spy_get_row)
    monkeypatch.setattr(Dataset, "get_table", _spy_get_table)

    controller.run_create_columns("dummy_op", row_ids)

    assert counts["get_row"] == 0, (
        f"run_create_columns() called Dataset.get_row() {counts['get_row']} "
        f"times; it must take one snapshot instead of reading row by row"
    )
    assert counts["get_table"] <= 1, (
        f"run_create_columns() called Dataset.get_table() {counts['get_table']} "
        f"times; expected at most one full-table copy"
    )


# ---------------------------------------------------------------------------
# 2b. snapshot/row_id alignment (P0.2a follow-up fix 1)
#
# OperatorRegistry._run_create_columns_worker pairs snapshot.iloc[i] with
# row_ids[i] by position. If snapshot ever ends up shorter than row_ids --
# e.g. because a requested row_id no longer exists in the table -- every
# row from that point on would silently receive another row's data: wrong
# output, no error. These tests guard the two lines of defence against
# that (snapshot_rows() refusing to build a short frame, and
# run_create_columns() refusing to run with mismatched lengths) and the
# alignment invariant itself, end to end.
# ---------------------------------------------------------------------------

def test_snapshot_rows_raises_on_unknown_row_id():
    from models.dataset import Dataset

    ds = Dataset()
    ds.load_folder(TEST_IMAGES)

    with pytest.raises(KeyError) as exc_info:
        ds.snapshot_rows("frames", ["not_a_real_row_id"])
    assert "not_a_real_row_id" in str(exc_info.value), (
        f"KeyError should name the missing row_id; got: {exc_info.value}"
    )


def test_run_create_columns_raises_on_snapshot_length_mismatch():
    from operators.operator_registry import OperatorRegistry
    from operators.base import BaseOperator

    class _NoOpOperator(BaseOperator):
        name                  = "noop"
        create_columns_label  = "No-op"
        output_columns        = [("probe", "numeric")]
        requires_image        = False

        def create_columns(self, row_id, image, metadata):
            return {"probe": 0}

    op_registry = OperatorRegistry()
    op_registry.register(_NoOpOperator())

    snapshot = pd.DataFrame({"row_id": ["000001"]})   # 1 row
    row_ids  = ["000001", "000002"]                   # 2 ids -- mismatch

    with pytest.raises(ValueError):
        op_registry.run_create_columns("noop", snapshot, row_ids, "frames")


def test_run_create_columns_pairs_snapshot_rows_with_correct_row_id(monkeypatch, tmp_path):
    # End-to-end alignment invariant: whatever changes in the future about
    # how the snapshot is built or paired with row_ids, each row's result
    # must be derived from that row's own row_id, never a neighbour's.
    import threading as threading_module
    from PySide6.QtWidgets import QApplication
    if QApplication.instance() is None:
        QApplication(sys.argv)

    from models.dataset import Dataset
    from models.query_engine import QueryEngine
    from artifacts.artifact_store import ArtifactStore
    from column_types.registry import ColumnTypeRegistry
    from operators.operator_registry import OperatorRegistry
    from operators.base import BaseOperator
    from controller import AppController

    class _EchoRowIdOperator(BaseOperator):
        name                  = "echo_row_id"
        create_columns_label  = "Echo row id"
        output_columns        = [("probe", "numeric")]
        requires_image        = False

        def create_columns(self, row_id, image, metadata):
            # The result is derived purely from row_id, so a shifted
            # pairing shows up as a wrong value at every affected row.
            return {"probe": int(row_id)}

    store    = ArtifactStore(tmp_path / "artifacts")
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)

    dataset = Dataset()
    dataset.load_folder(TEST_IMAGES)

    op_registry = OperatorRegistry()
    op_registry.register(_EchoRowIdOperator())

    controller = AppController(dataset, QueryEngine(), store, registry, op_registry)
    row_ids    = list(dataset.get_table("frames")["row_id"])

    # Capture the worker thread run_create_columns starts, so we can join
    # it before asserting -- the controller's own QTimer never fires here
    # since no Qt event loop is running.
    created_threads: list[threading_module.Thread] = []
    RealThread = threading_module.Thread

    class _TrackingThread(RealThread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created_threads.append(self)

    monkeypatch.setattr(threading_module, "Thread", _TrackingThread)

    controller.run_create_columns("echo_row_id", row_ids)

    assert created_threads, "run_create_columns() did not start a worker thread"
    created_threads[-1].join(timeout=5)
    assert not created_threads[-1].is_alive(), "worker thread did not finish in time"

    # Apply the queued results the same way the QTimer-driven drain would.
    controller._drain_queues()

    for row_id in row_ids:
        row = dataset.get_row(row_id, "frames")
        assert row.get("probe") == int(row_id), (
            f"row {row_id} has probe={row.get('probe')!r}, expected "
            f"{int(row_id)} -- the snapshot/row_id pairing shifted"
        )


# ---------------------------------------------------------------------------
# 3. The index survives every mutating path
# ---------------------------------------------------------------------------

def _case_update_row(ds, tmp_path):
    df     = ds.get_table("frames")
    row_id = df["row_id"].iloc[0]
    ds.update_row(row_id, {"probe": 1.0}, "frames")
    return "frames"


def _case_apply_row_updates(ds, tmp_path):
    df  = ds.get_table("frames")
    ids = [df["row_id"].iloc[0], df["row_id"].iloc[len(df) // 2], df["row_id"].iloc[-1]]
    ds.apply_row_updates("frames", {rid: {"probe": i} for i, rid in enumerate(ids)})
    return "frames"


def _case_add_column(ds, tmp_path):
    df     = ds.get_table("frames")
    values = pd.Series({rid: i for i, rid in enumerate(df["row_id"])})
    ds.add_column("probe", values, "numeric", "frames")
    return "frames"


def _case_add_computed_column(ds, tmp_path):
    ds.confirm_merge(ds.merge_csv(METADATA_CSV, join_on="file_name"))
    ds.add_computed_column("probe", "timestamp * 2", "numeric", "frames")
    return "frames"


def _case_confirm_merge(ds, tmp_path):
    ds.confirm_merge(ds.merge_csv(METADATA_CSV, join_on="file_name"))
    return "frames"


def _case_create_table_from_rows(ds, tmp_path):
    ids = list(ds.get_table("frames")["row_id"])
    ds.create_table_from_rows("subset", ids, source_table="frames")
    return "subset"


def _case_create_table_from_df(ds, tmp_path):
    ds.create_table_from_df("derived", pd.DataFrame({"value": range(5)}))
    return "derived"


def _case_aggregate(ds, tmp_path):
    ds.confirm_merge(ds.merge_csv(METADATA_CSV, join_on="file_name"))
    ds.aggregate(
        "agg", source_table="frames", group_by="condition",
        aggregations={"timestamp": "mean"},
    )
    return "agg"


def _case_load_folder(ds, tmp_path):
    ds.load_folder(TEST_IMAGES)
    return "frames"


def _case_load_csv_as_primary(ds, tmp_path):
    ds.load_csv_as_primary(METADATA_CSV)
    return "frames"


def _case_load(ds, tmp_path):
    project = tmp_path / "proj"
    ds.save(project)
    ds.load(project)
    return "frames"


MUTATION_CASES = {
    "update_row":             _case_update_row,
    "apply_row_updates":      _case_apply_row_updates,
    "add_column":              _case_add_column,
    "add_computed_column":     _case_add_computed_column,
    "confirm_merge":           _case_confirm_merge,
    "create_table_from_rows":  _case_create_table_from_rows,
    "create_table_from_df":    _case_create_table_from_df,
    "aggregate":               _case_aggregate,
    "load_folder":             _case_load_folder,
    "load_csv_as_primary":     _case_load_csv_as_primary,
    "load":                    _case_load,
}


@pytest.mark.parametrize("case_name", sorted(MUTATION_CASES.keys()))
def test_index_survives_mutation(case_name, tmp_path):
    if case_name in NEEDS_CSV and not METADATA_CSV.exists():
        pytest.skip("metadata.csv not found -- run create_test_csv.py first")

    from models.dataset import Dataset

    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    table_name = MUTATION_CASES[case_name](ds, tmp_path)

    # get_table() is the independently-verified [NOW] ground truth
    # (test_get_table_still_returns_a_copy below); get_row() must agree
    # with it for the first, a middle, and the last row after this
    # mutation, whether or not this specific mutation touched that row.
    ground_truth = ds.get_table(table_name)
    assert len(ground_truth) > 0, f"[{case_name}] produced an empty table"

    positions = sorted({0, len(ground_truth) // 2, len(ground_truth) - 1})
    for pos in positions:
        expected = ground_truth.iloc[pos].to_dict()
        actual   = ds.get_row(expected["row_id"], table_name)
        assert _rows_equal(actual, expected), (
            f"[{case_name}] get_row() mismatch at position {pos}: "
            f"expected {expected}, got {actual}"
        )


# ---------------------------------------------------------------------------
# 4. The index is self-healing
# ---------------------------------------------------------------------------

def test_index_self_heals_after_direct_tables_write_different_length():
    # A replacement frame with a DIFFERENT row count invalidates the
    # cached index on length alone -- the weakref half of the stamp is
    # never exercised by this case. Kept alongside the same-length case
    # below, which is the one that actually tests the weakref.
    from models.dataset import Dataset

    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    stale_row_id = ds.get_table("frames")["row_id"].iloc[0]

    # Prime the cached index against the original frame.
    assert ds.get_row(stale_row_id) != {}

    # Bypass every Dataset method, the way several tests in
    # tests/test_dataset.py already do (search that file for `_tables[`).
    ds._tables["frames"] = pd.DataFrame({
        "row_id":    ["900001", "900002"],
        "full_path": ["a.jpg", "b.jpg"],
        "file_name": ["a.jpg", "b.jpg"],
    })

    assert ds.get_row(stale_row_id) == {}, (
        "get_row() returned a row from the old, no-longer-stored frame"
    )
    fresh = ds.get_row("900002")
    assert fresh["file_name"] == "b.jpg", (
        f"get_row() did not read from the newly-assigned frame; got {fresh}"
    )


def test_index_self_heals_after_direct_tables_write_same_length():
    # A replacement frame with the SAME row count as the one the cached
    # index was built from. The stamp's length check alone would pass
    # here, so this is the case that actually exercises the weakref half
    # of the stamp -- a stale index that only checked length would
    # return the old positions (or old row_ids entirely) against the new
    # frame's data.
    from models.dataset import Dataset

    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    original = ds.get_table("frames")
    n        = len(original)
    stale_row_id = original["row_id"].iloc[0]

    # Prime the cached index against the original frame.
    assert ds.get_row(stale_row_id) != {}

    replacement = pd.DataFrame({
        "row_id":    [f"800{i:03d}" for i in range(n)],
        "full_path": [f"replacement_{i}.jpg" for i in range(n)],
        "file_name": [f"replacement_{i}.jpg" for i in range(n)],
    })
    assert len(replacement) == n, "sanity: replacement must match the original row count"
    ds._tables["frames"] = replacement  # bypasses every Dataset method

    assert ds.get_row(stale_row_id) == {}, (
        "get_row() returned a row_id from the old frame after a "
        "same-length replacement -- the index did not notice the swap"
    )
    fresh = ds.get_row("800000")
    assert fresh["file_name"] == "replacement_0.jpg", (
        f"get_row() did not read from the newly-assigned, same-length "
        f"frame; got {fresh}"
    )


# ---------------------------------------------------------------------------
# 5. apply_row_updates() equals sequential update_row()
# ---------------------------------------------------------------------------

def test_apply_row_updates_equals_sequential_update_row():
    from models.dataset import Dataset

    ds_batch      = Dataset()
    ds_sequential = Dataset()
    ds_batch.load_folder(TEST_IMAGES)
    ds_sequential.load_folder(TEST_IMAGES)

    ids     = list(ds_batch.get_table("frames")["row_id"])[:5]
    updates = {rid: {"probe_a": i, "probe_b": i * 2.0} for i, rid in enumerate(ids)}

    ds_batch.apply_row_updates("frames", updates)
    for rid, col_updates in updates.items():
        ds_sequential.update_row(rid, col_updates, "frames")

    batch_df      = ds_batch.get_table("frames").sort_index(axis=1)
    sequential_df = ds_sequential.get_table("frames").sort_index(axis=1)
    pd.testing.assert_frame_equal(batch_df, sequential_df)


# ---------------------------------------------------------------------------
# 6. read_only_view() does not copy
# ---------------------------------------------------------------------------

def test_read_only_view_does_not_copy(monkeypatch):
    from models.dataset import Dataset

    ds = Dataset()
    ds.load_folder(TEST_IMAGES)

    copy_calls     = {"n": 0}
    original_copy  = pd.DataFrame.copy

    def _spy_copy(self, *args, **kwargs):
        copy_calls["n"] += 1
        return original_copy(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "copy", _spy_copy)

    view = ds.read_only_view("frames")

    assert copy_calls["n"] == 0, "read_only_view() must not copy the DataFrame"
    assert view is ds._tables["frames"], "read_only_view() must return the live table"


# ---------------------------------------------------------------------------
# 7. get_table() still returns a copy ([NOW], must not regress)
# ---------------------------------------------------------------------------

def test_get_table_still_returns_a_copy():
    from models.dataset import Dataset

    ds = Dataset()
    ds.load_folder(TEST_IMAGES)

    df = ds.get_table("frames")
    df.loc[df.index[0], "file_name"] = "mutated_outside_dataset.jpg"

    assert ds.get_table("frames")["file_name"].iloc[0] != "mutated_outside_dataset.jpg", (
        "get_table() must return a copy -- mutating it changed the stored table"
    )
