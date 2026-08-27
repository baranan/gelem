"""
tests/test_result_delivery.py

Tests for P0.2b (docs/media_architecture.md section 6.1, P0.2 items
6-9): bounded result draining, one operation_id per run carried through
every callback, staleness keyed on run liveness (never on the active
table), and batched frozen notification payloads.

Written from the work-item specification, not the implementation. Each
test states, in a comment, what would still pass if the rule it guards
were violated.

New file rather than added to tests/test_dataset.py, which runs its
whole body twice at import.

Run with:
    python -m pytest tests/test_result_delivery.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

from PySide6.QtWidgets import QApplication

from models.dataset import Dataset
from models.query_engine import QueryEngine
from models.notifications import RowsUpdated, ThumbnailsReady
from artifacts.artifact_store import ArtifactStore
from column_types.registry import ColumnTypeRegistry
from operators.operator_registry import OperatorRegistry
from controller import AppController

TEST_IMAGES = project_root / "test_images"
CONTROLLER_FILE = project_root / "controller.py"


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    if QApplication.instance() is None:
        QApplication(sys.argv)


def _make_controller(tmp_path, *, drain_budget=200):
    """A real controller over the 20-row test_images 'frames' table."""
    store    = ArtifactStore(tmp_path / "artifacts")
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)

    dataset = Dataset()
    dataset.set_registry(registry)
    dataset.load_folder(TEST_IMAGES)

    op_registry = OperatorRegistry()
    controller  = AppController(
        dataset, QueryEngine(), store, registry, op_registry,
        drain_budget=drain_budget,
    )
    controller.set_filters([])   # publish an initial result
    return controller, dataset, op_registry


# ---------------------------------------------------------------------------
# 1. A result from a run that is no longer live is never applied.
# ---------------------------------------------------------------------------

def test_result_from_dead_run_is_not_applied(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path)
    row_id = controller.get_visible_row_ids()[0]

    op_id = "dead-run"
    controller._register_run(op_id, "Probe", "frames")
    # Clear the registry exactly as load_folder() / load_project() do.
    controller._live_runs.clear()

    controller._on_item_complete(op_id, "frames", row_id, {"probe": 42})
    controller._drain_queues()

    # Would still pass if violated? No. If the drain applied dead-run
    # results, apply_row_updates() would have created the 'probe' column
    # and written 42 into it. Asserting the column never appeared catches
    # exactly the folder-reload data-corruption bug this item fixes.
    assert "probe" not in dataset.get_table("frames").columns


# ---------------------------------------------------------------------------
# 2. A live result lands in its own table even after the user switched
#    tables. This is what stops (1) being satisfied by dropping
#    everything, and stops staleness being keyed on _active_table.
# ---------------------------------------------------------------------------

def test_live_result_lands_in_its_own_table_after_table_switch(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path)
    frames_row = controller.get_visible_row_ids()[0]

    # A second table, then switch the active table away from 'frames'.
    dataset.create_table_from_rows(
        "other", list(dataset.get_table("frames")["row_id"])[:5],
        source_table="frames",
    )
    controller.set_active_table("other")
    assert controller.get_active_table() == "other"

    # A run that targets 'frames' completes while 'other' is active.
    op_id = "run-frames"
    controller._register_run(op_id, "Probe", "frames")
    controller._on_item_complete(op_id, "frames", frames_row, {"probe": 7})
    controller._drain_queues()

    # Would still pass if violated? No. If staleness were keyed on
    # _active_table the result would be dropped ('frames' is not active);
    # if the drain used _active_table as the write target it would land
    # in 'other'. Only "apply to the payload's own table, gated on run
    # liveness" puts probe=7 on the frames row and nowhere else.
    frames_after = dataset.get_table("frames").set_index("row_id")
    assert frames_after.loc[frames_row, "probe"] == 7
    assert "probe" not in dataset.get_table("other").columns


# ---------------------------------------------------------------------------
# 3. The drain is bounded: at most `drain_budget` items per queue per tick.
# ---------------------------------------------------------------------------

def test_drain_is_bounded(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path, drain_budget=5)
    k = controller._drain_budget          # read the configured bound, never a literal
    row_ids = controller.get_visible_row_ids()[: 3 * k]
    assert len(row_ids) == 3 * k, "need 3k distinct rows for this test"

    op_id = "big-run"
    controller._register_run(op_id, "Probe", "frames")
    for i, rid in enumerate(row_ids):
        controller._on_item_complete(op_id, "frames", rid, {"probe": i})

    controller._drain_queues()   # exactly one tick

    # Would still pass if violated? No. An unbounded drain would empty the
    # queue in this one tick (qsize 0). Asserting exactly `3k - k` remain
    # asserts the bound itself, and does so relative to the configured
    # budget rather than a hard-coded number.
    assert controller._item_result_queue.qsize() == 3 * k - k

    applied = dataset.get_table("frames")["probe"].notna().sum()
    assert applied == k


# ---------------------------------------------------------------------------
# 4. ...and it still completes over later ticks -- a bounded drain that
#    made no progress would also satisfy test 3.
# ---------------------------------------------------------------------------

def test_drain_completes_over_later_ticks(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path, drain_budget=5)
    k = controller._drain_budget
    row_ids = controller.get_visible_row_ids()[: 3 * k]

    op_id = "big-run"
    controller._register_run(op_id, "Probe", "frames")
    for i, rid in enumerate(row_ids):
        controller._on_item_complete(op_id, "frames", rid, {"probe": i})

    # Tick until the queue is empty (with a hard stop so a broken drain
    # cannot loop forever).
    for _ in range(20):
        if controller._item_result_queue.qsize() == 0:
            break
        controller._drain_queues()

    # Would still pass if violated? No. A drain that is bounded but never
    # advances (e.g. re-queues everything it takes) would leave the queue
    # non-empty and these rows unwritten forever. Asserting the actual
    # value written (row i got probe == i) also rules out an `is not None`
    # check that a NaN would slip past.
    assert controller._item_result_queue.qsize() == 0
    frames = dataset.get_table("frames").set_index("row_id")
    for i, rid in enumerate(row_ids):
        assert frames.loc[rid, "probe"] == i


# ---------------------------------------------------------------------------
# 5. No _drain* method uses list.pop(0). AST check -- asserts the
#    property, so a SimpleQueue or a deque both pass.
# ---------------------------------------------------------------------------

def test_no_drain_method_uses_pop_zero():
    source = CONTROLLER_FILE.read_text(encoding="utf-8")
    tree   = ast.parse(source)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_drain"):
            seg = ast.get_source_segment(source, node) or ""
            if "pop(0)" in seg:
                offenders.append(node.name)

    # Would still pass if violated? No -- any list-front drain fails.
    assert not offenders, f"_drain* methods using pop(0): {offenders}"


# ---------------------------------------------------------------------------
# 6. Progress is coalesced: many values within one tick -> one emission,
#    carrying the last value.
# ---------------------------------------------------------------------------

def test_progress_is_coalesced(tmp_path):
    controller, _, _ = _make_controller(tmp_path)

    seen: list[int] = []
    controller.operator_progress.connect(seen.append)

    for percent in range(100):
        controller._on_progress(percent)
    controller._drain_queues()

    # Would still pass if violated? No. If progress were a per-item queue,
    # `seen` would be list(range(100)) (or the first `drain_budget` of
    # them). Exactly [99] asserts both the coalescing and that it is the
    # LAST value that survives.
    assert seen == [99]

    # A tick with no new progress emits nothing more.
    controller._drain_queues()
    assert seen == [99]


# ---------------------------------------------------------------------------
# 7. Row ids that could not be placed are counted and reported.
# ---------------------------------------------------------------------------

def test_unplaceable_row_ids_are_reported(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path)

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    op_id = "run-x"
    controller._register_run(op_id, "My Op", "frames")
    controller._on_item_complete(op_id, "frames", "no-such-row", {"probe": 1})
    controller._drain_queues()                      # apply: records the miss
    controller._on_create_columns_complete(op_id, "my_op", 1)   # 1 result emitted
    controller._drain_queues()                      # completion: reports it

    # Would still pass if violated? No. If apply_row_updates() still
    # skipped an unknown row_id silently (its pre-P0.2b behaviour) no
    # error would fire. Requiring an error that names the count catches
    # the silent drop, at the layer the user sees.
    assert any("1" in m and "My Op" in m for m in errors), errors


# ---------------------------------------------------------------------------
# 7b. A create_columns completion is held back until this run's own
#     per-row results have all been applied -- and a second, faster run
#     feeding the shared queue does not delay it.
# ---------------------------------------------------------------------------

def test_completion_waits_for_its_own_row_results_only(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path, drain_budget=5)
    k = controller._drain_budget
    ids = controller.get_visible_row_ids()

    completed: list[str] = []
    controller.operator_complete.connect(completed.append)

    # Run A: k results, then its completion.
    a_ids = ids[:k]
    controller._register_run("A", "Op A", "frames")
    for i, rid in enumerate(a_ids):
        controller._on_item_complete("A", "frames", rid, {"a": i})
    controller._on_create_columns_complete("A", "op_a", len(a_ids))

    # Run B (started after A): a full budget of results still queued
    # ahead of nothing -- it is here only to keep the shared queue
    # non-empty on the tick A's completion is first seen.
    b_ids = ids[k:2 * k]
    controller._register_run("B", "Op B", "frames")
    for i, rid in enumerate(b_ids):
        controller._on_item_complete("B", "frames", rid, {"b": i})

    # First tick: drains k items (all of A's) but the queue still holds
    # B's, so a queue-emptiness test would wrongly defer A. A's own
    # applied count has reached len(a_ids), so A's completion fires.
    controller._drain_queues()

    # Would still pass if violated? If deferral were keyed on the shared
    # queue being empty, A's completion would be withheld here and
    # `completed` would be empty. Keying on A's own count lets it through.
    assert "op_a" in completed
    assert "A" not in controller._live_runs, "run A should be deregistered"


# ---------------------------------------------------------------------------
# 7c. A run that never starts does not linger in _live_runs.
# ---------------------------------------------------------------------------

def test_failed_operator_start_leaves_no_live_run(tmp_path, monkeypatch):
    from operators.base import BaseOperator

    class _Probe(BaseOperator):
        name                 = "probe"
        create_columns_label = "Probe"
        output_columns       = [("probe", "numeric")]
        requires_image       = False

        def create_columns(self, row_id, image, metadata):
            return {"probe": 1.0}

    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_Probe())
    ids = controller.get_visible_row_ids()[:3]

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    # (a) unknown operator: controller's own guard rejects it before it
    #     ever registers a run.
    controller.run_create_columns("does_not_exist", ids)
    assert errors and controller._live_runs == {}

    # (b) the registry declines the run without raising and without a
    #     callback (return False). The controller registered the run
    #     just before the call, so it must deregister on the False.
    monkeypatch.setattr(op_registry, "run_create_columns",
                        lambda *a, **k: False)
    controller.run_create_columns("probe", ids)

    # Would still pass if violated? No. Before this fix the entry stayed
    # in _live_runs forever -- no callback path deregisters a run that
    # never started.
    assert controller._live_runs == {}


# ---------------------------------------------------------------------------
# 7d. A create_table / create_display run whose operator does not
#     implement that mode is deregistered, not leaked.
# ---------------------------------------------------------------------------

def test_unimplemented_mode_run_leaves_no_live_run(tmp_path, monkeypatch):
    import threading as _threading
    from operators.base import BaseOperator

    class _NoTable(BaseOperator):
        # A label is set, so the controller's guard lets the run start;
        # create_table() is BaseOperator's default, which raises
        # NotImplementedError inside the worker.
        name               = "no_table"
        create_table_label = "No table"

    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_NoTable())
    ids = controller.get_visible_row_ids()[:3]

    created: list[_threading.Thread] = []
    real_thread = _threading.Thread

    class _Tracked(real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    monkeypatch.setattr(_threading, "Thread", _Tracked)
    controller.run_create_table("no_table", ids)
    monkeypatch.undo()

    assert created, "run_create_table did not start a worker thread"
    created[-1].join(timeout=5)
    assert not created[-1].is_alive()
    controller._drain_queues()

    # Would still pass if violated? No. Before this fix the worker caught
    # NotImplementedError and returned with no callback at all, so the run
    # the controller registered before starting it stayed live forever.
    assert controller._live_runs == {}


# ---------------------------------------------------------------------------
# 8. Notifications are batched frozen payloads carrying the table name.
# ---------------------------------------------------------------------------

def test_rows_updated_is_a_batched_payload_with_table_name(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path)
    row_ids = controller.get_visible_row_ids()[:3]

    payloads: list[object] = []
    controller.rows_updated.connect(payloads.append)

    op_id = "run-b"
    controller._register_run(op_id, "Probe", "frames")
    for i, rid in enumerate(row_ids):
        controller._on_item_complete(op_id, "frames", rid, {"probe": i})
    controller._drain_queues()

    # Would still pass if violated? No. A per-row signal would emit three
    # times; a bare-row_id signal would carry no table. One RowsUpdated
    # for the whole tick, naming 'frames', is the coalescing this item
    # asks for.
    assert len(payloads) == 1
    payload = payloads[0]
    assert isinstance(payload, RowsUpdated)
    assert payload.table_name == "frames"
    assert set(payload.row_ids) == set(row_ids)


# ---------------------------------------------------------------------------
# 9. Only the three dataset-replacing paths clear the run registry;
#    set_active_table() does not. Source check -- directly answers review
#    check #1.
# ---------------------------------------------------------------------------

def test_only_dataset_replacing_paths_clear_the_run_registry():
    source = CONTROLLER_FILE.read_text(encoding="utf-8")
    tree   = ast.parse(source)
    methods = {
        n.name: (ast.get_source_segment(source, n) or "")
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
    }

    clears = "_live_runs.clear()"
    for name in ("load_folder", "load_csv_as_primary", "load_project"):
        assert name in methods, f"{name} not found in controller.py"
        assert clears in methods[name], (
            f"{name} must clear the run registry -- it replaces the dataset"
        )

    assert clears not in methods["set_active_table"], (
        "set_active_table() must NOT clear the run registry: a live run's "
        "result belongs in its own table whatever is on screen"
    )

    # And nothing else clears it (keeps 'only those three paths' honest).
    other_clearers = [
        name for name, src in methods.items()
        if clears in src and name not in (
            "load_folder", "load_csv_as_primary", "load_project", "__init__",
        )
    ]
    assert not other_clearers, (
        f"unexpected methods clear the run registry: {other_clearers}"
    )
