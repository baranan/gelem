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
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import pytest

from PySide6.QtWidgets import QApplication

from models.dataset import Dataset
from models.query_engine import QueryEngine
from models.notifications import RowsUpdated, ThumbnailsReady
from artifacts.artifact_store import ArtifactStore
from column_types.registry import ColumnTypeRegistry
from operators.base import BaseOperator
from operators.operator_registry import OperatorRegistry
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    ModeDescriptor,
    NewTableNameParameter,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
)
from operators.run_context import CancellationToken
from controller import (
    AppController,
    format_cancel_message,
    format_row_error_summary,
    format_run_indicator_text,
    format_table_name_changed_message,
    numbered_run_choices,
    resolve_table_name,
    write_read_conflict_warnings,
)
from media.resolver import MediaResolver


def _active_table_input():
    return (
        InputSpec(
            name="active_table",
            label="Active table",
            kind=InputKind.ACTIVE_TABLE,
        ),
    )


def _columns_descriptor(name, label, output_columns):
    """A minimal COLUMNS-mode descriptor for a test double (P1.12d-2a:
    every operator the controller runs carries one)."""
    return OperatorDescriptor(
        name=name,
        version="1.0",
        description=f"Test double: {label}.",
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.COLUMNS,
                label=label,
                inputs=_active_table_input(),
                parameters=(),
                output=OutputSpec(
                    columns=tuple(
                        OutputColumn(name=col_name, type_tag=col_tag)
                        for col_name, col_tag in output_columns
                    )
                ),
            ),
        ),
    )


def _table_descriptor(name, label):
    """A minimal TABLE-mode descriptor for a test double. The double may
    still not implement create_table() -- the descriptor only declares the
    mode; BaseOperator.create_table() then raises NotImplementedError,
    which is exactly the path the 'unimplemented mode' test exercises."""
    return OperatorDescriptor(
        name=name,
        version="1.0",
        description=f"Test double: {label}.",
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.TABLE,
                label=label,
                inputs=_active_table_input(),
                parameters=(),
                output=OutputSpec(creates_table=True),
            ),
        ),
    )


def _table_descriptor_with_name_param(name, label, default_name):
    """Same as _table_descriptor, but the mode also declares a
    NewTableNameParameter -- fix round item 1's naming tests need an
    operator that lets the researcher (or the test) supply a table name."""
    return OperatorDescriptor(
        name=name,
        version="1.0",
        description=f"Test double: {label}.",
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.TABLE,
                label=label,
                inputs=_active_table_input(),
                parameters=(
                    NewTableNameParameter(
                        name="output_table",
                        label="New table name",
                        default=default_name,
                    ),
                ),
                output=OutputSpec(creates_table=True),
            ),
        ),
    )

TEST_IMAGES = project_root / "test_images"
CONTROLLER_FILE = project_root / "controller.py"


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    if QApplication.instance() is None:
        QApplication(sys.argv)


def _make_controller(tmp_path, *, drain_budget=200):
    """A real controller over the 20-row test_images 'frames' table."""
    store    = ArtifactStore(tmp_path / "artifacts", resolver=MediaResolver(max_open_decoders=4))
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)

    dataset = Dataset()
    dataset.load_folder(TEST_IMAGES)

    op_registry = OperatorRegistry()
    controller  = AppController(
        dataset, QueryEngine(), store, registry, op_registry,
        drain_budget=drain_budget,
    resolver=MediaResolver(max_open_decoders=4))
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
        descriptor = _columns_descriptor("probe", "Probe", [("probe", "numeric")])

        def create_columns(self, row_id, media, metadata, run):
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
        # A label AND a TABLE-mode descriptor are set, so the controller
        # builds the run and starts the worker; create_table() is still
        # BaseOperator's default, which raises NotImplementedError inside
        # the worker -- the path this test exercises.
        name               = "no_table"
        create_table_label = "No table"
        descriptor         = _table_descriptor("no_table", "No table")

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


# ---------------------------------------------------------------------------
# 10. P1.12f-1: the operator-run record and input staleness.
#
# These drive the controller's queues directly, the same way tests 1-9
# above do, rather than through run_create_columns/_build_operator_run --
# that lets each test set up the run's "inputs" and start version
# explicitly, which is the thing under test.
# ---------------------------------------------------------------------------

def _operator_run_entries(dataset):
    return [e for e in dataset.provenance.to_list() if e["action"] == "operator_run"]


def test_clean_columns_run_is_recorded_complete_and_not_superseded(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path)
    row_ids = controller.get_visible_row_ids()[:3]
    start_version = dataset.table_version("frames")

    op_id = "op-clean"
    controller._register_run(
        op_id, "Probe", "frames",
        operator_name="probe", mode_name="COLUMNS", target_table="frames",
        parameters={"threshold": 0.5}, rows_requested=len(row_ids),
        inputs={"active_table": {"table": "frames", "version": start_version}},
    )
    for i, rid in enumerate(row_ids):
        controller._on_item_complete(op_id, "frames", rid, {"probe": i})
    controller._on_create_columns_complete(op_id, "probe", len(row_ids))
    controller._drain_queues()

    # Would still pass if violated? No. An operator run leaves no
    # provenance trace at all today (the gap this item closes); asserting
    # the shape of the one entry it now writes catches a run that never
    # got recorded, and a "superseded" false positive on an ordinary run
    # that touched nothing else.
    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    params = entries[0]["params"]
    assert params["operator"] == "probe"
    assert params["mode"] == "COLUMNS"
    assert params["label"] == "Probe"
    assert params["target_table"] == "frames"
    assert params["rows_requested"] == len(row_ids)
    assert params["rows_applied"] == len(row_ids)
    assert params["unplaceable_row_ids"] == []
    assert params["outcome"] == "complete"
    assert params["superseded_tables"] == []
    assert op_id not in controller._live_runs


# ---------------------------------------------------------------------------
# 10a. THE FALSE-POSITIVE TRAP (step 0b of the spec): a COLUMNS run writes
# into the very table it read. Its own apply_row_updates() bumps that
# table's write-ticket version before the completion is processed. This
# must not read back as "superseded by itself".
# ---------------------------------------------------------------------------

def test_columns_run_is_not_superseded_by_its_own_writes(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path)
    row_ids = controller.get_visible_row_ids()[:5]
    start_version = dataset.table_version("frames")

    op_id = "op-self-write"
    controller._register_run(
        op_id, "Probe", "frames",
        operator_name="probe", mode_name="COLUMNS", target_table="frames",
        rows_requested=len(row_ids),
        inputs={"active_table": {"table": "frames", "version": start_version}},
    )
    for i, rid in enumerate(row_ids):
        controller._on_item_complete(op_id, "frames", rid, {"probe": i})
    controller._on_create_columns_complete(op_id, "probe", len(row_ids))
    # One tick: _drain_item_results applies all 5 results (bumping
    # 'frames' past start_version and recording that bump as THIS run's
    # own on run["own_versions"]), then _drain_completions sees
    # applied == emitted and processes the completion in the same tick.
    controller._drain_queues()

    # Would still pass if violated? No. A naive comparison of
    # table_version("frames") against start_version alone would see a
    # version bump (this run's own apply) and wrongly report
    # superseded_tables == ["frames"] on a run that only ever wrote its
    # own results.
    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["superseded_tables"] == []
    assert dataset.table_version("frames") > start_version, (
        "sanity: the run's own writes should actually have bumped the version"
    )


def test_columns_run_is_superseded_by_a_different_write(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path)
    row_ids = controller.get_visible_row_ids()[:5]
    start_version = dataset.table_version("frames")

    op_id = "op-foreign-write"
    controller._register_run(
        op_id, "Probe", "frames",
        operator_name="probe", mode_name="COLUMNS", target_table="frames",
        rows_requested=len(row_ids),
        inputs={"active_table": {"table": "frames", "version": start_version}},
    )
    for i, rid in enumerate(row_ids):
        controller._on_item_complete(op_id, "frames", rid, {"probe": i})
    controller._drain_queues()   # applies this run's own writes only

    # A second, unrelated write lands on 'frames' before this run's
    # completion is processed -- e.g. a computed column added from the
    # menu while the operator was still running.
    dataset.add_computed_column("unrelated", "1", table_name="frames")

    controller._on_create_columns_complete(op_id, "probe", len(row_ids))
    controller._drain_queues()

    # Would still pass if violated? No. If supersession were judged
    # against the run's own last-caused version alone (never re-checking
    # the CURRENT version), this foreign write would go unnoticed and
    # superseded_tables would wrongly stay empty.
    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["superseded_tables"] == ["frames"]


# ---------------------------------------------------------------------------
# 10a-fix (P1.12f-1-fix): a foreign write SANDWICHED between two of this
# run's own drain ticks is not visible at arrival -- this run's own later
# write becomes the table's last commit by completion time. Only a latch
# taken at the moment the foreign write is still the most recent thing
# that happened catches it.
# ---------------------------------------------------------------------------

def test_columns_run_latches_a_foreign_write_sandwiched_between_its_own_ticks(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path)
    row_ids = controller.get_visible_row_ids()[:3]
    start_version = dataset.table_version("frames")

    op_id = "op-sandwiched"
    controller._register_run(
        op_id, "Probe", "frames",
        operator_name="probe", mode_name="COLUMNS", target_table="frames",
        rows_requested=len(row_ids),
        inputs={"active_table": {"table": "frames", "version": start_version}},
    )

    # tick 1: this run applies results -> frames moves to v1, ours.
    controller._on_item_complete(op_id, "frames", row_ids[0], {"probe": 0})
    controller._drain_queues()

    # tick 2: this run applies results -> v2, ours.
    controller._on_item_complete(op_id, "frames", row_ids[1], {"probe": 1})
    controller._drain_queues()

    # between ticks: a FOREIGN write -> v3, not ours.
    dataset.add_computed_column("unrelated", "1", table_name="frames")

    # tick 3: this run applies results -> v4, ours.
    controller._on_item_complete(op_id, "frames", row_ids[2], {"probe": 2})
    controller._drain_queues()

    controller._on_create_columns_complete(op_id, "probe", len(row_ids))
    controller._drain_queues()

    # Would still pass if violated? No. At completion time the CURRENT
    # version (v4) equals the version this run's own tick 3 write just
    # caused, so a comparison made only at arrival sees nothing wrong --
    # exactly the missed detection this fix closes. Only the latch,
    # taken during tick 3's PRE-apply check (current version v3 there,
    # against this run's own last-caused version v2), ever sees the
    # foreign write. A missed detection here is a wrong number in a
    # paper; this pins that it is no longer missed.
    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["superseded_tables"] == ["frames"]


def test_columns_run_not_superseded_across_three_clean_ticks(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path)
    row_ids = controller.get_visible_row_ids()[:3]
    start_version = dataset.table_version("frames")

    op_id = "op-clean-three-tick"
    controller._register_run(
        op_id, "Probe", "frames",
        operator_name="probe", mode_name="COLUMNS", target_table="frames",
        rows_requested=len(row_ids),
        inputs={"active_table": {"table": "frames", "version": start_version}},
    )

    # Three separate drain ticks, one result each, no foreign write at
    # any point between them.
    for rid in row_ids:
        controller._on_item_complete(op_id, "frames", rid, {"probe": 1})
        controller._drain_queues()

    controller._on_create_columns_complete(op_id, "probe", len(row_ids))
    controller._drain_queues()

    # Would still pass if violated? No. If the pre-apply latch check
    # compared against the wrong baseline (e.g. always the start version
    # rather than this run's own last-caused version), an ordinary
    # multi-tick COLUMNS run would latch itself as superseded on its own
    # second and third ticks, exactly the false-positive trap P1.12f-1
    # already guards against -- this pins that the fix did not reopen it.
    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["superseded_tables"] == []


def test_superseded_run_emits_one_researcher_facing_message(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path)
    row_ids = controller.get_visible_row_ids()[:2]
    start_version = dataset.table_version("frames")

    messages: list[str] = []
    controller.error_occurred.connect(messages.append)

    op_id = "op-message"
    controller._register_run(
        op_id, "My Op", "frames",
        operator_name="my_op", mode_name="COLUMNS", target_table="frames",
        rows_requested=len(row_ids),
        inputs={"active_table": {"table": "frames", "version": start_version}},
    )
    for i, rid in enumerate(row_ids):
        controller._on_item_complete(op_id, "frames", rid, {"probe": i})
    controller._drain_queues()

    dataset.add_computed_column("unrelated", "1", table_name="frames")

    controller._on_create_columns_complete(op_id, "my_op", len(row_ids))
    controller._drain_queues()

    # Would still pass if violated? No. Silence here (no message, or one
    # that names neither the operator nor the table) would leave the
    # researcher trusting a result that no longer matches the data on
    # screen -- exactly what step 5 of the spec exists to prevent. No
    # re-run button and no blocking is asserted here on purpose: this
    # item is wording only.
    staleness_messages = [m for m in messages if "My Op" in m and "frames" in m]
    assert len(staleness_messages) == 1, messages


# ---------------------------------------------------------------------------
# 10b. Outcome: partial (unplaceable rows, or a setup_error / row_errors
# landed for this run) and failed (the "error" branch fired).
# ---------------------------------------------------------------------------

def test_operator_run_outcome_partial_on_unplaceable_rows(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path)
    start_version = dataset.table_version("frames")

    op_id = "op-unplaceable"
    controller._register_run(
        op_id, "My Op", "frames",
        operator_name="my_op", mode_name="COLUMNS", target_table="frames",
        rows_requested=1,
        inputs={"active_table": {"table": "frames", "version": start_version}},
    )
    controller._on_item_complete(op_id, "frames", "no-such-row", {"probe": 1})
    controller._on_create_columns_complete(op_id, "my_op", 1)
    controller._drain_queues()

    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    params = entries[0]["params"]
    assert params["outcome"] == "partial"
    assert params["unplaceable_row_ids"] == ["no-such-row"]
    assert params["unplaceable_row_count"] == 1
    assert params["rows_applied"] == 0
    # run-indicator-3-fix: nobody clicked Cancel here (_register_run was
    # never given a token) -- "partial" is entirely the unplaceable row's
    # doing, and cancellation_requested must say so.
    assert params["cancellation_requested"] is False


def test_operator_run_outcome_partial_after_setup_error(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path)
    start_version = dataset.table_version("frames")

    op_id = "op-setup-error"
    controller._register_run(
        op_id, "My Op", "frames",
        operator_name="my_op", mode_name="COLUMNS", target_table="frames",
        rows_requested=0,
        inputs={"active_table": {"table": "frames", "version": start_version}},
    )
    controller._on_operator_setup_error(op_id, "My Op", "model file missing")
    controller._drain_queues()
    controller._on_create_columns_complete(op_id, "my_op", 0)
    controller._drain_queues()

    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["outcome"] == "partial"


def test_operator_run_outcome_failed_on_error_branch(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path)
    start_version = dataset.table_version("frames")

    op_id = "op-error"
    controller._register_run(
        op_id, "My Op", "frames",
        operator_name="my_op", mode_name="COLUMNS", target_table="frames",
        rows_requested=1,
        inputs={"active_table": {"table": "frames", "version": start_version}},
    )
    controller._on_operator_error(op_id, "my_op", "boom")
    controller._drain_queues()

    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["outcome"] == "failed"


# ---------------------------------------------------------------------------
# 10c. Step 6: a run no longer live at arrival (the project was replaced)
# is not recorded. Same "arrived after the project changed" path every
# other mode already has -- this only checks the new recording does not
# fire on top of it.
# ---------------------------------------------------------------------------

def test_dead_run_completion_is_not_recorded(tmp_path):
    controller, dataset, _ = _make_controller(tmp_path)
    start_version = dataset.table_version("frames")

    op_id = "op-dead"
    controller._register_run(
        op_id, "Probe", "frames",
        operator_name="probe", mode_name="COLUMNS", target_table="frames",
        rows_requested=1,
        inputs={"active_table": {"table": "frames", "version": start_version}},
    )
    controller._live_runs.clear()   # e.g. load_folder() replaced the project

    controller._on_create_columns_complete(op_id, "probe", 0)
    controller._drain_queues()

    # Would still pass if violated? No. If the completion path recorded
    # unconditionally, this dead run would still leave a provenance entry
    # naming a run that no longer exists and rows that mean nothing after
    # the reload.
    assert _operator_run_entries(dataset) == []


# ---------------------------------------------------------------------------
# 11. P1.12f-2: the pre-emptive write/read conflict check.
#
# 11a. The pure, Qt-free function: write_read_conflict_warnings() takes
# live runs as plain data plus the about-to-start run's own read/write
# sets and returns plain-English sentences. Tested with no controller and
# no Qt widget at all.
# ---------------------------------------------------------------------------

def test_no_live_runs_means_no_warnings():
    # Would still pass if violated? No. A function that warned regardless
    # of live_runs being empty would produce a dialog on every single run.
    assert write_read_conflict_warnings([], "New run", {"frames"}, {"frames"}) == []


def test_disjoint_tables_produce_no_warning():
    live_runs = [{"label": "Other run", "reads": {"clips"}, "writes": {"clips"}}]
    # The about-to-start run touches "frames" only -- no table in common
    # with the live run's "clips" -- so nothing should be reported.
    assert write_read_conflict_warnings(
        live_runs, "New run", {"frames"}, {"frames"}
    ) == []


def test_new_run_reading_a_table_a_live_run_is_writing_warns():
    live_runs = [{"label": "Blendshapes", "reads": {"frames"}, "writes": {"frames"}}]
    # The about-to-start run only READS "frames"; it writes nothing.
    warnings = write_read_conflict_warnings(
        live_runs, "Summary stats", {"frames"}, set()
    )
    assert len(warnings) == 1
    assert "Blendshapes" in warnings[0]
    assert "Summary stats" in warnings[0]
    assert "frames" in warnings[0]


def test_new_run_writing_a_table_a_live_run_is_reading_warns():
    live_runs = [{"label": "Summary stats", "reads": {"frames"}, "writes": set()}]
    # The about-to-start run only WRITES "frames"; it reads nothing of
    # its own (an unrealistic but still valid input to the pure function).
    warnings = write_read_conflict_warnings(
        live_runs, "Blendshapes", set(), {"frames"}
    )
    assert len(warnings) == 1
    assert "Summary stats" in warnings[0]
    assert "Blendshapes" in warnings[0]
    assert "frames" in warnings[0]


def test_read_conflict_and_write_conflict_are_worded_differently():
    # Would still pass if violated? No. If both situations produced the
    # same sentence, a researcher could not tell a "your result may be
    # incomplete" risk from a "the other run's result may go stale" risk
    # -- the spec calls these two different situations that deserve two
    # different sentences.
    reads_conflict = write_read_conflict_warnings(
        [{"label": "A", "reads": {"t"}, "writes": {"t"}}], "B", {"t"}, set()
    )
    writes_conflict = write_read_conflict_warnings(
        [{"label": "A", "reads": {"t"}, "writes": {"t"}}], "B", set(), {"t"}
    )
    assert reads_conflict != writes_conflict


def test_each_conflicting_live_run_gets_its_own_sentence():
    live_runs = [
        {"label": "Run one", "reads": set(), "writes": {"frames"}},
        {"label": "Run two", "reads": set(), "writes": {"frames"}},
    ]
    warnings = write_read_conflict_warnings(
        live_runs, "New run", {"frames"}, set()
    )
    assert len(warnings) == 2
    assert any("Run one" in w for w in warnings)
    assert any("Run two" in w for w in warnings)


def test_warning_wording_avoids_technical_jargon():
    # Undergraduates have never heard "table version" -- the spec bans
    # these specific words from the researcher-facing sentences.
    warnings = write_read_conflict_warnings(
        [{"label": "A", "reads": {"t"}, "writes": {"t"}}], "B", {"t"}, {"t"},
    )
    banned = ("superseded", "stale", "write-ticket", "version")
    for sentence in warnings:
        lowered = sentence.lower()
        for word in banned:
            assert word not in lowered, f"{word!r} found in: {sentence!r}"


# ---------------------------------------------------------------------------
# 11b. AppController.get_write_read_conflict_warnings(): assembles the
# plain data from self._live_runs and the operator's descriptor, and
# returns no warnings for anything it cannot resolve.
# ---------------------------------------------------------------------------

def _register_probe(op_registry, name, label, output_columns):
    from operators.base import BaseOperator

    class _Probe(BaseOperator):
        create_columns_label = label
        descriptor = _columns_descriptor(name, label, output_columns)

        def create_columns(self, row_id, media, metadata, run):
            return {}

    _Probe.name = name
    op = _Probe()
    op_registry.register(op)
    return op


def _register_table_probe(op_registry, name, label):
    from operators.base import BaseOperator

    class _TableProbe(BaseOperator):
        create_table_label = label
        descriptor = _table_descriptor(name, label)

    _TableProbe.name = name
    op = _TableProbe()
    op_registry.register(op)
    return op


def test_get_write_read_conflict_warnings_is_empty_with_nothing_live(tmp_path):
    controller, dataset, op_registry = _make_controller(tmp_path)
    _register_probe(op_registry, "probe", "Probe", [("probe", "numeric")])

    # Would still pass if violated? No. A version that always compared
    # against an empty descriptor set would also pass here for the wrong
    # reason -- the sibling tests below pin the non-empty cases.
    assert controller.get_write_read_conflict_warnings("probe", "COLUMNS") == []


def test_get_write_read_conflict_warnings_when_new_run_reads_a_live_writer(tmp_path):
    controller, dataset, op_registry = _make_controller(tmp_path)
    _register_table_probe(op_registry, "aggregate", "Aggregate")
    start_version = dataset.table_version("frames")

    # A live COLUMNS run whose target table is "frames" -- its write set
    # is {"frames"} (P1.12f-1's target_table field).
    controller._register_run(
        "live-columns-run", "Blendshapes", "frames",
        operator_name="blendshapes", mode_name="COLUMNS", target_table="frames",
        rows_requested=5,
        inputs={"active_table": {"table": "frames", "version": start_version}},
    )

    # The about-to-start TABLE-mode run reads the active table ("frames")
    # and writes nothing -- so it conflicts with the live run's write.
    warnings = controller.get_write_read_conflict_warnings("aggregate", "TABLE")
    assert len(warnings) == 1
    assert "Blendshapes" in warnings[0]
    assert "frames" in warnings[0]


def test_get_write_read_conflict_warnings_when_new_run_writes_a_live_reader(tmp_path):
    controller, dataset, op_registry = _make_controller(tmp_path)
    _register_probe(op_registry, "probe", "Probe", [("probe", "numeric")])
    start_version = dataset.table_version("frames")

    # A live TABLE-mode run: it reads "frames" (its declared input) but
    # its target_table is "" -- TABLE mode writes nothing to an existing
    # table -- so its write set is empty.
    controller._register_run(
        "live-table-run", "Aggregate", "frames",
        operator_name="aggregate", mode_name="TABLE", target_table="",
        rows_requested=5,
        inputs={"active_table": {"table": "frames", "version": start_version}},
    )

    # The about-to-start COLUMNS run writes into "frames" -- so it
    # conflicts with the live run's read.
    warnings = controller.get_write_read_conflict_warnings("probe", "COLUMNS")
    assert len(warnings) == 1
    assert "Aggregate" in warnings[0]
    assert "frames" in warnings[0]


def test_get_write_read_conflict_warnings_unknown_operator_returns_empty(tmp_path):
    controller, dataset, op_registry = _make_controller(tmp_path)
    # Would still pass if violated? No. A version that raised or crashed
    # on an unknown name would fail this test loudly instead of quietly
    # returning no warnings.
    assert controller.get_write_read_conflict_warnings("does_not_exist", "COLUMNS") == []


def test_get_write_read_conflict_warnings_missing_mode_returns_empty(tmp_path):
    controller, dataset, op_registry = _make_controller(tmp_path)
    _register_probe(op_registry, "probe", "Probe", [("probe", "numeric")])
    # "probe" only declares a COLUMNS mode -- asking about TABLE mode
    # must return no warnings, not raise.
    assert controller.get_write_read_conflict_warnings("probe", "TABLE") == []


# ---------------------------------------------------------------------------
# 12. run-indicator-1: showing that an operator is running.
#
# 12a. format_run_indicator_text(): the pure, Qt-free sentence builder.
# Tested with plain lists and ints -- no controller, no Qt.
# ---------------------------------------------------------------------------

def test_indicator_text_is_empty_with_no_live_runs():
    # Would still pass if violated? No. A version that always produced
    # some text (even a generic "idle" message) would fail this, and the
    # empty string is exactly what tells the widget to hide itself.
    assert format_run_indicator_text([], None) == ""
    assert format_run_indicator_text([], 50) == ""


def test_indicator_text_one_run_no_progress_yet():
    live_runs = [{"label": "Extract blendshapes", "table_name": "frames"}]
    text = format_run_indicator_text(live_runs, None)
    assert "Extract blendshapes" in text
    assert "frames" in text
    # No progress tick has arrived -- no percentage should appear.
    assert "%" not in text


def test_indicator_text_one_run_with_progress():
    live_runs = [{"label": "Extract blendshapes", "table_name": "frames"}]
    text = format_run_indicator_text(live_runs, 42)
    assert "Extract blendshapes" in text
    assert "frames" in text
    assert "42" in text
    assert "%" in text


def test_indicator_text_two_runs_has_no_percentage():
    # Would still pass if violated? No. AppController coalesces progress
    # into one number for the whole application; showing that number
    # next to either run's label would misattribute it. Even when a
    # percent is supplied, two live runs must suppress it entirely.
    live_runs = [
        {"label": "Extract blendshapes", "table_name": "frames"},
        {"label": "Summary statistics", "table_name": "frames"},
    ]
    text = format_run_indicator_text(live_runs, 77)
    assert "%" not in text
    assert "77" not in text
    assert "Extract blendshapes" in text
    assert "Summary statistics" in text
    assert "2" in text


# ---------------------------------------------------------------------------
# 12b. AppController.get_live_runs() and the live_runs_changed signal.
# ---------------------------------------------------------------------------

def test_get_live_runs_empty_initially(tmp_path):
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    assert controller.get_live_runs() == []


def test_get_live_runs_returns_only_the_four_public_fields(tmp_path):
    # NOTE: this test's name and asserted shape changed for run-indicator-2.
    # It previously pinned exactly three keys; get_live_runs() now also
    # carries "message" (the run's newest run.log() text, None until one
    # arrives) per that item's explicit design decision -- see
    # controller.py's get_live_runs() docstring and operators/CLAUDE.md's
    # "Progress messages -- run.log". The property this test guards is
    # unchanged: only the fields a caller may rely on, never "token",
    # "column_tags" or the rest of the internal bookkeeping.
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    controller._register_run("op-1", "Probe", "frames")

    # Would still pass if violated? No. A version that returned the raw
    # _live_runs dict would also carry "token", "column_tags" and the
    # rest of the internal bookkeeping -- this pins the public read down
    # to exactly the four fields a caller may rely on: the run's opaque
    # identity (run-indicator-1-fix), its label and table, and its
    # newest log message (run-indicator-2, None before one arrives).
    live_runs = controller.get_live_runs()
    assert live_runs == [
        {
            "operation_id": "op-1",
            "label": "Probe",
            "table_name": "frames",
            "message": None,
        }
    ]


def test_get_live_runs_identity_matches_what_the_run_was_registered_under(
    tmp_path,
):
    # run-indicator-1-fix: the identity get_live_runs() hands back must be
    # the SAME value the caller (e.g. run_create_columns) registered the
    # run under -- it is the handle a future Cancel control passes back
    # to name which run to stop, so it cannot be some other derived id.
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    controller._register_run("the-exact-operation-id", "Probe", "frames")

    live_runs = controller.get_live_runs()
    assert len(live_runs) == 1
    assert live_runs[0]["operation_id"] == "the-exact-operation-id"


def test_two_concurrent_runs_have_different_identities(tmp_path):
    # Would still pass if violated? No. If get_live_runs() invented its
    # own identity (e.g. list position) rather than echoing back
    # operation_id, two runs with the same label on the same table could
    # come back indistinguishable -- exactly the case a Cancel control
    # needs to tell apart.
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    controller._register_run("op-1", "Plot columns (bar chart)", "frames")
    controller._register_run("op-2", "Plot columns (bar chart)", "frames")

    live_runs = controller.get_live_runs()
    ids = {run["operation_id"] for run in live_runs}
    assert ids == {"op-1", "op-2"}


def test_live_runs_changed_emitted_on_register(tmp_path):
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    events = []
    controller.live_runs_changed.connect(lambda: events.append(True))

    controller._register_run("op-1", "Probe", "frames")

    assert len(events) == 1


def test_live_runs_changed_emitted_on_deregister(tmp_path):
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    controller._register_run("op-1", "Probe", "frames")
    events = []
    controller.live_runs_changed.connect(lambda: events.append(True))

    controller._deregister_run("op-1")

    assert len(events) == 1
    assert controller.get_live_runs() == []


def test_live_runs_changed_not_emitted_on_deregister_of_unknown_id(tmp_path):
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    events = []
    controller.live_runs_changed.connect(lambda: events.append(True))

    # Would still pass if violated? No. An unconditional emit here would
    # fire the signal for a no-op pop, e.g. a second deregister of an
    # already-dead run -- this pins "emits only on an actual change".
    controller._deregister_run("never-registered")

    assert events == []


def test_load_folder_emits_live_runs_changed_once_when_a_run_was_live(tmp_path):
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    controller._register_run("op-1", "Probe", "frames")
    controller._register_run("op-2", "Other probe", "frames")
    events = []
    controller.live_runs_changed.connect(lambda: events.append(True))

    # load_folder() replaces the dataset, so it clears the run registry
    # (test 9 above pins that this is one of exactly three such paths).
    controller.load_folder(TEST_IMAGES)

    assert len(events) == 1
    assert controller.get_live_runs() == []


def test_load_folder_emits_no_live_runs_changed_when_nothing_was_live(tmp_path):
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    events = []
    controller.live_runs_changed.connect(lambda: events.append(True))

    # Would still pass if violated? No. An unconditional emit would fire
    # this signal on every load_folder() call even when no run was ever
    # started, which is not a change to the live-run set.
    controller.load_folder(TEST_IMAGES)

    assert events == []


def test_two_concurrent_runs_produce_a_sentence_with_no_percentage():
    # End-to-end through the same seam ui/main_window.py uses: plain data
    # from get_live_runs() (two entries) fed straight into
    # format_run_indicator_text(), pinning that the two layers agree.
    live_runs = [
        {"label": "Run one", "table_name": "frames"},
        {"label": "Run two", "table_name": "frames"},
    ]
    text = format_run_indicator_text(live_runs, 10)
    assert "%" not in text
    assert "10" not in text


# ---------------------------------------------------------------------------
# 13. run-indicator-2: free-text run.log() messages.
#
# 13a. format_run_indicator_text(): the message half of the sentence
# builder. Pure, Qt-free -- no controller.
# ---------------------------------------------------------------------------

def test_indicator_text_includes_a_message_for_one_run():
    live_runs = [
        {"label": "Extract frames", "table_name": "videos",
         "message": "video 3 of 40: clip.mp4"},
    ]
    text = format_run_indicator_text(live_runs, None)
    assert "Extract frames" in text
    assert "video 3 of 40: clip.mp4" in text


def test_indicator_text_omits_the_message_when_none_has_arrived_yet():
    # A run may carry "message": None (never logged) or omit the key
    # entirely (an older caller's plain dict, e.g. this file's own
    # pre-existing live_runs literals) -- both must render identically.
    text_none = format_run_indicator_text(
        [{"label": "Op", "table_name": "frames", "message": None}], None
    )
    text_missing = format_run_indicator_text(
        [{"label": "Op", "table_name": "frames"}], None
    )
    assert text_none == text_missing == 'Running "Op" on "frames"'


def test_indicator_text_truncates_an_absurdly_long_message():
    # Would still pass if violated? No. A version that showed the message
    # verbatim would put all 500 characters in the text; asserting a hard
    # ceiling well under that catches a missing truncation.
    huge = "x" * 500
    live_runs = [{"label": "Op", "table_name": "frames", "message": huge}]
    text = format_run_indicator_text(live_runs, None)
    assert len(text) < 200
    assert "..." in text
    assert huge not in text


def test_indicator_text_shows_each_runs_own_message_with_two_live_runs():
    # Unlike the coalesced application-wide percentage -- which two live
    # runs must suppress entirely (test 12 above) -- a run.log() message
    # is stored per operation_id (AppController._latest_logs), so there
    # is no such ambiguity. Both runs' own messages may appear together.
    live_runs = [
        {"label": "Extract blendshapes", "table_name": "frames",
         "message": "no face detected in row r7"},
        {"label": "Summary statistics", "table_name": "frames",
         "message": None},
    ]
    text = format_run_indicator_text(live_runs, 77)
    assert "%" not in text
    assert "Extract blendshapes" in text
    assert "no face detected in row r7" in text
    assert "Summary statistics" in text


# ---------------------------------------------------------------------------
# 13b. AppController: run.log() reaches get_live_runs() through the drain,
# two concurrent runs keep separate messages, and a message for a run that
# has already ended is dropped rather than raising.
# ---------------------------------------------------------------------------

def _message_for(controller, operation_id):
    for run in controller.get_live_runs():
        if run["operation_id"] == operation_id:
            return run["message"]
    return "<not live>"


def test_on_run_log_moves_the_message_onto_the_live_run_entry(tmp_path):
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    controller._register_run("op-1", "Probe", "frames")

    controller._on_run_log("op-1", "clip 3 of 40")
    controller._drain_queues()

    # Would still pass if violated? No. Before _apply_run_logs runs,
    # get_live_runs() still shows the "message": None it was registered
    # with -- this pins that the drain is what moves it across.
    assert _message_for(controller, "op-1") == "clip 3 of 40"


def test_two_concurrent_runs_keep_separate_messages(tmp_path):
    # Required by the work item: LATEST WINS, PER RUN means per
    # operation_id, not one shared value the way progress is shared.
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    controller._register_run("op-a", "Extract frames", "videos")
    controller._register_run("op-b", "Extract blendshapes", "frames")

    controller._on_run_log("op-a", "video 3 of 40: clip.mp4")
    controller._on_run_log("op-b", "no face detected in row r7")
    controller._drain_queues()

    # Would still pass if violated? No. A single shared "latest message"
    # variable (the progress pattern, applied naively) would leave both
    # runs showing whichever call happened to land last -- this fails
    # unless each operation_id keeps its own value.
    assert _message_for(controller, "op-a") == "video 3 of 40: clip.mp4"
    assert _message_for(controller, "op-b") == "no face detected in row r7"


def test_only_the_newest_message_per_run_is_kept(tmp_path):
    # LATEST WINS: several calls between two ticks collapse to one value,
    # the same coalescing the progress percentage already gets.
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    controller._register_run("op-1", "Probe", "frames")

    for i in range(50):
        controller._on_run_log("op-1", f"row {i}")
    controller._drain_queues()

    assert _message_for(controller, "op-1") == "row 49"


def test_logging_after_a_run_has_ended_does_not_raise(tmp_path):
    # Required by the work item. A worker thread may still be running
    # (or a straggling call may already be queued) after the main thread
    # has deregistered its run -- run.log() must not be able to crash
    # anything by calling in at that moment.
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    controller._register_run("op-1", "Probe", "frames")
    controller._deregister_run("op-1")

    controller._on_run_log("op-1", "the run this belonged to is gone")
    controller._drain_queues()   # must not raise

    assert controller.get_live_runs() == []


def test_operator_log_changed_emitted_once_per_tick_when_a_message_lands(
    tmp_path,
):
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    controller._register_run("op-1", "Probe", "frames")
    events: list[None] = []
    controller.operator_log_changed.connect(lambda: events.append(None))

    controller._on_run_log("op-1", "row 1")
    controller._on_run_log("op-1", "row 2")
    controller._drain_queues()

    # Would still pass if violated? No. Emitting once per _on_run_log call
    # (rather than once per tick, after coalescing) would leave `events`
    # with 2 entries here.
    assert len(events) == 1

    # A tick with no new message emits nothing more -- the same discipline
    # _emit_progress_if_changed already applies to operator_progress.
    controller._drain_queues()
    assert len(events) == 1


def test_operator_log_changed_not_emitted_for_a_message_on_a_dead_run(
    tmp_path,
):
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    controller._register_run("op-1", "Probe", "frames")
    controller._deregister_run("op-1")
    events: list[None] = []
    controller.operator_log_changed.connect(lambda: events.append(None))

    controller._on_run_log("op-1", "too late")
    controller._drain_queues()

    assert events == []


def test_run_dot_log_reaches_get_live_runs_through_the_real_wiring(tmp_path):
    # End-to-end through _build_operator_run's actual _log_fn wiring,
    # rather than calling controller._on_run_log() directly -- proves the
    # OperatorRun an operator actually receives is connected all the way
    # to get_live_runs(), the same seam ui/main_window.py reads.
    from operators.base import BaseOperator

    class _Probe(BaseOperator):
        name = "probe_table"
        descriptor = _table_descriptor("probe_table", "Probe table")

    controller, dataset, op_registry = _make_controller(tmp_path)
    operator = _Probe()
    op_registry.register(operator)
    ids = controller.get_visible_row_ids()[:3]
    snapshot = dataset.snapshot_rows("frames", ids)

    run, token = controller._build_operator_run(
        operator, ExecutionMode.TABLE, "op-real", "frames", {}, snapshot,
    )
    controller._register_run("op-real", "Probe table", "frames", token=token)

    run.log("clip 1 of 1")
    controller._drain_queues()

    assert _message_for(controller, "op-real") == "clip 1 of 1"


# ===========================================================================
# run-indicator-3: the Cancel control.
#
# AppController.cancel_run(), _run_outcome()'s cancellation check, and the
# discard-on-arrival behaviour for TABLE/DISPLAY are all exercised here by
# driving the controller's registry and queues directly -- the same style
# section 10 above uses -- so each test isolates what THE CONTROLLER does
# once a run's token is cancelled. Whether the COLUMNS row loop itself
# actually stops between rows (operators/operator_registry.py) is a
# separate concern, end-to-end tested in tests/test_run_cancellation.py.
#
# Written from the work-item specification, not the implementation.
# ===========================================================================

def test_cancel_run_sets_the_lives_runs_token(tmp_path):
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    token = CancellationToken()
    controller._register_run("op-1", "Probe", "frames", token=token)

    controller.cancel_run("op-1")

    # Would still pass if violated? No. A version that did nothing, or
    # that flipped some OTHER run's token, would leave this token
    # uncancelled.
    assert token.is_cancelled() is True


def test_cancel_run_on_an_id_that_is_not_live_is_a_noop_and_raises_nothing(
    tmp_path,
):
    # Required by the work item: the run may have finished (or never
    # existed) between the researcher's click and this call arriving.
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    token = CancellationToken()
    controller._register_run("op-1", "Probe", "frames", token=token)
    controller._deregister_run("op-1")   # the run already finished

    controller.cancel_run("op-1")             # must not raise
    controller.cancel_run("never-registered")  # must not raise either

    # Would still pass if violated? No. A version that looked up the id
    # and raised KeyError on a miss would never reach this line. A
    # version that cancelled SOME live run regardless of id would leave
    # this token (still held here, though no longer in _live_runs)
    # flipped -- it is not, because cancel_run() only ever reads
    # _live_runs, which no longer holds "op-1".
    assert token.is_cancelled() is False


def test_get_live_runs_returns_runs_in_registration_order(tmp_path):
    # run-indicator-3's Cancel picker numbers live runs "1.", "2." and so
    # on for the researcher to tell apart -- meaningless unless this order
    # really is the order the runs started, survives a deregister in the
    # middle, and is not just an accident of today's dict iteration.
    controller, _dataset, _op_registry = _make_controller(tmp_path)
    controller._register_run("op-1", "First", "frames")
    controller._register_run("op-2", "Second", "frames")
    controller._register_run("op-3", "Third", "frames")
    controller._deregister_run("op-2")
    controller._register_run("op-4", "Fourth", "frames")

    ids = [run["operation_id"] for run in controller.get_live_runs()]

    # Would still pass if violated? No. A version that sorted by label,
    # or that re-inserted a deregistered id's slot for the next
    # registration, would produce a different order here.
    assert ids == ["op-1", "op-3", "op-4"]


def test_cancelled_columns_run_keeps_applied_rows_and_is_recorded_partial(
    tmp_path,
):
    # Pins the work item's central guarantee: a COLUMNS run cancelled
    # after some rows were already applied keeps those rows' values (they
    # are never rolled back) and is recorded "partial" -- not "complete"
    # (it did not process every row it was asked to) and not "failed"
    # (nothing errored; the rows it produced are real, kept results).
    #
    # run-indicator-3-fix: "partial" here comes from actually applying
    # fewer rows than requested (2 of 5), NOT from cancellation alone --
    # see test_cancelled_columns_run_that_applied_every_row_is_recorded_complete
    # below for the case that tells the two apart. cancellation_requested
    # records the separate fact that Cancel was clicked, regardless of
    # what the outcome turned out to be.
    controller, dataset, _ = _make_controller(tmp_path)
    row_ids = controller.get_visible_row_ids()[:5]
    start_version = dataset.table_version("frames")

    op_id = "op-cancelled"
    token = CancellationToken()
    controller._register_run(
        op_id, "Probe", "frames",
        operator_name="probe", mode_name="COLUMNS", target_table="frames",
        rows_requested=len(row_ids),
        inputs={"active_table": {"table": "frames", "version": start_version}},
        token=token,
    )

    # Only 2 of the 5 requested rows were processed before the researcher
    # clicked Cancel; the per-row runner's between-rows check is what
    # would stop it there in the real wiring (see
    # tests/test_run_cancellation.py for that end-to-end behaviour). Here
    # we feed the controller exactly what such a worker would have handed
    # it: two results, then cancellation, then a completion reporting only
    # those two as emitted.
    cancelled_ids = row_ids[:2]
    for i, rid in enumerate(cancelled_ids):
        controller._on_item_complete(op_id, "frames", rid, {"probe": i})
    controller.cancel_run(op_id)
    controller._on_create_columns_complete(op_id, "probe", len(cancelled_ids))
    controller._drain_queues()

    # The two rows the run did produce keep their new values -- nothing
    # already written is undone.
    frames_after = dataset.get_table("frames").set_index("row_id")
    for i, rid in enumerate(cancelled_ids):
        assert frames_after.loc[rid, "probe"] == i

    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    params = entries[0]["params"]
    assert params["rows_applied"] == len(cancelled_ids)
    assert params["outcome"] == "partial"
    assert params["cancellation_requested"] is True
    assert op_id not in controller._live_runs


def test_cancelled_columns_run_that_applied_every_row_is_recorded_complete(
    tmp_path,
):
    # run-indicator-3-fix RULING: outcome describes what the run
    # PRODUCED, not what the researcher asked for. A COLUMNS run that
    # applied every row it was asked to is "complete" even if Cancel was
    # clicked -- e.g. after the last row had already finished, before the
    # controller got a chance to process the completion. Recording that
    # as "partial" would write a false "this is incomplete" into
    # provenance a researcher may read long after the session ended.
    # cancellation_requested still records that the click happened, so
    # that fact is not lost either.
    controller, dataset, _ = _make_controller(tmp_path)
    row_ids = controller.get_visible_row_ids()[:3]
    start_version = dataset.table_version("frames")

    op_id = "op-cancelled-too-late"
    token = CancellationToken()
    controller._register_run(
        op_id, "Probe", "frames",
        operator_name="probe", mode_name="COLUMNS", target_table="frames",
        rows_requested=len(row_ids),
        inputs={"active_table": {"table": "frames", "version": start_version}},
        token=token,
    )

    # Every requested row was applied BEFORE cancel_run() is called --
    # standing in for a Cancel click that reaches the controller only
    # after the worker thread had already finished every row.
    for i, rid in enumerate(row_ids):
        controller._on_item_complete(op_id, "frames", rid, {"probe": i})
    controller.cancel_run(op_id)
    controller._on_create_columns_complete(op_id, "probe", len(row_ids))
    controller._drain_queues()

    # Would still pass if violated? No. Deriving "partial" from
    # cancellation alone (the pre-fix behaviour) would report "partial"
    # here even though every requested row was actually applied; this
    # asserts "complete" specifically.
    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    params = entries[0]["params"]
    assert params["rows_applied"] == len(row_ids)
    assert params["outcome"] == "complete"
    assert params["cancellation_requested"] is True


def test_cancelled_create_table_result_is_discarded_and_recorded_partial(
    tmp_path,
):
    # TABLE mode is single-shot: the runner cannot interrupt it, so
    # cancelling one does not stop the work. Its finished result must be
    # discarded when it arrives rather than stored -- an honest limit, not
    # a bug -- and the run is still recorded "partial", never "complete".
    controller, dataset, _ = _make_controller(tmp_path)
    tables_before = dataset.list_tables()

    op_id = "op-table-cancelled"
    token = CancellationToken()
    controller._register_run(
        op_id, "Mean face", "frames",
        operator_name="mean_face", mode_name="TABLE", target_table="",
        token=token,
    )
    controller.cancel_run(op_id)

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)
    tables_updated: list[list[str]] = []
    controller.tables_updated.connect(tables_updated.append)

    result_df = pd.DataFrame({"x": [1, 2, 3]})
    controller._on_operator_complete(
        "create_table", (op_id, "mean_face", result_df)
    )

    # Would still pass if violated? No. A version that stored the result
    # before checking cancellation would add "mean_face_result" to the
    # table list and emit tables_updated; neither happens here.
    assert dataset.list_tables() == tables_before
    assert tables_updated == []
    assert op_id not in controller._live_runs
    assert any("cancelled" in e.lower() for e in errors)

    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["outcome"] == "partial"
    # run-indicator-3-fix ruling item 4: a discarded single-shot result
    # produced nothing, so this stays "partial" -- unlike COLUMNS, there
    # is no "cancelled but still complete" case for TABLE/DISPLAY in this
    # codebase, because the store-or-discard decision and this outcome
    # come from the exact same cancellation check at the exact same
    # moment (see AppController._run_outcome's docstring).
    assert entries[0]["params"]["cancellation_requested"] is True


def test_cancelled_create_display_result_is_discarded_and_recorded_partial(
    tmp_path,
):
    # Same "cannot interrupt, discard on arrival" limit as TABLE mode
    # above, for DISPLAY.
    controller, dataset, _ = _make_controller(tmp_path)

    op_id = "op-display-cancelled"
    token = CancellationToken()
    controller._register_run(
        op_id, "Summary stats", "frames",
        operator_name="summary_stats", mode_name="DISPLAY", target_table="",
        token=token,
    )
    controller.cancel_run(op_id)

    shown: list[dict] = []
    controller.display_result_ready.connect(shown.append)
    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    controller._on_operator_complete(
        "create_display", (op_id, "summary_stats", {"summary": {}})
    )

    assert shown == []
    assert op_id not in controller._live_runs
    assert any("cancelled" in e.lower() for e in errors)

    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["outcome"] == "partial"
    assert entries[0]["params"]["cancellation_requested"] is True


# ---------------------------------------------------------------------------
# P1.7-1 fix round, items 1-5: row-error reporting for a TABLE-mode run.
#
# format_row_error_summary() itself (items 2, 3) is tested directly, Qt-free,
# with no controller. Item 4 (a discarded run's row-error report must not
# reach the researcher) and item 5 (end-to-end: a real controller, a real
# Dataset, checked by RECORDED outcome and by the table actually existing --
# not by callback order) are tested against a real AppController below.
# ---------------------------------------------------------------------------

def test_format_row_error_summary_counts_distinct_rows_not_entries():
    # Item 3: the same row_id reported twice (once via report_row_error,
    # once via a caught exception, say) must count as ONE affected row,
    # while still keeping both lines of detail.
    errors = [
        ("r1", "Flagged", "flagged first"),
        ("r1", "ValueError", "then raised"),
        ("r2", "Flagged", "flagged too"),
    ]
    message = format_row_error_summary(
        "My op", errors, mode_name="COLUMNS", result_stored=True,
    )
    # Would still pass if violated? No. len(errors) is 3; if the count
    # were len(errors) instead of distinct row_ids, this would read
    # "3 row(s)" instead of "2 row(s)".
    assert "2 row(s)" in message
    assert "Flagged" in message and "ValueError" in message


def test_format_row_error_summary_wording_by_mode_and_storage():
    errors = [("r1", "Flagged", "msg")]

    columns_msg = format_row_error_summary(
        "Op", errors, mode_name="COLUMNS", result_stored=True,
    )
    table_stored_msg = format_row_error_summary(
        "Op", errors, mode_name="TABLE", result_stored=True,
    )
    table_not_stored_msg = format_row_error_summary(
        "Op", errors, mode_name="TABLE", result_stored=False,
    )

    # Would still pass if violated? No. Before this fix every mode used
    # the COLUMNS wording ("no values for the new columns"), which is
    # false for a TABLE operator -- those source rows were never columns
    # of anything; they were left out of a table, or no table exists.
    assert "no values for the new columns" in columns_msg
    assert "no values for the new columns" not in table_stored_msg
    assert "no values for the new columns" not in table_not_stored_msg

    assert "left out of the stored table" in table_stored_msg
    assert "No table was stored" in table_not_stored_msg
    # The two TABLE messages must actually differ -- a wording bug that
    # made both branches return the same string would still pass every
    # assertion above on its own.
    assert table_stored_msg != table_not_stored_msg


# ---------------------------------------------------------------------------
# Real-controller test doubles for items 4 and 5. Each create_table()
# drops row_id from every row it returns (operators/CLAUDE.md: "Do not
# add row_id -- it is generated when the table is stored") -- unlike the
# registry-seam test double in tests/test_table_output_contract.py, these
# actually go through Dataset.create_table_from_df's real row_id
# insertion, which raises on a duplicate column if row_id survives.
# ---------------------------------------------------------------------------

class _SkipsEveryOtherRow(BaseOperator):
    """A real TABLE operator: reports every even-position row through
    run.report_row_error() and excludes it from the result."""

    name = "skips_every_other_row_e2e"
    descriptor = _table_descriptor(
        "skips_every_other_row_e2e", "Skips every other row",
    )

    def create_table(self, df, run):
        kept_rows = []
        for position, (_, row) in enumerate(df.iterrows()):
            if position % 2 == 0:
                run.report_row_error(
                    row["row_id"], "Flagged",
                    f"row {row['row_id']} was flagged for skip",
                )
                continue
            kept_rows.append({"file_name": row["file_name"]})
        return pd.DataFrame(kept_rows)


class _ReportsThenRaises(BaseOperator):
    """Reports one row through run.report_row_error(), then raises --
    item 1's exact failure-path scenario, driven end to end."""

    name = "reports_then_raises_e2e"
    descriptor = _table_descriptor(
        "reports_then_raises_e2e", "Reports then raises",
    )

    def create_table(self, df, run):
        first_row_id = df.iloc[0]["row_id"]
        run.report_row_error(first_row_id, "Flagged", "reported before raising")
        raise ValueError("boom")


class _ReturnsARowIdColumn(BaseOperator):
    """create_table() itself succeeds -- unlike _ReportsThenRaises above,
    nothing here raises -- but the frame it returns already carries a
    "row_id" column, which operators/CLAUDE.md forbids ("do not add row_id
    -- it is generated when the table is stored"). Dataset.create_table_from_df
    inserts its own fresh "row_id" regardless, producing a duplicate column
    name that _prepare_table refuses. Fix round item 3's exact scenario: the
    STORE fails after the run itself already succeeded."""

    name = "returns_row_id_e2e"
    descriptor = _table_descriptor(
        "returns_row_id_e2e", "Returns a row_id column",
    )

    def create_table(self, df, run):
        return pd.DataFrame({"row_id": ["bogus_a", "bogus_b"], "value": [1, 2]})


class _NamesItsTable(BaseOperator):
    """A real TABLE operator declaring a NewTableNameParameter -- fix round
    item 1's get_operator() resolution test needs one whose descriptor
    actually carries the parameter to check."""

    name = "names_its_table_e2e"
    descriptor = _table_descriptor_with_name_param(
        "names_its_table_e2e", "Names its table", "segments",
    )

    def create_table(self, df, run):
        return pd.DataFrame({"value": [1, 2, 3]})


def _run_table_via_controller_and_join(controller, operator_name, row_ids, monkeypatch):
    """Calls controller.run_create_table() through the real public API,
    tracks and joins every background thread it spawns (the same pattern
    test_unimplemented_mode_run_leaves_no_live_run above uses), and
    returns the operation_id the controller assigned -- read off
    get_live_runs() immediately after the call, since _register_run runs
    synchronously on the calling thread before the worker starts."""
    created: list[threading.Thread] = []
    real_thread = threading.Thread

    class _Tracked(real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    monkeypatch.setattr(threading, "Thread", _Tracked)
    try:
        controller.run_create_table(operator_name, row_ids)
        live = controller.get_live_runs()
        assert len(live) == 1, "expected exactly one live run to have started"
        operation_id = live[0]["operation_id"]
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)

    for thread in created:
        thread.join(timeout=10)
        assert not thread.is_alive(), "worker thread did not finish in time"

    return operation_id


def _drain_until_run_ends(controller, operation_id, *, max_ticks=20):
    """Repeatedly calls the SAME drain loop the real 50ms QTimer calls
    (_drain_queues), so a small _drain_budget (item 5's ordering-cannot-
    hide-behind-a-large-budget check) still gets everything through over
    several ticks, the way it would in the running app."""
    for _ in range(max_ticks):
        controller._drain_queues()
        if operation_id not in controller._live_runs:
            return
    raise AssertionError(
        f"run {operation_id} was still live after {max_ticks} drain ticks"
    )


def test_e2e_table_operator_that_skips_rows_is_recorded_partial_and_stores_the_table(
    tmp_path, monkeypatch,
):
    # Item 5, case 1: real controller, real Dataset. Proves the RECORDED
    # outcome, not callback order, and that a real table is actually
    # accepted through Dataset.create_table_from_df's row_id insertion
    # (which a test double that kept row_id would never reach).
    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_SkipsEveryOtherRow())
    row_ids = controller.get_visible_row_ids()

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    operation_id = _run_table_via_controller_and_join(
        controller, "skips_every_other_row_e2e", row_ids, monkeypatch
    )
    _drain_until_run_ends(controller, operation_id)

    # Would still pass if violated? No. If report_row_error() were lost,
    # or never merged into the outcome check, this would read "complete".
    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["outcome"] == "partial"

    # The table must actually exist, with only the kept (odd-position)
    # rows -- proving create_table_from_df really accepted this
    # operator's real output (row_id dropped, as operators/CLAUDE.md
    # requires -- a test double that kept it would have raised here
    # instead, per this fix round's item 5 finding).
    assert "skips_every_other_row_e2e_result" in dataset.list_tables()
    stored = dataset.get_table("skips_every_other_row_e2e_result")
    kept_expected = len([i for i in range(len(row_ids)) if i % 2 == 1])
    assert len(stored) == kept_expected

    assert any("row(s) hit unexpected errors" in e for e in errors)
    assert any("left out of the stored table" in e for e in errors)


def test_e2e_table_operator_skipped_rows_with_drain_budget_one(tmp_path, monkeypatch):
    # Item 5, case 2: the same scenario, but with _drain_budget=1 so the
    # "row_errors" and "create_table" queue messages are necessarily
    # drained on SEPARATE ticks -- a small budget must not change the
    # outcome or hide a wrong delivery order.
    controller, dataset, op_registry = _make_controller(tmp_path, drain_budget=1)
    op_registry.register(_SkipsEveryOtherRow())
    row_ids = controller.get_visible_row_ids()

    operation_id = _run_table_via_controller_and_join(
        controller, "skips_every_other_row_e2e", row_ids, monkeypatch
    )
    _drain_until_run_ends(controller, operation_id, max_ticks=50)

    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["outcome"] == "partial"
    assert "skips_every_other_row_e2e_result" in dataset.list_tables()


def test_e2e_table_operator_reports_then_raises_is_recorded_failed(
    tmp_path, monkeypatch,
):
    # Item 5, case 3: reports a row, then raises. The report must still
    # reach the researcher (item 1), and the RECORDED outcome must be
    # "failed" -- never downgraded to "partial" just because a row-error
    # report also arrived first.
    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_ReportsThenRaises())
    row_ids = controller.get_visible_row_ids()

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    operation_id = _run_table_via_controller_and_join(
        controller, "reports_then_raises_e2e", row_ids, monkeypatch
    )
    _drain_until_run_ends(controller, operation_id)

    # Would still pass if violated? No. The bug this fix round's item 1
    # describes drops the report entirely on this exact path -- before
    # the fix, `errors` here would contain only the "boom" message, never
    # one mentioning the reported row.
    assert any("reported before raising" in e for e in errors)
    assert any("boom" in e for e in errors)

    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["outcome"] == "failed"
    assert "reports_then_raises_e2e_result" not in dataset.list_tables()


# ---------------------------------------------------------------------------
# P1.7-1 fix round, item 3: the run's recorded outcome must reflect whether
# the table was actually stored, not just whether create_table() itself
# raised.
# ---------------------------------------------------------------------------

def test_e2e_store_failure_is_recorded_failed_not_complete(tmp_path, monkeypatch):
    # create_table() itself succeeds here -- unlike
    # test_e2e_table_operator_reports_then_raises_is_recorded_failed above,
    # this run never goes through the "error" completion mode at all. The
    # STORE fails afterwards, inside _on_operator_complete's "create_table"
    # branch, when Dataset.create_table_from_df refuses the returned
    # frame's duplicate "row_id" column. Before this fix round,
    # _finish_run_provenance ran before this store was even attempted, so
    # the provenance entry recorded "complete" while the researcher was
    # separately told the store failed -- this proves the two now agree.
    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_ReturnsARowIdColumn())
    row_ids = controller.get_visible_row_ids()
    tables_before = dataset.list_tables()

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    operation_id = _run_table_via_controller_and_join(
        controller, "returns_row_id_e2e", row_ids, monkeypatch
    )
    _drain_until_run_ends(controller, operation_id)

    # Would still pass if violated? No. Before this fix, this same
    # sequence still stored nothing (create_table_from_df's rejection is
    # unaffected by this fix round) but recorded "complete" -- only the
    # outcome value distinguishes the fixed behaviour from the broken one.
    assert dataset.list_tables() == tables_before
    assert any("Failed to store table" in e for e in errors)

    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["outcome"] == "failed"


def test_row_errors_from_a_dead_run_are_suppressed(tmp_path):
    # Item 4: the project changed (or this run otherwise ended) between
    # its worker reporting rows and the drain reaching that report. No
    # dialog should reach the researcher about a run nobody is tracking
    # any more -- the SAME suppression every other dead-run completion
    # already gets (see test_result_from_dead_run_is_not_applied above).
    controller, dataset, _ = _make_controller(tmp_path)

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    # Never registered -- stands in for a run whose entry is already gone
    # by the time this message is drained.
    controller._on_operator_complete(
        "row_errors", ("dead-op", "Some op", [("r1", "Flagged", "msg")])
    )

    # Would still pass if violated? No. Before this fix the branch only
    # skipped setting had_row_errors for a dead run; it still built and
    # emitted the summary regardless of liveness.
    assert errors == []


# ---------------------------------------------------------------------------
# P1.7-1 fix round (round 4), item 1: a TABLE operator's NewTableNameParameter
# is resolved by the controller, never silently discarded, and never lets a
# collision overwrite an existing table.
# ---------------------------------------------------------------------------

def test_resolve_table_name_returns_suggestion_when_free():
    assert resolve_table_name("segments", set()) == "segments"


def test_resolve_table_name_suffixes_on_collision():
    assert resolve_table_name("segments", {"segments"}) == "segments_1"


def test_resolve_table_name_finds_the_first_free_suffix():
    # Would still pass if violated? No. A version that only ever tried
    # "_1" and gave up, or that restarted numbering from "_1" regardless
    # of what is already taken, would both return "segments_1" here even
    # though it is taken too.
    existing = {"segments", "segments_1"}
    assert resolve_table_name("segments", existing) == "segments_2"


def test_format_table_name_changed_message_names_both_names():
    message = format_table_name_changed_message("segments", "segments_2")
    assert "segments_2" in message
    assert "segments" in message


def test_get_operator_shows_the_resolved_suggestion_in_the_descriptor(tmp_path):
    # Fix round item 1, bullet 3: the parameter form must be built from the
    # REAL suggestion, not the operator's static declared default, so the
    # box never shows a name that is about to be resolved out from under
    # the researcher.
    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_NamesItsTable())
    dataset.create_table_from_df("segments", pd.DataFrame({"a": [1]}))

    operator = controller.get_operator("names_its_table_e2e")
    mode_descriptor = operator.descriptor.mode_for(ExecutionMode.TABLE)
    name_param = next(
        p for p in mode_descriptor.parameters if p.name == "output_table"
    )

    # Would still pass if violated? No. The operator's own declared
    # default is the plain string "segments"; if get_operator() returned
    # it unchanged, this would read "segments", not "segments_1".
    assert name_param.default == "segments_1"

    # The substitution never touches the real singleton, and every other
    # attribute -- in particular the execution method -- still reaches it:
    # this must not become a dead end for actually running the operator.
    assert op_registry.get("names_its_table_e2e").descriptor.mode_for(
        ExecutionMode.TABLE
    ).parameters[0].default == "segments"
    result = operator.create_table(pd.DataFrame({"value": [1]}), None)
    assert list(result["value"]) == [1, 2, 3]


def test_get_operator_returns_the_operator_unchanged_when_nothing_collides(tmp_path):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_NamesItsTable())

    operator = controller.get_operator("names_its_table_e2e")
    assert operator is op_registry.get("names_its_table_e2e")


def test_untouched_suggestion_still_resolves_when_the_dataset_moves_before_the_run_starts(
    tmp_path,
):
    # Fix round item 2 (second pass): pins that was_auto is decided by
    # REMEMBERING the suggestion the form showed, not by recomputing it at
    # run start. get_operator() shows "segments" (nothing exists under
    # that name yet); the set of existing tables then changes -- another
    # table is created under exactly that name -- BEFORE run_create_table
    # is ever called, standing in for a second live run's own dialog
    # completing while this one was still open (a modal QDialog keeps the
    # drain timer running). The researcher submits the suggestion
    # UNCHANGED. This must still resolve automatically: nothing was
    # edited, so a version that recomputed "what would I suggest now"
    # instead of remembering "what did I show" would see the collision
    # and wrongly refuse.
    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_NamesItsTable())

    operator = controller.get_operator("names_its_table_e2e")
    mode_descriptor = operator.descriptor.mode_for(ExecutionMode.TABLE)
    shown_default = mode_descriptor.parameters[0].default
    # Sanity: nothing collided yet, so the shown suggestion is the bare
    # declared default -- the case this test needs to be meaningful.
    assert shown_default == "segments"

    # The set of existing tables changes AFTER the dialog was built but
    # BEFORE run_create_table (and therefore _attach_table_name_resolution)
    # is ever called.
    dataset.create_table_from_df("segments", pd.DataFrame({"a": [1]}))

    op_id = "op-shown-vs-now"
    controller._register_run(
        op_id, "Names its table", "frames",
        operator_name="names_its_table_e2e", mode_name="TABLE", target_table="",
    )
    controller._attach_table_name_resolution(
        op_id, "names_its_table_e2e", mode_descriptor,
        {"output_table": shown_default},   # submitted exactly as shown
    )

    # Would still pass if violated? No. Recomputing instead of remembering
    # would compare "segments" against resolve_table_name("segments",
    # {"frames", "segments"}) == "segments_1" here -- unequal -- and read
    # was_auto False.
    run = controller._live_runs[op_id]
    assert run["new_table_name_was_auto"] is True
    assert run["new_table_name_requested"] == "segments"

    # And the visible behaviour matches: the run resolves to a fresh name
    # rather than being refused.
    messages: list[str] = []
    controller.error_occurred.connect(messages.append)
    controller._on_operator_complete(
        "create_table",
        (op_id, "names_its_table_e2e", pd.DataFrame({"value": [1, 2]})),
    )
    assert "segments_1" in dataset.list_tables()
    assert not any("already exists" in m for m in messages)

    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["outcome"] == "complete"


def test_auto_suggested_name_is_re_resolved_if_taken_before_the_store(tmp_path):
    # Fix round item 1, bullet 4: another run can take the shown name
    # between the dialog and the store. Simulated directly through the
    # run-registration seam (the same pattern
    # test_cancelled_create_table_result_is_discarded_and_recorded_partial
    # above uses) rather than racing two real background threads.
    controller, dataset, _ = _make_controller(tmp_path)

    op_id = "op-name-race"
    controller._register_run(
        op_id, "Cut into segments", "frames",
        operator_name="segment", mode_name="TABLE", target_table="",
    )
    controller._attach_table_name_resolution(
        op_id, "segment",
        _table_descriptor_with_name_param(
            "segment", "Cut into segments", "segments",
        ).mode_for(ExecutionMode.TABLE),
        {"output_table": "segments"},
    )

    # The name the researcher saw ("segments") is taken by another table
    # created AFTER this run started but BEFORE its result arrives.
    dataset.create_table_from_df("segments", pd.DataFrame({"a": [1]}))

    messages: list[str] = []
    controller.error_occurred.connect(messages.append)

    result_df = pd.DataFrame({"value": [1, 2]})
    controller._on_operator_complete(
        "create_table", (op_id, "segment", result_df)
    )

    # Would still pass if violated? No. If the collision were refused
    # instead of re-resolved, "segments_1" would never be created.
    assert "segments_1" in dataset.list_tables()
    assert any("segments_1" in m and "segments" in m for m in messages)

    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["outcome"] == "complete"


def test_edited_name_that_already_exists_is_refused_not_overwritten(tmp_path):
    # Fix round item 1, bullet 6: the researcher typed over the suggested
    # default with the name of a table that already exists. This must be
    # refused, matching Dataset.confirm_merge, never auto-suffixed and
    # never overwritten.
    controller, dataset, _ = _make_controller(tmp_path)
    dataset.create_table_from_df("frames_copy", pd.DataFrame({"a": [1]}))
    tables_before = dataset.list_tables()

    op_id = "op-name-edited"
    controller._register_run(
        op_id, "Cut into segments", "frames",
        operator_name="segment", mode_name="TABLE", target_table="",
    )
    controller._attach_table_name_resolution(
        op_id, "segment",
        _table_descriptor_with_name_param(
            "segment", "Cut into segments", "segments",
        ).mode_for(ExecutionMode.TABLE),
        {"output_table": "frames_copy"},
    )

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    result_df = pd.DataFrame({"value": [1, 2]})
    controller._on_operator_complete(
        "create_table", (op_id, "segment", result_df)
    )

    # Would still pass if violated? No. Auto-suffixing here instead of
    # refusing would create "frames_copy_1", changing dataset.list_tables()
    # from tables_before by more than nothing.
    assert dataset.list_tables() == tables_before
    assert any("already exists" in e for e in errors)

    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["outcome"] == "failed"


def test_operator_without_name_parameter_auto_suffixes_on_rerun(tmp_path):
    # No NewTableNameParameter declared -- the old fixed
    # f"{operator_name}_result" name is now just another auto suggestion
    # (fix round item 1), so re-running the same TABLE operator no longer
    # silently overwrites its own previous result the way it used to:
    # item 2 refuses that collision outright, and this is what stops the
    # refusal from ever being reached on this path.
    controller, dataset, _ = _make_controller(tmp_path)

    def _run_once(op_id):
        controller._register_run(
            op_id, "Mean face", "frames",
            operator_name="mean_face", mode_name="TABLE", target_table="",
        )
        controller._attach_table_name_resolution(
            op_id, "mean_face",
            _table_descriptor("mean_face", "Mean face").mode_for(
                ExecutionMode.TABLE
            ),
            {},
        )
        controller._on_operator_complete(
            "create_table",
            (op_id, "mean_face", pd.DataFrame({"value": [1]})),
        )

    _run_once("op-mean-face-1")
    _run_once("op-mean-face-2")

    # Would still pass if violated? No. If item 2's refusal reached this
    # path, the second call would leave "mean_face_result" as the only
    # stored table and report the run "failed" instead.
    assert "mean_face_result" in dataset.list_tables()
    assert "mean_face_result_1" in dataset.list_tables()
    entries = _operator_run_entries(dataset)
    assert [e["params"]["outcome"] for e in entries] == ["complete", "complete"]


# ---------------------------------------------------------------------------
# P1.7-1 fix round (third pass), item 1: a run's provenance is recorded
# exactly once, enforced rather than left to control flow, and a failure in
# a post-store notification is never reinterpreted as a store failure.
# ---------------------------------------------------------------------------

def test_finish_run_provenance_raises_if_called_twice_for_the_same_run(tmp_path):
    # Pins the exactly-once guard itself, independent of any particular
    # caller's control flow: whatever the reason, a second call for the
    # same run dict must be impossible to do silently.
    controller, dataset, _ = _make_controller(tmp_path)
    op_id = "op-double-finish"
    controller._register_run(
        op_id, "Probe", "frames",
        operator_name="probe", mode_name="COLUMNS", target_table="frames",
    )
    run = controller._live_runs[op_id]
    controller._finish_run_provenance("create_columns", run)

    # Would still pass if violated? No. Without the guard, this second
    # call would silently append a second, contradictory "operator_run"
    # provenance entry for the same run.
    with pytest.raises(RuntimeError):
        controller._finish_run_provenance("create_columns", run)

    entries = _operator_run_entries(dataset)
    assert len(entries) == 1


def test_a_failure_after_a_successful_store_does_not_reinterpret_or_duplicate_the_outcome(
    tmp_path, monkeypatch,
):
    # Fix round item 1 (third pass): something plain-Python AFTER the
    # store succeeds -- here, building the "stored under a different
    # name" notice -- raises. (A raising Qt *signal subscriber* was tried
    # first and turned out not to exercise this: PySide6 catches an
    # exception raised inside a connected slot and reports it through
    # sys.excepthook rather than letting it propagate back through
    # .emit(), so a broken subscriber can never reach this branch's own
    # except either way. A plain function call in the success path is
    # what a real bug here looks like.)
    #
    # This must not be caught and reported as "Failed to store table"
    # (the OLD shape of this branch wrapped this call in the same try as
    # the store itself), and the provenance entry it produces must be
    # exactly the one the NEW ordering writes before attempting this call
    # at all -- "complete", once.
    import controller as controller_module

    controller, dataset, _ = _make_controller(tmp_path)

    op_id = "op-post-store-boom"
    controller._register_run(
        op_id, "Cut into segments", "frames",
        operator_name="segment", mode_name="TABLE", target_table="",
    )
    # "segments" is free when the run starts, so the submitted, untouched
    # suggestion is recorded as auto.
    controller._attach_table_name_resolution(
        op_id, "segment",
        _table_descriptor_with_name_param(
            "segment", "Cut into segments", "segments",
        ).mode_for(ExecutionMode.TABLE),
        {"output_table": "segments"},
    )
    # It is taken AFTER the run starts but BEFORE its result arrives --
    # the same race test_auto_suggested_name_is_re_resolved_if_taken_before_the_store
    # above drives -- so the store resolves to "segments_1" and the
    # rename-notice code path (the one this test breaks) actually runs.
    dataset.create_table_from_df("segments", pd.DataFrame({"a": [1]}))

    def _boom(shown_name, stored_name):
        raise RuntimeError("wording bug")

    monkeypatch.setattr(
        controller_module, "format_table_name_changed_message", _boom
    )

    # Would still pass if violated? No. Under the OLD ordering, this
    # exception would land in the except clause, which would report
    # "Failed to store table from 'segment'" and record the run "failed"
    # -- even though the table was genuinely, successfully stored.
    with pytest.raises(RuntimeError, match="wording bug"):
        controller._on_operator_complete(
            "create_table",
            (op_id, "segment", pd.DataFrame({"value": [1, 2]})),
        )

    # The store itself genuinely succeeded, under the resolved name --
    # and provenance was recorded before the wording bug ever ran.
    assert "segments_1" in dataset.list_tables()

    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["outcome"] == "complete"


# ---------------------------------------------------------------------------
# P1.7-1 fix round (third pass), item 2: a run that fails during setup --
# after _register_run() but before its worker actually starts -- must not
# remain stuck in the live-run list.
# ---------------------------------------------------------------------------

def test_a_run_that_fails_during_setup_does_not_remain_live(tmp_path, monkeypatch):
    # A reviewer found that _attach_table_name_resolution() sat OUTSIDE
    # run_create_table()'s deregister-on-failure try/except: an exception
    # there left the run permanently stuck in self._live_runs --
    # uncancellable and uncompletable. Broken generically here (any setup
    # step raising, not only the one KeyError this item's fallback fix
    # already prevents), so the guarantee covers every step between
    # registration and the worker starting, not just this one method.
    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_NamesItsTable())

    def _boom(*args, **kwargs):
        raise RuntimeError("setup bug")

    monkeypatch.setattr(controller, "_attach_table_name_resolution", _boom)

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    controller.run_create_table(
        "names_its_table_e2e", controller.get_visible_row_ids(),
        # output_table is required, so a run must actually get past
        # _build_operator_run() and _register_run() to reach the
        # monkeypatched setup step below -- an empty/missing parameters
        # dict would be refused earlier and never register a live run at
        # all, which would make this test pass vacuously.
        {"output_table": "segments"},
    )

    # Would still pass if violated? No. Before this fix, this run would
    # still be live here -- get_live_runs() would return one entry, not
    # zero, and no future cancel_run() or completion could ever remove it.
    assert controller.get_live_runs() == []
    assert any("setup bug" in e for e in errors)


# ---------------------------------------------------------------------------
# run-indicator-3: numbered_run_choices() and format_cancel_message(), the
# Qt-free wording and numbering ui/main_window.py's Cancel control reads
# from rather than composing itself.
# ---------------------------------------------------------------------------

def test_numbered_run_choices_numbers_in_start_order():
    live_runs = [
        {
            "operation_id": "op-1", "label": "Extract blendshapes",
            "table_name": "frames",
        },
        {
            "operation_id": "op-2", "label": "Extract blendshapes",
            "table_name": "frames",
        },
    ]

    choices = numbered_run_choices(live_runs)

    assert [op_id for op_id, _label, _text in choices] == ["op-1", "op-2"]
    assert [label for _op_id, label, _text in choices] == [
        "Extract blendshapes", "Extract blendshapes",
    ]
    # Would still pass if violated? No. Without the number, two runs of
    # the same operator on the same table would produce identical display
    # text -- exactly the ambiguity numbering exists to remove.
    _op1, _label1, text1 = choices[0]
    _op2, _label2, text2 = choices[1]
    assert text1 != text2
    assert text1.startswith("1.")
    assert text2.startswith("2.")


def test_format_cancel_message_names_the_run_and_says_nothing_is_undone():
    message = format_cancel_message("Extract blendshapes")

    assert "Extract blendshapes" in message
    assert "undone" in message
