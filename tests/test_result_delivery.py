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
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    ModeDescriptor,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
)
from controller import AppController, write_read_conflict_warnings


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
