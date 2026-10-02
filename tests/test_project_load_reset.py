"""
tests/test_project_load_reset.py

Guards AppController._reset_project_state(), the single place every
per-project piece of controller state is reset -- called by all three
load paths (load_folder, load_csv_as_primary, load_project) so they
cannot drift apart the way they had:

  1. The active table used to not be reset (or healed) by load_folder()
     or load_csv_as_primary() the way an earlier fix already made
     load_project() do. If the researcher switched to a table the
     previous project created (e.g. an operator's output table) and then
     opened a new folder or CSV, the controller kept pointing at a table
     name the new project did not have, and _refresh_result() failed
     with "Table '<name>' does not exist in this project." The gallery
     went empty and stayed that way -- nothing in the UI could select a
     table the combo box already appeared to show.
  2. No load path fired a signal that drove DetailWidget.clear() -- the
     only candidate, active_table_changed, is emitted by
     set_active_table() alone, and none of the three load paths went
     through it (load_project() assigns self._active_table directly). A
     stimulus shown before the load kept showing in the Detail tab after
     it, for all three load paths. project_loaded (emitted by
     _reset_project_state()) is what MainWindow now wires to
     DetailWidget.clear() instead.

Also exercises the user's literal repro steps -- a COLUMNS run started,
then cancelled, then a new project opened -- and the related case of a
project load arriving while a run is still live. Both drive a real
background worker thread through the controller's public
run_create_columns(), tracked and joined the same way
tests/test_result_delivery.py's thread-tracking helpers do. The
still-live case also checks that _reset_project_state() cancels the run
through cancel_run() -- the same mechanism the Cancel button uses --
rather than merely dropping it from the live-run registry.

Every test here asserts the correct post-load behaviour, so reverting
_reset_project_state() (or dropping its call from any one load path)
must turn at least one of these red -- see the reversal check this work
item's report describes.

A real background thread is involved in every "live run" test, so run this
module with faulthandler guards against a stuck join:

    python -m pytest tests/test_project_load_reset.py \
        -o faulthandler_timeout=60 -o faulthandler_exit_on_timeout=true
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

from models.dataset import Dataset
from operators.base import BaseOperator
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    ModeDescriptor,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
)
from ui.detail_widget import DetailWidget

TEST_IMAGES = project_root / "test_images"


# ---------------------------------------------------------------------------
# Descriptor / operator scaffolding. Duplicated from other test modules'
# builders rather than imported -- a test module is not a library.
# ---------------------------------------------------------------------------

def _active_table_input():
    return (
        InputSpec(
            name="active_table", label="Active table",
            kind=InputKind.ACTIVE_TABLE,
        ),
    )


def _columns_descriptor(name, label, output_columns):
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


class _BlockingProbe(BaseOperator):
    """A COLUMNS operator whose row work blocks on an Event, so a test can
    pause a real worker thread mid-run deterministically instead of racing
    it. `started` is set as soon as the first row begins; the call then
    waits on `proceed`, which the test sets once it has done whatever it
    needed to do while the run was still live.
    """

    name = "blocking_probe"
    descriptor = _columns_descriptor(
        "blocking_probe", "Blocking probe", [("probe", "numeric")]
    )

    def __init__(self):
        self.started = threading.Event()
        self.proceed = threading.Event()
        self.processed_row_ids: list[str] = []

    def create_columns(self, row_id, media, metadata, run):
        self.processed_row_ids.append(row_id)
        self.started.set()
        self.proceed.wait(timeout=10)
        return {"probe": len(self.processed_row_ids)}


def _start_blocking_run(controller, op_registry, monkeypatch, row_ids):
    """Registers a fresh _BlockingProbe and starts it through the real
    controller.run_create_columns(), tracking every thread it spawns.
    Returns once the worker has begun its first row (operator.started is
    set), so the caller can rely on the run being live. Returns
    (operator, operation_id, created_threads).
    """
    operator = _BlockingProbe()
    op_registry.register(operator)

    created: list[threading.Thread] = []
    real_thread = threading.Thread

    class _Tracked(real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    monkeypatch.setattr(threading, "Thread", _Tracked)
    try:
        controller.run_create_columns(operator.name, row_ids)
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)

    live = controller.get_live_runs()
    assert len(live) == 1, "run_create_columns did not register one live run"
    operation_id = live[0]["operation_id"]

    assert operator.started.wait(timeout=10), (
        "worker never started processing its first row"
    )
    return operator, operation_id, created


def _finish_blocking_run(operator, created_threads, controller):
    """Releases a _BlockingProbe's block, joins every thread the run
    spawned, and drains the controller's queues until nothing is left to
    apply -- the same shape tests/test_result_delivery.py's
    _drain_until_run_ends uses, inlined here since this module only needs
    it for cleanup, not as its own assertion target.
    """
    operator.proceed.set()
    for thread in created_threads:
        thread.join(timeout=10)
        assert not thread.is_alive(), "worker thread did not finish in time"
    for _ in range(20):
        controller._drain_queues()


# ---------------------------------------------------------------------------
# The three load paths, as plain callables -- so the tests below can be
# parametrized across all three rather than tripled by hand.
# ---------------------------------------------------------------------------

def _load_via_folder(controller, tmp_path):
    controller.load_folder(TEST_IMAGES)


def _load_via_csv(controller, tmp_path):
    csv_path = tmp_path / "other_project.csv"
    csv_path.write_text("value\n1\n2\n3\n")
    controller.load_csv_as_primary(csv_path)


def _build_other_project(tmp_path: Path) -> Path:
    csv_path = tmp_path / "other_project_source.csv"
    csv_path.write_text("value\n10\n20\n")
    ds = Dataset()
    ds.load_csv_as_primary(csv_path)
    proj = tmp_path / "other_project"
    ds.save(proj)
    return proj


def _load_via_project(controller, tmp_path):
    controller.load_project(_build_other_project(tmp_path))


LOAD_PATHS = [
    ("load_folder", _load_via_folder),
    ("load_csv_as_primary", _load_via_csv),
    ("load_project", _load_via_project),
]
LOAD_PATH_IDS = [name for name, _ in LOAD_PATHS]


# ---------------------------------------------------------------------------
# Defect 1: a stale active table left behind by a load path, reached the
# plain way -- just switch tables, no operator run involved.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, loader", LOAD_PATHS, ids=LOAD_PATH_IDS)
def test_a_stale_active_table_does_not_survive_any_load_path(
    name, loader, make_controller, tmp_path
):
    controller, dataset, _ = make_controller(tmp_path)
    frame_ids = list(dataset.get_table("frames")["row_id"])
    dataset.create_table_from_rows("frames_wors", frame_ids[:3])
    controller.set_active_table("frames_wors")
    assert controller.get_active_table() == "frames_wors"

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    loader(controller, tmp_path)

    assert controller.get_active_table() in dataset.list_tables(), (
        f"{name}() left the active table pointing at {controller.get_active_table()!r}, "
        f"which the new project does not have: {dataset.list_tables()!r}"
    )
    assert not any("does not exist in this project" in e for e in errors), (
        f"{name}() surfaced the stale-active-table error instead of "
        f"recovering to a valid table: {errors}"
    )


# ---------------------------------------------------------------------------
# Defect 1, the user's literal steps: a COLUMNS run on the custom table,
# cancelled mid-run, THEN a load. Real background thread, joined by hand.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, loader", LOAD_PATHS, ids=LOAD_PATH_IDS)
def test_a_load_after_a_cancelled_run_does_not_leave_the_active_table_stale(
    name, loader, make_controller, tmp_path, monkeypatch
):
    controller, dataset, op_registry = make_controller(tmp_path)
    frame_ids = list(dataset.get_table("frames")["row_id"])
    dataset.create_table_from_rows("frames_wors", frame_ids[:3])
    controller.set_active_table("frames_wors")

    operator, operation_id, created = _start_blocking_run(
        controller, op_registry, monkeypatch, frame_ids[:3]
    )

    # The researcher clicks Cancel while the run is mid-flight, then lets
    # the row already in progress finish -- the between-rows check (not
    # mid-row) is what the runner's cancellation contract promises.
    controller.cancel_run(operation_id)
    _finish_blocking_run(operator, created, controller)
    assert operation_id not in controller._live_runs

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    loader(controller, tmp_path)

    assert controller.get_active_table() in dataset.list_tables(), (
        f"{name}() after a cancelled run left the active table pointing "
        f"at {controller.get_active_table()!r}, which the new project "
        f"does not have: {dataset.list_tables()!r}"
    )
    assert not any("does not exist in this project" in e for e in errors), (
        f"{name}() surfaced the stale-active-table error instead of "
        f"recovering to a valid table: {errors}"
    )


# ---------------------------------------------------------------------------
# Defect 1 variant + the thread-cancellation question: a load arrives
# while the run is STILL live (not yet cancelled at all). The settled
# design requires the live run to be cancelled through the existing
# cancel_run() mechanism, not merely dropped from _live_runs -- today's
# three load paths only clear the dict, so the CancellationToken is never
# told, and the worker thread runs on regardless of the project switch.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, loader", LOAD_PATHS, ids=LOAD_PATH_IDS)
def test_a_load_while_a_run_is_still_live_cancels_it_through_cancel_run(
    name, loader, make_controller, tmp_path, monkeypatch
):
    controller, dataset, op_registry = make_controller(tmp_path)
    frame_ids = list(dataset.get_table("frames")["row_id"])
    dataset.create_table_from_rows("frames_wors", frame_ids[:3])
    controller.set_active_table("frames_wors")

    operator, operation_id, created = _start_blocking_run(
        controller, op_registry, monkeypatch, frame_ids[:3]
    )
    # Read the token before the load clears _live_runs -- the dict entry
    # disappears, but the CancellationToken object it pointed at does not.
    token = controller._live_runs[operation_id]["token"]

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    loader(controller, tmp_path)

    assert token.is_cancelled(), (
        f"{name}() replaced the project while \"blocking_probe\" was "
        f"still live. Its CancellationToken must be cancelled through "
        f"the same mechanism cancel_run() uses, not merely dropped from "
        f"the live-run registry -- otherwise the worker thread keeps "
        f"running against a project that is no longer open."
    )

    _finish_blocking_run(operator, created, controller)

    assert controller.get_active_table() in dataset.list_tables(), (
        f"{name}() left the active table pointing at "
        f"{controller.get_active_table()!r}, which the new project does "
        f"not have: {dataset.list_tables()!r}"
    )
    assert not any("does not exist in this project" in e for e in errors), (
        f"{name}() surfaced the stale-active-table error instead of "
        f"recovering to a valid table: {errors}"
    )
    # The dead run's results must never land in the replacement project,
    # whatever table ends up active there.
    assert "probe" not in dataset.get_table(controller.get_active_table()).columns


# ---------------------------------------------------------------------------
# Defect 2: no load path fires a signal that clears the Detail tab. Wired
# the same way ui/main_window.py wires it --
# controller.project_loaded.connect(detail.clear) -- rather than building
# a whole MainWindow.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, loader", LOAD_PATHS, ids=LOAD_PATH_IDS)
def test_detail_tab_clears_to_placeholder_on_every_load_path(
    name, loader, make_controller, realize_widget, qapp, tmp_path
):
    controller, dataset, _ = make_controller(tmp_path)
    detail = DetailWidget(controller)
    realize_widget(detail)
    controller.project_loaded.connect(detail.clear)

    row_id = list(dataset.get_table("frames")["row_id"])[0]
    detail.show_rows([row_id])
    qapp.processEvents()
    assert detail._label.text() != "No item selected", (
        "sanity: the detail view must actually be showing something "
        "before the load, or clearing it proves nothing"
    )

    loader(controller, tmp_path)
    qapp.processEvents()

    assert detail._label.text() == "No item selected", (
        f"{name}() left the previous project's stimulus showing in the "
        f"Detail tab -- it must clear to the empty placeholder on every "
        f"project load"
    )
    assert detail._media_widget is None
    assert not detail._close_btn.isEnabled()


# ---------------------------------------------------------------------------
# Bounded question: once _reset_project_state() cancels a live run, is it
# removed from _live_runs immediately, or does it linger until its worker
# finishes? load_folder() always names its table "frames", so a run still
# writing to "frames" in project A when a folder is opened -- which also
# gets a table named "frames" in project B -- is the one scenario where a
# late, wrongly-applied result could land in a table that merely looks
# like the right one because the name matches.
# ---------------------------------------------------------------------------

def test_a_cancelled_runs_late_completion_cannot_write_into_a_same_named_new_table(
    make_controller, tmp_path, monkeypatch
):
    controller, dataset, op_registry = make_controller(tmp_path)
    frame_ids = list(dataset.get_table("frames")["row_id"])

    operator, operation_id, created = _start_blocking_run(
        controller, op_registry, monkeypatch, frame_ids[:3]
    )

    columns_updates: list[list[str]] = []
    controller.columns_updated.connect(columns_updates.append)
    tables_updates: list[list[str]] = []
    controller.tables_updated.connect(tables_updates.append)

    controller.load_folder(TEST_IMAGES)

    # The bounded question itself: the run must be gone from the registry
    # in the SAME call that cancelled it, not left registered until its
    # worker eventually gets around to finishing.
    assert operation_id not in controller._live_runs, (
        "a cancelled run must be removed from _live_runs immediately when "
        "_reset_project_state() returns, not left registered until its "
        "worker finishes"
    )

    columns_updates.clear()
    tables_updates.clear()
    _finish_blocking_run(operator, created, controller)

    assert "probe" not in dataset.get_table("frames").columns, (
        "the cancelled run's output must never reach the replacement "
        "project's 'frames' table, even though both tables share that name"
    )
    # _on_operator_complete's "run is None" branch (controller.py) answers
    # the bounded question's second half directly: a dead run's belated
    # completion still emits plain operator_complete -- deliberately, so a
    # "running" UI indicator clears -- but returns before doing anything
    # that could affect a table, so neither signal below fires for it.
    assert columns_updates == [], (
        f"a dead run's belated completion must not refresh columns for "
        f"the replacement project: {columns_updates}"
    )
    assert tables_updates == [], (
        f"a dead run's belated completion must not refresh the table "
        f"list for the replacement project: {tables_updates}"
    )
