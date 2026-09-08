"""
tests/test_operator_run_wiring.py

P1.12d-2a: the OperatorRun reaches the operator, and per-run parameter
values travel in an immutable OperatorRunSpec instead of being written
onto the operator singleton.

Written from the work-item specification. Each test states, in a comment,
what would still pass -- or fail -- if the behaviour it guards were
broken.

No Qt widget is shown here (a real AppController is built, as
tests/test_settings_gateway.py does, but nothing is realised), so the
run-tests substring heuristic classifies this module into the combined
group, which is correct. No `# run-tests:` token is needed.

Run with:
    python -m pytest tests/test_operator_run_wiring.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

from models.dataset import Dataset
from models.query_engine import QueryEngine
from artifacts.artifact_store import ArtifactStore
from column_types.registry import ColumnTypeRegistry
from operators.operator_registry import OperatorRegistry
from operators.base import BaseOperator
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    ModeDescriptor,
    NumberParameter,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
)
from controller import AppController
from ui.main_window import MainWindow

TEST_IMAGES = project_root / "test_images"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_controller(tmp_path):
    """A real AppController over the test_images 'frames' table. No
    QApplication -- the same construction tests/test_settings_gateway.py
    uses in the combined group."""
    store = ArtifactStore(tmp_path / "artifacts")
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)

    dataset = Dataset()
    dataset.load_folder(TEST_IMAGES)

    op_registry = OperatorRegistry()
    controller = AppController(
        dataset, QueryEngine(), store, registry, op_registry
    )
    controller.set_filters([])  # publish an initial query result
    return controller, dataset, op_registry


def _run_columns_and_wait(controller, operator_name, row_ids, parameters,
                          monkeypatch):
    """Start a create_columns run, join every worker thread it spawned,
    then pump the controller's drain by hand (no Qt event loop here)."""
    created: list[threading.Thread] = []
    real_thread = threading.Thread

    class _Tracked(real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    monkeypatch.setattr(threading, "Thread", _Tracked)
    try:
        controller.run_create_columns(operator_name, row_ids, parameters)
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)

    for thread in created:
        thread.join(timeout=10)
        assert not thread.is_alive(), "worker thread did not finish in time"

    # One tick applies the per-row results; a later tick lets the deferred
    # create_columns completion through.
    for _ in range(4):
        controller._drain_queues()
    return created


def _columns_mode(parameters):
    return ModeDescriptor(
        mode=ExecutionMode.COLUMNS,
        label="Run",
        inputs=(
            InputSpec(
                name="active_table",
                label="Active table",
                kind=InputKind.ACTIVE_TABLE,
            ),
        ),
        parameters=tuple(parameters),
        output=OutputSpec(
            columns=(OutputColumn(name="out", type_tag="numeric"),)
        ),
    )


class _FactorOperator(BaseOperator):
    """COLUMNS operator with one required numeric parameter, 'factor'.
    create_columns() writes the run's factor into every row and records
    the OperatorRun it was handed for each row, so a test can inspect
    both the run object and the parameters it carried."""

    name = "factor_op"
    create_columns_label = "Write factor"
    output_columns = [("out", "numeric")]
    descriptor = OperatorDescriptor(
        name="factor_op",
        version="1.0",
        description="Writes the run's 'factor' parameter into every row.",
        modes=(
            _columns_mode(
                (NumberParameter(
                    name="factor", label="Factor", minimum=0, maximum=1000
                ),)
            ),
        ),
    )

    def __init__(self):
        super().__init__()
        self.runs_seen: list = []

    def create_columns(self, row_id, media, metadata, run):
        self.runs_seen.append(run)
        return {"out": float(run.parameters["factor"])}


class _NoParamOperator(BaseOperator):
    """COLUMNS operator that declares no parameters at all."""

    name = "no_param_op"
    create_columns_label = "No params"
    output_columns = [("out", "numeric")]
    descriptor = OperatorDescriptor(
        name="no_param_op",
        version="1.0",
        description="Declares no parameters.",
        modes=(_columns_mode(()),),
    )

    def __init__(self):
        super().__init__()
        self.seen_empty: list[bool] = []

    def create_columns(self, row_id, media, metadata, run):
        self.seen_empty.append(dict(run.parameters) == {})
        return {"out": 1.0}


class _NoDescriptorOperator(BaseOperator):
    """A create_columns operator with NO descriptor -- not a runnable
    operator as of P1.12d-2a. The controller must refuse to start it."""

    name = "no_descriptor_op"
    create_columns_label = "No descriptor"
    output_columns = [("out", "numeric")]
    # descriptor deliberately left as BaseOperator's None.

    def create_columns(self, row_id, media, metadata, run):
        return {"out": 1.0}


# ---------------------------------------------------------------------------
# THE ONE THAT MATTERS
# ---------------------------------------------------------------------------

def test_two_runs_of_one_operator_hold_independent_parameters(
    tmp_path, monkeypatch
):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _FactorOperator()
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:3]

    # Capture the OperatorRun the controller builds and passes to the
    # registry, without disturbing the real run.
    captured: list = []
    real_run_columns = op_registry.run_create_columns

    def _wrap(*a, **k):
        captured.append(k.get("run"))
        return real_run_columns(*a, **k)

    monkeypatch.setattr(op_registry, "run_create_columns", _wrap)

    _run_columns_and_wait(controller, "factor_op", row_ids, {"factor": 2},
                          monkeypatch)
    _run_columns_and_wait(controller, "factor_op", row_ids, {"factor": 7},
                          monkeypatch)

    run_1, run_2 = captured

    # Each run carries its own frozen parameter set...
    assert run_1.parameters["factor"] == 2
    assert run_2.parameters["factor"] == 7
    # ...and building/running the second did not disturb the first.
    assert run_1.spec.parameters["factor"] == 2
    assert run_1 is not run_2

    # The operator is one shared singleton instance. It observed factor 2
    # on every row of the first run and factor 7 on every row of the
    # second, because each row's value came from that row's own run
    # object.
    seen_1 = op.runs_seen[: len(row_ids)]
    seen_2 = op.runs_seen[len(row_ids):]
    assert seen_1 and all(r is run_1 for r in seen_1)
    assert seen_2 and all(r is run_2 for r in seen_2)
    assert all(r.parameters["factor"] == 2 for r in seen_1)
    assert all(r.parameters["factor"] == 7 for r in seen_2)

    # Would this pass against the OLD singleton behaviour? No, on two
    # counts. There was no `run` argument -- create_columns() was
    # (self, row_id, image, metadata) -- so `op.runs_seen` and
    # `captured[i].parameters` simply did not exist. And the value the
    # operator read was self._factor, a single mutable slot on the shared
    # instance that the second dialog overwrote: both runs would report 7.


# ---------------------------------------------------------------------------
# Parameter validation at run start
# ---------------------------------------------------------------------------

def test_undeclared_parameter_does_not_start_a_run(tmp_path, monkeypatch):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_FactorOperator())
    row_ids = controller.get_visible_row_ids()[:3]

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    started: list = []
    monkeypatch.setattr(
        op_registry, "run_create_columns",
        lambda *a, **k: (started.append(True), True)[1],
    )

    controller.run_create_columns(
        "factor_op", row_ids, {"factor": 2, "bogus": 9}
    )

    # Would still pass if broken? No. Without the spec check the run would
    # start with a parameter the operator never asked for.
    assert started == [], "a run started despite an undeclared parameter"
    assert errors and "bogus" in errors[-1]
    assert controller._live_runs == {}


def test_missing_required_parameter_does_not_start_a_run(tmp_path, monkeypatch):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_FactorOperator())
    row_ids = controller.get_visible_row_ids()[:3]

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    started: list = []
    monkeypatch.setattr(
        op_registry, "run_create_columns",
        lambda *a, **k: (started.append(True), True)[1],
    )

    controller.run_create_columns("factor_op", row_ids, {})

    # Would still pass if broken? No. Without the required-parameter check
    # create_columns() would run and KeyError on run.parameters["factor"]
    # inside every worker row instead of failing cleanly at start.
    assert started == [], "a run started with a required parameter missing"
    assert errors and "factor" in errors[-1]
    assert controller._live_runs == {}


# ---------------------------------------------------------------------------
# RunData version
# ---------------------------------------------------------------------------

def test_run_data_reports_the_active_tables_current_version(
    tmp_path, monkeypatch
):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_FactorOperator())
    row_ids = controller.get_visible_row_ids()[:3]

    captured: list = []
    monkeypatch.setattr(
        op_registry, "run_create_columns",
        lambda *a, **k: (captured.append(k.get("run")), True)[1],
    )

    # The fake registry returns True without starting a worker, so nothing
    # writes to the table and its version does not move after the call.
    controller.run_create_columns("factor_op", row_ids, {"factor": 2})

    run = captured[0]
    assert run.data.versions() == {
        "frames": dataset.table_version("frames")
    }
    # Would still pass if broken? No -- a run that captured no snapshot,
    # or captured a hard-coded version, would not match table_version().


# ---------------------------------------------------------------------------
# The registry hands the operator the run the controller built
# ---------------------------------------------------------------------------

def test_registry_passes_the_operator_the_run_the_controller_built(
    tmp_path, monkeypatch
):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _FactorOperator()
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:2]

    captured: list = []
    real_run_columns = op_registry.run_create_columns

    def _wrap(*a, **k):
        captured.append(k.get("run"))
        return real_run_columns(*a, **k)

    monkeypatch.setattr(op_registry, "run_create_columns", _wrap)

    _run_columns_and_wait(controller, "factor_op", row_ids, {"factor": 3},
                          monkeypatch)

    assert op.runs_seen, "operator.create_columns() was never called"
    # The exact object the controller built reached the operator -- the
    # registry passed it straight through, it did not rebuild one.
    assert all(seen is captured[0] for seen in op.runs_seen)


# ---------------------------------------------------------------------------
# A run with no parameters
# ---------------------------------------------------------------------------

def test_a_run_with_no_parameters_works(tmp_path, monkeypatch):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _NoParamOperator()
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:3]

    _run_columns_and_wait(controller, "no_param_op", row_ids, {}, monkeypatch)

    assert op.seen_empty and all(op.seen_empty), (
        "operator saw a non-empty run.parameters for a no-parameter run"
    )
    stored = dataset.get_table("frames")
    assert "out" in stored.columns
    assert (stored["out"].dropna() == 1.0).any()


# ---------------------------------------------------------------------------
# STEP 7 deletion-check guard (amendment 5): an operator with no
# descriptor must fail to START, and the message must say so.
# ---------------------------------------------------------------------------

def test_operator_without_a_descriptor_does_not_start_and_says_so(
    tmp_path, monkeypatch
):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_NoDescriptorOperator())
    row_ids = controller.get_visible_row_ids()[:3]

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    started: list = []
    monkeypatch.setattr(
        op_registry, "run_create_columns",
        lambda *a, **k: (started.append(True), True)[1],
    )

    controller.run_create_columns("no_descriptor_op", row_ids, {})

    # Would still pass if the explicit `operator.descriptor is None` guard
    # in AppController._build_operator_run were removed? No. Without it
    # `operator.descriptor.mode_for(...)` raises a bare AttributeError
    # ("'NoneType' object has no attribute 'mode_for'"), which the outer
    # handler still surfaces as an error and still does not start a run --
    # but that message does not contain the word "descriptor", so the
    # last assertion here fails.
    assert started == [], "a run started for a descriptor-less operator"
    assert errors, "no error surfaced for the descriptor-less operator"
    assert "descriptor" in errors[-1].lower()
    assert controller._live_runs == {}


# ---------------------------------------------------------------------------
# MainWindow refuses a parameter dialog that does not hand its values back
# by name (STEP 3 / STEP 7 deletion check). Tested through the extracted
# MainWindow._parameters_from_dialog staticmethod so no widget is realised.
# ---------------------------------------------------------------------------

def test_mainwindow_rejects_a_dialog_with_no_parameter_values():
    class _BadDialog:
        pass  # no parameter_values() method

    # Would still pass if broken? No. Without the raise, MainWindow would
    # fall back to an empty parameter set and the run would start with no
    # parameters -- the wrong-number failure this item removes.
    with pytest.raises(RuntimeError):
        MainWindow._parameters_from_dialog(_BadDialog(), "some_op")


def test_mainwindow_reads_parameter_values_from_a_conforming_dialog():
    class _GoodDialog:
        def parameter_values(self):
            return {"factor": 5}

    assert MainWindow._parameters_from_dialog(
        _GoodDialog(), "some_op"
    ) == {"factor": 5}
