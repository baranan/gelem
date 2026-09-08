"""
tests/test_model_lifecycle.py

P1.12d-2b-2: the create_columns runner honours the mode's declared
``model_lifecycle`` (operators/descriptor.py -> ModelLifecycle). The
operator no longer builds or caches a model itself -- it provides a
factory, ``build_model()``, and reads ``run.model``.

  * NONE         -- no model is built; ``run.model`` is None.
  * PER_WORKER   -- built once per run, inside the worker, before the row
                    loop; the same instance serves every row of that run,
                    and two concurrent runs get different instances.
  * SHARED       -- built once for the application, cached on
                    OperatorRegistry under a lock, reused across runs.
  * PER_SEQUENCE -- the per-row runner cannot honour it; the run is
                    refused before it starts, in AppController, with a
                    message naming the operator and the lifecycle.

If ``build_model()`` raises OperatorSetupError the run aborts through
``on_setup_error`` having processed no rows -- including the case where
every row would have failed to decode, which the old lazy load inside
create_columns swallowed.

Written from the work-item specification, not the implementation. Each
test says, in a comment, what would still pass if the rule it guards were
broken.

No Qt widget is shown here (a real AppController is built the way
tests/test_media_requirement.py does), so the run-tests substring
heuristic puts this module in the combined group, which is correct. No
``# run-tests:`` token is needed.

Run with:
    python -m pytest tests/test_model_lifecycle.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.dataset import Dataset
from models.query_engine import QueryEngine
from artifacts.artifact_store import ArtifactStore
from column_types.registry import ColumnTypeRegistry
from operators.operator_registry import OperatorRegistry
from operators.base import BaseOperator, OperatorSetupError
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModeDescriptor,
    ModelLifecycle,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
)
from controller import AppController

TEST_IMAGES = project_root / "test_images"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_controller(tmp_path):
    """A real AppController over the test_images 'frames' table -- the same
    no-widget construction tests/test_media_requirement.py uses."""
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


def _columns_descriptor(name, lifecycle, media_requirement):
    """A one-mode COLUMNS descriptor declaring a given model_lifecycle and
    media_requirement. The single output column is 'out' / numeric."""
    return OperatorDescriptor(
        name=name,
        version="1.0",
        description=f"Test double: model_lifecycle={lifecycle.name}.",
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.COLUMNS,
                label="Model op",
                inputs=(
                    InputSpec(
                        name="active_table",
                        label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                media_requirement=media_requirement,
                parameters=(),
                output=OutputSpec(
                    columns=(OutputColumn(name="out", type_tag="numeric"),)
                ),
                model_lifecycle=lifecycle,
            ),
        ),
    )


class _ModelOperator(BaseOperator):
    """A COLUMNS operator whose model_lifecycle is set per instance.

    ``build_model()`` records the call and returns a fresh sentinel object
    (or raises, if ``build_error`` was supplied). ``create_columns()``
    records the ``run.model`` it was handed for every row. ``events`` is
    the interleaved order of build/row calls, so a test can assert the
    model was built before the first row.
    """

    def __init__(
        self,
        lifecycle,
        *,
        media_requirement=MediaRequirement.METADATA,
        build_error=None,
    ):
        super().__init__()
        self.name = "model_op"
        self.create_columns_label = "Model op"
        self.output_columns = [("out", "numeric")]
        self.descriptor = _columns_descriptor(
            self.name, lifecycle, media_requirement
        )
        self._build_error = build_error
        self.events: list[str] = []          # ordered "build" / "row"
        self.built_models: list = []          # every object build_model returned
        self.models_seen: list = []           # run.model for every row

    def build_model(self):
        self.events.append("build")
        if self._build_error is not None:
            raise self._build_error
        model = object()
        self.built_models.append(model)
        return model

    def create_columns(self, row_id, media, metadata, run):
        self.events.append("row")
        self.models_seen.append(run.model)
        return {"out": 1.0}


def _run_columns_and_wait(controller, operator_name, row_ids, monkeypatch):
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
        controller.run_create_columns(operator_name, row_ids)
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)

    for thread in created:
        thread.join(timeout=10)
        assert not thread.is_alive(), "worker thread did not finish in time"

    # One tick applies the per-row results; a later tick lets the deferred
    # completion (and any setup-error message) through.
    for _ in range(4):
        controller._drain_queues()
    return created


# ---------------------------------------------------------------------------
# NONE -- no model is built, run.model is None
# ---------------------------------------------------------------------------

def test_none_lifecycle_builds_no_model_and_hands_the_operator_none(
    tmp_path, monkeypatch
):
    controller, _dataset, op_registry = _make_controller(tmp_path)
    op = _ModelOperator(ModelLifecycle.NONE)
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:3]

    _run_columns_and_wait(controller, op.name, row_ids, monkeypatch)

    assert op.events.count("build") == 0, (
        "build_model() was called for a NONE-lifecycle run"
    )
    assert len(op.models_seen) == len(row_ids), "not every row was processed"
    assert all(model is None for model in op.models_seen)

    # Would still pass if the runner built a model anyway? No -- a build
    # would be counted and models_seen would hold objects, not None.


# ---------------------------------------------------------------------------
# PER_WORKER -- built once per run, before the first row
# ---------------------------------------------------------------------------

def test_per_worker_builds_once_before_the_first_row_and_every_row_sees_it(
    tmp_path, monkeypatch
):
    controller, _dataset, op_registry = _make_controller(tmp_path)
    op = _ModelOperator(ModelLifecycle.PER_WORKER)
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:4]

    _run_columns_and_wait(controller, op.name, row_ids, monkeypatch)

    # Exactly one build for the run...
    assert op.events.count("build") == 1
    # ...and it came before any row was processed.
    assert op.events[0] == "build"
    assert op.events[1:] == ["row"] * len(row_ids)
    # Every row saw the one built instance.
    assert len(op.built_models) == 1
    assert op.models_seen == [op.built_models[0]] * len(row_ids)

    # Would still pass if the runner built the model per row? No -- events
    # would interleave build/row and built_models would have len 4.


def test_per_worker_isolation_two_runs_get_different_model_objects(
    tmp_path, monkeypatch
):
    controller, _dataset, op_registry = _make_controller(tmp_path)
    op = _ModelOperator(ModelLifecycle.PER_WORKER)
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:2]

    _run_columns_and_wait(controller, op.name, row_ids, monkeypatch)
    first_models = list(op.models_seen)

    _run_columns_and_wait(controller, op.name, row_ids, monkeypatch)
    second_models = op.models_seen[len(first_models):]

    assert op.events.count("build") == 2, "expected exactly one build per run"
    assert len({id(model) for model in first_models}) == 1
    assert len({id(model) for model in second_models}) == 1
    assert first_models[0] is not second_models[0], (
        "two PER_WORKER runs shared a model instance -- the per-run "
        "isolation this item establishes is gone"
    )

    # This is the test that would have passed vacuously before P1.12d-2b-2:
    # the operator kept ONE landmarker on self and reused it for every run,
    # so first_models[0] would BE second_models[0].


# ---------------------------------------------------------------------------
# SHARED -- built once for the application, reused across runs
# ---------------------------------------------------------------------------

def test_shared_builds_once_across_two_runs_and_both_see_the_same_object(
    tmp_path, monkeypatch
):
    controller, _dataset, op_registry = _make_controller(tmp_path)
    op = _ModelOperator(ModelLifecycle.SHARED)
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:2]

    _run_columns_and_wait(controller, op.name, row_ids, monkeypatch)
    _run_columns_and_wait(controller, op.name, row_ids, monkeypatch)

    assert op.events.count("build") == 1, "the SHARED model was built more than once"
    assert len(op.models_seen) == 2 * len(row_ids)
    assert len({id(model) for model in op.models_seen}) == 1
    assert op.models_seen[0] is op.built_models[0]

    # Would still pass if SHARED were treated like PER_WORKER? No -- build
    # would run twice and the two runs would see different objects.


# ---------------------------------------------------------------------------
# PER_SEQUENCE -- refused before the run starts
# ---------------------------------------------------------------------------

def test_per_sequence_run_does_not_start_and_message_names_operator_and_lifecycle(
    tmp_path, monkeypatch
):
    controller, _dataset, op_registry = _make_controller(tmp_path)
    op = _ModelOperator(ModelLifecycle.PER_SEQUENCE)
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:3]

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    # If the registry is reached at all, record it -- it must not be.
    started: list = []
    monkeypatch.setattr(
        op_registry, "run_create_columns",
        lambda *a, **k: (started.append(True), True)[1],
    )

    controller.run_create_columns(op.name, row_ids)

    assert started == [], (
        "a PER_SEQUENCE run reached the registry; it must be refused before "
        "it starts"
    )
    assert op.events == [], "build_model() or create_columns() ran for a refused run"
    assert errors, "no error surfaced for the PER_SEQUENCE run"
    message = errors[-1]
    assert op.display_label in message, message
    assert ModelLifecycle.PER_SEQUENCE.name in message, message
    # The run left no live-run entry behind.
    assert controller._live_runs == {}

    # Would still pass if AppController silently handed the operator a
    # PER_WORKER-built model instead? No -- the run would start, `started`
    # would be non-empty and there would be no error naming the lifecycle.


# ---------------------------------------------------------------------------
# build_model() raising OperatorSetupError -- abort with zero rows
# ---------------------------------------------------------------------------

def test_build_model_setup_error_aborts_the_run_with_zero_rows_processed(
    tmp_path, monkeypatch
):
    controller, _dataset, op_registry = _make_controller(tmp_path)
    op = _ModelOperator(
        ModelLifecycle.PER_WORKER,
        build_error=OperatorSetupError("model file missing (test)"),
    )
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:3]

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    _run_columns_and_wait(controller, op.name, row_ids, monkeypatch)

    assert op.models_seen == [], "a row was processed after build_model() failed"
    assert op.events == ["build"], "the run continued past the failed build"
    assert errors, "the setup error never reached the researcher"
    assert "model file missing (test)" in errors[-1]
    # The aborted run left no live-run entry behind.
    assert controller._live_runs == {}

    # Would still pass if the runner ignored a build failure and ran the
    # rows anyway? No -- models_seen would be non-empty (rows processed
    # with run.model still None) and there would be no error.


def test_an_unexpected_build_error_also_aborts_the_run_and_deregisters_it(
    tmp_path, monkeypatch
):
    # build_model() is documented to raise OperatorSetupError, but a
    # corrupt model file can make the underlying library raise something
    # else (RuntimeError, ValueError). That must still tear the run down --
    # surface a message and deregister -- not kill the worker thread and
    # leave the run live forever.
    controller, _dataset, op_registry = _make_controller(tmp_path)
    op = _ModelOperator(
        ModelLifecycle.PER_WORKER,
        build_error=RuntimeError("landmarker init blew up (test)"),
    )
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:3]

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    _run_columns_and_wait(controller, op.name, row_ids, monkeypatch)

    assert op.models_seen == [], "a row was processed after the build failed"
    assert errors, "an unexpected build error was swallowed"
    assert "landmarker init blew up (test)" in errors[-1]
    assert "RuntimeError" in errors[-1]
    assert controller._live_runs == {}, "the aborted run was left live"

    # Would still pass if only OperatorSetupError were caught around the
    # build? No -- the RuntimeError would escape the worker, no callback
    # would fire, and _live_runs would still hold the run.


def test_setup_error_surfaces_even_when_every_row_would_fail_to_decode(
    tmp_path, monkeypatch
):
    # Amendment 3 / the answer to STEP 4's question. The pre-P1.12d-2b-2
    # lazy load raised OperatorSetupError from inside create_columns, which
    # the runner only calls for a row whose image decoded. A run where
    # every row failed to decode therefore completed "successfully" having
    # processed zero rows, never mentioning the missing model. Building the
    # model before the row loop fixes that unconditionally.
    controller, _dataset, op_registry = _make_controller(tmp_path)
    op = _ModelOperator(
        ModelLifecycle.PER_WORKER,
        media_requirement=MediaRequirement.FRAME,
        build_error=OperatorSetupError("model file missing (test)"),
    )
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:3]

    # Every row fails to decode.
    monkeypatch.setattr(op, "load_image", lambda *a, **k: None)

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    _run_columns_and_wait(controller, op.name, row_ids, monkeypatch)

    assert op.models_seen == []
    assert errors, "the missing-model error was swallowed when no row decoded"
    assert "model file missing (test)" in errors[-1]

    # Would still pass under the old lazy-load behaviour? No -- load_image
    # returns None for every row, so create_columns (and the lazy loader
    # it used to hold) was never reached, and the run completed clean.


# ---------------------------------------------------------------------------
# The base-class factory
# ---------------------------------------------------------------------------

def test_base_operator_build_model_returns_none():
    assert BaseOperator().build_model() is None

    class _Plain(BaseOperator):
        name = "plain_op"

    assert _Plain().build_model() is None
