"""
tests/test_media_requirement.py

P1.12d-2b-1: the create_columns runner decides what to hand the operator
from the run's declared ``media_requirement`` (operators/descriptor.py ->
MediaRequirement), and the old boolean ``requires_image`` is gone.

  * FRAME    -- the operator is handed a decoded frame;
  * METADATA -- the operator is handed None;
  * ADDRESS  -- the operator is handed None (it resolves the media itself);
  * VIDEO_SPAN / AUDIO_SPAN -- the per-row runner cannot supply a span, so
    the run is refused before it starts, in AppController, with a message
    naming the operator and the requirement. It never reaches the worker.

Written from the work-item specification, not the implementation. Each
test says, in a comment, what would still pass if the rule it guards were
broken.

No Qt widget is shown here (a real AppController is built the way
tests/test_operator_run_wiring.py does), so the run-tests substring
heuristic puts this module in the combined group, which is correct. No
`# run-tests:` token is needed.

Run with:
    python -m pytest tests/test_media_requirement.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

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
    MediaRequirement,
    ModeDescriptor,
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
    no-widget construction tests/test_operator_run_wiring.py uses."""
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


def _columns_descriptor(name, label, media_requirement):
    """A one-mode COLUMNS descriptor declaring a given media_requirement.
    The single output column is 'out' / numeric."""
    return OperatorDescriptor(
        name=name,
        version="1.0",
        description=f"Test double declaring {media_requirement.name}.",
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.COLUMNS,
                label=label,
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
            ),
        ),
    )


class _RecordingOperator(BaseOperator):
    """A COLUMNS operator that records the ``image`` argument it is handed
    for every row. The descriptor -- and therefore the declared
    media_requirement -- is supplied per instance so one class covers the
    FRAME, METADATA and ADDRESS cases."""

    name = "recording_op"
    create_columns_label = "Recording"
    output_columns = [("out", "numeric")]

    def __init__(self, media_requirement):
        super().__init__()
        self.name = f"recording_op_{media_requirement.name.lower()}"
        self.create_columns_label = "Recording"
        self.descriptor = _columns_descriptor(
            self.name, "Recording", media_requirement
        )
        self.images_seen: list = []

    def create_columns(self, row_id, image, metadata, run):
        self.images_seen.append(image)
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
    # completion through.
    for _ in range(4):
        controller._drain_queues()
    return created


# ---------------------------------------------------------------------------
# FRAME -- the operator receives a decoded image
# ---------------------------------------------------------------------------

def test_frame_mode_run_receives_a_decoded_image(tmp_path, monkeypatch):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _RecordingOperator(MediaRequirement.FRAME)
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:3]

    _run_columns_and_wait(controller, op.name, row_ids, monkeypatch)

    # Every row of a FRAME run is handed a real decoded frame: an RGB uint8
    # array of shape (H, W, 3).
    assert op.images_seen, "create_columns() was never called"
    assert len(op.images_seen) == len(row_ids)
    for image in op.images_seen:
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3 and image.shape[2] == 3
        assert image.dtype == np.uint8

    # Would still pass if the runner ignored media_requirement and handed
    # None (the METADATA behaviour)? No -- the isinstance check fails.


# ---------------------------------------------------------------------------
# METADATA -- the operator receives None
# ---------------------------------------------------------------------------

def test_metadata_mode_run_receives_none(tmp_path, monkeypatch):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _RecordingOperator(MediaRequirement.METADATA)
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:3]

    _run_columns_and_wait(controller, op.name, row_ids, monkeypatch)

    assert op.images_seen, "create_columns() was never called"
    assert len(op.images_seen) == len(row_ids)
    assert all(image is None for image in op.images_seen)

    # Would still pass if the runner decoded a frame anyway? No -- the rows
    # have real image paths, so a decode would put an ndarray here.


# ---------------------------------------------------------------------------
# ADDRESS -- the operator receives None (it resolves the media itself)
# ---------------------------------------------------------------------------

def test_address_mode_run_receives_none(tmp_path, monkeypatch):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _RecordingOperator(MediaRequirement.ADDRESS)
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:3]

    _run_columns_and_wait(controller, op.name, row_ids, monkeypatch)

    assert op.images_seen, "create_columns() was never called"
    assert len(op.images_seen) == len(row_ids)
    assert all(image is None for image in op.images_seen)

    # Would still pass if ADDRESS were treated like FRAME? No -- a decode
    # would put an ndarray here. ADDRESS means the runner decodes nothing;
    # the operator walks the address from its metadata.


# ---------------------------------------------------------------------------
# VIDEO_SPAN / AUDIO_SPAN -- refused before the run starts
# ---------------------------------------------------------------------------

def _assert_span_run_is_refused(tmp_path, monkeypatch, requirement):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _RecordingOperator(requirement)
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:3]

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    # If the runner is reached at all, record it -- it must not be.
    started: list = []
    monkeypatch.setattr(
        op_registry, "run_create_columns",
        lambda *a, **k: (started.append(True), True)[1],
    )

    controller.run_create_columns(op.name, row_ids)

    assert started == [], (
        f"a {requirement.name} run reached the worker; it must be refused "
        f"before it starts"
    )
    assert op.images_seen == [], "create_columns() ran for a span run"
    assert errors, f"no error surfaced for the {requirement.name} run"
    message = errors[-1]
    # The message names the operator...
    assert op.display_label in message, message
    # ...and the requirement that cannot be supplied.
    assert requirement.name in message, message
    # The run left no live-run entry behind.
    assert controller._live_runs == {}


def test_video_span_declaration_does_not_start_a_run(tmp_path, monkeypatch):
    # Would still pass if AppController silently handed the operator None
    # for a VIDEO_SPAN run? No -- the run would start, `started` would be
    # non-empty and there would be no error naming the requirement.
    _assert_span_run_is_refused(
        tmp_path, monkeypatch, MediaRequirement.VIDEO_SPAN
    )


def test_audio_span_declaration_does_not_start_a_run(tmp_path, monkeypatch):
    # Same guarantee for AUDIO_SPAN.
    _assert_span_run_is_refused(
        tmp_path, monkeypatch, MediaRequirement.AUDIO_SPAN
    )


# ---------------------------------------------------------------------------
# The boolean is gone
# ---------------------------------------------------------------------------

def test_base_operator_has_no_requires_image_attribute():
    # P1.12d-2b-1 deletes the class attribute outright -- it is not
    # renamed, not defaulted. hasattr walks the MRO, so this also catches a
    # stray definition on BaseOperator itself.
    assert not hasattr(BaseOperator, "requires_image")

    # And a plain concrete operator does not carry one either.
    class _Plain(BaseOperator):
        name = "plain_op"

    assert not hasattr(_Plain(), "requires_image")
