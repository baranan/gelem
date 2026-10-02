"""
tests/test_one_row_sequences.py

Which rows of a sequence run are one-row sequences (a still image, or a
``#t=`` time point) is decided by ONE pure function in
operators/operator_registry.py, ``address_forms_one_row_sequence``. The
runner and the controller's pre-run warning both use it.

  * the pure function's truth table;
  * ``AppController.count_one_row_sequence_rows`` -- the warning's count:
    non-zero only when tracking (the sequence option) is on;
  * an AST guard that the runner and the controller call the one function
    and that the controller does not repeat the decision inline.

Written from the work-item specification, not the implementation.

Run with:
    python -m pytest tests/test_one_row_sequences.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import pytest

from artifacts.artifact_store import ArtifactStore
from column_types.registry import ColumnTypeRegistry
from controller import AppController
from media.media_address import parse as parse_address
from media.resolver import MediaResolver
from models.dataset import Dataset
from models.query_engine import QueryEngine
from operators.base import BaseOperator
from operators.descriptor import (
    BooleanParameter,
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModeDescriptor,
    ModelLifecycle,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
    SequenceOption,
)
from operators.operator_registry import (
    OperatorRegistry,
    address_forms_one_row_sequence,
)


# ---------------------------------------------------------------------------
# The pure decision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        ("photo.jpg#f=0", True),      # a still's own frame address
        ("photo.png", True),          # a bare still
        ("clip.mp4#t=1.0", True),     # a time point on a video
        ("clip.mp4#t=1.0-2.0", True), # a time range on a video
        ("clip.mp4#f=12", False),     # a video frame: part of a sequence
        ("clip.mp4", False),          # whole video: not a photo, not counted
    ],
)
def test_address_forms_one_row_sequence_truth_table(value, expected):
    assert address_forms_one_row_sequence(parse_address(value)) is expected


# ---------------------------------------------------------------------------
# The controller's count
# ---------------------------------------------------------------------------

class _TrackingOp(BaseOperator):
    name = "TrackingOp"

    def __init__(self, media_requirement=MediaRequirement.FRAME):
        super().__init__()
        self.descriptor = OperatorDescriptor(
            name=self.name,
            version="1.0",
            description="Test double: tracking option.",
            modes=(
                ModeDescriptor(
                    mode=ExecutionMode.COLUMNS,
                    label="Tracking op",
                    inputs=(
                        InputSpec(
                            name="active_table", label="Active table",
                            kind=InputKind.ACTIVE_TABLE,
                        ),
                    ),
                    media_requirement=media_requirement,
                    parameters=(
                        BooleanParameter(
                            name="track", label="Track",
                            required=False, default=False,
                        ),
                    ),
                    output=OutputSpec(columns=(
                        OutputColumn(name="out", type_tag="numeric"),
                    )),
                    model_lifecycle=ModelLifecycle.PER_WORKER,
                    sequence_option=SequenceOption(enabled_by="track"),
                ),
            ),
        )


_MEDIA_VALUES = [
    "a.mp4#f=1",        # video frame
    "a.mp4#f=2",        # video frame
    "photo.jpg#f=0",    # still
    "b.mp4#t=1.5",      # time point
    "c.mp4",            # whole video: not counted
    "",                 # missing media: not counted
]


def _controller_with_media(tmp_path, operator):
    csv_path = tmp_path / "rows.csv"
    pd.DataFrame({"media": _MEDIA_VALUES}).to_csv(csv_path, index=False)
    store = ArtifactStore(
        tmp_path / "artifacts", resolver=MediaResolver(max_open_decoders=4)
    )
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)
    dataset = Dataset()
    op_registry = OperatorRegistry()
    op_registry.register(operator)
    controller = AppController(
        dataset, QueryEngine(), store, registry, op_registry,
        resolver=MediaResolver(max_open_decoders=4),
    )
    controller.load_csv_as_primary(csv_path, image_column="media")
    controller.set_filters([])
    table = dataset.get_table(controller.get_active_table())
    return controller, list(table["row_id"])


def test_count_is_nonzero_only_with_tracking_on(tmp_path):
    controller, row_ids = _controller_with_media(tmp_path, _TrackingOp())
    table = controller.get_active_table()
    # Tracking on: the still and the time point are counted; video frames,
    # the whole-video row and the empty row are not.
    assert controller.count_one_row_sequence_rows(
        "TrackingOp", "COLUMNS", table, row_ids, {"track": True}
    ) == 2
    # Tracking off: nothing is a sequence, so nothing is warned about.
    assert controller.count_one_row_sequence_rows(
        "TrackingOp", "COLUMNS", table, row_ids, {"track": False}
    ) == 0


def test_count_is_zero_when_only_video_frame_rows_are_chosen(tmp_path):
    controller, row_ids = _controller_with_media(tmp_path, _TrackingOp())
    table = controller.get_active_table()
    frames_only = row_ids[:2]
    assert controller.count_one_row_sequence_rows(
        "TrackingOp", "COLUMNS", table, frames_only, {"track": True}
    ) == 0


def test_count_only_covers_the_chosen_rows(tmp_path):
    controller, row_ids = _controller_with_media(tmp_path, _TrackingOp())
    table = controller.get_active_table()
    assert controller.count_one_row_sequence_rows(
        "TrackingOp", "COLUMNS", table, [row_ids[2]], {"track": True}
    ) == 1


def test_count_is_zero_for_an_unknown_operator(tmp_path):
    controller, row_ids = _controller_with_media(tmp_path, _TrackingOp())
    table = controller.get_active_table()
    assert controller.count_one_row_sequence_rows(
        "NoSuchOp", "COLUMNS", table, row_ids, {"track": True}
    ) == 0


# ---------------------------------------------------------------------------
# One decision, one place (AST guard)
# ---------------------------------------------------------------------------

def _parse(relative: str) -> ast.Module:
    return ast.parse((project_root / relative).read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_sequenced_runner_and_controller_call_the_one_function():
    registry_tree = _parse("operators/operator_registry.py")
    runner = _function(registry_tree, "_run_create_columns_sequenced")
    assert "address_forms_one_row_sequence" in _called_names(runner)

    controller_tree = _parse("controller.py")
    counter = _function(controller_tree, "count_one_row_sequence_rows")
    called = _called_names(counter)
    assert "address_forms_one_row_sequence" in called
    assert "runs_as_sequences" in called
    # The decision is not repeated inline in the controller.
    assert "is_image_path" not in called


def test_the_decision_is_not_repeated_inline_in_the_sequence_runner():
    runner = _function(
        _parse("operators/operator_registry.py"), "_run_create_columns_sequenced"
    )
    assert "is_image_path" not in _called_names(runner)
