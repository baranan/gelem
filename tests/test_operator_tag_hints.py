"""
tests/test_operator_tag_hints.py

P1.8d-2b-2: an operator's declared output_columns tags reach the target
table's TableSchema as ColumnHints on the accept path, instead of being
written into ColumnTypeRegistry's column-name map.

Written from the work-item specification. Each test states, in a comment,
what would still pass if the rule it guards were broken.

New file rather than added to tests/test_dataset.py, which runs its whole
body twice at import.

Run with:
    python -m pytest tests/test_operator_tag_hints.py
"""

from __future__ import annotations

import sys
import threading as _threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import pytest

from PySide6.QtWidgets import QApplication

from models.dataset import Dataset
from models.query_engine import QueryEngine
from models.table_schema import infer_type_tag
from artifacts.artifact_store import ArtifactStore
from column_types.registry import ColumnTypeRegistry
from operators.operator_registry import OperatorRegistry
from operators.base import BaseOperator
from controller import AppController

TEST_IMAGES = project_root / "test_images"


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    if QApplication.instance() is None:
        QApplication(sys.argv)


def _make_controller(tmp_path):
    """A real controller over the test_images 'frames' table."""
    store = ArtifactStore(tmp_path / "artifacts")
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)

    dataset = Dataset()
    dataset.load_folder(TEST_IMAGES)

    op_registry = OperatorRegistry()
    controller = AppController(
        dataset, QueryEngine(), store, registry, op_registry
    )
    controller.set_filters([])  # publish an initial result
    return controller, dataset, op_registry


def _run_operator_to_completion(controller, operator_name, row_ids):
    """Start a create_columns run, join its worker thread, then pump the
    controller's drain by hand -- there is no Qt event loop in the test."""
    created: list[_threading.Thread] = []
    real_thread = _threading.Thread

    class _Tracked(real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    original = _threading.Thread
    _threading.Thread = _Tracked
    try:
        controller.run_create_columns(operator_name, row_ids)
    finally:
        _threading.Thread = original

    assert created, "run_create_columns did not start a worker thread"
    created[-1].join(timeout=10)
    assert not created[-1].is_alive(), "worker thread did not finish in time"

    # Drain a few times: one tick applies the per-row results, the next
    # lets the deferred create_columns completion through.
    for _ in range(4):
        controller._drain_queues()


# ---------------------------------------------------------------------------
# Check 1 -- the operator's declared tag wins over value inference.
# ---------------------------------------------------------------------------

class _FlagDeclaringOperator(BaseOperator):
    # Declares "boolean_flag" for a column it fills with the integer 1.
    # infer_type_tag() on an all-integer column returns "numeric", so the
    # declared tag and the inferred tag genuinely disagree.
    name = "flag_declaring"
    create_columns_label = "Flag declaring"
    output_columns = [("mood_flag", "boolean_flag")]
    requires_image = False

    def create_columns(self, row_id, image, metadata):
        return {"mood_flag": 1}


def test_declared_tag_reaches_schema_and_beats_inference(tmp_path):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_FlagDeclaringOperator())
    row_ids = controller.get_visible_row_ids()[:5]

    _run_operator_to_completion(controller, "flag_declaring", row_ids)

    stored_df = dataset.get_table("frames")
    assert "mood_flag" in stored_df.columns

    # Sanity: value inference on the stored column would NOT pick
    # "boolean_flag" -- it is an integer column, so inference says "numeric".
    assert infer_type_tag(stored_df["mood_flag"]) == "numeric"

    # The schema must carry what the operator declared, not the inferred tag.
    spec = dataset.schema_for("frames").spec_for("mood_flag")
    assert spec.type_tag == "boolean_flag", (
        f"schema tag for mood_flag is {spec.type_tag!r}; the operator "
        f"declared 'boolean_flag' and that must reach the schema"
    )

    # Would still pass if broken? No. Without the hint plumbing the column
    # would be inferred from its integer values and tagged "numeric", and
    # this assertion would fail.


# ---------------------------------------------------------------------------
# Check 2 -- a tag the registry does not know still reaches the schema, the
#            run does not raise, and the controller warns once.
# ---------------------------------------------------------------------------

class _UnknownTagOperator(BaseOperator):
    name = "unknown_tag_op"
    create_columns_label = "Unknown tag"
    output_columns = [("odd_col", "no_such_tag_anywhere")]
    requires_image = False

    def create_columns(self, row_id, image, metadata):
        return {"odd_col": "some text"}


def test_unknown_tag_reaches_schema_without_raising(tmp_path, capsys):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_UnknownTagOperator())
    row_ids = controller.get_visible_row_ids()[:5]

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    # Sanity: the registry really does not know this tag.
    assert controller._registry.type_for_tag("no_such_tag_anywhere") is None

    _run_operator_to_completion(controller, "unknown_tag_op", row_ids)

    # (a) the tag reached the schema unchanged -- the schema does not police tags.
    spec = dataset.schema_for("frames").spec_for("odd_col")
    assert spec.type_tag == "no_such_tag_anywhere"

    # (b) nothing raised into the caller: no error_occurred, column stored.
    assert errors == [], errors
    assert "odd_col" in dataset.get_table("frames").columns

    # (c) the controller warned exactly once, naming the tag, at the same
    #     point register_by_tag's KeyError used to be caught.
    warnings = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("[Controller] Warning:")
    ]
    assert len(warnings) == 1, warnings
    assert "no_such_tag_anywhere" in warnings[0]

    # Would still pass if broken? No. If the tag were still routed through
    # register_by_tag the run path would swallow a KeyError per column and
    # nothing would land in the schema; if the warning moved or doubled,
    # the count assertion fails.


# ---------------------------------------------------------------------------
# Follow-up round -- an operator that declares 'media_path' for a column it
# fills with a path to a PNG it wrote gets that tag in the schema, and the
# controller prints NO unknown-tag warning. This is the shape of the two
# bundled operators PlotOperator ('plot_path') and BlendshapeAvatarOperator
# ('avatar_path') after the fix.
# ---------------------------------------------------------------------------

def test_media_path_declaring_operator_tags_schema_and_warns_nothing(
    tmp_path, capsys
):
    controller, dataset, op_registry = _make_controller(tmp_path)

    out_dir = tmp_path / "charts"
    out_dir.mkdir()

    class _ChartOperator(BaseOperator):
        # Declares 'media_path' and fills the column with the path to a real
        # PNG on disk -- exactly what PlotOperator / BlendshapeAvatarOperator
        # do. 'media_path' IS a registered type, so no warning is expected.
        name = "chart_op"
        create_columns_label = "Chart"
        output_columns = [("chart_path", "media_path")]
        requires_image = False

        def create_columns(self, row_id, image, metadata):
            from PIL import Image
            path = out_dir / f"{row_id}_chart.png"
            Image.new("RGB", (16, 16), color=(10, 20, 30)).save(path, "PNG")
            return {"chart_path": str(path)}

    op_registry.register(_ChartOperator())
    row_ids = controller.get_visible_row_ids()[:5]

    # Clear registration noise so the warning check below only sees the run.
    capsys.readouterr()

    _run_operator_to_completion(controller, "chart_op", row_ids)

    spec = dataset.schema_for("frames").spec_for("chart_path")
    assert spec.type_tag == "media_path", (
        f"schema tag for chart_path is {spec.type_tag!r}; the operator "
        f"declared 'media_path' and that must reach the schema"
    )

    warnings = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("[Controller] Warning:")
    ]
    assert warnings == [], (
        f"a run declaring only registered tags must print no unknown-tag "
        f"warning; got: {warnings}"
    )

    # Would still pass if broken? No. If 'media_path' stopped being applied
    # as a hint the column would still infer to 'media_path' from the .png
    # paths here -- so the discriminating half of this test is the warning
    # assertion: declaring an unregistered tag (the pre-fix state) prints a
    # line, declaring 'media_path' must not.


# ---------------------------------------------------------------------------
# Check -- STEP 3: add_column / add_computed_column col_type is not dead.
# ---------------------------------------------------------------------------

def test_add_computed_column_col_type_reaches_schema(tmp_path):
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    row_ids = list(ds.get_table("frames")["row_id"])

    # A numeric base column to compute from (a bare folder load has none).
    ds.add_column(
        "base", pd.Series({rid: float(i) for i, rid in enumerate(row_ids)}),
        "numeric", table_name="frames",
    )

    # The expression is numeric; value inference would tag the new column
    # "numeric". Ask for "text" instead and it must be honoured.
    ds.add_computed_column(
        "ts_label", "base * 2", col_type="text", table_name="frames"
    )
    spec = ds.schema_for("frames").spec_for("ts_label")
    assert spec.type_tag == "text", (
        f"add_computed_column col_type='text' produced schema tag "
        f"{spec.type_tag!r}"
    )


def test_add_column_col_type_reaches_schema(tmp_path):
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    row_ids = list(ds.get_table("frames")["row_id"])

    # Integer values -> inference would say "numeric". Ask for "boolean_flag".
    values = pd.Series({rid: 1 for rid in row_ids})
    ds.add_column("flagged", values, "boolean_flag", table_name="frames")
    spec = ds.schema_for("frames").spec_for("flagged")
    assert spec.type_tag == "boolean_flag"


def test_recompute_keeps_stored_tag(tmp_path):
    # A column that already exists keeps its stored spec: the stored schema
    # wins in _prepare_table, so a second col_type is ignored.
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    row_ids = list(ds.get_table("frames")["row_id"])
    ds.add_column(
        "base", pd.Series({rid: float(i) for i, rid in enumerate(row_ids)}),
        "numeric", table_name="frames",
    )

    ds.add_computed_column(
        "again", "base * 2", col_type="text", table_name="frames"
    )
    assert ds.schema_for("frames").spec_for("again").type_tag == "text"

    ds.add_computed_column(
        "again", "base * 3", col_type="numeric", table_name="frames"
    )
    assert ds.schema_for("frames").spec_for("again").type_tag == "text", (
        "a recompute must not re-tag an existing column"
    )
