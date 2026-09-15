"""
tests/test_run_carry_columns.py

P1.6a: AppController._build_operator_run computes each declared
ACTIVE_TABLE input's carry set (Dataset.columns_to_carry(),
docs/architecture.md §4.2) on the main thread and puts it on that
input's frozen TableSnapshot as carry_columns, so a splitting operator
(P1.6b, not built here) can read it without ever touching Dataset.

Written from the work item and §4.2, not from the implementation. Real
Dataset + AppController, no Qt widget shown -- the same construction
tests/test_operator_run_wiring.py and the P1.12f-1 section of
tests/test_result_delivery.py use for _build_operator_run.

Run with:
    python -m pytest tests/test_run_carry_columns.py
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from models.dataset import Dataset
from models.query_engine import QueryEngine
from models.table_schema import ColumnRole, ColumnSpec, TableSchema
from artifacts.artifact_store import ArtifactStore
from column_types.registry import ColumnTypeRegistry
from operators.operator_registry import OperatorRegistry
from operators.base import BaseOperator
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    ModeDescriptor,
    OperatorDescriptor,
    OutputSpec,
)
from controller import AppController


def _table_with_explicit_roles(ds: Dataset) -> None:
    """One column of each role, one flagged measurement and one unflagged
    one -- built with an explicit schema so the test controls every role
    directly. The same fixture tests/test_column_roles.py uses to test
    Dataset.columns_to_carry() itself, so a failure here that is not also
    a failure there means the break is in the controller's wiring, not in
    Dataset."""
    df = pd.DataFrame(
        {
            "row_id": ["1", "2"],
            "participant_id": ["p07", "p08"],
            "trial_id": pd.Series([1, 2], dtype="int64"),
            "reaction_time": pd.Series([0.5, 0.6], dtype="float64"),
            "blendshape_score": pd.Series([0.1, 0.2], dtype="float64"),
        }
    )
    schema = TableSchema(
        columns=(
            ColumnSpec("participant_id", "text", "object", ColumnRole.identifier, True),
            ColumnSpec("trial_id", "numeric", "int64", ColumnRole.index, True),
            ColumnSpec("reaction_time", "numeric", "float64", ColumnRole.measurement, True),
            ColumnSpec("blendshape_score", "numeric", "float64", ColumnRole.measurement, False),
        )
    )
    ds._accept_table("source", df, schema=schema, source="test")


def _probe_descriptor():
    """A minimal TABLE-mode descriptor declaring one ACTIVE_TABLE input
    named 'active_table' -- the input _build_operator_run looks for when
    it builds the carried snapshot."""
    return OperatorDescriptor(
        name="probe",
        version="1.0",
        description="Test double for P1.6a.",
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.TABLE,
                label="Probe",
                inputs=(
                    InputSpec(
                        name="active_table",
                        label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                parameters=(),
                output=OutputSpec(creates_table=True),
            ),
        ),
    )


class _ProbeOperator(BaseOperator):
    """Never actually run here -- only its descriptor is consulted by
    _build_operator_run."""

    name = "probe"
    descriptor = _probe_descriptor()


def _make_controller(tmp_path):
    """A real AppController over a Dataset holding only the 'source'
    table built by _table_with_explicit_roles. No test_images folder and
    no worker thread: this item never starts a run, only builds one."""
    store = ArtifactStore(tmp_path / "artifacts")
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)

    dataset = Dataset()
    _table_with_explicit_roles(dataset)

    op_registry = OperatorRegistry()
    controller = AppController(
        dataset, QueryEngine(), store, registry, op_registry
    )
    return controller, dataset


def _carry_columns_for(controller, dataset):
    """Build a run through _build_operator_run exactly as
    run_create_columns/run_create_table do, and return the carry_columns
    tuple the controller put on the 'source' input's snapshot."""
    run, _token = controller._build_operator_run(
        _ProbeOperator(),
        ExecutionMode.TABLE,
        "op-1",
        "source",
        {},
        dataset.get_table("source"),
    )
    return run.data.snapshot("active_table").carry_columns


def test_carry_columns_includes_identifier_and_index_columns(tmp_path):
    controller, dataset = _make_controller(tmp_path)
    carried = _carry_columns_for(controller, dataset)

    assert "participant_id" in carried
    assert "trial_id" in carried
    # Would still pass if the controller never called columns_to_carry()
    # at all? No -- TableSnapshot.carry_columns defaults to ()
    # (operators/run_context.py), so an untouched snapshot fails this.


def test_carry_columns_includes_a_flagged_measurement(tmp_path):
    controller, dataset = _make_controller(tmp_path)
    carried = _carry_columns_for(controller, dataset)

    assert "reaction_time" in carried
    # Would still pass if the controller carried only identifier/index
    # columns, ignoring carry_to_children measurements? No.


def test_carry_columns_excludes_an_unflagged_measurement(tmp_path):
    controller, dataset = _make_controller(tmp_path)
    carried = _carry_columns_for(controller, dataset)

    assert "blendshape_score" not in carried
    # Would still pass if the controller carried every column regardless
    # of carry_to_children? No -- this is the one column in the fixture
    # with carry_to_children False, and it must stay dropped.


def test_carry_columns_is_in_schema_order(tmp_path):
    controller, dataset = _make_controller(tmp_path)
    carried = _carry_columns_for(controller, dataset)

    assert carried == ("participant_id", "trial_id", "reaction_time")
    # Pins both the schema order Dataset.columns_to_carry() promises and
    # that the controller stores exactly that sequence, not a re-sorted
    # or differently-ordered one.
