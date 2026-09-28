"""
tests/test_rerun_choice.py

P2.2a: when a COLUMNS run's declared output columns already exist on the
active table, the researcher is asked before it starts -- overwrite every
chosen row, fill in only the rows still empty, or cancel. Gelem keeps no
per-row "done" record of its own; this choice is how the researcher
decides each time.

Covers, at the public seams named in the work item:
  - Dataset.empty_rows_for_columns() -- which chosen rows are empty.
  - Dataset.most_recent_operator_run() -- provenance lookup for the hint.
  - Dataset.record_operator_run(operator_version=...).
  - Dataset.clear_output_columns() -- the "Overwrite all" clearing step.
  - AppController.describe_existing_outputs() -- the Qt-free check and
    message the dialog is built from.
  - AppController.run_create_columns(fill_only_empty=...,
    clear_existing_outputs=...).

Written from the work-item specification, not the implementation. Each
test says, in a comment, what would still pass -- or fail -- if the rule
it guards were broken.

No Qt widget is shown here (a real AppController is built the way
tests/test_operator_run_wiring.py does), so the run-tests substring
heuristic puts this module in the combined group, which is correct. No
`# run-tests:` token is needed.

Run with:
    python -m pytest tests/test_rerun_choice.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
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
    from media.resolver import MediaResolver

    store = ArtifactStore(tmp_path / "artifacts", resolver=MediaResolver(max_open_decoders=4))
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)

    dataset = Dataset()
    dataset.load_folder(TEST_IMAGES)

    op_registry = OperatorRegistry()
    controller = AppController(
        dataset, QueryEngine(), store, registry, op_registry,
        resolver=MediaResolver(max_open_decoders=4),
    )
    controller.set_filters([])  # publish an initial query result
    return controller, dataset, op_registry


def _run_columns_and_wait(controller, operator_name, row_ids, parameters,
                          monkeypatch, **kwargs):
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
        controller.run_create_columns(operator_name, row_ids, parameters, **kwargs)
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


def _operator_run_entries(dataset):
    return [e for e in dataset.provenance.to_list() if e["action"] == "operator_run"]


def _out_value(dataset, table_name, row_id, column="out"):
    table = dataset.get_table(table_name)
    return table.loc[table["row_id"] == row_id, column].iloc[0]


def _columns_mode(*, media_requirement, label="Run"):
    return ModeDescriptor(
        mode=ExecutionMode.COLUMNS,
        label=label,
        inputs=(
            InputSpec(
                name="active_table", label="Active table",
                kind=InputKind.ACTIVE_TABLE,
            ),
        ),
        media_requirement=media_requirement,
        output=OutputSpec(columns=(OutputColumn(name="out", type_tag="numeric"),)),
    )


class _WriteFixedValueOperator(BaseOperator):
    """METADATA-mode operator: writes a fixed value into 'out' for every
    row it actually processes, and records which rows those were --
    no media dependency, so this is the simple case for narrowing tests."""

    def __init__(self, name="write_fixed_value_op", value=999.0, version="1.0"):
        super().__init__()
        self.name = name
        self._value = value
        self.descriptor = OperatorDescriptor(
            name=name,
            version=version,
            description="Writes a fixed value into 'out' for every row it processes.",
            modes=(_columns_mode(media_requirement=MediaRequirement.METADATA),),
        )
        self.rows_seen: list[str] = []

    def create_columns(self, row_id, media, metadata, run):
        self.rows_seen.append(row_id)
        return {"out": self._value}


class _FrameEchoOperator(BaseOperator):
    """FRAME-mode operator: writes a fixed value into 'out' for every row
    whose media actually decodes. A row with no media value never reaches
    create_columns() at all (operators/operator_registry.py's MissingMedia
    path) -- the case the clear_output_columns fix is about."""

    def __init__(self, name="frame_echo_op", value=42.0):
        super().__init__()
        self.name = name
        self._value = value
        self.descriptor = OperatorDescriptor(
            name=name,
            version="1.0",
            description="Writes a fixed value into 'out' for every row whose media decodes.",
            modes=(_columns_mode(media_requirement=MediaRequirement.FRAME),),
        )
        self.rows_seen: list[str] = []

    def create_columns(self, row_id, media, metadata, run):
        self.rows_seen.append(row_id)
        return {"out": self._value}


# ---------------------------------------------------------------------------
# Dataset.empty_rows_for_columns()
# ---------------------------------------------------------------------------

def test_empty_rows_for_columns_all_empty_when_column_does_not_exist_yet():
    # Would still pass if the method required the column to already
    # exist? No -- a first-ever run has no output column yet, and every
    # chosen row must count as empty in that case.
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    row_ids = ds.get_table("frames")["row_id"].tolist()[:4]

    result = ds.empty_rows_for_columns("frames", ["out"], row_ids)

    assert result == row_ids


def test_empty_rows_for_columns_partly_filled():
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    row_ids = ds.get_table("frames")["row_id"].tolist()[:4]
    ds.apply_row_updates("frames", {
        row_ids[0]: {"out": 1.0},
        row_ids[1]: {"out": 2.0},
    })
    # row_ids[2], row_ids[3] were never in an update batch, so the column
    # created for them defaults to None -- empty.

    result = ds.empty_rows_for_columns("frames", ["out"], row_ids)

    assert result == [row_ids[2], row_ids[3]]


def test_empty_rows_for_columns_a_declared_column_missing_from_the_table_disqualifies_no_row():
    # Would still pass if a not-yet-existing column were treated as
    # "every row fails it"? No -- that would make "fill only empty"
    # refuse to run at all the first time an operator adds a SECOND
    # output column alongside one it already wrote.
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    row_ids = ds.get_table("frames")["row_id"].tolist()[:3]
    ds.apply_row_updates("frames", {row_ids[0]: {"out": 1.0}})

    result = ds.empty_rows_for_columns(
        "frames", ["out", "does_not_exist_yet"], row_ids
    )

    assert result == [row_ids[1], row_ids[2]]


def test_empty_rows_for_columns_preserves_the_given_order():
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    row_ids = ds.get_table("frames")["row_id"].tolist()[:4]
    ds.apply_row_updates("frames", {row_ids[0]: {"out": 1.0}})
    shuffled = [row_ids[3], row_ids[0], row_ids[2], row_ids[1]]

    result = ds.empty_rows_for_columns("frames", ["out"], shuffled)

    assert result == [row_ids[3], row_ids[2], row_ids[1]]


# ---------------------------------------------------------------------------
# Dataset.most_recent_operator_run() / record_operator_run(operator_version=)
# ---------------------------------------------------------------------------

def _record(ds, *, operator, mode, table, parameters, version=None):
    ds.record_operator_run(
        operator_name=operator, mode=mode, label="L", parameters=parameters,
        target_table=table, inputs={}, rows_requested=1, rows_applied=1,
        unplaceable_row_ids=[], outcome="complete", superseded_tables=[],
        operator_version=version,
    )


def test_most_recent_operator_run_picks_the_latest_and_ignores_other_operators_modes_tables():
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    _record(ds, operator="op_a", mode="COLUMNS", table="frames", parameters={"x": 1}, version="1.0")
    _record(ds, operator="op_b", mode="COLUMNS", table="frames", parameters={"x": 99}, version="9.0")
    _record(ds, operator="op_a", mode="TABLE", table="frames", parameters={"x": 2}, version="1.0")
    _record(ds, operator="op_a", mode="COLUMNS", table="other_table", parameters={"x": 3}, version="1.0")
    _record(ds, operator="op_a", mode="COLUMNS", table="frames", parameters={"x": 4}, version="1.1")

    result = ds.most_recent_operator_run("op_a", "COLUMNS", "frames")

    assert result is not None
    assert result["parameters"] == {"x": 4}
    assert result["operator_version"] == "1.1"


def test_most_recent_operator_run_returns_none_when_nothing_is_recorded():
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)

    assert ds.most_recent_operator_run("nope", "COLUMNS", "frames") is None


def test_record_operator_run_stores_operator_version():
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    ds.record_operator_run(
        operator_name="op", mode="COLUMNS", label="L", parameters={},
        target_table="frames", inputs={}, rows_requested=1, rows_applied=1,
        unplaceable_row_ids=[], outcome="complete", superseded_tables=[],
        operator_version="2.3",
    )

    entries = [e for e in ds.provenance.to_list() if e["action"] == "operator_run"]
    assert entries[-1]["params"]["operator_version"] == "2.3"


def test_record_operator_run_operator_version_defaults_to_none():
    # Every caller before P2.2a never mentions operator_version -- must
    # keep working exactly as before.
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    ds.record_operator_run(
        operator_name="op", mode="COLUMNS", label="L", parameters={},
        target_table="frames", inputs={}, rows_requested=1, rows_applied=1,
        unplaceable_row_ids=[], outcome="complete", superseded_tables=[],
    )

    entries = [e for e in ds.provenance.to_list() if e["action"] == "operator_run"]
    assert entries[-1]["params"]["operator_version"] is None


# ---------------------------------------------------------------------------
# Dataset.clear_output_columns()
# ---------------------------------------------------------------------------

def test_clear_output_columns_touches_only_chosen_rows_and_only_named_columns():
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    row_ids = ds.get_table("frames")["row_id"].tolist()[:4]
    ds.apply_row_updates("frames", {
        row_ids[0]: {"out": 1.0, "other": 10.0},
        row_ids[1]: {"out": 2.0, "other": 20.0},
        row_ids[2]: {"out": 3.0, "other": 30.0},
        row_ids[3]: {"out": 4.0, "other": 40.0},
    })

    ds.clear_output_columns("frames", ["out"], [row_ids[0], row_ids[1]])

    # Chosen rows, named column: cleared.
    assert pd.isna(_out_value(ds, "frames", row_ids[0], "out"))
    assert pd.isna(_out_value(ds, "frames", row_ids[1], "out"))
    # Rows outside the chosen set: untouched.
    assert _out_value(ds, "frames", row_ids[2], "out") == 3.0
    assert _out_value(ds, "frames", row_ids[3], "out") == 4.0
    # A column not named: untouched, even on chosen rows.
    assert _out_value(ds, "frames", row_ids[0], "other") == 10.0
    assert _out_value(ds, "frames", row_ids[1], "other") == 20.0


def test_clear_output_columns_skips_a_column_not_on_the_table():
    ds = Dataset()
    ds.load_folder(TEST_IMAGES)
    row_ids = ds.get_table("frames")["row_id"].tolist()[:2]

    # Must not raise: a declared output column that has never been
    # written yet has nothing to clear.
    ds.clear_output_columns("frames", ["does_not_exist"], row_ids)


# ---------------------------------------------------------------------------
# AppController.describe_existing_outputs()
# ---------------------------------------------------------------------------

def test_describe_existing_outputs_returns_none_when_no_output_column_exists(tmp_path):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _WriteFixedValueOperator()
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:3]

    info = controller.describe_existing_outputs(
        op.name, "COLUMNS", row_ids, {}
    )

    assert info is None


def test_describe_existing_outputs_counts_are_right(tmp_path):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _WriteFixedValueOperator()
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:4]
    dataset.apply_row_updates("frames", {
        row_ids[0]: {"out": 1.0},
        row_ids[1]: {"out": 2.0},
    })

    info = controller.describe_existing_outputs(
        op.name, "COLUMNS", row_ids, {}
    )

    assert info is not None
    assert info.existing_columns == ("out",)
    assert info.chosen_row_count == 4
    assert info.empty_row_count == 2


def test_describe_existing_outputs_message_lists_changed_parameters_and_omits_unchanged(tmp_path):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _WriteFixedValueOperator()
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:2]
    dataset.apply_row_updates("frames", {row_ids[0]: {"out": 1.0}})
    _record(
        dataset, operator=op.name, mode="COLUMNS", table="frames",
        parameters={"factor": 2, "threshold": 0.1}, version="1.0",
    )

    info = controller.describe_existing_outputs(
        op.name, "COLUMNS", row_ids, {"factor": 9, "threshold": 0.1}
    )

    assert info is not None
    assert "factor: was 2, now 9" in info.message
    assert "threshold" not in info.message


def test_describe_existing_outputs_no_settings_line_when_nothing_is_recorded(tmp_path):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _WriteFixedValueOperator()
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:2]
    dataset.apply_row_updates("frames", {row_ids[0]: {"out": 1.0}})

    info = controller.describe_existing_outputs(
        op.name, "COLUMNS", row_ids, {"factor": 9}
    )

    assert info is not None
    assert "Since the last run" not in info.message


def test_describe_existing_outputs_no_version_line_when_the_old_entry_lacks_a_version(tmp_path):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _WriteFixedValueOperator(version="1.0")
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:2]
    dataset.apply_row_updates("frames", {row_ids[0]: {"out": 1.0}})
    _record(
        dataset, operator=op.name, mode="COLUMNS", table="frames",
        parameters={"factor": 1}, version=None,
    )

    info = controller.describe_existing_outputs(
        op.name, "COLUMNS", row_ids, {"factor": 2}
    )

    assert info is not None
    assert "factor: was 1, now 2" in info.message
    assert "version:" not in info.message


def test_describe_existing_outputs_version_line_when_both_are_known_and_differ(tmp_path):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _WriteFixedValueOperator(version="2.0")
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:2]
    dataset.apply_row_updates("frames", {row_ids[0]: {"out": 1.0}})
    _record(
        dataset, operator=op.name, mode="COLUMNS", table="frames",
        parameters={"factor": 1}, version="1.0",
    )

    info = controller.describe_existing_outputs(
        op.name, "COLUMNS", row_ids, {"factor": 1}
    )

    assert info is not None
    assert "version: was 1.0, now 2.0" in info.message


# ---------------------------------------------------------------------------
# AppController.run_create_columns(fill_only_empty=True)
# ---------------------------------------------------------------------------

def test_fill_only_empty_processes_only_empty_rows_and_leaves_filled_rows_untouched(tmp_path, monkeypatch):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _WriteFixedValueOperator()
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:4]
    dataset.apply_row_updates("frames", {
        row_ids[0]: {"out": 1.0},
        row_ids[1]: {"out": 2.0},
    })

    _run_columns_and_wait(
        controller, op.name, row_ids, {}, monkeypatch, fill_only_empty=True,
    )

    # Only the two empty rows ever reached create_columns().
    assert set(op.rows_seen) == {row_ids[2], row_ids[3]}
    # The already-filled rows kept their exact old values.
    assert _out_value(dataset, "frames", row_ids[0]) == 1.0
    assert _out_value(dataset, "frames", row_ids[1]) == 2.0
    # The empty rows were filled.
    assert _out_value(dataset, "frames", row_ids[2]) == 999.0
    assert _out_value(dataset, "frames", row_ids[3]) == 999.0


def test_fill_only_empty_records_the_narrowed_row_count(tmp_path, monkeypatch):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _WriteFixedValueOperator()
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:4]
    dataset.apply_row_updates("frames", {
        row_ids[0]: {"out": 1.0},
        row_ids[1]: {"out": 2.0},
    })

    _run_columns_and_wait(
        controller, op.name, row_ids, {}, monkeypatch, fill_only_empty=True,
    )

    entries = _operator_run_entries(dataset)
    assert len(entries) == 1
    assert entries[0]["params"]["rows_requested"] == 2


def test_fill_only_empty_with_zero_empty_rows_registers_no_run(tmp_path):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _WriteFixedValueOperator()
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:3]
    dataset.apply_row_updates("frames", {rid: {"out": 1.0} for rid in row_ids})
    before = len(_operator_run_entries(dataset))

    controller.run_create_columns(op.name, row_ids, {}, fill_only_empty=True)

    assert controller._live_runs == {}
    assert len(_operator_run_entries(dataset)) == before
    assert op.rows_seen == []


# ---------------------------------------------------------------------------
# AppController.run_create_columns(clear_existing_outputs=True)
# ---------------------------------------------------------------------------

def test_fill_only_empty_and_clear_existing_outputs_together_is_a_value_error(tmp_path):
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _WriteFixedValueOperator()
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:2]

    with pytest.raises(ValueError):
        controller.run_create_columns(
            op.name, row_ids, {},
            fill_only_empty=True, clear_existing_outputs=True,
        )

    assert controller._live_runs == {}


def test_overwrite_all_clears_a_row_with_missing_media_instead_of_keeping_its_old_value(tmp_path, monkeypatch):
    # This is the exact failure clear_output_columns fixes: a row the
    # per-row runner never delivers a result for at all (its media cell
    # is blank, so operators/operator_registry.py reports MissingMedia
    # and never calls create_columns() for it) must not silently keep
    # whatever an earlier run under different settings left in its
    # output column.
    controller, dataset, op_registry = _make_controller(tmp_path)
    op = _FrameEchoOperator()
    op_registry.register(op)
    row_ids = controller.get_visible_row_ids()[:4]

    # First run: every row has real media, so every row gets a value.
    _run_columns_and_wait(controller, op.name, row_ids, {}, monkeypatch)
    for rid in row_ids:
        assert _out_value(dataset, "frames", rid) == 42.0

    # Now break one row's media, and re-run with "Overwrite all" over the
    # same chosen rows.
    dataset.apply_row_updates("frames", {row_ids[0]: {"full_path": ""}})
    op.rows_seen = []

    _run_columns_and_wait(
        controller, op.name, row_ids, {}, monkeypatch,
        clear_existing_outputs=True,
    )

    # The broken row was never delivered a result this run...
    assert row_ids[0] not in op.rows_seen
    # ...and must therefore be empty, not still 42.0 from the first run.
    assert pd.isna(_out_value(dataset, "frames", row_ids[0]))
    # The other three rows were processed normally.
    for rid in row_ids[1:]:
        assert rid in op.rows_seen
        assert _out_value(dataset, "frames", rid) == 42.0
