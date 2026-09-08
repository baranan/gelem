"""
tests/test_run_context.py -- behaviour tests for operators/run_context.py.

No Qt. Each rule gets its own named test; nothing loops over a list of
rules. The module under test has no consumers yet (P1.12c builds the
contract; P1.12d wires it in), so every test here constructs the objects
directly.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from operators.descriptor import (
    BooleanParameter,
    ChoiceParameter,
    ExecutionMode,
    ModeDescriptor,
    NumberParameter,
    OutputColumn,
    OutputSpec,
    TextParameter,
)
from operators.run_context import (
    CancellationToken,
    OperatorRun,
    OperatorRunError,
    OperatorRunSpec,
    RunData,
    TableSnapshot,
)


# ---------------------------------------------------------------------------
# Small builders, so each test reads as one rule rather than ten lines of
# descriptor scaffolding.
# ---------------------------------------------------------------------------
def _columns_mode(parameters=()):
    return ModeDescriptor(
        mode=ExecutionMode.COLUMNS,
        label="Compute the score",
        inputs=(),
        parameters=tuple(parameters),
        output=OutputSpec(columns=(OutputColumn(name="score", type_tag="numeric"),)),
    )


def _table_mode(parameters=()):
    return ModeDescriptor(
        mode=ExecutionMode.TABLE,
        label="Make a table",
        inputs=(),
        parameters=tuple(parameters),
        output=OutputSpec(creates_table=True),
    )


def _display_mode(parameters=()):
    return ModeDescriptor(
        mode=ExecutionMode.DISPLAY,
        label="Show a result",
        inputs=(),
        parameters=tuple(parameters),
        output=OutputSpec(is_display_only=True),
    )


def _spec(
    mode_descriptor,
    *,
    parameters=None,
    target_table=None,
    operation_id="op-1",
    operator_name="my_operator",
    mode=None,
):
    if mode is None:
        mode = mode_descriptor.mode
    if parameters is None:
        parameters = {}
    if target_table is None:
        target_table = "faces" if mode is ExecutionMode.COLUMNS else ""
    return OperatorRunSpec(
        operation_id=operation_id,
        operator_name=operator_name,
        mode=mode,
        mode_descriptor=mode_descriptor,
        parameters=parameters,
        target_table=target_table,
    )


def _empty_run_data():
    return RunData(tables={}, projects={})


def _run(spec, *, token=None, emit_fn=None):
    return OperatorRun(
        spec=spec,
        data=_empty_run_data(),
        paths=object(),
        _token=token if token is not None else CancellationToken(),
        _emit_fn=emit_fn,
    )


# ---------------------------------------------------------------------------
# OperatorRunSpec -- one test per validation rule.
# ---------------------------------------------------------------------------
def test_valid_columns_spec_constructs():
    spec = _spec(_columns_mode(), target_table="faces")
    assert spec.mode is ExecutionMode.COLUMNS
    assert spec.target_table == "faces"


def test_operation_id_must_be_non_empty():
    with pytest.raises(OperatorRunError):
        _spec(_columns_mode(), operation_id="")


def test_operator_name_must_be_non_empty():
    with pytest.raises(OperatorRunError):
        _spec(_columns_mode(), operator_name="   ")


def test_mode_must_match_mode_descriptor():
    # A TABLE mode value handed a COLUMNS-mode descriptor.
    with pytest.raises(OperatorRunError):
        _spec(_columns_mode(), mode=ExecutionMode.TABLE)


def test_missing_required_parameter_raises():
    mode = _columns_mode(
        parameters=(NumberParameter(name="window_ms", label="Window (ms)"),)
    )
    with pytest.raises(OperatorRunError):
        _spec(mode, parameters={})


def test_undeclared_parameter_raises():
    with pytest.raises(OperatorRunError):
        _spec(_columns_mode(), parameters={"bogus": 3})


def test_choice_parameter_value_must_be_a_declared_choice():
    mode = _columns_mode(
        parameters=(
            ChoiceParameter(
                name="method",
                label="Method",
                choices=(("mean", "Mean"), ("median", "Median")),
            ),
        )
    )
    with pytest.raises(OperatorRunError):
        _spec(mode, parameters={"method": "mode"})


def test_number_parameter_value_below_minimum_raises():
    mode = _columns_mode(
        parameters=(
            NumberParameter(name="w", label="W", minimum=0, maximum=10),
        )
    )
    with pytest.raises(OperatorRunError):
        _spec(mode, parameters={"w": -1})


def test_number_parameter_value_above_maximum_raises():
    mode = _columns_mode(
        parameters=(
            NumberParameter(name="w", label="W", minimum=0, maximum=10),
        )
    )
    with pytest.raises(OperatorRunError):
        _spec(mode, parameters={"w": 11})


def test_number_parameter_non_numeric_value_raises_operator_run_error():
    # A string value must fail as OperatorRunError, not leak a bare
    # TypeError from the bound comparison.
    mode = _columns_mode(
        parameters=(
            NumberParameter(name="w", label="W", minimum=0, maximum=10),
        )
    )
    with pytest.raises(OperatorRunError):
        _spec(mode, parameters={"w": "300"})


def test_boolean_parameter_value_must_be_an_actual_bool():
    mode = _columns_mode(
        parameters=(BooleanParameter(name="flag", label="Flag"),)
    )
    with pytest.raises(OperatorRunError):
        _spec(mode, parameters={"flag": 1})


def test_columns_mode_requires_non_empty_target_table():
    with pytest.raises(OperatorRunError):
        _spec(_columns_mode(), target_table="")


def test_table_mode_requires_empty_target_table():
    with pytest.raises(OperatorRunError):
        _spec(_table_mode(), target_table="faces")


def test_display_mode_requires_empty_target_table():
    with pytest.raises(OperatorRunError):
        _spec(_display_mode(), target_table="faces")


def test_valid_table_spec_constructs():
    spec = _spec(_table_mode())
    assert spec.mode is ExecutionMode.TABLE
    assert spec.target_table == ""


def test_optional_parameter_may_be_absent():
    mode = _columns_mode(
        parameters=(
            TextParameter(name="note", label="Note", required=False),
        )
    )
    spec = _spec(mode, parameters={})
    assert "note" not in spec.parameters


# ---------------------------------------------------------------------------
# OperatorRunSpec.parameters is genuinely read-only.
# ---------------------------------------------------------------------------
def test_parameters_mapping_is_read_only():
    mode = _columns_mode(
        parameters=(NumberParameter(name="w", label="W"),)
    )
    spec = _spec(mode, parameters={"w": 5})
    with pytest.raises(TypeError):
        spec.parameters["w"] = 6


# ---------------------------------------------------------------------------
# RunData.
# ---------------------------------------------------------------------------
def test_table_returns_the_identical_object_it_was_handed():
    frame = object()
    data = RunData(
        tables={"faces": TableSnapshot(table_name="faces", frame=frame, version=1)},
        projects={},
    )
    assert data.table("faces") is frame
    # And the same object on a second call -- no per-call copy.
    assert data.table("faces") is data.table("faces")


def test_table_on_a_whole_project_input_raises_and_names_tables():
    data = RunData(
        tables={},
        projects={
            "everything": {
                "faces": TableSnapshot(table_name="faces", frame=object(), version=1)
            }
        },
    )
    with pytest.raises(OperatorRunError) as excinfo:
        data.table("everything")
    assert "tables()" in str(excinfo.value)


def test_tables_on_a_single_table_input_raises_and_names_table():
    data = RunData(
        tables={"faces": TableSnapshot(table_name="faces", frame=object(), version=1)},
        projects={},
    )
    with pytest.raises(OperatorRunError) as excinfo:
        data.tables("faces")
    assert "table()" in str(excinfo.value)


def test_table_on_an_unknown_input_raises():
    with pytest.raises(OperatorRunError):
        _empty_run_data().table("nope")


def test_tables_on_an_unknown_input_raises():
    with pytest.raises(OperatorRunError):
        _empty_run_data().tables("nope")


def test_tables_returns_frame_objects_by_table_name():
    frame_a = object()
    frame_b = object()
    data = RunData(
        tables={},
        projects={
            "everything": {
                "a": TableSnapshot(table_name="a", frame=frame_a, version=2),
                "b": TableSnapshot(table_name="b", frame=frame_b, version=2),
            }
        },
    )
    got = data.tables("everything")
    assert got == {"a": frame_a, "b": frame_b}
    assert got["a"] is frame_a


def test_snapshot_returns_the_name_and_version():
    snap = TableSnapshot(table_name="faces", frame=object(), version=7)
    data = RunData(tables={"faces": snap}, projects={})
    assert data.snapshot("faces") is snap
    assert data.snapshot("faces").version == 7


def test_snapshot_on_a_whole_project_input_raises():
    data = RunData(
        tables={},
        projects={
            "everything": {
                "faces": TableSnapshot(table_name="faces", frame=object(), version=1)
            }
        },
    )
    with pytest.raises(OperatorRunError):
        data.snapshot("everything")


def test_versions_flattens_by_table_name():
    data = RunData(
        tables={
            "in_a": TableSnapshot(table_name="faces", frame=object(), version=3),
        },
        projects={
            "in_b": {
                "faces": TableSnapshot(table_name="faces", frame=object(), version=3),
                "trials": TableSnapshot(table_name="trials", frame=object(), version=5),
            }
        },
    )
    assert data.versions() == {"faces": 3, "trials": 5}


def test_same_table_through_two_inputs_at_the_same_version_is_fine():
    data = RunData(
        tables={
            "primary": TableSnapshot(table_name="faces", frame=object(), version=4),
        },
        projects={
            "everything": {
                "faces": TableSnapshot(table_name="faces", frame=object(), version=4),
            }
        },
    )
    assert data.versions() == {"faces": 4}


def test_same_table_through_two_inputs_at_different_versions_raises_in_constructor():
    with pytest.raises(OperatorRunError):
        RunData(
            tables={
                "primary": TableSnapshot(table_name="faces", frame=object(), version=4),
            },
            projects={
                "everything": {
                    "faces": TableSnapshot(
                        table_name="faces", frame=object(), version=5
                    ),
                }
            },
        )


def test_an_input_name_in_both_mappings_raises():
    with pytest.raises(OperatorRunError):
        RunData(
            tables={
                "shared": TableSnapshot(table_name="faces", frame=object(), version=1),
            },
            projects={
                "shared": {
                    "faces": TableSnapshot(
                        table_name="faces", frame=object(), version=1
                    ),
                }
            },
        )


def test_input_names_lists_single_table_inputs_then_project_inputs():
    data = RunData(
        tables={
            "primary": TableSnapshot(table_name="faces", frame=object(), version=1),
        },
        projects={
            "everything": {
                "trials": TableSnapshot(table_name="trials", frame=object(), version=1),
            }
        },
    )
    assert data.input_names() == ("primary", "everything")


def test_run_data_rejects_a_non_snapshot_table_value():
    with pytest.raises(OperatorRunError):
        RunData(tables={"faces": None}, projects={})


def test_run_data_rejects_a_non_mapping_project_value():
    snap = TableSnapshot(table_name="faces", frame=object(), version=1)
    with pytest.raises(OperatorRunError):
        RunData(tables={}, projects={"everything": [snap]})


def test_run_data_rejects_a_non_snapshot_project_value():
    with pytest.raises(OperatorRunError):
        RunData(tables={}, projects={"everything": {"faces": "not a snapshot"}})


def test_table_snapshot_rejects_empty_name():
    with pytest.raises(OperatorRunError):
        TableSnapshot(table_name="", frame=object(), version=1)


def test_table_snapshot_rejects_version_below_one():
    with pytest.raises(OperatorRunError):
        TableSnapshot(table_name="faces", frame=object(), version=0)


# ---------------------------------------------------------------------------
# CancellationToken and OperatorRun.cancelled().
# ---------------------------------------------------------------------------
def test_cancelled_is_false_then_true_after_the_token_is_cancelled():
    token = CancellationToken()
    run = _run(_spec(_columns_mode()), token=token)
    assert run.cancelled() is False
    token.cancel()
    assert run.cancelled() is True


# ---------------------------------------------------------------------------
# OperatorRun.emit().
# ---------------------------------------------------------------------------
def test_emit_forwards_exactly_operation_id_target_table_row_id_values():
    received = []

    def sink(operation_id, target_table, row_id, values):
        received.append((operation_id, target_table, row_id, values))

    spec = _spec(_columns_mode(), target_table="faces", operation_id="op-9")
    run = _run(spec, emit_fn=sink)

    payload = {"score": 1.0}
    run.emit("r1", payload)

    assert received == [("op-9", "faces", "r1", {"score": 1.0})]
    # The receiver must not get the same dict object the operator passed.
    assert received[0][3] is not payload


def test_emit_with_no_channel_raises():
    run = _run(_spec(_columns_mode()), emit_fn=None)
    with pytest.raises(OperatorRunError):
        run.emit("r1", {"score": 1.0})


def test_emit_on_a_table_mode_run_raises():
    # A sink is wired, so this exercises the mode check, not the None check.
    def sink(operation_id, target_table, row_id, values):
        raise AssertionError("sink must not be called for a TABLE-mode run")

    run = _run(_spec(_table_mode()), emit_fn=sink)
    with pytest.raises(OperatorRunError):
        run.emit("r1", {"anything": 1})


def test_emit_on_a_table_mode_run_with_no_sink_reports_the_mode_not_the_sink():
    # A TABLE-mode run legitimately has no sink; the diagnostic must be
    # about the mode, not the missing sink.
    run = _run(_spec(_table_mode()), emit_fn=None)
    with pytest.raises(OperatorRunError) as excinfo:
        run.emit("r1", {"anything": 1})
    assert "COLUMNS mode" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Source scan: the module stays Qt-free and does not import models,
# controller or ui, and imports operator_config only under TYPE_CHECKING.
# ---------------------------------------------------------------------------
def _module_path():
    return (
        pathlib.Path(__file__).resolve().parent.parent
        / "operators"
        / "run_context.py"
    )


# run-tests: combined -- names the Qt binding in a source-scan assertion but never imports it
def test_module_source_has_no_qt_binding_import():
    source = _module_path().read_text(encoding="utf-8")
    # The binding names are written plainly; the declaration comment above
    # keeps this module out of run_tests.py's isolated group even though its
    # text now contains the literal marker.
    qt_bindings = ("PySide6", "PyQt5", "PyQt6")
    for binding in qt_bindings:
        assert binding not in source, f"run_context.py must not import {binding}"


def test_module_does_not_runtime_import_models_controller_ui_or_operator_config():
    source = _module_path().read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_runtime = ("models", "controller", "ui")
    guarded_only = "operators.operator_config"

    # Walk only the module-level statements. Anything inside an
    # `if TYPE_CHECKING:` block is a type-check-time import and does not run.
    for node in tree.body:
        if isinstance(node, ast.If):
            # Skip the TYPE_CHECKING guard entirely -- its body never runs.
            continue
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            assert root not in forbidden_runtime, f"runtime import of {module}"
            assert module != guarded_only, (
                "operator_config must only be imported under TYPE_CHECKING"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden_runtime, f"runtime import of {alias.name}"
                assert alias.name != guarded_only


def test_operator_config_is_imported_under_type_checking():
    # Positive check: the name IS available for annotations, i.e. it appears
    # inside a TYPE_CHECKING block.
    source = _module_path().read_text(encoding="utf-8")
    tree = ast.parse(source)

    found_guarded_import = False
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        # The guard is `if TYPE_CHECKING:`.
        test = node.test
        is_type_checking = (
            isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"
        )
        if not is_type_checking:
            continue
        for inner in node.body:
            if isinstance(inner, ast.ImportFrom) and inner.module == (
                "operators.operator_config"
            ):
                found_guarded_import = True

    assert found_guarded_import
