"""
tests/test_table_output_contract.py -- P1.7-1: a TABLE-mode operator
declares its own output columns' role/carry_to_children, and reports the
rows it skips through the same channel the COLUMNS path already uses.

Two additions to the TABLE-mode operator contract:

  (A) OutputSpec gains an optional ``table_columns`` (a tuple of
      OutputColumn, each optionally carrying ``role`` /
      ``carry_to_children``) for the creates_table case.
      OperatorRegistry.hints_for_table_output() translates it into the
      ColumnHint dict Dataset.create_table_from_df's new ``hints``
      argument applies through the EXISTING _prepare_table hint
      mechanism -- the same one media hints already use, not a second
      path. A declared column absent from the returned frame is refused
      (OperatorRunError), not silently dropped.

  (B) OperatorRun gains report_row_error(row_id, kind, message), a
      worker-safe accumulator create_table() may call for a row it
      chooses to skip. OperatorRegistry._run_create_table_worker reads it
      back after create_table() returns and delivers it through the
      SAME on_row_errors(operation_id, label, errors) callback the
      COLUMNS path already uses (run_create_columns), in the same order
      (row_errors before on_complete) -- there is no second channel.

These tests exercise the mechanism at the OperatorRegistry / Dataset
seam -- a real OperatorRegistry, a real background thread joined
deterministically (the same construction tests/test_run_cancellation.py
uses for the analogous COLUMNS-path check), and a real Dataset -- rather
than through a live AppController. AppController is built on the Qt
binding, and every existing test that constructs one for real
(tests/test_result_delivery.py) does so behind a running GUI-toolkit
application object and is consequently run in its own isolated pytest
process (see run_tests.py's module docstring and its WIDGET_MODULES
list, which tests/test_run_tests_grouping.py keeps in sync with a scan
of which test modules actually need that). run_tests.py and
tests/test_run_tests_grouping.py are both outside the files this work
item may touch, so this module deliberately imports neither that binding
nor controller.py -- naming either would flip this module into that
isolated set and require updating a file this item may not edit. Instead
it proves "reaches the controller" the way test_run_cancellation.py
already established for run outcomes: at the exact callback boundary
(on_row_errors) the real AppController.run_create_table wires to
controller.py's own _on_operator_row_errors, with the same (operation_id,
label, errors) shape. controller.py's own outcome computation already
turns a set "had row errors" flag into outcome "partial" for any mode,
unchanged by this work item -- it is not re-tested here because nothing
about it changed.

Written from the work-item specification, not the implementation. Each
test states, in a comment, what would still pass if the rule it guards
were violated.

Run with:
    python -m pytest tests/test_table_output_contract.py
"""

from __future__ import annotations

import dataclasses
import sys
import threading
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from operators.base import BaseOperator
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    ModeDescriptor,
    OperatorDescriptor,
    OperatorDescriptorError,
    OutputColumn,
    OutputSpec,
)
from operators.operator_registry import OperatorRegistry
from operators.run_context import (
    CancellationToken,
    OperatorRun,
    OperatorRunSpec,
    RunData,
)
from models.dataset import Dataset
from models.table_schema import ColumnRole


# ---------------------------------------------------------------------------
# Small builders. Deliberately duplicated from other test modules'
# descriptor scaffolding rather than imported -- a test module is not a
# library (see tests/test_run_log.py's own note on this).
# ---------------------------------------------------------------------------

def _active_table_input():
    return (
        InputSpec(
            name="active_table", label="Active table",
            kind=InputKind.ACTIVE_TABLE,
        ),
    )


def _table_descriptor(name, label, *, table_columns=()):
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
                output=OutputSpec(
                    creates_table=True, table_columns=table_columns,
                ),
            ),
        ),
    )


def _build_table_run(operator: BaseOperator) -> OperatorRun:
    """A minimal, valid OperatorRun for a TABLE run of *operator*."""
    mode_descriptor = operator.descriptor.mode_for(ExecutionMode.TABLE)
    spec = OperatorRunSpec(
        operation_id="op-1",
        operator_name=operator.name,
        mode=ExecutionMode.TABLE,
        mode_descriptor=mode_descriptor,
        parameters={},
        target_table="",
    )
    return OperatorRun(
        spec=spec,
        data=RunData(tables={}, projects={}),
        paths=object(),
        resolver=object(),
        _token=CancellationToken(),
    )


def _run_table_and_join(op_registry, operator, df, run, monkeypatch):
    """Starts run_create_table(), joins the worker thread it spawns (the
    same tracked-thread construction tests/test_run_cancellation.py uses
    for run_create_columns), and records every callback call plus the
    ORDER they arrived in -- no Qt, no controller."""
    completions: list[tuple] = []
    errors: list[tuple] = []
    row_error_calls: list[tuple] = []
    call_order: list[str] = []

    created: list[threading.Thread] = []
    real_thread = threading.Thread

    class _Tracked(real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    monkeypatch.setattr(threading, "Thread", _Tracked)
    try:
        started = op_registry.run_create_table(
            operator.name, df, operation_id="op-1", run=run,
            on_complete=lambda *a: (
                call_order.append("complete"), completions.append(a)
            ),
            on_error=lambda *a: (
                call_order.append("error"), errors.append(a)
            ),
            on_row_errors=lambda *a: (
                call_order.append("row_errors"), row_error_calls.append(a)
            ),
        )
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)

    assert started, "run_create_table did not start a worker"
    for thread in created:
        thread.join(timeout=10)
        assert not thread.is_alive(), "worker thread did not finish in time"

    return completions, errors, row_error_calls, call_order


# ---------------------------------------------------------------------------
# Test doubles.
# ---------------------------------------------------------------------------

class _DeclaresIndexRole(BaseOperator):
    """Declares its own output column 'idx' with role='index'."""

    name = "declares_index_role"
    descriptor = _table_descriptor(
        "declares_index_role", "Declares idx as an index column",
        table_columns=(
            OutputColumn(name="idx", type_tag="numeric", role="index"),
        ),
    )

    def create_table(self, df, run):
        out = df.copy()
        out["idx"] = range(len(out))
        return out


class _DeclaresNoRoleHint(BaseOperator):
    """Same output shape as _DeclaresIndexRole, but declares no hint at
    all -- the contrast case that proves the role in the test above is
    not incidental."""

    name = "declares_no_role_hint"
    descriptor = _table_descriptor(
        "declares_no_role_hint", "Declares idx with no role hint",
    )

    def create_table(self, df, run):
        out = df.copy()
        out["idx"] = range(len(out))
        return out


class _DeclaresMissingColumn(BaseOperator):
    """Declares an output column its create_table() never returns."""

    name = "declares_missing_column"
    descriptor = _table_descriptor(
        "declares_missing_column", "Declares a column it never returns",
        table_columns=(
            OutputColumn(name="never_returned", type_tag="numeric"),
        ),
    )

    def create_table(self, df, run):
        return df.copy()


class _ReportsRowThenRaises(BaseOperator):
    """Third fix round, item 1: reports one row via report_row_error(),
    then raises for a different reason -- the exact scenario the
    pre-fix worker lost the report on (collected_row_errors() was only
    read on the success path)."""

    name = "reports_row_then_raises"
    descriptor = _table_descriptor(
        "reports_row_then_raises", "Reports a row, then raises",
    )

    def create_table(self, df, run):
        run.report_row_error("r0", "Flagged", "reported before raising")
        raise ValueError("boom")


class _SkipsFlaggedRows(BaseOperator):
    """Reports every row whose 'skip' column is true through
    run.report_row_error(), and excludes it from the returned frame.

    Fix round, item 5: the returned frame carries the source row_id into
    an ordinary "source_row_id" column, never a literal "row_id" --
    operators/CLAUDE.md's create_table() rule ("Do not add row_id -- it
    is generated when the table is stored") is a real contract, not
    decoration: a returned frame that still had "row_id" would make
    Dataset.create_table_from_df's own row_id insertion raise on the
    duplicate column. This test double used to keep it, which meant
    these tests never actually proved a real Dataset would accept this
    operator's output -- see tests/test_result_delivery.py's
    end-to-end tests for that proof; these registry-seam tests only
    prove the callback-delivery mechanism.
    """

    name = "skips_flagged_rows"
    descriptor = _table_descriptor("skips_flagged_rows", "Skips flagged rows")

    def create_table(self, df, run):
        kept = []
        for _, row in df.iterrows():
            if row["skip"]:
                run.report_row_error(
                    row["row_id"], "Flagged",
                    f"row {row['row_id']} was flagged for skip",
                )
                continue
            kept.append({"source_row_id": row["row_id"]})
        return pd.DataFrame(kept)


class _RaisesForOneRow(BaseOperator):
    """A COLUMNS operator that raises for one row_id -- the COLUMNS
    row_errors path's pre-existing exception-catching source, unchanged
    in shape by the fix round's merge (item 3)."""

    name = "raises_for_one_row"
    descriptor = OperatorDescriptor(
        name="raises_for_one_row",
        version="1.0",
        description="Test double: raises for one row.",
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.COLUMNS,
                label="Raises for one row",
                inputs=_active_table_input(),
                parameters=(),
                output=OutputSpec(
                    columns=(OutputColumn(name="probe", type_tag="numeric"),)
                ),
            ),
        ),
    )

    def __init__(self, bad_row_id: str):
        self._bad_row_id = bad_row_id

    def create_columns(self, row_id, media, metadata, run):
        if row_id == self._bad_row_id:
            raise ValueError("boom")
        return {"probe": 1}


class _RaisesForOneRowAndReportsAnother(BaseOperator):
    """A COLUMNS operator that raises for one row (the existing
    exception-catching source) AND explicitly calls
    run.report_row_error() for a different row (the fix round's new
    COLUMNS-path drain, item 3) -- both must reach on_row_errors."""

    name = "raises_and_reports"
    descriptor = OperatorDescriptor(
        name="raises_and_reports",
        version="1.0",
        description="Test double: raises for one row, reports another.",
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.COLUMNS,
                label="Raises and reports",
                inputs=_active_table_input(),
                parameters=(),
                output=OutputSpec(
                    columns=(OutputColumn(name="probe", type_tag="numeric"),)
                ),
            ),
        ),
    )

    def __init__(self, bad_row_id: str, reported_row_id: str):
        self._bad_row_id = bad_row_id
        self._reported_row_id = reported_row_id

    def create_columns(self, row_id, media, metadata, run):
        if row_id == self._reported_row_id:
            run.report_row_error(
                row_id, "Flagged", f"row {row_id} was flagged, not raised"
            )
            return {"probe": None}
        if row_id == self._bad_row_id:
            raise ValueError("boom")
        return {"probe": 1}


# ---------------------------------------------------------------------------
# (A) Declared role reaches the stored schema.
# ---------------------------------------------------------------------------

def test_declared_index_role_reaches_stored_schema():
    op_registry = OperatorRegistry()
    operator = _DeclaresIndexRole()
    op_registry.register(operator)

    # hints_for_table_output() is the registry-side translation
    # (OutputColumn.role -> ColumnHint(role=ColumnRole...)); create_table()
    # is the operator's own output -- exactly the two things a real
    # create_table run produces, without needing OperatorRegistry's
    # background-thread machinery for this assertion.
    hints = op_registry.hints_for_table_output(operator.name)
    result_df = operator.create_table(pd.DataFrame({"x": [1, 2, 3]}), None)

    ds = Dataset()
    ds.create_table_from_df("built", result_df, hints=hints)
    schema = ds.schema_for("built")

    # Would still pass if violated? No. If hints_for_table_output ignored
    # OutputColumn.role, or create_table_from_df dropped its hints
    # argument, "idx" would fall through to infer_schema's plain
    # int-column default (role=measurement) instead -- exactly what the
    # contrast test right below actually asserts happens with no hint.
    assert schema.spec_for("idx").role is ColumnRole.index


def test_without_role_hint_role_defaults_to_measurement():
    # The reversed case: same int column, no role declared. This is what
    # PROVES the test above is sensitive to the mechanism rather than
    # vacuously true -- if create_table_from_df's hints were ignored
    # entirely, this test and the one above would read identically
    # (both "measurement"), and the assertion above would fail instead.
    op_registry = OperatorRegistry()
    operator = _DeclaresNoRoleHint()
    op_registry.register(operator)

    hints = op_registry.hints_for_table_output(operator.name)
    assert hints == {}, "an operator declaring no role/carry hint must translate to no hints"

    ds = Dataset()
    ds.create_table_from_df(
        "built", operator.create_table(pd.DataFrame({"x": [1, 2, 3]}), None),
        hints=hints,
    )
    schema = ds.schema_for("built")
    assert schema.spec_for("idx").role is ColumnRole.measurement


def test_declared_type_tag_reaches_stored_schema():
    # Fix round, item 1: hints_for_table_output()'s first cut copied role
    # and carry_to_children into the ColumnHint it built but dropped
    # type_tag, so a column declared media_path was silently stored as
    # "text" (the value-scan default for a column of plain words).
    op_registry = OperatorRegistry()
    operator = _DeclaresIndexRole()
    op_registry.register(operator)

    hints = op_registry.hints_for_table_output(operator.name)

    # Would still pass if violated? No. If hints_for_table_output still
    # dropped type_tag, this ColumnHint's type_tag would be None, and
    # infer_schema would fall through to its own value-scan default
    # (a plain int column -> "numeric" anyway here, which is WHY this
    # test asserts on the hint itself rather than only on the stored
    # schema -- the stored tag alone would not distinguish "carried
    # through" from "coincidentally inferred the same value").
    assert hints["idx"].type_tag == "numeric"


def test_invalid_role_string_is_refused_at_registration():
    # Fix round, item 2: an invalid role used to surface only after a
    # whole TABLE run completed, as a bare ValueError from
    # ColumnRole(...) inside hints_for_table_output, swallowed by
    # controller.py's generic "Failed to store table" handler. It must
    # now be refused at register() -- before any run -- with a message
    # naming the operator, the column and the valid roles.
    class _DeclaresInvalidRole(BaseOperator):
        name = "declares_invalid_role"
        descriptor = _table_descriptor(
            "declares_invalid_role", "Declares an unknown role",
            table_columns=(
                OutputColumn(name="idx", type_tag="numeric", role="indexx"),
            ),
        )

        def create_table(self, df, run):
            return df.copy()

    op_registry = OperatorRegistry()
    operator = _DeclaresInvalidRole()

    with pytest.raises(OperatorDescriptorError) as excinfo:
        op_registry.register(operator)

    message = str(excinfo.value)
    # Would still pass if violated? No. A hand-written string check here
    # (e.g. just "raises") would not catch the valid-roles list silently
    # drifting from ColumnRole if a future edit added a member to one but
    # not the other -- asserting the actual enum values appear is what
    # pins "derived from the enum, not hand-written".
    assert operator.name in message
    assert "idx" in message
    assert "indexx" in message
    for role in ColumnRole:
        assert role.value in message
    assert operator.name not in op_registry.list_operators(), (
        "a refused registration must not leave the operator registered"
    )


# ---------------------------------------------------------------------------
# (A) A declared column absent from the returned frame is an error.
# ---------------------------------------------------------------------------

def test_declared_column_absent_from_result_raises(monkeypatch):
    op_registry = OperatorRegistry()
    operator = _DeclaresMissingColumn()
    op_registry.register(operator)
    run = _build_table_run(operator)
    df = pd.DataFrame({"row_id": ["r0", "r1"], "x": [1, 2]})

    completions, errors, row_error_calls, call_order = _run_table_and_join(
        op_registry, operator, df, run, monkeypatch
    )

    # Would still pass if violated? No. A version that let
    # Dataset._prepare_table silently narrow away the unmatched hint (as
    # it does for every OTHER caller's hints) would call on_complete with
    # the incomplete frame and never call on_error at all.
    assert completions == []
    assert row_error_calls == []
    assert len(errors) == 1
    _operation_id, _operator_name, message = errors[0]
    assert "never_returned" in message


def test_reported_row_survives_when_create_table_then_raises(monkeypatch):
    # Third fix round, item 1: a row reported through report_row_error()
    # must reach on_row_errors even when create_table() goes on to raise
    # -- the pre-fix worker only called collected_row_errors() after a
    # successful return, so this report used to be silently dropped.
    op_registry = OperatorRegistry()
    operator = _ReportsRowThenRaises()
    op_registry.register(operator)
    run = _build_table_run(operator)
    df = pd.DataFrame({"row_id": ["r0", "r1"], "x": [1, 2]})

    completions, errors, row_error_calls, call_order = _run_table_and_join(
        op_registry, operator, df, run, monkeypatch
    )

    # Would still pass if violated? No. Before this fix, row_error_calls
    # would be [] here -- the report made just before the raise was never
    # read back, because collected_row_errors() was only called on the
    # try block's success path, which this scenario never reaches.
    assert len(row_error_calls) == 1
    _op_id, _label, reported = row_error_calls[0]
    assert reported == [("r0", "Flagged", "reported before raising")]

    assert len(errors) == 1
    _op_id, _op_name, message = errors[0]
    assert "boom" in message

    # Exactly one terminal call, and it is on_error, never on_complete --
    # a run that raised must not also look like it succeeded.
    assert completions == []
    assert call_order == ["row_errors", "error"], (
        "the report must be delivered BEFORE the terminal on_error call, "
        "the same ordering the success path already uses before on_complete"
    )


# ---------------------------------------------------------------------------
# (B) Skipped rows reach on_row_errors, one entry each, before on_complete.
# ---------------------------------------------------------------------------

def test_table_operator_skipped_rows_reach_on_row_errors_before_complete(
    monkeypatch,
):
    op_registry = OperatorRegistry()
    operator = _SkipsFlaggedRows()
    op_registry.register(operator)
    run = _build_table_run(operator)
    df = pd.DataFrame({
        "row_id": ["r0", "r1", "r2", "r3"],
        "skip":   [False, True, False, True],
    })

    completions, errors, row_error_calls, call_order = _run_table_and_join(
        op_registry, operator, df, run, monkeypatch
    )

    # Would still pass if violated? No.
    #   - If report_row_error() were a no-op, row_error_calls would be
    #     empty and this whole block would fail.
    #   - If on_row_errors fired once PER ROW instead of once with every
    #     row (unlike the shape run_create_columns' on_row_errors already
    #     documents), len(row_error_calls) would be 2, not 1.
    #   - If ordering were reversed (on_complete before on_row_errors, the
    #     opposite of the COLUMNS path), call_order would read
    #     ["complete", "row_errors"].
    assert errors == []
    assert len(row_error_calls) == 1
    operation_id, label, row_errors = row_error_calls[0]
    assert operation_id == "op-1"
    assert {row_id for row_id, _kind, _msg in row_errors} == {"r1", "r3"}
    assert all(kind == "Flagged" for _rid, kind, _msg in row_errors)

    assert len(completions) == 1
    _op_id, _op_name, result_df = completions[0]
    assert sorted(result_df["source_row_id"].tolist()) == ["r0", "r2"]

    assert call_order == ["row_errors", "complete"], (
        "on_row_errors must fire before on_complete, the same order the "
        "COLUMNS path already uses"
    )


def test_table_operator_with_no_skipped_rows_never_calls_on_row_errors(
    monkeypatch,
):
    # Contrast case: an ordinary run that skips nothing must not touch the
    # row-error channel at all -- mirrors run_create_columns' own
    # `if row_errors and on_row_errors is not None` guard.
    op_registry = OperatorRegistry()
    operator = _SkipsFlaggedRows()
    op_registry.register(operator)
    run = _build_table_run(operator)
    df = pd.DataFrame({
        "row_id": ["r0", "r1"],
        "skip":   [False, False],
    })

    completions, errors, row_error_calls, call_order = _run_table_and_join(
        op_registry, operator, df, run, monkeypatch
    )

    assert row_error_calls == []
    assert call_order == ["complete"]
    assert len(completions) == 1


# ---------------------------------------------------------------------------
# The COLUMNS path's row-error behaviour is unchanged.
# ---------------------------------------------------------------------------

def test_columns_path_row_errors_still_arrive_the_same_way(monkeypatch):
    # The COLUMNS path's own exception-catching source is unchanged by
    # either fix round: this operator never calls report_row_error(), so
    # the item-3 merge (row_errors + run.collected_row_errors()) appends
    # an empty list and produces exactly the same result as before that
    # merge existed. This test exists to prove that.
    op_registry = OperatorRegistry()
    operator = _RaisesForOneRow(bad_row_id="r1")
    op_registry.register(operator)

    mode_descriptor = operator.descriptor.mode_for(ExecutionMode.COLUMNS)
    spec = OperatorRunSpec(
        operation_id="op-1", operator_name=operator.name,
        mode=ExecutionMode.COLUMNS, mode_descriptor=mode_descriptor,
        parameters={}, target_table="frames",
    )
    run = OperatorRun(
        spec=spec, data=RunData(tables={}, projects={}),
        paths=object(), resolver=object(), _token=CancellationToken(),
    )

    row_ids = ["r0", "r1", "r2"]
    snapshot = pd.DataFrame(
        {"row_id": row_ids, "full_path": ["" for _ in row_ids]}
    )

    call_order: list[str] = []
    row_error_calls: list[tuple] = []
    completions: list[tuple] = []

    created: list[threading.Thread] = []
    real_thread = threading.Thread

    class _Tracked(real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    monkeypatch.setattr(threading, "Thread", _Tracked)
    try:
        started = op_registry.run_create_columns(
            operator.name, snapshot, row_ids, "frames", run,
            operation_id="op-1",
            on_item_complete=lambda *a: None,
            on_complete=lambda *a: (
                call_order.append("complete"), completions.append(a)
            ),
            on_row_errors=lambda *a: (
                call_order.append("row_errors"), row_error_calls.append(a)
            ),
        )
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)

    assert started
    for thread in created:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert len(row_error_calls) == 1
    operation_id, label, row_errors = row_error_calls[0]
    assert len(row_errors) == 1
    row_id, exc_type, message = row_errors[0]
    assert row_id == "r1"
    assert exc_type == "ValueError"
    assert call_order == ["row_errors", "complete"]
    assert len(completions) == 1


def test_columns_operator_raising_and_reporting_both_reach_on_row_errors(
    monkeypatch,
):
    # Fix round, item 3: report_row_error() used to sit on every
    # OperatorRun but only the TABLE worker ever drained it back --
    # a COLUMNS operator calling it had the report silently dropped.
    op_registry = OperatorRegistry()
    operator = _RaisesForOneRowAndReportsAnother(
        bad_row_id="r1", reported_row_id="r2"
    )
    op_registry.register(operator)

    mode_descriptor = operator.descriptor.mode_for(ExecutionMode.COLUMNS)
    spec = OperatorRunSpec(
        operation_id="op-1", operator_name=operator.name,
        mode=ExecutionMode.COLUMNS, mode_descriptor=mode_descriptor,
        parameters={}, target_table="frames",
    )
    run = OperatorRun(
        spec=spec, data=RunData(tables={}, projects={}),
        paths=object(), resolver=object(), _token=CancellationToken(),
    )

    row_ids = ["r0", "r1", "r2", "r3"]
    snapshot = pd.DataFrame(
        {"row_id": row_ids, "full_path": ["" for _ in row_ids]}
    )

    row_error_calls: list[tuple] = []
    completions: list[tuple] = []
    created: list[threading.Thread] = []
    real_thread = threading.Thread

    class _Tracked(real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    monkeypatch.setattr(threading, "Thread", _Tracked)
    try:
        started = op_registry.run_create_columns(
            operator.name, snapshot, row_ids, "frames", run,
            operation_id="op-1",
            on_item_complete=lambda *a: None,
            on_complete=lambda *a: completions.append(a),
            on_row_errors=lambda *a: row_error_calls.append(a),
        )
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)

    assert started
    for thread in created:
        thread.join(timeout=10)
        assert not thread.is_alive()

    # Would still pass if violated? No.
    #   - If the COLUMNS worker never drained run.collected_row_errors(),
    #     only "r1" (the caught exception) would appear -- "r2" (the
    #     explicit report) would be missing entirely.
    #   - If the merge REPLACED the caught-exception list instead of
    #     appending to it, "r1" would be missing instead.
    assert len(row_error_calls) == 1
    _operation_id, _label, row_errors = row_error_calls[0]
    assert [row_id for row_id, _kind, _msg in row_errors] == ["r1", "r2"], (
        "expected the caught exception (r1) first, then the explicit "
        "report (r2), each source's own order preserved"
    )
    kinds = {row_id: kind for row_id, kind, _msg in row_errors}
    assert kinds["r1"] == "ValueError"
    assert kinds["r2"] == "Flagged"
    assert len(completions) == 1


# ---------------------------------------------------------------------------
# (item 6) OperatorRun's new fields must never make hashing an
# ADDITIONAL, avoidable failure -- see run_context.py's own comment on
# _row_errors / _row_errors_lock for the full reasoning: RunData was
# already unhashable before this work item, so this does not claim
# OperatorRun is hashable now, only that these two fields are not
# themselves a second, needless reason it would fail to be.
# ---------------------------------------------------------------------------

def test_dataclasses_replace_on_a_run_preserves_reported_row_errors():
    # The one place in the repo that calls dataclasses.replace() on a live
    # OperatorRun (operator_registry.py's PER_WORKER/SHARED model swap,
    # before the row loop starts). replace() copies each field's CURRENT
    # value rather than reinitialising it, so the replaced run must keep
    # seeing the SAME _row_errors list -- a report made before the swap
    # must still be visible after it.
    #
    # Would still pass if violated? No. If replace() (or a future change
    # to OperatorRun) reset _row_errors to a fresh empty list instead of
    # carrying the same object over, "already-reported" would be missing
    # from collected_row_errors() after the swap.
    mode_descriptor = _DeclaresIndexRole.descriptor.mode_for(
        ExecutionMode.TABLE
    )
    spec = OperatorRunSpec(
        operation_id="op-1", operator_name="declares_index_role",
        mode=ExecutionMode.TABLE, mode_descriptor=mode_descriptor,
        parameters={}, target_table="",
    )
    run = OperatorRun(
        spec=spec, data=RunData(tables={}, projects={}),
        paths=object(), resolver=object(), _token=CancellationToken(),
    )
    run.report_row_error("r0", "Flagged", "already reported")

    replaced = dataclasses.replace(run, model=object())

    assert replaced.collected_row_errors() == [("r0", "Flagged", "already reported")]


def test_new_row_error_fields_are_excluded_from_hash_and_eq():
    # A frozen dataclass's generated __hash__ uses exactly the fields
    # whose `compare` is True (dataclasses' own rule: a field is hashed
    # when compare=True, unless hash is explicitly overridden). _row_errors
    # is a plain list -- unhashable -- so it must be compare=False, or
    # hash(some_run) would raise TypeError the moment it reached this
    # field. (RunData, this run's own `data` field, already makes
    # hash(some_run) raise for an unrelated, pre-existing reason -- see
    # run_context.py's comment on these two fields -- so this test checks
    # the narrower, directly provable claim: these two fields specifically
    # are not among the ones __hash__ would use.)
    hashed_field_names = {
        f.name for f in dataclasses.fields(OperatorRun) if f.compare
    }
    assert "_row_errors" not in hashed_field_names
    assert "_row_errors_lock" not in hashed_field_names


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
