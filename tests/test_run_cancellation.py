"""
tests/test_run_cancellation.py -- P1.12f-3: the COLUMNS runner's
between-rows cancellation check.

operators/operator_registry.py's _run_create_columns_worker checks
run.cancelled() at the top of its row loop and stops there, keeping every
result already handed to on_item_complete for an earlier row. This is the
mechanism CLAUDE.md's "Long-running work" rule and
AppController.cancel_run() rely on for COLUMNS mode. These tests exercise
it directly at the OperatorRegistry seam -- a real background thread,
joined deterministically the same way tests/test_operator_run_wiring.py's
_run_columns_and_wait does -- rather than through the full AppController
wiring. What AppController itself does once a run ends cancelled (outcome
recording, and the TABLE/DISPLAY discard-on-arrival limit) is covered by
tests/test_result_delivery.py's run-indicator-3 section.

New module because none of the three files this work item may edit
(tests/test_result_delivery.py, tests/test_parameter_dialog.py,
tests/test_run_log.py) already owns OperatorRegistry-level worker tests --
tests/test_operator_run_wiring.py is the natural home for this shape of
test but is not one of them.

No Qt widget is shown here (a real OperatorRegistry and a background
worker thread, joined by hand -- the same construction
tests/test_operator_run_wiring.py uses in the combined, non-widget group).
No `# run-tests:` token is needed.

Written from the work-item specification, not the implementation. Each
test states, in a comment, what would still pass if the rule it guards
were violated.

Run with:
    python -m pytest tests/test_run_cancellation.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
from operators.operator_registry import OperatorRegistry
from operators.run_context import (
    CancellationToken,
    OperatorRun,
    OperatorRunSpec,
    RunData,
)


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


def _columns_descriptor(name, label):
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
                    columns=(OutputColumn(name="probe", type_tag="numeric"),)
                ),
            ),
        ),
    )


class _CancelsAfterRow(BaseOperator):
    """A COLUMNS operator that calls token.cancel() itself, from inside
    create_columns(), once it has processed a chosen row -- standing in
    for "the researcher clicked Cancel while this run was in progress"
    deterministically, without needing real inter-thread timing.

    Recording every row it actually processed (self.processed_row_ids)
    is what lets a test tell "the row that requested cancellation still
    finished its own work" (checked BETWEEN rows, never mid-row) apart
    from "cancellation also skipped that row".
    """

    name = "cancels_after_row"
    descriptor = _columns_descriptor("cancels_after_row", "Cancels after row")

    def __init__(self, token: CancellationToken, cancel_after_row_id: str):
        self._token = token
        self._cancel_after_row_id = cancel_after_row_id
        self.processed_row_ids: list[str] = []

    def create_columns(self, row_id, media, metadata, run):
        self.processed_row_ids.append(row_id)
        if row_id == self._cancel_after_row_id:
            self._token.cancel()
        return {"probe": len(self.processed_row_ids)}


def _build_run(operator: BaseOperator, token: CancellationToken) -> OperatorRun:
    """A minimal, valid OperatorRun for a COLUMNS run of *operator*,
    carrying *token* -- the SAME token object the test controls, so
    cancelling it from the test (or from inside create_columns(), as
    _CancelsAfterRow does) is what the worker's run.cancelled() sees.
    """
    mode_descriptor = operator.descriptor.mode_for(ExecutionMode.COLUMNS)
    spec = OperatorRunSpec(
        operation_id="op-1",
        operator_name=operator.name,
        mode=ExecutionMode.COLUMNS,
        mode_descriptor=mode_descriptor,
        parameters={},
        target_table="frames",
    )
    return OperatorRun(
        spec=spec,
        data=RunData(tables={}, projects={}),
        paths=object(),
        _token=token,
    )


def _run_and_join(op_registry, operator, row_ids, run, monkeypatch):
    """Starts run_create_columns(), joins every worker thread it spawns
    (tracked the same way tests/test_operator_run_wiring.py's
    _run_columns_and_wait does), and collects on_item_complete /
    on_complete into plain lists -- no Qt event loop, no controller."""
    snapshot = pd.DataFrame(
        {"row_id": row_ids, "full_path": ["" for _ in row_ids]}
    )

    item_results: list[tuple] = []
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
            on_item_complete=lambda *a: item_results.append(a),
            on_complete=lambda *a: completions.append(a),
        )
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)

    assert started, "run_create_columns did not start a worker"
    for thread in created:
        thread.join(timeout=10)
        assert not thread.is_alive(), "worker thread did not finish in time"

    return item_results, completions


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

def test_cancellation_between_rows_stops_the_loop_and_keeps_earlier_results(
    monkeypatch,
):
    row_ids = ["r1", "r2", "r3", "r4", "r5"]
    token = CancellationToken()
    operator = _CancelsAfterRow(token, cancel_after_row_id="r2")
    op_registry = OperatorRegistry()
    op_registry.register(operator)
    run = _build_run(operator, token)

    item_results, completions = _run_and_join(
        op_registry, operator, row_ids, run, monkeypatch
    )

    # r1 and r2 were processed. r2 is the row whose OWN create_columns()
    # call requests cancellation and it still completes and is kept --
    # "check BETWEEN rows, never mid-row" means the row already being
    # worked on finishes. r3, r4 and r5 are never started at all.
    #
    # Would still pass if violated? No. A mid-row check (aborting r2's
    # own call the instant it cancels) would leave processed_row_ids as
    # just ["r1"] with no result emitted for r2. A check that came too
    # late (e.g. only before on_complete) would process all 5 rows.
    assert operator.processed_row_ids == ["r1", "r2"]
    processed_from_callback = [row_id for _op, _table, row_id, _r in item_results]
    assert processed_from_callback == ["r1", "r2"]

    assert len(completions) == 1
    _operation_id, _operator_name, emitted = completions[0]
    assert emitted == 2

    assert token.is_cancelled() is True


def test_cancelling_before_the_run_starts_processes_no_rows(monkeypatch):
    # The check is at the TOP of the loop, so a token already cancelled
    # before the worker's first iteration stops it before row 1.
    row_ids = ["r1", "r2", "r3"]
    token = CancellationToken()
    token.cancel()
    operator = _CancelsAfterRow(token, cancel_after_row_id="unused")
    op_registry = OperatorRegistry()
    op_registry.register(operator)
    run = _build_run(operator, token)

    item_results, completions = _run_and_join(
        op_registry, operator, row_ids, run, monkeypatch
    )

    assert item_results == []
    assert operator.processed_row_ids == []
    assert len(completions) == 1
    _operation_id, _operator_name, emitted = completions[0]
    assert emitted == 0


def test_without_cancellation_every_row_is_still_processed(monkeypatch):
    # A contrast case: the between-rows check must not affect an ordinary
    # run that is never cancelled.
    row_ids = ["r1", "r2", "r3"]
    token = CancellationToken()
    operator = _CancelsAfterRow(token, cancel_after_row_id="never-matches")
    op_registry = OperatorRegistry()
    op_registry.register(operator)
    run = _build_run(operator, token)

    item_results, completions = _run_and_join(
        op_registry, operator, row_ids, run, monkeypatch
    )

    assert operator.processed_row_ids == row_ids
    assert [row_id for _op, _table, row_id, _r in item_results] == row_ids
    assert len(completions) == 1
    assert completions[0][2] == len(row_ids)
    assert token.is_cancelled() is False
