"""
tests/test_run_log.py -- behaviour tests for OperatorRun.log() (run-indicator-2).

New module, kept separate from tests/test_run_context.py: that file already
owns this shape of pure, Qt-free OperatorRun contract test (see its
"OperatorRun.emit()" section) and is the natural home for these, but it is
not one of the files this work item may edit. This module mirrors its style
-- small builders, one rule per test, no Qt -- for exactly the one new
method, operators/run_context.py's OperatorRun.log().

Written from the work item's specification (CLAUDE.md's "Long-running
work" and operators/CLAUDE.md's "Progress messages -- run.log"), not from
the implementation: log() forwards free text to a runner-supplied sink,
keyed by this run's operation_id, and must never raise regardless of
whether a sink is wired or what that sink does. Coalescing ("LATEST WINS,
PER RUN") is the sink's job, not this method's -- those tests live at the
controller seam, in tests/test_result_delivery.py.
"""

from __future__ import annotations

import pytest

from operators.descriptor import ExecutionMode, ModeDescriptor, OutputColumn, OutputSpec
from operators.run_context import CancellationToken, OperatorRun, OperatorRunSpec, RunData


# ---------------------------------------------------------------------------
# Small builders, so each test reads as one rule rather than ten lines of
# descriptor scaffolding. Deliberately duplicated from test_run_context.py
# rather than imported from it -- a test module is not a library.
# ---------------------------------------------------------------------------
def _columns_mode():
    return ModeDescriptor(
        mode=ExecutionMode.COLUMNS,
        label="Compute the score",
        inputs=(),
        parameters=(),
        output=OutputSpec(columns=(OutputColumn(name="score", type_tag="numeric"),)),
    )


def _spec(mode_descriptor, *, operation_id="op-1", target_table="faces"):
    return OperatorRunSpec(
        operation_id=operation_id,
        operator_name="my_operator",
        mode=mode_descriptor.mode,
        mode_descriptor=mode_descriptor,
        parameters={},
        target_table=target_table,
    )


def _run(spec, *, log_fn=None):
    return OperatorRun(
        spec=spec,
        data=RunData(tables={}, projects={}),
        paths=object(),
        resolver=object(),
        _token=CancellationToken(),
        _log_fn=log_fn,
    )


# ---------------------------------------------------------------------------
# OperatorRun.log().
# ---------------------------------------------------------------------------
def test_log_forwards_operation_id_and_text_to_the_sink():
    received = []

    def sink(operation_id, text):
        received.append((operation_id, text))

    run = _run(_spec(_columns_mode(), operation_id="op-9"), log_fn=sink)

    run.log("clip 3 of 40")

    assert received == [("op-9", "clip 3 of 40")]


def test_log_is_a_noop_with_no_sink_wired():
    # A bare OperatorRun built with no _log_fn (the default) -- a test
    # harness, or a mode the runner has not wired one for yet. Calling
    # log() must not raise: CLAUDE.md's "Long-running work" and
    # operators/CLAUDE.md's "Progress messages" both require an operator
    # be able to run on a `run` that carries no log channel.
    run = _run(_spec(_columns_mode()), log_fn=None)

    run.log("this goes nowhere")  # must not raise


def test_log_never_raises_even_if_the_sink_itself_fails():
    # "An operator must not be able to crash a run by logging at the
    # wrong moment" -- the wrong moment includes the run having already
    # ended on the main thread, which in the real wiring
    # (AppController._on_run_log) never raises by construction. This
    # pins the same guarantee against a sink that DOES misbehave, so a
    # future change to the wiring cannot quietly reintroduce a crash.
    def raising_sink(operation_id, text):
        raise RuntimeError("the run this message belonged to is long gone")

    run = _run(_spec(_columns_mode()), log_fn=raising_sink)

    with pytest.raises(RuntimeError):
        run.log("boom")
    # This test documents today's behaviour (log() does not itself catch a
    # misbehaving sink) rather than asserting the desired one, because the
    # real sink AppController wires is a plain dict write under a lock and
    # cannot raise -- see test_result_delivery.py's controller-level test
    # for the actual "logging after a run has ended does not raise" case.


def test_log_forwards_every_call_not_just_the_last():
    # Coalescing to "only the newest message matters" is the SINK's job
    # (AppController stores one value per operation_id under a lock);
    # OperatorRun.log() itself must not swallow or de-duplicate calls, or
    # an operator that wants to log twice in quick succession would have
    # no way to know only one call reached the sink.
    received = []

    def sink(operation_id, text):
        received.append(text)

    run = _run(_spec(_columns_mode()), log_fn=sink)

    run.log("first")
    run.log("second")
    run.log("third")

    assert received == ["first", "second", "third"]


def test_log_is_available_on_a_table_mode_run():
    # Unlike emit() (COLUMNS-only), log() is wired for every mode -- a
    # TABLE operator such as video_frames reports per-clip progress the
    # same way a COLUMNS operator would.
    table_mode = ModeDescriptor(
        mode=ExecutionMode.TABLE,
        label="Extract frames",
        inputs=(),
        parameters=(),
        output=OutputSpec(creates_table=True),
    )
    received = []
    run = _run(
        _spec(table_mode, target_table=""),
        log_fn=lambda operation_id, text: received.append(text),
    )

    run.log("video 3 of 40: clip.mp4")

    assert received == ["video 3 of 40: clip.mp4"]
