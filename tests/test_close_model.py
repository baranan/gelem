"""
tests/test_close_model.py

P2.4b follow-up: operators/base.py's close_model(model) hook and
operators/operator_registry.py's call sites for it (the serial COLUMNS
path, the parallel path's consumers / upfront build / coordinator
fallback, and the sequence runner's per-sequence model, including a
row-error reset, a sequence's normal end, an operator abort, and
cancellation).

No real operator and no MediaPipe here (operators/blendshapes.py's own
close_model() is exercised for real by tests/test_blendshapes_tracking.py).
Every operator under test is _FakeCloseModelOperator, defined in this
file, whose build_model()/build_sequence_model() hand out small objects
with their own identity and whose close_model() records which model was
closed, on which thread. _assert_closed_correctly (below) is the one
place every test checks the same two invariants: every built model is
closed EXACTLY ONCE, and it is closed on the thread that used it LAST
(or, if it was never used at all, the thread that built it).

Groups D's two tests use real vids/ fixture videos (gitignored, the same
directory tests/test_sequence_runner.py uses) to form a genuine
MULTI-ROW grouped sequence -- under MediaRequirement.METADATA (every
other group here) the sequence runner always treats a row as its own
one-row sequence, so a reset that is followed by MORE rows on the fresh
model cannot be exercised without real frame addresses. They skip
cleanly when vids/ is absent.

Written from the work-item spec, not the implementation. Each test
states, in a comment, what would still pass if the rule it guards were
broken. Every test that waits on threads joins with a timeout.

Run with:
    python -m pytest tests/test_close_model.py
"""

from __future__ import annotations

import dataclasses
import itertools
import sys
import threading
from pathlib import Path

import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from operators.base import BaseOperator, OperatorSetupError
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
from operators.operator_registry import OperatorRegistry
from operators.run_context import (
    CancellationToken,
    OperatorRun,
    OperatorRunSpec,
    RunData,
)
from media.media_address import format as format_address, from_path
from media.resolver import MediaResolver

VIDS_DIR = project_root / "vids"
VIDEO_PATHS = sorted(VIDS_DIR.glob("*.mp4")) if VIDS_DIR.is_dir() else []
_needs_video = pytest.mark.skipif(
    not VIDEO_PATHS, reason="vids/ fixture videos are not present on this machine"
)


class _FatalTestError(BaseException):
    """Deliberately NOT an Exception subclass -- stands in for a thread
    dying from something that is not a per-row exception, the same way
    tests/test_parallel_columns_runner.py's own _FatalTestError does."""


class _FakeModel:
    """A trivial model object with its own identity -- nothing to
    release, but distinct instances must be told apart."""

    _next_id = itertools.count(1)

    def __init__(self):
        self.id = next(_FakeModel._next_id)


class _FakeCloseModelOperator(BaseOperator):
    """A COLUMNS operator exercising every site operators/operator_registry.py
    discards a model at. model_lifecycle PER_WORKER (serial, parallel) and
    a SequenceOption named "use_sequence" (the sequence runner) share one
    descriptor, matching operators/blendshapes.py's own shape.

    build_model() / build_sequence_model() both hand out a fresh
    _FakeModel and record (model id -> building thread id).
    create_columns() / create_columns_in_sequence() both record (model id
    -> using thread id) and can be told to raise for one row_id:
        raise_for_row_id    -- an ordinary Exception (ValueError): a
                                ROW ERROR, never a model-lifecycle event
                                by itself (the parallel/serial paths keep
                                the model; the sequence runner resets it).
        crash_once_for_row_id -- _FatalTestError, but only the FIRST time
                                that row is attempted -- a thread "dying"
                                in a way _classify_row_result's mapping
                                does not catch; the retry (on a fallback
                                or reset model) succeeds.
        abort_for_row_id    -- OperatorSetupError: aborts the whole run.
        crash_every_original_model (with original_model_count set to the
                                run's worker_count) -- EVERY ONE of the
                                first `original_model_count` models built
                                crashes on ITS OWN first use, succeeding
                                on any retry -- what lets a test force
                                every parallel consumer to die on its own
                                first row while the coordinator's later
                                fallback model (built only once a death is
                                observed) never does. Mirrors
                                tests/test_parallel_columns_runner.py's
                                own _FakeParallelOperator
                                (die_once_for_every_row / consumer_count).
    close_model() records (model id, closing thread id).
    """

    name = "fake_close_model_op"

    def __init__(
        self,
        *,
        media_requirement=MediaRequirement.METADATA,
        raise_for_row_id: str | None = None,
        crash_once_for_row_id: str | None = None,
        abort_for_row_id: str | None = None,
        build_fails_on_call: int | None = None,
        crash_every_original_model: bool = False,
        original_model_count: int = 0,
    ):
        super().__init__()
        self.descriptor = OperatorDescriptor(
            name=self.name,
            version="1.0",
            description="Test double: close_model call sites.",
            modes=(
                ModeDescriptor(
                    mode=ExecutionMode.COLUMNS,
                    label="Fake close_model op",
                    inputs=(
                        InputSpec(
                            name="active_table", label="Active table",
                            kind=InputKind.ACTIVE_TABLE,
                        ),
                    ),
                    media_requirement=media_requirement,
                    parameters=(
                        BooleanParameter(
                            name="use_sequence", label="Use sequence",
                            required=False, default=False,
                        ),
                    ),
                    output=OutputSpec(columns=(
                        OutputColumn(name="out", type_tag="numeric"),
                    )),
                    model_lifecycle=ModelLifecycle.PER_WORKER,
                    sequence_option=SequenceOption(enabled_by="use_sequence"),
                ),
            ),
        )
        self._raise_for_row_id = raise_for_row_id
        self._crash_once_for_row_id = crash_once_for_row_id
        self._abort_for_row_id = abort_for_row_id
        self._build_fails_on_call = build_fails_on_call
        self._crash_every_original_model = crash_every_original_model
        self._original_model_count = original_model_count

        self.lock = threading.Lock()
        self.build_call_count = 0
        self.build_thread_by_model: dict[int, int] = {}
        self.used_thread_by_model: dict[int, list[int]] = {}
        self.closed: list[tuple[int, int]] = []
        self._crashed_once: set = set()
        self._original_model_ids: set = set()
        self._crashed_models: set = set()

    # -- model lifecycle --------------------------------------------------

    def _new_model(self) -> _FakeModel:
        with self.lock:
            self.build_call_count += 1
            call_number = self.build_call_count
            is_original = (
                self._crash_every_original_model
                and call_number <= self._original_model_count
            )
        if self._build_fails_on_call is not None and call_number == self._build_fails_on_call:
            raise OperatorSetupError(f"model build failed on call {call_number} (test)")
        model = _FakeModel()
        with self.lock:
            self.build_thread_by_model[model.id] = threading.get_ident()
            if is_original:
                self._original_model_ids.add(model.id)
        return model

    def build_model(self):
        return self._new_model()

    def build_sequence_model(self, run):
        return self._new_model()

    def close_model(self, model) -> None:
        with self.lock:
            self.closed.append((model.id, threading.get_ident()))

    # -- per-row work -------------------------------------------------------

    def _use(self, model: _FakeModel, row_id: str) -> dict:
        with self.lock:
            self.used_thread_by_model.setdefault(model.id, []).append(
                threading.get_ident()
            )
            should_crash_original = (
                model.id in self._original_model_ids
                and model.id not in self._crashed_models
            )
            if should_crash_original:
                self._crashed_models.add(model.id)
        if should_crash_original:
            raise _FatalTestError(f"simulated crash on model {model.id} (test)")
        if self._crash_once_for_row_id == row_id and row_id not in self._crashed_once:
            self._crashed_once.add(row_id)
            raise _FatalTestError(f"simulated crash on {row_id} (test)")
        if self._abort_for_row_id == row_id:
            raise OperatorSetupError(f"simulated abort on {row_id} (test)")
        if self._raise_for_row_id == row_id:
            raise ValueError(f"boom on {row_id}")
        return {"out": float(len(row_id))}

    def create_columns(self, row_id, media, metadata, run):
        return self._use(run.model, row_id)

    def create_columns_in_sequence(self, row_id, media, metadata, run, media_time_us):
        return self._use(run.model, row_id)


def _assert_closed_correctly(op: _FakeCloseModelOperator) -> None:
    """Every built model is closed EXACTLY ONCE, and -- for every model
    that was actually used at least once -- on the same thread that used
    it LAST. The one check every test in this file ends with.

    A model NEVER used at all (e.g. a parallel consumer whose queue
    happened to run dry before it was handed a row) is deliberately NOT
    thread-checked here: which thread is "responsible" for closing an
    unused model is a call-site detail (the owning consumer thread in the
    parallel path, the coordinator in an upfront build failure, the
    builder in the sequenced path) that a cross-cutting helper has no way
    to know -- only that it must still be closed exactly once, which the
    first two assertions below already require.
    """
    closed_ids = [model_id for model_id, _tid in op.closed]
    assert len(closed_ids) == len(set(closed_ids)), (
        f"a model was closed more than once: {op.closed}"
    )
    built_ids = set(op.build_thread_by_model)
    assert set(closed_ids) == built_ids, (
        f"built models {sorted(built_ids)} do not match closed models "
        f"{sorted(set(closed_ids))} -- something was built and never "
        f"closed, or closed without ever being built"
    )
    for model_id, close_thread_id in op.closed:
        used_threads = op.used_thread_by_model.get(model_id, [])
        if not used_threads:
            continue
        expected_thread_id = used_threads[-1]
        assert close_thread_id == expected_thread_id, (
            f"model {model_id} closed on thread {close_thread_id}, "
            f"expected {expected_thread_id} (the thread that used it last)"
        )


def _frame_cell(path: Path, ordinal: int) -> str:
    return format_address(dataclasses.replace(from_path(str(path)), frame=ordinal))


def _build_run(operator, parameters, resolver=None, token=None) -> OperatorRun:
    mode_descriptor = operator.descriptor.mode_for(ExecutionMode.COLUMNS)
    spec = OperatorRunSpec(
        operation_id="op-1",
        operator_name=operator.name,
        mode=ExecutionMode.COLUMNS,
        mode_descriptor=mode_descriptor,
        parameters=parameters,
        target_table="rows",
    )
    return OperatorRun(
        spec=spec,
        data=RunData(tables={}, projects={}),
        paths=object(),
        resolver=resolver if resolver is not None else object(),
        _token=token or CancellationToken(),
    )


def _metadata_snapshot(row_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"row_id": row_ids})


def _run_and_join(
    op_registry, operator, snapshot, row_ids, run, monkeypatch, *, worker_count,
    media_column=None,
):
    op_registry.register(operator)
    results: dict[str, dict] = {}
    row_errors: list[tuple] = []
    setup_errors: list[str] = []

    created: list[threading.Thread] = []
    real_thread = threading.Thread

    class _Tracked(real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    def _on_item_complete(operation_id, table_name, row_id, result):
        results[row_id] = result

    def _on_row_errors(operation_id, label, errors):
        row_errors.extend(errors)

    def _on_progress(percent):
        pass

    def _on_setup_error(operation_id, label, message):
        setup_errors.append(message)

    def _on_complete(operation_id, operator_name, emitted, elapsed_seconds=0.0, worker_count=1):
        pass

    monkeypatch.setattr(threading, "Thread", _Tracked)
    try:
        started = op_registry.run_create_columns(
            operator.name, snapshot, row_ids, "rows", run,
            operation_id="op-1",
            on_item_complete=_on_item_complete,
            on_progress=_on_progress,
            on_complete=_on_complete,
            on_setup_error=_on_setup_error,
            on_row_errors=_on_row_errors,
            media_column=media_column,
            worker_count=worker_count,
        )
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)

    assert started, "run_create_columns did not start a worker"
    for thread in created:
        thread.join(timeout=15)
        assert not thread.is_alive(), "a thread did not finish in time (possible hang)"

    return {"results": results, "row_errors": row_errors, "setup_errors": setup_errors}


# ===========================================================================
# A. Serial path (worker_count=1, use_sequence off -- model_lifecycle
# PER_WORKER is honoured directly, no sub-runner).
# ===========================================================================

def test_serial_closes_the_one_model_once_on_the_serial_thread(monkeypatch):
    """worker_count=1, no errors: exactly one model built, used by every
    row, closed exactly once, all on the same (serial) thread.

    Would still pass if the serial path never called close_model() at
    all? No -- _assert_closed_correctly would fail on an empty op.closed
    not matching the one built model.
    """
    row_ids = [f"r{i}" for i in range(6)]
    op = _FakeCloseModelOperator()
    run = _build_run(op, {"use_sequence": False})
    out = _run_and_join(
        OperatorRegistry(), op, _metadata_snapshot(row_ids), row_ids, run,
        monkeypatch, worker_count=1,
    )

    assert not out["row_errors"]
    assert set(out["results"]) == set(row_ids)
    assert op.build_call_count == 1
    _assert_closed_correctly(op)


def test_serial_row_error_keeps_and_still_closes_the_one_model(monkeypatch):
    """An ordinary row-level exception in the serial path is a row
    error, not a model-lifecycle event -- the SAME PER_WORKER model keeps
    serving every other row, and is still closed exactly once at the end.

    Would still pass if the serial path rebuilt a model per row error (it
    must not -- PER_WORKER means one model for the whole run)? No --
    op.build_call_count would be > 1.
    """
    row_ids = [f"r{i}" for i in range(6)]
    op = _FakeCloseModelOperator(raise_for_row_id="r3")
    run = _build_run(op, {"use_sequence": False})
    out = _run_and_join(
        OperatorRegistry(), op, _metadata_snapshot(row_ids), row_ids, run,
        monkeypatch, worker_count=1,
    )

    assert {row_id for row_id, _k, _m in out["row_errors"]} == {"r3"}
    assert op.build_call_count == 1, "PER_WORKER must not rebuild on a row error"
    _assert_closed_correctly(op)


# ===========================================================================
# B. Parallel path (worker_count >= 2).
# ===========================================================================

def test_parallel_closes_each_consumers_model_once_on_its_own_thread(monkeypatch):
    """3 consumers, no errors: 3 models built, each closed exactly once,
    each on the SAME thread that used it (never the coordinator's).

    Would still pass if every consumer's model were left for the
    coordinator to close instead? No -- the per-model thread check inside
    _assert_closed_correctly would fail (closing thread != using thread).
    """
    row_ids = [f"r{i}" for i in range(12)]
    op = _FakeCloseModelOperator()
    run = _build_run(op, {"use_sequence": False})
    out = _run_and_join(
        OperatorRegistry(), op, _metadata_snapshot(row_ids), row_ids, run,
        monkeypatch, worker_count=3,
    )

    assert not out["row_errors"]
    assert set(out["results"]) == set(row_ids)
    assert op.build_call_count == 3
    _assert_closed_correctly(op)


def test_parallel_upfront_build_failure_closes_already_built_models(monkeypatch):
    """worker_count=4, the 3rd consumer's build_model() fails: the first
    2 (already built, never handed to any thread) are closed on the
    COORDINATOR thread (the only one that ever touched them), and the run
    aborts through on_setup_error with zero rows processed.

    Would still pass if the already-built models were simply dropped,
    unclosed, on a build failure? No -- _assert_closed_correctly would
    find built models with no matching close().
    """
    row_ids = [f"r{i}" for i in range(5)]
    op = _FakeCloseModelOperator(build_fails_on_call=3)
    run = _build_run(op, {"use_sequence": False})
    out = _run_and_join(
        OperatorRegistry(), op, _metadata_snapshot(row_ids), row_ids, run,
        monkeypatch, worker_count=4,
    )

    assert out["results"] == {}
    assert out["setup_errors"], "no setup error reached the researcher"
    assert op.build_call_count == 3
    assert len(op.closed) == 2, "only the 2 successfully built models exist to close"
    _assert_closed_correctly(op)


def test_parallel_consumer_crash_still_closes_that_consumers_model(monkeypatch):
    """A consumer that dies from a non-Exception crash (_FatalTestError)
    still closes its own model, via the consumer's own `finally` --
    reprocessing on the fallback model (a different model) must not skip
    the dead consumer's own cleanup.

    Would still pass if a crashed consumer's model were leaked instead of
    closed? No -- _assert_closed_correctly would find it unclosed.
    """
    row_ids = [f"r{i}" for i in range(8)]
    op = _FakeCloseModelOperator(crash_once_for_row_id="r2")
    run = _build_run(op, {"use_sequence": False})
    out = _run_and_join(
        OperatorRegistry(), op, _metadata_snapshot(row_ids), row_ids, run,
        monkeypatch, worker_count=3,
    )

    assert not out["row_errors"]
    assert set(out["results"]) == set(row_ids), "the crashed row must still complete (fallback)"
    _assert_closed_correctly(op)


def test_parallel_every_consumer_dies_fallback_model_closed_once(monkeypatch):
    """worker_count=2, EVERY original consumer model crashing on its own
    first use: every consumer has died, so the coordinator takes over
    using its own lazily-built fallback model for the rest of the run.
    Both dead consumers' own models AND the fallback model must each be
    closed exactly once.

    Note: worker_count must be >= 2 for this run to be eligible for the
    parallel path at all (operators/operator_registry.py's own
    eligibility check) -- at worker_count=1 the SAME crash would escape
    uncaught from the serial path instead, which has no fallback
    machinery; that is a different, non-recoverable shape, not this one.

    Would still pass if the fallback model were never closed? No --
    _assert_closed_correctly would find it unclosed (built_thread_by_model
    has it, closed does not).
    """
    row_ids = [f"r{i}" for i in range(6)]
    op = _FakeCloseModelOperator(
        crash_every_original_model=True, original_model_count=2,
    )
    run = _build_run(op, {"use_sequence": False})
    out = _run_and_join(
        OperatorRegistry(), op, _metadata_snapshot(row_ids), row_ids, run,
        monkeypatch, worker_count=2,
    )

    assert not out["row_errors"]
    assert set(out["results"]) == set(row_ids)
    assert op.build_call_count == 3, "the two consumers' models, plus the fallback's"
    _assert_closed_correctly(op)


# ===========================================================================
# C. Sequenced path (use_sequence=True), METADATA -- every row is its own
# one-row sequence (operators/CLAUDE.md's "the sequence runner").
# ===========================================================================

def test_sequenced_one_row_sequences_each_closed_once_on_its_own_thread(monkeypatch):
    """Several independent one-row sequences: one model per row, each
    closed exactly once, on whichever worker thread processed it.

    Would still pass if a sequence's model were left for the NEXT
    sequence's worker iteration to close? No -- the per-model thread
    check would fail whenever two sequences land on different threads.
    """
    row_ids = [f"r{i}" for i in range(9)]
    op = _FakeCloseModelOperator()
    run = _build_run(op, {"use_sequence": True})
    out = _run_and_join(
        OperatorRegistry(), op, _metadata_snapshot(row_ids), row_ids, run,
        monkeypatch, worker_count=3,
    )

    assert not out["row_errors"]
    assert set(out["results"]) == set(row_ids)
    assert op.build_call_count == len(row_ids), "one model per one-row sequence"
    _assert_closed_correctly(op)


def test_sequenced_row_error_closes_old_model_and_the_reset_model(monkeypatch):
    """A one-row sequence whose only row errors: the model that raised is
    closed, a fresh one is built to reset the (now-ending) sequence, and
    that fresh model is ALSO closed since the sequence has no more rows.
    Two models, two closes, both on the one worker thread involved.

    Would still pass if the discarded (raising) model were left for the
    garbage collector instead of closed here? No -- this is the exact
    hang this item exists to fix (docs/review/p2_4b_mediapipe_close_hang.md);
    _assert_closed_correctly would find it unclosed.
    """
    row_ids = ["r0"]
    op = _FakeCloseModelOperator(raise_for_row_id="r0")
    run = _build_run(op, {"use_sequence": True})
    out = _run_and_join(
        OperatorRegistry(), op, _metadata_snapshot(row_ids), row_ids, run,
        monkeypatch, worker_count=1,
    )

    assert len(out["row_errors"]) == 1
    assert out["row_errors"][0][0] == "r0"
    assert op.build_call_count == 2, "the original model, plus the reset"
    _assert_closed_correctly(op)


def test_sequenced_operator_abort_closes_the_model_once(monkeypatch):
    """A row that raises OperatorSetupError aborts the WHOLE run; the
    sequence's one model is still closed exactly once before the run
    reports the setup error.

    Would still pass if an abort outcome skipped closing the model (only
    a reset's rebuild closed it)? No -- _assert_closed_correctly would
    find the one built model unclosed.
    """
    row_ids = ["r0", "r1"]
    op = _FakeCloseModelOperator(abort_for_row_id="r0")
    run = _build_run(op, {"use_sequence": True})
    out = _run_and_join(
        OperatorRegistry(), op, _metadata_snapshot(row_ids), row_ids, run,
        monkeypatch, worker_count=1,
    )

    assert out["setup_errors"], "no setup error reached the researcher"
    _assert_closed_correctly(op)


# ===========================================================================
# D. Sequenced path with a REAL grouped (multi-row) video sequence --
# the only way to exercise "reset, then MORE rows on the fresh model" and
# "cancelled mid-sequence" (METADATA always makes a row its own sequence).
# ===========================================================================

@_needs_video
def test_grouped_sequence_row_error_resets_and_both_models_get_closed(monkeypatch):
    """One video, several #f= rows, one ungrouped sequence: a row error
    partway through closes the model that raised, the sequence continues
    on a fresh model for the remaining rows, and THAT model is closed
    once the sequence (and the whole run) ends.

    Would still pass if the reset model, used successfully for the rest
    of the sequence, were left unclosed at the sequence's natural end? No
    -- _assert_closed_correctly would find it unclosed.
    """
    video_path = VIDEO_PATHS[0]
    resolver = MediaResolver(max_open_decoders=4)
    try:
        frame_times = resolver.get_frame_times(from_path(str(video_path)))
        n_frames = min(len(frame_times), 8)
        assert n_frames >= 4, "fixture too short for a meaningful reset test"
        ordinals = list(range(n_frames))
        row_ids = [f"r{i}" for i in ordinals]
        cells = [_frame_cell(video_path, i) for i in ordinals]
        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells})

        bad_row_id = row_ids[n_frames // 2]
        op = _FakeCloseModelOperator(
            media_requirement=MediaRequirement.FRAME, raise_for_row_id=bad_row_id,
        )
        run = _build_run(op, {"use_sequence": True}, resolver=resolver)
        out = _run_and_join(
            OperatorRegistry(), op, snapshot, row_ids, run, monkeypatch,
            worker_count=1, media_column="media",
        )

        assert {row_id for row_id, _k, _m in out["row_errors"]} == {bad_row_id}
        assert set(out["results"]) == set(row_ids)
        assert op.build_call_count == 2, "the original model, plus the reset"
        _assert_closed_correctly(op)
    finally:
        resolver.close()


@_needs_video
def test_grouped_sequence_cancellation_closes_the_current_model(monkeypatch):
    """One video, several #f= rows, one ungrouped sequence, cancelled
    partway through: the model in use at the moment of cancellation is
    still closed exactly once, on the worker thread that was using it.

    Would still pass if a cancelled sequence left its current model
    unclosed? No -- _assert_closed_correctly would find it unclosed.
    """
    video_path = VIDEO_PATHS[0]
    resolver = MediaResolver(max_open_decoders=4)
    try:
        frame_times = resolver.get_frame_times(from_path(str(video_path)))
        n_frames = min(len(frame_times), 10)
        assert n_frames >= 6, "fixture too short for a meaningful cancel test"
        ordinals = list(range(n_frames))
        row_ids = [f"r{i}" for i in ordinals]
        cells = [_frame_cell(video_path, i) for i in ordinals]
        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells})

        token = CancellationToken()
        cancel_after = row_ids[2]
        op = _FakeCloseModelOperator(
            media_requirement=MediaRequirement.FRAME,
            crash_once_for_row_id=None,
        )
        # Cancel the token from inside _use() the first time the target
        # row is processed -- reuse crash hook plumbing by wrapping _use.
        original_use = op._use

        def _use_and_maybe_cancel(model, row_id):
            if row_id == cancel_after:
                token.cancel()
            return original_use(model, row_id)

        op._use = _use_and_maybe_cancel

        run = _build_run(op, {"use_sequence": True}, resolver=resolver, token=token)
        out = _run_and_join(
            OperatorRegistry(), op, snapshot, row_ids, run, monkeypatch,
            worker_count=1, media_column="media",
        )

        assert not out["row_errors"]
        assert cancel_after in out["results"]
        assert op.build_call_count == 1, "cancellation must not trigger a reset"
        _assert_closed_correctly(op)
    finally:
        resolver.close()
