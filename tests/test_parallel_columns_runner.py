"""
tests/test_parallel_columns_runner.py -- the parallel COLUMNS
runner (operators/operator_registry.py's _run_create_columns_parallel).

Exercises OperatorRegistry.run_create_columns(worker_count=N) directly --
a real background thread tree (one producer, N consumers, the calling
thread as coordinator), joined deterministically, the same seam
tests/test_run_cancellation.py and tests/test_model_lifecycle.py already
use. No Qt, no controller, and no MediaPipe: every operator here is a
fake. Most tests use MediaRequirement.METADATA (media=None); the
UnreadableMedia tests use MediaRequirement.FRAME with a fake resolver
double, so nothing here ever decodes a real image.

Written from the work-item specification, not the implementation. Each
test states, in a comment, what would still pass if the rule it guards
were violated.

Run with:
    python -m pytest tests/test_parallel_columns_runner.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from operators.base import BaseOperator, OperatorSetupError
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModeDescriptor,
    ModelLifecycle,
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
from media.media_address import parse as parse_media_address
from media.resolver import MediaResolverError


# ---------------------------------------------------------------------------
# Scaffolding. Deliberately duplicated from other test modules' descriptor
# builders rather than imported -- a test module is not a library (see
# tests/test_run_log.py's own note on this).
# ---------------------------------------------------------------------------

def _active_table_input():
    return (
        InputSpec(
            name="active_table", label="Active table",
            kind=InputKind.ACTIVE_TABLE,
        ),
    )


def _columns_descriptor(
    name, label, *, model_lifecycle,
    media_requirement=MediaRequirement.METADATA,
):
    return OperatorDescriptor(
        name=name,
        version="1.0",
        description=f"Test double: {label}.",
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.COLUMNS,
                label=label,
                inputs=_active_table_input(),
                media_requirement=media_requirement,
                parameters=(),
                output=OutputSpec(
                    columns=(OutputColumn(name="out", type_tag="numeric"),)
                ),
                model_lifecycle=model_lifecycle,
            ),
        ),
    )


class _FatalTestError(BaseException):
    """Deliberately NOT an Exception subclass. _classify_row_result's
    exception mapping only ever catches Exception (matching what
    create_columns() is documented to raise: NotImplementedError,
    OperatorSetupError, or an ordinary Exception) -- this escapes that
    mapping entirely and reaches the parallel runner's consumer loop,
    standing in for "a consumer thread died from something that was
    never a per-row exception"."""


class _FakeParallelOperator(BaseOperator):
    """A PER_WORKER COLUMNS operator with no media (METADATA), whose
    create_columns() and build_model() both record the calling thread's
    ident, so a test can tell "which thread actually computed this" apart
    from "which thread recorded the outcome". Deterministic output (a
    function of row_id alone), so a parallel run and a serial run over
    the same rows must produce identical results.

    raise_for_row_id: create_columns() raises ValueError for this one
    row_id, every time -- an ordinary per-row failure.

    die_once_for_row_id: create_columns() raises _FatalTestError for this
    row_id, but only on its FIRST call for that row -- standing in for a
    transient consumer-thread death, not a permanently poisoned row (a
    real per-row exception is unconditional; this is not one).

    die_once_for_every_row (with consumer_count set to the run's
    worker_count): every row dies on the FIRST call made with one of the
    original per-consumer models, succeeding on retry -- but a call made
    with the FALLBACK's model (built later, lazily, only once a death is
    observed -- see OperatorRegistry._get_fallback_run) never dies,
    whichever thread makes it. This is what lets a test force EVERY
    consumer to die (each dies on its own first row) while guaranteeing
    the coordinator's own reprocessing of every other row still succeeds,
    rather than the coordinator immediately dying too on the very next
    never-before-seen row it reprocesses.

    abort_on_first_call: the very first create_columns() call across all
    threads raises OperatorSetupError -- an ordinary per-row abort that
    happens to land near the start of a run, e.g. while the work queue is
    still full of many other rows.

    build_fails_on_call: build_model() raises OperatorSetupError on its
    Nth call (1-indexed) -- simulates one PER_WORKER consumer's model
    build failing while building N models sequentially before any thread
    starts.
    """

    def __init__(
        self,
        *,
        media_requirement=MediaRequirement.METADATA,
        raise_for_row_id: str | None = None,
        die_once_for_row_id: str | None = None,
        die_once_for_every_row: bool = False,
        consumer_count: int = 0,
        abort_on_first_call: bool = False,
        build_fails_on_call: int | None = None,
        sleep_seconds: float = 0.0,
    ):
        super().__init__()
        self.name = "fake_parallel_op"
        self.descriptor = _columns_descriptor(
            self.name, "Fake parallel op",
            model_lifecycle=ModelLifecycle.PER_WORKER,
            media_requirement=media_requirement,
        )
        self._raise_for_row_id = raise_for_row_id
        self._die_once_for_row_id = die_once_for_row_id
        self._die_once_for_every_row = die_once_for_every_row
        self._consumer_count = consumer_count
        self._abort_on_first_call = abort_on_first_call
        self._build_fails_on_call = build_fails_on_call
        self._sleep_seconds = sleep_seconds
        self._lock = threading.Lock()
        self._died_once_for: set = set()
        self._consumer_model_ids: set = set()
        self._first_call_done = False
        self.build_call_count = 0
        self.build_thread_ids: list[int] = []
        self.row_thread_ids: dict[str, int] = {}

    def build_model(self):
        model = object()
        with self._lock:
            self.build_call_count += 1
            call_number = self.build_call_count
            self.build_thread_ids.append(threading.get_ident())
            if self._die_once_for_every_row and call_number <= self._consumer_count:
                self._consumer_model_ids.add(id(model))
        if self._build_fails_on_call is not None and call_number == self._build_fails_on_call:
            raise OperatorSetupError(f"model build failed on call {call_number} (test)")
        return model

    def create_columns(self, row_id, media, metadata, run):
        if self._sleep_seconds:
            time.sleep(self._sleep_seconds)
        with self._lock:
            self.row_thread_ids[row_id] = threading.get_ident()
            is_first_call = not self._first_call_done
            self._first_call_done = True
        if self._abort_on_first_call and is_first_call:
            raise OperatorSetupError("simulated row-outcome abort (test)")
        should_die = (
            self._die_once_for_row_id == row_id
            or (self._die_once_for_every_row and id(run.model) in self._consumer_model_ids)
        )
        if should_die and row_id not in self._died_once_for:
            self._died_once_for.add(row_id)
            raise _FatalTestError("simulated consumer death (test)")
        if self._raise_for_row_id == row_id:
            raise ValueError(f"boom on {row_id}")
        # Deterministic: a function of row_id alone, so serial and
        # parallel runs over the same rows must agree exactly.
        return {"out": float(len(row_id))}


class _OverridingIterOperator(_FakeParallelOperator):
    """Same as _FakeParallelOperator, but overrides iter_column_updates()
    -- the eligibility check must see this and refuse the parallel path
    regardless of model_lifecycle or worker_count, since an overriding
    operator may carry cross-frame state (P2.4) the parallel consumers'
    independent per-row calls cannot honour. The override's own body is
    never expected to run in these tests (METADATA rows are never
    grouped), so it only needs to exist."""

    def iter_column_updates(self, rows, run):
        raise NotImplementedError("not exercised by these tests")


def _build_run(
    operator: BaseOperator,
    token: CancellationToken | None = None,
    resolver=None,
) -> OperatorRun:
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
        resolver=resolver if resolver is not None else object(),
        _token=token or CancellationToken(),
    )


def _run_and_join(
    op_registry, operator, row_ids, run, monkeypatch, *, worker_count,
    snapshot=None, media_column=None,
):
    """Starts run_create_columns(worker_count=worker_count), joins every
    thread it spawns (producer + consumers, tracked the same way
    tests/test_run_cancellation.py does), and collects every callback
    into plain structures alongside the thread ident that made each
    call -- no Qt event loop, no controller.

    Returns a dict: results (row_id -> value dict), row_errors (list of
    tuples), completions (list of (operation_id, operator_name, emitted,
    elapsed_seconds, worker_count) tuples), setup_errors (list of
    messages), recording_thread_ids (every thread ident that called
    on_item_complete / on_row_errors / on_progress / on_setup_error --
    must all be the SAME thread, the coordinator).
    """
    op_registry.register(operator)
    if snapshot is None:
        snapshot = pd.DataFrame({"row_id": row_ids})

    results: dict[str, dict] = {}
    row_errors: list[tuple] = []
    completions: list[tuple] = []
    setup_errors: list[str] = []
    recording_thread_ids: list[int] = []

    created: list[threading.Thread] = []
    real_thread = threading.Thread

    class _Tracked(real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    def _on_item_complete(operation_id, table_name, row_id, result):
        recording_thread_ids.append(threading.get_ident())
        results[row_id] = result

    def _on_row_errors(operation_id, label, errors):
        recording_thread_ids.append(threading.get_ident())
        row_errors.extend(errors)

    def _on_progress(percent):
        recording_thread_ids.append(threading.get_ident())

    def _on_setup_error(operation_id, label, message):
        recording_thread_ids.append(threading.get_ident())
        setup_errors.append(message)

    def _on_complete(operation_id, operator_name, emitted, elapsed_seconds=0.0, worker_count=1):
        completions.append(
            (operation_id, operator_name, emitted, elapsed_seconds, worker_count)
        )

    monkeypatch.setattr(threading, "Thread", _Tracked)
    try:
        started = op_registry.run_create_columns(
            operator.name, snapshot, row_ids, "frames", run,
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
        thread.join(timeout=10)
        assert not thread.is_alive(), "a thread did not finish in time (possible hang)"

    return {
        "results": results,
        "row_errors": row_errors,
        "completions": completions,
        "setup_errors": setup_errors,
        "recording_thread_ids": recording_thread_ids,
        "threads_created": created,
    }


# ---------------------------------------------------------------------------
# 1. Parallel results equal serial results, row for row.
# ---------------------------------------------------------------------------

def test_parallel_results_equal_serial_results_row_for_row(monkeypatch):
    row_ids = [f"r{i}" for i in range(20)]

    serial_op = _FakeParallelOperator()
    serial_out = _run_and_join(
        OperatorRegistry(), serial_op, row_ids, _build_run(serial_op),
        monkeypatch, worker_count=1,
    )

    parallel_op = _FakeParallelOperator()
    parallel_out = _run_and_join(
        OperatorRegistry(), parallel_op, row_ids, _build_run(parallel_op),
        monkeypatch, worker_count=4,
    )

    assert serial_out["results"] == parallel_out["results"]
    assert set(serial_out["results"].keys()) == set(row_ids)

    # Would still pass if the parallel path silently dropped or
    # reordered a row's result? No -- the dict-equality check above
    # compares every row_id's value, and a dropped row would leave the
    # two dicts different lengths.


# ---------------------------------------------------------------------------
# 2. build_model called once per consumer (parallel) / once (serial).
# ---------------------------------------------------------------------------

def test_build_model_called_once_per_consumer_in_parallel_and_once_in_serial(monkeypatch):
    row_ids = [f"r{i}" for i in range(10)]

    serial_op = _FakeParallelOperator()
    _run_and_join(
        OperatorRegistry(), serial_op, row_ids, _build_run(serial_op),
        monkeypatch, worker_count=1,
    )
    assert serial_op.build_call_count == 1

    parallel_op = _FakeParallelOperator()
    _run_and_join(
        OperatorRegistry(), parallel_op, row_ids, _build_run(parallel_op),
        monkeypatch, worker_count=4,
    )
    assert parallel_op.build_call_count == 4
    # Every one of the 4 rows this run's consumers handled did NOT all
    # land on one thread (i.e. real concurrency, not four sequential
    # calls disguised as parallel) -- see test 8 below for the stronger,
    # dedicated version of this check.

    # Would still pass if the parallel path built one shared model for
    # all 4 consumers? No -- build_call_count would be 1, not 4.


# ---------------------------------------------------------------------------
# 3. worker_count == 1, NONE / SHARED lifecycle, and an overriding
#    operator all take the serial path -- _run_create_columns_parallel is
#    never even reached.
# ---------------------------------------------------------------------------

def test_ineligible_combinations_never_reach_the_parallel_method(monkeypatch):
    row_ids = ["r0", "r1", "r2"]
    parallel_calls: list = []

    def _spy(self, operator, snapshot, row_ids, table_name, run,
             operation_id, on_item_complete, on_progress, on_complete,
             on_setup_error, on_row_errors, media_column, needs_frame,
             label, worker_count, start_time, on_row_finished=None):
        parallel_calls.append(True)
        # Fall through to a harmless no-op completion so the run still
        # ends cleanly and the test can inspect `started`.
        if on_complete is not None:
            on_complete(operation_id, operator.name, 0)

    monkeypatch.setattr(
        OperatorRegistry, "_run_create_columns_parallel", _spy,
    )

    # (a) worker_count == 1 with an eligible (PER_WORKER, no override) operator.
    op = _FakeParallelOperator()
    _run_and_join(
        OperatorRegistry(), op, row_ids, _build_run(op),
        monkeypatch, worker_count=1,
    )
    assert parallel_calls == [], "worker_count=1 reached the parallel method"

    # (b) NONE lifecycle, worker_count >= 2.
    none_op = _FakeParallelOperator()
    none_op.descriptor = _columns_descriptor(
        none_op.name, "none lifecycle", model_lifecycle=ModelLifecycle.NONE,
    )
    _run_and_join(
        OperatorRegistry(), none_op, row_ids, _build_run(none_op),
        monkeypatch, worker_count=4,
    )
    assert parallel_calls == [], "a NONE-lifecycle run reached the parallel method"

    # (c) SHARED lifecycle, worker_count >= 2 -- "per run" only in the
    # sense the design's own PER_RUN wording covers; only PER_WORKER is
    # eligible.
    shared_op = _FakeParallelOperator()
    shared_op.descriptor = _columns_descriptor(
        shared_op.name, "shared lifecycle", model_lifecycle=ModelLifecycle.SHARED,
    )
    _run_and_join(
        OperatorRegistry(), shared_op, row_ids, _build_run(shared_op),
        monkeypatch, worker_count=4,
    )
    assert parallel_calls == [], "a SHARED-lifecycle run reached the parallel method"

    # (d) An operator overriding iter_column_updates(), PER_WORKER, worker_count >= 2.
    overriding_op = _OverridingIterOperator()
    _run_and_join(
        OperatorRegistry(), overriding_op, row_ids, _build_run(overriding_op),
        monkeypatch, worker_count=4,
    )
    assert parallel_calls == [], (
        "an operator overriding iter_column_updates() reached the parallel method"
    )

    # Sanity: an eligible run (PER_WORKER, no override, worker_count>=2)
    # DOES reach it -- otherwise this test would pass vacuously because
    # the spy is simply never wired correctly.
    eligible_op = _FakeParallelOperator()
    _run_and_join(
        OperatorRegistry(), eligible_op, row_ids, _build_run(eligible_op),
        monkeypatch, worker_count=4,
    )
    assert parallel_calls == [True], (
        "an eligible run did not reach the parallel method -- the spy "
        "itself is not wired to the real routing condition"
    )

    # Would still pass if the eligibility check were deleted entirely
    # (every run always parallel)? No -- (a)-(d) would each append to
    # parallel_calls and the assertions right after them would fail.


# ---------------------------------------------------------------------------
# 4. One row raising = exactly one row error; every other row completes.
# ---------------------------------------------------------------------------

def test_one_row_raising_is_exactly_one_row_error_other_rows_complete(monkeypatch):
    row_ids = [f"r{i}" for i in range(12)]
    op = _FakeParallelOperator(raise_for_row_id="r5")

    out = _run_and_join(
        OperatorRegistry(), op, row_ids, _build_run(op),
        monkeypatch, worker_count=4,
    )

    assert [e[0] for e in out["row_errors"]] == ["r5"]
    assert out["row_errors"][0][1] == "ValueError"
    # The failing row still gets a result delivered (all-None), same as
    # the serial path's own unexpected-exception branch.
    assert out["results"]["r5"] == {"out": None}
    # Every OTHER row completed normally.
    for row_id in row_ids:
        if row_id == "r5":
            continue
        assert out["results"][row_id] == {"out": float(len(row_id))}
    assert len(out["results"]) == len(row_ids)

    # Would still pass if one bad row silently took down its whole
    # consumer's remaining share? No -- some row_id after "r5" in that
    # consumer's share would be missing from out["results"].


# ---------------------------------------------------------------------------
# 5. OperatorSetupError in one consumer's build_model aborts the run.
# ---------------------------------------------------------------------------

def test_setup_error_in_one_consumers_build_model_aborts_the_run(monkeypatch):
    row_ids = [f"r{i}" for i in range(8)]
    # Building 4 models sequentially, upfront; the 3rd raises.
    op = _FakeParallelOperator(build_fails_on_call=3)

    out = _run_and_join(
        OperatorRegistry(), op, row_ids, _build_run(op),
        monkeypatch, worker_count=4,
    )

    assert op.build_call_count == 3, (
        "building should stop at the first failure, not continue to "
        "build a 4th model"
    )
    assert out["results"] == {}, "a row was processed after a build failure"
    assert out["setup_errors"], "the setup error never reached the researcher"
    assert "model build failed on call 3" in out["setup_errors"][-1]
    assert out["completions"] and out["completions"][-1][2] == 0, (
        "on_complete must still fire, with 0 rows emitted"
    )
    # No producer/consumer thread was ever started -- nothing to join
    # beyond whatever run_create_columns itself started, which
    # _run_and_join already confirmed finished.

    # Would still pass if the runner ignored the build failure and
    # started the run with only 3 (of 4) consumers anyway? No --
    # out["results"] would be non-empty and there would be no setup error.


# ---------------------------------------------------------------------------
# 6. Cancellation mid-run stops the run and keeps delivered results.
# ---------------------------------------------------------------------------

def test_cancellation_mid_run_keeps_delivered_results_and_does_not_hang(monkeypatch):
    row_ids = [f"r{i}" for i in range(40)]
    op = _FakeParallelOperator(sleep_seconds=0.01)
    token = CancellationToken()
    run = _build_run(op, token=token)

    delivered_count = {"n": 0}

    # Cancel once a handful of rows have actually landed, so the test
    # does not depend on exact thread timing to exercise the check.
    real_create_columns = op.create_columns

    def _create_columns_then_maybe_cancel(row_id, media, metadata, run_arg):
        result = real_create_columns(row_id, media, metadata, run_arg)
        delivered_count["n"] += 1
        if delivered_count["n"] == 5:
            token.cancel()
        return result

    op.create_columns = _create_columns_then_maybe_cancel

    out = _run_and_join(
        OperatorRegistry(), op, row_ids, run,
        monkeypatch, worker_count=4,
    )

    assert 0 < len(out["results"]) < len(row_ids), (
        "cancellation should stop the run before every row is processed, "
        "but keep whatever was already delivered"
    )
    assert out["completions"], "on_complete never fired -- the run hung"

    # Would still pass if cancellation were ignored? No -- len(results)
    # would equal len(row_ids). Would still pass if cancellation dropped
    # already-delivered results? No -- len(results) would be 0.


# ---------------------------------------------------------------------------
# 7. A consumer dying from a non-row exception triggers the serial
#    fallback; every row ends up done; the run does not hang.
# ---------------------------------------------------------------------------

def test_consumer_death_triggers_fallback_every_row_done_no_hang(monkeypatch):
    row_ids = [f"r{i}" for i in range(16)]
    op = _FakeParallelOperator(die_once_for_row_id="r7")

    # The timeout inside _run_and_join (thread.join(timeout=10)) is this
    # test's own hang guard: if the run hangs, this test fails on the
    # "did not finish in time" assertion rather than blocking forever.
    out = _run_and_join(
        OperatorRegistry(), op, row_ids, _build_run(op),
        monkeypatch, worker_count=4,
    )

    assert len(out["results"]) == len(row_ids), (
        "every row must end up done, including the one whose consumer died"
    )
    assert out["results"]["r7"] == {"out": float(len("r7"))}
    for row_id in row_ids:
        assert out["results"][row_id] == {"out": float(len(row_id))}
    assert not out["row_errors"], (
        "a consumer death is not a row_error -- it is recovered by the "
        "fallback, not reported as a failure"
    )

    # Would still pass if the fallback were missing entirely? No -- "r7"
    # would simply never appear in out["results"], and the length
    # assertion above would fail. Would still pass if the fallback hung
    # waiting for the dead consumer? No -- _run_and_join's own
    # thread.join(timeout=10) would fail first.


# ---------------------------------------------------------------------------
# 8. Recording (results / row errors / progress) happens only on the
#    coordinator thread; consumers only compute.
# ---------------------------------------------------------------------------

def test_recording_happens_only_on_the_coordinator_thread(monkeypatch):
    row_ids = [f"r{i}" for i in range(24)]
    op = _FakeParallelOperator(raise_for_row_id="r10", sleep_seconds=0.005)

    out = _run_and_join(
        OperatorRegistry(), op, row_ids, _build_run(op),
        monkeypatch, worker_count=4,
    )

    assert len(out["results"]) == len(row_ids)
    coordinator_idents = set(out["recording_thread_ids"])
    assert len(coordinator_idents) == 1, (
        f"recording (on_item_complete / on_row_errors / on_progress) "
        f"happened on more than one thread: {coordinator_idents}"
    )

    # The computing side (create_columns) really did run on several
    # threads -- otherwise this test would pass vacuously with N=1
    # consumer in practice.
    compute_idents = set(op.row_thread_ids.values())
    assert len(compute_idents) > 1, (
        "create_columns() only ever ran on one thread -- this run was "
        "not actually parallel, so the single-recording-thread assertion "
        "above proves nothing"
    )

    # The coordinator is not any of the consumer threads.
    assert not (coordinator_idents & compute_idents), (
        "the coordinator thread is also one of the consumer threads that "
        "computed a row -- recording and computing are not separated"
    )

    # Would still pass if a consumer called on_item_complete itself
    # directly, instead of handing its outcome back through the results
    # queue? No -- coordinator_idents would gain that consumer's ident
    # and len(coordinator_idents) would be > 1 (see the reversal check
    # performed by hand for this item, reported in the work item's reply).


# ---------------------------------------------------------------------------
# Fixtures for the fix-round tests (hang-on-full-queue, UnreadableMedia).
# ---------------------------------------------------------------------------

class _FailingFrameResolver:
    """A resolver double for MediaRequirement.FRAME rows: resolve_frame()
    raises MediaResolverError for one specific (already-parsed) path, and
    returns a trivial payload otherwise -- exercises a real decode
    failure without touching a real media file."""

    class _Payload:
        def __init__(self):
            self.pixels = object()

    def __init__(self, fail_for_path: str):
        self._fail_for_path = fail_for_path

    def resolve_frame(self, addr, purpose):
        if addr.path == self._fail_for_path:
            raise MediaResolverError("forced decode failure (test)")
        return _FailingFrameResolver._Payload()


_QUEUE_SIZE_MULTIPLIER = 2  # matches operator_registry.py's own work_queue maxsize = 2 * worker_count


# ---------------------------------------------------------------------------
# 9. Cancel mid-run with a full queue: the run completes within the
#    timeout, and no row is processed after cancellation beyond those
#    already in flight.
# ---------------------------------------------------------------------------

def test_cancel_mid_run_with_full_queue_completes_within_timeout(monkeypatch):
    worker_count = 2
    queue_size = _QUEUE_SIZE_MULTIPLIER * worker_count
    row_ids = [f"r{i}" for i in range(queue_size * 10)]
    # Slow consumers, a fast (no-sleep) producer: the bounded work queue
    # fills up almost immediately, long before the first row finishes.
    op = _FakeParallelOperator(sleep_seconds=0.05)
    token = CancellationToken()
    run = _build_run(op, token=token)

    delivered_count = {"n": 0}
    real_create_columns = op.create_columns

    def _create_columns_then_cancel_after_first(row_id, media, metadata, run_arg):
        result = real_create_columns(row_id, media, metadata, run_arg)
        delivered_count["n"] += 1
        if delivered_count["n"] == 1:
            token.cancel()
        return result

    op.create_columns = _create_columns_then_cancel_after_first

    # _run_and_join's own thread.join(timeout=10) is this test's hang
    # guard: a still-hanging coordinator fails this test by timeout
    # rather than freezing the suite.
    out = _run_and_join(
        OperatorRegistry(), op, row_ids, run,
        monkeypatch, worker_count=worker_count,
    )

    assert out["completions"], "on_complete never fired -- the run hung"
    assert 0 < len(out["results"]) < len(row_ids), (
        "cancellation should stop the run well before every row is "
        "processed, but keep whatever was already delivered"
    )

    # Would still pass if the producer's blocking put() were restored
    # (the pre-fix bug)? No -- see the reversal check performed by hand
    # for this item, reported in the work item's reply: it fails by
    # timeout, not by hanging the suite.


# ---------------------------------------------------------------------------
# 10. A row outcome that aborts (OperatorSetupError from create_columns)
#     with a full queue: the run completes within the timeout.
# ---------------------------------------------------------------------------

def test_abort_from_row_outcome_with_full_queue_completes_within_timeout(monkeypatch):
    worker_count = 2
    queue_size = _QUEUE_SIZE_MULTIPLIER * worker_count
    row_ids = [f"r{i}" for i in range(queue_size * 10)]
    # Slow consumers again, so the queue is still full of unconsumed rows
    # when the very first create_columns() call raises and aborts the run.
    op = _FakeParallelOperator(sleep_seconds=0.05, abort_on_first_call=True)

    out = _run_and_join(
        OperatorRegistry(), op, row_ids, _build_run(op),
        monkeypatch, worker_count=worker_count,
    )

    assert out["completions"], "on_complete never fired -- the run hung"
    assert out["setup_errors"], "the setup error never reached the researcher"
    assert len(out["results"]) < len(row_ids), (
        "an aborting row outcome should stop the run well short of "
        "every row being processed"
    )

    # Would still pass if the abort path did not set abort_event, or the
    # producer ignored it? No -- with dozens of rows still queued behind
    # the aborting one and a blocking put(), the producer (and thus
    # producer_thread.join()) would hang, and out["completions"] would
    # be empty within the test's own timeout.


# ---------------------------------------------------------------------------
# 11. Every consumer dies: the run completes, and every row is still
#     done, via the fallback -- including rows the producer had not
#     queued yet when the deaths happened.
# ---------------------------------------------------------------------------

def test_every_consumer_dies_every_row_still_done_by_fallback(monkeypatch):
    worker_count = 2
    row_ids = [f"r{i}" for i in range(_QUEUE_SIZE_MULTIPLIER * worker_count * 10)]
    op = _FakeParallelOperator(
        die_once_for_every_row=True, consumer_count=worker_count,
    )

    out = _run_and_join(
        OperatorRegistry(), op, row_ids, _build_run(op),
        monkeypatch, worker_count=worker_count,
    )

    assert out["completions"], "on_complete never fired -- the run hung"
    assert len(out["results"]) == len(row_ids), (
        "every row must end up done, including rows the producer had "
        "not queued yet when every consumer died"
    )
    for row_id in row_ids:
        assert out["results"][row_id] == {"out": float(len(row_id))}
    assert not out["row_errors"], (
        "a consumer death is not a row_error -- it is recovered by the "
        "fallback, not reported as a failure"
    )

    # Would still pass if the fallback only drained whatever was already
    # sitting in the work queue at the moment of the last death, instead
    # of continuing to take over from the still-running producer? No --
    # with 40 rows and only 2 consumers (both dying on their first row),
    # the vast majority of rows are produced AFTER both consumers are
    # already gone; a snapshot-only drain would leave most of
    # out["results"] missing.


# ---------------------------------------------------------------------------
# 12. UnreadableMedia: a row whose resolve_frame() raises becomes exactly
#     one row error, in both the serial and the parallel path.
# ---------------------------------------------------------------------------

def test_unreadable_media_becomes_one_row_error_in_both_paths(monkeypatch):
    row_ids = ["r0", "r1", "r2"]
    bad_path = "C:/fake/image_bad.jpg"
    media_paths = ["C:/fake/image0.jpg", bad_path, "C:/fake/image2.jpg"]
    fail_for_path = parse_media_address(bad_path).path

    for worker_count in (1, 4):
        op = _FakeParallelOperator(media_requirement=MediaRequirement.FRAME)
        resolver = _FailingFrameResolver(fail_for_path=fail_for_path)
        run = _build_run(op, resolver=resolver)
        snapshot = pd.DataFrame({"row_id": row_ids, "media": media_paths})

        out = _run_and_join(
            OperatorRegistry(), op, row_ids, run, monkeypatch,
            worker_count=worker_count, snapshot=snapshot,
            media_column="media",
        )

        assert [e[0] for e in out["row_errors"]] == ["r1"], (
            f"worker_count={worker_count}: expected exactly one row "
            f"error, for r1; got {out['row_errors']}"
        )
        assert out["row_errors"][0][1] == "UnreadableMedia"
        assert set(out["results"].keys()) == {"r0", "r2"}, (
            f"worker_count={worker_count}: r1 must not silently vanish "
            f"with neither a result nor a row error, and must not gain "
            f"a spurious result either"
        )

    # Would still pass under the old behaviour (print and skip, no row
    # error)? No -- out["row_errors"] would be empty in both the
    # worker_count=1 (serial) and worker_count=4 (parallel) cases, and
    # this is exactly the silent-drop this fix round closes.
