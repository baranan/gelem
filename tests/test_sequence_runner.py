"""
tests/test_sequence_runner.py

P2.4a: the COLUMNS runner learns ordered sequences. Exercises the seam
operators/operator_registry.py's _run_create_columns_sequenced, reached
whenever operators.descriptor.runs_as_sequences() is true for a run's
mode and parameters -- either because the mode declares model_lifecycle
PER_SEQUENCE outright, or because it declares a SequenceOption and the
run's parameters turn it on.

No real operator is touched here -- every operator under test is a fake
defined in this file (_FakeSequenceOperator below), instrumented to
record exactly which model instance processed which row, in what order,
with what media_time_us. tests/test_model_lifecycle.py already covers
the DEFAULT delegation from build_sequence_model()/
create_columns_in_sequence() to build_model()/create_columns() using a
plain PER_SEQUENCE operator that does not override either; this file
covers the sequence runner's own mechanics.

Written from the work-item spec, not the implementation. Each test
states, in a comment, what would still pass if the rule it guards were
broken. Every test that waits on threads joins with a timeout, so a hang
fails instead of freezing the suite.

Tests 3-5, 8 and 12 use real media: tests 3-5 and 8 need the vids/
fixture videos at the project root (gitignored -- the same directory
tests/test_ordered_frame_runner.py and tests/test_clip_frame_cache.py
use) and skip cleanly when absent; test 12 uses the tracked test_images/
fixture, so it needs no skip.

No Qt widget is shown here, so the run-tests substring heuristic puts
this module in the combined group. No `# run-tests:` token is needed.

Run with:
    python -m pytest tests/test_sequence_runner.py
"""

from __future__ import annotations

import itertools
import random
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
    ColumnParameter,
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModeDescriptor,
    ModelLifecycle,
    OperatorDescriptor,
    OperatorDescriptorError,
    OutputColumn,
    OutputSpec,
    SequenceOption,
    runs_as_sequences,
    sequence_group_column,
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

TEST_IMAGES = project_root / "test_images"
VIDS_DIR = project_root / "vids"
VIDEO_PATHS = sorted(VIDS_DIR.glob("*.mp4")) if VIDS_DIR.is_dir() else []

_needs_video = pytest.mark.skipif(
    not VIDEO_PATHS, reason="vids/ fixture videos are not present on this machine"
)
_needs_two_videos = pytest.mark.skipif(
    len(VIDEO_PATHS) < 2, reason="need at least two vids/ fixture videos"
)


# ---------------------------------------------------------------------------
# Scaffolding. Deliberately duplicated from other test modules' own
# descriptor/harness builders rather than imported -- a test module is
# not a library (see tests/test_run_log.py's own note on this).
# ---------------------------------------------------------------------------

def _frame_cell(path: Path, ordinal: int) -> str:
    """The stored-cell string for a #f=<ordinal> address on `path`."""
    import dataclasses
    return format_address(dataclasses.replace(from_path(str(path)), frame=ordinal))


class _FakeSequenceOperator(BaseOperator):
    """Instrumented COLUMNS operator for exercising the sequence runner
    directly. build_sequence_model() and create_columns_in_sequence()
    are BOTH overridden (never the defaults), so a test can see exactly
    which model instance processed which row, in what order, with what
    media_time_us.

    build_model() / create_columns() are ALSO implemented, separately
    instrumented, so a test can confirm the CLASSIC path (option off)
    never touches the sequence-only methods.
    """

    name = "fake_sequence_op"

    def __init__(
        self,
        *,
        media_requirement=MediaRequirement.METADATA,
        group_by: bool = False,
        raise_for_row_id: str | None = None,
        build_error: Exception | None = None,
        cancel_token: CancellationToken | None = None,
        cancel_after_row_id: str | None = None,
    ):
        super().__init__()
        parameters = [
            BooleanParameter(
                name="use_sequence", label="Use sequence",
                required=False, default=False,
            ),
        ]
        if group_by:
            parameters.append(
                ColumnParameter(
                    name="group_col", label="Group by",
                    from_input="active_table",
                )
            )
        self.descriptor = OperatorDescriptor(
            name=self.name,
            version="1.0",
            description="Test double: sequence runner (P2.4a).",
            modes=(
                ModeDescriptor(
                    mode=ExecutionMode.COLUMNS,
                    label="Fake sequence op",
                    inputs=(
                        InputSpec(
                            name="active_table", label="Active table",
                            kind=InputKind.ACTIVE_TABLE,
                        ),
                    ),
                    media_requirement=media_requirement,
                    parameters=tuple(parameters),
                    output=OutputSpec(columns=(
                        OutputColumn(name="out", type_tag="numeric"),
                    )),
                    model_lifecycle=ModelLifecycle.PER_WORKER,
                    sequence_option=SequenceOption(
                        enabled_by="use_sequence",
                        group_by="group_col" if group_by else None,
                    ),
                ),
            ),
        )
        self._raise_for_row_id = raise_for_row_id
        self._build_error = build_error
        self._cancel_token = cancel_token
        self._cancel_after_row_id = cancel_after_row_id

        self.lock = threading.Lock()
        self._next_model_id = itertools.count(1)

        # Sequence-path instrumentation.
        self.build_calls: list[int] = []           # model ids, in build order
        self.build_thread_ids: list[int] = []
        self.calls: list[tuple] = []                # (model_id, row_id, media_time_us, thread_id)

        # Classic-path instrumentation (option-off test only).
        self.classic_build_count = 0
        self.classic_rows_seen: list[str] = []

    # -- sequence path ----------------------------------------------------

    def build_sequence_model(self, run):
        if self._build_error is not None:
            raise self._build_error
        model_id = next(self._next_model_id)
        with self.lock:
            self.build_calls.append(model_id)
            self.build_thread_ids.append(threading.get_ident())
        return {"id": model_id}

    def create_columns_in_sequence(self, row_id, media, metadata, run, media_time_us):
        model_id = run.model["id"]
        with self.lock:
            self.calls.append((model_id, row_id, media_time_us, threading.get_ident()))
        if self._cancel_token is not None and row_id == self._cancel_after_row_id:
            self._cancel_token.cancel()
        if self._raise_for_row_id == row_id:
            raise ValueError(f"boom on {row_id}")
        return {"out": float(len(row_id))}

    # -- classic path (option-off test only) -------------------------------

    def build_model(self):
        with self.lock:
            self.classic_build_count += 1
        return {"id": "classic"}

    def create_columns(self, row_id, media, metadata, run):
        with self.lock:
            self.classic_rows_seen.append(row_id)
        return {"out": float(len(row_id))}


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
    op_registry, operator, snapshot, row_ids, run, monkeypatch, *,
    worker_count=2, media_column=None,
):
    """Starts run_create_columns() at the OperatorRegistry seam (a real
    background thread tree, joined deterministically), and collects
    every callback into plain structures -- no Qt event loop, no
    controller. Returns a dict: results, row_errors, completions,
    setup_errors, recording_thread_ids (every thread ident that called
    on_item_complete / on_row_errors / on_progress / on_setup_error /
    on_row_finished -- must all be the SAME thread, the coordinator)."""
    op_registry.register(operator)
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

    def _on_row_finished(operation_id):
        recording_thread_ids.append(threading.get_ident())

    def _on_complete(operation_id, operator_name, emitted, elapsed_seconds=0.0, worker_count=1):
        completions.append((operation_id, operator_name, emitted, worker_count))

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
            on_row_finished=_on_row_finished,
        )
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)

    assert started, "run_create_columns did not start a worker"
    for thread in created:
        thread.join(timeout=15)
        assert not thread.is_alive(), "a thread did not finish in time (possible hang)"

    return {
        "results": results,
        "row_errors": row_errors,
        "completions": completions,
        "setup_errors": setup_errors,
        "recording_thread_ids": recording_thread_ids,
    }


# ===========================================================================
# 1. Descriptor validation.
# ===========================================================================

def test_sequence_option_validation():
    """SequenceOption naming a missing or wrong-typed parameter raises;
    on a non-COLUMNS mode raises; combined with model_lifecycle SHARED
    raises.

    Would still pass if SequenceOption's cross-checks were skipped
    entirely? No -- every pytest.raises block below would fail with
    "DID NOT RAISE", and the valid-baseline construction at the top
    would be the only thing that ran.
    """
    active_table_input = (
        InputSpec(
            name="active_table", label="Active table", kind=InputKind.ACTIVE_TABLE,
        ),
    )

    def _mode(mode=ExecutionMode.COLUMNS, parameters=(), model_lifecycle=ModelLifecycle.NONE,
              sequence_option=None):
        if mode is ExecutionMode.COLUMNS:
            output = OutputSpec(columns=(OutputColumn(name="out", type_tag="numeric"),))
        elif mode is ExecutionMode.TABLE:
            output = OutputSpec(creates_table=True)
        else:
            output = OutputSpec(is_display_only=True)
        return ModeDescriptor(
            mode=mode,
            label="Test mode",
            inputs=active_table_input,
            parameters=parameters,
            output=output,
            model_lifecycle=model_lifecycle,
            sequence_option=sequence_option,
        )

    good_bool = BooleanParameter(name="use_sequence", label="Use sequence")
    good_col = ColumnParameter(name="group_col", label="Group", from_input="active_table")

    # Valid baseline -- must NOT raise.
    _mode(parameters=(good_bool,), sequence_option=SequenceOption(enabled_by="use_sequence"))

    # enabled_by names nothing declared by this mode.
    with pytest.raises(OperatorDescriptorError):
        _mode(parameters=(good_bool,), sequence_option=SequenceOption(enabled_by="missing"))

    # enabled_by names the wrong type (a ColumnParameter, not Boolean).
    with pytest.raises(OperatorDescriptorError):
        _mode(parameters=(good_col,), sequence_option=SequenceOption(enabled_by="group_col"))

    # group_by names nothing declared by this mode.
    with pytest.raises(OperatorDescriptorError):
        _mode(
            parameters=(good_bool,),
            sequence_option=SequenceOption(enabled_by="use_sequence", group_by="missing"),
        )

    # group_by names the wrong type (a BooleanParameter, not Column).
    with pytest.raises(OperatorDescriptorError):
        _mode(
            parameters=(good_bool, BooleanParameter(name="not_a_column", label="Not a column")),
            sequence_option=SequenceOption(enabled_by="use_sequence", group_by="not_a_column"),
        )

    # Non-COLUMNS mode.
    with pytest.raises(OperatorDescriptorError):
        _mode(
            mode=ExecutionMode.TABLE,
            parameters=(good_bool,),
            sequence_option=SequenceOption(enabled_by="use_sequence"),
        )

    # Combined with model_lifecycle SHARED.
    with pytest.raises(OperatorDescriptorError):
        _mode(
            parameters=(good_bool,),
            model_lifecycle=ModelLifecycle.SHARED,
            sequence_option=SequenceOption(enabled_by="use_sequence"),
        )


# ===========================================================================
# 2. runs_as_sequences / sequence_group_column truth table.
# ===========================================================================

def test_runs_as_sequences_truth_table():
    """Declared PER_SEQUENCE; option on; option off; no option at all.

    Would still pass if runs_as_sequences() ignored model_lifecycle
    PER_SEQUENCE and only ever consulted sequence_option? No -- the
    "declared" assertions below would fail (that mode has no
    sequence_option at all).
    """
    active_table_input = (
        InputSpec(
            name="active_table", label="Active table", kind=InputKind.ACTIVE_TABLE,
        ),
    )
    bool_param = BooleanParameter(
        name="use_sequence", label="Use sequence", required=False, default=False,
    )
    col_param = ColumnParameter(name="group_col", label="Group", from_input="active_table")
    output = OutputSpec(columns=(OutputColumn(name="out", type_tag="numeric"),))

    declared = ModeDescriptor(
        mode=ExecutionMode.COLUMNS, label="Test mode", inputs=active_table_input,
        parameters=(bool_param, col_param), output=output,
        model_lifecycle=ModelLifecycle.PER_SEQUENCE,
    )
    with_option_no_group = ModeDescriptor(
        mode=ExecutionMode.COLUMNS, label="Test mode", inputs=active_table_input,
        parameters=(bool_param,), output=output,
        sequence_option=SequenceOption(enabled_by="use_sequence"),
    )
    with_option_and_group = ModeDescriptor(
        mode=ExecutionMode.COLUMNS, label="Test mode", inputs=active_table_input,
        parameters=(bool_param, col_param), output=output,
        sequence_option=SequenceOption(enabled_by="use_sequence", group_by="group_col"),
    )
    no_option = ModeDescriptor(
        mode=ExecutionMode.COLUMNS, label="Test mode", inputs=active_table_input,
        parameters=(bool_param, col_param), output=output,
    )

    # Declared PER_SEQUENCE -- always True, whatever the parameters say.
    assert runs_as_sequences(declared, {}) is True
    assert runs_as_sequences(declared, {"use_sequence": False}) is True
    assert sequence_group_column(declared, {}) is None

    # Option present, turned on.
    params_on = {"use_sequence": True, "group_col": "cond"}
    assert runs_as_sequences(with_option_and_group, params_on) is True
    assert sequence_group_column(with_option_and_group, params_on) == "cond"

    # Option present, turned off.
    params_off = {"use_sequence": False, "group_col": "cond"}
    assert runs_as_sequences(with_option_and_group, params_off) is False

    # Option present but this mode declares no group_by parameter.
    assert runs_as_sequences(with_option_no_group, {"use_sequence": True}) is True
    assert sequence_group_column(with_option_no_group, {"use_sequence": True}) is None

    # No sequence_option at all -- never runs as sequences.
    assert runs_as_sequences(no_option, {"use_sequence": True}) is False
    assert sequence_group_column(no_option, {}) is None


# ===========================================================================
# 3-5. Sequence formation over real video.
# ===========================================================================

@_needs_video
def test_one_video_no_group_one_model_ascending_order(monkeypatch):
    """One video, no group column: one model instance; rows arrive in
    ascending media_time_us; every row delivered once."""
    video_path = VIDEO_PATHS[0]
    resolver = MediaResolver(max_open_decoders=4)
    try:
        frame_times = resolver.get_frame_times(from_path(str(video_path)))
        n_frames = min(len(frame_times), 15)
        ordinals = list(range(n_frames))
        random.Random(7).shuffle(ordinals)  # shuffled table row order
        row_ids = [f"r{i}" for i in ordinals]
        cells = [_frame_cell(video_path, i) for i in ordinals]
        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells})

        op = _FakeSequenceOperator(media_requirement=MediaRequirement.FRAME)
        run = _build_run(op, {"use_sequence": True}, resolver=resolver)
        out = _run_and_join(
            OperatorRegistry(), op, snapshot, row_ids, run, monkeypatch,
            worker_count=3, media_column="media",
        )

        assert not out["row_errors"], out["row_errors"]
        assert set(out["results"]) == set(row_ids)
        assert len(op.build_calls) == 1, (
            "one video, no grouping column -- one sequence, one model"
        )
        assert len(op.calls) == n_frames

        delivered_times = [media_time_us for (_m, _r, media_time_us, _th) in op.calls]
        assert delivered_times == sorted(delivered_times), (
            "rows were not delivered in ascending media_time_us order"
        )
        assert None not in delivered_times

        delivered_row_ids = [row_id for (_m, row_id, _t, _th) in op.calls]
        assert sorted(delivered_row_ids) == sorted(row_ids)
    finally:
        resolver.close()


@_needs_video
def test_group_column_three_values_three_models_no_crossing(monkeypatch):
    """Group column with 3 values over one video (interleaved in the
    table's row order): 3 distinct model instances; no model ever sees
    rows of two groups; ascending order within each."""
    video_path = VIDEO_PATHS[0]
    resolver = MediaResolver(max_open_decoders=4)
    try:
        frame_times = resolver.get_frame_times(from_path(str(video_path)))
        assert len(frame_times) >= 12, "fixture too short for a meaningful 3-group test"

        groups = {"a": [0, 3, 6], "b": [1, 4, 7], "c": [2, 5, 8]}
        entries = [
            (group_value, ordinal)
            for group_value, ordinals in groups.items()
            for ordinal in ordinals
        ]
        random.Random(42).shuffle(entries)  # interleaved table row order

        row_ids = [f"{g}_{i}" for i, (g, _o) in enumerate(entries)]
        cells = [_frame_cell(video_path, o) for (_g, o) in entries]
        conds = [g for (g, _o) in entries]
        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells, "cond": conds})
        expected_group_by_row = dict(zip(row_ids, conds))

        op = _FakeSequenceOperator(media_requirement=MediaRequirement.FRAME, group_by=True)
        run = _build_run(op, {"use_sequence": True, "group_col": "cond"}, resolver=resolver)
        out = _run_and_join(
            OperatorRegistry(), op, snapshot, row_ids, run, monkeypatch,
            worker_count=3, media_column="media",
        )

        assert not out["row_errors"], out["row_errors"]
        assert set(out["results"]) == set(row_ids)
        assert len(op.build_calls) == 3, "3 distinct group values -- 3 sequences, 3 models"

        rows_by_model: dict = {}
        for model_id, row_id, media_time_us, _th in op.calls:
            rows_by_model.setdefault(model_id, []).append((row_id, media_time_us))
        assert len(rows_by_model) == 3
        for model_id, rows in rows_by_model.items():
            groups_seen = {expected_group_by_row[row_id] for row_id, _t in rows}
            assert len(groups_seen) == 1, (
                f"model {model_id} saw rows from more than one group: {groups_seen}"
            )
            times = [t for _r, t in rows]
            assert times == sorted(times)
    finally:
        resolver.close()


@_needs_video
def test_missing_group_value_forms_one_sequence_not_one_row_each(monkeypatch):
    """Every row of one source whose group value is missing (None/NaN)
    forms ONE sequence together, in ascending frame order -- exactly
    like any other group value, never split into one-row sequences (that
    would give each a cold model and no tracking). A row whose group
    value happens to be the literal STRING "_missing_" is a real value
    and must stay in its own group, never folded into the missing-value
    group by the sentinel the runner uses internally to mean "missing".
    """
    video_path = VIDEO_PATHS[0]
    resolver = MediaResolver(max_open_decoders=4)
    try:
        frame_times = resolver.get_frame_times(from_path(str(video_path)))
        assert len(frame_times) >= 7, "fixture too short for this test"

        # Interleaved, in table row order: A, missing, B, missing, A,
        # missing, then one more row whose group value is the literal
        # string "_missing_" -- not a missing value at all.
        values = ["A", None, "B", None, "A", None, "_missing_"]
        ordinals = list(range(len(values)))
        row_ids = [f"r{i}" for i in ordinals]
        cells = [_frame_cell(video_path, i) for i in ordinals]
        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells, "cond": values})

        op = _FakeSequenceOperator(media_requirement=MediaRequirement.FRAME, group_by=True)
        run = _build_run(op, {"use_sequence": True, "group_col": "cond"}, resolver=resolver)
        out = _run_and_join(
            OperatorRegistry(), op, snapshot, row_ids, run, monkeypatch,
            worker_count=3, media_column="media",
        )

        assert not out["row_errors"], out["row_errors"]
        assert set(out["results"]) == set(row_ids)

        model_id_by_row = {row_id: model_id for (model_id, row_id, _t, _th) in op.calls}

        a_rows = ["r0", "r4"]
        missing_rows = ["r1", "r3", "r5"]
        b_rows = ["r2"]
        literal_row = "r6"

        # The A / missing / B groups: exactly 3 sequences, 3 models.
        core_model_ids = {model_id_by_row[r] for r in a_rows + missing_rows + b_rows}
        assert len(core_model_ids) == 3, (
            "expected exactly 3 sequences for the A / missing / B groups"
        )
        assert len({model_id_by_row[r] for r in a_rows}) == 1, (
            "both 'A' rows must share one model"
        )
        assert len({model_id_by_row[r] for r in missing_rows}) == 1, (
            "all three missing-value rows must share ONE model -- a "
            "missing value is not its own one-row group"
        )
        assert len({model_id_by_row[r] for r in b_rows}) == 1

        # The missing-value rows are delivered in ascending
        # media_time_us order, on their shared model -- preserved even
        # though other sequences' rows are interleaved with theirs in
        # op.calls' own (global, cross-sequence) delivery order.
        missing_calls_in_order = [
            (row_id, media_time_us)
            for (_m, row_id, media_time_us, _th) in op.calls
            if row_id in missing_rows
        ]
        assert {row_id for row_id, _t in missing_calls_in_order} == set(missing_rows)
        times = [t for _r, t in missing_calls_in_order]
        assert times == sorted(times), (
            "missing-value rows were not delivered in ascending "
            "media_time_us order"
        )

        # The literal string "_missing_" is a REAL value: its row must
        # NOT share a model with the actual missing-value rows, and must
        # not be folded into the 3-model core count above.
        assert model_id_by_row[literal_row] != model_id_by_row[missing_rows[0]], (
            "a group value equal to the literal string '_missing_' was "
            "folded into the missing-value group"
        )
        assert model_id_by_row[literal_row] not in core_model_ids
    finally:
        resolver.close()


@_needs_two_videos
def test_two_videos_at_least_two_models(monkeypatch):
    """Two videos: at least 2 model instances even with no group
    column -- two sources must never share one sequence's model."""
    video_a, video_b = VIDEO_PATHS[0], VIDEO_PATHS[1]
    resolver = MediaResolver(max_open_decoders=4)
    try:
        row_ids = [f"a{i}" for i in range(5)] + [f"b{i}" for i in range(5)]
        cells = (
            [_frame_cell(video_a, i) for i in range(5)]
            + [_frame_cell(video_b, i) for i in range(5)]
        )
        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells})

        op = _FakeSequenceOperator(media_requirement=MediaRequirement.FRAME)
        run = _build_run(op, {"use_sequence": True}, resolver=resolver)
        out = _run_and_join(
            OperatorRegistry(), op, snapshot, row_ids, run, monkeypatch,
            worker_count=3, media_column="media",
        )

        assert not out["row_errors"], out["row_errors"]
        assert set(out["results"]) == set(row_ids)
        assert len(op.build_calls) >= 2
    finally:
        resolver.close()


# ===========================================================================
# 6. worker_count 1 vs 3: identical results.
# ===========================================================================

def test_worker_count_1_vs_3_identical_results(monkeypatch):
    """Same table, worker_count 1 vs 3: identical results per row.

    Would still pass if worker_count silently changed a row's computed
    value (e.g. by leaking state between models)? No -- dict equality
    below compares every row's own result.
    """
    row_ids = [f"r{i}" for i in range(12)]
    snapshot = _metadata_snapshot(row_ids)

    op1 = _FakeSequenceOperator()
    out1 = _run_and_join(
        OperatorRegistry(), op1, snapshot, row_ids,
        _build_run(op1, {"use_sequence": True}), monkeypatch, worker_count=1,
    )

    op3 = _FakeSequenceOperator()
    out3 = _run_and_join(
        OperatorRegistry(), op3, snapshot, row_ids,
        _build_run(op3, {"use_sequence": True}), monkeypatch, worker_count=3,
    )

    assert not out1["row_errors"] and not out3["row_errors"]
    assert set(out1["results"]) == set(row_ids)
    assert out1["results"] == out3["results"]


# ===========================================================================
# 7. Only the coordinator thread records.
# ===========================================================================

def test_only_coordinator_records(monkeypatch):
    """Only the coordinator thread calls on_item_complete /
    on_row_finished / on_progress / on_setup_error (same style as
    tests/test_parallel_columns_runner.py).

    Would still pass if a worker thread called on_item_complete
    directly? No -- recording_thread_ids would contain more than one
    distinct thread ident.
    """
    row_ids = [f"r{i}" for i in range(9)]
    snapshot = _metadata_snapshot(row_ids)
    op = _FakeSequenceOperator()
    out = _run_and_join(
        OperatorRegistry(), op, snapshot, row_ids,
        _build_run(op, {"use_sequence": True}), monkeypatch, worker_count=4,
    )

    assert not out["row_errors"]
    assert set(out["results"]) == set(row_ids)
    assert out["recording_thread_ids"], "no recording callback fired at all"
    assert len(set(out["recording_thread_ids"])) == 1


# ===========================================================================
# 8. Row error resets the sequence's model.
# ===========================================================================

@_needs_video
def test_row_error_resets_the_sequence_model(monkeypatch):
    """Operator raises on one row: that row is a row error, the next
    row of the same sequence is delivered by a NEW model instance."""
    video_path = VIDEO_PATHS[0]
    resolver = MediaResolver(max_open_decoders=4)
    try:
        frame_times = resolver.get_frame_times(from_path(str(video_path)))
        n_frames = min(len(frame_times), 10)
        assert n_frames >= 4, "fixture too short for a meaningful reset test"
        ordinals = list(range(n_frames))
        row_ids = [f"r{i}" for i in ordinals]
        cells = [_frame_cell(video_path, i) for i in ordinals]
        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells})

        bad_row_id = row_ids[n_frames // 2]
        op = _FakeSequenceOperator(
            media_requirement=MediaRequirement.FRAME, raise_for_row_id=bad_row_id,
        )
        run = _build_run(op, {"use_sequence": True}, resolver=resolver)
        out = _run_and_join(
            OperatorRegistry(), op, snapshot, row_ids, run, monkeypatch,
            worker_count=1, media_column="media",
        )

        error_row_ids = {row_id for row_id, _kind, _msg in out["row_errors"]}
        assert error_row_ids == {bad_row_id}
        assert set(out["results"]) == set(row_ids)

        delivered_order = [
            (row_id, model_id) for (model_id, row_id, _t, _th) in op.calls
        ]
        bad_index = next(
            i for i, (row_id, _m) in enumerate(delivered_order) if row_id == bad_row_id
        )
        assert bad_index + 1 < len(delivered_order), (
            "the bad row must not be the last row of its sequence for "
            "this test to be meaningful"
        )
        bad_model_id = delivered_order[bad_index][1]
        next_model_id = delivered_order[bad_index + 1][1]
        assert next_model_id != bad_model_id, (
            "the sequence's model was not reset after a row error"
        )
        assert len(op.build_calls) >= 2, (
            "expected at least the initial build plus one reset build"
        )
    finally:
        resolver.close()


# ===========================================================================
# 9. build_sequence_model raising aborts the run with zero rows delivered.
# ===========================================================================

def test_build_sequence_model_setup_error_aborts_with_zero_rows(monkeypatch):
    row_ids = [f"r{i}" for i in range(5)]
    snapshot = _metadata_snapshot(row_ids)
    op = _FakeSequenceOperator(build_error=OperatorSetupError("model file missing (test)"))
    out = _run_and_join(
        OperatorRegistry(), op, snapshot, row_ids,
        _build_run(op, {"use_sequence": True}), monkeypatch, worker_count=3,
    )

    assert out["results"] == {}
    assert out["row_errors"] == []
    assert out["setup_errors"], "no setup error reached the researcher"
    assert "model file missing (test)" in out["setup_errors"][-1]
    assert out["completions"], "on_complete must still fire so the run is not left live"
    assert out["completions"][0][2] == 0, "no row should have been processed"


# ===========================================================================
# 10. Cancellation: no new row started after cancel.
# ===========================================================================

def test_cancel_after_first_rows_no_new_row_started(monkeypatch):
    row_ids = [f"r{i}" for i in range(10)]
    snapshot = _metadata_snapshot(row_ids)
    token = CancellationToken()
    op = _FakeSequenceOperator(cancel_token=token, cancel_after_row_id="r2")
    run = _build_run(op, {"use_sequence": True}, token=token)
    # worker_count=1: sequences are taken strictly in queue (row_ids)
    # order, so the outcome is deterministic rather than racing threads.
    out = _run_and_join(
        OperatorRegistry(), op, snapshot, row_ids, run, monkeypatch, worker_count=1,
    )

    seen_row_ids = [row_id for (_m, row_id, _t, _th) in op.calls]
    assert seen_row_ids == ["r0", "r1", "r2"], (
        "a row was started after cancellation"
    )
    assert set(out["results"]) == {"r0", "r1", "r2"}
    assert out["completions"], "on_complete must still fire for a cancelled run"


# ===========================================================================
# 11. Option off: classic path, not the sequence runner.
# ===========================================================================

def test_option_off_uses_classic_path(monkeypatch):
    row_ids = [f"r{i}" for i in range(5)]
    snapshot = _metadata_snapshot(row_ids)
    op = _FakeSequenceOperator()
    out = _run_and_join(
        OperatorRegistry(), op, snapshot, row_ids,
        _build_run(op, {"use_sequence": False}), monkeypatch, worker_count=1,
    )

    assert not out["row_errors"]
    assert set(out["results"]) == set(row_ids)
    assert op.classic_build_count == 1, (
        "expected exactly one build_model() call (PER_WORKER, serial path)"
    )
    assert sorted(op.classic_rows_seen) == sorted(row_ids)
    assert op.build_calls == [], "build_sequence_model() must not be called with the option off"
    assert op.calls == [], "create_columns_in_sequence() must not be called with the option off"


# ===========================================================================
# 12. A still-image row with the option on.
# ===========================================================================

def test_still_image_sequence_media_time_us_is_none(monkeypatch):
    still_paths = sorted(TEST_IMAGES.glob("*.jpg"))
    assert still_paths, "need at least one test_images fixture"
    still_path = still_paths[0]
    cell = format_address(from_path(str(still_path)))

    row_ids = ["only_row"]
    snapshot = pd.DataFrame({"row_id": row_ids, "media": [cell]})

    resolver = MediaResolver(max_open_decoders=4)
    try:
        op = _FakeSequenceOperator(media_requirement=MediaRequirement.FRAME)
        run = _build_run(op, {"use_sequence": True}, resolver=resolver)
        out = _run_and_join(
            OperatorRegistry(), op, snapshot, row_ids, run, monkeypatch,
            worker_count=2, media_column="media",
        )

        assert not out["row_errors"], out["row_errors"]
        assert set(out["results"]) == {"only_row"}
        assert len(op.build_calls) == 1, "a still image is its own one-row sequence"
        assert len(op.calls) == 1
        _model_id, row_id, media_time_us, _thread_id = op.calls[0]
        assert row_id == "only_row"
        assert media_time_us is None, "a still image has no timeline"
    finally:
        resolver.close()
