"""
tests/test_frame_operator.py

Tests for operators/frame_operator.py, written from the work-item
specification (docs/media_architecture.md §6.2's "P1.7 Frame operator"
paragraph, operators/CLAUDE.md's "Carry the lineage columns" and
"Reporting a row you could not process") -- not from the implementation.

The video fixture is generated on demand with ffmpeg, following the
pattern of tests/test_media_resolver.py's own
_generate_known_frame_video -- that file is not imported (the work item's
file list does not permit editing it), so the small piece of the pattern
this module needs is copied here. Every frame-time expectation below is
computed independently from the fixture's own known, constant-frame-rate
spacing (fps=25 -> exactly 40,000 microseconds per frame -- confirmed
exact, not merely close, by tests/test_media_resolver.py's own
test_get_frame_times_sorted_and_matches_frame_count), never by calling
the same frames_in_range()/get_frame_times() machinery under test to
build its own expected answer.

Most of these tests exercise FrameOperator.create_table() directly (a
plain Python call, not through OperatorRegistry's background-thread
machinery) plus a real MediaResolver against a real fixture. The
OSError-escape regression test (P1.7-2 round 5) is the one exception: it
drives the real OperatorRegistry.run_create_table() path instead, because
the defect it guards is specifically about what escapes TO the registry.
Neither this module nor that one test imports Qt or controller.py --
matching tests/test_table_output_contract.py's own reasoning for staying
out of run_tests.py's isolated widget group; proving the literal
AppController-level run "outcome" field (controller.py's own
_run_outcome()) would need a real AppController, which needs a live
QApplication, which would move this module into that isolated group and
require editing run_tests.py's WIDGET_MODULES -- out of scope this round.
That test instead proves the mechanism CLAUDE.md's "Long-running work"
rule says decides the outcome: on_row_errors fires (which is what makes
AppController record "partial") and on_error never fires (which is what
would make it "failed").

Run with: python -m pytest tests/test_frame_operator.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import pytest
from PIL import Image

from media.media_address import MediaAddress, format as format_address, parse as parse_address
from media.resolver import MediaResolver
from models.dataset import Dataset
from models.table_schema import ColumnRole
from operators.base import OperatorSetupError
from operators.descriptor import ExecutionMode
from operators.frame_operator import FrameOperator
from operators.operator_registry import OperatorRegistry
from operators.run_context import (
    CancellationToken,
    OperatorRun,
    OperatorRunSpec,
    RunData,
    TableSnapshot,
)

FFMPEG_MISSING = shutil.which("ffmpeg") is None
pytestmark = pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg is not on PATH")

FPS = 25
DURATION_S = 2
FRAME_PERIOD_US = 1_000_000 // FPS  # exactly 40,000us -- see module docstring
N_FRAMES = FPS * DURATION_S  # 50


def _generate_known_frame_video(tmp_path, width=64, height=64):
    """A lossless, constant-frame-rate video. Copied from
    tests/test_media_resolver.py's helper of the same purpose (that file
    is not imported, per the work item's file list).
    """
    out_path = tmp_path / "known_frames.mkv"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s={width}x{height}:r={FPS}:d={DURATION_S}",
            "-vf", "format=gray,geq=lum='N'",
            "-pix_fmt", "gray",
            "-c:v", "ffv1",
            str(out_path),
        ],
        check=True, capture_output=True,
    )
    return out_path


def _build_run(operator, resolver, df, carry_columns, parameters=None):
    """A minimal, valid OperatorRun for a TABLE run of `operator`, wrapping
    a real MediaResolver -- mirrors
    tests/test_table_output_contract.py's _build_table_run, extended with
    a real resolver and a controllable carry_columns tuple (advisory data
    this operator reads directly, per run_context.py's TableSnapshot
    docstring).
    """
    mode_descriptor = operator.descriptor.mode_for(ExecutionMode.TABLE)
    if parameters is None:
        parameters = {
            "media_column": "full_path",
            "frame_step": 1,
            "output_table": "frame_rows",
        }
    spec = OperatorRunSpec(
        operation_id="op-1",
        operator_name=operator.name,
        mode=ExecutionMode.TABLE,
        mode_descriptor=mode_descriptor,
        parameters=parameters,
        target_table="",
    )
    snapshot = TableSnapshot(
        table_name="active_table_name",
        frame=df,
        version=1,
        carry_columns=tuple(carry_columns),
    )
    return OperatorRun(
        spec=spec,
        data=RunData(tables={"active_table": snapshot}, projects={}),
        paths=object(),
        resolver=resolver,
        _token=CancellationToken(),
    )


# ---------------------------------------------------------------------------
# A bare whole-video row yields real #f= addresses whose ordinals match the
# file, one row per frame at the default step.
# ---------------------------------------------------------------------------

def test_bare_video_row_produces_real_f_addresses_matching_file_ordinals(tmp_path):
    video_path = _generate_known_frame_video(tmp_path)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        df = pd.DataFrame({
            "row_id": ["r0"],
            "full_path": [str(video_path)],
        })
        operator = FrameOperator()
        run = _build_run(operator, resolver, df, carry_columns=())

        result = operator.create_table(df, run)

        # Would still pass if the operator fabricated ordinals (e.g. its
        # own enumeration counter) instead of asking the resolver? No --
        # this fixture is CFR from frame 0, so a fabricated counter would
        # coincidentally match here. The assertion below on the ADDRESS
        # STRING itself (not just frame_index) is what catches that: each
        # address must parse back to the same ordinal AND the same source
        # path, which only the resolver's real index guarantees.
        assert len(result) == N_FRAMES
        assert sorted(result["frame_index"].tolist()) == list(range(N_FRAMES))

        for _, row in result.iterrows():
            addr = parse_address(row["frame_address"])
            assert addr.frame == row["frame_index"]
            assert addr.time_range_us is None and addr.time_us is None
            assert Path(addr.path) == Path(video_path)
            expected_time = row["frame_index"] * FRAME_PERIOD_US / 1_000_000
            assert row["frame_time"] == pytest.approx(expected_time)
            # A bare row's range starts at the stream's own zero (decision
            # 10), so time_within_segment must equal frame_time exactly --
            # not merely be present.
            assert row["time_within_segment"] == pytest.approx(row["frame_time"])
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# A segment row yields only the frames inside its own range -- decision 3's
# half-open membership, computed here from the fixture's own known constant
# frame period, independently of frames_in_range().
# ---------------------------------------------------------------------------

def test_segment_row_yields_only_frames_inside_its_range(tmp_path):
    video_path = _generate_known_frame_video(tmp_path)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        start_us, end_us = 500_000, 1_000_000
        segment_addr = MediaAddress(path=str(video_path), time_range_us=(start_us, end_us))
        df = pd.DataFrame({
            "row_id": ["r0"],
            "full_path": [format_address(segment_addr)],
        })
        operator = FrameOperator()
        run = _build_run(operator, resolver, df, carry_columns=())

        result = operator.create_table(df, run)

        # Independent membership computation: frame n's time is exactly
        # n * FRAME_PERIOD_US (this fixture's own known CFR spacing), and
        # decision 3's rule is start <= p < end.
        expected_indices = [
            n for n in range(N_FRAMES)
            if start_us <= n * FRAME_PERIOD_US < end_us
        ]
        assert len(expected_indices) > 1, "test fixture parameters produced a degenerate range"

        assert sorted(result["frame_index"].tolist()) == expected_indices

        # Every produced address must fall inside the SAME range -- guards
        # against an off-by-one that leaks in a neighbouring frame (the
        # same failure mode operators/CLAUDE.md's segment-thumbnail rule
        # warns about for the analogous case).
        for _, row in result.iterrows():
            pts = row["frame_index"] * FRAME_PERIOD_US
            assert start_us <= pts < end_us
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# time_within_segment is measured from the segment's own declared start,
# not from the first frame the range happens to capture.
# ---------------------------------------------------------------------------

def test_time_within_segment_measured_from_segment_start_not_first_captured_frame(tmp_path):
    video_path = _generate_known_frame_video(tmp_path)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        # 510,000us does not land on a frame boundary (frames land on
        # multiples of 40,000us): the first frame actually captured is at
        # 520,000us (n=13), ten milliseconds later. If time_within_segment
        # were measured from that first captured frame instead of the
        # declared start, its value for that frame would be 0 -- this test
        # is exactly what tells the two apart.
        start_us, end_us = 510_000, 900_000
        segment_addr = MediaAddress(path=str(video_path), time_range_us=(start_us, end_us))
        df = pd.DataFrame({
            "row_id": ["r0"],
            "full_path": [format_address(segment_addr)],
        })
        operator = FrameOperator()
        run = _build_run(operator, resolver, df, carry_columns=())

        result = operator.create_table(df, run).sort_values("frame_index")

        first_row = result.iloc[0]
        assert first_row["frame_index"] == 13  # 13 * 40_000 = 520_000
        expected_time_within_segment = (13 * FRAME_PERIOD_US - start_us) / 1_000_000
        assert expected_time_within_segment == pytest.approx(0.01)
        assert first_row["time_within_segment"] == pytest.approx(expected_time_within_segment)
        assert first_row["time_within_segment"] != pytest.approx(0.0)
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# frame_step keeps every Nth frame of the row's own range.
# ---------------------------------------------------------------------------

def test_frame_step_keeps_every_nth_frame(tmp_path):
    video_path = _generate_known_frame_video(tmp_path)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        df = pd.DataFrame({
            "row_id": ["r0"],
            "full_path": [str(video_path)],
        })
        operator = FrameOperator()
        parameters = {"media_column": "full_path", "frame_step": 5, "output_table": "frame_rows"}
        run = _build_run(operator, resolver, df, carry_columns=(), parameters=parameters)

        result = operator.create_table(df, run)

        # Would still pass if step were ignored? No -- that would produce
        # 50 rows, not 10.
        assert sorted(result["frame_index"].tolist()) == list(range(0, N_FRAMES, 5))
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# Carried columns match carry_columns, narrowed of the media column and of
# segment_index (which the operator re-adds itself -- see the module's own
# note on why).
# ---------------------------------------------------------------------------

def test_carried_columns_match_carry_columns_and_narrow_media_and_segment_index(tmp_path):
    video_path = _generate_known_frame_video(tmp_path)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        df = pd.DataFrame({
            "row_id": ["r0"],
            "full_path": [str(video_path)],
            "participant_id": ["p07"],
            "reaction_time": [0.842],
            "segment_index": [3],
            "untracked_column": ["should not appear"],
        })
        operator = FrameOperator()
        # carry_columns deliberately includes full_path (the media column)
        # and segment_index -- exactly what Dataset.columns_to_carry()
        # would hand a real caller, since both are always-carried
        # (identifier / index) columns on a real segment table. It
        # deliberately excludes untracked_column, to prove narrowing is
        # respected in both directions.
        carry_columns = ("participant_id", "full_path", "reaction_time", "segment_index")
        run = _build_run(operator, resolver, df, carry_columns=carry_columns)

        result = operator.create_table(df, run)

        # Would still pass if the operator carried everything in df,
        # ignoring carry_columns? No -- untracked_column would appear.
        assert "untracked_column" not in result.columns
        assert "participant_id" in result.columns
        assert "reaction_time" in result.columns
        assert (result["participant_id"] == "p07").all()
        assert (result["reaction_time"] == 0.842).all()

        # full_path itself must not survive as a second copy of the media
        # reference (operators/CLAUDE.md's own full_path example).
        assert "full_path" not in result.columns

        # segment_index must appear exactly once (the operator's own
        # column, not a duplicate), carrying the SOURCE row's value
        # through -- not the bare-row fallback of 0.
        assert list(result.columns).count("segment_index") == 1
        assert (result["segment_index"] == 3).all()
    finally:
        resolver.close()


def test_bare_row_with_no_segment_index_column_gets_synthesised_zero(tmp_path):
    video_path = _generate_known_frame_video(tmp_path)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        df = pd.DataFrame({"row_id": ["r0"], "full_path": [str(video_path)]})
        operator = FrameOperator()
        run = _build_run(operator, resolver, df, carry_columns=())

        result = operator.create_table(df, run)

        assert (result["segment_index"] == 0).all()
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# The declared role/type_tag hints reach the stored table's schema -- the
# first real consumer of OutputSpec.table_columns for a media-address
# output column (operators/CLAUDE.md, P1.7-1).
# ---------------------------------------------------------------------------

def test_declared_roles_and_type_tag_reach_stored_schema(tmp_path):
    video_path = _generate_known_frame_video(tmp_path)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        df = pd.DataFrame({"row_id": ["r0"], "full_path": [str(video_path)]})
        operator = FrameOperator()
        op_registry = OperatorRegistry()
        op_registry.register(operator)
        run = _build_run(operator, resolver, df, carry_columns=())

        hints = op_registry.hints_for_table_output(operator.name)
        result = operator.create_table(df, run)

        ds = Dataset()
        ds.create_table_from_df("built", result, hints=hints)
        schema = ds.schema_for("built")

        # Would still pass if the declared hints were dropped? No -- an
        # undeclared int column defaults to role=measurement
        # (infer_schema's own rule), which is wrong for a lineage index;
        # an undeclared text column would still often read as media_path
        # by value-scan coincidence, which is exactly why frame_index and
        # segment_index (both plain ints, no coincidental default) are the
        # sharper half of this assertion.
        assert schema.spec_for("frame_index").role is ColumnRole.index
        assert schema.spec_for("segment_index").role is ColumnRole.index
        assert schema.spec_for("frame_address").type_tag == "media_path"
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# A bad row is reported through run.report_row_error(), not silently
# dropped -- and does not stop the rest of the run.
# ---------------------------------------------------------------------------

def test_missing_media_row_is_reported_and_other_rows_still_process(tmp_path):
    video_path = _generate_known_frame_video(tmp_path)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        df = pd.DataFrame({
            "row_id": ["r0", "r1"],
            "full_path": [None, str(video_path)],
        })
        operator = FrameOperator()
        run = _build_run(operator, resolver, df, carry_columns=())

        result = operator.create_table(df, run)

        # Would still pass if the bad row silently vanished with no
        # report? No -- collected_row_errors() would be empty.
        errors = run.collected_row_errors()
        assert len(errors) == 1
        row_id, kind, _message = errors[0]
        assert row_id == "r0"
        assert kind == "MissingMedia"

        # The good row (r1) still produced its full 50 frames -- one bad
        # row must not kill the run (operators/CLAUDE.md, "Rules").
        assert len(result) == N_FRAMES
    finally:
        resolver.close()


def test_time_range_outside_the_file_is_reported_not_raised(tmp_path):
    video_path = _generate_known_frame_video(tmp_path)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        far_future = MediaAddress(path=str(video_path), time_range_us=(10_000_000, 10_500_000))
        df = pd.DataFrame({
            "row_id": ["r0", "r1"],
            "full_path": [format_address(far_future), str(video_path)],
        })
        operator = FrameOperator()
        run = _build_run(operator, resolver, df, carry_columns=())

        # Must not raise -- the whole point of report_row_error() is that
        # a bad row is reported, never allowed to propagate as an
        # exception that kills the run.
        result = operator.create_table(df, run)

        errors = run.collected_row_errors()
        assert len(errors) == 1
        row_id, kind, _message = errors[0]
        assert row_id == "r0"
        assert kind == "InvalidTimeRange"

        # r1 (the ordinary bare row) still produced its frames.
        assert len(result) == N_FRAMES
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# A still image row -- decided: one frame row at #f=0, not a skipped row.
# See operators/frame_operator.py's own comment at the call site for why.
# ---------------------------------------------------------------------------

def test_still_image_row_emits_one_frame_at_f_equals_zero(tmp_path):
    image_path = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path, "PNG")

    resolver = MediaResolver(max_open_decoders=2)
    try:
        df = pd.DataFrame({"row_id": ["r0"], "full_path": [str(image_path)]})
        operator = FrameOperator()
        run = _build_run(operator, resolver, df, carry_columns=())

        result = operator.create_table(df, run)

        assert len(run.collected_row_errors()) == 0
        assert len(result) == 1
        row = result.iloc[0]
        addr = parse_address(row["frame_address"])
        assert addr.frame == 0
        assert row["frame_index"] == 0
        assert pd.isna(row["frame_time"])
        assert pd.isna(row["time_within_segment"])
        assert row["segment_index"] == 0
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# P1.7-2 round 5: an OSError-family exception from get_frame_times() (a
# missing, permission-denied or otherwise unreadable file reaching
# av.open() inside the resolver's decoder pool) must be reported for that
# one row through run.report_row_error(), not allowed to escape create_table()
# and fail the whole run at the registry level.
#
# Driven through the real OperatorRegistry.run_create_table() path -- not
# a direct create_table() call -- because the defect is specifically about
# what escapes TO the registry: the registry's own try/except around
# create_table() is what turns an uncaught exception into a whole-run
# failure that discards every row already produced, which is exactly what
# operators/CLAUDE.md's "one bad row must not kill a run" rule forbids.
#
# The OSError is entirely real, not mocked: row "r_bad" names a PATH THAT
# IS A DIRECTORY, not a file that does not exist (P1.7-2 round 6's own fix
# 2 added an upfront os.path.exists() check that reports a genuinely
# nonexistent path as MissingMedia before get_frame_times() is ever
# called -- that is a different, earlier report, guarded by round 6's own
# test, and would no longer reach this one). A directory still passes
# that exists() check (a directory exists) but is not a video file, so
# MediaResolver.get_frame_times() -> the decoder pool's av.open() still
# raises a real OSError-family exception -- confirmed by hand:
# av.error.PermissionError, which subclasses both av.error.FFmpegError and
# Python's builtin PermissionError, itself an OSError -- no monkeypatching
# of the resolver or the operator.
# ---------------------------------------------------------------------------

def _run_table_and_join(op_registry, operator, df, run, monkeypatch):
    """Starts run_create_table(), joins the worker thread it spawns, and
    records every callback call plus the order they arrived in -- no Qt,
    no controller. Mirrors tests/test_table_output_contract.py's helper of
    the same name and purpose (that module is not imported -- a test
    module is not a library -- so the small piece of the pattern this
    module needs is duplicated here).
    """
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
        thread.join(timeout=15)
        assert not thread.is_alive(), "worker thread did not finish in time"

    return completions, errors, row_error_calls, call_order


def test_unreadable_media_row_is_reported_not_raised_through_the_real_registry(
    tmp_path, monkeypatch,
):
    video_path = _generate_known_frame_video(tmp_path)
    # A directory, not a missing path -- see the block comment above for
    # why round 6's new existence check means a merely-nonexistent path no
    # longer reaches this test's scenario.
    bad_path = tmp_path / "a_directory_not_a_video.mp4"
    bad_path.mkdir()
    assert bad_path.exists()

    resolver = MediaResolver(max_open_decoders=2)
    try:
        df = pd.DataFrame({
            "row_id": ["r_before", "r_bad", "r_after"],
            "full_path": [str(video_path), str(bad_path), str(video_path)],
            "source_label": ["before", "bad", "after"],
        })
        operator = FrameOperator()
        op_registry = OperatorRegistry()
        op_registry.register(operator)
        run = _build_run(operator, resolver, df, carry_columns=("source_label",))

        completions, errors, row_error_calls, call_order = _run_table_and_join(
            op_registry, operator, df, run, monkeypatch,
        )

        # Would still pass if the OSError escaped and the registry treated
        # the whole run as failed? No -- errors would be non-empty and
        # completions would be empty, which is exactly the defect
        # /code-review found (operators/operator_registry.py's whole-run
        # `except Exception` catching an uncaught OSError from
        # create_table()).
        assert errors == [], f"the run must not fail at the registry level: {errors}"
        assert len(completions) == 1

        # The bad row is reported, not silently dropped and not merged
        # into some other row's report.
        assert len(row_error_calls) == 1
        _operation_id, _label, reported = row_error_calls[0]
        assert len(reported) == 1
        row_id, kind, _message = reported[0]
        assert row_id == "r_bad"
        assert kind == "UnreadableMedia"

        # The mechanism CLAUDE.md's "Long-running work" rule says decides
        # AppController's recorded outcome: on_row_errors fired (-> the
        # controller records "partial") and on_error never fired (-> it is
        # never "failed"). See the module docstring for why the literal
        # outcome field itself is not asserted here.
        assert call_order == ["row_errors", "complete"]

        # The rows BEFORE and AFTER the bad one still produced their full
        # set of frames -- one bad row must not kill the run
        # (operators/CLAUDE.md, "Rules": "Fail gracefully per row").
        _op_id, _op_name, result_df = completions[0]
        labels_present = set(result_df["source_label"].unique())
        assert labels_present == {"before", "after"}, (
            "a row before and a row after the bad one must both still "
            f"produce frames; got labels {labels_present}"
        )
        assert (result_df["source_label"] == "before").sum() == N_FRAMES
        assert (result_df["source_label"] == "after").sum() == N_FRAMES
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# P1.7-2 round 6 -- three defects an independent review found.
# ---------------------------------------------------------------------------

def test_a_carried_column_colliding_with_an_own_column_refuses_the_run(tmp_path):
    # Fix 1. A table this operator itself produced earlier already has a
    # "frame_index" column; running the operator on it again must refuse
    # before touching any row, name the colliding column, and leave the
    # researcher's input table exactly as it was.
    video_path = _generate_known_frame_video(tmp_path)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        df = pd.DataFrame({
            "row_id": ["r0"],
            "full_path": [str(video_path)],
            "frame_index": [999],
        })
        df_before = df.copy(deep=True)
        operator = FrameOperator()
        run = _build_run(operator, resolver, df, carry_columns=("frame_index",))

        # Would still pass if the operator instead silently dropped the
        # carried "frame_index" or renamed it? No -- no exception would be
        # raised at all, and this assertion is the only thing that proves
        # one was.
        with pytest.raises(OperatorSetupError) as excinfo:
            operator.create_table(df, run)
        assert "frame_index" in str(excinfo.value)

        # The researcher's input table is untouched -- create_table() must
        # not mutate df even on the refusal path.
        pd.testing.assert_frame_equal(df, df_before)
    finally:
        resolver.close()


def test_nonexistent_image_row_is_reported_and_other_rows_still_produce_frames(tmp_path):
    # Fix 2. The middle row names a .png that was never created. It must
    # be reported through report_row_error with kind MissingMedia -- not
    # silently turned into a successful #f=0 output row -- and the rows
    # before and after it must still produce their frames.
    video_path = _generate_known_frame_video(tmp_path)
    missing_image_path = tmp_path / "does_not_exist.png"
    assert not missing_image_path.exists()

    resolver = MediaResolver(max_open_decoders=2)
    try:
        df = pd.DataFrame({
            "row_id": ["r_before", "r_missing_image", "r_after"],
            "full_path": [str(video_path), str(missing_image_path), str(video_path)],
            "source_label": ["before", "missing", "after"],
        })
        operator = FrameOperator()
        run = _build_run(operator, resolver, df, carry_columns=("source_label",))

        result = operator.create_table(df, run)

        # Would still pass if the missing image were silently accepted?
        # No -- collected_row_errors() would be empty and "missing" would
        # appear in result's source_label.
        errors = run.collected_row_errors()
        assert len(errors) == 1
        row_id, kind, _message = errors[0]
        assert row_id == "r_missing_image"
        assert kind == "MissingMedia"

        labels_present = set(result["source_label"].unique())
        assert labels_present == {"before", "after"}
        assert (result["source_label"] == "before").sum() == N_FRAMES
        assert (result["source_label"] == "after").sum() == N_FRAMES
    finally:
        resolver.close()


def test_segment_row_with_no_source_segment_index_gets_missing_value_bare_row_gets_zero(tmp_path):
    # Fix 3. Neither row carries a source segment_index column at all. The
    # bare row is the trivial single-segment case, so 0 is right for it;
    # the #t= row names some particular range of the source and has no
    # principled number, so it must come out missing, not a fabricated 0.
    # A carried "source_label" column tells the two rows' output apart
    # unambiguously -- both address the SAME file, so their frame_index
    # values overlap and cannot be used for that on their own.
    video_path = _generate_known_frame_video(tmp_path)
    segment_addr = MediaAddress(path=str(video_path), time_range_us=(500_000, 1_000_000))

    resolver = MediaResolver(max_open_decoders=2)
    try:
        df = pd.DataFrame({
            "row_id": ["r_bare", "r_segment"],
            "full_path": [str(video_path), format_address(segment_addr)],
            "source_label": ["bare", "segment"],
        })
        assert "segment_index" not in df.columns
        operator = FrameOperator()
        run = _build_run(operator, resolver, df, carry_columns=("source_label",))

        result = operator.create_table(df, run)

        bare_out = result[result["source_label"] == "bare"]
        segment_out = result[result["source_label"] == "segment"]
        assert len(bare_out) == N_FRAMES
        assert len(segment_out) > 0

        # Would still pass if both rows still got the pre-fix 0? No --
        # segment_out's assertion below (isna(), not == 0) would fail.
        assert (bare_out["segment_index"] == 0).all()
        assert segment_out["segment_index"].isna().all()
    finally:
        resolver.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
