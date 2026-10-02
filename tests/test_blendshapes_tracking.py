"""
tests/test_blendshapes_tracking.py

P2.4b: "Extract blendshapes" gains a tracking option built on the
P2.4a sequence runner (operators/descriptor.py's SequenceOption,
runs_as_sequences, sequence_group_column; operators/base.py's
build_sequence_model / create_columns_in_sequence;
operators/operator_registry.py's _run_create_columns_sequenced).

Written from the work-item spec, not the implementation. The descriptor
test needs only mediapipe importable; every other test here calls the
REAL BlendshapeOperator with the REAL face_landmarker.task model over the
vids/ fixture videos (gitignored, the same directory
tests/test_sequence_runner.py uses) and skips cleanly when either is
absent. Every test that waits on threads joins with a timeout, so a hang
fails instead of freezing the suite.

Run with:
    python -m pytest tests/test_blendshapes_tracking.py
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

pytest.importorskip("mediapipe", reason="mediapipe not installed")

import mediapipe as mp

import operators.blendshapes as blendshapes_module
from operators.blendshapes import BLENDSHAPE_NAMES, BlendshapeOperator, _MODEL_PATH
from operators.descriptor import ExecutionMode, runs_as_sequences, sequence_group_column
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
TEST_IMAGES = project_root / "test_images"

_MODEL_MISSING = not _MODEL_PATH.exists()

_needs_model_and_video = pytest.mark.skipif(
    _MODEL_MISSING or not VIDEO_PATHS,
    reason=(
        "face_landmarker.task is not downloaded, or the vids/ fixture "
        "videos are not present on this machine"
    ),
)
_needs_model_and_stills = pytest.mark.skipif(
    _MODEL_MISSING or not sorted(TEST_IMAGES.glob("*.jpg")),
    reason="face_landmarker.task is not downloaded, or test_images/ is empty",
)


# ---------------------------------------------------------------------------
# Scaffolding. Deliberately duplicated from tests/test_sequence_runner.py's
# own builders rather than imported -- a test module is not a library.
# ---------------------------------------------------------------------------

def _frame_cell(path: Path, ordinal: int) -> str:
    return format_address(dataclasses.replace(from_path(str(path)), frame=ordinal))


def _wrap_build_model(monkeypatch):
    """Wraps BlendshapeOperator.build_model to record every call. Does
    NOT close anything -- operators/operator_registry.py now closes every
    model it builds itself, via operator.close_model(), on the thread
    that used it last; these tests rely on that rather than managing
    landmarker lifetime themselves. (An earlier version of this file had
    these wrappers build and close their own tracked list of landmarkers,
    as a test-side workaround before the runner did this -- see
    docs/review/p2_4b_mediapipe_close_hang.md for why that was needed
    then and is not needed now.)"""
    calls: list[bool] = []
    original = BlendshapeOperator.build_model

    def _tracked(self):
        calls.append(True)
        return original(self)

    monkeypatch.setattr(BlendshapeOperator, "build_model", _tracked)
    return calls


def _wrap_build_sequence_model(monkeypatch):
    """Same as _wrap_build_model, for build_sequence_model."""
    calls: list[bool] = []
    original = BlendshapeOperator.build_sequence_model

    def _tracked(self, run):
        calls.append(True)
        return original(self, run)

    monkeypatch.setattr(BlendshapeOperator, "build_sequence_model", _tracked)
    return calls


def _build_run(operator, parameters, resolver) -> OperatorRun:
    mode_descriptor = operator.descriptor.mode_for(ExecutionMode.COLUMNS)
    spec = OperatorRunSpec(
        operation_id="op-1",
        operator_name=operator.name,
        mode=ExecutionMode.COLUMNS,
        mode_descriptor=mode_descriptor,
        parameters=parameters,
        target_table="frames",
    )
    return OperatorRun(
        spec=spec,
        data=RunData(tables={}, projects={}),
        paths=object(),
        resolver=resolver,
        _token=CancellationToken(),
    )


def _run_and_join(
    op_registry, operator, snapshot, row_ids, run, monkeypatch, *,
    worker_count=2, media_column="media",
):
    """Runs run_create_columns() through the real OperatorRegistry seam, a
    real background thread tree joined deterministically -- the same
    style tests/test_sequence_runner.py uses, but exercising the REAL
    BlendshapeOperator and REAL MediaPipe rather than a fake."""
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
        thread.join(timeout=90)
        assert not thread.is_alive(), "a thread did not finish in time (possible hang)"

    return {"results": results, "row_errors": row_errors, "setup_errors": setup_errors}


# ===========================================================================
# 1. Descriptor.
# ===========================================================================

def test_descriptor_declares_tracking_and_group_parameters():
    """Both parameters are declared; sequence_option names them; version
    is "1.1"; runs_as_sequences is False with the default parameters and
    True with track_face=True.

    Would still pass if track_face/sequence_column were never wired into
    a SequenceOption? No -- md.sequence_option would be None and the
    runs_as_sequences() assertions below would fail.
    """
    md = BlendshapeOperator.descriptor.mode_for(ExecutionMode.COLUMNS)
    param_names = {p.name for p in md.parameters}

    assert "track_face" in param_names
    assert "sequence_column" in param_names
    assert md.sequence_option is not None
    assert md.sequence_option.enabled_by == "track_face"
    assert md.sequence_option.group_by == "sequence_column"
    assert BlendshapeOperator.descriptor.version == "1.1"

    assert runs_as_sequences(md, {}) is False
    assert runs_as_sequences(md, {"track_face": False}) is False
    assert runs_as_sequences(md, {"track_face": True}) is True
    assert sequence_group_column(md, {}) is None
    assert sequence_group_column(md, {"sequence_column": "trial_id"}) == "trial_id"


# ===========================================================================
# 2. Tracking off: identical to today's per-image behaviour.
# ===========================================================================

@_needs_model_and_video
def test_tracking_off_matches_build_model_and_detect(monkeypatch):
    """The output for rows of a vids/ video, tracking off, is identical
    to calling build_model() + detect() directly (today's behaviour,
    unrouted through any sequence machinery).

    Would still pass if create_columns() started calling detect_for_video
    or otherwise changed its own per-row behaviour? No -- the per-row
    dict equality check below would fail on any changed score.
    """
    video_path = VIDEO_PATHS[0]
    resolver = MediaResolver(max_open_decoders=4)
    try:
        frame_times = resolver.get_frame_times(from_path(str(video_path)))
        n_frames = min(len(frame_times), 8)
        ordinals = list(range(n_frames))
        row_ids = [f"r{i}" for i in ordinals]
        cells = [_frame_cell(video_path, i) for i in ordinals]
        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells})

        # Reference: build_model() + detect(), directly, one call per frame.
        reference_op = BlendshapeOperator()
        reference_run = _build_run(reference_op, {}, resolver=resolver)
        landmarker = reference_op.build_model()
        try:
            expected = {}
            for row_id, ordinal in zip(row_ids, ordinals):
                payload = resolver.resolve_frame(
                    _frame_cell(video_path, ordinal), "analysis"
                )
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=payload.pixels
                )
                detection_result = landmarker.detect(mp_image)
                expected[row_id] = blendshapes_module._scores_from_detection_result(
                    detection_result, row_id, reference_run
                )
        finally:
            landmarker.close()

        _build_calls = _wrap_build_model(monkeypatch)
        op = BlendshapeOperator()
        run = _build_run(op, {}, resolver=resolver)
        out = _run_and_join(
            OperatorRegistry(), op, snapshot, row_ids, run, monkeypatch,
            worker_count=1,
        )

        assert not out["row_errors"], out["row_errors"]
        assert out["results"] == expected
    finally:
        resolver.close()


def test_option_off_never_calls_the_sequence_methods(monkeypatch):
    """With track_face left at its default (False), the classic PER_WORKER
    path is used: build_sequence_model() and create_columns_in_sequence()
    are never called. Uses METADATA media so it needs no video fixture.

    Would still pass if the descriptor's sequence_option accidentally
    made every run sequenced regardless of track_face? No -- the call
    counters below would be nonzero.
    """
    build_seq_calls = []
    seq_calls = []
    original_build_seq = BlendshapeOperator.build_sequence_model
    original_seq = BlendshapeOperator.create_columns_in_sequence

    def _tracked_build_seq(self, run):
        build_seq_calls.append(True)
        return original_build_seq(self, run)

    def _tracked_seq(self, row_id, media, metadata, run, media_time_us):
        seq_calls.append(True)
        return original_seq(self, row_id, media, metadata, run, media_time_us)

    monkeypatch.setattr(BlendshapeOperator, "build_sequence_model", _tracked_build_seq)
    monkeypatch.setattr(BlendshapeOperator, "create_columns_in_sequence", _tracked_seq)

    if _MODEL_MISSING:
        pytest.skip("face_landmarker.task is not downloaded")

    row_ids = ["r0", "r1", "r2"]
    still_paths = sorted(TEST_IMAGES.glob("*.jpg"))
    if not still_paths:
        pytest.skip("need at least one test_images fixture")
    cell = format_address(from_path(str(still_paths[0])))
    snapshot = pd.DataFrame({"row_id": row_ids, "media": [cell] * 3})

    _build_model_calls = _wrap_build_model(monkeypatch)
    op = BlendshapeOperator()
    resolver = MediaResolver(max_open_decoders=2)
    try:
        run = _build_run(op, {"track_face": False}, resolver=resolver)
        out = _run_and_join(
            OperatorRegistry(), op, snapshot, row_ids, run, monkeypatch,
            worker_count=1,
        )
        assert not out["row_errors"], out["row_errors"]
        assert set(out["results"]) == set(row_ids)
    finally:
        resolver.close()

    assert build_seq_calls == [], "build_sequence_model() must not be called with tracking off"
    assert seq_calls == [], "create_columns_in_sequence() must not be called with tracking off"


# ===========================================================================
# 3. Tracking on: one VIDEO-mode model per sequence, deterministic.
# ===========================================================================

@_needs_model_and_video
def test_tracking_on_one_video_mode_model_per_sequence_deterministic(monkeypatch):
    """A run completes over the frame rows of one vids/ video with
    tracking on: exactly one model is built for the one (ungrouped)
    sequence, every _landmarker_options() call for this run asked for
    VIDEO running mode, and the results are identical across two runs.

    Would still pass if build_sequence_model() silently built an
    IMAGE-mode landmarker? No -- the recorded running_modes list would
    contain None (IMAGE, mediapipe's default) instead of VIDEO.
    """
    video_path = VIDEO_PATHS[0]

    def _one_run():
        running_modes = []
        original_options = blendshapes_module._landmarker_options

        def _tracked_options(running_mode=None):
            running_modes.append(running_mode)
            return original_options(running_mode=running_mode)

        monkeypatch.setattr(blendshapes_module, "_landmarker_options", _tracked_options)
        build_seq_calls = _wrap_build_sequence_model(monkeypatch)

        resolver = MediaResolver(max_open_decoders=4)
        try:
            frame_times = resolver.get_frame_times(from_path(str(video_path)))
            n_frames = min(len(frame_times), 10)
            ordinals = list(range(n_frames))
            row_ids = [f"r{i}" for i in ordinals]
            cells = [_frame_cell(video_path, i) for i in ordinals]
            snapshot = pd.DataFrame({"row_id": row_ids, "media": cells})

            op = BlendshapeOperator()
            run = _build_run(op, {"track_face": True}, resolver=resolver)
            out = _run_and_join(
                OperatorRegistry(), op, snapshot, row_ids, run, monkeypatch,
                worker_count=3,
            )
            return out, build_seq_calls, running_modes, row_ids
        finally:
            resolver.close()

    out1, build_seq_calls1, running_modes1, row_ids = _one_run()
    assert not out1["row_errors"], out1["row_errors"]
    assert set(out1["results"]) == set(row_ids)
    assert len(build_seq_calls1) == 1, "one video, no grouping -- one sequence, one model"
    assert running_modes1, "no _landmarker_options() call was recorded"
    assert all(mode == mp.tasks.vision.RunningMode.VIDEO for mode in running_modes1), (
        f"expected every model built for a tracking-on run to be VIDEO mode, "
        f"got {running_modes1}"
    )

    out2, _build_seq_calls2, _running_modes2, _row_ids2 = _one_run()
    assert not out2["row_errors"], out2["row_errors"]
    assert out1["results"] == out2["results"], (
        "tracking-on results were not deterministic across two runs"
    )


# ===========================================================================
# 4. sequence_column splits one source into more than one sequence.
# ===========================================================================

@_needs_model_and_video
def test_sequence_column_splits_rows_into_separate_sequences(monkeypatch):
    """A sequence_column with 2 distinct values over one video: 2
    distinct build_sequence_model() calls -- rows split by group value,
    not treated as one sequence.

    Would still pass if sequence_column were ignored by the runner? No
    -- build_seq_calls would have length 1 (one video, no real split).
    """
    video_path = VIDEO_PATHS[0]
    resolver = MediaResolver(max_open_decoders=4)
    try:
        frame_times = resolver.get_frame_times(from_path(str(video_path)))
        assert len(frame_times) >= 8, "fixture too short for a meaningful split test"

        groups = {"a": [0, 2, 4], "b": [1, 3, 5]}
        entries = [
            (group_value, ordinal)
            for group_value, ordinals in groups.items()
            for ordinal in ordinals
        ]
        row_ids = [f"{g}_{i}" for i, (g, _o) in enumerate(entries)]
        cells = [_frame_cell(video_path, o) for (_g, o) in entries]
        conds = [g for (g, _o) in entries]
        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells, "cond": conds})

        build_seq_calls = _wrap_build_sequence_model(monkeypatch)

        op = BlendshapeOperator()
        run = _build_run(
            op, {"track_face": True, "sequence_column": "cond"}, resolver=resolver,
        )
        out = _run_and_join(
            OperatorRegistry(), op, snapshot, row_ids, run, monkeypatch,
            worker_count=3,
        )

        assert not out["row_errors"], out["row_errors"]
        assert set(out["results"]) == set(row_ids)
        assert len(build_seq_calls) == 2, "2 distinct group values -- 2 sequences, 2 models"
    finally:
        resolver.close()


# ===========================================================================
# 5. Still-image rows with tracking on: each is its own one-row sequence.
# ===========================================================================

@_needs_model_and_stills
def test_still_images_tracking_on_each_is_its_own_sequence(monkeypatch):
    """Two still-image rows with tracking on: each is its own one-row
    sequence -- 2 build_sequence_model() calls -- and the run completes.

    Would still pass if still images were folded into a single shared
    sequence? No -- build_seq_calls would have length 1.
    """
    still_paths = sorted(TEST_IMAGES.glob("*.jpg"))
    assert len(still_paths) >= 1, "need at least one test_images fixture"
    # Re-use the same image twice if only one is available -- two ROWS,
    # each its own one-row sequence, is the property under test, not two
    # distinct source files.
    paths = [still_paths[0], still_paths[min(1, len(still_paths) - 1)]]
    cells = [format_address(from_path(str(p))) for p in paths]
    row_ids = ["r0", "r1"]
    snapshot = pd.DataFrame({"row_id": row_ids, "media": cells})

    build_seq_calls = _wrap_build_sequence_model(monkeypatch)

    op = BlendshapeOperator()
    resolver = MediaResolver(max_open_decoders=2)
    try:
        run = _build_run(op, {"track_face": True}, resolver=resolver)
        out = _run_and_join(
            OperatorRegistry(), op, snapshot, row_ids, run, monkeypatch,
            worker_count=2,
        )
    finally:
        resolver.close()

    assert not out["row_errors"], out["row_errors"]
    assert set(out["results"]) == set(row_ids)
    assert len(build_seq_calls) == 2, "each still image must be its own one-row sequence"


# ===========================================================================
# 6. A duplicate #f= address inside one sequence: MediaPipe's own
# monotonically-increasing-timestamp requirement surfaces as a row error,
# and the sequence's model is reset (Step 1 item 3).
# ===========================================================================

@_needs_model_and_video
def test_duplicate_frame_in_a_sequence_is_a_row_error_and_resets_the_model(monkeypatch):
    """Two rows naming the SAME #f= frame inside one ungrouped sequence:
    decode_frames_in_order yields that frame twice (once per row, its own
    documented behaviour for a duplicate address), so the second
    detect_for_video() call receives a timestamp equal to the first --
    MediaPipe raises ValueError ("Input timestamp must be monotonically
    increasing."), which _classify_row_result maps to a row error, and
    the sequence continues on a freshly reset model.

    Would still pass if the runner silently skipped a duplicate address
    instead of feeding it to MediaPipe twice? No -- out["row_errors"]
    would be empty.
    """
    video_path = VIDEO_PATHS[0]
    resolver = MediaResolver(max_open_decoders=4)
    try:
        frame_times = resolver.get_frame_times(from_path(str(video_path)))
        assert len(frame_times) >= 4, "fixture too short for this test"

        # r0 and r1 both name frame 0 -- decode_frames_in_order answers
        # both, in input order, at the SAME presentation time. r2 names a
        # later frame and must still be delivered (the sequence
        # continues after the reset).
        row_ids = ["r0", "r1", "r2"]
        cells = [
            _frame_cell(video_path, 0),
            _frame_cell(video_path, 0),
            _frame_cell(video_path, min(3, len(frame_times) - 1)),
        ]
        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells})

        build_seq_calls = _wrap_build_sequence_model(monkeypatch)

        op = BlendshapeOperator()
        run = _build_run(op, {"track_face": True}, resolver=resolver)
        # No test-side model closing here -- operators/operator_registry.py
        # now closes the discarded (reset) landmarker itself, via
        # operator.close_model(), right after the row error, on the
        # worker thread that used it. See docs/review/p2_4b_mediapipe_close_hang.md
        # for why relying on garbage collection to do this used to hang.
        out = _run_and_join(
            OperatorRegistry(), op, snapshot, row_ids, run, monkeypatch,
            worker_count=1,
        )

        assert len(out["row_errors"]) == 1, out["row_errors"]
        error_row_id, error_kind, error_message = out["row_errors"][0]
        assert error_row_id == "r1", (
            "the SECOND row naming the duplicated frame must be the one "
            "that fails -- the first establishes the timestamp"
        )
        assert error_kind == "ValueError"
        assert "monotonically increasing" in error_message

        # Every row still gets an item_complete entry, including the
        # error row -- _run_create_columns_sequenced's own _record calls
        # on_item_complete with an all-None dict for a "row_error" outcome
        # (the same thing _classify_row_result hands back), exactly as the
        # non-sequenced paths already do.
        assert set(out["results"]) == {"r0", "r1", "r2"}
        assert out["results"]["r1"] == {name: None for name in BLENDSHAPE_NAMES}

        # The failure reset the sequence's model: at least the initial
        # build plus one reset build.
        assert len(build_seq_calls) >= 2, (
            "expected at least the initial build plus one reset build "
            "after the row error"
        )
    finally:
        resolver.close()
