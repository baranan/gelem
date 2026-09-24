"""
tests/test_ordered_frame_runner.py

P2.1: the ordered sequential frame runner for a FRAME-mode COLUMNS run.
Exercises two seams:

  (1) MediaResolver.decode_frames_in_order() -- decodes a list of #f=
      addresses on ONE source in one sequential pass, yielding
      (address, FramePayload) pairs in ascending ordinal order.
  (2) OperatorRegistry._run_create_columns_worker's grouping of a
      FRAME-mode COLUMNS run's #f= rows by source, decoded through (1)
      and fed to operator.iter_column_updates() (operators/base.py's
      serial reference, unless an operator overrides it -- none does
      here).

Also covers the media-column fix folded into this item: the worker no
longer hardcodes "full_path" -- AppController.run_create_columns decides
the column (get_detail_media_column()) and passes it through as
media_column, so a table whose media lives under a different column name
(e.g. "Split into frames"' own frame_address) is no longer silently
refused.

Written from the work-item spec, not the implementation. Each test
states, in a comment, what would still pass if the rule it guards were
broken.

Uses the real, short recordings in vids/ at the project root (vid11.mp4,
vid22.mp4, vid33.mp4, vid44.mp4 -- none longer than a few seconds), the
same fixture directory tests/test_clip_frame_cache.py uses. That
directory is gitignored, so this whole module skips cleanly when it is
not present on this machine, rather than failing.

No Qt widget is shown here, so the run-tests substring heuristic puts
this module in the combined group. No `# run-tests:` token is needed.

Run with:
    python -m pytest tests/test_ordered_frame_runner.py
"""

from __future__ import annotations

import dataclasses
import random
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
from operators.operator_registry import OperatorRegistry
from operators.run_context import (
    CancellationToken,
    OperatorRun,
    OperatorRunSpec,
    RunData,
)
from media.media_address import MediaAddressError, format as format_address, from_path
from media.resolver import MediaResolver
from models.dataset import Dataset
from models.table_schema import ColumnHint
from models.query_engine import QueryEngine
from artifacts.artifact_store import ArtifactStore
from column_types.registry import ColumnTypeRegistry
from controller import AppController

TEST_IMAGES = project_root / "test_images"
VIDS_DIR = project_root / "vids"
VIDEO_PATHS = sorted(VIDS_DIR.glob("*.mp4")) if VIDS_DIR.is_dir() else []

pytestmark = pytest.mark.skipif(
    not VIDEO_PATHS, reason="vids/ fixture videos are not present on this machine"
)


# ---------------------------------------------------------------------------
# Small builders. Deliberately duplicated from other test modules' own
# descriptor scaffolding rather than imported -- a test module is not a
# library (see tests/test_run_log.py's own note on this).
# ---------------------------------------------------------------------------

def _pixel_stats_descriptor(name, label="Pixel stats"):
    return OperatorDescriptor(
        name=name,
        version="1.0",
        description="Test double: per-channel mean of the decoded frame.",
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.COLUMNS,
                label=label,
                inputs=(
                    InputSpec(
                        name="active_table", label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                media_requirement=MediaRequirement.FRAME,
                parameters=(),
                output=OutputSpec(columns=(
                    OutputColumn(name="mean_r", type_tag="numeric"),
                    OutputColumn(name="mean_g", type_tag="numeric"),
                    OutputColumn(name="mean_b", type_tag="numeric"),
                )),
            ),
        ),
    )


def _pixel_stats(media) -> dict:
    return {
        "mean_r": float(media[:, :, 0].mean()),
        "mean_g": float(media[:, :, 1].mean()),
        "mean_b": float(media[:, :, 2].mean()),
    }


class _PixelStatsOperator(BaseOperator):
    """FRAME-requirement operator returning the per-channel mean of the
    decoded frame. Does NOT override iter_column_updates() -- every test
    below exercises BaseOperator's own serial reference implementation,
    per the work item's "none does in this item"."""

    name = "pixel_stats"
    descriptor = _pixel_stats_descriptor("pixel_stats")

    def __init__(self):
        super().__init__()
        self.rows_seen: list[str] = []

    def create_columns(self, row_id, media, metadata, run):
        self.rows_seen.append(row_id)
        return _pixel_stats(media)


class _CancelsAfterRow(BaseOperator):
    """Like _PixelStatsOperator, but cancels the run's own token once it
    has processed a chosen row -- standing in for "the researcher
    clicked Cancel while this run was in progress" deterministically."""

    name = "cancels_after_row"
    descriptor = _pixel_stats_descriptor("cancels_after_row", "Cancels after row")

    def __init__(self, token: CancellationToken, cancel_after_row_id: str):
        super().__init__()
        self._token = token
        self._cancel_after_row_id = cancel_after_row_id
        self.rows_seen: list[str] = []

    def create_columns(self, row_id, media, metadata, run):
        self.rows_seen.append(row_id)
        if row_id == self._cancel_after_row_id:
            self._token.cancel()
        return _pixel_stats(media)


class _RaisesOnOneFrame(BaseOperator):
    """Like _PixelStatsOperator, but raises on exactly one chosen row --
    an ordinary operator bug (a malformed frame, a library crash), not a
    setup error and not "does not implement create_columns()". Does NOT
    override iter_column_updates(), so the runner must isolate this
    row's failure the same way the per-row path always has."""

    name = "raises_on_one_frame"
    descriptor = _pixel_stats_descriptor("raises_on_one_frame", "Raises on one frame")

    def __init__(self, bad_row_id: str):
        super().__init__()
        self._bad_row_id = bad_row_id
        self.rows_seen: list[str] = []

    def create_columns(self, row_id, media, metadata, run):
        self.rows_seen.append(row_id)
        if row_id == self._bad_row_id:
            raise RuntimeError("simulated per-row failure")
        return _pixel_stats(media)


def _frame_cell(path: Path, ordinal: int) -> str:
    """The stored-cell string for a #f=<ordinal> address on `path`."""
    return format_address(dataclasses.replace(from_path(str(path)), frame=ordinal))


def _expected_means(resolver: MediaResolver, cell: str) -> dict:
    return _pixel_stats(resolver.resolve_frame(cell, "analysis").pixels)


def _build_run(operator: BaseOperator, resolver: MediaResolver, token=None) -> OperatorRun:
    mode_descriptor = operator.descriptor.mode_for(ExecutionMode.COLUMNS)
    spec = OperatorRunSpec(
        operation_id="op-1",
        operator_name=operator.name,
        mode=ExecutionMode.COLUMNS,
        mode_descriptor=mode_descriptor,
        parameters={},
        target_table="frame_rows",
    )
    return OperatorRun(
        spec=spec,
        data=RunData(tables={}, projects={}),
        paths=object(),
        resolver=resolver,
        _token=token or CancellationToken(),
    )


def _run_columns_and_collect(
    op_registry, operator, snapshot, row_ids, run, monkeypatch,
    media_column="media",
):
    """Starts run_create_columns() at the OperatorRegistry seam (a real
    background thread, joined deterministically), and collects every
    callback into plain structures -- no Qt event loop, no controller.
    Returns (results: dict[row_id, dict], row_errors: list[tuple], on_
    complete_calls: list[tuple])."""
    results: dict[str, dict] = {}
    row_errors: list[tuple] = []
    completions: list[tuple] = []

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

    def _on_complete(operation_id, operator_name, emitted):
        completions.append((operation_id, operator_name, emitted))

    monkeypatch.setattr(threading, "Thread", _Tracked)
    try:
        started = op_registry.run_create_columns(
            operator.name, snapshot, row_ids, "frame_rows", run,
            operation_id="op-1",
            on_item_complete=_on_item_complete,
            on_progress=None,
            on_complete=_on_complete,
            on_setup_error=None,
            on_row_errors=_on_row_errors,
            media_column=media_column,
        )
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)

    assert started, "run_create_columns did not start a worker"
    for thread in created:
        thread.join(timeout=30)
        assert not thread.is_alive(), "worker thread did not finish in time"

    return results, row_errors, completions


# ===========================================================================
# MediaResolver.decode_frames_in_order()
# ===========================================================================

def test_decode_frames_in_order_matches_resolve_frame_pixel_for_pixel():
    """Every frame of a short clip, decoded through decode_frames_in_order,
    is pixel-identical to the same frame decoded independently through
    resolve_frame(#f=n) one at a time, and comes back in ascending
    ordinal order -- the ordered path changes HOW frames are decoded,
    never WHAT is decoded.

    Would still pass if decode_frames_in_order silently returned the
    wrong frame for some ordinal (an off-by-one, say)? No -- the
    array_equal check would catch a pixel mismatch even if the count and
    ordering were otherwise right.
    """
    video_path = VIDEO_PATHS[0]
    resolver = MediaResolver(max_open_decoders=4)
    try:
        n_frames = len(resolver.get_frame_times(from_path(str(video_path))))
        addresses = [
            dataclasses.replace(from_path(str(video_path)), frame=i)
            for i in range(n_frames)
        ]
        expected = {
            i: resolver.resolve_frame(addresses[i], "analysis").pixels
            for i in range(n_frames)
        }

        seen_ordinals: list[int] = []
        for addr, payload in resolver.decode_frames_in_order(addresses, "analysis"):
            assert addr.frame not in seen_ordinals, "same ordinal yielded twice"
            seen_ordinals.append(addr.frame)
            np.testing.assert_array_equal(payload.pixels, expected[addr.frame])
            assert payload.frame_ordinal == addr.frame

        assert seen_ordinals == sorted(seen_ordinals), (
            "decode_frames_in_order did not yield in ascending ordinal order"
        )
        assert seen_ordinals == list(range(n_frames))
    finally:
        resolver.close()


def test_decode_frames_in_order_sparse_shuffled_with_duplicate_ordinal():
    """A sparse, shuffled, duplicated request list: every INPUT element
    gets its own yield, in ascending ordinal order (ties -- the
    duplicate -- in input order), and nothing earlier than the smallest
    requested ordinal's own presentation time is ever yielded.
    """
    video_path = VIDEO_PATHS[0]
    resolver = MediaResolver(max_open_decoders=4)
    try:
        frame_times = resolver.get_frame_times(from_path(str(video_path)))
        n_frames = len(frame_times)
        assert n_frames >= 40, "fixture too short for a meaningful sparse test"

        # Sparse (gaps), shuffled (not ascending), duplicated (10 twice).
        requested_ordinals = [30, 10, 25, 10, 35]
        addresses = [
            dataclasses.replace(from_path(str(video_path)), frame=i)
            for i in requested_ordinals
        ]
        expected_by_ordinal = {
            i: resolver.resolve_frame(
                dataclasses.replace(from_path(str(video_path)), frame=i),
                "analysis",
            ).pixels
            for i in set(requested_ordinals)
        }

        pairs = list(resolver.decode_frames_in_order(addresses, "analysis"))
        assert len(pairs) == len(addresses), (
            "expected one yield per requested address, including the duplicate"
        )

        yielded_ordinals = [addr.frame for addr, _payload in pairs]
        assert yielded_ordinals == sorted(requested_ordinals), (
            "not yielded in ascending ordinal order (with duplicate-ordinal "
            "ties in input order)"
        )
        # The two elements requesting ordinal 10 (input positions 1 and 3)
        # must each get their OWN yield -- not be collapsed to one.
        assert yielded_ordinals.count(10) == 2

        min_pts = min(payload.presentation_time_us for _addr, payload in pairs)
        assert min_pts == frame_times[10], (
            "a yielded frame's presentation time is earlier than the "
            "smallest requested ordinal's own time -- something before "
            "the first requested frame was returned"
        )

        for addr, payload in pairs:
            np.testing.assert_array_equal(
                payload.pixels, expected_by_ordinal[addr.frame]
            )
            assert payload.frame_ordinal == addr.frame
    finally:
        resolver.close()


def test_decode_frames_in_order_rejects_mismatched_sources():
    # Would still pass if the method silently decoded from whichever
    # source the FIRST address named, ignoring the rest? No -- this
    # asserts it refuses outright rather than guessing.
    if len(VIDEO_PATHS) < 2:
        pytest.skip("need at least two vids/ fixture videos")
    video_a, video_b = VIDEO_PATHS[0], VIDEO_PATHS[1]
    resolver = MediaResolver(max_open_decoders=4)
    try:
        addresses = [
            dataclasses.replace(from_path(str(video_a)), frame=0),
            dataclasses.replace(from_path(str(video_b)), frame=0),
        ]
        with pytest.raises(MediaAddressError):
            resolver.decode_frames_in_order(addresses, "analysis")
    finally:
        resolver.close()


def test_decode_frames_in_order_rejects_empty_list():
    resolver = MediaResolver(max_open_decoders=4)
    try:
        with pytest.raises(ValueError):
            resolver.decode_frames_in_order([], "analysis")
    finally:
        resolver.close()


# ===========================================================================
# OperatorRegistry._run_create_columns_worker -- the grouped runner.
# ===========================================================================

def test_ordered_path_equivalence_through_the_registry(monkeypatch):
    """Every #f= row of one clip, run through the real grouped/ordered
    OperatorRegistry path, gets exactly the value an independent
    resolve_frame()-per-row computation gives for the same frame.

    Would still pass if the runner mis-routed a row to the wrong group,
    or silently dropped one? No -- every row_id is checked individually
    against its own frame's independently computed value.
    """
    video_path = VIDEO_PATHS[0]
    resolver_expected = MediaResolver(max_open_decoders=4)
    resolver_actual = MediaResolver(max_open_decoders=4)
    try:
        n_frames = len(resolver_expected.get_frame_times(from_path(str(video_path))))
        row_ids = [f"r{i}" for i in range(n_frames)]
        cells = [_frame_cell(video_path, i) for i in range(n_frames)]
        expected = {
            row_ids[i]: _expected_means(resolver_expected, cells[i])
            for i in range(n_frames)
        }

        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells})
        op_registry = OperatorRegistry()
        operator = _PixelStatsOperator()
        op_registry.register(operator)
        run = _build_run(operator, resolver_actual)

        results, row_errors, completions = _run_columns_and_collect(
            op_registry, operator, snapshot, row_ids, run, monkeypatch,
        )

        assert not row_errors, f"unexpected row errors: {row_errors}"
        assert set(results) == set(row_ids)
        for row_id in row_ids:
            for key in ("mean_r", "mean_g", "mean_b"):
                assert results[row_id][key] == pytest.approx(expected[row_id][key]), (
                    row_id, key,
                )
        assert completions and completions[0][2] == n_frames
    finally:
        resolver_expected.close()
        resolver_actual.close()


def test_shuffled_rows_from_two_videos_each_row_gets_its_own_frame(monkeypatch):
    """Rows from two different source videos, interleaved and shuffled in
    the input row order, each land in their own group -- grouping by
    source must never cross-contaminate results between two files.
    """
    if len(VIDEO_PATHS) < 2:
        pytest.skip("need at least two vids/ fixture videos")
    video_a, video_b = VIDEO_PATHS[0], VIDEO_PATHS[1]
    resolver_expected = MediaResolver(max_open_decoders=4)
    resolver_actual = MediaResolver(max_open_decoders=4)
    try:
        ordinals_a = [0, 5, 10, 15, 20]
        ordinals_b = [1, 6, 11, 16]
        entries = (
            [(f"a{i}", video_a, i) for i in ordinals_a]
            + [(f"b{i}", video_b, i) for i in ordinals_b]
        )
        random.Random(1234).shuffle(entries)

        row_ids = [row_id for row_id, _path, _i in entries]
        cells = [_frame_cell(path, i) for _row_id, path, i in entries]
        expected = {
            row_id: _expected_means(resolver_expected, cell)
            for row_id, cell in zip(row_ids, cells)
        }

        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells})
        op_registry = OperatorRegistry()
        operator = _PixelStatsOperator()
        op_registry.register(operator)
        run = _build_run(operator, resolver_actual)

        results, row_errors, completions = _run_columns_and_collect(
            op_registry, operator, snapshot, row_ids, run, monkeypatch,
        )

        assert not row_errors, f"unexpected row errors: {row_errors}"
        assert set(results) == set(row_ids)
        for row_id in row_ids:
            for key in ("mean_r", "mean_g", "mean_b"):
                assert results[row_id][key] == pytest.approx(expected[row_id][key])
    finally:
        resolver_expected.close()
        resolver_actual.close()


def test_duplicate_frame_rows_both_get_results(monkeypatch):
    """Two different rows naming the EXACT SAME frame of the same video
    each get their own, correct result -- decode_frames_in_order's
    "duplicated ordinals yield once per requested address" must survive
    all the way through row delivery.
    """
    video_path = VIDEO_PATHS[0]
    resolver_expected = MediaResolver(max_open_decoders=4)
    resolver_actual = MediaResolver(max_open_decoders=4)
    try:
        dup_cell = _frame_cell(video_path, 7)
        row_ids = ["dup_a", "dup_b", "other"]
        cells = [dup_cell, dup_cell, _frame_cell(video_path, 20)]
        expected = {
            row_id: _expected_means(resolver_expected, cell)
            for row_id, cell in zip(row_ids, cells)
        }

        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells})
        op_registry = OperatorRegistry()
        operator = _PixelStatsOperator()
        op_registry.register(operator)
        run = _build_run(operator, resolver_actual)

        results, row_errors, completions = _run_columns_and_collect(
            op_registry, operator, snapshot, row_ids, run, monkeypatch,
        )

        assert not row_errors, f"unexpected row errors: {row_errors}"
        assert set(results) == set(row_ids)
        for row_id in row_ids:
            for key in ("mean_r", "mean_g", "mean_b"):
                assert results[row_id][key] == pytest.approx(expected[row_id][key])
        assert results["dup_a"] == results["dup_b"]
    finally:
        resolver_expected.close()
        resolver_actual.close()


def test_cancellation_after_n_frames_keeps_exactly_the_delivered_results(monkeypatch):
    """Cancelling mid-group keeps every result already delivered and
    delivers nothing after the row that triggered cancellation -- the
    same between-units guarantee the per-row path already had (checked
    between rows, never mid-row), extended to the ordered path's
    between-frames check.
    """
    video_path = VIDEO_PATHS[0]
    resolver = MediaResolver(max_open_decoders=4)
    try:
        n_frames = min(20, len(resolver.get_frame_times(from_path(str(video_path)))))
        row_ids = [f"r{i}" for i in range(n_frames)]
        cells = [_frame_cell(video_path, i) for i in range(n_frames)]
        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells})

        token = CancellationToken()
        operator = _CancelsAfterRow(token, cancel_after_row_id="r4")
        op_registry = OperatorRegistry()
        op_registry.register(operator)
        run = _build_run(operator, resolver, token=token)

        results, row_errors, completions = _run_columns_and_collect(
            op_registry, operator, snapshot, row_ids, run, monkeypatch,
        )

        assert not row_errors, (
            f"cancellation must not itself be reported as a row error: {row_errors}"
        )
        # r4's own result is kept (cancellation is checked BETWEEN rows,
        # never mid-row -- r4 already finished when it cancelled); nothing
        # from r5 onward was ever attempted.
        assert set(results) == {f"r{i}" for i in range(5)}
        assert operator.rows_seen == [f"r{i}" for i in range(5)]
    finally:
        resolver.close()


def test_damaged_file_in_one_group_reports_row_errors_other_group_completes(
    tmp_path, monkeypatch,
):
    """A decode failure in one source's group (a file that is not
    actually a video) reports every row of THAT group through
    on_row_errors with one shared reason, and the run continues with the
    next group -- one bad source must not kill the rest of the run.
    """
    good_video = VIDEO_PATHS[0]
    fake_video = tmp_path / "fake.mp4"
    fake_video.write_bytes(b"this is not a real video file")

    resolver_expected = MediaResolver(max_open_decoders=4)
    resolver_actual = MediaResolver(max_open_decoders=4)
    try:
        good_row_ids = [f"good{i}" for i in range(5)]
        good_cells = [_frame_cell(good_video, i) for i in range(5)]
        expected = {
            row_id: _expected_means(resolver_expected, cell)
            for row_id, cell in zip(good_row_ids, good_cells)
        }

        bad_row_ids = ["bad0", "bad1"]
        bad_cells = [_frame_cell(fake_video, 0), _frame_cell(fake_video, 1)]

        row_ids = good_row_ids + bad_row_ids
        cells = good_cells + bad_cells
        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells})

        op_registry = OperatorRegistry()
        operator = _PixelStatsOperator()
        op_registry.register(operator)
        run = _build_run(operator, resolver_actual)

        results, row_errors, completions = _run_columns_and_collect(
            op_registry, operator, snapshot, row_ids, run, monkeypatch,
        )

        assert set(results) == set(good_row_ids), (
            "the good group's rows must all complete despite the other "
            "group's decode failure"
        )
        for row_id in good_row_ids:
            for key in ("mean_r", "mean_g", "mean_b"):
                assert results[row_id][key] == pytest.approx(expected[row_id][key])

        error_row_ids = {row_id for row_id, _kind, _msg in row_errors}
        assert error_row_ids == set(bad_row_ids), (
            f"expected exactly the fake-file rows in row_errors, got {error_row_ids}"
        )
        assert completions, "on_complete was never called -- the run must not abort"
    finally:
        resolver_expected.close()
        resolver_actual.close()


def test_mixed_table_frames_and_a_still_still_uses_old_path(monkeypatch):
    """A plain still-image row (no #f= address at all) mixed into a FRAME
    run alongside grouped video-frame rows keeps using the ORIGINAL
    per-row resolve_frame() path -- grouping is only for a #f= address on
    a video.
    """
    video_path = VIDEO_PATHS[0]
    still_paths = sorted(TEST_IMAGES.glob("*.jpg"))
    assert still_paths, "need at least one test_images fixture"
    still_path = still_paths[0]

    resolver_expected = MediaResolver(max_open_decoders=4)
    resolver_actual = MediaResolver(max_open_decoders=4)
    try:
        frame_row_ids = [f"f{i}" for i in range(5)]
        frame_cells = [_frame_cell(video_path, i) for i in range(5)]
        still_cell = format_address(from_path(str(still_path)))

        row_ids = frame_row_ids + ["still"]
        cells = frame_cells + [still_cell]
        expected = {
            row_id: _expected_means(resolver_expected, cell)
            for row_id, cell in zip(row_ids, cells)
        }

        snapshot = pd.DataFrame({"row_id": row_ids, "media": cells})
        op_registry = OperatorRegistry()
        operator = _PixelStatsOperator()
        op_registry.register(operator)
        run = _build_run(operator, resolver_actual)

        results, row_errors, completions = _run_columns_and_collect(
            op_registry, operator, snapshot, row_ids, run, monkeypatch,
        )

        assert not row_errors, f"unexpected row errors: {row_errors}"
        assert set(results) == set(row_ids)
        for row_id in row_ids:
            for key in ("mean_r", "mean_g", "mean_b"):
                assert results[row_id][key] == pytest.approx(expected[row_id][key])
    finally:
        resolver_expected.close()
        resolver_actual.close()


# ===========================================================================
# AppController end to end -- the media-column fix.
# ===========================================================================

def _make_controller(tmp_path):
    store = ArtifactStore(
        tmp_path / "artifacts", resolver=MediaResolver(max_open_decoders=4)
    )
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)
    dataset = Dataset()
    dataset.load_folder(TEST_IMAGES)
    op_registry = OperatorRegistry()
    controller = AppController(
        dataset, QueryEngine(), store, registry, op_registry,
        resolver=MediaResolver(max_open_decoders=4),
    )
    controller.set_filters([])
    return controller, dataset, op_registry


def _run_columns_via_controller_and_join(controller, operator_name, row_ids, monkeypatch):
    created: list[threading.Thread] = []
    real_thread = threading.Thread

    class _Tracked(real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    monkeypatch.setattr(threading, "Thread", _Tracked)
    try:
        controller.run_create_columns(operator_name, row_ids)
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)

    for thread in created:
        thread.join(timeout=30)
        assert not thread.is_alive(), "worker thread did not finish in time"
    for _ in range(6):
        controller._drain_queues()


def _run_table_via_controller_and_join(
    controller, operator_name, row_ids, parameters, monkeypatch,
):
    created: list[threading.Thread] = []
    real_thread = threading.Thread

    class _Tracked(real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    monkeypatch.setattr(threading, "Thread", _Tracked)
    try:
        controller.run_create_table(operator_name, row_ids, parameters)
        live = controller.get_live_runs()
        assert len(live) == 1, "expected exactly one live run to have started"
        operation_id = live[0]["operation_id"]
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)

    for thread in created:
        thread.join(timeout=30)
        assert not thread.is_alive(), "worker thread did not finish in time"

    for _ in range(20):
        controller._drain_queues()
        if operation_id not in controller._live_runs:
            break
    return operation_id


def test_end_to_end_split_into_frames_then_pixel_operator_every_row_gets_a_result(
    tmp_path, monkeypatch,
):
    """The premise this whole item exists to fix: a FRAME operator run on
    the REAL output of "Split into frames" -- a table with NO full_path
    column at all, only frame_address. Before the media-column fix this
    silently refused every row (MissingMedia), because the worker read
    only the literal "full_path" key.

    Would still pass if AppController still hardcoded "full_path"? No --
    frame_rows has no such column, so every row would come back with no
    mean_r/mean_g/mean_b at all.
    """
    from operators.frame_operator import FrameOperator

    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(FrameOperator())
    op_registry.register(_PixelStatsOperator())

    all_row_ids = controller.get_visible_row_ids()
    video_row_ids = all_row_ids[:2]
    second_video = VIDEO_PATHS[1] if len(VIDEO_PATHS) > 1 else VIDEO_PATHS[0]
    dataset.apply_row_updates("frames", {
        video_row_ids[0]: {"full_path": str(VIDEO_PATHS[0])},
        video_row_ids[1]: {"full_path": str(second_video)},
    })

    _run_table_via_controller_and_join(
        controller, "frame", video_row_ids,
        {"media_column": "full_path", "frame_step": 10, "output_table": "frame_rows"},
        monkeypatch,
    )
    assert "frame_rows" in dataset.list_tables(), (
        "Split into frames did not produce frame_rows"
    )

    frame_table = dataset.get_table("frame_rows")
    assert "full_path" not in frame_table.columns, (
        "test premise broken: frame_rows unexpectedly has a full_path column"
    )
    assert "frame_address" in frame_table.columns
    assert len(frame_table) > 0, "Split into frames produced no rows"

    controller.set_active_table("frame_rows")
    controller.set_filters([])
    frame_row_ids = controller.get_visible_row_ids()
    assert frame_row_ids

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    _run_columns_via_controller_and_join(
        controller, "pixel_stats", frame_row_ids, monkeypatch,
    )

    assert not errors, f"unexpected error_occurred messages: {errors}"

    result_table = dataset.get_table("frame_rows").set_index("row_id")
    for row_id in frame_row_ids:
        for key in ("mean_r", "mean_g", "mean_b"):
            value = result_table.loc[row_id, key]
            assert value is not None and not pd.isna(value), (
                f"row {row_id!r} got no {key} result -- the media-column "
                f"fix did not reach the ordered runner"
            )


def test_frame_run_on_a_table_with_a_renamed_media_column_reads_that_column(
    tmp_path, monkeypatch,
):
    """A table whose media lives under a column NOT named full_path still
    gets its media read correctly (per-row path, still images) --
    AppController resolves the column from the table's own display
    column, never a hardcoded literal.
    """
    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_PixelStatsOperator())

    still_paths = sorted(TEST_IMAGES.glob("*.jpg"))[:3]
    assert len(still_paths) >= 2, "need at least two test_images fixtures"

    df = pd.DataFrame({"clip_path": [str(p) for p in still_paths]})
    dataset.create_table_from_df(
        "renamed_media", df, hints={"clip_path": ColumnHint(type_tag="media_path")},
    )

    controller.set_active_table("renamed_media")
    controller.set_filters([])
    row_ids = controller.get_visible_row_ids()
    assert row_ids

    before = dataset.get_table("renamed_media")
    cell_by_row = dict(zip(before["row_id"], before["clip_path"]))

    resolver_expected = MediaResolver(max_open_decoders=4)
    try:
        expected = {
            row_id: _expected_means(resolver_expected, cell)
            for row_id, cell in cell_by_row.items()
        }
    finally:
        resolver_expected.close()

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    _run_columns_via_controller_and_join(
        controller, "pixel_stats", row_ids, monkeypatch,
    )

    assert not errors, f"unexpected error_occurred messages: {errors}"

    after = dataset.get_table("renamed_media").set_index("row_id")
    for row_id in row_ids:
        for key in ("mean_r", "mean_g", "mean_b"):
            assert after.loc[row_id, key] == pytest.approx(expected[row_id][key]), (
                row_id, key,
            )


def test_frame_run_refused_when_table_has_no_visual_column(tmp_path, monkeypatch):
    """A table with no visual column to read media from refuses a
    FRAME-mode run before it starts, naming the table -- there is
    nothing for AppController's media-column decision to resolve to.
    """
    controller, dataset, op_registry = _make_controller(tmp_path)
    op_registry.register(_PixelStatsOperator())

    df = pd.DataFrame({"value": [1, 2, 3]})
    dataset.create_table_from_df("no_media", df)

    controller.set_active_table("no_media")
    controller.set_filters([])
    row_ids = controller.get_visible_row_ids()
    assert row_ids

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    started: list = []
    monkeypatch.setattr(
        op_registry, "run_create_columns",
        lambda *a, **k: (started.append(True), True)[1],
    )

    controller.run_create_columns("pixel_stats", row_ids)

    assert started == [], (
        "the run reached the worker; it must be refused before it starts"
    )
    assert errors, "no error surfaced for the no-visual-column table"
    message = errors[-1]
    assert "no_media" in message, message
    assert controller._live_runs == {}


# ===========================================================================
# One bad frame must not fail the rest of its video.
# ===========================================================================

def test_one_bad_frame_inside_a_group_does_not_fail_the_rest_of_its_video(
    tmp_path, monkeypatch,
):
    """A review finding: for an operator that does NOT override
    iter_column_updates (every operator today), an ordinary exception
    from create_columns() on ONE row of a group must be isolated to that
    row -- exactly as the per-row path already isolates it -- not turn
    every later frame of the same source into a row error too. Checked
    through the real AppController so the run's recorded outcome
    ("partial", never "failed" or silently "complete") is the real
    provenance value, not a registry-level approximation of it.

    Would still pass if the pre-fix group-level `except Exception` still
    caught this? No -- every row after the bad one would be missing a
    result and would instead appear in row_errors, not just the one row
    this test names.
    """
    controller, dataset, op_registry = _make_controller(tmp_path)

    video_path = VIDEO_PATHS[0]
    resolver_expected = MediaResolver(max_open_decoders=4)
    try:
        n_frames = min(
            30, len(resolver_expected.get_frame_times(from_path(str(video_path))))
        )
        assert n_frames >= 30, "fixture too short for a meaningful group"
        cells = [_frame_cell(video_path, i) for i in range(n_frames)]

        df = pd.DataFrame({"clip": cells})
        dataset.create_table_from_df(
            "frame_group", df, hints={"clip": ColumnHint(type_tag="media_path")},
        )
        controller.set_active_table("frame_group")
        controller.set_filters([])
        row_ids = controller.get_visible_row_ids()
        assert len(row_ids) == n_frames

        table = dataset.get_table("frame_group")
        cell_by_row = dict(zip(table["row_id"], table["clip"]))
        row_by_cell = {cell: row_id for row_id, cell in cell_by_row.items()}
        bad_row_id = row_by_cell[cells[5]]

        expected = {
            row_id: _expected_means(resolver_expected, cell)
            for row_id, cell in cell_by_row.items()
            if row_id != bad_row_id
        }
    finally:
        resolver_expected.close()

    operator = _RaisesOnOneFrame(bad_row_id)
    op_registry.register(operator)

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    _run_columns_via_controller_and_join(
        controller, "raises_on_one_frame", row_ids, monkeypatch,
    )

    assert operator.rows_seen == row_ids, (
        "every frame of the source must still be attempted, not just up "
        "to the bad one"
    )

    result_table = dataset.get_table("frame_group").set_index("row_id")
    for row_id in row_ids:
        if row_id == bad_row_id:
            continue
        for key in ("mean_r", "mean_g", "mean_b"):
            value = result_table.loc[row_id, key]
            assert value == pytest.approx(expected[row_id][key]), (row_id, key)

    bad_values = result_table.loc[bad_row_id, ["mean_r", "mean_g", "mean_b"]]
    assert all(pd.isna(v) for v in bad_values), (
        "the one bad row should have an all-None result, not a real one"
    )

    entries = [
        e for e in dataset.provenance.to_list() if e["action"] == "operator_run"
    ]
    assert entries, "no operator_run provenance entry was recorded"
    assert entries[-1]["params"]["outcome"] == "partial", (
        "a run with one row error must not be recorded as a clean complete"
    )
