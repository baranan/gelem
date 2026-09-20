"""
tests/test_video_frames_operator.py

Standalone test for VideoFramesOperator.

Run as a script:
    python tests/test_video_frames_operator.py

Or under pytest:
    pytest tests/test_video_frames_operator.py

Verifies the v5 spec for VideoFramesOperator:
  - One row per kept frame, across multiple input videos.
  - Source-row metadata (e.g. participant_id) is copied onto every frame row.
  - full_path on each output row points at a JPEG that actually exists.
  - frame_number is the frame's real position in its source file.
  - video_file holds the source video's filename.
  - Frames from different videos do not collide on disk
    (different videos with the same frame_number must not overwrite each other).
  - frame_step downsamples correctly.
  - The input DataFrame is not mutated.

P1.2c-2 routed the operator's decoding through the shared MediaResolver
(media/resolver.py) instead of cv2.VideoCapture, so this file's fixtures
switched from a cv2-written .mp4 to the same lossless, known-frame FFV1
fixture the rest of the P1.2 test suite uses (tests/test_media_resolver.py's
own pattern, copied here rather than imported, per that file's own note) --
its exact, predictable per-frame pixel value is what makes a frame's
real position in the file checkable at all, and it needs no assumption
about what a particular cv2 codec/container combination happens to do
with presentation timestamps.
"""

from __future__ import annotations
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from operators.video_frames import VideoFramesOperator
from operators.descriptor import ExecutionMode
from operators.run_context import (
    CancellationToken,
    OperatorRun,
    OperatorRunSpec,
    RunData,
)
from models.project_paths import build_project_paths
from media.resolver import MediaResolver

FFMPEG_MISSING = shutil.which("ffmpeg") is None
pytestmark = pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg is not on PATH")


def _run_ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-y", *args], check=True, capture_output=True)


def _write_test_video(path: Path, n_frames: int, fps: int = 12) -> None:
    """A lossless video whose every pixel of frame N equals N (copied from
    tests/test_media_resolver.py's _generate_known_frame_video pattern --
    that file is not imported, per the work item's file list). Choosing
    duration_s = n_frames / fps gives exactly n_frames output frames.
    """
    duration_s = n_frames / fps
    _run_ffmpeg([
        "-f", "lavfi", "-i", f"color=c=black:s=32x32:r={fps}:d={duration_s}",
        "-vf", "format=gray,geq=lum='N'",
        "-pix_fmt", "gray",
        "-c:v", "ffv1",
        "-frames:v", str(n_frames),
        str(path),
    ])


def _run(op, *, video_column, frame_step, td, resolver):
    """A minimal OperatorRun for a direct create_table() call. The
    parameters are built from and validated against the operator's OWN
    descriptor (P1.12d-2a) -- so this also proves video_column / frame_step
    reach the operator through run.parameters, which is the behaviour that
    item introduced. Replaces the old `op._video_column = ...` /
    `op._frame_step = ...` instance-attribute setup.

    paths is a real ProjectPaths rooted under td/"project" (P1.9a): the
    operator no longer takes a constructor output_dir, so every call needs
    a working run.paths.outputs_dir to write frames under.

    resolver is a real media.resolver.MediaResolver, supplied by the
    caller so it can be closed after the test -- create_table() now
    decodes through run.resolver (P1.2c-2) rather than opening files
    itself, so a bare stand-in object no longer works here.
    """
    mode_descriptor = op.descriptor.mode_for(ExecutionMode.TABLE)
    spec = OperatorRunSpec(
        operation_id="test-run",
        operator_name=op.name,
        mode=ExecutionMode.TABLE,
        mode_descriptor=mode_descriptor,
        parameters={"video_column": video_column, "frame_step": frame_step},
        target_table="",
    )
    return OperatorRun(
        spec=spec,
        data=RunData(tables={}, projects={}),
        paths=build_project_paths(Path(td) / "project", is_workspace=False),
        resolver=resolver,
        _token=CancellationToken(),
    )


def _make_df(tmp: Path) -> pd.DataFrame:
    v1 = tmp / "participant_01.mkv"
    v2 = tmp / "participant_02.mkv"
    _write_test_video(v1, n_frames=12)
    _write_test_video(v2, n_frames=8)
    return pd.DataFrame([
        {"row_id": "r1", "full_path": str(v1), "participant_id": "P01"},
        {"row_id": "r2", "full_path": str(v2), "participant_id": "P02"},
    ])


def test_step_1_keeps_every_frame():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = _make_df(td)
        op = VideoFramesOperator()
        resolver = MediaResolver(max_open_decoders=4)
        try:
            result = op.create_table(
                df, _run(op, video_column="full_path", frame_step=1, td=td,
                         resolver=resolver)
            )
        finally:
            resolver.close()

        assert len(result) == 12 + 8, \
            f"expected 20 frame rows, got {len(result)}"

        v1_rows = result[result["video_file"] == "participant_01.mkv"]
        v2_rows = result[result["video_file"] == "participant_02.mkv"]
        assert len(v1_rows) == 12
        assert len(v2_rows) == 8

        for _, row in result.iterrows():
            assert Path(row["full_path"]).exists(), \
                f"frame missing on disk: {row['full_path']}"

        assert (v1_rows["participant_id"] == "P01").all()
        assert (v2_rows["participant_id"] == "P02").all()

        assert list(v1_rows["frame_number"]) == list(range(12))
        assert list(v2_rows["frame_number"]) == list(range(8))

        assert "row_id" not in result.columns or result["row_id"].isna().all(), \
            "OperatorRegistry must assign row_ids; create_table should not"


def test_step_n_downsamples():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = _make_df(td)
        op = VideoFramesOperator()
        resolver = MediaResolver(max_open_decoders=4)
        try:
            result = op.create_table(
                df, _run(op, video_column="full_path", frame_step=4, td=td,
                         resolver=resolver)
            )
        finally:
            resolver.close()

        v1_rows = result[result["video_file"] == "participant_01.mkv"]
        v2_rows = result[result["video_file"] == "participant_02.mkv"]
        assert list(v1_rows["frame_number"]) == [0, 4, 8]
        assert list(v2_rows["frame_number"]) == [0, 4]


def test_frames_from_different_videos_do_not_collide():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = _make_df(td)
        op = VideoFramesOperator()
        resolver = MediaResolver(max_open_decoders=4)
        try:
            result = op.create_table(
                df, _run(op, video_column="full_path", frame_step=1, td=td,
                         resolver=resolver)
            )
        finally:
            resolver.close()
        paths = result["full_path"].tolist()
        assert len(paths) == len(set(paths)), \
            "frame paths must be unique across videos"


def test_two_time_point_rows_on_the_same_video_give_distinct_files():
    """Two #t= point rows -- never a #f= or a span -- on the SAME video,
    at two different times. frame_ordinal comes back None for both (a
    bare #t= point resolve never builds the per-file frame-time index on
    its own), so the output filename cannot fall back to a constant like
    0: the second row's JPEG would silently overwrite the first's on
    disk, even though the table still lists two distinct rows. This pins
    the presentation_time_us fallback instead, which differs for the two
    times by construction.
    """
    from PIL import Image
    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        video = td / "clip.mkv"
        _write_test_video(video, n_frames=10, fps=10)  # frame k at t=k/10s

        df = pd.DataFrame([
            {"row_id": "r1", "full_path": f"{video.as_posix()}#t=0.2"},
            {"row_id": "r2", "full_path": f"{video.as_posix()}#t=0.7"},
        ])

        op = VideoFramesOperator()
        resolver = MediaResolver(max_open_decoders=4)
        try:
            result = op.create_table(
                df, _run(op, video_column="full_path", frame_step=1, td=td,
                         resolver=resolver)
            )
        finally:
            resolver.close()

        assert len(result) == 2

        paths = result["full_path"].tolist()
        assert len(set(paths)) == 2, (
            f"the two #t= point rows wrote the same file: {paths}"
        )
        for p in paths:
            assert Path(p).exists(), f"frame missing on disk: {p}"

        # frame_number is honestly None, not a fabricated 0 -- the
        # resolver never built the index for a bare #t= point resolve.
        assert result["frame_number"].isna().all()

        # The known-frame fixture's own "every pixel of frame N equals N"
        # rule: frame 2 and frame 7 must not read back as the same value.
        pixel_values = [
            int(np.asarray(Image.open(p).convert("L")).max()) for p in paths
        ]
        assert pixel_values[0] != pixel_values[1], (
            f"both files hold the same frame's pixels: {pixel_values}"
        )


def _generate_quad_video(path: Path, width=64, height=64) -> None:
    """A one-frame, lossless video with four distinct, asymmetric
    quadrant colours -- copied from tests/test_media_resolver.py's
    _generate_quad_video pattern (that file is not imported, per this
    file's own convention), trimmed to one frame since these tests only
    ever address #f=0. Top-left red, top-right blue, bottom-left cyan,
    bottom-right white: no two quadrants share a colour, so a region
    crop's average colour alone identifies which quadrant it came from.
    """
    r_expr = "if(lt(X,W/2), if(lt(Y,H/2),255,0), if(lt(Y,H/2),0,255))"
    g_expr = "if(lt(X,W/2), if(lt(Y,H/2),0,255), if(lt(Y,H/2),0,255))"
    b_expr = "if(lt(X,W/2), if(lt(Y,H/2),0,255), if(lt(Y,H/2),255,255))"
    _run_ffmpeg([
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=1:d=1",
        "-vf", f"geq=r='{r_expr}':g='{g_expr}':b='{b_expr}'",
        # gbrp (planar RGB) so FFV1 is truly lossless here -- no YUV
        # chroma subsampling to blur the quadrant boundaries or shift
        # the saturated colours this test's tolerances check against.
        "-pix_fmt", "gbrp",
        "-c:v", "ffv1",
        "-frames:v", "1",
        str(path),
    ])


def test_two_rows_selecting_different_regions_of_the_same_frame_give_distinct_files():
    """Review round 5: frame_number / presentation_time_us name a MOMENT
    in the file, never a region or a stream selector. Two rows addressing
    the SAME frame (#f=0) but different #r= crops of it used to write the
    identical filename, so the second row's JPEG would silently overwrite
    the first's and both rows would end up pointing at the second crop.
    This pins the fix: the address's own hash, folded into the filename,
    keeps the two files (and the two rows' full_path values) apart.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        video = td / "quad.mkv"
        _generate_quad_video(video)

        # Top-left quadrant (red) and top-right quadrant (blue) of the
        # SAME frame, #f=0.
        top_left = f"{video.as_posix()}#f=0&r=0,0,0.5,0.5"
        top_right = f"{video.as_posix()}#f=0&r=0.5,0,0.5,0.5"
        df = pd.DataFrame([
            {"row_id": "r1", "full_path": top_left},
            {"row_id": "r2", "full_path": top_right},
        ])

        op = VideoFramesOperator()
        resolver = MediaResolver(max_open_decoders=4)
        try:
            result = op.create_table(
                df, _run(op, video_column="full_path", frame_step=1, td=td,
                         resolver=resolver)
            )
        finally:
            resolver.close()

        assert len(result) == 2

        paths = result["full_path"].tolist()
        assert len(set(paths)) == 2, (
            f"two different regions of the same frame wrote the same "
            f"file: {paths}"
        )
        for p in paths:
            assert Path(p).exists(), f"frame missing on disk: {p}"

        from PIL import Image
        import numpy as np

        # Each output is a small solid-colour crop (JPEG re-encode of a
        # uniform field: a couple of levels of drift, never enough to be
        # mistaken for a different quadrant's colour). Row 1 (top-left)
        # must read back red; row 2 (top-right) must read back blue --
        # not the other way around, and not the same colour twice, which
        # is exactly what "both rows point at their own file" means.
        row1_rgb = np.asarray(Image.open(paths[0]).convert("RGB")).reshape(-1, 3).mean(axis=0)
        row2_rgb = np.asarray(Image.open(paths[1]).convert("RGB")).reshape(-1, 3).mean(axis=0)

        assert row1_rgb[0] > 200 and row1_rgb[2] < 50, (
            f"row 1 (top-left, expected red) read back as {row1_rgb}"
        )
        assert row2_rgb[2] > 200 and row2_rgb[0] < 50, (
            f"row 2 (top-right, expected blue) read back as {row2_rgb}"
        )
        assert np.abs(row1_rgb - row2_rgb).max() > 100, (
            f"the two crops' pixels do not differ: {row1_rgb} vs {row2_rgb}"
        )


def test_input_dataframe_not_mutated():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = _make_df(td)
        snapshot = df.copy()
        op = VideoFramesOperator()
        resolver = MediaResolver(max_open_decoders=4)
        try:
            op.create_table(
                df, _run(op, video_column="full_path", frame_step=1, td=td,
                         resolver=resolver)
            )
        finally:
            resolver.close()
        pd.testing.assert_frame_equal(df, snapshot)


def test_missing_column_returns_empty():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = _make_df(td)
        op = VideoFramesOperator()
        resolver = MediaResolver(max_open_decoders=4)
        try:
            result = op.create_table(
                df, _run(op, video_column="does_not_exist", frame_step=1,
                         td=td, resolver=resolver)
            )
        finally:
            resolver.close()
        assert len(result) == 0


def test_non_video_extensions_are_skipped(capsys):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = _make_df(td)
        # Add a .jpg row alongside the two video rows.
        from PIL import Image
        jpg_path = td / "stray.jpg"
        Image.new("RGB", (32, 32), color=(0, 0, 0)).save(str(jpg_path))
        df = pd.concat(
            [
                df,
                pd.DataFrame([{
                    "row_id": "r3",
                    "full_path": str(jpg_path),
                    "participant_id": "P03",
                }]),
            ],
            ignore_index=True,
        )

        op = VideoFramesOperator()
        resolver = MediaResolver(max_open_decoders=4)
        try:
            capsys.readouterr()  # discard setup output
            result = op.create_table(
                df, _run(op, video_column="full_path", frame_step=1, td=td,
                         resolver=resolver)
            )
        finally:
            resolver.close()
        printed = capsys.readouterr().out

        assert "P03" not in set(result["participant_id"]), \
            "rows with non-video paths must be skipped"
        assert set(result["participant_id"]) == {"P01", "P02"}
        assert len(result) == 12 + 8

        # Pins the fix (P1.2c-2 review round 4): a still-image cell must
        # be caught by the cheap, cell-level extension gate -- "Not a
        # video, skipping" -- and never reach the resolver at all. Before
        # the fix, looks_like_media_extension() accepted an image
        # extension too, so "stray.jpg" passed the gate and only failed
        # deep inside decode_video_span() (MediaAddressError: "looks like
        # a still image"), which the end-of-run summary counted as a
        # failed VIDEO ("0 non-video, 1 skipped") instead of a non-video
        # skip ("1 non-video, 0 skipped") -- exactly the count mismatch
        # the gate's own comment claims does not happen.
        assert "Not a video, skipping: " in printed and "stray.jpg" in printed, (
            f"expected the cheap cell-level gate to reject stray.jpg by "
            f"name; got: {printed!r}"
        )
        assert "(1 non-video, 0 skipped)" in printed, (
            f"expected the end-of-run summary to count stray.jpg as the "
            f"one non-video skip, not a failed video; got: {printed!r}"
        )


def test_zero_videos_raises():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        from PIL import Image
        jpg_path = td / "face.jpg"
        Image.new("RGB", (32, 32), color=(0, 0, 0)).save(str(jpg_path))
        df = pd.DataFrame([{
            "row_id": "r1",
            "full_path": str(jpg_path),
            "participant_id": "P01",
        }])

        op = VideoFramesOperator()
        resolver = MediaResolver(max_open_decoders=4)

        try:
            try:
                op.create_table(
                    df, _run(op, video_column="full_path", frame_step=1,
                             td=td, resolver=resolver)
                )
            except ValueError as e:
                msg = str(e)
                # No longer names a fixed extension list -- there is no
                # single authoritative video-only extension set
                # (docs/media_architecture.md section 3.3's own note on
                # media/resolver.py's local, minimal image/video split).
                assert "0 usable video" in msg
                assert "full_path" in msg
            else:
                raise AssertionError(
                    "expected ValueError when no video files were found"
                )
        finally:
            resolver.close()


def test_frames_are_written_only_under_run_paths_outputs_dir():
    # P1.9a: the operator no longer takes a constructor output_dir -- it
    # must write every frame under run.paths.outputs_dir, and nowhere
    # else (the source videos, which live directly under td, are the one
    # thing this test allows outside outputs_dir).
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = _make_df(td)
        files_before = {p.resolve() for p in td.rglob("*") if p.is_file()}

        op = VideoFramesOperator()
        resolver = MediaResolver(max_open_decoders=4)
        try:
            run = _run(op, video_column="full_path", frame_step=1, td=td,
                       resolver=resolver)
            result = op.create_table(df, run)

            assert len(result) > 0
            outputs_dir = run.paths.outputs_dir.resolve()

            for _, row in result.iterrows():
                written = Path(row["full_path"]).resolve()
                assert written.exists(), f"frame missing on disk: {written}"
                assert outputs_dir in written.parents, (
                    f"{written} was not written under {outputs_dir}"
                )

            new_files = {
                p.resolve() for p in td.rglob("*") if p.is_file()
            } - files_before
            stray = [p for p in new_files if outputs_dir not in p.parents]
            assert not stray, f"files written outside outputs_dir: {stray}"
        finally:
            resolver.close()


if __name__ == "__main__":
    test_step_1_keeps_every_frame()
    test_step_n_downsamples()
    test_frames_from_different_videos_do_not_collide()
    test_frames_are_written_only_under_run_paths_outputs_dir()
    test_input_dataframe_not_mutated()
    test_missing_column_returns_empty()
    test_non_video_extensions_are_skipped()
    test_zero_videos_raises()
    print("\nAll VideoFramesOperator tests passed.")
