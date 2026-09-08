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
  - frame_number increments within each video.
  - video_file holds the source video's filename.
  - Frames from different videos do not collide on disk
    (different videos with the same frame_number must not overwrite each other).
  - frame_step downsamples correctly.
  - The input DataFrame is not mutated.
"""

from __future__ import annotations
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

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


def _run(op, *, video_column, frame_step):
    """A minimal OperatorRun for a direct create_table() call. The
    parameters are built from and validated against the operator's OWN
    descriptor (P1.12d-2a) -- so this also proves video_column / frame_step
    reach the operator through run.parameters, which is the behaviour this
    item introduces. Replaces the old `op._video_column = ...` /
    `op._frame_step = ...` instance-attribute setup."""
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
        paths=None,
        _token=CancellationToken(),
    )


def _write_test_video(path: Path, n_frames: int, color_seed: int) -> None:
    """
    Writes a tiny .mp4 with n_frames solid-colour frames.
    color_seed makes each video distinguishable visually.
    """
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (32, 32))
    try:
        for i in range(n_frames):
            frame = np.full(
                (32, 32, 3),
                fill_value=(color_seed + i) % 256,
                dtype=np.uint8,
            )
            writer.write(frame)
    finally:
        writer.release()


def _make_df(tmp: Path) -> pd.DataFrame:
    v1 = tmp / "participant_01.mp4"
    v2 = tmp / "participant_02.mp4"
    _write_test_video(v1, n_frames=12, color_seed=0)
    _write_test_video(v2, n_frames=8, color_seed=100)
    return pd.DataFrame([
        {"row_id": "r1", "full_path": str(v1), "participant_id": "P01"},
        {"row_id": "r2", "full_path": str(v2), "participant_id": "P02"},
    ])


def test_step_1_keeps_every_frame():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = _make_df(td)
        op = VideoFramesOperator(output_dir=td / "out")

        result = op.create_table(
            df, _run(op, video_column="full_path", frame_step=1)
        )

        assert len(result) == 12 + 8, \
            f"expected 20 frame rows, got {len(result)}"

        v1_rows = result[result["video_file"] == "participant_01.mp4"]
        v2_rows = result[result["video_file"] == "participant_02.mp4"]
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
        op = VideoFramesOperator(output_dir=td / "out")

        result = op.create_table(
            df, _run(op, video_column="full_path", frame_step=4)
        )

        v1_rows = result[result["video_file"] == "participant_01.mp4"]
        v2_rows = result[result["video_file"] == "participant_02.mp4"]
        assert list(v1_rows["frame_number"]) == [0, 4, 8]
        assert list(v2_rows["frame_number"]) == [0, 4]


def test_frames_from_different_videos_do_not_collide():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = _make_df(td)
        op = VideoFramesOperator(output_dir=td / "out")

        result = op.create_table(
            df, _run(op, video_column="full_path", frame_step=1)
        )
        paths = result["full_path"].tolist()
        assert len(paths) == len(set(paths)), \
            "frame paths must be unique across videos"


def test_input_dataframe_not_mutated():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = _make_df(td)
        snapshot = df.copy()
        op = VideoFramesOperator(output_dir=td / "out")
        op.create_table(
            df, _run(op, video_column="full_path", frame_step=1)
        )
        pd.testing.assert_frame_equal(df, snapshot)


def test_missing_column_returns_empty():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = _make_df(td)
        op = VideoFramesOperator(output_dir=td / "out")
        result = op.create_table(
            df, _run(op, video_column="does_not_exist", frame_step=1)
        )
        assert len(result) == 0


def test_non_video_extensions_are_skipped():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df = _make_df(td)
        # Add a .jpg row alongside the two .mp4 rows.
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

        op = VideoFramesOperator(output_dir=td / "out")

        result = op.create_table(
            df, _run(op, video_column="full_path", frame_step=1)
        )

        assert "P03" not in set(result["participant_id"]), \
            "rows with non-video paths must be skipped"
        assert set(result["participant_id"]) == {"P01", "P02"}
        assert len(result) == 12 + 8


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

        op = VideoFramesOperator(output_dir=td / "out")

        try:
            op.create_table(
                df, _run(op, video_column="full_path", frame_step=1)
            )
        except ValueError as e:
            msg = str(e)
            assert "only supports videos" in msg
            assert ".mp4" in msg
        else:
            raise AssertionError(
                "expected ValueError when no video files were found"
            )


if __name__ == "__main__":
    test_step_1_keeps_every_frame()
    test_step_n_downsamples()
    test_frames_from_different_videos_do_not_collide()
    test_input_dataframe_not_mutated()
    test_missing_column_returns_empty()
    test_non_video_extensions_are_skipped()
    test_zero_videos_raises()
    print("\nAll VideoFramesOperator tests passed.")
