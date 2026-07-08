"""
operators/video_frames.py

VideoFramesOperator extracts frames from videos into a frame-level table.

Input table: one row per video, with at least one column holding the
video file path (default: full_path) and any number of metadata columns
(participant_id, condition, session, etc.).

Output table: one row per kept frame. Every column from the source
video row is copied onto each of its frame rows, full_path is
overwritten to point at the saved frame JPEG, and two new columns are
added: frame_number (0-indexed within the video) and video_file
(the source video's filename).

The researcher chooses two parameters via the dialog:
    self._video_column  -- which column holds the video path
    self._frame_step    -- keep every Nth frame (1 keeps everything)

Frame JPEGs are saved under self._output_dir, one fresh subfolder per
create_table() call (named run_YYYY.MM.DD_HH.MM.SS.cs) so re-runs never
overwrite previous extractions. Within a run, filenames are prefixed
with the source video's stem so frames from different videos do not
collide on disk.

Student C is responsible for implementing this operator.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

from operators.base import BaseOperator
from column_types.renderers import VIDEO_EXTENSIONS


class VideoFramesOperator(BaseOperator):
    """
    Extracts frames from each video row into a new frame-level table.
    """

    name = "video_frames"
    create_table_label = "Extract frames from videos"
    output_columns = []
    requires_image = False

    def __init__(self, output_dir: Path | None = None):
        # Default to a project-relative folder, not the system Temp
        # directory. Temp wasn't durable (Disk Cleanup / Storage Sense
        # could wipe saved frame paths) and accumulated leftovers
        # across runs. main.py passes an explicit output_dir.
        self._output_dir = output_dir or (
            Path.cwd() / "gelem_project" / "frames"
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._video_column: str = "full_path"
        self._frame_step: int = 1

    def get_parameters_dialog(self, parent=None, columns=None):
        from PySide6.QtWidgets import (
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QFormLayout,
            QSpinBox,
            QVBoxLayout,
        )

        available = list(columns) if columns else ["full_path"]

        dialog = QDialog(parent)
        dialog.setWindowTitle("Extract frames from videos")

        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        column_combo = QComboBox()
        column_combo.addItems(available)
        if "full_path" in available:
            column_combo.setCurrentText("full_path")
        form.addRow("Video path column:", column_combo)

        step_spin = QSpinBox()
        step_spin.setMinimum(1)
        step_spin.setMaximum(10_000)
        step_spin.setValue(self._frame_step)
        form.addRow("Frame step (keep every Nth):", step_spin)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        def _store():
            self._video_column = column_combo.currentText()
            self._frame_step = step_spin.value()

        dialog.accepted.connect(_store)
        return dialog

    def create_table(
        self,
        df: pd.DataFrame,
        group_by: str | list[str] | None = None,
    ) -> pd.DataFrame:
        import cv2
        from datetime import datetime

        work = df.copy()
        if self._video_column not in work.columns:
            print(
                f"[VideoFramesOperator] Column "
                f"'{self._video_column}' not in input table. "
                f"Available: {list(work.columns)}"
            )
            return pd.DataFrame()

        # Each create_table() call writes into its own subfolder so
        # re-runs never overwrite a previous run's frames. The 2-digit
        # centiseconds suffix keeps the name unique even for back-to-back
        # runs within the same second.
        now = datetime.now()
        stamp = (
            now.strftime("%Y.%m.%d_%H.%M.%S")
            + f".{now.microsecond // 10000:02d}"
        )
        run_dir = self._output_dir / f"run_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        out_rows: list[dict] = []
        videos_seen = 0
        videos_skipped = 0
        non_videos_skipped = 0

        for _, src_row in work.iterrows():
            video_path_raw = src_row[self._video_column]
            if not video_path_raw:
                videos_skipped += 1
                continue
            video_path = Path(str(video_path_raw))
            if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                print(
                    f"[VideoFramesOperator] Not a video, skipping: "
                    f"{video_path.name}"
                )
                non_videos_skipped += 1
                continue
            if not video_path.exists():
                print(
                    f"[VideoFramesOperator] Video not found: {video_path}"
                )
                videos_skipped += 1
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(
                    f"[VideoFramesOperator] Could not open: {video_path}"
                )
                videos_skipped += 1
                cap.release()
                continue

            videos_seen += 1
            video_stem = video_path.stem
            video_filename = video_path.name
            frame_idx = 0
            kept = 0

            try:
                while True:
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    if frame_idx % self._frame_step == 0:
                        frame_rgb = cv2.cvtColor(
                            frame_bgr, cv2.COLOR_BGR2RGB
                        )
                        out_path = (
                            run_dir
                            / f"{video_stem}_frame_{frame_idx:06d}.jpg"
                        )
                        self.save_image(frame_rgb, out_path)

                        row = src_row.to_dict()
                        row["full_path"] = str(out_path)
                        row["frame_number"] = frame_idx
                        row["video_file"] = video_filename
                        row.pop("row_id", None)
                        out_rows.append(row)
                        kept += 1
                    frame_idx += 1
            finally:
                cap.release()

            print(
                f"[VideoFramesOperator] {video_filename}: "
                f"{kept} frames kept (step={self._frame_step})"
            )

        if videos_seen == 0:
            raise ValueError(
                f"VideoFramesOperator only supports videos "
                f"({', '.join(sorted(VIDEO_EXTENSIONS))}). "
                f"Found 0 video files in column '{self._video_column}'."
            )

        print(
            f"[VideoFramesOperator] Done — "
            f"{len(out_rows)} frames from {videos_seen} videos "
            f"({non_videos_skipped} non-video, "
            f"{videos_skipped} missing/empty)"
        )
        return pd.DataFrame(out_rows)
