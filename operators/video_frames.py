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

The researcher chooses two parameters via the dialog; both travel in
run.parameters, never on the operator instance:
    video_column  -- which column holds the video path
    frame_step    -- keep every Nth frame (1 keeps everything)

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
from operators.descriptor import (
    ColumnParameter,
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModelLifecycle,
    ModeDescriptor,
    NumberParameter,
    OperatorDescriptor,
    OutputSpec,
)
from column_types.renderers import VIDEO_EXTENSIONS


class VideoFramesOperator(BaseOperator):
    """
    Extracts frames from each video row into a new frame-level table.
    """

    name = "video_frames"
    create_table_label = "Extract frames from videos"
    output_columns = []
    requires_image = False

    # ------------------------------------------------------------------
    # Descriptor (P1.12d-1). What create_table() ACTUALLY does today:
    #  - one TABLE mode: it builds a brand-new frame-level table
    #    (creates_table), one row per kept frame;
    #  - TWO parameters, both read by create_table() from run.parameters
    #    (P1.12d-2a -- they no longer touch the operator instance):
    #      video_column -- a column of the active table holding the video
    #                      path
    #      frame_step   -- keep every Nth frame; dialog min 1, max 10_000,
    #                      default 1
    #  - media_requirement ADDRESS: create_table() opens the video files
    #    itself --
    #        cap = cv2.VideoCapture(str(video_path))
    #    -- rather than being handed decoded media by the runner. This is
    #    the ADDRESS case the descriptor enum calls out by name.
    #  - model_lifecycle NONE;
    #  - deterministic FALSE / cacheable FALSE: every run writes into a
    #    fresh timestamped subfolder --
    #        now = datetime.now()
    #        stamp = now.strftime("%Y.%m.%d_%H.%M.%S") + ...
    #        run_dir = self._output_dir / f"run_{stamp}"
    #    -- and the output table's overwritten full_path values carry
    #    that timestamp, so identical inputs give different output.
    # ------------------------------------------------------------------
    descriptor = OperatorDescriptor(
        name="video_frames",
        version="1.0",
        description=(
            "Reads a video-path column of the active table, decodes each "
            "video with OpenCV, and builds a new frame-level table with one "
            "row per kept frame (every Nth frame). Each frame row copies the "
            "source video row's columns, repoints full_path at the saved "
            "frame JPEG, and adds frame_number and video_file."
        ),
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.TABLE,
                label="Extract frames from videos",
                inputs=(
                    InputSpec(
                        name="active_table",
                        label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                media_requirement=MediaRequirement.ADDRESS,
                parameters=(
                    ColumnParameter(
                        name="video_column",
                        label="Video path column",
                        from_input="active_table",
                    ),
                    NumberParameter(
                        name="frame_step",
                        label="Frame step (keep every Nth)",
                        minimum=1,
                        maximum=10_000,
                        decimals=0,
                        default=1,
                    ),
                ),
                output=OutputSpec(creates_table=True),
                model_lifecycle=ModelLifecycle.NONE,
                deterministic=False,
                cacheable=False,
            ),
        ),
    )

    def __init__(self, output_dir: Path | None = None):
        # Default to a project-relative folder, not the system Temp
        # directory. Temp wasn't durable (Disk Cleanup / Storage Sense
        # could wipe saved frame paths) and accumulated leftovers
        # across runs. main.py passes an explicit output_dir.
        #
        # output_dir is a genuine construction-time value and stays here.
        # The two run parameters (video_column, frame_step) used to be
        # stored on self by get_parameters_dialog; as of P1.12d-2a they
        # travel in run.parameters and nothing about them lives on the
        # instance.
        self._output_dir = output_dir or (
            Path.cwd() / "gelem_project" / "frames"
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

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
        step_spin.setValue(1)
        form.addRow("Frame step (keep every Nth):", step_spin)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        # The dialog hands its answers back through parameter_values(),
        # keyed by the descriptor's parameter names. It stores nothing on
        # the operator instance -- two concurrent runs must not share one
        # set of values.
        chosen: dict = {}

        def _store():
            chosen["video_column"] = column_combo.currentText()
            chosen["frame_step"] = int(step_spin.value())

        dialog.accepted.connect(_store)
        dialog.parameter_values = lambda: dict(chosen)
        return dialog

    def create_table(
        self,
        df: pd.DataFrame,
        run,
    ) -> pd.DataFrame:
        import cv2
        from datetime import datetime

        # Parameters arrive in run.parameters, validated against this
        # operator's descriptor before the run started -- never off self.
        video_column: str = run.parameters["video_column"]
        frame_step: int = int(run.parameters["frame_step"])

        work = df.copy()
        if video_column not in work.columns:
            print(
                f"[VideoFramesOperator] Column "
                f"'{video_column}' not in input table. "
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
            video_path_raw = src_row[video_column]
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
                    if frame_idx % frame_step == 0:
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
                f"{kept} frames kept (step={frame_step})"
            )

        if videos_seen == 0:
            raise ValueError(
                f"VideoFramesOperator only supports videos "
                f"({', '.join(sorted(VIDEO_EXTENSIONS))}). "
                f"Found 0 video files in column '{video_column}'."
            )

        print(
            f"[VideoFramesOperator] Done — "
            f"{len(out_rows)} frames from {videos_seen} videos "
            f"({non_videos_skipped} non-video, "
            f"{videos_skipped} missing/empty)"
        )
        return pd.DataFrame(out_rows)
