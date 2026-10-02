"""
operators/blendshapes.py

BlendshapeOperator extracts blendshape values from face images using
Google's mediapipe library.

Blendshapes are numerical scores (0.0 to 1.0) representing the
activation of specific facial muscle movements, such as jaw opening,
lip corner raising, or brow lowering. There are approximately 52
blendshape values per face.

Student C is responsible for implementing this operator.

Dependencies:
    pip install mediapipe

Model setup:
    Download the face landmarker model file into operators/models/ before first use.
    Copy and run:

    curl -L -o operators/models/face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

Reference:
    https://developers.google.com/mediapipe/solutions/vision/face_landmarker
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import mediapipe as mp

from operators.base import BaseOperator, OperatorSetupError
from operators.descriptor import (
    BooleanParameter,
    ColumnParameter,
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModelLifecycle,
    ModeDescriptor,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
    SequenceOption,
)

# Where the model file lives, and where to download it from. Kept in one
# place so the missing-model error message and the docstring above stay
# consistent.
_MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_RELATIVE = "operators/models/face_landmarker.task"


# The 52 blendshape names mediapipe returns, in order.
# Used to name the output columns.
BLENDSHAPE_NAMES = [
    "bs_browDownLeft", "bs_browDownRight",
    "bs_browInnerUp", "bs_browOuterUpLeft", "bs_browOuterUpRight",
    "bs_cheekPuff", "bs_cheekSquintLeft", "bs_cheekSquintRight",
    "bs_eyeBlinkLeft", "bs_eyeBlinkRight",
    "bs_eyeLookDownLeft", "bs_eyeLookDownRight",
    "bs_eyeLookInLeft", "bs_eyeLookInRight",
    "bs_eyeLookOutLeft", "bs_eyeLookOutRight",
    "bs_eyeLookUpLeft", "bs_eyeLookUpRight",
    "bs_eyeSquintLeft", "bs_eyeSquintRight",
    "bs_eyeWideLeft", "bs_eyeWideRight",
    "bs_jawForward", "bs_jawLeft", "bs_jawOpen", "bs_jawRight",
    "bs_mouthClose",
    "bs_mouthDimpleLeft", "bs_mouthDimpleRight",
    "bs_mouthFrownLeft", "bs_mouthFrownRight",
    "bs_mouthFunnel",
    "bs_mouthLeft", "bs_mouthLowerDownLeft", "bs_mouthLowerDownRight",
    "bs_mouthPressLeft", "bs_mouthPressRight",
    "bs_mouthPucker", "bs_mouthRight",
    "bs_mouthRollLower", "bs_mouthRollUpper",
    "bs_mouthShrugLower", "bs_mouthShrugUpper",
    "bs_mouthSmileLeft", "bs_mouthSmileRight",
    "bs_mouthStretchLeft", "bs_mouthStretchRight",
    "bs_mouthUpperUpLeft", "bs_mouthUpperUpRight",
    "bs_noseSneerLeft", "bs_noseSneerRight",
    # MediaPipe currently does not emit tongueOut (see MediaPipe issue #4403),
    # so this column will evaluate to None. Kept here for compatibility with
    # the ARKit blendshape set.
    "bs_tongueOut",
]


def _landmarker_options(running_mode=None):
    """
    The FaceLandmarkerOptions shared by build_model() (IMAGE, the default
    running mode MediaPipe uses when none is given) and
    BlendshapeOperator.build_sequence_model() (VIDEO, P2.4b) -- ONE place
    for the missing-model check and every option that must stay identical
    between the two, so they cannot drift apart.

    Raises OperatorSetupError if the model file has not been downloaded
    yet.
    """
    if not _MODEL_PATH.exists():
        raise OperatorSetupError(
            f"The MediaPipe face-landmarker model file is missing.\n"
            f"Download it once with this command:\n"
            f'  curl -L -o {_MODEL_RELATIVE} "{_MODEL_URL}"'
        )
    kwargs = dict(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(_MODEL_PATH)),
        output_face_blendshapes=True,
        num_faces=1,
    )
    if running_mode is not None:
        kwargs["running_mode"] = running_mode
    return mp.tasks.vision.FaceLandmarkerOptions(**kwargs)


def _scores_from_detection_result(detection_result, row_id, run):
    """
    Turns one MediaPipe FaceLandmarkerResult into the blendshape score
    dict create_columns() and create_columns_in_sequence() both return --
    ONE place for the no-face handling, the run.log line, and the
    category-name -> column-name mapping, so the two call sites cannot
    drift apart.
    """
    if not detection_result.face_blendshapes:
        # run-indicator-2: worth more than a percentage on a run where
        # many rows come back empty. LATEST WINS, PER RUN, so this
        # costs one dict write per row, same as any other row.
        run.log(f"no face detected in row {row_id}")
        return {name: None for name in BLENDSHAPE_NAMES}

    detected_scores = detection_result.face_blendshapes[0]
    # The order is NOT the same as BLENDSHAPE_NAMES — its first entry is "_neutral".
    # tongueOut is currently absent from MediaPipe output, so it will be None.
    score_by_name = {bs.category_name: bs.score for bs in detected_scores}
    return {
        bs_name: score_by_name.get(bs_name.removeprefix("bs_"))
        for bs_name in BLENDSHAPE_NAMES
    }


class BlendshapeOperator(BaseOperator):
    """
    Extracts blendshape values from face images using mediapipe.

    For each image, runs mediapipe's FaceLandmarker model and returns
    the 52 blendshape scores as numeric column values.

    If no face is detected in the image, returns None for all blendshape
    columns so the row is marked as missing rather than silently wrong.
    """

    name = "blendshapes"

    # ------------------------------------------------------------------
    # Descriptor (P1.12d-1). Describes what create_columns() does:
    #  - one COLUMNS mode, over the active table;
    #  - two parameters (P2.4b): track_face (opt into MediaPipe's VIDEO
    #    running mode, tracked across frames) and sequence_column (an
    #    optional column that splits one source into more than one
    #    sequence). Both are read by the runner alone, through
    #    operators.descriptor's SequenceOption / runs_as_sequences() /
    #    sequence_group_column() -- this operator itself never reads
    #    run.parameters["track_face"] or ["sequence_column"];
    #  - media_requirement FRAME: mediapipe needs the face image, so the
    #    runner decodes one frame and hands it in as `media`;
    #  - model_lifecycle PER_WORKER (honoured by the runner as of
    #    P1.12d-2b-2) for the default, tracking-off path: the FaceLandmarker
    #    is created in IMAGE running mode (build_model(), create_columns())
    #    and carries no cross-frame tracking state, so PER_SEQUENCE is not
    #    required there; but MediaPipe landmarkers are not documented as
    #    thread-safe, so SHARED cannot be claimed either -- PER_WORKER is
    #    the fallback operators/CLAUDE.md's "Where a model lives" section
    #    prescribes when thread-safety is unknown. With track_face on, the
    #    sequence_option below routes the run to the P2.4a sequence runner
    #    instead: build_sequence_model() builds one VIDEO-mode landmarker
    #    per sequence, and create_columns_in_sequence() calls
    #    detect_for_video() in ascending presentation order. Either way the
    #    runner builds the model and hands it in as run.model; this class
    #    stores nothing on self.
    #  - deterministic: the same image (or the same sequence, in order) and
    #    model give the same scores.
    # ------------------------------------------------------------------
    descriptor = OperatorDescriptor(
        name="blendshapes",
        version="1.1",
        description=(
            "Runs MediaPipe's FaceLandmarker on each row's face image and "
            "writes the 52 ARKit blendshape activation scores (0.0-1.0) as "
            "numeric columns. Rows with no detected face get None for every "
            "blendshape column. Optionally tracks the face across frames "
            "instead of treating each frame independently."
        ),
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.COLUMNS,
                label="Extract blendshapes",
                inputs=(
                    InputSpec(
                        name="active_table",
                        label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                media_requirement=MediaRequirement.FRAME,
                parameters=(
                    BooleanParameter(
                        name="track_face",
                        label="Track the face across frames",
                        required=False,
                        default=False,
                        help_text=(
                            "Tracking follows the face from one frame to "
                            "the next, which smooths the results and gives "
                            "different numbers from per-image mode – use "
                            "one mode for a whole study. With tracking on, "
                            "each still image (and each time-range row) "
                            "builds its own model, which is slow on large "
                            "tables of still images."
                        ),
                    ),
                    ColumnParameter(
                        name="sequence_column",
                        label=(
                            "Split each video into separate sequences by "
                            "this column"
                        ),
                        from_input="active_table",
                        required=False,
                        help_text=(
                            "Optional. When empty, each video file is one "
                            "sequence."
                        ),
                    ),
                ),
                output=OutputSpec(
                    columns=tuple(
                        OutputColumn(name=bs_name, type_tag="numeric")
                        for bs_name in BLENDSHAPE_NAMES
                    ),
                ),
                model_lifecycle=ModelLifecycle.PER_WORKER,
                sequence_option=SequenceOption(
                    enabled_by="track_face",
                    group_by="sequence_column",
                ),
                deterministic=True,
                cacheable=True,
            ),
        ),
    )

    def build_model(self):
        """
        FACTORY for the MediaPipe FaceLandmarker in IMAGE running mode --
        the tracking-off, per-frame-independent default (see
        operators/base.py -> build_model and operators/CLAUDE.md -> "Where
        a model lives").

        The runner calls this once per worker -- the descriptor declares
        model_lifecycle PER_WORKER -- and hands the result to
        create_columns() as run.model. Nothing is stored on self: a
        landmarker on this singleton would be shared by every concurrent
        run, which is exactly what PER_WORKER forbids.

        Raises OperatorSetupError if the model file has not been
        downloaded yet. The runner aborts the run before any row is
        processed and surfaces this message to the researcher.
        """
        landmarker_config = _landmarker_options()
        return mp.tasks.vision.FaceLandmarker.create_from_options(
            landmarker_config
        )

    def close_model(self, model) -> None:
        """
        Releases the FaceLandmarker's native resources. The runner calls
        this on the thread that used `model` last, right after its last
        use (operators/base.py's own docstring lists every call site) --
        never relying on garbage collection, which can call `close()`
        from an unrelated thread at an unpredictable moment and hang.
        """
        model.close()

    def build_sequence_model(self, run):
        """
        FACTORY for one sequence's MediaPipe FaceLandmarker, in VIDEO
        running mode (P2.4b). Reached only when the researcher turns on
        the "Track the face across frames" parameter -- the descriptor's
        sequence_option routes that run through the P2.4a sequence
        runner, which calls this once per sequence rather than
        build_model() once per worker.

        VIDEO mode carries tracking state forward from one
        detect_for_video() call to the next on the SAME landmarker
        instance, which is exactly why the sequence runner gives every
        sequence its own fresh one: two sequences (two sources, or two
        groups of one source) must never share a landmarker's tracking
        state.

        Raises OperatorSetupError if the model file has not been
        downloaded yet -- the same check build_model() makes, through the
        same _landmarker_options() helper, so the message cannot drift
        between the two paths. The runner aborts the whole run through
        on_setup_error, exactly like a build_model() failure does.
        """
        landmarker_config = _landmarker_options(
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(
            landmarker_config
        )

    def create_columns(
        self,
        row_id: str,
        media: np.ndarray,
        metadata: dict,
        run,
    ) -> dict:
        """
        Runs mediapipe face detection on one image and returns blendshape scores.

        Args:
            row_id:   Unique ID of the row being processed.
            media:    The payload the runner decoded for this row, decided
                      by the mode's media_requirement (None for METADATA
                      and ADDRESS). This operator declares FRAME, so it is
                      the face frame as a numpy array (height, width, 3), RGB.
            metadata: Existing column values for this row (not used here).
            run:      The OperatorRun for this run. Read per-run values from
                      run.parameters if needed; this method itself does
                      not -- the runner decides whether tracking is on and
                      routes to create_columns_in_sequence() instead when
                      it is. The landmarker the runner built once for this
                      run (the descriptor declares model_lifecycle
                      PER_WORKER) is run.model; this operator never builds
                      or caches one itself.

        Returns:
            Dict mapping each blendshape name to its score (0.0–1.0).
            If no face is detected, all values are None.
        """
        landmarker = run.model

        # Unexpected exceptions are intentionally NOT caught here — the
        # worker catches them, marks the row as missing, and reports them
        # in an end-of-run summary so they're distinguishable from the
        # normal "no face detected" case below.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=media)
        detection_result = landmarker.detect(mp_image)

        return _scores_from_detection_result(detection_result, row_id, run)

    def create_columns_in_sequence(
        self,
        row_id: str,
        media: np.ndarray,
        metadata: dict,
        run,
        media_time_us: int | None,
    ) -> dict:
        """
        Runs mediapipe face detection, WITH TRACKING, on one row of a
        sequence (P2.4b). Reached only when "Track the face across
        frames" is on; run.model is the VIDEO-mode landmarker
        build_sequence_model() built for THIS sequence.

        media_time_us is this row's presentation time in microseconds, or
        None when its media has no timeline (a still image is its own
        one-row sequence -- see operators/CLAUDE.md's "the sequence
        runner"). MediaPipe's detect_for_video() takes a millisecond
        timestamp and requires it to strictly increase within one
        landmarker's lifetime; a one-row sequence has no earlier
        timestamp to increase from, so None becomes 0.

        Returns / raises: same contract as create_columns(). A row whose
        timestamp does not strictly increase (two rows naming the same
        instant, or a grouping column mixing two sources) is exactly the
        kind of failure the sequence runner's own row-error handling
        covers: whatever MediaPipe raises becomes a row error and resets
        this sequence's model before its next row.
        """
        landmarker = run.model
        timestamp_ms = 0 if media_time_us is None else media_time_us // 1000

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=media)
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        return _scores_from_detection_result(detection_result, row_id, run)

# TODO: for a one-off terminal check, build a landmarker with build_model()
# and call landmarker.detect() on a loaded image directly. create_columns()
# now needs an OperatorRun carrying run.model, so it is no longer the
# simplest entry point.
