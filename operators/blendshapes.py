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
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModelLifecycle,
    ModeDescriptor,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
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
    #  - no parameters (get_parameters_dialog is not overridden);
    #  - media_requirement FRAME: mediapipe needs the face image, so the
    #    runner decodes one frame and hands it in as `media`;
    #  - model_lifecycle PER_WORKER (honoured by the runner as of
    #    P1.12d-2b-2). The FaceLandmarker is created in IMAGE running mode
    #    (no running_mode is passed to FaceLandmarkerOptions, and
    #    create_columns() calls run.model.detect()), so it carries no
    #    cross-frame tracking state and PER_SEQUENCE is not required; but
    #    MediaPipe landmarkers are not documented as thread-safe, so
    #    SHARED cannot be claimed either -- PER_WORKER is the fallback
    #    operators/CLAUDE.md's "Where a model lives" section prescribes
    #    when thread-safety is unknown. The runner calls build_model()
    #    once per worker and hands the landmarker in as run.model; this
    #    class stores nothing on self.
    #  - deterministic: the same image and model give the same scores.
    # ------------------------------------------------------------------
    descriptor = OperatorDescriptor(
        name="blendshapes",
        version="1.0",
        description=(
            "Runs MediaPipe's FaceLandmarker on each row's face image and "
            "writes the 52 ARKit blendshape activation scores (0.0-1.0) as "
            "numeric columns. Rows with no detected face get None for every "
            "blendshape column."
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
                parameters=(),
                output=OutputSpec(
                    columns=tuple(
                        OutputColumn(name=bs_name, type_tag="numeric")
                        for bs_name in BLENDSHAPE_NAMES
                    ),
                ),
                model_lifecycle=ModelLifecycle.PER_WORKER,
                deterministic=True,
                cacheable=True,
            ),
        ),
    )

    def build_model(self):
        """
        FACTORY for the MediaPipe FaceLandmarker (see operators/base.py ->
        build_model and operators/CLAUDE.md -> "Where a model lives").

        The runner calls this once per worker -- the descriptor declares
        model_lifecycle PER_WORKER -- and hands the result to
        create_columns() as run.model. Nothing is stored on self: a
        landmarker on this singleton would be shared by every concurrent
        run, which is exactly what PER_WORKER forbids.

        Raises OperatorSetupError if the model file has not been
        downloaded yet. The runner aborts the run before any row is
        processed and surfaces this message to the researcher.
        """
        if not _MODEL_PATH.exists():
            raise OperatorSetupError(
                f"The MediaPipe face-landmarker model file is missing.\n"
                f"Download it once with this command:\n"
                f'  curl -L -o {_MODEL_RELATIVE} "{_MODEL_URL}"'
            )
        landmarker_config = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(_MODEL_PATH)),
            output_face_blendshapes=True,
            num_faces=1,
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
            run:      The OperatorRun for this run. This operator declares
                      no parameters, so run.parameters is empty. The
                      landmarker the runner built once for this run (the
                      descriptor declares model_lifecycle PER_WORKER) is
                      run.model; this operator never builds or caches one
                      itself.

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

        if not detection_result.face_blendshapes:
            return {name: None for name in BLENDSHAPE_NAMES}

        detected_scores = detection_result.face_blendshapes[0]
        # The order is NOT the same as BLENDSHAPE_NAMES — its first entry is "_neutral".
        # tongueOut is currently absent from MediaPipe output, so it will be None.
        score_by_name = {bs.category_name: bs.score for bs in detected_scores}
        return {
            bs_name: score_by_name.get(bs_name.removeprefix("bs_"))
            for bs_name in BLENDSHAPE_NAMES
        }

# TODO: for a one-off terminal check, build a landmarker with build_model()
# and call landmarker.detect() on a loaded image directly. create_columns()
# now needs an OperatorRun carrying run.model, so it is no longer the
# simplest entry point.
