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
    create_columns_label = "Extract blendshapes"
    output_columns = [(bs_name, "numeric") for bs_name in BLENDSHAPE_NAMES]
    requires_image = True  # Needs the face image to run mediapipe.

    def __init__(self):
        # Defer loading the mediapipe model until first use, so creating the
        # operator (and starting the app) does not require the model file.
        self._landmarker = None

    def _load_model(self):
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
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            landmarker_config
        )

    def create_columns(
        self,
        row_id: str,
        image: np.ndarray,
        metadata: dict,
    ) -> dict:
        """
        Runs mediapipe face detection on one image and returns blendshape scores.

        Args:
            row_id:   Unique ID of the row being processed.
            image:    The face image as a numpy array (height, width, 3), RGB.
            metadata: Existing column values for this row (not used here).

        Returns:
            Dict mapping each blendshape name to its score (0.0–1.0).
            If no face is detected, all values are None.
        """
        if self._landmarker is None:
            # Lets OperatorSetupError propagate so the worker can abort the
            # whole run and surface the message to the user.
            self._load_model()

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detection_result = self._landmarker.detect(mp_image)

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

        except Exception as e:
            import traceback
            print(
                f"[BlendshapeOperator] UNEXPECTED ERROR processing {row_id}: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()
            return {name: None for name in BLENDSHAPE_NAMES}

# TODO: until the integration will be completed with the ui, we can print the result for a single picture by running this in the terminal:
# (only change to the correct picture name from this folder)

# python -c "
# from operators.blendshapes import BlendshapeOperator
# op = BlendshapeOperator()
# image = op.load_image('test_images/001_08.jpg')
# scores = op.create_columns('test_001', image, {})
# for name, value in scores.items():
#     print(f'{name}: {value}')
# "
