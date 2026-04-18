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

Reference:
    https://developers.google.com/mediapipe/solutions/vision/face_landmarker
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from operators.base import BaseOperator


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
        """
        Initialises the mediapipe FaceLandmarker.

        TODO (Student C): Initialise the mediapipe FaceLandmarker here.
        The model should be loaded once when the operator is created,
        not once per image — loading it per image would be very slow.

        See: https://developers.google.com/mediapipe/solutions/vision/face_landmarker/python
        """
        # PLACEHOLDER: model not yet loaded.
        self._landmarker = None

    def create_columns(
        self,
        row_id: str,
        image: np.ndarray,
        metadata: dict,
    ) -> dict:
        """
        Extracts blendshape values from one face image.

        Args:
            row_id:   The row being processed.
            image:    RGB numpy array (height, width, 3), uint8.
            metadata: Existing column values for this row. Not used here.

        Returns:
            Dict mapping blendshape column names to float values (0.0-1.0).
            If no face is detected, all values are None.

        TODO (Student C): Implement this method.

        Suggested approach:
            1. Convert the numpy array to a mediapipe Image object:
               mp_image = mediapipe.Image(
                   image_format=mediapipe.ImageFormat.SRGB, data=image
               )
            2. Run self._landmarker.detect(mp_image).
            3. If no face detected (results.face_blendshapes is empty):
               return {name: None for name in BLENDSHAPE_NAMES}
            4. Extract the score for each blendshape from the result.
            5. Return a dict mapping BLENDSHAPE_NAMES to float scores.

        Example return value:
            {
                'bs_jawOpen': 0.42,
                'bs_mouthSmileLeft': 0.18,
                'bs_mouthSmileRight': 0.21,
                ... (all 52 blendshapes)
            }
        """
        # PLACEHOLDER: returns zeros for all blendshapes.
        # Replace with real mediapipe extraction.
        print(
            f"[BlendshapeOperator] PLACEHOLDER — returning zeros for {row_id}"
        )
        return {bs_name: 0.0 for bs_name in BLENDSHAPE_NAMES}