"""
tests/test_blendshape_operator.py

Standalone test for BlendshapeOperator.

Run as a script:
    python tests/test_blendshape_operator.py

Or under pytest (skips automatically if mediapipe / the model file are
missing, so the suite stays green for reviewers who haven't downloaded
the model yet):
    pytest tests/test_blendshape_operator.py

Verifies the v5 spec for BlendshapeOperator:
  - On a face image, returns the 52 blendshape scores (some non-zero).
  - On an image with no detectable face, returns None for every score.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("mediapipe", reason="mediapipe not installed")

MODEL_PATH = ROOT / "operators" / "models" / "face_landmarker.task"
if not MODEL_PATH.exists():
    pytest.skip(
        f"face_landmarker.task not found at {MODEL_PATH}. "
        "Download it with the curl command in operators/blendshapes.py.",
        allow_module_level=True,
    )

from operators.blendshapes import BlendshapeOperator, BLENDSHAPE_NAMES


def test_face_image_returns_scores():
    op = BlendshapeOperator()
    image_path = ROOT / "test_images" / "001_08.jpg"
    image = op.load_image(image_path)
    assert image is not None, f"could not load {image_path}"

    scores = op.create_columns("test_face", image, {})

    assert set(scores.keys()) == set(BLENDSHAPE_NAMES), \
        "result keys do not match BLENDSHAPE_NAMES"
    # MediaPipe currently does not emit tongueOut (issue #4403), so allow it to
    # be None while requiring every other blendshape to have a real score.
    non_tongue = {k: v for k, v in scores.items() if k != "bs_tongueOut"}
    assert all(v is not None for v in non_tongue.values()), \
        "expected every blendshape (except bs_tongueOut) to be detected on a face image"
    assert scores["bs_tongueOut"] is None, \
        "bs_tongueOut is expected to be None until MediaPipe issue #4403 is resolved"
    assert any(v is not None and v > 0 for v in scores.values()), \
        "expected at least one blendshape > 0 on a real face"

    print(
        f"[face image] {image_path.name}: {len(scores)} scores, "
        f"jawOpen={scores['bs_jawOpen']:.3f}, "
        f"eyeBlinkLeft={scores['bs_eyeBlinkLeft']:.3f}"
    )


def test_no_face_image_returns_all_none():
    op = BlendshapeOperator()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)

    scores = op.create_columns("test_blank", blank, {})

    assert set(scores.keys()) == set(BLENDSHAPE_NAMES), \
        "result keys do not match BLENDSHAPE_NAMES"
    assert all(v is None for v in scores.values()), \
        "expected every blendshape to be None when no face is detected"

    print(f"[no-face image] all {len(scores)} values None as expected")


if __name__ == "__main__":
    test_face_image_returns_scores()
    test_no_face_image_returns_all_none()
    print("\nAll BlendshapeOperator tests passed.")
