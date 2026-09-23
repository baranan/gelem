"""
tests/test_detail_frame_still.py

CC-35: in detail mode, an address that selects exactly one frame (#f=N,
or a #t= time POINT) must render as a still of that exact frame, not the
whole video in a QMediaPlayer. The old renderer dispatched on file
extension alone (column_types/renderers.py) and could not see the
fragment, so any video-extension cell opened a player regardless of the
address. A bare video path, or a #t= time RANGE, must keep today's
whole-video player (range playback is P1.10, not this item).

Written from the work-item spec, not from the implementation:
  1. a #f=N detail render returns a still widget (a ZoomableImageView),
     never a player, and its pixels match resolve_frame() for frame N,
     not frame 0;
  2. a bare video path still returns the whole-video player.

Video fixture generation is copied from tests/test_media_resolver.py's
_generate_known_frame_video pattern (that module is not imported, per
this item's file list) -- a lossless video whose every pixel of frame N
equals N, so two frames are guaranteed to differ without depending on
real footage. Skips cleanly if ffmpeg is not on PATH.

Run with:
    python -m pytest tests/test_detail_frame_still.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication
from PySide6.QtMultimediaWidgets import QVideoWidget

from shared_widgets.zoomable_image_view import ZoomableImageView
from media.resolver import MediaResolver
from column_types.renderers import _pil_to_pixmap
from PIL import Image as PILImage

FFMPEG_MISSING = shutil.which("ffmpeg") is None
pytestmark = pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg is not on PATH")

# The controller factory lives in tests/conftest.py (make_controller
# fixture).


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    if QApplication.instance() is None:
        QApplication(sys.argv)


def _run_ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-y", *args], check=True, capture_output=True)


def _generate_known_frame_video(tmp_path, width=64, height=64, fps=25, duration_s=2):
    """A lossless video whose every pixel of frame N equals N. Copied from
    tests/test_media_resolver.py's helper of the same purpose (that file
    is not imported, per this item's file list -- only tests/ may change).
    """
    out_path = tmp_path / "known_frames.mkv"
    _run_ffmpeg([
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration_s}",
        "-vf", "format=gray,geq=lum='N'",
        "-pix_fmt", "gray",
        "-c:v", "ffv1",
        str(out_path),
    ])
    return out_path


def _make_clips_table(make_controller, tmp_path, cell_value):
    controller, dataset, _ = make_controller(tmp_path)
    dataset.create_table_from_df(
        "clips",
        pd.DataFrame({"clip": [cell_value]}),
    )
    assert dataset.schema_for("clips").spec_for("clip").type_tag == "media_path", (
        "sanity: 'clip' must actually be tagged media_path for this test "
        "to exercise the rule it claims to"
    )
    controller.set_active_table("clips")
    row_id = str(dataset.get_table("clips")["row_id"].iloc[0])
    metadata = controller.get_row(row_id)
    return controller, row_id, metadata


# ---------------------------------------------------------------------------
# 1. A #f=N detail render is a still, and it is the RIGHT still.
# ---------------------------------------------------------------------------

def test_frame_address_detail_render_is_a_still_of_the_requested_frame(
    make_controller, tmp_path
):
    video_path = _generate_known_frame_video(tmp_path)
    video_str  = str(video_path).replace("\\", "/")
    n = 22
    address_n = f"{video_str}#f={n}"

    controller, row_id, metadata = _make_clips_table(make_controller, tmp_path, address_n)

    widget = controller.render_column_value(
        "clip", metadata["clip"], size=400, mode="detail",
        context={"row_id": row_id, "column_name": "clip"},
    )

    assert isinstance(widget, ZoomableImageView), (
        f"a #f={n} address must render as a still (ZoomableImageView), "
        f"not a player; got {type(widget)}"
    )

    resolver = MediaResolver(max_open_decoders=2)
    try:
        expected_n = resolver.resolve_frame(address_n, "display", policy="first")
        expected_0 = resolver.resolve_frame(f"{video_str}#f=0", "display", policy="first")
    finally:
        resolver.close()

    assert not np.array_equal(expected_n.pixels, expected_0.pixels), (
        "sanity: frame N and frame 0 must actually differ for this test "
        "to prove anything beyond 'some frame was decoded'"
    )

    actual_pixmap   = widget.current_pixmap()
    expected_pixmap = _pil_to_pixmap(PILImage.fromarray(expected_n.pixels))
    assert actual_pixmap is not None
    assert actual_pixmap.toImage() == expected_pixmap.toImage(), (
        "the still's pixels do not match resolve_frame() for the "
        "requested frame N"
    )


# ---------------------------------------------------------------------------
# 2. A bare video path keeps today's whole-video player.
# ---------------------------------------------------------------------------

def test_bare_video_path_still_returns_the_player(make_controller, tmp_path):
    video_path = _generate_known_frame_video(tmp_path)
    video_str  = str(video_path).replace("\\", "/")

    controller, row_id, metadata = _make_clips_table(make_controller, tmp_path, video_str)

    widget = controller.render_column_value(
        "clip", metadata["clip"], size=400, mode="detail",
        context={"row_id": row_id, "column_name": "clip"},
    )

    assert not isinstance(widget, ZoomableImageView), (
        "a bare video path must not render as a still"
    )
    assert widget.findChild(QVideoWidget) is not None, (
        "a bare video path must still open the whole-video player"
    )


# ---------------------------------------------------------------------------
# 3. CC-36: a resolve failure on a single-frame address shows nothing and
#    tells the researcher why, instead of silently opening the wrong thing.
# ---------------------------------------------------------------------------

def test_resolve_failure_on_single_frame_address_emits_error_occurred(
    make_controller, tmp_path
):
    video_path = _generate_known_frame_video(tmp_path)  # 2s @ 25fps = 50 frames
    video_str  = str(video_path).replace("\\", "/")
    # One past the last real frame -- select_frame() refuses this at
    # resolve time with MediaAddressError (decision 11), an existing,
    # reliable way to make resolve_frame() fail without a corrupt file.
    address_beyond_end = f"{video_str}#f=999999"

    controller, row_id, metadata = _make_clips_table(
        make_controller, tmp_path, address_beyond_end
    )

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    widget = controller.render_column_value(
        "clip", metadata["clip"], size=400, mode="detail",
        context={"row_id": row_id, "column_name": "clip"},
    )

    assert widget is None, (
        "a failed single-frame decode must show nothing, not fall back "
        "to the whole-video player"
    )
    assert errors, (
        "expected an error_occurred signal telling the researcher why "
        "the detail panel is empty"
    )
    assert "999999" in errors[0], (
        f"error message should name the address that failed: {errors[0]!r}"
    )
