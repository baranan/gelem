"""
tests/test_media_resolver.py

Tests for media/resolver.py -- P1.2a, the core media resolver. Written
from docs/media_architecture.md section 3.6 ("Address semantics --
settled") and the P1.2a work item text, not from the implementation.

All video fixtures are generated on demand with ffmpeg (skip cleanly if
it is missing), following the pattern of
tests/test_media_address.py::_generate_known_frame_video -- that file is
not imported or modified; the small pieces of the pattern this module
needs are copied here.

Every picture check compares against either a lossless fixture's own
known pixel values, or a value independently sampled by decoding the
same file directly with PyAV or reading it back with an unrelated tool
(ffmpeg's own CLI auto-rotation) -- never against a re-encoded JPEG.

Run with: python -m pytest tests/test_media_resolver.py
"""

import ast
import dataclasses
import gc
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import threading
import time

# Add project root to Python path, matching the other test modules.
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import av
import numpy as np
import pytest
from PIL import Image, ImageOps

from media.media_address import MediaAddressError, from_path
from media.resolver import MediaResolver, MediaResolverError

FFMPEG_MISSING = shutil.which("ffmpeg") is None
FFPROBE_MISSING = shutil.which("ffprobe") is None
pytestmark = pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg is not on PATH")

# P1.2b: the real recordings this module's GELEM_FIXTURES-gated tests use,
# following the pattern in tests/test_media_address.py -- skip cleanly
# rather than fail when the folder is not set (docs/fixtures.md).
GELEM_FIXTURES = os.environ.get("GELEM_FIXTURES")


# ---------------------------------------------------------------------------
# Fixture generation helpers.
# ---------------------------------------------------------------------------

def _run_ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-y", *args], check=True, capture_output=True)


def _generate_known_frame_video(tmp_path, width=64, height=64, fps=25, duration_s=2):
    """A lossless video whose every pixel of frame N equals N. Copied from
    tests/test_media_address.py's helper of the same purpose (that file is
    not imported, per the work item's file list).
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


def _generate_bframe_video(tmp_path, width=64, height=64, fps=25, duration_s=2):
    """An H.264 file with real B-frames: demuxed packet order (decode
    order) differs from presentation order, which is exactly the case
    P1.2b's frame-time index must sort its way out of rather than trust.

    x264's lossless mode (crf 0) turns out to disable the B-pyramid on
    this content -- verified empirically while building this fixture, not
    assumed -- so this uses a lossy crf with a forced, non-adaptive
    B-frame pattern instead. Content is the same grayscale geq=lum='N'
    gradient as _generate_known_frame_video's, but because it is lossy,
    tests must compare against an independent full decode
    (see _full_decode), never against pixel value N directly.
    """
    out_path = tmp_path / "bframes.mp4"
    _run_ffmpeg([
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration_s}",
        "-vf", "format=gray,geq=lum='N'",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18",
        "-x264-params", "bframes=3:b-adapt=0:scenecut=0",
        str(out_path),
    ])
    return out_path


def _generate_duplicate_pts_video(tmp_path, width=32, height=32, fps=25, duration_s=1):
    """A file where two presented frames genuinely share one presentation
    time. `setpts` maps source frames N and N+1 (for even N) onto the
    same output timestamp; `-fps_mode passthrough` is required to stop
    the muxer silently dropping the resulting duplicate (verified
    empirically -- without it, ffmpeg drops every frame whose pts
    collides with the one before it, `drop=11` in its own summary, and
    the file ends up with no duplicates at all). Content still varies
    per source frame (geq=lum='N'), so what collides is purely the
    timestamp, not the picture.
    """
    out_path = tmp_path / "duplicate_pts.mkv"
    _run_ffmpeg([
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration_s}",
        "-vf", f"format=gray,geq=lum='N',setpts=floor(N/2)/({fps}*TB)",
        "-fps_mode", "passthrough",
        "-pix_fmt", "gray", "-c:v", "ffv1",
        str(out_path),
    ])
    return out_path


def _full_decode(path):
    """Ground truth: an independent full decode pass over stream 0, in
    presentation order (PyAV reorders internally, as demonstrated by
    test_frame_index_matches_a_full_decode_with_b_frames), returning
    (raw pts in stream time_base ticks, rgb24 pixel array) per frame.

    Used only to check the resolver's #f= answers against -- this
    function is not part of resolver.py and duplicates none of its
    caching or pool logic, so it is a genuinely independent check.
    """
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        container.seek(0, backward=True, any_frame=False, stream=stream)
        return [
            (frame.pts, frame.to_ndarray(format="rgb24").copy())
            for frame in container.decode(stream)
        ]
    finally:
        container.close()


def _run_with_timeout(func, timeout=20):
    """Run `func` (no arguments) on a background thread and wait up to
    `timeout` seconds. Fails the test loudly if it is still running past
    the deadline, rather than letting a deadlock regression hang the
    whole suite. Returns a dict with 'value' or 'error' (any exception
    `func` raised, captured so the caller can assert on it from the main
    thread rather than losing it in the background thread).
    """
    outcome = {}

    def _target():
        try:
            outcome["value"] = func()
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            outcome["error"] = exc

    thread = threading.Thread(target=_target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        pytest.fail(
            f"operation did not complete within {timeout}s -- "
            f"this looks like a hang/deadlock, not a slow pass"
        )
    return outcome


def _generate_quad_video(tmp_path, name="quad.mp4", width=64, height=64, fps=5, duration_s=1):
    """A lossless-ish video with four distinct, asymmetric quadrant colours:
    top-left red, top-right blue, bottom-left cyan, bottom-right white.
    Used to test region cropping, RGB channel order, and (via a matrix
    remux, see _apply_display_rotation below) orientation -- all of which
    need real spatial variation, unlike the grayscale known-frame fixture.
    """
    out_path = tmp_path / name
    r_expr = "if(lt(X,W/2), if(lt(Y,H/2),255,0), if(lt(Y,H/2),0,255))"
    g_expr = "if(lt(X,W/2), if(lt(Y,H/2),0,255), if(lt(Y,H/2),0,255))"
    b_expr = "if(lt(X,W/2), if(lt(Y,H/2),0,255), if(lt(Y,H/2),255,255))"
    _run_ffmpeg([
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration_s}",
        "-vf", f"geq=r='{r_expr}':g='{g_expr}':b='{b_expr}'",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "0",
        str(out_path),
    ])
    return out_path


def _apply_display_matrix(tmp_path, source_path, name, rotation_degrees, hflip=False):
    """Remux `source_path` with a display-matrix side data added (pixel
    data byte-identical, -c copy) -- what a real rotated recording's
    container metadata looks like. `-display_rotation` is an INPUT option
    (must precede -i); ffprobe/ffmpeg logs confirm the resulting side data.
    """
    out_path = tmp_path / name
    args = ["-display_rotation:v:0", str(rotation_degrees)]
    if hflip:
        args += ["-display_hflip:v:0"]
    args += ["-i", str(source_path), "-c", "copy", str(out_path)]
    _run_ffmpeg(args)
    return out_path


def _ffmpeg_auto_rotated_frame(path, tmp_path, out_name="auto.png"):
    """The ground truth for decision 6: ffmpeg's own CLI applies a
    container's display matrix automatically when extracting a frame.
    This is an independent rendering path from both PyAV and resolver.py.
    """
    out_path = tmp_path / out_name
    _run_ffmpeg(["-i", str(path), "-update", "1", "-frames:v", "1", str(out_path)])
    return Image.open(out_path).convert("RGB")


def _corners(image_or_array, margin=5):
    """Sample the four corners of a PIL Image or an (H, W, 3) ndarray,
    a fixed margin in from each edge so chroma-subsampling blur at a
    quadrant boundary never contaminates a sample.
    """
    if isinstance(image_or_array, Image.Image):
        w, h = image_or_array.size
        get = image_or_array.getpixel
        return {
            "TL": get((margin, margin)),
            "TR": get((w - 1 - margin, margin)),
            "BL": get((margin, h - 1 - margin)),
            "BR": get((w - 1 - margin, h - 1 - margin)),
        }
    arr = image_or_array
    h, w = arr.shape[:2]
    return {
        "TL": tuple(int(v) for v in arr[margin, margin]),
        "TR": tuple(int(v) for v in arr[margin, w - 1 - margin]),
        "BL": tuple(int(v) for v in arr[h - 1 - margin, margin]),
        "BR": tuple(int(v) for v in arr[h - 1 - margin, w - 1 - margin]),
    }


def _sq_dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _nearest_label(pixel, references):
    return min(references, key=lambda name: _sq_dist(pixel, references[name]))


# ---------------------------------------------------------------------------
# Instrumented av.open() proxies, for the pool-bound and cost-rule tests.
# Both patch media.resolver.av.open so the resolver's own decoder pool is
# exercised exactly as in production; nothing in resolver.py is touched.
# ---------------------------------------------------------------------------

class _FrameCountingContainer:
    """Proxies a real PyAV container, counting every frame yielded by
    decode() into a shared, thread-safe counter.
    """

    def __init__(self, real, counter, lock):
        self._real = real
        self._counter = counter
        self._lock = lock

    def decode(self, *args, **kwargs):
        for frame in self._real.decode(*args, **kwargs):
            with self._lock:
                self._counter[0] += 1
            yield frame

    def __getattr__(self, name):
        return getattr(self._real, name)


class _OpenTrackingContainer:
    """Proxies a real PyAV container, reporting to a shared tracker when
    opened and when closed, so a test can assert the peak number of
    simultaneously open containers.
    """

    def __init__(self, real, tracker):
        self._real = real
        self._tracker = tracker
        self._tracker.opened()

    def close(self):
        self._tracker.closed()
        self._real.close()

    def __getattr__(self, name):
        return getattr(self._real, name)


class _OpenTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._current = 0
        self.peak = 0

    def opened(self):
        with self._lock:
            self._current += 1
            self.peak = max(self.peak, self._current)

    def closed(self):
        with self._lock:
            self._current -= 1


# ---------------------------------------------------------------------------
# Decision 2 -- time point.
# ---------------------------------------------------------------------------

def test_time_point_not_on_a_frame_boundary(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        addr = from_path(str(video_path))
        addr = dataclasses.replace(addr, time_us=50_000)  # 0.05s; frame period is 0.04s
        payload = resolver.resolve_frame(addr, purpose="analysis")
        # Frame 1 covers [0.04, 0.08) -- decision 2: the frame whose
        # interval contains the query time.
        assert set(payload.pixels.flatten().tolist()) == {1}
        assert payload.presentation_time_us == 40_000
        assert payload.width == 64 and payload.height == 64
    finally:
        resolver.close()


def test_time_point_beyond_the_end_of_stream_is_refused(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        addr = dataclasses.replace(from_path(str(video_path)), time_us=100_000_000)
        with pytest.raises(MediaAddressError):
            resolver.resolve_frame(addr, purpose="display")
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# Decisions 3 and 4 -- range membership and representative-frame policy.
# ---------------------------------------------------------------------------

def test_range_policy_first(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        addr = dataclasses.replace(
            from_path(str(video_path)), time_range_us=(100_000, 300_000)
        )
        # Frames at 0.12, 0.16, 0.20, 0.24, 0.28 (indices 3..7) are members.
        payload = resolver.resolve_frame(addr, purpose="display", policy="first")
        assert set(payload.pixels.flatten().tolist()) == {3}
        assert payload.presentation_time_us == 120_000
    finally:
        resolver.close()


def test_range_policy_midpoint(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        addr = dataclasses.replace(
            from_path(str(video_path)), time_range_us=(100_000, 300_000)
        )
        # Midpoint of [0.10, 0.30) is 0.20, which is exactly frame 5's time.
        payload = resolver.resolve_frame(addr, purpose="display", policy="midpoint")
        assert set(payload.pixels.flatten().tolist()) == {5}
        assert payload.presentation_time_us == 200_000
    finally:
        resolver.close()


def test_range_start_between_frames_never_picks_the_earlier_frame(tmp_path):
    """The P0.3 amendment to decision 4: 'first' must choose only from the
    frames the range actually contains, never a frame outside it whose
    own interval happens to overlap the range's start.
    """
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        # Frame 2 covers [0.08, 0.12); frame 3 is at 0.12. A range starting
        # at 0.10 (inside frame 2's own interval) must not select frame 2.
        addr = dataclasses.replace(
            from_path(str(video_path)), time_range_us=(100_000, 140_000)
        )
        payload = resolver.resolve_frame(addr, purpose="display", policy="first")
        assert set(payload.pixels.flatten().tolist()) == {3}
    finally:
        resolver.close()


def test_range_containing_no_frame_is_refused(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        # Frame 3 is at 0.12, frame 4 at 0.16 -- nothing lies in between.
        addr = dataclasses.replace(
            from_path(str(video_path)), time_range_us=(121_000, 139_000)
        )
        with pytest.raises(MediaAddressError):
            resolver.resolve_frame(addr, purpose="display", policy="first")
    finally:
        resolver.close()


def test_bare_path_first_is_frame_zero(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        payload = resolver.resolve_frame(str(video_path), purpose="display")
        assert set(payload.pixels.flatten().tolist()) == {0}
        assert payload.presentation_time_us == 0
    finally:
        resolver.close()


def test_bare_path_midpoint_is_near_the_middle_frame(tmp_path):
    # 50 frames (0..49); the true middle frame is 25 (t=1.0s of a 2.0s file).
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        payload = resolver.resolve_frame(str(video_path), purpose="display", policy="midpoint")
        value = payload.pixels.flatten().tolist()[0]
        assert 20 <= value <= 30
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# Decision 8 -- frame_ordinal is a file-wide presentation-order position.
# P1.2a reported None unconditionally for video. P1.2b's frame-time index
# makes a real value possible, but only once it exists: a #t=, range or
# bare-path resolve never builds the index itself (the cost rule), so it
# reports the real ordinal only when an earlier #f= resolve on the same
# MediaResolver already built it, and None otherwise.
# ---------------------------------------------------------------------------

def test_video_frame_ordinal_is_none_before_any_index_exists(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        point_addr = dataclasses.replace(from_path(str(video_path)), time_us=50_000)
        range_addr = dataclasses.replace(
            from_path(str(video_path)), time_range_us=(100_000, 300_000)
        )
        bare_payload = resolver.resolve_frame(str(video_path), purpose="display")
        point_payload = resolver.resolve_frame(point_addr, purpose="display")
        range_payload = resolver.resolve_frame(range_addr, purpose="display")

        assert bare_payload.frame_ordinal is None
        assert point_payload.frame_ordinal is None
        assert range_payload.frame_ordinal is None
    finally:
        resolver.close()


def test_video_frame_ordinal_is_filled_once_an_index_exists(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        # Build the index via one #f= resolve.
        frame_addr = dataclasses.replace(from_path(str(video_path)), frame=0)
        resolver.resolve_frame(frame_addr, purpose="display")

        # Frame 1 covers [0.04, 0.08) -- same fact test_time_point_not_on_a
        # _frame_boundary uses.
        point_addr = dataclasses.replace(from_path(str(video_path)), time_us=50_000)
        point_payload = resolver.resolve_frame(point_addr, purpose="display")
        assert point_payload.frame_ordinal == 1
        assert point_payload.presentation_time_us == 40_000

        bare_payload = resolver.resolve_frame(str(video_path), purpose="display")
        assert bare_payload.frame_ordinal == 0

        # Frames at indices 3..7 -- same range as test_range_policy_first.
        range_addr = dataclasses.replace(
            from_path(str(video_path)), time_range_us=(100_000, 300_000)
        )
        range_payload = resolver.resolve_frame(range_addr, purpose="display", policy="first")
        assert range_payload.frame_ordinal == 3
    finally:
        resolver.close()


def test_time_point_and_bare_and_range_resolves_never_build_the_index(tmp_path, monkeypatch):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    build_calls = []
    original_build = MediaResolver._build_frame_index

    def _counting_build(self, container, stream, epoch_us):
        build_calls.append(1)
        return original_build(self, container, stream, epoch_us)

    monkeypatch.setattr(MediaResolver, "_build_frame_index", _counting_build)
    try:
        resolver.resolve_frame(str(video_path), purpose="display")  # bare
        point_addr = dataclasses.replace(from_path(str(video_path)), time_us=50_000)
        resolver.resolve_frame(point_addr, purpose="display")
        range_addr = dataclasses.replace(
            from_path(str(video_path)), time_range_us=(100_000, 300_000)
        )
        resolver.resolve_frame(range_addr, purpose="display")
        assert build_calls == []

        frame_addr = dataclasses.replace(from_path(str(video_path)), frame=0)
        resolver.resolve_frame(frame_addr, purpose="display")
        assert len(build_calls) == 1

        # A second #f= resolve reuses the cached index rather than
        # rebuilding it.
        frame_addr_2 = dataclasses.replace(from_path(str(video_path)), frame=1)
        resolver.resolve_frame(frame_addr_2, purpose="display")
        assert len(build_calls) == 1
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# Decision 5 -- region.
# ---------------------------------------------------------------------------

def test_region_crop_selects_the_right_quadrant(tmp_path):
    video_path = _generate_quad_video(tmp_path)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        from media.media_address import Region

        addr = dataclasses.replace(from_path(str(video_path)), region=Region(
            x=0, y=0, w=500_000, h=500_000  # top-left quadrant
        ))
        payload = resolver.resolve_frame(addr, purpose="display")
        assert payload.width == 32 and payload.height == 32
        # Top-left quadrant is red -- see _generate_quad_video.
        r, g, b = payload.pixels[16, 16]
        assert int(r) > 200 and int(g) < 50 and int(b) < 50
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# Decision 6 -- orientation.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degrees,hflip", [(90, False), (-90, False), (180, False), (90, True)])
def test_rotated_fixture_is_upright_against_ffmpegs_own_auto_rotation(tmp_path, degrees, hflip):
    base = _generate_quad_video(tmp_path)
    name = f"rot_{degrees}_{hflip}.mp4"
    rotated = _apply_display_matrix(tmp_path, base, name, degrees, hflip=hflip)

    ground_truth = _ffmpeg_auto_rotated_frame(rotated, tmp_path, f"gt_{degrees}_{hflip}.png")
    gt_w, gt_h = ground_truth.size
    gt_corners = _corners(ground_truth)

    resolver = MediaResolver(max_open_decoders=2)
    try:
        payload = resolver.resolve_frame(str(rotated), purpose="display")
        assert (payload.width, payload.height) == (gt_w, gt_h)
        resolved_corners = _corners(payload.pixels)
        for position in ("TL", "TR", "BL", "BR"):
            distance = _sq_dist(resolved_corners[position], gt_corners[position])
            assert distance < 900, (
                f"{position}: resolver={resolved_corners[position]} "
                f"ffmpeg-auto-rotated={gt_corners[position]}"
            )
    finally:
        resolver.close()


def test_unsupported_display_matrix_raises():
    # A matrix that is not one of the eight quarter-turn/mirror signatures.
    class _FakeSideData:
        def __init__(self, raw):
            self._raw = raw

        def __bytes__(self):
            return self._raw

    class _FakeFrame:
        def __init__(self, matrix_values):
            from av.sidedata.sidedata import Type

            raw = struct.pack("<9i", *matrix_values)
            self.side_data = {Type.DISPLAYMATRIX: _FakeSideData(raw)}

    from media.resolver import _orientation_for_frame

    # A 45-degree-ish rotation: not in the lookup table.
    skewed = (46341, -46341, 0, 46341, 46341, 0, 0, 0, 1 << 30)
    with pytest.raises(MediaResolverError):
        _orientation_for_frame(_FakeFrame(skewed))


# ---------------------------------------------------------------------------
# Decision 7 -- stream selection.
# ---------------------------------------------------------------------------

def test_stream_selection_default_and_explicit(tmp_path):
    out_path = tmp_path / "multi_stream.mp4"
    _run_ffmpeg([
        "-f", "lavfi", "-i", "color=c=red:s=32x32:r=5:d=1",
        "-f", "lavfi", "-i", "color=c=blue:s=32x32:r=5:d=1",
        "-map", "0:v", "-map", "1:v",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "0",
        str(out_path),
    ])
    resolver = MediaResolver(max_open_decoders=2)
    try:
        default_payload = resolver.resolve_frame(str(out_path), purpose="display")
        r, g, b = default_payload.pixels[16, 16]
        assert int(r) > 200 and int(b) < 50  # stream 0 (default) is red

        from media.media_address import StreamSelector

        addr_v1 = dataclasses.replace(
            from_path(str(out_path)), stream=StreamSelector(kind="v", index=1)
        )
        payload_v1 = resolver.resolve_frame(addr_v1, purpose="display")
        r, g, b = payload_v1.pixels[16, 16]
        assert int(b) > 200 and int(r) < 50  # stream 1 is blue
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# Relative address and #f=.
# ---------------------------------------------------------------------------

def test_relative_address_raises():
    resolver = MediaResolver(max_open_decoders=2)
    try:
        addr = from_path("relative/clip.mp4")
        with pytest.raises(MediaAddressError):
            resolver.resolve_frame(addr, purpose="display")
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# P1.2b -- #f= resolution against the known-frame fixture (no B-frames):
# ordinal N must select exactly the frame whose pixels report N, and one
# past the last frame must be refused (decision 11).
# ---------------------------------------------------------------------------

def test_frame_ordinal_selects_the_exact_frame(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)  # 50 frames: 0..49
    resolver = MediaResolver(max_open_decoders=2)
    try:
        for n in (0, 25, 49):
            addr = dataclasses.replace(from_path(str(video_path)), frame=n)
            payload = resolver.resolve_frame(addr, purpose="analysis")
            assert set(payload.pixels.flatten().tolist()) == {n}
            assert payload.frame_ordinal == n
            assert payload.presentation_time_us == n * 40_000  # 1/25s frame period
    finally:
        resolver.close()


def test_frame_ordinal_one_past_the_end_is_refused(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)  # 50 frames: 0..49
    resolver = MediaResolver(max_open_decoders=2)
    try:
        addr = dataclasses.replace(from_path(str(video_path)), frame=50)
        with pytest.raises(MediaAddressError):
            resolver.resolve_frame(addr, purpose="display")
    finally:
        resolver.close()


def test_frame_ordinal_combined_with_region_and_stream_selector(tmp_path):
    out_path = tmp_path / "multi_stream.mp4"
    _run_ffmpeg([
        "-f", "lavfi", "-i", "color=c=red:s=32x32:r=5:d=1",
        "-f", "lavfi", "-i", "color=c=blue:s=32x32:r=5:d=1",
        "-map", "0:v", "-map", "1:v",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "0",
        str(out_path),
    ])
    resolver = MediaResolver(max_open_decoders=2)
    try:
        from media.media_address import Region, StreamSelector

        addr = dataclasses.replace(
            from_path(str(out_path)),
            frame=0,
            stream=StreamSelector(kind="v", index=1),
            region=Region(x=0, y=0, w=500_000, h=500_000),
        )
        payload = resolver.resolve_frame(addr, purpose="display")
        assert payload.frame_ordinal == 0
        assert payload.width == 16 and payload.height == 16
        r, g, b = payload.pixels[8, 8]
        assert int(b) > 200 and int(r) < 50  # stream 1 is blue
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# P1.2b -- the index must sort by presentation time, not trust demux
# (decode) order. An H.264 fixture with real, forced B-frames is the case
# where the two orders genuinely differ (verified while building the
# fixture generator: demuxed packet pts is not ascending on this file, but
# container.decode()'s frames are). Compared against an independent full
# decode, never against pixel value N (this fixture is lossy).
# ---------------------------------------------------------------------------

def test_frame_index_matches_a_full_decode_with_b_frames(tmp_path):
    video_path = _generate_bframe_video(tmp_path, fps=25, duration_s=2)
    ground_truth = _full_decode(video_path)
    n_frames = len(ground_truth)
    assert n_frames > 1

    resolver = MediaResolver(max_open_decoders=2)
    try:
        for n in (0, n_frames // 2, n_frames - 1):
            addr = dataclasses.replace(from_path(str(video_path)), frame=n)
            payload = resolver.resolve_frame(addr, purpose="analysis")
            assert payload.frame_ordinal == n
            np.testing.assert_array_equal(payload.pixels, ground_truth[n][1])

        # Index length equals the full decode's frame count (decision 11):
        # the last valid ordinal succeeds, one past it is refused.
        last_addr = dataclasses.replace(from_path(str(video_path)), frame=n_frames - 1)
        resolver.resolve_frame(last_addr, purpose="display")
        one_past_addr = dataclasses.replace(from_path(str(video_path)), frame=n_frames)
        with pytest.raises(MediaAddressError):
            resolver.resolve_frame(one_past_addr, purpose="display")
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# P1.2b -- decode_video_span.
# ---------------------------------------------------------------------------

def test_decode_video_span_rejects_point_and_frame_addresses(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        point_addr = dataclasses.replace(from_path(str(video_path)), time_us=50_000)
        with pytest.raises(MediaAddressError):
            resolver.decode_video_span(point_addr, purpose="display")

        frame_addr = dataclasses.replace(from_path(str(video_path)), frame=0)
        with pytest.raises(MediaAddressError):
            resolver.decode_video_span(frame_addr, purpose="display")
    finally:
        resolver.close()


def test_decode_video_span_acquires_lazily_not_at_call_time(tmp_path):
    resolver = MediaResolver(max_open_decoders=2)
    try:
        missing_path = tmp_path / "does_not_exist.mp4"
        span = resolver.decode_video_span(str(missing_path), purpose="display")
        # Constructing the iterator did not touch the file -- only
        # iterating it does, which is where the pool acquire (and the
        # inevitable failure to open a missing file) happens.
        with pytest.raises(Exception):
            next(span)
    finally:
        resolver.close()


def test_decode_video_span_empty_range_is_refused(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        # Frame 3 is at 0.12, frame 4 at 0.16 -- same empty gap as
        # test_range_containing_no_frame_is_refused.
        addr = dataclasses.replace(
            from_path(str(video_path)), time_range_us=(121_000, 139_000)
        )
        with pytest.raises(MediaAddressError):
            list(resolver.decode_video_span(addr, purpose="display"))
    finally:
        resolver.close()


def test_span_yields_frames_in_range_with_consecutive_ordinals(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        # Build the index first -- a span never builds it itself (rule 3
        # extended to spans), so without this every ordinal below would
        # be None rather than consecutive.
        frame_addr = dataclasses.replace(from_path(str(video_path)), frame=0)
        resolver.resolve_frame(frame_addr, purpose="display")

        # Frames at indices 3..7 -- same range as test_range_policy_first.
        range_addr = dataclasses.replace(
            from_path(str(video_path)), time_range_us=(100_000, 300_000)
        )
        payloads = list(resolver.decode_video_span(range_addr, purpose="display"))

        assert [set(p.pixels.flatten().tolist()) for p in payloads] == [
            {n} for n in range(3, 8)
        ]
        assert [p.frame_ordinal for p in payloads] == list(range(3, 8))
        assert [p.presentation_time_us for p in payloads] == [
            40_000 * n for n in range(3, 8)
        ]
    finally:
        resolver.close()


def test_span_with_ordinals_true_yields_real_ordinals_with_no_manual_priming(tmp_path):
    """decode_video_span(..., with_ordinals=True) builds the per-file
    frame-time index itself -- unlike test_span_yields_frames_in_range_
    with_consecutive_ordinals above, this test never resolves a #f=
    address first. The fixture's own "every pixel of frame N equals N"
    rule is the ground truth the yielded ordinals are checked against,
    not merely internal consistency.
    """
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        # Frames at indices 3..7 -- same range as test_range_policy_first.
        range_addr = dataclasses.replace(
            from_path(str(video_path)), time_range_us=(100_000, 300_000)
        )
        payloads = list(
            resolver.decode_video_span(
                range_addr, purpose="display", with_ordinals=True
            )
        )

        assert [p.frame_ordinal for p in payloads] == list(range(3, 8))
        assert [set(p.pixels.flatten().tolist()) for p in payloads] == [
            {n} for n in range(3, 8)
        ]
    finally:
        resolver.close()


def test_span_with_ordinals_false_never_builds_the_index(tmp_path, monkeypatch):
    """The default (with_ordinals=False) must cost nothing extra: no demux
    pass over the whole file, for a bare path OR a #t= range. Asserted on
    the same mechanism tests/test_media_resolver.py's own
    test_time_point_and_bare_and_range_resolves_never_build_the_index
    uses for resolve_frame -- counting calls to _build_frame_index,
    the one place a demux pass actually happens.
    """
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    build_calls = []
    original_build = MediaResolver._build_frame_index

    def _counting_build(self, container, stream, epoch_us):
        build_calls.append(1)
        return original_build(self, container, stream, epoch_us)

    monkeypatch.setattr(MediaResolver, "_build_frame_index", _counting_build)
    try:
        bare_payloads = list(
            resolver.decode_video_span(str(video_path), purpose="display")
        )
        assert all(p.frame_ordinal is None for p in bare_payloads)

        range_addr = dataclasses.replace(
            from_path(str(video_path)), time_range_us=(100_000, 300_000)
        )
        range_payloads = list(
            resolver.decode_video_span(range_addr, purpose="display")
        )
        assert all(p.frame_ordinal is None for p in range_payloads)

        assert build_calls == [], (
            "with_ordinals=False (the default) must never build the "
            "per-file frame-time index"
        )
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# CC-17 -- get_frame_times(): the public frame-time-index call. Written
# from the work item text, not from the implementation: enumerating a
# file's frame times without decoding pixels, reusing the same cache and
# demux machinery #f= and with_ordinals=True already share, is the whole
# point, so every test here checks one of those properties directly rather
# than merely pinning the method's return shape.
# ---------------------------------------------------------------------------

def test_get_frame_times_sorted_and_matches_frame_count(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)  # 50 frames
    resolver = MediaResolver(max_open_decoders=2)
    try:
        times = resolver.get_frame_times(str(video_path))
        assert times == sorted(times)
        assert len(times) == 50
        assert times == [n * 40_000 for n in range(50)]  # 1/25s frame period
    finally:
        resolver.close()


def test_get_frame_times_matches_full_decode_with_b_frames(tmp_path):
    """The same case test_frame_index_matches_a_full_decode_with_b_frames
    uses for #f=: demux (decode) order is not presentation order on this
    fixture, so getting the sort right is not a no-op here the way it
    would be on a fixture with no B-frames.
    """
    from media.resolver import _ticks_to_us

    video_path = _generate_bframe_video(tmp_path, fps=25, duration_s=2)
    ground_truth = _full_decode(video_path)
    n_frames = len(ground_truth)
    assert n_frames > 1

    container = av.open(str(video_path))
    time_base = container.streams.video[0].time_base
    container.close()
    sorted_pts = sorted(pts for pts, _pixels in ground_truth)
    epoch_us = _ticks_to_us(sorted_pts[0], time_base)
    expected = [_ticks_to_us(pts, time_base) - epoch_us for pts in sorted_pts]

    resolver = MediaResolver(max_open_decoders=2)
    try:
        times = resolver.get_frame_times(str(video_path))
        assert times == expected
        assert len(times) == n_frames
    finally:
        resolver.close()


def test_get_frame_times_decodes_no_pixels(tmp_path, monkeypatch):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)

    def _fail_if_called(*args, **kwargs):
        pytest.fail("get_frame_times must not decode any pixels")

    monkeypatch.setattr("media.resolver._decoded_frame_to_pixels", _fail_if_called)
    try:
        times = resolver.get_frame_times(str(video_path))
        assert len(times) == 50
    finally:
        resolver.close()


def test_get_frame_times_second_call_reuses_cache_no_second_demux(tmp_path, monkeypatch):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    build_calls = []
    original_build = MediaResolver._build_frame_index

    def _counting_build(self, container, stream, epoch_us):
        build_calls.append(1)
        return original_build(self, container, stream, epoch_us)

    monkeypatch.setattr(MediaResolver, "_build_frame_index", _counting_build)
    try:
        first = resolver.get_frame_times(str(video_path))
        assert len(build_calls) == 1

        second = resolver.get_frame_times(str(video_path))
        assert len(build_calls) == 1, "a second call must reuse the cache, not demux again"
        assert second == first
    finally:
        resolver.close()


def test_get_frame_times_reuses_the_index_an_earlier_f_resolve_built(tmp_path, monkeypatch):
    """Proof there is no second index-building path: an earlier #f=
    resolve_frame() call already paid the demux cost, so get_frame_times()
    must reuse that result rather than building its own.
    """
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        frame_addr = dataclasses.replace(from_path(str(video_path)), frame=0)
        resolver.resolve_frame(frame_addr, purpose="display")  # builds the index

        build_calls = []
        original_build = MediaResolver._build_frame_index

        def _counting_build(self, container, stream, epoch_us):
            build_calls.append(1)
            return original_build(self, container, stream, epoch_us)

        monkeypatch.setattr(MediaResolver, "_build_frame_index", _counting_build)

        times = resolver.get_frame_times(str(video_path))
        assert build_calls == [], (
            "get_frame_times must reuse the index #f= already built, not "
            "demux the file a second time"
        )
        assert len(times) == 50
    finally:
        resolver.close()


def test_get_frame_times_builds_an_index_later_reused_by_an_f_resolve(tmp_path, monkeypatch):
    """The other direction of cache-sharing: an index get_frame_times()
    built must be reused by a later #f= resolve_frame() call, not rebuilt.
    """
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        resolver.get_frame_times(str(video_path))  # builds the index

        build_calls = []
        original_build = MediaResolver._build_frame_index

        def _counting_build(self, container, stream, epoch_us):
            build_calls.append(1)
            return original_build(self, container, stream, epoch_us)

        monkeypatch.setattr(MediaResolver, "_build_frame_index", _counting_build)

        frame_addr = dataclasses.replace(from_path(str(video_path)), frame=10)
        payload = resolver.resolve_frame(frame_addr, purpose="display")
        assert payload.frame_ordinal == 10
        assert build_calls == [], (
            "resolve_frame(#f=) must reuse the index get_frame_times() "
            "already built, not demux the file a second time"
        )
    finally:
        resolver.close()


def test_get_frame_times_returns_a_copy_not_the_cached_list(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        times = resolver.get_frame_times(str(video_path))
        times.append(999_999)  # mutate the caller's copy

        times_again = resolver.get_frame_times(str(video_path))
        assert 999_999 not in times_again
        assert len(times_again) == 50
    finally:
        resolver.close()


def test_get_frame_times_stream_selector_picks_the_right_stream(tmp_path):
    out_path = tmp_path / "multi_stream.mp4"
    _run_ffmpeg([
        "-f", "lavfi", "-i", "color=c=red:s=32x32:r=5:d=1",
        "-f", "lavfi", "-i", "color=c=blue:s=32x32:r=10:d=1",
        "-map", "0:v", "-map", "1:v",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "0",
        str(out_path),
    ])
    resolver = MediaResolver(max_open_decoders=2)
    try:
        default_times = resolver.get_frame_times(str(out_path))
        assert len(default_times) == 5  # stream 0 (default), 5fps, 1s

        from media.media_address import StreamSelector

        addr_v1 = dataclasses.replace(
            from_path(str(out_path)), stream=StreamSelector(kind="v", index=1)
        )
        v1_times = resolver.get_frame_times(addr_v1)
        assert len(v1_times) == 10  # stream 1, 10fps, 1s
    finally:
        resolver.close()


def test_get_frame_times_ignores_region_in_the_address(tmp_path):
    video_path = _generate_quad_video(tmp_path)  # 5fps, 1s -> 5 frames
    resolver = MediaResolver(max_open_decoders=2)
    try:
        from media.media_address import Region

        plain_times = resolver.get_frame_times(str(video_path))

        region_addr = dataclasses.replace(
            from_path(str(video_path)),
            region=Region(x=0, y=0, w=500_000, h=500_000),
        )
        region_times = resolver.get_frame_times(region_addr)
        assert region_times == plain_times
    finally:
        resolver.close()


def test_get_frame_times_accepts_a_frame_or_time_fragment(tmp_path):
    # The fragment is accepted (not rejected) even though it is ignored
    # beyond parsing -- the returned index always describes the whole
    # stream, never a slice named by #f= or #t=.
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        plain_times = resolver.get_frame_times(str(video_path))

        frame_addr = dataclasses.replace(from_path(str(video_path)), frame=10)
        assert resolver.get_frame_times(frame_addr) == plain_times

        point_addr = dataclasses.replace(from_path(str(video_path)), time_us=50_000)
        assert resolver.get_frame_times(point_addr) == plain_times
    finally:
        resolver.close()


def test_get_frame_times_relative_address_raises():
    resolver = MediaResolver(max_open_decoders=2)
    try:
        addr = from_path("relative/clip.mp4")
        with pytest.raises(MediaAddressError):
            resolver.get_frame_times(addr)
    finally:
        resolver.close()


def test_get_frame_times_still_image_raises(tmp_path):
    image = Image.new("RGB", (10, 10), color=(255, 0, 0))
    jpeg_path = tmp_path / "solid.jpg"
    image.save(jpeg_path, format="JPEG", quality=100)

    resolver = MediaResolver(max_open_decoders=2)
    try:
        with pytest.raises(MediaResolverError):
            resolver.get_frame_times(str(jpeg_path))
    finally:
        resolver.close()


@pytest.mark.skipif(GELEM_FIXTURES is None, reason="GELEM_FIXTURES is not set")
def test_span_with_ordinals_matches_resolve_frame_on_a_vfr_fixture():
    """The phone recording (variable frame rate, real per-frame timings,
    not a nominal frame rate) is the real input decision 8's per-file
    index exists for. A range span's with_ordinals=True ordinals must
    name exactly the same frames resolve_frame(#f=N) names, one at a
    time, on the same file.
    """
    from media.resolver import _ticks_to_us

    path = pathlib.Path(GELEM_FIXTURES) / "VID_20260826_100749315.mp4"
    if not path.exists():
        pytest.skip(f"expected fixture not found: {path}")

    ground_truth = _full_decode(path)
    n_frames = len(ground_truth)
    assert n_frames > 1010, "fixture is shorter than the requested range assumes"

    container = av.open(str(path))
    time_base = container.streams.video[0].time_base
    container.close()
    epoch_us = _ticks_to_us(ground_truth[0][0], time_base)

    start_n, end_n = 1000, 1005  # a small, arbitrary mid-file window
    start_us = _ticks_to_us(ground_truth[start_n][0], time_base) - epoch_us
    end_us = _ticks_to_us(ground_truth[end_n][0], time_base) - epoch_us

    resolver = MediaResolver(max_open_decoders=2)
    try:
        range_addr = dataclasses.replace(
            from_path(str(path)), time_range_us=(start_us, end_us)
        )
        span_payloads = list(
            resolver.decode_video_span(
                range_addr, purpose="analysis", with_ordinals=True
            )
        )
        assert [p.frame_ordinal for p in span_payloads] == list(
            range(start_n, end_n)
        )

        for payload in span_payloads:
            frame_addr = dataclasses.replace(
                from_path(str(path)), frame=payload.frame_ordinal
            )
            direct = resolver.resolve_frame(frame_addr, purpose="analysis")
            assert direct.presentation_time_us == payload.presentation_time_us
            np.testing.assert_array_equal(direct.pixels, payload.pixels)
    finally:
        resolver.close()


def test_abandoned_span_frees_its_handle_so_a_later_resolve_succeeds(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    resolver = MediaResolver(max_open_decoders=1)
    try:
        span = resolver.decode_video_span(str(video_path), purpose="display")
        next(span)  # forces the lazy pool acquire

        # Abandon it without calling close() explicitly -- garbage
        # collection alone must release the handle.
        del span
        gc.collect()

        # With only one decoder slot, this hangs forever if the handle
        # was not actually freed.
        outcome = _run_with_timeout(
            lambda: resolver.resolve_frame(str(video_path), purpose="display")
        )
        assert "error" not in outcome, outcome.get("error")
        assert outcome["value"] is not None
    finally:
        resolver.close()


def test_same_thread_reentry_at_the_bound_raises_instead_of_hanging(tmp_path):
    video_a = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    video_b = _generate_quad_video(tmp_path)
    resolver = MediaResolver(max_open_decoders=1)

    def _reentrant_call():
        span = resolver.decode_video_span(str(video_a), purpose="display")
        try:
            next(span)  # holds the pool's only slot, for file A
            # Same thread, still holding it: a different file cannot be
            # waited for -- only this thread could ever release the slot,
            # and it cannot while blocked here.
            with pytest.raises(MediaResolverError):
                resolver.resolve_frame(str(video_b), purpose="display")
        finally:
            span.close()

    try:
        outcome = _run_with_timeout(_reentrant_call)
        assert "error" not in outcome, outcome.get("error")
    finally:
        resolver.close()


def test_decoder_pool_bound_holds_with_concurrent_span_iterators(tmp_path, monkeypatch):
    # Five distinct files -- a span holds its handle for its whole
    # iteration, unlike resolve_frame's brief acquire/release, so this
    # exercises sustained contention against the bound.
    paths = []
    for i in range(5):
        p = _generate_known_frame_video(tmp_path, width=16, height=16, fps=5, duration_s=1)
        renamed = tmp_path / f"span_pool_{i}.mkv"
        p.rename(renamed)
        paths.append(renamed)

    tracker = _OpenTracker()
    import media.resolver as resolver_module

    real_open = resolver_module.av.open

    def _tracking_open(path, *args, **kwargs):
        return _OpenTrackingContainer(real_open(path, *args, **kwargs), tracker)

    monkeypatch.setattr(resolver_module.av, "open", _tracking_open)

    max_open_decoders = 2
    resolver = MediaResolver(max_open_decoders=max_open_decoders)
    try:
        errors = []

        def _worker(path):
            try:
                for _payload in resolver.decode_video_span(str(path), purpose="display"):
                    pass
            except Exception as exc:  # pragma: no cover - surfaced via `errors`
                errors.append(exc)

        threads = [
            threading.Thread(target=_worker, args=(paths[i % len(paths)],))
            for i in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not any(t.is_alive() for t in threads), "a span iterator thread hung"
        assert not errors, errors
        assert tracker.peak <= max_open_decoders
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# Still image, EXIF orientation.
# ---------------------------------------------------------------------------

def test_still_jpeg_exif_orientation(tmp_path):
    # Quadrant colours, non-square, so a 90-degree correction is visible
    # as a dimension swap, not just a relabelling.
    width, height = 40, 24
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            if x < width // 2:
                pixels[x, y] = (255, 0, 0) if y < height // 2 else (0, 255, 255)
            else:
                pixels[x, y] = (0, 0, 255) if y < height // 2 else (255, 255, 255)

    jpeg_path = tmp_path / "photo.jpg"
    exif = image.getexif()
    exif[274] = 6  # Orientation tag: a common real-camera value.
    image.save(jpeg_path, format="JPEG", quality=100, exif=exif)

    # Independent ground truth: re-open fresh and let PIL itself correct
    # the orientation -- this checks resolver.py's integration (does it
    # call exif_transpose, convert to RGB, crop correctly), not PIL's own
    # correctness.
    expected = ImageOps.exif_transpose(Image.open(jpeg_path)).convert("RGB")
    expected_arr = np.array(expected, dtype=np.uint8)

    resolver = MediaResolver(max_open_decoders=2)
    try:
        payload = resolver.resolve_frame(str(jpeg_path), purpose="display")
        assert (payload.width, payload.height) == expected.size
        assert payload.frame_ordinal == 0
        assert payload.presentation_time_us is None
        np.testing.assert_array_equal(payload.pixels, expected_arr)
        # Orientation 6 rotates a 40x24 source to 24x40 -- a real swap,
        # not a no-op -- so this test would fail if exif_transpose were
        # silently skipped.
        assert (payload.width, payload.height) == (height, width)
    finally:
        resolver.close()


def test_still_image_frame_zero_succeeds_frame_one_is_refused(tmp_path):
    # A still image has exactly one frame, at ordinal 0 (the FramePayload
    # docstring's "0 for a still image, which has exactly one frame").
    # #f=0 is that frame; #f=1 is beyond the last frame (decision 11).
    image = Image.new("RGB", (10, 10), color=(255, 0, 0))
    jpeg_path = tmp_path / "solid.jpg"
    image.save(jpeg_path, format="JPEG", quality=100)

    resolver = MediaResolver(max_open_decoders=2)
    try:
        zero_addr = dataclasses.replace(from_path(str(jpeg_path)), frame=0)
        payload = resolver.resolve_frame(zero_addr, purpose="display")
        assert payload.frame_ordinal == 0

        one_addr = dataclasses.replace(from_path(str(jpeg_path)), frame=1)
        with pytest.raises(MediaAddressError):
            resolver.resolve_frame(one_addr, purpose="display")
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# RGB channel order.
# ---------------------------------------------------------------------------

def test_rgb_channel_order_on_a_coloured_fixture(tmp_path):
    video_path = _generate_quad_video(tmp_path)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        payload = resolver.resolve_frame(str(video_path), purpose="display")
        # Top-left quadrant is authored pure red (see _generate_quad_video).
        # If channels were swapped (e.g. BGR), the red channel would not
        # dominate here.
        r, g, b = (int(v) for v in payload.pixels[8, 8])
        assert r > 200
        assert g < 50
        assert b < 50
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# Decoder pool bound.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_open_decoders", [1, 2])
def test_decoder_pool_never_exceeds_its_bound(tmp_path, monkeypatch, max_open_decoders):
    # Five distinct files, so with a bound below 5, contention and
    # eviction both actually happen.
    paths = []
    for i in range(5):
        p = _generate_known_frame_video(tmp_path, width=16, height=16, fps=5, duration_s=1)
        renamed = tmp_path / f"pool_{i}.mkv"
        p.rename(renamed)
        paths.append(renamed)

    tracker = _OpenTracker()
    import media.resolver as resolver_module

    real_open = resolver_module.av.open

    def _tracking_open(path, *args, **kwargs):
        return _OpenTrackingContainer(real_open(path, *args, **kwargs), tracker)

    monkeypatch.setattr(resolver_module.av, "open", _tracking_open)

    resolver = MediaResolver(max_open_decoders=max_open_decoders)
    try:
        errors = []

        def _worker(path):
            try:
                for _ in range(3):
                    resolver.resolve_frame(str(path), purpose="display")
            except Exception as exc:  # pragma: no cover - surfaced via `errors`
                errors.append(exc)

        threads = [
            threading.Thread(target=_worker, args=(paths[i % len(paths)],))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, errors
        assert tracker.peak <= max_open_decoders
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# Cost rule.
# ---------------------------------------------------------------------------

def test_cost_rule_late_time_point_decodes_a_bounded_number_of_frames(tmp_path):
    fps = 25
    total_frames = 300  # 12 seconds
    gop_frames = 25  # 1-second GOPs -- short, per the work item's wording
    out_path = tmp_path / "longgop.mp4"
    _run_ffmpeg([
        "-f", "lavfi", "-i", f"color=c=black:s=32x32:r={fps}:d={total_frames / fps}",
        "-vf", "format=gray,geq=lum='mod(N,256)'",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-x264-params",
        f"keyint={gop_frames}:min-keyint={gop_frames}:scenecut=0:open-gop=0",
        str(out_path),
    ])

    counter = [0]
    lock = threading.Lock()
    import media.resolver as resolver_module

    real_open = resolver_module.av.open

    def _counting_open(path, *args, **kwargs):
        return _FrameCountingContainer(real_open(path, *args, **kwargs), counter, lock)

    resolver = MediaResolver(max_open_decoders=2)
    original_open = resolver_module.av.open
    resolver_module.av.open = _counting_open
    try:
        # A late point: 11.5s into a 12s file. If the resolver decoded
        # from the start, this would cost hundreds of frames; seeking to
        # the nearest keyframe (11.0s) should cost roughly one GOP's worth.
        addr = dataclasses.replace(from_path(str(out_path)), time_us=11_500_000)
        payload = resolver.resolve_frame(addr, purpose="analysis")
        assert payload.presentation_time_us is not None
        assert abs(payload.presentation_time_us - 11_500_000) < 200_000

        assert counter[0] < gop_frames * 2, (
            f"decoded {counter[0]} frames resolving a point 11.5s into a "
            f"12s file with 1s GOPs -- expected roughly one GOP's worth, "
            f"not a decode proportional to the file's full length"
        )
    finally:
        resolver_module.av.open = original_open
        resolver.close()


# ---------------------------------------------------------------------------
# P1.2b follow-up -- code-review fixes.
# ---------------------------------------------------------------------------

def test_decoder_pool_bound_holds_under_concurrent_opens(tmp_path, monkeypatch):
    """Regression test for a race in _DecoderPool.acquire(): the original
    code decremented _pending_opens and inserted the newly opened entry
    in two SEPARATE lock acquisitions, so a concurrent acquire could see
    free capacity for a container that was already open but not yet
    counted anywhere -- transiently exceeding max_open_decoders.

    The race window itself is only a handful of bytecode instructions, so
    reproducing it needs two things working together: (1) a slow fake
    open() so many threads are genuinely mid-open at once, all polling
    acquire()'s retry loop, and (2) a much smaller
    sys.setswitchinterval() so CPython's GIL actually hands off between
    threads often enough to land inside that narrow window -- the
    default ~5ms interval rarely does, which is why an earlier version of
    this test passed against the unfixed code by sheer luck.
    """
    paths = []
    for i in range(5):
        p = _generate_known_frame_video(tmp_path, width=16, height=16, fps=5, duration_s=1)
        renamed = tmp_path / f"race_{i}.mkv"
        p.rename(renamed)
        paths.append(renamed)

    import media.resolver as resolver_module

    real_open = resolver_module.av.open

    original_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(0.00001)
    try:
        for max_open_decoders in (1, 2):
            tracker = _OpenTracker()

            def _slow_tracking_open(path, *args, **kwargs):
                time.sleep(0.005)
                return _OpenTrackingContainer(real_open(path, *args, **kwargs), tracker)

            monkeypatch.setattr(resolver_module.av, "open", _slow_tracking_open)
            resolver = MediaResolver(max_open_decoders=max_open_decoders)
            try:
                errors = []

                def _worker(path):
                    try:
                        for _ in range(6):
                            resolver.resolve_frame(str(path), purpose="display")
                    except Exception as exc:  # pragma: no cover - surfaced via `errors`
                        errors.append(exc)

                threads = [
                    threading.Thread(target=_worker, args=(paths[i % len(paths)],))
                    for i in range(16)
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=30)

                assert not errors, errors
                assert tracker.peak <= max_open_decoders, (
                    f"peak simultaneously open containers {tracker.peak} exceeded "
                    f"max_open_decoders={max_open_decoders}"
                )
            finally:
                resolver.close()
    finally:
        sys.setswitchinterval(original_switch_interval)


def test_epoch_cache_is_invalidated_when_the_file_at_the_same_path_changes(tmp_path):
    """_epoch_cache was keyed on (path, stream index) only, while the
    frame-index cache also keys on file size and mtime. Replacing the
    file at a path made a subsequent #f= resolve compare a fresh index
    (built from the new file) against a stale epoch (cached from the old
    file), raising a spurious mismatch error for a perfectly valid file.
    """
    video_path = tmp_path / "same_path.mkv"
    original = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    original.rename(video_path)
    other_path = _generate_quad_video(tmp_path, name="evict_me.mp4")

    # max_open_decoders=1 so that resolving a second, different file
    # below forces the pool to evict and actually close() video_path's
    # container -- on Windows, PyAV holds an OS-level handle open for as
    # long as a container sits idle in the pool, which would otherwise
    # make the on-disk replace below fail with a sharing-violation
    # PermissionError that has nothing to do with the bug under test.
    resolver = MediaResolver(max_open_decoders=1)
    try:
        # Cache the epoch for the original file via a #t= resolve.
        point_addr = dataclasses.replace(from_path(str(video_path)), time_us=0)
        original_payload = resolver.resolve_frame(point_addr, purpose="display")
        assert original_payload.presentation_time_us == 0

        # Evict and close video_path's container.
        resolver.resolve_frame(str(other_path), purpose="display")

        # Replace the file at the SAME path with a different one whose
        # first frame's raw pts is genuinely offset (a real epoch
        # change), and force a new size/mtime.
        time.sleep(0.05)  # coarse mtime resolution on some filesystems
        replacement = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
        shifted = tmp_path / "shifted.mkv"
        _run_ffmpeg(["-itsoffset", "5", "-i", str(replacement), "-c", "copy", str(shifted)])
        shifted.replace(video_path)

        frame_addr = dataclasses.replace(from_path(str(video_path)), frame=0)
        payload = resolver.resolve_frame(frame_addr, purpose="display")
        assert payload.frame_ordinal == 0
        assert set(payload.pixels.flatten().tolist()) == {0}
    finally:
        resolver.close()


def test_duplicate_presentation_time_refuses_rather_than_guesses(tmp_path):
    video_path = _generate_duplicate_pts_video(tmp_path)

    # Sanity-check the fixture actually has the property this test needs,
    # so a future ffmpeg change that stops producing duplicates fails
    # loudly here rather than this test silently passing for the wrong
    # reason.
    ground_truth = _full_decode(video_path)
    pts_values = [pts for pts, _ in ground_truth]
    if len(pts_values) == len(set(pts_values)):
        pytest.skip(
            "this ffmpeg could not be made to produce a duplicate "
            "presentation time on this fixture -- duplicate-pts handling "
            "stays unverified on this machine"
        )

    resolver = MediaResolver(max_open_decoders=2)
    try:
        frame_addr = dataclasses.replace(from_path(str(video_path)), frame=0)
        with pytest.raises(MediaResolverError):
            resolver.resolve_frame(frame_addr, purpose="display")

        # Consequently, #t= on the same file must not fill in an ordinal
        # either -- the index was never cached (the build raised).
        point_addr = dataclasses.replace(from_path(str(video_path)), time_us=0)
        point_payload = resolver.resolve_frame(point_addr, purpose="display")
        assert point_payload.frame_ordinal is None
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# Review round 5: two DISTINCT raw ticks that round to the SAME microsecond
# are just as ambiguous as an exact raw-tick duplicate (the check just
# above), but no real encoder places two presented frames a fraction of a
# microsecond apart -- video frame rates are milliseconds apart at their
# very fastest. A genuine file cannot be made to demonstrate this the way
# _generate_duplicate_pts_video demonstrates an exact tie, so this test
# fabricates the one thing _build_frame_index actually reads -- a
# container/stream pair narrow enough to demux fake packets and report a
# fine time_base -- and calls the resolver's own, unmodified
# _build_frame_index with it. This is the same "proxy the pieces PyAV
# would supply" technique the pool-bound and cost-rule tests above use
# (_FrameCountingContainer, _OpenTrackingContainer), applied to a
# container two orders of magnitude cheaper to fake than a real decode.
# ---------------------------------------------------------------------------

class _FakePacket:
    def __init__(self, pts, is_discard=False):
        self.pts = pts
        self.is_discard = is_discard


class _FakeStream:
    def __init__(self, time_base, index=0):
        self.time_base = time_base
        self.index = index


class _FakeDemuxContainer:
    """The minimum surface _build_frame_index actually calls: seek() (a
    no-op here -- there is no real file position to move), demux(stream)
    (yields the canned packets, ignoring which stream was asked for -- one
    fake stream is all this test needs), and .name (what
    MediaResolver._container_path reads).
    """

    def __init__(self, name, packets):
        self.name = name
        self._packets = packets

    def seek(self, *args, **kwargs):
        pass

    def demux(self, stream):
        return iter(self._packets)


def test_frame_index_refuses_when_two_ticks_round_to_the_same_microsecond():
    from fractions import Fraction
    from media.resolver import _ticks_to_us

    # time_base of 1/2,000,000 -- two ticks per microsecond -- so raw
    # ticks 100 and 101 (a genuine, non-tied pair: 100 != 101, so the
    # raw-tick duplicate check above does not fire) both round to
    # microsecond 50 (100/2 = 50.0; 101/2 = 50.5, and Python's
    # round-half-to-even rounds 50.5 down to 50). Tick 200 (-> 100us) is
    # an unambiguous third, later frame, so this fixture isolates the
    # rounding collision from every other refusal _build_frame_index can
    # raise.
    time_base = Fraction(1, 2_000_000)
    packets = [
        _FakePacket(pts=100),
        _FakePacket(pts=101),
        _FakePacket(pts=200),
    ]
    stream = _FakeStream(time_base)
    container = _FakeDemuxContainer("fake_collision.mkv", packets)
    epoch_us = _ticks_to_us(100, time_base)  # matches the first entry, 50

    resolver = MediaResolver(max_open_decoders=2)
    try:
        with pytest.raises(MediaResolverError, match="collide at microsecond resolution"):
            resolver._build_frame_index(container, stream, epoch_us)
    finally:
        resolver.close()


def test_close_does_not_close_a_container_in_use_by_a_live_span(tmp_path):
    video_path = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)  # 50 frames
    resolver = MediaResolver(max_open_decoders=2)

    span = resolver.decode_video_span(str(video_path), purpose="display")
    next(span)  # forces the lazy acquire; the container is now busy

    resolver.close()  # must not close the busy container out from under the span

    # Continuing the span to the end must not raise -- the container it
    # is using stays open until the span itself releases it.
    remaining = list(span)
    assert len(remaining) == 49  # frame 0 already consumed above

    # A new resolve after close() is refused, rather than silently
    # opening a fresh container from a closed pool.
    with pytest.raises(MediaResolverError):
        resolver.resolve_frame(str(video_path), purpose="display")


def test_deadlock_guard_waits_when_another_thread_will_release(tmp_path):
    """The original deadlock guard raised whenever the calling thread
    held ANY container at all, even when the pool's other busy container
    was held by a different thread that would release it on its own --
    refusing a perfectly safe wait. It must raise only when the calling
    thread holds every busy container in the pool (so no other thread
    could ever free one).
    """
    video_1 = _generate_known_frame_video(tmp_path, fps=25, duration_s=2)
    video_2 = _generate_quad_video(tmp_path, name="quad_b.mp4")
    video_3 = _generate_quad_video(tmp_path, name="quad_c.mp4")
    resolver = MediaResolver(max_open_decoders=2)

    b_acquired = threading.Event()
    b_released = threading.Event()

    def _hold_and_release_b():
        span_b = resolver.decode_video_span(str(video_2), purpose="display")
        next(span_b)  # holds the pool's second slot, for file2
        b_acquired.set()
        time.sleep(0.2)  # a short, deliberate delay before releasing
        span_b.close()
        b_released.set()

    def _hold_then_resolve_a():
        span_a = resolver.decode_video_span(str(video_1), purpose="display")
        next(span_a)  # holds the pool's first slot, for file1
        try:
            assert b_acquired.wait(timeout=10), "B never acquired its handle"
            # Pool is now genuinely at its bound (2/2): this thread holds
            # one container, thread B holds the other -- not every busy
            # container is held by THIS thread, so this must wait for
            # B's release rather than raise.
            payload = resolver.resolve_frame(str(video_3), purpose="display")
            assert payload is not None
            assert b_released.is_set(), (
                "resolve_frame returned before B released its container "
                "-- it did not actually wait for it"
            )
        finally:
            span_a.close()

    thread_b = threading.Thread(target=_hold_and_release_b)
    try:
        thread_b.start()
        outcome = _run_with_timeout(_hold_then_resolve_a, timeout=15)
        assert "error" not in outcome, outcome.get("error")
    finally:
        thread_b.join(timeout=10)
        resolver.close()


# ---------------------------------------------------------------------------
# GELEM_FIXTURES-gated -- real recordings (docs/fixtures.md). Skip cleanly,
# not fail, when the folder is unset, following test_media_address.py's
# pattern. These are the only tests in this module that decode a real,
# large file, so they are opt-in rather than part of the default run.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(GELEM_FIXTURES is None, reason="GELEM_FIXTURES is not set")
def test_phone_recording_frame_ordinals_match_a_full_decode():
    """The phone recording: variable frame rate, rotation=90 (portrait
    stored as landscape). Real per-file frame timings, not a nominal frame
    rate, and real orientation, in one fixture.
    """
    from media.resolver import _ticks_to_us

    path = pathlib.Path(GELEM_FIXTURES) / "VID_20260826_100749315.mp4"
    if not path.exists():
        pytest.skip(f"expected fixture not found: {path}")

    ground_truth = _full_decode(path)
    n_frames = len(ground_truth)
    assert n_frames > 1000, "fixture is shorter than the requested N values assume"

    container = av.open(str(path))
    time_base = container.streams.video[0].time_base
    container.close()
    epoch_us = _ticks_to_us(ground_truth[0][0], time_base)

    resolver = MediaResolver(max_open_decoders=2)
    try:
        for n in (0, 1, 1000, n_frames - 1):
            addr = dataclasses.replace(from_path(str(path)), frame=n)
            payload = resolver.resolve_frame(addr, purpose="analysis")
            assert payload.frame_ordinal == n
            expected_us = _ticks_to_us(ground_truth[n][0], time_base) - epoch_us
            assert payload.presentation_time_us == expected_us

        # rotation=90: stored landscape, displayed portrait.
        bare_payload = resolver.resolve_frame(str(path), purpose="display")
        assert bare_payload.height > bare_payload.width

        # Index length equals the full decode's frame count (decision 11).
        one_past_addr = dataclasses.replace(from_path(str(path)), frame=n_frames)
        with pytest.raises(MediaAddressError):
            resolver.resolve_frame(one_past_addr, purpose="display")
    finally:
        resolver.close()


@pytest.mark.skipif(FFPROBE_MISSING, reason="ffprobe is not on PATH")
@pytest.mark.skipif(GELEM_FIXTURES is None, reason="GELEM_FIXTURES is not set")
def test_decision12_frame_zero_after_edit_list_matches_time_zero(tmp_path):
    """Regenerates the same non-keyframe-aligned cut as
    test_media_address.py::test_decision12_edit_list_on_video_stream_attempt,
    which produces a genuine video-stream edit list on this machine
    (verified there by reading the container back with ffprobe, not
    assumed). #f=0 must be the same frame as #t=0 (decision 12: frame 0
    and time 0 are the same frame), which only holds if the index excludes
    the edit list's discarded pre-roll packets.
    """
    source = pathlib.Path(GELEM_FIXTURES) / "sid89_video.mp4"
    if not source.exists():
        pytest.skip(f"expected fixture not found: {source}")

    out_path = tmp_path / "elst_attempt.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-y",
        "-ss", "10.3", "-i", str(source), "-t", "3",
        "-c", "copy", "-map", "0:v:0",
        str(out_path),
    ]
    subprocess.run(command, check=True, capture_output=True)

    probe = subprocess.run(
        ["ffprobe", "-hide_banner", "-v", "debug", str(out_path)],
        capture_output=True, text=True,
    )
    if "Processing st: 0, edit list" not in probe.stderr:
        pytest.skip(
            "this attempt did not produce a video-stream edit list; "
            "decision 12 stays unverified -- see docs/media_architecture.md "
            "section 3.6, item 12"
        )

    ground_truth = _full_decode(out_path)
    n_frames = len(ground_truth)

    resolver = MediaResolver(max_open_decoders=2)
    try:
        frame_addr = dataclasses.replace(from_path(str(out_path)), frame=0)
        frame_payload = resolver.resolve_frame(frame_addr, purpose="display")
        assert frame_payload.frame_ordinal == 0
        assert frame_payload.presentation_time_us == 0

        point_addr = dataclasses.replace(from_path(str(out_path)), time_us=0)
        point_payload = resolver.resolve_frame(point_addr, purpose="display")
        np.testing.assert_array_equal(frame_payload.pixels, point_payload.pixels)

        # Index length equals the full decode's frame count (decision 11).
        last_addr = dataclasses.replace(from_path(str(out_path)), frame=n_frames - 1)
        resolver.resolve_frame(last_addr, purpose="display")
        one_past_addr = dataclasses.replace(from_path(str(out_path)), frame=n_frames)
        with pytest.raises(MediaAddressError):
            resolver.resolve_frame(one_past_addr, purpose="display")
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# P1.2c-2 -- routing operators/video_frames.py and the FRAME/ADDRESS
# per-row runner through this same resolver. These three tests reach one
# level above media/resolver.py itself (the operator, and the
# controller/registry layers), because that is what the work item asks
# this file to cover for this sub-item. The controller/dataset/registry
# scaffolding is copied from tests/test_resolver_wiring.py's own pattern
# rather than imported, per that file's own note that the pattern is
# copied, not shared, between the P1.2 test modules.
# ---------------------------------------------------------------------------

TEST_IMAGES = project_root / "test_images"


def _video_frames_run(op, *, video_column, frame_step, tmp_path, resolver):
    from operators.descriptor import ExecutionMode
    from operators.run_context import (
        CancellationToken, OperatorRun, OperatorRunSpec, RunData,
    )
    from models.project_paths import build_project_paths

    mode_descriptor = op.descriptor.mode_for(ExecutionMode.TABLE)
    spec = OperatorRunSpec(
        operation_id="p1.2c2-test-run",
        operator_name=op.name,
        mode=ExecutionMode.TABLE,
        mode_descriptor=mode_descriptor,
        parameters={"video_column": video_column, "frame_step": frame_step},
        target_table="",
    )
    return OperatorRun(
        spec=spec,
        data=RunData(tables={}, projects={}),
        paths=build_project_paths(tmp_path / "project", is_workspace=False),
        resolver=resolver,
        _token=CancellationToken(),
    )


def test_video_frames_bare_path_and_time_range_give_real_frame_numbers(tmp_path):
    """(a) A bare video path walks the whole file, keeping every Nth frame
    (frame_step=2 keeps 0, 2, 4, ...); a #t= range walks only the frames
    that range contains (decision 3), still keeping every Nth of THOSE.
    frame_number in both cases is the frame's real, file-wide position
    (payload.frame_ordinal), not a position local to the range.
    """
    import pandas as pd
    from operators.video_frames import VideoFramesOperator

    # fps=10, duration_s=2 -> 20 frames at t = 0.0, 0.1, ..., 1.9 seconds,
    # each frame's every pixel equal to its (bare-path) ordinal.
    video_path = _generate_known_frame_video(
        tmp_path, width=32, height=32, fps=10, duration_s=2
    )
    op = VideoFramesOperator()
    resolver = MediaResolver(max_open_decoders=4)
    try:
        # -- Bare path: every one of the 20 frames is in range; step 2
        #    keeps ordinals 0, 2, 4, ..., 18. --
        df = pd.DataFrame([{"row_id": "r1", "full_path": str(video_path)}])
        run = _video_frames_run(
            op, video_column="full_path", frame_step=2, tmp_path=tmp_path,
            resolver=resolver,
        )
        result = op.create_table(df, run)
        assert list(result["frame_number"]) == list(range(0, 20, 2))
        for _, row in result.iterrows():
            pixel = np.asarray(Image.open(row["full_path"]).convert("L"))
            # JPEG re-encode of a uniform field: the known-frame fixture's
            # "every pixel of frame N equals N" rule survives to within a
            # couple of levels, never enough to be mistaken for a
            # different frame's value.
            assert abs(int(pixel.max()) - row["frame_number"]) <= 2

        # -- #t= range 0.5-1.0 (half-open) contains ordinals 5..9; step 2
        #    keeps the 1st and 3rd and 5th MEMBERS OF THAT RANGE -- real
        #    ordinals 5, 7, 9, not a range-local 0, 2, 4. --
        df_range = pd.DataFrame(
            [{"row_id": "r1", "full_path": f"{video_path.as_posix()}#t=0.5-1.0"}]
        )
        run_range = _video_frames_run(
            op, video_column="full_path", frame_step=2, tmp_path=tmp_path,
            resolver=resolver,
        )
        result_range = op.create_table(df_range, run_range)
        assert list(result_range["frame_number"]) == [5, 7, 9]
        for _, row in result_range.iterrows():
            pixel = np.asarray(Image.open(row["full_path"]).convert("L"))
            assert abs(int(pixel.max()) - row["frame_number"]) <= 2
    finally:
        resolver.close()


def _run_columns_and_wait(controller, operator_name, row_ids, monkeypatch):
    """Start a create_columns run, join every worker thread it spawned,
    then pump the controller's drain by hand (no Qt event loop here) --
    copied from tests/test_resolver_wiring.py's helper of the same name.
    """
    created: list[threading.Thread] = []
    real_thread = threading.Thread

    class _Tracked(real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    monkeypatch.setattr(threading, "Thread", _Tracked)
    try:
        controller.run_create_columns(operator_name, row_ids)
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)

    for thread in created:
        thread.join(timeout=10)
        assert not thread.is_alive(), "worker thread did not finish in time"

    for _ in range(4):
        controller._drain_queues()


def _make_frame_recording_operator():
    """A minimal FRAME-requirement COLUMNS operator, recording the media
    it is handed per row_id (absent for a row_id the runner refused).
    """
    from operators.base import BaseOperator
    from operators.descriptor import (
        ExecutionMode, InputKind, InputSpec, MediaRequirement,
        ModeDescriptor, OperatorDescriptor, OutputColumn, OutputSpec,
    )

    class _FrameRecordingOperator(BaseOperator):
        def __init__(self):
            super().__init__()
            self.name = "p1_2c2_frame_recorder"
            self.descriptor = OperatorDescriptor(
                name=self.name,
                version="1.0",
                description="Test double recording the media it is handed.",
                modes=(
                    ModeDescriptor(
                        mode=ExecutionMode.COLUMNS,
                        label="Recording",
                        inputs=(
                            InputSpec(
                                name="active_table", label="Active table",
                                kind=InputKind.ACTIVE_TABLE,
                            ),
                        ),
                        media_requirement=MediaRequirement.FRAME,
                        parameters=(),
                        output=OutputSpec(
                            columns=(OutputColumn(name="out", type_tag="numeric"),)
                        ),
                    ),
                ),
            )
            self.media_by_row: dict = {}

        def create_columns(self, row_id, media, metadata, run):
            self.media_by_row[row_id] = media
            return {"out": 1.0}

    return _FrameRecordingOperator()


def _make_frame_controller(tmp_path, resolver):
    from models.dataset import Dataset
    from models.query_engine import QueryEngine
    from artifacts.artifact_store import ArtifactStore
    from column_types.registry import ColumnTypeRegistry
    from operators.operator_registry import OperatorRegistry
    from controller import AppController

    store = ArtifactStore(tmp_path / "artifacts", resolver=resolver)
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)
    dataset = Dataset()
    dataset.load_folder(TEST_IMAGES)
    op_registry = OperatorRegistry()
    controller = AppController(
        dataset, QueryEngine(), store, registry, op_registry, resolver=resolver
    )
    controller.set_filters([])
    return controller, dataset, op_registry


def test_frame_run_refuses_a_whole_video_row_but_still_processes_an_image_row(
    tmp_path, monkeypatch
):
    """(b) A FRAME operator declares that it reads one addressed frame. A
    row whose media cell is a BARE video path names no frame at all, so
    the per-row runner must refuse it (rather than silently decoding that
    video's first frame) while an ordinary image row in the same run is
    unaffected.
    """
    resolver = MediaResolver(max_open_decoders=4)
    try:
        controller, dataset, op_registry = _make_frame_controller(tmp_path, resolver)
        row_ids = controller.get_visible_row_ids()
        image_row_id = row_ids[0]
        video_row_id = row_ids[1]

        video_path = _generate_known_frame_video(tmp_path, fps=10, duration_s=1)
        dataset.apply_row_updates(
            "frames", {video_row_id: {"full_path": str(video_path)}}
        )

        op = _make_frame_recording_operator()
        op_registry.register(op)
        _run_columns_and_wait(
            controller, op.name, [image_row_id, video_row_id], monkeypatch
        )

        assert image_row_id in op.media_by_row, (
            "the image row must still be processed"
        )
        assert video_row_id not in op.media_by_row, (
            "a bare video path names no single frame -- the runner must "
            "refuse it rather than default to that video's first frame"
        )
    finally:
        resolver.close()


def test_address_run_resolves_a_relative_cell_in_a_non_full_path_media_column(
    tmp_path, monkeypatch
):
    """(c) An ADDRESS-requirement operator resolves its own media from
    metadata. The controller's pre-resolution (AppController.
    _absolutise_media_columns) must reach every column the active table's
    schema tags media_path -- not only a column literally named
    "full_path" -- so a relative cell in a differently-named media column
    still resolves against the project root before the operator reads it.
    """
    from operators.base import BaseOperator
    from operators.descriptor import (
        ExecutionMode, InputKind, InputSpec, MediaRequirement,
        ModeDescriptor, OperatorDescriptor, OutputColumn, OutputSpec,
    )
    from models.dataset import Dataset
    from models.query_engine import QueryEngine
    from artifacts.artifact_store import ArtifactStore
    from column_types.registry import ColumnTypeRegistry
    from operators.operator_registry import OperatorRegistry
    from controller import AppController

    class _AddressRecordingOperator(BaseOperator):
        def __init__(self):
            super().__init__()
            self.name = "p1_2c2_address_recorder"
            self.descriptor = OperatorDescriptor(
                name=self.name,
                version="1.0",
                description="Resolves its own media from a non-full_path column.",
                modes=(
                    ModeDescriptor(
                        mode=ExecutionMode.COLUMNS,
                        label="Recording",
                        inputs=(
                            InputSpec(
                                name="active_table", label="Active table",
                                kind=InputKind.ACTIVE_TABLE,
                            ),
                        ),
                        media_requirement=MediaRequirement.ADDRESS,
                        parameters=(),
                        output=OutputSpec(
                            columns=(OutputColumn(name="out", type_tag="numeric"),)
                        ),
                    ),
                ),
            )
            self.pixels_by_row: dict = {}

        def create_columns(self, row_id, media, metadata, run):
            payload = run.resolver.resolve_frame(metadata["clip"], "analysis")
            self.pixels_by_row[row_id] = payload.pixels
            return {"out": 1.0}

    project_root_dir = tmp_path / "project_root"
    (project_root_dir / "videos").mkdir(parents=True)
    video_path = project_root_dir / "videos" / "known_frames.mkv"
    _run_ffmpeg([
        "-f", "lavfi", "-i", "color=c=black:s=16x16:r=5:d=1",
        "-vf", "format=gray,geq=lum=0",
        "-pix_fmt", "gray", "-c:v", "ffv1",
        str(video_path),
    ])

    import pandas as pd

    csv_path = project_root_dir / "data.csv"
    pd.DataFrame({
        "clip": ["videos/known_frames.mkv"],
        "label": ["x"],
    }).to_csv(csv_path, index=False)

    resolver = MediaResolver(max_open_decoders=4)
    try:
        store = ArtifactStore(tmp_path / "artifacts", resolver=resolver)
        registry = ColumnTypeRegistry()
        registry.setup_defaults(store)
        dataset = Dataset()
        op_registry = OperatorRegistry()
        controller = AppController(
            dataset, QueryEngine(), store, registry, op_registry, resolver=resolver
        )

        # image_column=None: no full_path is set from a chosen column, but
        # every CSV column is still copied verbatim onto each row -- so
        # "clip" survives as its own column and, since its value has a
        # directory separator and a media extension, type inference tags
        # it media_path on accept (models/table_schema.py's
        # _looks_like_media_path).
        controller.load_csv_as_primary(csv_path, image_column=None)
        row_id = controller.get_visible_row_ids()[0]

        stored_cell = dataset.get_row(row_id, "frames")["clip"]
        assert "/" in stored_cell, (
            f"test setup did not produce a relative media cell: {stored_cell!r}"
        )
        schema = dataset.schema_for("frames")
        media_columns = {spec.name for spec in schema.columns_with_tag("media_path")}
        assert "clip" in media_columns, (
            f"'clip' was not inferred as a media_path column: {media_columns!r}"
        )
        assert controller._project_root == project_root_dir

        op = _AddressRecordingOperator()
        op_registry.register(op)
        _run_columns_and_wait(controller, op.name, [row_id], monkeypatch)

        assert row_id in op.pixels_by_row, (
            "the ADDRESS operator could not resolve its own media -- the "
            "relative cell in the non-full_path 'clip' column was never "
            "absolutised against the project root before the run started"
        )
        assert op.pixels_by_row[row_id].max() <= 2, (
            "resolved to something other than the all-black known-frame "
            "video"
        )

        # Dataset's own stored cell is untouched -- absolutisation happens
        # on the run's private snapshot, never on the stored table.
        assert dataset.get_row(row_id, "frames")["clip"] == stored_cell
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# P1.7-2 round 4: media/resolver.py must not hold its own copy of the
# still-image-path test. media/extensions.py::is_image_path is the single
# authority (moved there because two independent copies -- this module's
# former private _is_image_path and operators/frame_operator.py's own
# identical private copy -- will eventually disagree; sharing the
# IMAGE_EXTENSIONS constant while duplicating the test that reads it does
# not prevent that).
#
# Scoped to media/resolver.py only -- this file is that module's own test
# module. It is NOT a repo-wide sweep: column_types/renderers.py and
# operators/operator_registry.py each still carry a further inline copy of
# this same test, reported (not fixed) this round, so a repo-wide guard
# would fail on files this round does not touch.
# ---------------------------------------------------------------------------

def _module_ast(path: pathlib.Path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _names_referenced(tree) -> set[str]:
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _imports_is_image_path_from_media_extensions(tree) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "media.extensions":
            if any(alias.name == "is_image_path" for alias in node.names):
                return True
    return False


def test_resolver_does_not_reference_image_extensions_directly():
    # Would still pass if a future edit renamed the old private helper but
    # kept its body? No -- this checks for ANY reference to the
    # IMAGE_EXTENSIONS name in the module, not for a function called
    # "_is_image_path" -- so a reintroduced copy under a different name is
    # still caught.
    tree = _module_ast(pathlib.Path(__file__).parent.parent / "media" / "resolver.py")
    assert "IMAGE_EXTENSIONS" not in _names_referenced(tree), (
        "media/resolver.py references IMAGE_EXTENSIONS directly -- it "
        "should call media.extensions.is_image_path() instead of "
        "re-implementing the still-image test"
    )


def test_resolver_imports_the_shared_image_path_predicate():
    tree = _module_ast(pathlib.Path(__file__).parent.parent / "media" / "resolver.py")
    assert _imports_is_image_path_from_media_extensions(tree), (
        "media/resolver.py must import is_image_path from media.extensions"
    )


def test_the_image_extensions_reference_guard_catches_a_planted_violation(tmp_path):
    # Proves the guard is not vacuous: a module that reintroduces the old
    # inline check (even under a fresh name) fails it.
    planted = tmp_path / "planted_duplicate.py"
    planted.write_text(
        "import pathlib\n"
        "from media.extensions import IMAGE_EXTENSIONS\n"
        "\n"
        "def _looks_like_a_still_image(path):\n"
        "    return pathlib.PurePosixPath(path).suffix.lower() in IMAGE_EXTENSIONS\n",
        encoding="utf-8",
    )
    tree = _module_ast(planted)
    assert "IMAGE_EXTENSIONS" in _names_referenced(tree)


# ---------------------------------------------------------------------------
# CC-36: resolve_frame() must never leak a bare StopIteration (or PyAV's
# own av.error.EOFError) from next(container.decode(stream)) running dry
# -- both are internal iterator-protocol signals, not part of
# resolve_frame's documented (MediaResolverError, MediaAddressError)
# contract. _get_epoch_us and _resolve_range_or_bare's "policy=first on a
# bare path" branch are the two next(...) call sites in media/resolver.py.
#
# A NOTE ON WHY THESE TESTS USE A FAKE CONTAINER, NOT A REAL CORRUPT FILE.
# The natural way to trigger this is a real zero-frame or truncated video,
# and that was tried first and extensively: ffmpeg's own "-frames:v 0"
# produces a file so minimal that av.open() itself fails before either
# next() call is reached (a real gap, noted below, but a different one);
# truncating a valid fixture at every 5-byte offset from just past the
# header to well past the first frame found only three outcomes -- the
# container fails to open, container.seek(0, ...) itself fails (a THIRD
# exception type, also outside next()'s scope), or the file decodes fine
# -- with no byte offset in between where seek() succeeds and decode()
# then yields nothing. For this resolver's own seek(0, backward=True,
# any_frame=False) call, seek and decode appear to share the same
# keyframe data, so they fail or succeed together for a truncated MKV/
# FFV1 fixture. Since the two next() sites are what this finding is
# about, a minimal fake container/stream isolates exactly those two
# lines instead of fighting ffmpeg/PyAV internals for an artifact that
# may not be constructible at all for every container/codec pairing.
# ---------------------------------------------------------------------------

class _DryDecodeContainer:
    """A stand-in for an av.InputContainer whose .seek() succeeds (as a
    real one's does at the very start of a real file) but whose
    .decode() yields nothing -- next() on it raises StopIteration, the
    ordinary empty-iterator case."""

    def __init__(self, name="fake.mkv"):
        self.name = name

    def seek(self, *args, **kwargs):
        pass

    def decode(self, stream):
        return iter(())


class _EOFErrorIterator:
    def __iter__(self):
        return self

    def __next__(self):
        raise EOFError("synthetic PyAV end-of-file (av.error.EOFError shape)")


class _EOFErrorDecodeContainer(_DryDecodeContainer):
    """Like _DryDecodeContainer, but decode() raises EOFError from
    inside next() rather than exhausting cleanly -- what PyAV itself was
    confirmed (empirically, against a truncated fixture) to do for this
    exact situation, via av.error.EOFError, a subclass of the builtin
    EOFError."""

    def decode(self, stream):
        return _EOFErrorIterator()


class _DryDecodeStream:
    """A distinct name from the existing _FakeStream above (line ~1630,
    a different fixture's fake with a required time_base argument) --
    reusing that name would silently shadow it at module scope. index is
    read by _media_cache_key's stat-based cache key; nothing else about
    the stream is touched in either failure path -- the exception fires
    at next() itself, before stream.pts or stream.time_base would be
    read."""

    index = 0


def _real_file(tmp_path) -> str:
    # _get_epoch_us's cache-key lookup stats container.name before it
    # ever reaches seek()/decode(), so the fake container must still
    # name a real file on disk -- its content is irrelevant, only that
    # it exists.
    path = tmp_path / "fake.mkv"
    path.write_bytes(b"")
    return str(path)


@pytest.mark.parametrize(
    "container_cls", [_DryDecodeContainer, _EOFErrorDecodeContainer]
)
def test_get_epoch_us_converts_a_dry_decode_to_media_resolver_error(
    container_cls, tmp_path
):
    resolver = MediaResolver(max_open_decoders=2)
    try:
        with pytest.raises(MediaResolverError, match="no decodable frames"):
            resolver._get_epoch_us(
                container_cls(_real_file(tmp_path)), _DryDecodeStream()
            )
    finally:
        resolver.close()


@pytest.mark.parametrize(
    "container_cls", [_DryDecodeContainer, _EOFErrorDecodeContainer]
)
def test_resolve_range_or_bare_first_policy_converts_a_dry_decode_to_media_resolver_error(
    container_cls, tmp_path
):
    resolver = MediaResolver(max_open_decoders=2)
    try:
        real_path = _real_file(tmp_path)
        addr = from_path(real_path)  # bare path: is_bare and policy == "first"
        with pytest.raises(MediaResolverError, match="no decodable frames"):
            resolver._resolve_range_or_bare(
                container_cls(real_path), _DryDecodeStream(),
                epoch_us=0, addr=addr, policy="first",
            )
    finally:
        resolver.close()


def test_neither_dry_decode_conversion_raises_stop_iteration(tmp_path):
    # The exact regression this item fixes: pytest.raises(MediaResolverError)
    # above would also "pass" in a confusing way if the call raised
    # StopIteration inside a generator-based test helper (PEP 479 turns
    # that into a RuntimeError), so this checks the un-wrapped call
    # directly and explicitly names what must NOT come out.
    resolver = MediaResolver(max_open_decoders=2)
    try:
        try:
            resolver._get_epoch_us(
                _DryDecodeContainer(_real_file(tmp_path)), _DryDecodeStream()
            )
            pytest.fail("expected MediaResolverError")
        except StopIteration:
            pytest.fail(
                "_get_epoch_us leaked a bare StopIteration instead of "
                "raising MediaResolverError (CC-36)"
            )
        except MediaResolverError:
            pass  # expected
    finally:
        resolver.close()
