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

import dataclasses
import pathlib
import shutil
import struct
import subprocess
import sys
import threading

# Add project root to Python path, matching the other test modules.
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest
from PIL import Image, ImageOps

from media.media_address import MediaAddressError, from_path
from media.resolver import MediaResolver, MediaResolverError

FFMPEG_MISSING = shutil.which("ffmpeg") is None
pytestmark = pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg is not on PATH")


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
# Decision 8 -- frame_ordinal is a file-wide presentation-order position,
# which this module cannot supply without the per-file frame-time index
# P1.2b builds. Every video resolve must report None, whatever address
# shape asked for the frame -- a point, a range, or a bare path.
# ---------------------------------------------------------------------------

def test_video_frame_ordinal_is_always_none(tmp_path):
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


def test_frame_ordinal_address_raises_not_implemented():
    resolver = MediaResolver(max_open_decoders=2)
    try:
        addr = dataclasses.replace(from_path("C:/videos/clip.mp4"), frame=5)
        with pytest.raises(NotImplementedError):
            resolver.resolve_frame(addr, purpose="display")
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
