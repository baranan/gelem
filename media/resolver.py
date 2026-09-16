"""
media/resolver.py

MediaResolver -- the one module that decodes a user's media file into
pixels, per docs/media_architecture.md section 3.3. Video decoding uses
PyAV; still images use PIL with EXIF orientation applied. Nothing here
opens a Qt binding.

This sub-item (P1.2a) supports still images, bare video paths, #t= time
points and #t= time ranges. A #f= address raises NotImplementedError --
resolving a frame ordinal against a variable-frame-rate file needs a
per-file index of frame presentation times, which is P1.2b's job
(docs/media_architecture.md section 3.6, decision 8).

Reuses media/media_address.py's pure frame-selection functions
(select_frame, region_to_pixels) rather than re-deriving their rules --
this module's own job is only to supply real frame timings and real
pixels for those functions to work with.

Frame ordinal means only what decision 8 means: the zero-based position
of a frame in presentation order within the WHOLE selected stream. For
video, this module cannot supply that yet -- computing it would require
decoding every frame between the start of the file and the answer, which
is exactly what the cost rule below forbids, and the per-file index that
would make it both possible and cheap is P1.2b's job, not this module's.
So every video resolve reports frame_ordinal as None. A still image has
no such position to report against a file-wide timeline of frames, but
it does have exactly one frame, so it keeps 0. The index this module
finds a frame at, within whatever local window of frame times it
actually decoded, is used internally to pick the frame out of that
window and is never exposed -- it is not the value decision 8 means, and
reporting it as frame_ordinal would misrepresent it as one.

Cost rule: resolving a bare path, a #t= point, or a #t= range never
demuxes or decodes the whole file. The container is seeked to the
nearest keyframe at or before the point of interest, and frames are
decoded forward only until enough is known to answer the query --
one frame past the target for a time point, or through the end of the
requested range for a range or bare path. A bare path's default
('first') policy needs only its first frame; 'midpoint' on a bare path
uses the container's own duration metadata to find a seek target near
the middle, so it never decodes to the true end of file to find it.

Decoder pool: open PyAV containers are kept in a pool keyed by absolute
file path, bounded by max_open_decoders and evicted least-recently-used.
A container is used by one thread at a time; a caller wanting a file
whose container is busy, with the pool already at its bound and no idle
entry to evict, waits.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import struct
import threading
from collections import OrderedDict
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union

import av
import numpy as np
from av.sidedata.sidedata import Type as _SideDataType
from PIL import Image, ImageOps

from media.media_address import (
    MediaAddress,
    MediaAddressError,
    parse as parse_address,
    region_to_pixels,
    select_frame,
)

__all__ = ["MediaResolver", "FramePayload", "MediaResolverError"]


class MediaResolverError(ValueError):
    """Raised when the FILE, not the address, cannot be resolved as asked --
    an unsupported display matrix, a missing video stream, or (for a
    bare-path midpoint request) no duration the container will report
    without decoding to the true end. A problem with what the address
    itself asks for, given the file, is raised as MediaAddressError
    instead -- see decision 11's own resolve-time refusals, which this
    module reaches by calling select_frame().
    """


_PURPOSES = ("display", "analysis")

# Mirrors the image half of media/extensions.py's MEDIA_EXTENSIONS. No
# authoritative image/video split exists yet (see docs/review/p1.2-survey.md
# section 10's note on column_types/renderers.py's own, separate, private
# split) -- this is this module's own minimal, local dispatch, not a claim
# to be the authority other modules should import.
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"})

_DRIVE_LETTER_PATH = re.compile(r"^[A-Za-z]:/")


def _is_absolute(path: str) -> bool:
    """True for a posix-style absolute path, a UNC path ('//server/share'),
    or a drive-letter path ('C:/Users/x') -- the same three forms an
    address path may hold after media_address.py's forward-slash
    normalisation (decision 9). A UNC path already starts with '/' after
    that normalisation, so the first check covers it too.
    """
    if path.startswith("/"):
        return True
    return bool(_DRIVE_LETTER_PATH.match(path))


def _is_image_path(path: str) -> bool:
    return pathlib.PurePosixPath(path).suffix.lower() in _IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# FramePayload
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class FramePayload:
    """The result of resolving one address to a picture.

    pixels: RGB uint8 numpy array, shape (height, width, 3). Upright (the
        container's display matrix already applied for video; EXIF
        orientation already applied for a still image) and cropped to
        the address's region, if one was given.
    frame_ordinal: the zero-based position of this frame in presentation
        order within the whole selected stream (decision 8) -- 0 for a
        still image, which has exactly one frame. For video, None: this
        module has no per-file frame-time index yet to compute a
        file-wide position from, and reporting anything narrower (such
        as a position within whatever it happened to decode to answer
        this query) would claim a meaning decision 8 does not give it.
        P1.2b's index is what makes a real value possible here.
    presentation_time_us: the selected frame's presentation time in
        integer microseconds on the file's own clock (decision 10: zero
        is the first presented frame of the selected video stream). None
        for a still image, which has no timeline.
    width, height: the pixel array's own dimensions, after rotation and
        crop -- always consistent with pixels.shape, provided for a
        caller that wants them without indexing the array.
    purpose: 'display' or 'analysis', echoed back from the request.
    """

    pixels: np.ndarray
    frame_ordinal: Optional[int]
    presentation_time_us: Optional[int]
    width: int
    height: int
    purpose: str


# ---------------------------------------------------------------------------
# Orientation -- decision 6. Quarter turns and horizontal mirroring are
# read directly off the container's display matrix (AV_PKT_DATA_DISPLAYMATRIX),
# exposed by PyAV as decoded-frame side data. The eight signatures below are
# the exact Q16.16 fixed-point values FFmpeg writes for the eight members of
# the square's symmetry group (four quarter turns, each with and without a
# horizontal flip) -- derived empirically against ffmpeg's own auto-rotated
# output (docs/review/p1.2-survey.md was silent on this; verified by hand
# for this sub-item against `-display_rotation` / `-display_hflip` fixtures).
# Any other matrix -- a shear, a scale, an odd angle -- is refused rather
# than approximated, per decision 6.
# ---------------------------------------------------------------------------

_ROTATION_SIGNATURES: Dict[Tuple[int, int, int, int], Tuple[int, bool]] = {
    (65536, 0, 0, 65536): (0, False),
    (0, -65536, 65536, 0): (1, False),
    (-65536, 0, 0, -65536): (2, False),
    (0, 65536, -65536, 0): (3, False),
    (-65536, 0, 0, 65536): (0, True),
    (0, -65536, -65536, 0): (1, True),
    (65536, 0, 0, -65536): (2, True),
    (0, 65536, 65536, 0): (3, True),
}

_IDENTITY_FIXED_ROW3 = (0, 0, 0, 0, 1 << 30)  # matrix[2], [5], [6], [7], [8]


def _orientation_for_frame(frame: "av.VideoFrame") -> Tuple[int, bool]:
    """Return (k, hflip): rotate the decoded frame np.rot90(pixels, k=k)
    then, if hflip, np.fliplr the result, to make it upright (decision 6).

    No display matrix at all means no rotation -- the ordinary case.
    """
    side_data = frame.side_data.get(_SideDataType.DISPLAYMATRIX)
    if side_data is None:
        return 0, False

    raw = bytes(side_data)
    if len(raw) != 36:
        raise MediaResolverError(
            f"display matrix side data is {len(raw)} bytes, expected 36"
        )
    matrix = struct.unpack("<9i", raw)
    m0, m1, m2, m3, m4, m5, m6, m7, m8 = matrix
    if (m2, m5, m6, m7, m8) != _IDENTITY_FIXED_ROW3:
        raise MediaResolverError(
            f"unsupported display matrix (shear, scale or translation "
            f"component present): {matrix}"
        )
    signature = (m0, m1, m3, m4)
    if signature not in _ROTATION_SIGNATURES:
        raise MediaResolverError(
            f"unsupported display matrix (not a quarter turn or horizontal "
            f"mirror): {matrix}"
        )
    return _ROTATION_SIGNATURES[signature]


def _apply_orientation(pixels: np.ndarray, k: int, hflip: bool) -> np.ndarray:
    if k:
        pixels = np.rot90(pixels, k=k)
    if hflip:
        pixels = np.fliplr(pixels)
    return np.ascontiguousarray(pixels)


# ---------------------------------------------------------------------------
# Time <-> ticks. Exact Fraction arithmetic throughout (decision 13's
# "prefer integer microseconds or rational PTS to floating-point seconds").
# ---------------------------------------------------------------------------

def _ticks_to_us(ticks: int, time_base: Fraction) -> int:
    return round(Fraction(ticks) * time_base * 1_000_000)


def _us_to_ticks(us: int, time_base: Fraction) -> int:
    return round(Fraction(us, 1_000_000) / time_base)


# ---------------------------------------------------------------------------
# The decoder pool -- an LRU pool of open PyAV containers, keyed by
# absolute file path, bounded by max_open_decoders, with per-file mutual
# exclusion. See the module docstring.
# ---------------------------------------------------------------------------

class _PoolEntry:
    __slots__ = ("container", "busy")

    def __init__(self, container):
        self.container = container
        self.busy = True


class _DecoderPool:
    def __init__(self, max_open_decoders: int):
        if max_open_decoders < 1:
            raise ValueError("max_open_decoders must be at least 1")
        self._max_open = max_open_decoders
        self._lock = threading.Lock()
        self._not_busy = threading.Condition(self._lock)
        # Ordered oldest-idle-first is approximated by moving an entry to
        # the end on every acquire, so the front of the dict is always the
        # least-recently-acquired entry -- the LRU eviction candidate.
        self._entries: "OrderedDict[str, _PoolEntry]" = OrderedDict()

    def acquire(self, path: str) -> "av.container.InputContainer":
        with self._not_busy:
            while True:
                entry = self._entries.get(path)
                if entry is not None:
                    if not entry.busy:
                        entry.busy = True
                        self._entries.move_to_end(path)
                        return entry.container
                    self._not_busy.wait()
                    continue

                if len(self._entries) < self._max_open:
                    container = av.open(path)
                    self._entries[path] = _PoolEntry(container)
                    return container

                if self._evict_one_idle_locked():
                    container = av.open(path)
                    self._entries[path] = _PoolEntry(container)
                    return container

                self._not_busy.wait()

    def release(self, path: str) -> None:
        with self._not_busy:
            entry = self._entries.get(path)
            if entry is not None:
                entry.busy = False
                self._not_busy.notify_all()

    def _evict_one_idle_locked(self) -> bool:
        for candidate_path, entry in self._entries.items():
            if not entry.busy:
                entry.container.close()
                del self._entries[candidate_path]
                return True
        return False

    def close_all(self) -> None:
        with self._not_busy:
            for entry in self._entries.values():
                entry.container.close()
            self._entries.clear()


# ---------------------------------------------------------------------------
# MediaResolver
# ---------------------------------------------------------------------------

class MediaResolver:
    """Decodes a media address into a FramePayload. See the module
    docstring for scope, the cost rule, and the decoder pool.
    """

    def __init__(self, *, max_open_decoders: int):
        self._pool = _DecoderPool(max_open_decoders)
        self._stream_info_lock = threading.Lock()
        # (abs_path, container_stream_index) -> epoch_us, the raw
        # (unrebased) presentation time of that stream's first frame
        # (decision 10: this is where this stream's clock reads zero).
        self._epoch_cache: Dict[Tuple[str, int], int] = {}

    def close(self) -> None:
        """Close every pooled container. For test and shutdown cleanup."""
        self._pool.close_all()

    def resolve_frame(
        self,
        address: Union[str, MediaAddress],
        purpose: str,
        *,
        policy: str = "first",
    ) -> FramePayload:
        """Resolve one address to a picture.

        `policy` ('first' or 'midpoint', default 'first') chooses the
        representative frame for a range or bare-path address (decision
        4). It is not part of the address string -- it is a resolve-time
        argument, exactly as section 3.6 decision 4 specifies -- and is
        ignored for a time-point address, which decision 2 already
        selects a single frame for unambiguously.
        """
        if purpose not in _PURPOSES:
            raise ValueError(f"purpose must be one of {_PURPOSES}, got {purpose!r}")

        addr = address if isinstance(address, MediaAddress) else parse_address(address)

        if not _is_absolute(addr.path):
            raise MediaAddressError(
                f"resolve_frame requires an absolute address, got a relative "
                f"path: {addr.path!r}"
            )

        if addr.frame is not None:
            raise NotImplementedError(
                "#f= addresses are not resolved yet: resolving a frame "
                "ordinal against a file's real (possibly variable) frame "
                "timings needs a per-file frame-time index, which is "
                "P1.2b's job, not P1.2a's."
            )

        if _is_image_path(addr.path):
            return self._resolve_still_image(addr, purpose)
        return self._resolve_video_frame(addr, purpose, policy)

    # -- still images ------------------------------------------------------

    def _resolve_still_image(self, addr: MediaAddress, purpose: str) -> FramePayload:
        with Image.open(addr.path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            pixels = np.array(image, dtype=np.uint8)
        # The file handle is closed by the `with` block above -- an open
        # handle would otherwise lock the file on Windows for as long as
        # this MediaResolver instance lives, for no benefit: everything
        # after this point works from `pixels`, already fully decoded.
        height, width = pixels.shape[:2]

        if addr.region is not None:
            pixel_region = region_to_pixels(addr.region, width, height)
            pixels = np.ascontiguousarray(pixels[
                pixel_region.top: pixel_region.top + pixel_region.height,
                pixel_region.left: pixel_region.left + pixel_region.width,
            ])
            height, width = pixels.shape[:2]

        return FramePayload(
            pixels=pixels,
            frame_ordinal=0,
            presentation_time_us=None,
            width=width,
            height=height,
            purpose=purpose,
        )

    # -- video ---------------------------------------------------------

    def _resolve_video_frame(
        self, addr: MediaAddress, purpose: str, policy: str
    ) -> FramePayload:
        container = self._pool.acquire(addr.path)
        try:
            stream = self._select_video_stream(container, addr)
            epoch_us = self._get_epoch_us(container, stream)

            if addr.time_us is not None:
                # The second element is this resolve's own local index
                # into the frame-time window it decoded -- needed inside
                # the helper to pick `frame` out of that window, never a
                # file-wide position (decision 8), so it is not carried
                # any further than this unpacking.
                frame, _local_index, pts_us = self._resolve_point(
                    container, stream, epoch_us, addr
                )
            else:
                # A range, or a bare path (decision 4: the range covering
                # the whole file).
                frame, _local_index, pts_us = self._resolve_range_or_bare(
                    container, stream, epoch_us, addr, policy
                )

            k, hflip = _orientation_for_frame(frame)
            pixels = frame.to_ndarray(format="rgb24")
            pixels = _apply_orientation(pixels, k, hflip)
            height, width = pixels.shape[:2]

            if addr.region is not None:
                pixel_region = region_to_pixels(addr.region, width, height)
                pixels = np.ascontiguousarray(pixels[
                    pixel_region.top: pixel_region.top + pixel_region.height,
                    pixel_region.left: pixel_region.left + pixel_region.width,
                ])
                height, width = pixels.shape[:2]

            return FramePayload(
                pixels=pixels,
                # Decision 8's frame ordinal is a file-wide position that
                # this module cannot compute without the per-file index
                # P1.2b builds -- see the module and FramePayload
                # docstrings. _local_index above is not that value.
                frame_ordinal=None,
                presentation_time_us=pts_us,
                width=width,
                height=height,
                purpose=purpose,
            )
        finally:
            self._pool.release(addr.path)

    def _select_video_stream(self, container, addr: MediaAddress):
        video_streams = list(container.streams.video)
        if not video_streams:
            raise MediaResolverError(f"{addr.path!r} has no video stream")

        if addr.stream is None:
            index = 0
        else:
            if addr.stream.kind != "v":
                raise MediaAddressError(
                    f"resolve_frame resolves a video frame; the address "
                    f"names an audio stream selector: {addr.stream!r}"
                )
            index = addr.stream.index

        if index >= len(video_streams):
            raise MediaAddressError(
                f"stream selector v={index} is beyond the last video "
                f"stream ({len(video_streams)} video streams in "
                f"{addr.path!r})"
            )
        return video_streams[index]

    def _get_epoch_us(self, container, stream) -> int:
        cache_key = (self._container_path(container), stream.index)
        with self._stream_info_lock:
            cached = self._epoch_cache.get(cache_key)
        if cached is not None:
            return cached

        container.seek(0, backward=True, any_frame=False, stream=stream)
        first_frame = next(container.decode(stream))
        epoch_us = _ticks_to_us(first_frame.pts, stream.time_base)

        with self._stream_info_lock:
            self._epoch_cache[cache_key] = epoch_us
        return epoch_us

    @staticmethod
    def _container_path(container) -> str:
        # container.name is the path av.open() was given -- the same
        # absolute, forward-slash-normalised string used as the pool key,
        # so this is stable across the container being closed and reopened.
        return container.name

    def _resolve_point(
        self, container, stream, epoch_us: int, addr: MediaAddress
    ):
        target_us = addr.time_us
        raw_target_ticks = _us_to_ticks(target_us + epoch_us, stream.time_base)
        container.seek(raw_target_ticks, backward=True, any_frame=False, stream=stream)

        frame_times: List[int] = []
        frames: List = []
        for frame in container.decode(stream):
            pts_us = _ticks_to_us(frame.pts, stream.time_base) - epoch_us
            frame_times.append(pts_us)
            frames.append(frame)
            if pts_us > target_us:
                break

        # This index is local to the frame_times window just decoded, not
        # a file-wide position (decision 8) -- see the module docstring.
        local_index = select_frame(addr, frame_times)
        return frames[local_index], local_index, frame_times[local_index]

    def _resolve_range_or_bare(
        self, container, stream, epoch_us: int, addr: MediaAddress, policy: str
    ):
        is_bare = addr.time_range_us is None

        if is_bare and policy == "first":
            # Decision 4's default case for a bare path: just the first
            # frame. No seek needed beyond the start; no duration lookup.
            container.seek(0, backward=True, any_frame=False, stream=stream)
            frame = next(container.decode(stream))
            pts_us = _ticks_to_us(frame.pts, stream.time_base) - epoch_us
            local_index = select_frame(addr, [pts_us], policy=policy)
            return frame, local_index, pts_us

        if is_bare:
            # 'midpoint' on a bare path: decision 4 treats the whole file
            # as one range, but finding the nearest frame to its true
            # midpoint by decoding from the start would decode the whole
            # file. The nearest frame to any target time is always one of
            # the two frames immediately bracketing it in presentation
            # order (frame times are monotonic), so seeking near the
            # midpoint (from the container's own duration metadata -- no
            # decode) and decoding just enough to find that bracketing
            # pair answers decision 4's rule without decoding the file in
            # between.
            return self._resolve_bare_midpoint(container, stream, epoch_us, addr)

        start_us, end_us = addr.time_range_us
        raw_start_ticks = _us_to_ticks(start_us + epoch_us, stream.time_base)
        container.seek(raw_start_ticks, backward=True, any_frame=False, stream=stream)

        frame_times: List[int] = []
        frames: List = []
        for frame in container.decode(stream):
            pts_us = _ticks_to_us(frame.pts, stream.time_base) - epoch_us
            frame_times.append(pts_us)
            frames.append(frame)

            if policy == "first" and start_us <= pts_us < end_us:
                # The earliest member has been found; no need to decode
                # any further to answer a 'first' query.
                break
            if pts_us >= end_us:
                # We now know everything decision 3's membership test
                # needs to know about this range's upper edge.
                break

        local_index = select_frame(addr, frame_times, policy=policy)
        return frames[local_index], local_index, frame_times[local_index]

    def _resolve_bare_midpoint(self, container, stream, epoch_us: int, addr: MediaAddress):
        span_us = self._stream_span_us(container, stream)
        if span_us is None:
            raise MediaResolverError(
                f"cannot determine {addr.path!r}'s duration to find a "
                f"midpoint frame without decoding to the end of the file"
            )
        midpoint_us = span_us // 2
        raw_ticks = _us_to_ticks(midpoint_us + epoch_us, stream.time_base)
        container.seek(raw_ticks, backward=True, any_frame=False, stream=stream)

        frame_times: List[int] = []
        frames: List = []
        for frame in container.decode(stream):
            pts_us = _ticks_to_us(frame.pts, stream.time_base) - epoch_us
            frame_times.append(pts_us)
            frames.append(frame)
            if pts_us >= midpoint_us:
                break

        # frame_times now holds one or more frames at/before the midpoint,
        # optionally followed by exactly one frame at/after it -- the
        # bracketing pair decision 4's "nearest, ties to earlier" rule
        # needs. The one at/before the midpoint is always last-but-one
        # unless we hit end of stream before ever passing the midpoint, or
        # the very first decoded frame already sits at/after it.
        if frame_times[-1] < midpoint_us:
            # End of stream reached before the midpoint -- the last frame
            # decoded is the only candidate.
            return frames[-1], len(frame_times) - 1, frame_times[-1]

        if len(frame_times) == 1:
            # The seek landed on a keyframe already at/after the midpoint
            # (the file's very first frame, most likely) -- no earlier
            # candidate exists.
            return frames[0], 0, frame_times[0]

        before_index, after_index = len(frame_times) - 2, len(frame_times) - 1
        before_us, after_us = frame_times[before_index], frame_times[after_index]
        if midpoint_us - before_us <= after_us - midpoint_us:
            return frames[before_index], before_index, before_us
        return frames[after_index], after_index, after_us

    def _stream_span_us(self, container, stream) -> Optional[int]:
        if stream.duration is not None and stream.time_base is not None:
            return _ticks_to_us(stream.duration, stream.time_base)
        if container.duration is not None:
            return container.duration
        return None
