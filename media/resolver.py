"""
media/resolver.py

MediaResolver -- the one module that decodes a user's media file into
pixels, per docs/media_architecture.md section 3.3. Video decoding uses
PyAV; still images use PIL with EXIF orientation applied. Nothing here
opens a Qt binding.

This sub-item (P1.2b) adds #f= resolution, true frame ordinals, and
decode_video_span, on top of P1.2a's still images, bare video paths,
#t= time points and #t= time ranges.

Reuses media/media_address.py's pure frame-selection functions
(select_frame, region_to_pixels) rather than re-deriving their rules --
this module's own job is only to supply real frame timings and real
pixels for those functions to work with.

Frame ordinal means only what decision 8 means: the zero-based position
of a frame in presentation order within the WHOLE selected stream,
counted after any edit list (decision 12). A still image has no such
position to report against a file-wide timeline of frames, but it does
have exactly one frame, so it keeps 0.

For video, computing a real ordinal needs a per-file index of frame
presentation times, built by demuxing (not decoding) every packet of
the file once -- see "Frame-time index" below. A #f= resolve always
builds (or reuses) that index, because it cannot answer without one. A
#t= or bare-path resolve reports the true ordinal when the index for
that file already exists (built by an earlier #f= resolve on the same
MediaResolver instance) and None otherwise -- it never builds the index
itself, per the cost rule below.

Cost rule: resolving a bare path, a #t= point, or a #t= range never
demuxes or decodes the whole file. The container is seeked to the
nearest keyframe at or before the point of interest, and frames are
decoded forward only until enough is known to answer the query --
one frame past the target for a time point, or through the end of the
requested range for a range or bare path. A bare path's default
('first') policy needs only its first frame; 'midpoint' on a bare path
uses the container's own duration metadata to find a seek target near
the middle, so it never decodes to the true end of file to find it.
The one deliberate exception is #f=: decision 8's consequence for P1.2
is that resolving a frame ordinal against a file's real (possibly
variable) frame timings needs the per-file index, built on first use.

Frame-time index: for one (absolute path, video stream index), the
sorted presentation times of every frame the decoder actually presents.
Built by demuxing packets only -- no decode -- because a packet's pts
already carries its presentation time. Demux order is decode order, not
presentation order (a stream with B-frames interleaves out-of-order
packets), so the index sorts by pts rather than trusting packet order.
A packet with no pts (the empty flush packet libav yields at end of
stream) is skipped. A packet the edit list marks discard
(`packet.is_discard`) is skipped too -- it is reference data the
decoder needs but never presents, and per decision 12, frame 0 and time
0 are the same frame, which only holds if discarded pre-roll packets
are excluded. The index is cached per (path, stream index, file size,
mtime) for the life of the MediaResolver instance, so a file edited
after it was indexed is re-indexed rather than served a stale answer.
"Cached", not "built exactly once": the cache check and the build are
not one atomic step (`_get_frame_index` checks the cache, then builds
outside the lock on a miss so one slow build never blocks an unrelated
file's lookup), so two threads racing to resolve the same file's first
`#f=` address can both demux it and both write an equivalent result --
wasted work, not a wrong one, since the index is a pure function of the
file (see `docs/known_defects.md`).

Its first entry is required to equal the epoch this
module already caches from decoding (P1.2a's `_get_epoch_us`); if they
disagree, resolving raises MediaResolverError rather than silently
choosing one of the two candidate answers.

Decoder pool: open PyAV containers are kept in a pool bounded in total
by max_open_decoders and evicted least-recently-used. More than one
container may be open for the same file at once, within that total
bound, so a second thread wanting a busy file does not wait while
capacity remains elsewhere. A caller wanting a file whose every open
container is busy, with the pool already at its bound and nothing idle
to evict, waits -- unless the calling thread itself already holds every
busy container in the pool, in which case no other thread could ever
free one, so waiting would deadlock (a thread iterating a
decode_video_span alone at the bound, that also calls resolve_frame, for
instance) and MediaResolverError is raised instead. Holding some, but
not all, of the busy containers is not this case: another thread's own
release still frees capacity, so waiting is correct there. Opening and
closing a container (av.open, container.close) both happen outside the
pool's lock, so one slow open -- Google Drive Streaming, say -- never
blocks another file's resolve. MediaResolver.close() never closes a
container a live resolve_frame call or decode_video_span iterator is
using -- it closes every idle one immediately, marks a busy one to close
itself on release, and marks the pool closed so no later acquire
silently opens a fresh container from a resolver the caller believes is
shut down.
"""

from __future__ import annotations

import bisect
import dataclasses
import pathlib
import re
import struct
import threading
from collections import OrderedDict
from fractions import Fraction
from typing import Dict, Iterator, List, Optional, Tuple, Union

import av
import numpy as np
from av.sidedata.sidedata import Type as _SideDataType
from PIL import Image, ImageOps

from media.extensions import IMAGE_EXTENSIONS
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
    return pathlib.PurePosixPath(path).suffix.lower() in IMAGE_EXTENSIONS


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
        still image, which has exactly one frame. For video, the real
        value when a #f= address was resolved (which always builds or
        reuses the per-file frame-time index to answer) or when a #t=
        or bare-path address was resolved and that file's index already
        existed from an earlier #f= resolve on this MediaResolver
        instance; otherwise None, because computing it without the
        index would mean decoding the whole file, which the cost rule
        forbids for those address shapes.
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


def _decoded_frame_to_pixels(
    frame: "av.VideoFrame", region
) -> Tuple[np.ndarray, int, int]:
    """Turn one decoded PyAV video frame into an upright RGB array,
    cropped to `region` (a media_address.Region, or None), applying
    orientation (decision 6) before region (decision 5), exactly as
    decision 5 requires. Returns (pixels, width, height).

    Shared by resolve_frame's video path and decode_video_span, which
    otherwise independently reimplemented this same transform -- a fix
    to one applied without the other would make the same address decode
    to two different pictures depending only on which entry point asked.
    """
    k, hflip = _orientation_for_frame(frame)
    pixels = frame.to_ndarray(format="rgb24")
    pixels = _apply_orientation(pixels, k, hflip)
    height, width = pixels.shape[:2]

    if region is not None:
        pixel_region = region_to_pixels(region, width, height)
        pixels = np.ascontiguousarray(pixels[
            pixel_region.top: pixel_region.top + pixel_region.height,
            pixel_region.left: pixel_region.left + pixel_region.width,
        ])
        height, width = pixels.shape[:2]

    return pixels, width, height


# ---------------------------------------------------------------------------
# Time <-> ticks. Exact Fraction arithmetic throughout (decision 13's
# "prefer integer microseconds or rational PTS to floating-point seconds").
# ---------------------------------------------------------------------------

def _ticks_to_us(ticks: int, time_base: Fraction) -> int:
    return round(Fraction(ticks) * time_base * 1_000_000)


def _us_to_ticks(us: int, time_base: Fraction) -> int:
    return round(Fraction(us, 1_000_000) / time_base)


# ---------------------------------------------------------------------------
# The decoder pool -- an LRU pool of open PyAV containers, bounded in total
# by max_open_decoders. More than one container may be open for the same
# path at once (P1.2b rule 5a), so entries are keyed by an opaque identity,
# not by path -- a path lookup scans for an idle match instead. See the
# module docstring for the deadlock-avoidance and lock-scope rules.
# ---------------------------------------------------------------------------

class _PoolEntry:
    __slots__ = ("container", "path", "busy", "owner_thread", "close_on_release")

    def __init__(self, container, path: str):
        self.container = container
        self.path = path
        self.busy = True
        self.owner_thread: Optional[int] = None
        # Set when close_all() finds this entry busy: it cannot be closed
        # out from under whatever is using it, so release() closes it
        # instead, once that use ends.
        self.close_on_release = False


class _DecoderPool:
    def __init__(self, max_open_decoders: int):
        if max_open_decoders < 1:
            raise ValueError("max_open_decoders must be at least 1")
        self._max_open = max_open_decoders
        self._lock = threading.Lock()
        self._not_busy = threading.Condition(self._lock)
        # Ordered oldest-acquired-first: every acquire (new or reused) moves
        # its entry to the end, so the front is always the LRU eviction
        # candidate. Keyed by id(entry), since a path may now map to more
        # than one entry.
        self._entries: "OrderedDict[int, _PoolEntry]" = OrderedDict()
        # Opens reserved but not yet turned into an entry -- counted against
        # the bound so a slow, unlocked av.open() cannot let concurrent
        # acquires overshoot max_open_decoders.
        self._pending_opens = 0
        # Thread identity -> number of entries that thread currently holds
        # busy, across every path. Used only to detect the self-deadlock
        # rule 5b describes; never consulted to decide whose turn is next.
        self._held_counts: Dict[int, int] = {}
        # Set by close_all(): every later acquire() is refused outright,
        # rather than silently opening a fresh container from a pool the
        # caller believes is shut down.
        self._closed = False

    def acquire(self, path: str) -> "av.container.InputContainer":
        while True:
            with self._not_busy:
                if self._closed:
                    raise MediaResolverError(
                        "the decoder pool is closed -- "
                        "MediaResolver.close() was called"
                    )

                entry = self._find_idle_entry_locked(path)
                if entry is not None:
                    self._mark_busy_locked(entry)
                    return entry.container

                to_close: Optional[_PoolEntry] = None
                if len(self._entries) + self._pending_opens >= self._max_open:
                    to_close = self._pop_idle_entry_locked()
                    if to_close is None:
                        # Waiting can only ever end if some OTHER thread
                        # releases a container -- if the calling thread
                        # itself holds every busy container in the pool,
                        # no other thread ever will, and waiting would
                        # deadlock. Holding some, but not all, of them is
                        # fine: a different thread's release still frees
                        # capacity, so waiting is the right thing to do.
                        thread_id = threading.get_ident()
                        held_by_caller = self._held_counts.get(thread_id, 0)
                        total_busy = sum(
                            1 for e in self._entries.values() if e.busy
                        )
                        if held_by_caller > 0 and held_by_caller >= total_busy:
                            raise MediaResolverError(
                                f"cannot open {path!r}: the decoder pool is "
                                f"already at its bound ({self._max_open}) "
                                f"and every busy container is held by this "
                                f"thread, so waiting for a release could "
                                f"never end"
                            )
                        self._not_busy.wait()
                        continue
                self._pending_opens += 1

            # The slow part -- opening a file, and possibly closing the
            # evicted one -- happens with the lock released (rule 5c), so
            # one slow open never blocks another file's acquire or release.
            try:
                if to_close is not None:
                    to_close.container.close()
                container = av.open(path)
            except BaseException:
                with self._not_busy:
                    self._pending_opens -= 1
                    self._not_busy.notify_all()
                raise

            # The reservation is released and the new entry is registered
            # in the SAME lock acquisition: doing these as two separate
            # `with` blocks left a window where another thread's acquire()
            # could see the freed reservation but not yet see the entry it
            # was reserved for, undercounting a container that was, in
            # fact, already open -- and so let the bound be overshot.
            with self._not_busy:
                self._pending_opens -= 1
                entry = _PoolEntry(container, path)
                self._entries[id(entry)] = entry
                self._mark_busy_locked(entry)
                self._not_busy.notify_all()
                return entry.container

    def release(self, container) -> None:
        to_close: Optional[_PoolEntry] = None
        with self._not_busy:
            entry = self._find_entry_by_container_locked(container)
            if entry is None:
                return
            entry.busy = False
            thread_id = entry.owner_thread
            entry.owner_thread = None
            if thread_id is not None:
                remaining = self._held_counts.get(thread_id, 0) - 1
                if remaining > 0:
                    self._held_counts[thread_id] = remaining
                else:
                    self._held_counts.pop(thread_id, None)
            if entry.close_on_release:
                del self._entries[id(entry)]
                to_close = entry
            self._not_busy.notify_all()
        if to_close is not None:
            to_close.container.close()

    def _find_idle_entry_locked(self, path: str) -> Optional[_PoolEntry]:
        for entry in self._entries.values():
            if entry.path == path and not entry.busy:
                return entry
        return None

    def _find_entry_by_container_locked(self, container) -> Optional[_PoolEntry]:
        for entry in self._entries.values():
            if entry.container is container:
                return entry
        return None

    def _mark_busy_locked(self, entry: _PoolEntry) -> None:
        entry.busy = True
        entry.owner_thread = threading.get_ident()
        self._entries.move_to_end(id(entry))
        self._held_counts[entry.owner_thread] = (
            self._held_counts.get(entry.owner_thread, 0) + 1
        )

    def _pop_idle_entry_locked(self) -> Optional[_PoolEntry]:
        for entry_id, entry in list(self._entries.items()):
            if not entry.busy:
                del self._entries[entry_id]
                return entry
        return None

    def close_all(self) -> None:
        """Close every IDLE container now. A busy one -- in use by a live
        resolve_frame call or, for much longer, a decode_video_span
        iterator -- is not touched; it is marked to close itself when
        released instead, so a caller mid-iteration never has its
        container pulled out from under it. The pool is marked closed
        either way, so no later acquire() silently opens a fresh one.
        """
        with self._not_busy:
            self._closed = True
            idle_entries = []
            for entry_id, entry in list(self._entries.items()):
                if entry.busy:
                    entry.close_on_release = True
                else:
                    del self._entries[entry_id]
                    idle_entries.append(entry)
            self._not_busy.notify_all()
        for entry in idle_entries:
            entry.container.close()


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
        # (abs_path, container_stream_index, file_size, mtime_ns) ->
        # epoch_us, the raw (unrebased) presentation time of that stream's
        # first frame (decision 10: this is where this stream's clock
        # reads zero). Keyed identically to _frame_index_cache below,
        # through the same _media_cache_key(), so the two caches can never
        # disagree about which file version they each describe -- a file
        # replaced at the same path gets a new key in both, rather than a
        # fresh index compared against a stale epoch from the old file.
        self._epoch_cache: Dict[Tuple[str, int, int, int], int] = {}
        self._frame_index_lock = threading.Lock()
        # (abs_path, container_stream_index, file_size, mtime_ns) -> sorted
        # list of that stream's presented frame times, zero-based against
        # the same epoch as _epoch_cache (see the module docstring's "Frame-
        # time index"). Keying on size and mtime means an edited file is
        # re-indexed rather than served a stale index.
        self._frame_index_cache: Dict[Tuple[str, int, int, int], List[int]] = {}

    def close(self) -> None:
        """Close every idle pooled container now; a container a live
        resolve_frame call or decode_video_span iterator is using closes
        itself once that use ends, never out from under the caller. After
        this, every later acquire (any resolve_frame or decode_video_span
        call) raises MediaResolverError -- the resolver does not reopen
        itself. For test and shutdown cleanup.
        """
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
        selects a single frame for unambiguously, and for a frame-ordinal
        address, which decision 8 already selects a single frame for.

        A #f=N address always builds or reuses the file's frame-time
        index (see the module docstring) to answer -- unlike every other
        address shape, this one address's cost is not bounded by the cost
        rule, because decision 8 requires real per-file frame timings.
        """
        if purpose not in _PURPOSES:
            raise ValueError(f"purpose must be one of {_PURPOSES}, got {purpose!r}")

        addr = address if isinstance(address, MediaAddress) else parse_address(address)

        if not _is_absolute(addr.path):
            raise MediaAddressError(
                f"resolve_frame requires an absolute address, got a relative "
                f"path: {addr.path!r}"
            )

        if _is_image_path(addr.path):
            return self._resolve_still_image(addr, purpose)
        return self._resolve_video_frame(addr, purpose, policy)

    def decode_video_span(
        self, address: Union[str, MediaAddress], purpose: str,
        *, with_ordinals: bool = False,
    ) -> Iterator[FramePayload]:
        """Decode every frame of a range or bare-path video address, in
        presentation order, as an iterator of FramePayload (decision 3's
        half-open membership; a bare path is the whole stream, decision 4).

        `with_ordinals` (default False) decides whether every yielded
        FramePayload's frame_ordinal is real or None:

          * False (the default, and the only behaviour before this
            argument existed): the cost rule applies exactly as it does
            to resolve_frame's own #t=/bare cases -- this call does NOT
            BUILD the per-file frame-time index, but it DOES reuse one
            if some earlier #f= resolve (or an earlier
            with_ordinals=True call) already built and cached it for
            this file. frame_ordinal is None whenever no cached index
            exists yet, real otherwise.
          * True: the index is built (or reused, if it already exists)
            BEFORE any frame is yielded, so every yielded frame_ordinal
            is that frame's real, file-wide position -- never None. This
            costs one full demux pass over the file (no decode) -- the
            SAME cost a #f= address already pays, per the module
            docstring's "Frame-time index" -- and that cost is normally
            paid once per (path, stream, file size, mtime): the index is
            cached, so a LATER with_ordinals=True span, or a #f=
            resolve, on the same file within this resolver's lifetime
            reuses it for free. Two calls racing to resolve the SAME
            file's first such request can each miss the cache and each
            pay the cost once, concurrently, before either result is
            cached -- see the module docstring's "Frame-time index" and
            `docs/known_defects.md`; the two results agree, so this
            wastes work rather than producing a wrong answer. A caller
            must not substitute counting the frames it happens to yield
            for this: a span that starts
            partway through a file (a #t= range) yields frames whose
            first index is NOT 0, so an enumeration counter is only ever
            correct by coincidence, for a bare path starting at time 0.

        Raises immediately -- before any file is touched -- for a
        malformed purpose or address, a relative address, a #t= point or
        #f= address (only a range or bare path is valid here), or an
        image path. Raises MediaAddressError, lazily on the first
        `next()`, if the range contains no frame at all (mirroring
        resolve_frame's empty-range refusal -- decision 11 -- which this
        function cannot check any earlier, since knowing the range is
        empty requires reading the file).

        The pool handle this needs is acquired lazily, on the first
        `next()`, and released when the iterator is exhausted, when its
        `close()` method is called, or when it is abandoned and garbage
        collected -- ordinary Python generator semantics, since the
        implementation below is one.
        """
        if purpose not in _PURPOSES:
            raise ValueError(f"purpose must be one of {_PURPOSES}, got {purpose!r}")

        addr = address if isinstance(address, MediaAddress) else parse_address(address)

        if not _is_absolute(addr.path):
            raise MediaAddressError(
                f"decode_video_span requires an absolute address, got a "
                f"relative path: {addr.path!r}"
            )
        if addr.frame is not None or addr.time_us is not None:
            raise MediaAddressError(
                "decode_video_span resolves a range or a bare path, not a "
                "single frame ordinal or time point -- use resolve_frame "
                "for those"
            )
        if _is_image_path(addr.path):
            raise MediaAddressError(
                f"decode_video_span resolves a video stream; {addr.path!r} "
                f"looks like a still image"
            )

        return self._decode_video_span_frames(addr, purpose, with_ordinals)

    def _decode_video_span_frames(
        self, addr: MediaAddress, purpose: str, with_ordinals: bool
    ) -> Iterator[FramePayload]:
        # Nothing above this line runs until the first next() -- this is a
        # generator function, so the pool acquire below is the lazy
        # acquisition decode_video_span's docstring promises, and the
        # `finally` below runs on exhaustion, on an explicit close(), and
        # on garbage collection of an abandoned iterator (all three drive
        # a generator's own close() under the hood).
        container = self._pool.acquire(addr.path)
        try:
            stream = self._select_video_stream(container, addr)
            epoch_us = self._get_epoch_us(container, stream)
            if with_ordinals:
                # Builds the index if it does not already exist -- the
                # one extra demux pass with_ordinals=True's docstring
                # promises, paid at most once per (path, stream, size,
                # mtime).
                frame_times = self._get_frame_index(container, stream, epoch_us)
            else:
                # Never builds the index (rule 3, extended to spans) --
                # only reused if an earlier #f= resolve, or an earlier
                # with_ordinals=True call, already built it.
                frame_times = self._peek_frame_index(container, stream)

            is_bare = addr.time_range_us is None
            if is_bare:
                start_us, end_us = 0, None
            else:
                start_us, end_us = addr.time_range_us

            raw_start_ticks = _us_to_ticks(start_us + epoch_us, stream.time_base)
            container.seek(raw_start_ticks, backward=True, any_frame=False, stream=stream)

            yielded_any = False
            for frame in container.decode(stream):
                pts_us = _ticks_to_us(frame.pts, stream.time_base) - epoch_us
                if pts_us < start_us:
                    continue
                if end_us is not None and pts_us >= end_us:
                    break

                pixels, width, height = _decoded_frame_to_pixels(frame, addr.region)

                ordinal = None
                if frame_times is not None:
                    index = bisect.bisect_left(frame_times, pts_us)
                    if index < len(frame_times) and frame_times[index] == pts_us:
                        ordinal = index

                yielded_any = True
                yield FramePayload(
                    pixels=pixels,
                    frame_ordinal=ordinal,
                    presentation_time_us=pts_us,
                    width=width,
                    height=height,
                    purpose=purpose,
                )

            if not yielded_any:
                raise MediaAddressError(
                    f"the range {start_us} to {end_us} microseconds "
                    f"contains no frames in {addr.path!r}"
                )
        finally:
            self._pool.release(container)

    # -- still images ------------------------------------------------------

    def _resolve_still_image(self, addr: MediaAddress, purpose: str) -> FramePayload:
        # A still image has exactly one frame, at ordinal 0 (decision 8's
        # FramePayload docstring). #f=0 asks for that frame; any other
        # ordinal is beyond the last frame (decision 11).
        if addr.frame is not None and addr.frame != 0:
            raise MediaAddressError(
                f"frame ordinal {addr.frame} is beyond the last frame "
                f"(a still image has exactly one frame, ordinal 0): "
                f"{addr.path!r}"
            )
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

            if addr.frame is not None:
                # #f=N always builds or reuses the frame-time index to
                # answer -- ordinal is addr.frame itself (decision 8:
                # never converted), bounds-checked by select_frame().
                frame, ordinal, pts_us = self._resolve_frame_ordinal(
                    container, stream, epoch_us, addr
                )
            elif addr.time_us is not None:
                # The second element is this resolve's own local index
                # into the frame-time window it decoded -- needed inside
                # the helper to pick `frame` out of that window, never a
                # file-wide position (decision 8), so it is not carried
                # any further than this unpacking.
                frame, _local_index, pts_us = self._resolve_point(
                    container, stream, epoch_us, addr
                )
                ordinal = self._ordinal_for_time(container, stream, pts_us)
            else:
                # A range, or a bare path (decision 4: the range covering
                # the whole file).
                frame, _local_index, pts_us = self._resolve_range_or_bare(
                    container, stream, epoch_us, addr, policy
                )
                ordinal = self._ordinal_for_time(container, stream, pts_us)

            pixels, width, height = _decoded_frame_to_pixels(frame, addr.region)

            return FramePayload(
                pixels=pixels,
                frame_ordinal=ordinal,
                presentation_time_us=pts_us,
                width=width,
                height=height,
                purpose=purpose,
            )
        finally:
            self._pool.release(container)

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
        cache_key = self._media_cache_key(container, stream)
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

    def _media_cache_key(self, container, stream) -> Tuple[str, int, int, int]:
        """The cache key both _epoch_cache and _frame_index_cache use:
        path, stream index, file size, and mtime. One shared function so
        the two caches are keyed identically by construction -- see the
        note on _epoch_cache in __init__ for why they must be.
        """
        path = self._container_path(container)
        stat = pathlib.Path(path).stat()
        return (path, stream.index, stat.st_size, stat.st_mtime_ns)

    # -- frame-time index (P1.2b, decision 8) -------------------------------

    def _peek_frame_index(self, container, stream) -> Optional[List[int]]:
        """The cached frame-time index for this (file, stream), or None if
        it has not been built yet (or the file changed since it was). Never
        builds it -- the cost rule forbids that for a #t= or bare resolve.
        """
        cache_key = self._media_cache_key(container, stream)
        with self._frame_index_lock:
            return self._frame_index_cache.get(cache_key)

    def _get_frame_index(self, container, stream, epoch_us: int) -> List[int]:
        """The frame-time index for this (file, stream), building it if
        this is the first #f= resolve to ask for it. Only reached from the
        #f= path -- see the module docstring's cost rule.
        """
        cache_key = self._media_cache_key(container, stream)
        with self._frame_index_lock:
            cached = self._frame_index_cache.get(cache_key)
        if cached is not None:
            return cached

        frame_times = self._build_frame_index(container, stream, epoch_us)

        with self._frame_index_lock:
            self._frame_index_cache[cache_key] = frame_times
        return frame_times

    def _build_frame_index(self, container, stream, epoch_us: int) -> List[int]:
        """Demux every packet of `stream` (no decode) to collect the
        presentation time of every frame the decoder will present, sorted
        ascending (demux order is decode order, not presentation order --
        see the module docstring). A packet with no pts (the flush packet
        at end of stream) or marked discard by an edit list is not a
        presented frame and is excluded (decision 12).
        """
        container.seek(0, backward=True, any_frame=False, stream=stream)

        raw_ticks: List[int] = []
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            if packet.is_discard:
                continue
            raw_ticks.append(packet.pts)

        path = self._container_path(container)
        if not raw_ticks:
            raise MediaResolverError(
                f"{path!r} stream {stream.index} has no presented frames"
            )
        raw_ticks.sort()

        for previous, current in zip(raw_ticks, raw_ticks[1:]):
            if previous == current:
                # Two presented frames sharing one presentation time make
                # a frame ordinal ambiguous -- bisect-based lookup could
                # return either one. Refuse rather than guess.
                raise MediaResolverError(
                    f"{path!r} stream {stream.index} has two presented "
                    f"frames sharing presentation time {current} ticks -- "
                    f"refusing to guess which one a frame ordinal means"
                )

        raw_us = [_ticks_to_us(ticks, stream.time_base) for ticks in raw_ticks]

        for previous, current in zip(raw_us, raw_us[1:]):
            if previous == current:
                # raw_ticks are strictly increasing (the check above already
                # refused an exact tie there) but rounding to microseconds
                # can still collapse two DISTINCT raw ticks onto the same
                # microsecond value -- a fine enough time_base (a high
                # sample rate) makes this a real possibility, not merely a
                # tick-level tie under a coarser clock. The same ambiguity
                # the raw-tick check exists to prevent, one rounding step
                # later: refuse rather than guess which frame an ordinal or
                # a #f= address means.
                raise MediaResolverError(
                    f"{path!r} stream {stream.index} has two presented "
                    f"frames whose presentation times round to the same "
                    f"microsecond ({current}) -- the file's timestamps "
                    f"collide at microsecond resolution, so refusing to "
                    f"guess which one a frame ordinal means"
                )

        if raw_us[0] != epoch_us:
            raise MediaResolverError(
                f"the frame-time index's first entry ({raw_us[0]} "
                f"microseconds) does not match the cached epoch "
                f"({epoch_us} microseconds) for {path!r} stream "
                f"{stream.index} -- refusing to choose between them"
            )
        return [us - epoch_us for us in raw_us]

    def _ordinal_for_time(self, container, stream, pts_us: int) -> Optional[int]:
        """The frame ordinal for a presentation time already known to be a
        real frame's time, if this file's index already exists; otherwise
        None (rule 3 -- a #t= or bare resolve never builds the index).
        """
        frame_times = self._peek_frame_index(container, stream)
        if frame_times is None:
            return None
        index = bisect.bisect_left(frame_times, pts_us)
        if index < len(frame_times) and frame_times[index] == pts_us:
            return index
        return None

    def _resolve_frame_ordinal(
        self, container, stream, epoch_us: int, addr: MediaAddress
    ):
        frame_times = self._get_frame_index(container, stream, epoch_us)
        # Bounds-checked, unconverted (decision 8): the ordinal this
        # returns is exactly addr.frame.
        ordinal = select_frame(addr, frame_times)
        target_us = frame_times[ordinal]

        raw_target_ticks = _us_to_ticks(target_us + epoch_us, stream.time_base)
        container.seek(raw_target_ticks, backward=True, any_frame=False, stream=stream)

        for frame in container.decode(stream):
            pts_us = _ticks_to_us(frame.pts, stream.time_base) - epoch_us
            if pts_us == target_us:
                return frame, ordinal, pts_us
            if pts_us > target_us:
                break

        raise MediaResolverError(
            f"frame ordinal {addr.frame} (indexed at {target_us} "
            f"microseconds) was not found decoding {addr.path!r} -- the "
            f"frame-time index may be stale"
        )

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
