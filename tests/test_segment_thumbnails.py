"""
tests/test_segment_thumbnails.py

P1.7a -- a segment's (#t= range) thumbnail comes from inside that
segment's own time range, represented by the frame nearest its middle
(docs/media_architecture.md section 4.1b). A bare video path keeps the
first frame.

Written from the work-item spec, not from the implementation. New file
on purpose: tests/test_dataset.py runs everything in it twice, so
nothing new goes there.

Uses the real, short recordings in vids/ at the project root (vid11.mp4,
vid22.mp4, vid33.mp4, vid44.mp4), the same fixture directory
tests/test_clip_frame_cache.py and tests/test_project_load.py already
use. That directory is gitignored, so this whole module skips cleanly
when it is not present on this machine, rather than failing.

Run with: python -m pytest tests/test_segment_thumbnails.py
"""

from __future__ import annotations

import bisect
import dataclasses
import sys
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

from artifacts.artifact_store import ArtifactStore
from media.media_address import from_path, format as format_address
from media.resolver import MediaResolver

VIDS_DIR = project_root / "vids"
VIDEO_PATHS = sorted(VIDS_DIR.glob("*.mp4")) if VIDS_DIR.is_dir() else []

pytestmark = pytest.mark.skipif(
    not VIDEO_PATHS, reason="vids/ fixture videos are not present on this machine"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare_address(path: Path) -> str:
    return format_address(from_path(str(path)))


def _range_address(path: Path, start_us: int, end_us: int) -> str:
    addr = dataclasses.replace(from_path(str(path)), time_range_us=(start_us, end_us))
    return format_address(addr)


def _nearest_member_ground_truth(frame_times, start_us: int, end_us: int) -> int:
    """The same rule select_frame()/decision 4 define, computed by hand
    against the file's real (demuxed, not decoded) frame-time index, so
    the resolver's answer is checked against an independent computation
    rather than against its own internals."""
    lo = bisect.bisect_left(frame_times, start_us)
    hi = bisect.bisect_left(frame_times, end_us)
    assert lo < hi, f"range {start_us}-{end_us} contains no frame in the fixture"
    midpoint_us = start_us + (end_us - start_us) // 2
    return min(frame_times[lo:hi], key=lambda t: (abs(t - midpoint_us), t))


def _wait_for_thumbnail(store: ArtifactStore) -> threading.Event:
    """Attach a chained ready-callback that sets a fresh Event -- the same
    pattern tests/test_artifact_identity.py's _wait_for_thumbnail and
    tests/test_clip_frame_cache.py's _wait_for_clip use. Call .wait(timeout)
    then .clear() before the next request."""
    event = threading.Event()
    previous = store.on_thumbnail_ready

    def _on_ready(table_name, row_id):
        if previous is not None:
            previous(table_name, row_id)
        event.set()

    store.on_thumbnail_ready = _on_ready
    return event


class _SpyResolver:
    """Wraps a real MediaResolver, recording the (address, policy) every
    resolve_frame() call receives. ArtifactStore's only resolve_frame call
    site is _decode_source (CLAUDE.md's media rule; guarded by
    tests/test_source_decode_guard.py), so this is the one seam that can
    observe which policy a real request actually decoded under, without
    re-deriving it by hand-parsing the address the way
    ArtifactStore._policy_for_address itself does -- that would just test
    the implementation against itself.
    """

    def __init__(self, real: MediaResolver):
        self._real = real
        self.calls: list[tuple[str, str]] = []

    def resolve_frame(self, address, purpose, *, policy="first"):
        self.calls.append((str(address), policy))
        return self._real.resolve_frame(address, purpose, policy=policy)

    def get_duration_us(self, address):
        return self._real.get_duration_us(address)

    def get_frame_times(self, address):
        return self._real.get_frame_times(address)

    def decode_video_span(self, *args, **kwargs):
        return self._real.decode_video_span(*args, **kwargs)

    def close(self):
        self._real.close()


# ===========================================================================
# 1. resolve_frame(range, "display", policy="midpoint") returns a frame
#    inside [start, end) nearest the range's own midpoint -- checked
#    against an independent ground truth computed from get_frame_times(),
#    for several ranges including one whose start is not on a keyframe
#    and one near the end of the video.
# ===========================================================================

def test_midpoint_range_returns_the_member_nearest_the_midpoint():
    resolver = MediaResolver(max_open_decoders=4)
    for video_path in VIDEO_PATHS:
        duration_us = resolver.get_duration_us(str(video_path))
        frame_times = resolver.get_frame_times(str(video_path))
        if duration_us < 2_000_000:
            continue  # too short to carve the cases below out of

        cases = [
            # An arbitrary, deliberately non-round start -- real keyframes
            # in these fixtures land on much rounder boundaries (whole or
            # half seconds), so this reliably falls mid-GOP.
            (317_000, min(1_317_000, duration_us - 1)),
            # Near the end of the video.
            (max(0, duration_us - 900_000), duration_us - 100_000),
        ]
        for start_us, end_us in cases:
            if start_us >= end_us:
                continue
            address = _range_address(video_path, start_us, end_us)
            payload = resolver.resolve_frame(address, "display", policy="midpoint")

            assert start_us <= payload.presentation_time_us < end_us, (
                f"{video_path.name} {start_us}-{end_us}: "
                f"{payload.presentation_time_us} is not inside the range"
            )
            expected = _nearest_member_ground_truth(frame_times, start_us, end_us)
            assert payload.presentation_time_us == expected, (
                f"{video_path.name} {start_us}-{end_us}: got "
                f"{payload.presentation_time_us}, nearest-to-midpoint "
                f"member is {expected}"
            )
    resolver.close()


# ===========================================================================
# 2. ArtifactStore asks the resolver for 'midpoint' on a range address and
#    'first' on a bare path, and is_cached() agrees with what was
#    generated -- no re-request once a thumbnail is ready.
# ===========================================================================

def test_store_decodes_range_as_midpoint_and_bare_path_as_first(tmp_path):
    video_path = VIDEO_PATHS[0]
    real_resolver = MediaResolver(max_open_decoders=4)
    spy = _SpyResolver(real_resolver)
    store = ArtifactStore(tmp_path / "artifacts", resolver=spy, worker_count=2)
    event = _wait_for_thumbnail(store)

    range_address = _range_address(video_path, 200_000, 1_200_000)
    bare_address = _bare_address(video_path)

    store.request_thumbnail("row_range", range_address, video_path, "segments")
    assert event.wait(timeout=30), "range thumbnail was never generated"
    event.clear()

    store.request_thumbnail("row_bare", bare_address, video_path, "segments")
    assert event.wait(timeout=30), "bare-path thumbnail was never generated"

    range_policies = {policy for addr, policy in spy.calls if addr == range_address}
    bare_policies = {policy for addr, policy in spy.calls if addr == bare_address}
    assert range_policies == {"midpoint"}, (
        f"range address was decoded under {range_policies}, not {{'midpoint'}}"
    )
    assert bare_policies == {"first"}, (
        f"bare path was decoded under {bare_policies}, not {{'first'}}"
    )

    # is_cached() must agree with what was just generated -- this is the
    # exact failure P1.7a's design note warns about: is_cached() and the
    # decode disagreeing would make a tile re-request forever.
    assert store.is_cached(range_address) is True
    assert store.is_cached(bare_address) is True

    # A second request for the same (already-cached) range address must
    # short-circuit synchronously and must NOT call the resolver again.
    calls_before = len(spy.calls)
    event2 = _wait_for_thumbnail(store)
    store.request_thumbnail("row_range_2", range_address, video_path, "segments")
    assert event2.wait(timeout=5), "second request for a cached range never settled"
    assert len(spy.calls) == calls_before, (
        "a cached range address was re-decoded instead of short-circuiting -- "
        "is_cached() and the decode policy disagree"
    )
    real_resolver.close()


# ===========================================================================
# Report: what happens for a range that contains no frame at all.
#
# resolve_frame(range, "display", policy="midpoint") raises
# MediaAddressError("the range ... contains no frames") -- decision 11,
# unchanged by this item. The seek-to-middle branch in
# media/resolver.py's _resolve_range_or_bare reuses the SAME
# select_frame() call the 'first' branch already ends on, so an empty
# range is refused exactly the same way regardless of which policy asked
# for it; no new empty-range behaviour was introduced. Inside
# ArtifactStore, that raise is caught by _run_job's existing except
# block: no index entry, no fingerprint-memo entry, no
# on_thumbnail_ready callback -- the tile stays a placeholder and
# re-requests on every repaint (docs/known_defects.md; not fixed here,
# per this item's own scope).
# ===========================================================================
