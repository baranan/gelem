"""
tests/test_clip_frame_cache.py

P1.4b part 1 -- ArtifactStore's in-memory frame-by-frame clip cache.

A clip (a #t= range address or a bare video path) is decoded entirely,
frame by frame, through MediaResolver.decode_video_span(with_ordinals=True)
on a worker thread, resized to the store's preview resolution, and held in
memory only -- never written to disk. The cache holds at most two clips at
once (the current one and one more); a clip whose length exceeds the
frame_stepper_max_seconds setting is refused.

Written from the work-item spec, not from the implementation. Uses the
real, short recordings in vids/ at the project root (vid11.mp4, vid22.mp4,
vid33.mp4 -- none of them longer than a few seconds), the same fixture
directory tests/test_project_load.py's literal media-cell values name and
docs/known_defects.md's P1.4a note used directly against a real file. That
directory is gitignored, so this whole module skips cleanly when it is not
present on this machine, rather than failing.

Does not play media or construct a widget.

Run with: python -m pytest tests/test_clip_frame_cache.py
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
import threading
import time

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from PIL import Image

from artifacts.artifact_store import ArtifactStore
from media.media_address import from_path, format as format_address, parse as parse_address
from media.resolver import MediaResolver

VIDS_DIR = project_root / "vids"
VIDEO_PATHS = sorted(VIDS_DIR.glob("*.mp4")) if VIDS_DIR.is_dir() else []

pytestmark = pytest.mark.skipif(
    not VIDEO_PATHS, reason="vids/ fixture videos are not present on this machine"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare_address(path: pathlib.Path) -> str:
    return format_address(from_path(str(path)))


def _range_address(path: pathlib.Path, start_us: int, end_us: int) -> str:
    addr = dataclasses.replace(from_path(str(path)), time_range_us=(start_us, end_us))
    return format_address(addr)


def _wait_for_clip(store: ArtifactStore) -> threading.Event:
    """Attach a chained ready-callback that sets a fresh Event, the same
    pattern tests/test_artifact_identity.py's _wait_for_thumbnail uses for
    on_thumbnail_ready. Call .wait(timeout) then .clear() before the next
    request."""
    event = threading.Event()
    previous = store.on_clip_frames_ready

    def _on_ready(canonical_address):
        if previous is not None:
            previous(canonical_address)
        event.set()

    store.on_clip_frames_ready = _on_ready
    return event


def _spin_until(predicate, timeout: float = 5.0) -> None:
    """Poll `predicate` until it is truthy or the timeout expires. Mirrors
    tests/test_request_queue.py's helper of the same name and purpose."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met within the timeout")


# ===========================================================================
# Frame ordinals match MediaResolver.get_frame_times.
# ===========================================================================

def test_bare_path_clip_ordinals_match_frame_times(tmp_path):
    resolver = MediaResolver(max_open_decoders=4)
    store = ArtifactStore(tmp_path / "artifacts", resolver=resolver)
    try:
        address = _bare_address(VIDEO_PATHS[0])
        frame_times = resolver.get_frame_times(address)

        ready = _wait_for_clip(store)
        store.request_clip_frames(address)
        assert ready.wait(timeout=30), "clip frames never finished decoding"

        frames = store.clip_frames(address)
        assert frames is not None
        assert len(frames) == len(frame_times)
        assert [frame.frame_ordinal for frame in frames] == list(range(len(frame_times)))
    finally:
        resolver.close()


def test_time_range_clip_ordinals_are_a_contiguous_subset(tmp_path):
    resolver = MediaResolver(max_open_decoders=4)
    store = ArtifactStore(tmp_path / "artifacts", resolver=resolver)
    try:
        video_path = VIDEO_PATHS[0]
        frame_times = resolver.get_frame_times(_bare_address(video_path))

        # A sub-range comfortably inside the file's own duration.
        start_us, end_us = 200_000, 1_500_000
        range_address = _range_address(video_path, start_us, end_us)

        ready = _wait_for_clip(store)
        store.request_clip_frames(range_address)
        assert ready.wait(timeout=30), "clip frames never finished decoding"

        frames = store.clip_frames(range_address)
        assert frames is not None
        expected_ordinals = [
            index for index, t in enumerate(frame_times) if start_us <= t < end_us
        ]
        actual_ordinals = [frame.frame_ordinal for frame in frames]
        assert actual_ordinals == expected_ordinals
        # Contiguous: a run of consecutive integers, no gap.
        assert actual_ordinals == list(
            range(actual_ordinals[0], actual_ordinals[0] + len(actual_ordinals))
        )
    finally:
        resolver.close()


# ===========================================================================
# Resizing and the disk boundary.
# ===========================================================================

def test_cached_frames_are_resized_to_the_preview_side(tmp_path):
    resolver = MediaResolver(max_open_decoders=4)
    # A small preview side, well under the fixture videos' real resolution,
    # so the resize is meaningfully exercised rather than a no-op.
    store = ArtifactStore(
        tmp_path / "artifacts", resolver=resolver, preview_max_side=64,
    )
    try:
        address = _bare_address(VIDEO_PATHS[0])
        ready = _wait_for_clip(store)
        store.request_clip_frames(address)
        assert ready.wait(timeout=30), "clip frames never finished decoding"

        frames = store.clip_frames(address)
        assert frames
        for frame in frames:
            height, width = frame.pixels.shape[:2]
            assert max(height, width) <= 64
    finally:
        resolver.close()


def test_caching_a_clip_writes_no_files_to_the_artifacts_directory(tmp_path):
    resolver = MediaResolver(max_open_decoders=4)
    artifacts_dir = tmp_path / "artifacts"
    store = ArtifactStore(artifacts_dir, resolver=resolver)
    try:
        before = sorted(path.name for path in artifacts_dir.iterdir())

        address = _bare_address(VIDEO_PATHS[0])
        ready = _wait_for_clip(store)
        store.request_clip_frames(address)
        assert ready.wait(timeout=30), "clip frames never finished decoding"

        after = sorted(path.name for path in artifacts_dir.iterdir())
        assert after == before, "caching a clip changed the artifacts directory"
    finally:
        resolver.close()


# ===========================================================================
# The two-clip LRU.
# ===========================================================================

def test_third_clip_request_evicts_the_least_recently_requested(tmp_path):
    if len(VIDEO_PATHS) < 3:
        pytest.skip("need at least three vids/ fixture videos")

    resolver = MediaResolver(max_open_decoders=4)
    store = ArtifactStore(tmp_path / "artifacts", resolver=resolver)
    try:
        address_a = _bare_address(VIDEO_PATHS[0])
        address_b = _bare_address(VIDEO_PATHS[1])
        address_c = _bare_address(VIDEO_PATHS[2])

        ready = _wait_for_clip(store)
        store.request_clip_frames(address_a)
        assert ready.wait(timeout=30), "clip A never finished decoding"
        ready.clear()

        store.request_clip_frames(address_b)
        assert ready.wait(timeout=30), "clip B never finished decoding"
        ready.clear()

        # A third distinct clip evicts A, the least recently requested.
        store.request_clip_frames(address_c)
        assert ready.wait(timeout=30), "clip C never finished decoding"

        assert store.clip_frames(address_a) is None, "A survived a third request"
        assert store.clip_frames(address_b) is not None
        assert store.clip_frames(address_c) is not None
    finally:
        resolver.close()


# ===========================================================================
# The length ceiling.
# ===========================================================================

def test_clip_over_the_length_limit_is_refused(tmp_path):
    resolver = MediaResolver(max_open_decoders=4)
    # Every vids/ fixture is a few seconds long -- 1 second is comfortably
    # under all of them.
    store = ArtifactStore(
        tmp_path / "artifacts", resolver=resolver, frame_stepper_max_seconds=1,
    )
    try:
        address = _bare_address(VIDEO_PATHS[0])
        duration_us = resolver.get_duration_us(address)
        assert not store.clip_fits_frame_cache(duration_us)

        with pytest.raises(ValueError):
            store.request_clip_frames(address)

        assert store.clip_frames(address) is None
    finally:
        resolver.close()


# ===========================================================================
# get_duration_us: accuracy and the cost rule.
# ===========================================================================

def test_get_duration_us_matches_last_frame_time_and_builds_no_index(tmp_path):
    resolver = MediaResolver(max_open_decoders=4)
    try:
        video_path = VIDEO_PATHS[0]
        address = _bare_address(video_path)
        addr = parse_address(address)

        duration_us = resolver.get_duration_us(address)
        assert duration_us is not None

        # Cost rule: get_duration_us must not build the per-file frame-time
        # index -- checked the same way the resolver's own cost-rule tests
        # do, via the private _peek_frame_index.
        container = resolver._pool.acquire(addr.path)
        try:
            stream = resolver._select_video_stream(container, addr)
            assert resolver._peek_frame_index(container, stream) is None, (
                "get_duration_us built the frame-time index"
            )
        finally:
            resolver._pool.release(container)

        frame_times = resolver.get_frame_times(address)
        # The fixture is not a fixed frame rate (its inter-frame gaps
        # vary), so "one frame duration" is the largest gap actually
        # observed between consecutive presented frames, not a nominal
        # rate -- a real per-file measurement, exactly as decision 8
        # requires for a #f= resolve.
        frame_interval_us = max(
            later - earlier for earlier, later in zip(frame_times, frame_times[1:])
        )
        assert abs(duration_us - frame_times[-1]) <= frame_interval_us
    finally:
        resolver.close()


# ===========================================================================
# reset() cancellation.
# ===========================================================================

def test_reset_drops_the_cache_and_cancels_an_in_flight_decode(tmp_path):
    video_path = VIDEO_PATHS[0]
    address = _bare_address(video_path)

    in_decode = threading.Event()
    proceed = threading.Event()

    class Gated(ArtifactStore):
        def _decode_clip_span(self, address):
            in_decode.set()
            assert proceed.wait(timeout=10)
            return super()._decode_clip_span(address)

    resolver = MediaResolver(max_open_decoders=4)
    store = Gated(tmp_path / "artifacts", worker_count=1, resolver=resolver)
    try:
        notified: list[str] = []
        store.on_clip_frames_ready = lambda canonical_address: notified.append(
            canonical_address
        )

        store.request_clip_frames(address)
        assert in_decode.wait(timeout=5), "the clip decode never started"

        store.reset()   # cancel the job while it is mid-decode
        proceed.set()   # let the gated call return and the decode continue

        time.sleep(0.3)

        assert store.clip_frames(address) is None
        assert notified == [], "a dropped clip decode sent a notification"
    finally:
        resolver.close()


# ===========================================================================
# An exception outside the caught tuple still frees the slot.
# ===========================================================================

def test_an_uncaught_decode_error_still_frees_the_slot_for_a_retry(tmp_path):
    """RuntimeError stands in for a bug _run_clip_job's except tuple
    (MediaResolverError, MediaAddressError, ValueError) does not name.
    WorkerPool's own backstop (`except Exception` in _worker_loop) is
    what actually stops it taking the worker thread down -- but
    ArtifactStore's own cleanup must already have freed the address's
    slot by the time that happens, so a second request_clip_frames for
    the same address is not silently treated as already-in-flight; it
    queues a fresh, working job."""
    resolver = MediaResolver(max_open_decoders=4)
    address = _bare_address(VIDEO_PATHS[0])
    attempts = {"count": 0}

    class Flaky(ArtifactStore):
        def _decode_clip_span(self, address):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("boom")
            return super()._decode_clip_span(address)

    store = Flaky(tmp_path / "artifacts", resolver=resolver)
    try:
        store.request_clip_frames(address)
        _spin_until(lambda: attempts["count"] >= 1, timeout=5)
        # Wait for the worker's own cleanup to actually run -- its
        # finally frees the slot before WorkerPool's backstop even logs
        # the exception.
        _spin_until(lambda: address not in store._clip_slots, timeout=5)
        assert store.clip_frames(address) is None

        ready = _wait_for_clip(store)
        store.request_clip_frames(address)
        assert ready.wait(timeout=30), "the retried clip decode never finished"

        assert store.clip_frames(address) is not None
        assert attempts["count"] == 2
    finally:
        resolver.close()


# ===========================================================================
# clip_is_steppable -- the ONE home for stepper eligibility (P1.4b part 2).
# ===========================================================================

def test_clip_is_steppable_true_for_a_short_vids_file(tmp_path):
    resolver = MediaResolver(max_open_decoders=4)
    store = ArtifactStore(tmp_path / "artifacts", resolver=resolver)
    try:
        address = _bare_address(VIDEO_PATHS[0])
        assert store.clip_is_steppable(address)
    finally:
        resolver.close()


def test_clip_is_steppable_false_over_the_length_limit(tmp_path):
    resolver = MediaResolver(max_open_decoders=4)
    # Every vids/ fixture is a few seconds long -- 1 second is comfortably
    # under all of them, the same ceiling test_clip_over_the_length_limit_
    # is_refused above uses.
    store = ArtifactStore(
        tmp_path / "artifacts", resolver=resolver, frame_stepper_max_seconds=1,
    )
    try:
        address = _bare_address(VIDEO_PATHS[0])
        assert not store.clip_is_steppable(address)
    finally:
        resolver.close()


def test_clip_is_steppable_false_for_a_still_image(tmp_path):
    resolver = MediaResolver(max_open_decoders=4)
    store = ArtifactStore(tmp_path / "artifacts", resolver=resolver)
    try:
        image = Image.new("RGB", (10, 10), color=(255, 0, 0))
        jpeg_path = tmp_path / "solid.jpg"
        image.save(jpeg_path, format="JPEG", quality=100)
        address = _bare_address(jpeg_path)
        assert not store.clip_is_steppable(address)
    finally:
        resolver.close()


def test_clip_is_steppable_false_for_a_relative_path(tmp_path):
    resolver = MediaResolver(max_open_decoders=4)
    store = ArtifactStore(tmp_path / "artifacts", resolver=resolver)
    try:
        assert not store.clip_is_steppable("relative/clip.mp4")
    finally:
        resolver.close()
