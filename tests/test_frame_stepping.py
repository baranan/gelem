"""
tests/test_frame_stepping.py

P1.4b part 2 -- media/frame_stepping.py, the Qt-free Layer A behind the
detail view's frame-by-frame stepper: which cached frame is nearest a
paused player position, what player position a frame's own presentation
time names, and the stepper slider's page step.

Written from the work-item spec, not from the implementation. Does not
play media, decode anything beyond MediaResolver.get_frame_times (no
pixel decode), or build a widget.

Run with: python -m pytest tests/test_frame_stepping.py
"""

from __future__ import annotations

import pathlib
import sys

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

from media.frame_stepping import (
    can_step_back,
    can_step_forward,
    frame_slider_page_step,
    nearest_frame_index,
    player_position_ms_for_frame,
)
from media.resolver import MediaResolver

VIDS_DIR = project_root / "vids"
VIDEO_PATHS = sorted(VIDS_DIR.glob("*.mp4")) if VIDS_DIR.is_dir() else []


# ===========================================================================
# nearest_frame_index
# ===========================================================================

def test_nearest_frame_index_picks_the_nearest():
    frame_times_us = [0, 1_000_000, 2_000_000]  # -> player ms: 0, 1000, 2000
    # 1800ms is 800ms from frame 1 (1000ms) and 200ms from frame 2
    # (2000ms) -- nearer to frame 2, even though frame 1 is the last one
    # not past it (a floor/round-down would wrongly pick frame 1).
    assert nearest_frame_index(frame_times_us, 1_800) == 2


def test_nearest_frame_index_ties_go_to_the_earlier():
    frame_times_us = [0, 2_000]  # -> player ms: 0, 2
    # Equidistant (1ms) from both -- the earlier frame wins.
    assert nearest_frame_index(frame_times_us, 1) == 0


def test_nearest_frame_index_clamps_before_the_first():
    frame_times_us = [1_000_000, 2_000_000, 3_000_000]
    assert nearest_frame_index(frame_times_us, -500) == 0


def test_nearest_frame_index_clamps_after_the_last():
    frame_times_us = [1_000_000, 2_000_000, 3_000_000]
    assert nearest_frame_index(frame_times_us, 10_000) == 2


def test_nearest_frame_index_raises_on_an_empty_sequence():
    with pytest.raises(ValueError):
        nearest_frame_index([], 0)


# ===========================================================================
# can_step_back / can_step_forward
# ===========================================================================

def test_can_step_back_false_on_the_first_frame():
    assert not can_step_back(0)


def test_can_step_back_true_after_the_first_frame():
    assert can_step_back(1)


def test_can_step_forward_false_on_the_last_frame():
    assert not can_step_forward(4, total_frames=5)


def test_can_step_forward_true_before_the_last_frame():
    assert can_step_forward(3, total_frames=5)


def test_can_step_neither_direction_for_a_single_frame_clip():
    assert not can_step_back(0)
    assert not can_step_forward(0, total_frames=1)


# ===========================================================================
# frame_slider_page_step
# ===========================================================================

def test_frame_slider_page_step_is_a_tenth_of_the_frame_count():
    assert frame_slider_page_step(100) == 10
    assert frame_slider_page_step(250) == 25


def test_frame_slider_page_step_is_never_zero_for_n_equal_one():
    assert frame_slider_page_step(1) == 1
    assert frame_slider_page_step(5) == 1


# ===========================================================================
# Round trip against a real file: frame -> player ms -> nearest index
# returns the same frame, for every frame of a real vids/ clip.
# ===========================================================================

@pytest.mark.skipif(
    not VIDEO_PATHS, reason="vids/ fixture videos are not present on this machine"
)
def test_round_trip_recovers_every_frame_of_a_real_clip():
    resolver = MediaResolver(max_open_decoders=4)
    try:
        address = str(VIDEO_PATHS[0]).replace("\\", "/")
        frame_times_us = resolver.get_frame_times(address)
        assert len(frame_times_us) > 1, "sanity: need more than one frame"

        for k, presentation_time_us in enumerate(frame_times_us):
            player_ms = player_position_ms_for_frame(presentation_time_us)
            recovered = nearest_frame_index(frame_times_us, player_ms)
            assert recovered == k, (
                f"frame {k} (t={presentation_time_us}us -> {player_ms}ms) "
                f"round-tripped to frame {recovered} instead"
            )
    finally:
        resolver.close()
