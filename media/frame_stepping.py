"""
media/frame_stepping.py

Layer A behind the detail view's frame-by-frame stepper: which cached
frame is nearest a paused player position, what player position a
frame's own presentation time names, the stepper label text, and the
stepper slider's page step. No Qt, no decoding, no file access.

Every conversion between a frame's presentation_time_us (the file
clock's microseconds) and a QMediaPlayer position in milliseconds goes
through media/playback.py's address_us_to_player_ms -- the one home of
that time-origin conversion (CLAUDE.md). This module never re-derives
it.
"""

from __future__ import annotations

from typing import Sequence

from media.playback import address_us_to_player_ms


def player_position_ms_for_frame(presentation_time_us: int) -> int:
    """The QMediaPlayer position, in milliseconds, that names the same
    instant as a frame's presentation_time_us.

    A thin, differently-named call onto media/playback.py's
    address_us_to_player_ms -- kept as its own function so a caller
    naming "the player position for this frame" reads that way, without
    a second place deciding what the conversion actually is.
    """
    return address_us_to_player_ms(presentation_time_us)


def nearest_frame_index(
    frame_times_us: Sequence[int], player_position_ms: int
) -> int:
    """The index into frame_times_us whose player-clock position is
    nearest to player_position_ms.

    A tie between two frames equally close goes to the earlier one. A
    position before the first frame's time, or after the last frame's
    time, clamps to that end's index -- both fall out of the same
    nearest-neighbour scan, not a separate clamp step.

    Raises ValueError if frame_times_us is empty: there is then no frame
    to name.
    """
    if not frame_times_us:
        raise ValueError("frame_times_us must not be empty")

    best_index = 0
    best_distance: int | None = None
    for index, time_us in enumerate(frame_times_us):
        position_ms = player_position_ms_for_frame(time_us)
        distance = abs(position_ms - player_position_ms)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def frame_label_text(
    frame_ordinal: int,
    presentation_time_us: int,
    index: int,
    total_frames: int,
) -> str:
    """The stepper's label text for the frame currently shown:
    'Frame <frame_ordinal>  <time in seconds, 3 decimals>  (<k+1> of <n>)'.
    """
    seconds = presentation_time_us / 1_000_000
    return f"Frame {frame_ordinal}  {seconds:.3f}s  ({index + 1} of {total_frames})"


def can_step_back(index: int) -> bool:
    """True if there is an earlier frame than `index` to step to -- the
    stepper's "Previous frame" button is enabled exactly when this is
    True."""
    return index > 0


def can_step_forward(index: int, total_frames: int) -> bool:
    """True if there is a later frame than `index`, out of `total_frames`
    frames, to step to -- the stepper's "Next frame" button is enabled
    exactly when this is True."""
    return index < total_frames - 1


def frame_slider_page_step(total_frames: int) -> int:
    """How far Page Up/Down should move the stepper's frame slider: a
    tenth of the clip's frame count, never zero -- a zero step would
    leave Page Up/Down with no effect at all, the same reasoning
    media/playback.py's slider_page_step_ms applies to the player's own
    slider.
    """
    return max(1, total_frames // 10)
