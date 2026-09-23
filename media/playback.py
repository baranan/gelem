"""
media/playback.py

P1.10: the pure decisions behind range playback in the detail view -- which
milliseconds a video player should start at, pause at, and restart from for
a given media address. This module is Layer A: no Qt, no decoding, no file
access. column_types/playback_adapter.py is Layer B, the only place that
calls Qt with these results.

A #t=A-B RANGE address plays only that span. A bare path, or an address
with no time selector, plays the whole file (decision 4 in
docs/media_architecture.md section 3.6: a bare path means the range
covering the whole file). A single-frame address (#f=N, or a #t= time
POINT) has no playback span at all -- callers show a still instead
(CLAUDE.md) -- so playback_span() refuses one with ValueError rather than
inventing a zero-length span.

This module never opens a file, never reads a frame time, and never checks
whether a range fits inside the file's actual length: a range whose end
overruns the file is passed through unchanged, exactly as
docs/media_architecture.md section 3.6 decision 11 says the resolver itself
never shortens anything. It never imports pandas, numpy, PyAV, cv2, PIL or
a Qt binding.
"""

from __future__ import annotations

from dataclasses import dataclass

from media import media_address


@dataclass(frozen=True)
class PlaybackSpan:
    """What a player should play: an absolute source path, a start
    position, and an end position -- or None for "play to the end of the
    file".
    """

    source_path: str
    start_ms: int
    end_ms: int | None


def address_us_to_player_ms(us: int) -> int:
    """Convert an address time, expressed in the file clock's microseconds
    (decision 10, docs/media_architecture.md section 3.6: the clock's zero
    is the first presented frame of the primary video stream), to the
    millisecond position QMediaPlayer.setPosition() expects.

    Today this is a plain unit conversion. Whether QMediaPlayer's own
    position-zero actually coincides with decision 10's file-clock zero for
    every container has NOT been verified -- see docs/known_defects.md.
    """
    return us // 1000


def playback_span(canonical_address: str) -> PlaybackSpan:
    """The PlaybackSpan a player should play for `canonical_address`.

    A bare path (no time selector) plays the whole file: start_ms=0,
    end_ms=None. A #t=A-B range plays only that span. A single-frame
    address (#f=N, or a #t= time POINT) raises ValueError -- it names one
    frame, not a span, so a caller shows a still instead of building a
    player at all.
    """
    addr = media_address.parse(canonical_address)

    if addr.selects_single_frame:
        raise ValueError(
            f"a single-frame address has no playback span, only a still: "
            f"{canonical_address!r}"
        )

    if addr.time_range_us is not None:
        start_us, end_us = addr.time_range_us
        start_ms = address_us_to_player_ms(start_us)
        end_ms: int | None = address_us_to_player_ms(end_us)
    else:
        start_ms = 0
        end_ms = None

    return PlaybackSpan(source_path=addr.path, start_ms=start_ms, end_ms=end_ms)


def initial_position_ms(span: PlaybackSpan) -> int:
    """Where the player should seek to as soon as the media is loaded."""
    return span.start_ms


def should_pause(position_ms: int, span: PlaybackSpan) -> bool:
    """True once playback has reached or passed the span's end.

    A span with no end (a bare path playing to the end of the file) never
    pauses on its own.
    """
    return span.end_ms is not None and position_ms >= span.end_ms


def play_from_ms(
    position_ms: int, span: PlaybackSpan, *, at_end_of_media: bool = False
) -> int | None:
    """Where Play should seek to before resuming, or None to resume from
    wherever the player already is.

    At or past the span's end, Play restarts the span from its start
    rather than resuming past the end into whatever follows in the file. A
    span with no end never restarts this way from position alone.

    at_end_of_media is for a span whose end overruns the file: the player
    reaches the actual end of the file, at a position still short of
    end_ms, so position_ms alone cannot tell "at the end". When the
    caller reports the player itself is at end of media, this always
    restarts at the span's start -- 0 for a bare path, whose start already
    is the beginning of the file.
    """
    if at_end_of_media:
        return span.start_ms
    if span.end_ms is not None and position_ms >= span.end_ms:
        return span.start_ms
    return None
