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


def span_length_ms(span: PlaybackSpan, media_duration_ms: int | None) -> int | None:
    """How long the slider's track is, in milliseconds.

    A range's nominal length is end_ms - start_ms, but a range can overrun
    the file (docs/media_architecture.md section 3.6, decision 11 -- the
    resolver never shortens one), so once the file's real duration is
    known and is shorter than the nominal end, the length is clamped to
    what the file actually has: media_duration_ms - start_ms.

    A bare path (end_ms is None) plays to the end of the file, so its
    length IS media_duration_ms - start_ms -- None until the duration is
    known, because there is nothing else to measure it against.

    Never negative: a span's start_ms is never past a known duration
    because initial_position_ms() seeks to start_ms before there is
    anything to overrun.
    """
    if span.end_ms is not None:
        end_ms = span.end_ms
        if media_duration_ms is not None and media_duration_ms < end_ms:
            end_ms = media_duration_ms
        return max(0, end_ms - span.start_ms)

    if media_duration_ms is None:
        return None
    return max(0, media_duration_ms - span.start_ms)


def slider_value_for_position(
    position_ms: int, span: PlaybackSpan, media_duration_ms: int | None
) -> int:
    """The slider value for a player position, relative to the span's
    start and clamped to [0, span length].

    0 whenever the length itself is unknown -- the slider stays disabled
    in that state (column_types/playback_adapter.py), so its value does
    not matter, but it must still be a valid in-range int.
    """
    length_ms = span_length_ms(span, media_duration_ms)
    if length_ms is None:
        return 0
    value = position_ms - span.start_ms
    return max(0, min(value, length_ms))


def position_for_slider_value(
    value: int, span: PlaybackSpan, media_duration_ms: int | None
) -> int:
    """The player position a slider value names, clamped to the span
    itself -- [span.start_ms, span.start_ms + span length].

    A length of None (not yet known) clamps the position to span.start_ms:
    with no track length, no drag is possible; this is the position the
    player already sits at right after the initial seek.
    """
    length_ms = span_length_ms(span, media_duration_ms)
    if length_ms is None:
        return span.start_ms
    clamped_value = max(0, min(value, length_ms))
    return span.start_ms + clamped_value


def slider_page_step_ms(length_ms: int) -> int:
    """How far a page step (a click on the slider's bar, or Page Up/Down)
    should move the slider: a tenth of the track's length, never zero --
    a zero step would leave a click on the bar with no effect at all.
    """
    return max(1, length_ms // 10)


def slider_single_step_ms(length_ms: int) -> int:
    """How far a single step (an arrow key) should move the slider: a
    hundredth of the track's length, never zero, for the same reason
    slider_page_step_ms() is never zero.
    """
    return max(1, length_ms // 100)
