"""
tests/test_playback_span.py

P1.10: written from the work item's spec, not from media/playback.py's
implementation.

media/playback.py is the pure "Layer A" behind range playback in the
detail view: which PlaybackSpan a media address names, and the three
decisions a player widget makes from one (initial position, whether to
pause, where Play restarts from). It never opens a file or reads a frame
time, so every case below is checked with address strings alone.

Also guards the file-boundary half of CLAUDE.md's "Native playback is the
explicit exception" rule with two static AST checks: QMediaPlayer is
referenced in exactly one non-test source file
(column_types/playback_adapter.py), and media/playback.py imports no Qt
binding.

Run with: python -m pytest tests/test_playback_span.py

# run-tests: combined -- names "PySide6" only inside a source-scan assertion, never imports it
"""

from __future__ import annotations

import ast
import pathlib
import sys

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

from media.playback import (
    PlaybackSpan,
    address_us_to_player_ms,
    initial_position_ms,
    play_from_ms,
    playback_span,
    position_for_slider_value,
    should_pause,
    slider_page_step_ms,
    slider_single_step_ms,
    slider_value_for_position,
    span_length_ms,
)


# ---------------------------------------------------------------------------
# playback_span() -- what kind of address produces what span.
# ---------------------------------------------------------------------------

def test_bare_path_plays_the_whole_file():
    span = playback_span("C:/videos/clip.mp4")
    assert span == PlaybackSpan(
        source_path="C:/videos/clip.mp4", start_ms=0, end_ms=None
    )


def test_range_address_gives_start_and_end_in_milliseconds():
    span = playback_span("C:/videos/clip.mp4#t=1.0-2.5")
    assert span == PlaybackSpan(
        source_path="C:/videos/clip.mp4", start_ms=1000, end_ms=2500
    )


def test_frame_ordinal_address_raises_value_error():
    with pytest.raises(ValueError):
        playback_span("C:/videos/clip.mp4#f=5")


def test_time_point_address_raises_value_error():
    with pytest.raises(ValueError):
        playback_span("C:/videos/clip.mp4#t=1.5")


def test_a_range_whose_end_overruns_the_file_is_passed_through_unchanged():
    # media/playback.py never reads the file, so it has no way to know
    # this range is too long -- and must not guess. It converts the
    # address exactly as given (docs/media_architecture.md section 3.6,
    # decision 11: "the resolver itself never shortens anything").
    span = playback_span("C:/videos/clip.mp4#t=1.0-99999.0")
    assert span.start_ms == 1000
    assert span.end_ms == 99_999_000


def test_address_us_to_player_ms_is_a_unit_conversion():
    assert address_us_to_player_ms(0) == 0
    assert address_us_to_player_ms(1_000_000) == 1000
    assert address_us_to_player_ms(2_500_000) == 2500


# ---------------------------------------------------------------------------
# initial_position_ms() -- where the player seeks to once media is loaded.
# ---------------------------------------------------------------------------

def test_initial_position_is_the_spans_start():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert initial_position_ms(span) == 1000

    bare_span = PlaybackSpan(source_path="x.mp4", start_ms=0, end_ms=None)
    assert initial_position_ms(bare_span) == 0


# ---------------------------------------------------------------------------
# should_pause() -- at, before, and after the span's end; never for a span
# with no end.
# ---------------------------------------------------------------------------

def test_should_pause_before_the_end_is_false():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert should_pause(2000, span) is False


def test_should_pause_at_the_end_is_true():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert should_pause(2500, span) is True


def test_should_pause_after_the_end_is_true():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert should_pause(3000, span) is True


def test_should_pause_never_true_when_end_ms_is_none():
    span = PlaybackSpan(source_path="x.mp4", start_ms=0, end_ms=None)
    assert should_pause(0, span) is False
    assert should_pause(10**9, span) is False


# ---------------------------------------------------------------------------
# play_from_ms() -- restart at the span's start at/after the end, else
# resume from wherever the player already is (None).
# ---------------------------------------------------------------------------

def test_play_from_before_the_end_continues():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert play_from_ms(2000, span) is None


def test_play_from_at_the_end_restarts_at_the_start():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert play_from_ms(2500, span) == 1000


def test_play_from_after_the_end_restarts_at_the_start():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert play_from_ms(3000, span) == 1000


def test_play_from_never_restarts_when_end_ms_is_none():
    span = PlaybackSpan(source_path="x.mp4", start_ms=0, end_ms=None)
    assert play_from_ms(0, span) is None
    assert play_from_ms(10**9, span) is None


# ---------------------------------------------------------------------------
# play_from_ms(at_end_of_media=...) -- an overrunning range's end is never
# reached by position alone, because the file itself ends first. The player
# reporting it has hit end of media must restart the span regardless of
# position.
# ---------------------------------------------------------------------------

def test_at_end_of_media_restarts_a_range_whose_end_was_never_reached():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=99_999_000)
    # position_ms is short of end_ms -- the file ended first.
    assert play_from_ms(5000, span, at_end_of_media=True) == 1000


def test_at_end_of_media_restarts_a_bare_path_at_zero():
    span = PlaybackSpan(source_path="x.mp4", start_ms=0, end_ms=None)
    assert play_from_ms(5000, span, at_end_of_media=True) == 0


def test_at_end_of_media_false_keeps_todays_results():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert play_from_ms(2000, span, at_end_of_media=False) is None
    assert play_from_ms(2500, span, at_end_of_media=False) == 1000
    assert play_from_ms(3000, span, at_end_of_media=False) == 1000

    bare_span = PlaybackSpan(source_path="x.mp4", start_ms=0, end_ms=None)
    assert play_from_ms(10**9, bare_span, at_end_of_media=False) is None


# ---------------------------------------------------------------------------
# span_length_ms() -- the slider's track length, in milliseconds.
# ---------------------------------------------------------------------------

def test_range_length_is_end_minus_start():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert span_length_ms(span, media_duration_ms=None) == 1500
    assert span_length_ms(span, media_duration_ms=10_000) == 1500


def test_bare_path_length_is_duration_minus_start_when_duration_known():
    span = PlaybackSpan(source_path="x.mp4", start_ms=0, end_ms=None)
    assert span_length_ms(span, media_duration_ms=5000) == 5000

    started_late = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=None)
    assert span_length_ms(started_late, media_duration_ms=5000) == 4000


def test_bare_path_length_is_none_when_duration_unknown():
    span = PlaybackSpan(source_path="x.mp4", start_ms=0, end_ms=None)
    assert span_length_ms(span, media_duration_ms=None) is None


def test_range_overrunning_the_file_is_clamped_to_the_real_duration():
    # The address names a range past the end of the file (media/playback.py
    # never checks this itself -- see the module docstring); once the
    # player reports the file's real duration, the slider's track must not
    # extend past it.
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=99_999_000)
    assert span_length_ms(span, media_duration_ms=5000) == 4000


def test_range_not_yet_known_to_overrun_uses_its_nominal_length():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=99_999_000)
    assert span_length_ms(span, media_duration_ms=None) == 99_998_000


def test_span_length_is_never_negative():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert span_length_ms(span, media_duration_ms=500) == 0


# ---------------------------------------------------------------------------
# slider_value_for_position() -- player position -> slider value, relative
# to the span's start and clamped to [0, length].
# ---------------------------------------------------------------------------

def test_slider_value_is_position_relative_to_span_start():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert slider_value_for_position(1000, span, media_duration_ms=None) == 0
    assert slider_value_for_position(1600, span, media_duration_ms=None) == 600
    assert slider_value_for_position(2500, span, media_duration_ms=None) == 1500


def test_slider_value_is_zero_when_length_is_unknown():
    span = PlaybackSpan(source_path="x.mp4", start_ms=0, end_ms=None)
    assert slider_value_for_position(5000, span, media_duration_ms=None) == 0


def test_slider_value_clamps_below_zero():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert slider_value_for_position(0, span, media_duration_ms=None) == 0


def test_slider_value_clamps_above_the_length():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert slider_value_for_position(999_999, span, media_duration_ms=None) == 1500


# ---------------------------------------------------------------------------
# position_for_slider_value() -- slider value -> player position, clamped
# to [span.start_ms, span.start_ms + length].
# ---------------------------------------------------------------------------

def test_position_for_slider_value_offsets_from_span_start():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert position_for_slider_value(0, span, media_duration_ms=None) == 1000
    assert position_for_slider_value(600, span, media_duration_ms=None) == 1600
    assert position_for_slider_value(1500, span, media_duration_ms=None) == 2500


def test_position_for_slider_value_clamps_below_zero():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert position_for_slider_value(-100, span, media_duration_ms=None) == 1000


def test_position_for_slider_value_clamps_above_the_length():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    assert position_for_slider_value(999_999, span, media_duration_ms=None) == 2500


def test_position_for_slider_value_with_unknown_length_clamps_to_span_start():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=None)
    assert position_for_slider_value(500, span, media_duration_ms=None) == 1000


def test_slider_value_and_position_round_trip():
    span = PlaybackSpan(source_path="x.mp4", start_ms=1000, end_ms=2500)
    for value in (0, 750, 1500):
        position_ms = position_for_slider_value(value, span, media_duration_ms=None)
        assert slider_value_for_position(position_ms, span, media_duration_ms=None) == value


# ---------------------------------------------------------------------------
# slider_page_step_ms() / slider_single_step_ms() -- how far a page step
# (a click on the bar, Page Up/Down) or a single step (an arrow key) moves
# the slider, as a fraction of the track's length, never zero.
# ---------------------------------------------------------------------------

def test_page_step_is_a_tenth_of_the_length():
    assert slider_page_step_ms(1000) == 100
    assert slider_page_step_ms(250) == 25


def test_page_step_is_never_zero_for_a_short_track():
    assert slider_page_step_ms(5) == 1
    assert slider_page_step_ms(1) == 1


def test_single_step_is_a_hundredth_of_the_length():
    assert slider_single_step_ms(1000) == 10
    assert slider_single_step_ms(2500) == 25


def test_single_step_is_never_zero_for_a_short_track():
    assert slider_single_step_ms(50) == 1
    assert slider_single_step_ms(1) == 1


def test_single_step_is_never_larger_than_the_page_step():
    for length_ms in (1, 5, 50, 100, 1000, 99_999):
        assert slider_single_step_ms(length_ms) <= slider_page_step_ms(length_ms)


# ---------------------------------------------------------------------------
# Static guards: QMediaPlayer is built in exactly one non-test file, and
# media/playback.py (Layer A) imports no Qt binding.
# ---------------------------------------------------------------------------

_EXCLUDED_DIR_PARTS = {"tests", "manual_testing", "docs", "__pycache__", ".git"}


def _source_files() -> list[pathlib.Path]:
    """Every non-test .py file under the repository root, matching
    tests/test_source_decode_guard.py's own walk.
    """
    files = []
    for path in project_root.rglob("*.py"):
        relative_parts = path.relative_to(project_root).parts
        if any(part in _EXCLUDED_DIR_PARTS for part in relative_parts):
            continue
        if path.name.startswith("test_"):
            continue
        files.append(path)
    return sorted(files)


def _references_qmediaplayer(path: pathlib.Path) -> bool:
    """True if this file's AST contains a Name or Attribute node spelled
    'QMediaPlayer', under any import alias.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "QMediaPlayer":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "QMediaPlayer":
            return True
    return False


def test_qmediaplayer_is_referenced_in_exactly_one_non_test_file():
    hits = [
        path.relative_to(project_root).as_posix()
        for path in _source_files()
        if _references_qmediaplayer(path)
    ]
    assert hits == ["column_types/playback_adapter.py"], (
        f"expected QMediaPlayer to be referenced only in "
        f"column_types/playback_adapter.py, found: {hits}"
    )


def test_playback_module_imports_no_qt_binding():
    playback_path = project_root / "media" / "playback.py"
    tree = ast.parse(playback_path.read_text(encoding="utf-8"), filename=str(playback_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("PySide6"), (
                    f"media/playback.py must not import a Qt binding, "
                    f"found: import {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                assert not node.module.startswith("PySide6"), (
                    f"media/playback.py must not import a Qt binding, "
                    f"found: from {node.module} import ..."
                )
