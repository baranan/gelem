"""
tests/test_playback_mute.py

Mute toggle on the detail-view video player
(column_types/playback_adapter.py). Written from the work-item spec, not
from the implementation.

Spec:
  1. clicking the mute button flips QAudioOutput.isMuted() and the
     button's label (and tooltip);
  2. a second PlaybackAdapter built after a toggle starts in the toggled
     state -- the choice is session-lifetime, class-level state, not
     per-instance;
  3. class-level state is reset around every test so test order cannot
     change another test's result.

No test plays media -- the source path never has to exist for these
assertions to hold.

Run with:
    python -m pytest tests/test_playback_mute.py

# run-tests: isolate -- constructs PlaybackAdapter, a QWidget, in every test
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication

from column_types.playback_adapter import PlaybackAdapter
from media.playback import PlaybackSpan


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    if QApplication.instance() is None:
        QApplication(sys.argv)


@pytest.fixture(autouse=True)
def _reset_mute_state():
    """The mute choice is deliberately class-level, session-lifetime
    state -- reset it before and after each test so test order cannot
    affect another test's result."""
    PlaybackAdapter._muted = False
    yield
    PlaybackAdapter._muted = False


def _make_adapter():
    span = PlaybackSpan(source_path="nonexistent.mp4", start_ms=0, end_ms=None)
    return PlaybackAdapter(span)


def test_starts_unmuted_with_sound_on_label():
    widget = _make_adapter()
    assert widget._audio.isMuted() is False
    assert widget._mute_btn.text() == "\U0001F50A Sound on"


def test_clicking_mute_flips_muted_state_and_label():
    widget = _make_adapter()
    widget._mute_btn.click()
    assert widget._audio.isMuted() is True
    assert widget._mute_btn.text() == "\U0001F507 Muted"


def test_clicking_mute_twice_returns_to_unmuted():
    widget = _make_adapter()
    widget._mute_btn.click()
    widget._mute_btn.click()
    assert widget._audio.isMuted() is False
    assert widget._mute_btn.text() == "\U0001F50A Sound on"


def test_second_adapter_built_after_a_toggle_starts_muted():
    first = _make_adapter()
    first._mute_btn.click()
    assert first._audio.isMuted() is True

    second = _make_adapter()
    assert second._audio.isMuted() is True
    assert second._mute_btn.text() == "\U0001F507 Muted"


def test_tooltip_changes_with_state():
    widget = _make_adapter()
    unmuted_tooltip = widget._mute_btn.toolTip()
    assert unmuted_tooltip

    widget._mute_btn.click()
    muted_tooltip = widget._mute_btn.toolTip()
    assert muted_tooltip
    assert muted_tooltip != unmuted_tooltip


def test_second_players_own_button_mutes_that_player_not_the_first():
    # Regression: clicking the second player's button used to flip the
    # class-level flag back to whatever it already was (both start
    # unmuted, so the first click's own effect on this player was
    # invisible), instead of muting that player.
    first = _make_adapter()
    second = _make_adapter()
    assert first._audio.isMuted() is False
    assert second._audio.isMuted() is False

    first._mute_btn.click()
    assert first._audio.isMuted() is True

    second._mute_btn.click()
    assert second._audio.isMuted() is True

    third = _make_adapter()
    assert third._audio.isMuted() is True


# ---------------------------------------------------------------------------
# One Play/Pause toggle button. Its label is derived from the player's
# playbackState. Driving a real QMediaPlayer into PlayingState needs real
# media, so the tests emit playbackStateChanged on the real player and, for
# click routing, swap in a stub player that reports a chosen state.
# ---------------------------------------------------------------------------

from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QPushButton


def _buttons_with(widget, fragment):
    return [
        b for b in widget.findChildren(QPushButton) if fragment in b.text()
    ]


class _StubPlayer:
    def __init__(self, state):
        self._state = state
        self.calls = []

    def playbackState(self):
        return self._state

    def pause(self):
        self.calls.append("pause")

    def play(self):
        self.calls.append("play")

    def position(self):
        return 0

    def mediaStatus(self):
        return QMediaPlayer.MediaStatus.NoMedia

    def setPosition(self, ms):
        self.calls.append(("setPosition", ms))


def test_exactly_one_play_pause_button_and_no_separate_pause_button():
    widget = _make_adapter()
    # Not playing at construction, so the one button reads Play.
    assert len(_buttons_with(widget, "Play")) == 1
    assert len(_buttons_with(widget, "Pause")) == 0
    assert widget._play_pause_btn.text() == "▶ Play"


def test_label_follows_playback_state_signal():
    widget = _make_adapter()
    player = widget._player
    player.playbackStateChanged.emit(QMediaPlayer.PlaybackState.PlayingState)
    assert "Pause" in widget._play_pause_btn.text()
    player.playbackStateChanged.emit(QMediaPlayer.PlaybackState.PausedState)
    assert "Play" in widget._play_pause_btn.text()
    player.playbackStateChanged.emit(QMediaPlayer.PlaybackState.PlayingState)
    player.playbackStateChanged.emit(QMediaPlayer.PlaybackState.StoppedState)
    assert "Play" in widget._play_pause_btn.text()


def test_state_slot_maps_each_state_to_a_label():
    widget = _make_adapter()
    states = QMediaPlayer.PlaybackState
    widget._on_playback_state_changed(states.PlayingState)
    assert "Pause" in widget._play_pause_btn.text()
    widget._on_playback_state_changed(states.PausedState)
    assert "Play" in widget._play_pause_btn.text()
    widget._on_playback_state_changed(states.StoppedState)
    assert "Play" in widget._play_pause_btn.text()


def test_button_width_is_stable_across_labels():
    widget = _make_adapter()
    wider = max(
        QPushButton("▶ Play").sizeHint().width(),
        QPushButton("⏸ Pause").sizeHint().width(),
    )
    assert widget._play_pause_btn.minimumWidth() >= wider


def test_click_while_playing_pauses():
    widget = _make_adapter()
    stub = _StubPlayer(QMediaPlayer.PlaybackState.PlayingState)
    widget._player = stub
    widget._play_pause_btn.click()
    assert stub.calls == ["pause"]


def test_click_while_paused_goes_through_the_play_path():
    widget = _make_adapter()
    stub = _StubPlayer(QMediaPlayer.PlaybackState.PausedState)
    widget._player = stub
    called = []
    widget._on_play_clicked = lambda: called.append(True)
    widget._play_pause_btn.click()
    assert called == [True]
    assert "pause" not in stub.calls


def test_click_while_paused_plays():
    widget = _make_adapter()
    stub = _StubPlayer(QMediaPlayer.PlaybackState.StoppedState)
    widget._player = stub
    widget._play_pause_btn.click()
    assert stub.calls[-1] == "play"
    assert "pause" not in stub.calls
