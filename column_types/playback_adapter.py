"""
column_types/playback_adapter.py

P1.10: the one file that constructs QMediaPlayer for detail-view playback.
media/playback.py (Layer A) decides which millisecond to seek to, when to
pause, and where Play should restart from; this widget (Layer B) only calls
Qt with those results -- no position or pause logic is computed here.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from media.playback import (
    PlaybackSpan,
    initial_position_ms,
    play_from_ms,
    position_for_slider_value,
    should_pause,
    slider_page_step_ms,
    slider_single_step_ms,
    slider_value_for_position,
    span_length_ms,
)


class PlaybackAdapter(QWidget):
    """Plays exactly the span of video a PlaybackSpan names.

    A bare-path span (end_ms=None) plays like a whole-file player. A range
    span pauses itself at its own end, and Play restarts it from the
    span's start when pressed at or after that end -- both decisions come
    from media/playback.py, never computed here.
    """

    def __init__(self, span: PlaybackSpan, parent=None):
        super().__init__(parent)
        self._span = span
        # None until durationChanged (or LoadedMedia) reports it; every
        # slider computation goes through span_length_ms(), which already
        # treats an unknown duration correctly, so this is the only state
        # this widget adds.
        self._duration_ms: int | None = None
        # True once the initial seek-to-start-and-play has run -- see
        # _on_media_status_changed for why this must happen only once.
        self._initial_play_started = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._video_widget = QVideoWidget()
        layout.addWidget(self._video_widget)

        # Position slider, between the video and the elapsed label. Its
        # range is unknown (and it stays disabled) until the span's length
        # can be computed -- see _update_slider_range().
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setEnabled(False)
        layout.addWidget(self._slider)

        self._elapsed_label = QLabel(self._format_elapsed(span.start_ms))
        layout.addWidget(self._elapsed_label)

        controls = QHBoxLayout()
        play_btn = QPushButton("▶ Play")
        pause_btn = QPushButton("⏸ Pause")
        controls.addWidget(play_btn)
        controls.addWidget(pause_btn)
        controls.addStretch()
        layout.addLayout(controls)

        # QAudioOutput is required in Qt6 to route audio.
        self._player = QMediaPlayer(self)
        audio = QAudioOutput(self)
        self._player.setAudioOutput(audio)
        self._player.setVideoOutput(self._video_widget)

        # setPosition() before the media has finished loading is a no-op,
        # so the initial seek to the span's start waits for this signal.
        # LoadedMedia is also the first point self._player.duration() is
        # trustworthy, so the slider's range is set here too.
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)
        # Some containers report duration only after LoadedMedia already
        # fired (or revise it later); this keeps the slider's range current
        # whenever that happens.
        self._player.durationChanged.connect(self._on_duration_changed)
        # Watches playback so a range span pauses itself at its own end
        # instead of playing on into whatever follows it in the file, and
        # keeps the slider positioned to match -- except while the user is
        # holding it down, so the player's reports don't fight the drag.
        self._player.positionChanged.connect(self._on_position_changed)

        # valueChanged fires for every value change regardless of cause --
        # a click on the bar (a page step, QSlider's own default), a drag,
        # or a keyboard step -- so this single connection covers
        # sliderPressed and sliderMoved's cases and more; both are
        # unnecessary and are not connected. A programmatic setValue() (in
        # _on_position_changed) or setRange() (in _update_slider_range)
        # would also emit valueChanged, so both are wrapped in
        # blockSignals() to stop them from seeking the player right back
        # to where it already told the slider it was.
        self._slider.valueChanged.connect(self._on_slider_value_changed)

        # Play re-derives where to seek from (span start, if Play is
        # pressed at or after the span's end) before resuming.
        play_btn.clicked.connect(self._on_play_clicked)
        # Pause needs no span logic at all -- it always just pauses.
        pause_btn.clicked.connect(self._player.pause)

        self._player.setSource(QUrl.fromLocalFile(span.source_path))

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            if not self._initial_play_started:
                self._initial_play_started = True
                self._player.setPosition(initial_position_ms(self._span))
                # The player opens playing, from the span's start, rather
                # than opening paused. A range still pauses itself at its
                # own end through should_pause (_on_position_changed), so
                # this does not change what a range span shows once it
                # settles. Audio is not muted. What actually makes the
                # slider work is the guard below (self._initial_play_started),
                # not autoplay itself -- see docs/known_defects.md for the
                # defect this guard fixes.
                self._player.play()
            # LoadedMedia is re-emitted by this machine's Qt Multimedia
            # backend after every later setPosition() call too -- verified
            # by a scratchpad probe against a real file, not assumed. Were
            # the block above not guarded, every slider seek would
            # immediately be undone by a seek back to the span's start:
            # exactly the "clicking the slider restarts the video" defect
            # this guard fixes. A later LoadedMedia still refreshes the
            # duration and slider range below, since some containers only
            # report an accurate duration once playback is underway.
            self._duration_ms = self._player.duration()
            self._update_slider_range()

    def _on_duration_changed(self, duration_ms: int) -> None:
        self._duration_ms = duration_ms
        self._update_slider_range()

    def _update_slider_range(self) -> None:
        """Enables the slider and sets its track length once the span's
        length can be computed; keeps it disabled at 0 until then.

        setRange() can itself clamp the current value into the new bounds
        and emit valueChanged for that clamp -- a duration revised down
        after the slider had already moved is the realistic case. That
        would reach _on_slider_value_changed and seek the player to a
        value nobody chose, so signals are blocked around the call; the
        clamped value (if any) is still applied, just silently. setPageStep
        and setSingleStep do not themselves move the slider's value, so
        they cannot emit valueChanged, but are set inside the same blocked
        section for one clear boundary.
        """
        length_ms = span_length_ms(self._span, self._duration_ms)
        self._slider.blockSignals(True)
        if length_ms is None:
            self._slider.setEnabled(False)
            self._slider.setRange(0, 0)
        else:
            self._slider.setRange(0, length_ms)
            self._slider.setPageStep(slider_page_step_ms(length_ms))
            self._slider.setSingleStep(slider_single_step_ms(length_ms))
            self._slider.setEnabled(True)
        self._slider.blockSignals(False)

    def _on_position_changed(self, position_ms: int) -> None:
        self._elapsed_label.setText(self._format_elapsed(position_ms))
        if should_pause(position_ms, self._span):
            self._player.pause()
        # While the user holds the slider down, its value is theirs to
        # set -- following the player here would fight the drag. Signals
        # are blocked around setValue() so this programmatic update does
        # not itself trigger valueChanged and seek the player again.
        if not self._slider.isSliderDown():
            value = slider_value_for_position(position_ms, self._span, self._duration_ms)
            self._slider.blockSignals(True)
            self._slider.setValue(value)
            self._slider.blockSignals(False)

    def _on_slider_value_changed(self, value: int) -> None:
        self._player.setPosition(
            position_for_slider_value(value, self._span, self._duration_ms)
        )

    def _on_play_clicked(self) -> None:
        at_end = self._player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia
        restart_ms = play_from_ms(
            self._player.position(), self._span, at_end_of_media=at_end
        )
        if restart_ms is not None:
            self._player.setPosition(restart_ms)
        self._player.play()

    def pause(self) -> None:
        """Pauses playback. Used by the frame stepper (P1.4b part 2) to
        stop the player before it swaps in the still-frame view."""
        self._player.pause()

    def position_ms(self) -> int:
        """The player's current position in milliseconds -- what the
        frame stepper's toggle reads to find the nearest cached frame."""
        return self._player.position()

    def seek_to_ms(self, position_ms: int) -> None:
        """Seeks to an absolute player position, in milliseconds -- what
        the frame stepper's toggle uses to hand playback back to the
        player at the frame it was showing."""
        self._player.setPosition(position_ms)

    def _format_elapsed(self, position_ms: int) -> str:
        """'elapsed / length' in seconds, relative to the span's start."""
        elapsed_s = max(0, position_ms - self._span.start_ms) / 1000
        if self._span.end_ms is None:
            return f"{elapsed_s:.1f}s"
        length_s = (self._span.end_ms - self._span.start_ms) / 1000
        return f"{elapsed_s:.1f}s / {length_s:.1f}s"
