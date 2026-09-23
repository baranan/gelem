"""
column_types/playback_adapter.py

P1.10: the one file that constructs QMediaPlayer for detail-view playback.
media/playback.py (Layer A) decides which millisecond to seek to, when to
pause, and where Play should restart from; this widget (Layer B) only calls
Qt with those results -- no position or pause logic is computed here.
"""

from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from media.playback import PlaybackSpan, initial_position_ms, play_from_ms, should_pause


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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._video_widget = QVideoWidget()
        layout.addWidget(self._video_widget)

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
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)
        # Watches playback so a range span pauses itself at its own end
        # instead of playing on into whatever follows it in the file.
        self._player.positionChanged.connect(self._on_position_changed)

        # Play re-derives where to seek from (span start, if Play is
        # pressed at or after the span's end) before resuming.
        play_btn.clicked.connect(self._on_play_clicked)
        # Pause needs no span logic at all -- it always just pauses.
        pause_btn.clicked.connect(self._player.pause)

        self._player.setSource(QUrl.fromLocalFile(span.source_path))

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self._player.setPosition(initial_position_ms(self._span))

    def _on_position_changed(self, position_ms: int) -> None:
        self._elapsed_label.setText(self._format_elapsed(position_ms))
        if should_pause(position_ms, self._span):
            self._player.pause()

    def _on_play_clicked(self) -> None:
        at_end = self._player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia
        restart_ms = play_from_ms(
            self._player.position(), self._span, at_end_of_media=at_end
        )
        if restart_ms is not None:
            self._player.setPosition(restart_ms)
        self._player.play()

    def _format_elapsed(self, position_ms: int) -> str:
        """'elapsed / length' in seconds, relative to the span's start."""
        elapsed_s = max(0, position_ms - self._span.start_ms) / 1000
        if self._span.end_ms is None:
            return f"{elapsed_s:.1f}s"
        length_s = (self._span.end_ms - self._span.start_ms) / 1000
        return f"{elapsed_s:.1f}s / {length_s:.1f}s"
