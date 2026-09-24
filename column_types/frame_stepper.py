"""
column_types/frame_stepper.py

Layer B (P1.4b part 2): the detail-view video widget that lets a
researcher toggle from playing an eligible clip to stepping through its
decoded frames one at a time.

Eligibility is decided exactly once, before this widget is built, by
ArtifactStore.clip_is_steppable -- column_types/renderers.py's
_render_video calls it and only constructs FrameStepperWidget when it is
True. This file does not re-check eligibility.

All arithmetic -- which cached frame is nearest a paused player
position, what player position a frame's presentation time names, the
label text, the slider's page step -- lives in media/frame_stepping.py
(Layer A). This file only wires Qt widgets to those results and to
column_types/playback_adapter.py's PlaybackAdapter, the only place that
constructs QMediaPlayer.
"""

from __future__ import annotations

from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from column_types.playback_adapter import PlaybackAdapter
from column_types.renderers import _pil_to_pixmap
from media.frame_stepping import (
    can_step_back,
    can_step_forward,
    frame_label_text,
    frame_slider_page_step,
    nearest_frame_index,
    player_position_ms_for_frame,
)
from media.playback import PlaybackSpan
from shared_widgets.zoomable_image_view import ZoomableImageView


class FrameStepperWidget(QWidget):
    """Wraps a PlaybackAdapter with a frame-by-frame stepper for a clip
    ArtifactStore.clip_is_steppable has already approved.

    Page 0 (shown by default) is the ordinary PlaybackAdapter. Toggling
    "Frame by frame" ON pauses it and swaps to page 1: a
    ZoomableImageView showing the cached frame nearest the player's
    paused position (fit to the pane on this entry only -- stepping
    afterwards keeps whatever zoom the researcher chose), a slider over
    frame indices (which Left/Right/PageUp/PageDown/Home/End already
    drive through QAbstractSlider's own key handling once it has focus),
    "Previous frame" / "Next frame" buttons either side of a label, and
    the label itself. Toggling OFF ("Back to video") seeks the player to
    the shown frame's own time and swaps back to page 0, still paused.

    While the clip's frames are not yet cached, the toggle is disabled
    and reads "Frame by frame (preparing...)".
    """

    def __init__(
        self,
        span: PlaybackSpan,
        canonical_address: str,
        artifact_store,
        clip_frames_ready_signal,
        parent=None,
    ):
        super().__init__(parent)
        self._address = canonical_address
        self._store = artifact_store
        self._frames: tuple | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._toggle = QPushButton()
        self._toggle.setCheckable(True)
        self._toggle.setEnabled(False)
        self._update_toggle_label()
        layout.addWidget(self._toggle)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, stretch=1)

        self._player = PlaybackAdapter(span)
        self._stack.addWidget(self._player)

        self._stepper_page = QWidget()
        stepper_layout = QVBoxLayout(self._stepper_page)
        stepper_layout.setContentsMargins(0, 0, 0, 0)
        self._image_view = ZoomableImageView()
        stepper_layout.addWidget(self._image_view, stretch=1)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setSingleStep(1)
        stepper_layout.addWidget(self._slider)

        step_row = QHBoxLayout()
        self._prev_btn = QPushButton("◀ Previous frame")
        self._prev_btn.setToolTip("Previous frame (Left arrow)")
        self._prev_btn.setEnabled(False)
        self._next_btn = QPushButton("Next frame ▶")
        self._next_btn.setToolTip("Next frame (Right arrow)")
        self._next_btn.setEnabled(False)
        self._frame_label = QLabel("")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        step_row.addWidget(self._prev_btn)
        step_row.addWidget(self._frame_label, stretch=1)
        step_row.addWidget(self._next_btn)
        stepper_layout.addLayout(step_row)
        self._stack.addWidget(self._stepper_page)

        self._toggle.toggled.connect(self._on_toggled)
        self._slider.valueChanged.connect(self._on_slider_value_changed)
        self._prev_btn.clicked.connect(self._on_previous_clicked)
        self._next_btn.clicked.connect(self._on_next_clicked)

        # A signal owned by the long-lived AppController outlives this
        # widget, so the connection must be removed explicitly when
        # DetailWidget deleteLater()'s it -- otherwise a later emission
        # calls into an already-destroyed widget. The naive fix,
        # disconnecting a BOUND METHOD of this widget from the
        # `destroyed` callback, was tried and crashed: by the time
        # `destroyed` fires, shiboken has already invalidated this
        # widget's C++ side, and disconnect() must inspect a bound
        # method's owner to match the connection, which raises
        # "libshiboken: Internal C++ object (FrameStepperWidget) already
        # deleted" -- exactly the class of teardown crash this project
        # has been bitten by before. `_forward_ready` is a plain
        # function, not a method of this widget, so neither connecting
        # nor disconnecting it ever has to validate this widget's own
        # QObject -- only its own call body (run before destruction,
        # never during it) touches `self`.
        if clip_frames_ready_signal is not None:
            def _forward_ready(address: str) -> None:
                self._on_clip_frames_ready(address)

            def _disconnect_ready() -> None:
                try:
                    clip_frames_ready_signal.disconnect(_forward_ready)
                except RuntimeError:
                    # The signal's own source (AppController) can be torn
                    # down before this widget is garbage-collected -- a
                    # real ordering hit during a full test run, not a
                    # widget-side bug: there is then nothing left to
                    # disconnect from.
                    pass

            clip_frames_ready_signal.connect(_forward_ready)
            self.destroyed.connect(_disconnect_ready)

        # Prefetch: queue the decode as soon as an eligible clip's detail
        # widget is built. Several detail panels may each request their
        # own clip; the store's two-clip LRU cancelling an older queued
        # decode is acceptable (P1.4b part 2 spec).
        self._store.request_clip_frames(self._address)

        cached = self._store.clip_frames(self._address)
        if cached is not None:
            self._set_frames(cached)

    def _on_clip_frames_ready(self, canonical_address: str) -> None:
        if canonical_address != self._address:
            return
        frames = self._store.clip_frames(self._address)
        if frames is not None:
            self._set_frames(frames)

    def _set_frames(self, frames: tuple) -> None:
        self._frames = frames
        self._toggle.setEnabled(True)
        self._update_toggle_label()

    def _update_toggle_label(self) -> None:
        """Sets the toggle's text and tooltip for its current state:
        disabled (frames not yet cached), enabled and unchecked (the
        video is showing), or enabled and checked (frames are showing).
        """
        if not self._toggle.isEnabled():
            self._toggle.setText("Frame by frame (preparing...)")
            self._toggle.setToolTip("Getting the frames of this clip ready")
        elif self._toggle.isChecked():
            self._toggle.setText("Back to video")
            self._toggle.setToolTip("Return to the player at this frame")
        else:
            self._toggle.setText("Frame by frame")
            self._toggle.setToolTip(
                "Pause and look at this clip one frame at a time. "
                "Arrow keys step."
            )

    def _on_toggled(self, checked: bool) -> None:
        self._update_toggle_label()
        if not self._frames:
            return
        if checked:
            self._player.pause()
            position_ms = self._player.position_ms()
            frame_times_us = [frame.presentation_time_us for frame in self._frames]
            index = nearest_frame_index(frame_times_us, position_ms)
            self._show_frame(index)
            self._stack.setCurrentWidget(self._stepper_page)
            # Deferred so the fit runs once the stepper page is actually
            # current and laid out, rather than against whatever size it
            # had before becoming visible. Fires once, on this entry into
            # frame view only -- stepping afterwards
            # (_on_slider_value_changed) keeps whatever zoom the
            # researcher chooses.
            QTimer.singleShot(0, self._image_view.fit_to_view)
            self._slider.setFocus()
        else:
            frame = self._frames[self._slider.value()]
            self._player.seek_to_ms(
                player_position_ms_for_frame(frame.presentation_time_us)
            )
            self._stack.setCurrentWidget(self._player)

    def _on_slider_value_changed(self, index: int) -> None:
        self._show_frame(index)

    def _on_previous_clicked(self) -> None:
        self._slider.setValue(self._slider.value() - 1)
        self._slider.setFocus()

    def _on_next_clicked(self) -> None:
        self._slider.setValue(self._slider.value() + 1)
        self._slider.setFocus()

    def _show_frame(self, index: int) -> None:
        frame = self._frames[index]
        pixmap = _pil_to_pixmap(Image.fromarray(frame.pixels))
        if pixmap is not None:
            self._image_view.set_pixmap_keeping_view(pixmap)

        total_frames = len(self._frames)
        self._slider.blockSignals(True)
        self._slider.setRange(0, total_frames - 1)
        self._slider.setPageStep(frame_slider_page_step(total_frames))
        self._slider.setValue(index)
        self._slider.blockSignals(False)

        self._prev_btn.setEnabled(can_step_back(index))
        self._next_btn.setEnabled(can_step_forward(index, total_frames))

        self._frame_label.setText(
            frame_label_text(
                frame.frame_ordinal, frame.presentation_time_us, index, total_frames
            )
        )
