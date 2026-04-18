"""
column_types/renderers.py

Render functions for each built-in column type.

Each render function has the signature:
    render(value: Any, size: int, mode: str = 'thumbnail')
        -> QPixmap | QWidget | None

In 'thumbnail' mode (used by gallery tiles), the function returns a
QPixmap scaled to fit within size x size pixels.

In 'detail' mode (used by DetailWidget), the function returns a QWidget
suitable for full-size display — a ZoomableImageView for images, a
QVideoWidget+QMediaPlayer for videos, a QLabel for text values.

Student A is responsible for this file. For each new operator that
produces a new kind of output, Student A adds a render function here.

The built-in render functions below are provided centrally as reference
implementations.

Note on factory functions:
    The media_path renderer needs access to ArtifactStore for thumbnail
    caching. We use a factory function that takes ArtifactStore as an
    argument and returns a render function, keeping each renderer
    self-contained and testable.

Note on media type detection:
    The media_path renderer dispatches on file extension internally.
    It does not rely on any metadata stored in the row — the extension
    alone determines whether to treat the file as an image or video.
    This means a single column (e.g. full_path) can hold a mix of
    image and video paths and each will be rendered correctly.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

from PIL import Image


# ---------------------------------------------------------------------------
# Known media extensions
# ---------------------------------------------------------------------------

# Extensions treated as images (loaded via PIL).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Extensions treated as videos (first frame extracted via OpenCV).
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _is_image(path: Path) -> bool:
    """Returns True if the file extension is a known image format."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _is_video(path: Path) -> bool:
    """Returns True if the file extension is a known video format."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# Media path renderer (images and videos)
# ---------------------------------------------------------------------------

def make_media_path_renderer(artifact_store):
    """
    Factory function that creates the renderer for 'media_path' columns.

    The returned renderer handles both images and videos by dispatching
    on file extension. It supports two display modes:

        'thumbnail': Returns a QPixmap scaled to fit within size x size.
                     For images: loads via PIL (using ArtifactStore cache).
                     For videos: extracts first frame via OpenCV.

        'detail':    Returns a QWidget for full-size display.
                     For images: returns a ZoomableImageView.
                     For videos: returns a QVideoWidget with QMediaPlayer.

    Args:
        artifact_store: The ArtifactStore instance for thumbnail caching.

    Returns:
        A render function: (value, size, mode) -> QPixmap | QWidget | None
    """

    def render(value: Any, size: int, mode: str = "thumbnail", context: dict | None = None):
        """
        Args:
            value:   A file path string pointing to an image or video.
            size:    Target size in pixels (used in thumbnail mode).
            mode:    'thumbnail' or 'detail'.
            context: Optional dict with row-level metadata, e.g.
                     {'row_id': ..., 'column_name': ...}.

        Returns:
            QPixmap in thumbnail mode, QWidget in detail mode, or None.
        """
        try:
            path = Path(str(value))
            if not path.exists():
                return None

            if _is_image(path):
                return _render_image(path, size, mode, artifact_store, context)
            elif _is_video(path):
                return _render_video(path, size, mode)
            else:
                # Unknown extension — return None so placeholder shows.
                print(f"[Renderer] Unsupported media extension: {path.suffix}")
                return None

        except Exception as e:
            print(f"[Renderer] media_path render error: {e}")
            return None

    return render


_PREVIEW_SIZE_THRESHOLD = 200  # pixels — tiles larger than this use 'preview'


def _render_image(
    path: Path,
    size: int,
    mode: str,
    artifact_store,
    context: dict | None = None,
):
    """
    Renders an image file.

    In thumbnail mode: checks ArtifactStore cache first for speed,
    falls back to loading directly if not cached. The artifact type
    is chosen by size: tiles <= 200px use 'thumbnail', larger use
    'preview'. Requires context={'row_id': ...} to hit the cache.

    In detail mode: returns a ZoomableImageView widget with the
    full-resolution image loaded.

    Args:
        path:           Path to the image file.
        size:           Target size in pixels (thumbnail mode only).
        mode:           'thumbnail' or 'detail'.
        artifact_store: ArtifactStore for thumbnail caching.
        context:        Optional dict with row-level metadata.

    Returns:
        QPixmap (thumbnail mode) or ZoomableImageView (detail mode).
    """
    if mode == "detail":
        from PySide6.QtGui import QPixmap
        from shared_widgets.zoomable_image_view import ZoomableImageView

        widget = ZoomableImageView()
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            widget.show_pixmap(pixmap)
        return widget

    else:
        # Thumbnail mode: try ArtifactStore cache if row_id is available.
        row_id = context.get("row_id") if context else None
        if row_id and artifact_store is not None:
            artifact_type = "thumbnail" if size <= _PREVIEW_SIZE_THRESHOLD else "preview"
            pil_image = artifact_store.get_pixmap(row_id, artifact_type)
            if pil_image is not None:
                img = pil_image.copy()
                img.thumbnail((size, size), Image.LANCZOS)
                return _pil_to_pixmap(img)

        # Cache miss or no context — load directly from disk.
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((size, size), Image.LANCZOS)
            return _pil_to_pixmap(img)


def _render_video(path: Path, size: int, mode: str):
    """
    Renders a video file.

    In thumbnail mode: extracts the first frame using OpenCV and
    returns it as a QPixmap.

    In detail mode: returns a QWidget containing a QMediaPlayer
    and QVideoWidget so the researcher can play the video.

    Args:
        path: Path to the video file.
        size: Target size in pixels (thumbnail mode only).
        mode: 'thumbnail' or 'detail'.

    Returns:
        QPixmap (thumbnail mode) or QWidget with video player (detail mode).
    """
    if mode == "detail":
        # Return a video player widget.
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout
        from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
        from PySide6.QtMultimediaWidgets import QVideoWidget
        from PySide6.QtCore import QUrl

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Video display widget.
        video_widget = QVideoWidget()
        layout.addWidget(video_widget)

        # Playback controls.
        controls = QHBoxLayout()
        play_btn = QPushButton("▶ Play")
        pause_btn = QPushButton("⏸ Pause")
        controls.addWidget(play_btn)
        controls.addWidget(pause_btn)
        controls.addStretch()
        layout.addLayout(controls)

        # Wire up the media player.
        # QAudioOutput is required in Qt6 to route audio.
        player = QMediaPlayer(container)
        audio  = QAudioOutput(container)
        player.setAudioOutput(audio)
        player.setVideoOutput(video_widget)
        player.setSource(QUrl.fromLocalFile(str(path)))

        play_btn.clicked.connect(player.play)
        pause_btn.clicked.connect(player.pause)

        return container

    else:
        # Thumbnail mode: extract first frame with OpenCV.
        return _video_first_frame_pixmap(path, size)


def _video_first_frame_pixmap(path: Path, size: int):
    """
    Extracts the first frame of a video and returns it as a QPixmap
    scaled to fit within size x size pixels.

    Uses OpenCV (cv2) for frame extraction. If OpenCV is not installed
    or the video cannot be read, returns None.

    Args:
        path: Path to the video file.
        size: Target size in pixels.

    Returns:
        A QPixmap, or None if extraction fails.
    """
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            return None

        # OpenCV returns BGR — convert to RGB for PIL.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((size, size), Image.LANCZOS)
        return _pil_to_pixmap(img)

    except ImportError:
        print("[Renderer] OpenCV (cv2) not installed — cannot render video thumbnail.")
        return None
    except Exception as e:
        print(f"[Renderer] Video frame extraction error: {e}")
        return None


# ---------------------------------------------------------------------------
# Non-visual renderers (visual=False)
# These return a QPixmap containing rendered text for thumbnail mode,
# or a QLabel for detail mode.
# ---------------------------------------------------------------------------

def render_numeric(value: Any, size: int, mode: str = "thumbnail", _context: dict | None = None):
    """
    Renders a numeric value (int, float, timestamp, duration).

    In thumbnail mode: returns a QPixmap with the formatted number.
    In detail mode: returns a QLabel with a larger font.

    Args:
        value: A float or int.
        size:  Target size in pixels (thumbnail mode).
        mode:  'thumbnail' or 'detail'.

    Returns:
        QPixmap or QLabel, or None.
    """
    try:
        if isinstance(value, float):
            text = f"{value:.4g}"  # 4 significant figures, no trailing zeros.
        else:
            text = str(value)

        if mode == "detail":
            return _text_to_label(text)
        return _text_to_pixmap(text, size, bg_color=(240, 248, 255))
    except Exception:
        return None


def render_text(value: Any, size: int, mode: str = "thumbnail", _context: dict | None = None):
    """
    Renders a text (string) value.

    In thumbnail mode: returns a QPixmap text badge.
    In detail mode: returns a QLabel with the full text, word-wrapped.

    Args:
        value: A string.
        size:  Target size in pixels (thumbnail mode).
        mode:  'thumbnail' or 'detail'.

    Returns:
        QPixmap or QLabel, or None.
    """
    try:
        text = str(value)
        if mode == "detail":
            return _text_to_label(text, word_wrap=True)
        return _text_to_pixmap(text, size, bg_color=(255, 248, 220))
    except Exception:
        return None


def render_boolean_flag(value: Any, size: int, mode: str = "thumbnail", _context: dict | None = None):
    """
    Renders a boolean value as a checkmark or cross symbol.

    Args:
        value: A bool or value that can be interpreted as bool.
        size:  Target size in pixels (thumbnail mode).
        mode:  'thumbnail' or 'detail'.

    Returns:
        QPixmap or QLabel, or None.
    """
    try:
        text = "✓" if bool(value) else "✗"
        bg   = (220, 255, 220) if bool(value) else (255, 220, 220)
        if mode == "detail":
            return _text_to_label(text)
        return _text_to_pixmap(text, size, bg_color=bg)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pil_to_pixmap(pil_image: Image.Image):
    """
    Converts a PIL Image to a QPixmap.
    Must be called on the main thread.

    Args:
        pil_image: A PIL Image in RGB or RGBA mode.

    Returns:
        A QPixmap, or None if Qt is not available.
    """
    try:
        from PySide6.QtGui import QPixmap, QImage

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        data = pil_image.tobytes("raw", "RGB")
        qimage = QImage(
            data,
            pil_image.width,
            pil_image.height,
            pil_image.width * 3,
            QImage.Format.Format_RGB888,
        )
        return QPixmap.fromImage(qimage)

    except Exception as e:
        print(f"[Renderer] PIL to QPixmap conversion error: {e}")
        return None


def _text_to_pixmap(
    text: str,
    size: int,
    bg_color: tuple[int, int, int] = (240, 240, 240),
):
    """
    Creates a QPixmap containing centered text on a colored background.
    Used by non-visual renderers in thumbnail mode.

    Args:
        text:     The text to display.
        size:     Width and height of the pixmap in pixels.
        bg_color: RGB background color tuple.

    Returns:
        A QPixmap, or None if Qt is not available.
    """
    try:
        from PySide6.QtGui import QPixmap, QPainter, QColor, QFont
        from PySide6.QtCore import Qt

        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(*bg_color))

        painter = QPainter(pixmap)
        painter.setPen(QColor(50, 50, 50))
        font = QFont()
        font.setPointSize(max(8, size // 15))
        painter.setFont(font)
        painter.drawText(
            pixmap.rect(),
            Qt.AlignmentFlag.AlignCenter,
            text,
        )
        painter.end()
        return pixmap

    except Exception as e:
        print(f"[Renderer] text_to_pixmap error: {e}")
        return None


def _text_to_label(text: str, word_wrap: bool = False):
    """
    Creates a QLabel displaying the given text.
    Used by non-visual renderers in detail mode.

    Args:
        text:      The text to display.
        word_wrap: Whether to enable word wrapping.

    Returns:
        A QLabel, or None if Qt is not available.
    """
    try:
        from PySide6.QtWidgets import QLabel
        from PySide6.QtCore import Qt

        label = QLabel(text)
        label.setWordWrap(word_wrap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            "font-size: 14px; color: #323232; padding: 8px;"
        )
        return label

    except Exception as e:
        print(f"[Renderer] text_to_label error: {e}")
        return None
