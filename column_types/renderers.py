"""
column_types/renderers.py

Render functions for each built-in column type.

Each render function has the signature:
    render(value: Any, size: int, mode: str = 'thumbnail')
        -> QPixmap | QWidget | None

In 'thumbnail' mode (used by gallery tiles), the function returns a
QPixmap scaled to fit within size x size pixels. For media columns this
mode is demand-driven (P0.5b-3i): it returns the cached picture on a hit
and a grey placeholder on a miss, and NEVER opens or decodes a source
file. The generation request is queued by the controller, not here.

In 'detail' mode (used by DetailWidget), the function returns a QWidget
suitable for full-size display — a ZoomableImageView for images, a
PlaybackAdapter for videos, a QLabel for text values. Detail mode is the
deliberate exception that still opens the source.

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

    The one exception (CC-35): in detail mode, render_column_value()
    (controller.py) already decoded a #f=N or #t= point address through
    the shared MediaResolver, because a single-frame address on a video
    file needs to show that exact frame as a still, not the whole file
    in a player -- extension alone cannot see the fragment. It signals
    this through context['media_selects_single_frame'] and hands the
    already-decoded pixels in context['detail_frame_pixels'], so this
    file still does no address parsing and no second decode.
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
                     Cache hit -> the cached picture; cache miss -> a grey
                     placeholder, with the generation request queued by
                     the controller. No source file is opened here, for
                     images or videos (P0.5b-3i).

        'detail':    Returns a QWidget for full-size display, opening the
                     source directly.
                     For images: returns a ZoomableImageView.
                     For videos: returns a PlaybackAdapter -- a #t=A-B
                     RANGE address plays only that range; a bare path
                     plays the whole file (P1.10).

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
            # render_column_value() resolves the media cell once and drops
            # the canonical address and an absolute source path into the
            # context, so this renderer needs no project root and does no
            # address parsing of its own. In detail mode a direct caller
            # that skipped that step falls back to treating the raw value
            # as a path.
            ctx = context or {}

            # Thumbnail mode is demand-driven (P0.5b-3i): it is
            # cache-or-placeholder and NEVER stats, opens or decodes a
            # source file -- not for an image tile and not for a video
            # tile. On a hit the cache lookup touches no filesystem; on a
            # miss we return a placeholder immediately and the controller
            # (render_column_value) queues the generation request. The
            # ready notification repaints the tile, which then hits.
            if mode == "thumbnail":
                cached = _cached_thumbnail(ctx, size, artifact_store)
                if cached is not None:
                    return cached
                return _placeholder_pixmap(size)

            # Detail mode is the deliberate exception -- it opens the
            # source directly for a full-size view.
            source_path = ctx.get("source_path") or str(value)
            path = Path(source_path)
            if not path.exists():
                return None

            # CC-35: an address that selects exactly one frame (#f=N, or
            # a #t= time POINT) renders as a still of that frame, even
            # when the file extension is a video one -- render_column_
            # value() (controller.py) already decoded it through the
            # shared MediaResolver and dropped the upright RGB pixels
            # into the context, because deciding this from the extension
            # alone (the bug this fixes) cannot see the fragment. A bare
            # path or a #t= RANGE never sets this flag, so it falls
            # through to the ordinary image/video dispatch below, where a
            # RANGE reaches _render_video() and plays only that span
            # (P1.10).
            if ctx.get("media_selects_single_frame"):
                pixels = ctx.get("detail_frame_pixels")
                if pixels is None:
                    # The resolver failed to decode the requested frame --
                    # show nothing rather than silently opening the whole
                    # video and displaying an unrelated frame.
                    return None
                return _render_still_from_pixels(pixels)

            if _is_image(path):
                return _render_image(path)
            elif _is_video(path):
                return _render_video(ctx, source_path, artifact_store)
            else:
                # Unknown extension — return None so placeholder shows.
                print(f"[Renderer] Unsupported media extension: {path.suffix}")
                return None

        except Exception as e:
            print(f"[Renderer] media_path render error: {e}")
            return None

    return render


def _placeholder_pixmap(size: int):
    """A neutral grey QPixmap shown in a tile while its real picture is
    still being generated (P0.5b-3i).

    The renderer never decodes a source on the paint path, so a cache
    miss returns this immediately. It is paintable, not None, so the
    tile shows a stable grey slot rather than a blank canvas and the
    caller never has to special-case a missing return.
    """
    try:
        from PySide6.QtGui import QPixmap, QColor

        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(210, 210, 210))
        return pixmap

    except Exception as e:
        print(f"[Renderer] placeholder pixmap error: {e}")
        return None


def _cached_thumbnail(context: dict | None, size: int, artifact_store):
    """A QPixmap from the ArtifactStore cache for this context's canonical
    media address, or None on a miss.

    On a memory-cache hit this touches no filesystem -- no stat, no
    exists, no open. `render_column_value()` puts 'canonical_address' in
    the context; without it (a direct caller) this always misses and the
    source path is used instead.
    """
    if not context or artifact_store is None:
        return None
    address = context.get("canonical_address")
    if not address:
        return None
    # Both the tile-size -> purpose decision and the purpose -> resolution
    # mapping live only on the store instance, which is already injected
    # here -- this renderer no longer holds a size threshold constant or a
    # resolution constant of its own. purpose_for_tile_size() draws the
    # line at the store's configured thumbnail resolution.
    purpose = artifact_store.purpose_for_tile_size(size)
    resolution = artifact_store.resolution_for(purpose)
    pil_image = artifact_store.get_pixmap(address, purpose, resolution)
    if pil_image is None:
        return None
    img = pil_image.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    return _pil_to_pixmap(img)


def _render_image(path: Path):
    """
    Renders an image file for detail mode: a ZoomableImageView widget
    with the full-resolution image loaded through Qt's QImageReader.

    Thumbnail mode never reaches here -- render() serves it
    cache-or-placeholder and never decodes a source (P0.5b-3i). This is
    not a source decode by PIL or cv2; it is Qt loading a file for a
    full-size view, the same as detail mode has always done.

    QImageReader with setAutoTransform(True) applies the file's EXIF
    orientation the same way the resolver does for analysis (decision 6,
    docs/media_architecture.md section 3.6) -- a plain QPixmap(path)
    construction does not apply it, so a sideways-stored, upright-
    displayed photo would show sideways here otherwise.

    Args:
        path: Path to the image file.

    Returns:
        A ZoomableImageView widget.
    """
    from PySide6.QtGui import QImageReader, QPixmap
    from shared_widgets.zoomable_image_view import ZoomableImageView

    widget = ZoomableImageView()
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    image = reader.read()
    if not image.isNull():
        widget.show_pixmap(QPixmap.fromImage(image))
    return widget


def _render_still_from_pixels(pixels):
    """
    CC-35: renders an already-decoded frame (a #f=N or #t= point address)
    as a still, in the same ZoomableImageView widget an image column uses
    -- so zoom/pan and Save as PNG (DetailWidget checks
    isinstance(widget, ZoomableImageView)) work exactly as they do for an
    ordinary image.

    `pixels` is the upright RGB uint8 numpy array render_column_value()
    (controller.py) already decoded through the shared MediaResolver --
    this function does no decoding and opens no file itself.

    Args:
        pixels: RGB uint8 numpy array, shape (height, width, 3).

    Returns:
        A ZoomableImageView widget.
    """
    from shared_widgets.zoomable_image_view import ZoomableImageView

    widget = ZoomableImageView()
    pixmap = _pil_to_pixmap(Image.fromarray(pixels))
    if pixmap is not None:
        widget.show_pixmap(pixmap)
    return widget


def _render_video(ctx: dict, source_path: str, artifact_store):
    """
    Renders a video file for detail mode: a PlaybackAdapter widget that
    plays a #t=A-B RANGE address as only that range, or a bare path as the
    whole file (P1.10). Deciding the span is media/playback.py's job; this
    only reads its result and hands it to the widget.

    P1.4b part 2: when the clip is short enough for the frame-by-frame
    stepper (ArtifactStore.clip_is_steppable -- the ONE place that
    eligibility is decided), this returns column_types/frame_stepper.py's
    wrapper widget instead of the bare PlaybackAdapter. That wrapper holds
    a PlaybackAdapter itself, so this stays the only site deciding which
    widget the detail view gets.

    Thumbnail mode never reaches here -- render() serves it
    cache-or-placeholder and never opens a source (P0.5b-3i).

    Args:
        ctx:            The render context. Its 'canonical_address' (set
                        by controller.py's render_column_value) decides
                        the span; when absent -- the cell did not parse
                        as a media address -- the whole file plays, from
                        source_path, and the stepper is never offered
                        (clip_is_steppable needs a canonical address).
                        Its 'clip_frames_ready' (also set by
                        render_column_value in detail mode) is the
                        controller signal the stepper wrapper watches for
                        its clip's frames finishing decode.
        source_path:    Absolute path to the video file.
        artifact_store: The ArtifactStore instance -- for clip
                        eligibility and, inside the stepper wrapper, the
                        frame cache itself.

    Returns:
        A FrameStepperWidget for a steppable clip, otherwise a bare
        PlaybackAdapter.
    """
    from column_types.playback_adapter import PlaybackAdapter
    from media.playback import PlaybackSpan, playback_span

    canonical_address = ctx.get("canonical_address")
    if canonical_address:
        span = playback_span(canonical_address)
    else:
        span = PlaybackSpan(source_path=source_path, start_ms=0, end_ms=None)

    if (
        canonical_address
        and artifact_store is not None
        and artifact_store.clip_is_steppable(canonical_address)
    ):
        from column_types.frame_stepper import FrameStepperWidget

        return FrameStepperWidget(
            span,
            canonical_address,
            artifact_store,
            ctx.get("clip_frames_ready"),
        )

    return PlaybackAdapter(span)


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
