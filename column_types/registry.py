"""
column_types/registry.py

ColumnTypeRegistry maps a column *type tag* to a renderer -- nothing else.

A type tag is a short string such as 'media_path', 'numeric', 'text' or
'boolean_flag'. Each tag owns one ColumnType, which carries the tag's
human-readable label, whether it produces a visual (image-like) output,
and the render function that turns a cell value into a QPixmap
(thumbnail mode) or a QWidget (detail mode).

What the registry no longer does:
    Until P1.8d it also held a second map, column name -> ColumnType,
    that Dataset and the operator output path wrote to and that
    FilterPanel and the gallery read back. That map made two tables
    sharing a column name share one type, which is wrong. It is gone.
    The authority for what a named column is now the table's own
    TableSchema (docs/architecture.md 4.3); AppController reads the
    column's type tag off that schema and asks this registry only what
    the tag renders as.

Who populates the registry:
    - main.py calls setup_defaults() once at startup to register the
      four built-in tags against their render functions.
    - A new custom visual type is added with register_type().

Who reads the registry:
    - AppController.render_column_value() / render_result_image() call
      render_by_tag() to paint a cell or an operator result.
    - AppController.get_column_type() calls type_for_tag() to answer the
      UI's "what kind of column is this" question.

This file is written centrally (not by a student).
New render functions are added in renderers.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import pandas as pd


# ---------------------------------------------------------------------------
# ColumnType
# ---------------------------------------------------------------------------

@dataclass
class ColumnType:
    """
    Describes how a column's values should be displayed.

    Attributes:
        tag:     A short string identifier, e.g. 'media_path',
                 'numeric', 'text'.
        label:   A human-readable name shown in the column selector,
                 e.g. 'Media file', 'Number'.
        visual:  True if this column type produces an image-like output
                 that can be shown in the gallery tile. False for types
                 like 'numeric' and 'text' that produce text tiles.
        render:  A function with signature:
                     render(value: Any, size: int,
                            mode: str = 'thumbnail',
                            context: dict | None = None) -> QPixmap | QWidget | None
                 Takes a cell value, a target pixel size, a display
                 mode ('thumbnail' or 'detail'), and an optional context
                 dict (e.g. {'row_id': ..., 'column_name': ...}) that
                 renderers can use for cache lookups.
                 In 'thumbnail' mode, returns a QPixmap for the gallery.
                 In 'detail' mode, returns a QWidget for DetailWidget.
                 Returns None if the value cannot be rendered.
    """
    tag: str
    label: str
    visual: bool
    render: Callable[[Any, int, str, Any], Any]  # Returns QPixmap, QWidget, or None.


# ---------------------------------------------------------------------------
# ColumnTypeRegistry
# ---------------------------------------------------------------------------

class ColumnTypeRegistry:
    """
    Maps a type tag to its ColumnType (label, visual flag, renderer).

    It holds no knowledge of any particular column -- two tables whose
    schemas both call a column 'score' but tag it differently get
    different renderers, because the caller passes the tag, not the
    column name.

    Usage:
        registry = ColumnTypeRegistry()
        registry.setup_defaults(artifact_store)

        # Gallery tile (thumbnail mode):
        pixmap = registry.render_by_tag('media_path', '/path/to/video.mp4', 150)

        # Detail view (detail mode):
        widget = registry.render_by_tag('media_path', '/path/to/video.mp4',
                                        600, mode='detail')

        # "What kind of column is this?"
        col_type = registry.type_for_tag('numeric')
    """

    def __init__(self):
        # Maps type tag -> ColumnType.
        # Built-in tags are registered here by setup_defaults().
        self._types: dict[str, ColumnType] = {}

    def setup_defaults(self, artifact_store) -> None:
        """
        Registers all built-in column types with their render functions.
        Must be called once during application startup, after
        ArtifactStore is created.

        Built-in types:
            'media_path'  — any media file (image or video). The renderer
                            dispatches on file extension internally.
            'numeric'     — numbers, including durations and timestamps.
            'text'        — any string value. FilterPanel shows toggle
                            buttons for low-cardinality columns and a
                            text search input for high-cardinality ones.
            'boolean_flag'— True/False values.

        Args:
            artifact_store: The ArtifactStore instance, passed to
                            renderers that need to load cached images.
        """
        from column_types.renderers import (
            make_media_path_renderer,
            render_numeric,
            render_text,
            render_boolean_flag,
        )

        self._types = {
            "media_path": ColumnType(
                tag="media_path",
                label="Media file",
                visual=True,
                render=make_media_path_renderer(artifact_store),
            ),
            "numeric": ColumnType(
                tag="numeric",
                label="Number",
                visual=False,
                render=render_numeric,
            ),
            "text": ColumnType(
                tag="text",
                label="Text",
                visual=False,
                render=render_text,
            ),
            "boolean_flag": ColumnType(
                tag="boolean_flag",
                label="Flag",
                visual=False,
                render=render_boolean_flag,
            ),
        }

    def register_type(self, col_type: ColumnType) -> None:
        """
        Registers a new custom column type by tag.
        Used to add a new visual type alongside a new operator.

        Args:
            col_type: The new ColumnType to register.
        """
        self._types[col_type.tag] = col_type

    def type_for_tag(self, tag: str) -> ColumnType | None:
        """
        Returns the ColumnType registered under a type tag, or None if
        the tag is unknown.

        AppController reads a column's type tag off that table's
        TableSchema and asks this method what the tag renders as -- so
        two tables that share a column name no longer share one type.

        Args:
            tag: A type tag, e.g. 'media_path', 'numeric', 'text'.

        Returns:
            The ColumnType, or None.
        """
        return self._types.get(tag, None)

    def render_by_tag(
        self,
        tag: str | None,
        value: Any,
        size: int,
        mode: str = "thumbnail",
        context: dict | None = None,
        *,
        label: str | None = None,
    ) -> Any:
        """
        Renders a value using the renderer for a type tag.

        Placeholder behaviour:
          * an unknown tag (or None) -> the "unknown" placeholder;
          * a None or NaN value      -> the "not computed" placeholder;
          * a renderer that raises    -> the "error" placeholder.

        `label` is used only in the placeholder message. AppController
        passes the column name it was asked about, so the three messages
        read "Unknown column: <name>", "Not computed: <name>",
        "Error: <name>". A caller with no column name --
        AppController.render_result_image, rendering an operator's output
        file -- passes no label, and each message falls back to a plain
        sentence for a researcher that names no tag.

        Args:
            tag:     The type tag to render as, or None (treated as unknown).
            value:   The cell value from the DataFrame row.
            size:    Target size in pixels.
            mode:    'thumbnail' (default) or 'detail'.
            context: Optional dict with row-level metadata, passed through
                     to the renderer for cache lookups.
            label:   Name of the thing being rendered, for the placeholder
                     message only. None -> a plain no-name sentence.

        Returns:
            A QPixmap (thumbnail mode), QWidget (detail mode), or None.
        """
        col_type = self._types.get(tag, None) if tag is not None else None

        if col_type is None:
            message = (
                f"Unknown column:\n{label}" if label
                else "This value cannot be displayed."
            )
            return self._placeholder(mode, size, message)

        if value is None or (isinstance(value, float) and pd.isna(value)):
            message = (
                f"Not computed:\n{label}" if label
                else "Nothing to display yet."
            )
            return self._placeholder(mode, size, message)

        try:
            return col_type.render(value, size, mode, context)
        except Exception as e:
            print(f"[ColumnTypeRegistry] render_by_tag error for tag '{tag}': {e}")
            message = (
                f"Error:\n{label}" if label
                else "This image could not be displayed."
            )
            return self._placeholder(mode, size, message)

    @staticmethod
    def _placeholder(mode: str, size: int, message: str):
        """A placeholder widget (detail mode) or pixmap (thumbnail mode)
        carrying `message`. One place for the mode split that render_by_tag's
        three placeholder branches share."""
        if mode == "detail":
            return _make_placeholder_widget(message)
        return _make_placeholder_pixmap(size, message)


# ---------------------------------------------------------------------------
# Placeholder helpers
# ---------------------------------------------------------------------------

def _make_placeholder_pixmap(size: int, label: str):
    """
    Creates a gray QPixmap with a text label for use when a column
    value cannot be rendered. Returns None if Qt is not available.

    Args:
        size:  The width and height of the placeholder in pixels.
        label: Short text to display inside the placeholder.

    Returns:
        A QPixmap, or None.
    """
    try:
        from PySide6.QtGui import QPixmap, QPainter, QColor, QFont
        from PySide6.QtCore import Qt

        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(200, 200, 200))

        painter = QPainter(pixmap)
        painter.setPen(QColor(100, 100, 100))
        font = QFont()
        font.setPointSize(max(7, size // 20))
        painter.setFont(font)
        painter.drawText(
            pixmap.rect(),
            Qt.AlignmentFlag.AlignCenter,
            label,
        )
        painter.end()
        return pixmap

    except Exception:
        return None


def _make_placeholder_widget(label: str):
    """
    Creates a simple QLabel placeholder widget for use in detail mode
    when a value cannot be rendered.

    Args:
        label: Text to display.

    Returns:
        A QLabel widget, or None if Qt is not available.
    """
    try:
        from PySide6.QtWidgets import QLabel
        from PySide6.QtCore import Qt

        widget = QLabel(label)
        widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        widget.setStyleSheet(
            "background-color: #C8C8C8; color: #646464; padding: 8px;"
        )
        return widget

    except Exception:
        return None
