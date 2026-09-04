"""
column_types/registry.py

ColumnTypeRegistry is the bridge between data and display.
It maps each column name to a column type, and each column type
to a render function.

When a tile needs to display a column's value, it asks the registry
for the render function, calls it with the value, a target size, and
a display mode, and receives a QPixmap or QWidget back. The tile does
not know or care whether the value is a file path, a number, or a
label — the render function handles all of that.

Display modes:
    'thumbnail' — used by gallery tiles; always returns a QPixmap.
    'detail'    — used by DetailWidget; returns a QWidget (e.g. a
                  video player or zoomable image view).

Who populates the registry:
    - Dataset registers columns when it loads a folder or merges a CSV.
    - OperatorRegistry registers columns before an operator runs,
      so tiles can show informative placeholders immediately.

Who reads the registry:
    - ImageTile calls render() with mode='thumbnail' to get a QPixmap.
    - DetailWidget calls render() with mode='detail' to get a QWidget.
    - FilterPanel calls get() to decide what control to show.
    - GalleryWidget calls list_visual_columns() to populate the
      column selector.

This file is written centrally (not by a student).
Student A adds new render functions in renderers.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
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
    Maps column names to ColumnType objects and provides rendering.

    All components interact with columns through this registry rather
    than making assumptions about what a column contains.

    Usage:
        registry = ColumnTypeRegistry()
        registry.register_by_tag('full_path', 'media_path')

        # Gallery tile (thumbnail mode):
        pixmap = registry.render('full_path', '/path/to/video.mp4', 150)

        # Detail view (detail mode):
        widget = registry.render('full_path', '/path/to/video.mp4', 600,
                                 mode='detail')
    """

    def __init__(self):
        # Maps column name -> ColumnType.
        self._columns: dict[str, ColumnType] = {}

        # Maps type tag -> ColumnType.
        # Built-in types are registered here by setup_defaults().
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

    def register(self, column_name: str, col_type: ColumnType) -> None:
        """
        Registers a column name with a fully specified ColumnType object.
        Used when an operator wants to provide a custom label or render
        function for a column it produces.

        Args:
            column_name: The column name as it appears in the DataFrame.
            col_type:    The ColumnType object describing this column.
        """
        self._columns[column_name] = col_type

    def register_by_tag(self, column_name: str, tag: str) -> None:
        """
        Registers a column name using a built-in type tag.
        The most common way for Dataset and OperatorRegistry to register
        columns.

        Args:
            column_name: The column name as it appears in the DataFrame.
            tag:         A built-in type tag, e.g. 'media_path',
                         'numeric', 'text'.

        Raises:
            KeyError: If the tag is not a known built-in type.
        """
        if tag not in self._types:
            raise KeyError(
                f"Unknown column type tag '{tag}'. "
                f"Known tags: {list(self._types.keys())}"
            )
        self._columns[column_name] = self._types[tag]

    def clear_column_map(self) -> None:
        """
        Forgets every column-name -> ColumnType mapping, leaving the registered
        types (and their tags) intact.

        Dataset.load() calls this at its point of no return so a freshly opened
        project does not inherit the previously open project's column tags. The
        built-in types from setup_defaults() and any operator-registered types
        stay available, ready for the new project's columns to be re-registered
        against them.
        """
        self._columns.clear()

    def register_type(self, col_type: ColumnType) -> None:
        """
        Registers a new custom column type by tag.
        Used by Student A to add new visual types alongside new operators.

        Args:
            col_type: The new ColumnType to register.
        """
        self._types[col_type.tag] = col_type

    def get(self, column_name: str) -> ColumnType | None:
        """
        Returns the ColumnType for a given column name, or None if
        the column has not been registered.

        Args:
            column_name: The column name to look up.

        Returns:
            The ColumnType, or None.
        """
        return self._columns.get(column_name, None)

    def render(
        self,
        column_name: str,
        value: Any,
        size: int,
        mode: str = "thumbnail",
        context: dict | None = None,
    ) -> Any:
        """
        Looks up the render function for the column and calls it.

        In 'thumbnail' mode, returns a QPixmap ready for display in a
        gallery tile.

        In 'detail' mode, returns a QWidget ready for display in
        DetailWidget (e.g. a ZoomableImageView or a QVideoWidget).

        If the column is not registered, returns a gray placeholder
        QPixmap (thumbnail mode) or a placeholder QLabel (detail mode).

        If the value is None (operator has not run yet), returns an
        informative placeholder.

        Args:
            column_name: The column to render.
            value:       The cell value from the DataFrame row.
            size:        Target size in pixels.
            mode:        'thumbnail' (default) or 'detail'.
            context:     Optional dict with row-level metadata, e.g.
                         {'row_id': ..., 'column_name': ...}. Passed
                         through to renderers for cache lookups.

        Returns:
            A QPixmap (thumbnail mode), QWidget (detail mode), or None.
        """
        col_type = self._columns.get(column_name, None)

        if col_type is None:
            if mode == "detail":
                return _make_placeholder_widget(f"Unknown column:\n{column_name}")
            return _make_placeholder_pixmap(size, f"Unknown:\n{column_name}")

        if value is None or (isinstance(value, float) and pd.isna(value)):
            if mode == "detail":
                return _make_placeholder_widget(f"Not computed:\n{column_name}")
            return _make_placeholder_pixmap(size, f"Not computed:\n{column_name}")

        try:
            return col_type.render(value, size, mode, context)
        except Exception as e:
            print(f"[ColumnTypeRegistry] render error for '{column_name}': {e}")
            if mode == "detail":
                return _make_placeholder_widget(f"Error:\n{column_name}")
            return _make_placeholder_pixmap(size, f"Error:\n{column_name}")

    def type_for_tag(self, tag: str) -> ColumnType | None:
        """
        Returns the ColumnType registered under a built-in type tag, or
        None if the tag is unknown.

        The tag-keyed counterpart of get(). P1.8d-2a has AppController read
        a column's type tag off that table's TableSchema and ask the
        registry only what the tag renders as -- so two tables that share a
        column name no longer share one type. get() and the column-name map
        stay for the write-path callers P1.8d-2b removes.

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
        Renders a value using the renderer for a type tag, rather than for
        a column name. The tag-keyed counterpart of render().

        Carries render()'s placeholder behaviour exactly:
          * an unknown tag (or None) -> the "unknown" placeholder;
          * a None or NaN value      -> the "not computed" placeholder;
          * a renderer that raises    -> the "error" placeholder.

        `label` is used only in the placeholder message. AppController
        passes the column name it was asked about, so the three messages
        read exactly as render()'s did: "Unknown column: <name>",
        "Not computed: <name>", "Error: <name>". A caller with no column
        name -- AppController.render_result_image, rendering an operator's
        output file -- passes no label, and each message falls back to a
        plain sentence for a researcher that names no tag.

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

    def list_visual_columns(self) -> list[str]:
        """
        Returns the names of all registered columns whose type produces
        a visual output (visual=True). Used by the column selector in
        the gallery to show only renderable columns.

        Returns:
            List of column names with visual=True column types.
        """
        return [
            name for name, ct in self._columns.items()
            if ct.visual
        ]

    def list_all_columns(self) -> list[str]:
        """
        Returns the names of all registered columns regardless of type.
        Used by FilterPanel to know which columns are available.

        Returns:
            List of all registered column names.
        """
        return list(self._columns.keys())


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
