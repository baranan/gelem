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

    def infer_type(self, series: pd.Series) -> str:
        """
        Inspects a pandas Series and returns a best-guess column type
        tag. Used by Dataset.merge_csv() to automatically register
        column types for CSV columns.

        Rules applied in order:
            1. Boolean dtype       -> 'boolean_flag'
            2. Numeric dtype       -> 'numeric'
            3. String values that all end in a known media extension
                                   -> 'media_path'
            4. Everything else     -> 'text'

        Note: datetime columns are now inferred as 'numeric' since
        timestamps and durations are both numeric values with
        formatting handled by the renderer.

        Args:
            series: The column data to inspect.

        Returns:
            A column type tag string.
        """
        if pd.api.types.is_bool_dtype(series):
            return "boolean_flag"

        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        # Check if values look like media file paths.
        media_extensions = {
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
            ".mp4", ".mov", ".avi", ".mkv", ".webm",
        }
        non_null = series.dropna().astype(str)
        if len(non_null) > 0:
            looks_like_paths = all(
                any(v.lower().endswith(ext) for ext in media_extensions)
                for v in non_null
            )
            if looks_like_paths:
                return "media_path"

        # Default: treat as text.
        return "text"


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
