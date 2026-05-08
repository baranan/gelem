"""
ui/tiles/image_tile.py

ImageTile is a leaf tile that displays one column value for one row.

It calls controller.render_column_value(mode='thumbnail') and displays
the returned QPixmap. It does not know or care whether the column
contains an image path, a video path, a number, or a label — the
renderer handles all of that.

The ArtifactStore thumbnail cache is used automatically for media_path
columns: the renderer checks the cache first before loading from disk.
ImageTile no longer needs to know about ArtifactStore at all.

Student A is responsible for implementing this class.
"""

from __future__ import annotations
from ui.tiles.base_tile import BaseTile


class ImageTile(BaseTile):
    """
    A leaf tile displaying one column value for one row.

    Attributes:
        row_id:      The row this tile represents.
        column_name: The column whose value is displayed.
        controller:  The AppController, used to render values.
    """

    def __init__(
        self,
        row_id: str,
        column_name: str,
        controller,
    ):
        """
        Creates an ImageTile for one row and one column.

        Args:
            row_id:      The row_id of the item to display.
            column_name: The column name whose value to render.
            controller:  The AppController instance.
        """
        self.row_id      = row_id
        self.column_name = column_name
        self._controller = controller

        # Cache the last rendered pixmap so we do not re-render on
        # every paint event when nothing has changed.
        self._cached_pixmap = None
        self._cached_size   = None

    def _compose(self, pixmap, width: int, height: int):
        """
        Paints `pixmap` centered into a (width x height) white canvas,
        scaled with KeepAspectRatio. Returns the canvas QPixmap.
        """
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPainter, QPixmap

        canvas = QPixmap(width, height)
        canvas.fill(Qt.GlobalColor.white)
        if pixmap is None or pixmap.isNull():
            return canvas

        scaled = pixmap.scaled(
            width, height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (width - scaled.width()) // 2
        y = (height - scaled.height()) // 2
        painter = QPainter(canvas)
        painter.drawPixmap(x, y, scaled)
        painter.end()
        return canvas

    def render(self, width: int, height: int):
        """
        Renders the column value as a QPixmap at the given size.

        Calls controller.render_column_value(mode='thumbnail'), which
        routes through ColumnTypeRegistry to the appropriate renderer.
        The renderer handles caching, file loading, and thumbnail
        generation internally.

        Returns the cached pixmap if size has not changed since the
        last render call.

        Args:
            width:  The tile width in pixels.
            height: The tile height in pixels.

        Returns:
            A QPixmap, or None if rendering fails.
        """
        # Return cached result if nothing has changed.
        if self._cached_pixmap is not None and self._cached_size == (width, height):
            return self._cached_pixmap

        # Read the column value from the dataset via the controller.
        metadata = self._controller.get_row(self.row_id)
        value    = metadata.get(self.column_name)

        # The renderer takes a single size hint; use the larger dimension
        # so the source pixmap has enough resolution to fill the slot.
        size = max(width, height)

        # Ask the controller to render it. The renderer decides how —
        # thumbnail from ArtifactStore for media, formatted text for
        # numeric/text columns, etc.
        pixmap = self._controller.render_column_value(
            self.column_name, value, size, mode="thumbnail",
            context={"row_id": self.row_id, "column_name": self.column_name},
        )

        # Fit the rendered pixmap into a (width, height) canvas with
        # KeepAspectRatio so the tile exactly fills its slot.
        result = self._compose(pixmap, width, height)
        self._cached_pixmap = result
        self._cached_size   = (width, height)
        return result

    def get_row_ids(self) -> list[str]:
        """
        Returns the single row_id this tile represents.

        Returns:
            A list containing just self.row_id.
        """
        return [self.row_id]

    def get_children(self) -> list[BaseTile]:
        """
        Returns an empty list because ImageTile is a leaf tile.

        Returns:
            Empty list.
        """
        return []

    def invalidate_cache(self) -> None:
        """
        Clears the cached pixmap so the tile re-renders on next paint.
        Called by GalleryWidget when row_updated or thumbnail_ready
        signal is received for this tile's row_id.
        """
        self._cached_pixmap = None
        self._cached_size   = None

    def __repr__(self) -> str:
        return (
            f"ImageTile(row_id={self.row_id!r}, "
            f"column={self.column_name!r})"
        )
