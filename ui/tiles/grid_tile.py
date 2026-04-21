"""
ui/tiles/grid_tile.py

GridTile is a composite tile that contains multiple child BaseTiles
arranged side by side (horizontally) or stacked (vertically).

The most common use case is showing two visual columns for one item:
    GridTile
        ImageTile(row_id='000001', column='full_path')
        ImageTile(row_id='000001', column='avatar_path')

GridTile renders each child at a proportionally smaller size and
composites them into one QPixmap. The researcher sees both images
within a single gallery cell.

GridTile can also contain other GridTiles, allowing nested layouts.

Student A is responsible for implementing this class.
"""

from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap
from ui.tiles.base_tile import BaseTile


class GridTile(BaseTile):
    """
    A composite tile containing multiple child BaseTiles.

    Children are arranged either horizontally (side by side) or
    vertically (stacked). The default is horizontal.

    Attributes:
        children:   The list of child BaseTile objects.
        direction:  'horizontal' or 'vertical'.
        controller: The AppController instance.
    """

    def __init__(
        self,
        children: list[BaseTile],
        direction: str = "horizontal",
        controller=None,
    ):
        """
        Creates a GridTile with the given children.

        Args:
            children:   List of BaseTile objects to display.
                        Must have at least one child.
            direction:  'horizontal' — children side by side (default).
                        'vertical'   — children stacked top to bottom.
            controller: The AppController instance. May be None if all
                        children already have their own controller
                        references.
        """
        if not children:
            raise ValueError("GridTile must have at least one child.")

        self._children   = children
        self._direction  = direction
        self._controller = controller

        # Cache the last rendered pixmap.
        self._cached_pixmap = None
        self._cached_size   = None

    def render(self, size: int):
        """
        Renders all children at proportionally smaller sizes and
        composites them into a single QPixmap.

        For horizontal layout with N children, each child gets
        size // N pixels of width and the full size as height.
        For vertical layout, each child gets the full size as width
        and size // N pixels of height.

        Args:
            size: The total tile size in pixels (both width and height).

        Returns:
            A QPixmap containing all children composited together,
            or None if rendering fails.

        TODO (Student A): Implement this method.

        Suggested approach using QPainter:

            1. Return cached pixmap if size has not changed.

            2. Calculate child size:
               n = len(self._children)
               child_size = size // n

            3. Create a blank QPixmap of size x size:
               from PySide6.QtGui import QPixmap
               result = QPixmap(size, size)
               result.fill(Qt.GlobalColor.white)

            4. Create a QPainter on the result pixmap:
               from PySide6.QtGui import QPainter
               painter = QPainter(result)

            5. For each child, render it and draw it at the right offset:
               for i, child in enumerate(self._children):
                   child_pixmap = child.render(child_size)
                   if child_pixmap is None:
                       continue
                   if self._direction == 'horizontal':
                       x_offset = i * child_size
                       painter.drawPixmap(x_offset, 0, child_pixmap)
                   else:
                       y_offset = i * child_size
                       painter.drawPixmap(0, y_offset, child_pixmap)

            6. End the painter and cache the result:
               painter.end()
               self._cached_pixmap = result
               self._cached_size   = size
               return result
        """
        if self._cached_pixmap is not None and self._cached_size == size:
            return self._cached_pixmap

        n = len(self._children)
        child_size = size // n

        result = QPixmap(size, size)
        result.fill(Qt.GlobalColor.white)

        painter = QPainter(result)
        for i, child in enumerate(self._children):
            child_pixmap = child.render(child_size)
            if child_pixmap is None:
                continue
            if self._direction == "horizontal":
                painter.drawPixmap(i * child_size, 0, child_pixmap)
            else:
                painter.drawPixmap(0, i * child_size, child_pixmap)
        painter.end()

        self._cached_pixmap = result
        self._cached_size = size
        return result

    def get_row_ids(self) -> list[str]:
        """
        Returns all unique row_ids contained across all children.
        Preserves order but removes duplicates.

        For the common case where all children show the same row,
        returns a list with just that one row_id.

        Returns:
            Ordered list of unique row_id strings.
        """
        seen = set()
        result = []
        for child in self._children:
            for row_id in child.get_row_ids():
                if row_id not in seen:
                    seen.add(row_id)
                    result.append(row_id)
        return result

    def get_children(self) -> list[BaseTile]:
        """
        Returns the list of child BaseTile objects.

        Returns:
            List of child tiles.
        """
        return list(self._children)

    def on_double_click(self, detail_widget) -> None:
        """
        Opens all children's content in the detail view side by side.

        Args:
            detail_widget: The DetailWidget instance.
        """
        detail_widget.show_rows(self.get_row_ids())

    def invalidate_cache(self) -> None:
        """
        Clears this tile's cache and all children's caches.
        Called when row_updated signal arrives for any row in this tile.
        """
        self._cached_pixmap = None
        self._cached_size   = None
        for child in self._children:
            if hasattr(child, "invalidate_cache"):
                child.invalidate_cache()

    def add_child(self, tile: BaseTile) -> None:
        """
        Adds a new child tile to this GridTile.
        Invalidates the cache so the tile re-renders.

        Args:
            tile: The BaseTile to add.
        """
        self._children.append(tile)
        self.invalidate_cache()

    def __repr__(self) -> str:
        return (
            f"GridTile(children={len(self._children)}, "
            f"direction={self._direction!r})"
        )