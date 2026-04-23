"""
ui/tiles/base_tile.py

Defines the BaseTile interface that all tile types must implement.

A tile is one cell in the gallery grid. It knows how to:
    - render itself as a QPixmap at a given size
    - report which row_ids it contains
    - respond to a double-click

There are two concrete tile types:
    ImageTile  — displays one column value for one row
    GridTile   — displays multiple ImageTiles arranged side by side

Because both implement the same interface, GalleryWidget does not need
to know which kind it is working with. It just calls render(), and
whatever comes back is displayed.

Student A is responsible for implementing ImageTile and GridTile.
This base class is written centrally.
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class BaseTile(ABC):
    """
    Abstract base class defining the interface all tiles must implement.

    In Python, ABC means Abstract Base Class. A class that inherits from
    ABC and has @abstractmethod methods cannot be instantiated directly —
    you must create a subclass that implements all abstract methods.
    This is Python's equivalent of a pure virtual class in C++.
    """

    @abstractmethod
    def render(self, width: int, height: int):
        """
        Produces a QPixmap representing this tile at the given size.
        Called by GalleryWidget whenever a tile needs to be painted.

        For ImageTile: renders one column value via ColumnTypeRegistry.
        For GridTile:  renders all children side by side and composites
                       them into one QPixmap.

        Args:
            width:  The width of the tile in pixels.
            height: The height of the tile in pixels.
                    The rendered image should fit within width x height.

        Returns:
            A QPixmap ready for display, or None if rendering fails.
        """
        ...

    @abstractmethod
    def get_row_ids(self) -> list[str]:
        """
        Returns all row_ids contained within this tile.

        For ImageTile: returns [self.row_id] — a list with one item.
        For GridTile:  returns the combined row_ids of all children.
                       If all children show the same row (the common case),
                       this still returns a list with one unique row_id.

        Used by GalleryWidget to know which rows are selected when the
        user clicks or shift-clicks tiles.

        Returns:
            List of row_id strings.
        """
        ...

    @abstractmethod
    def get_children(self) -> list['BaseTile']:
        """
        Returns the child tiles contained within this tile.

        For ImageTile: returns [] (no children — it is a leaf tile).
        For GridTile:  returns the list of child BaseTile objects.

        Used by DetailWidget to know how to display a tile's content
        when the user double-clicks it.

        Returns:
            List of child BaseTile objects. Empty list for leaf tiles.
        """
        ...

    def on_double_click(self, detail_widget) -> None:
        """
        Called when the user double-clicks this tile.
        Opens the tile's content in the detail view.

        Default implementation tells DetailWidget to show all row_ids
        in this tile. Subclasses may override for custom behaviour.

        Args:
            detail_widget: The DetailWidget instance to show content in.
        """
        detail_widget.show_rows(self.get_row_ids())

    def is_leaf(self) -> bool:
        """
        Returns True if this tile has no children (i.e. is an ImageTile).
        Convenience method so callers do not need to check get_children().

        Returns:
            True if get_children() returns an empty list.
        """
        return len(self.get_children()) == 0