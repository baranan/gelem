"""
ui/gallery_widget.py

GalleryWidget is a virtual-scrolling grid of tiles. It receives an
ordered list of row_ids and renders only the tiles currently visible
on screen.

Virtual scrolling means the widget never creates more tile objects than
fit on screen at once. As the user scrolls, tiles that leave the viewport
are recycled and refilled with new content. This allows the gallery to
handle tens of thousands of items smoothly.

For the grouped gallery view, MainWindow creates multiple GalleryWidgets
(one per group) inside a shared QScrollArea. All share the same tile size.

Student A is responsible for implementing this class.

Key Qt concepts used here:
    QScrollArea:  A scrollable container widget.
    QWidget:      The base class for all Qt widgets.
    QGridLayout:  Arranges child widgets in a grid.
    QPainter:     Used to draw content on a widget.
    Signal:       A Qt mechanism for notifying other objects of events.
"""

from __future__ import annotations
from PySide6.QtWidgets import (
    QWidget, QScrollArea, QVBoxLayout, QGridLayout,
    QLabel, QSizePolicy
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap

from ui.tiles.image_tile import ImageTile
from ui.tiles.grid_tile import GridTile
from ui.tiles.base_tile import BaseTile


class TileWidget(QLabel):
    """
    A single cell in the gallery grid. Wraps a BaseTile and handles
    painting, clicking, and double-clicking.

    Inherits from QLabel because QLabel can display a QPixmap directly,
    which is the simplest way to show tile images in Qt.
    """

    # Signal emitted when this tile is single-clicked.
    # Carries the list of row_ids in this tile.
    clicked = Signal(list, bool, bool)  # row_ids, ctrl_held, shift_held

    # Signal emitted when this tile is double-clicked.
    double_clicked = Signal(list)

    def __init__(self, tile: BaseTile, size: int, parent=None):
        """
        Creates a TileWidget wrapping the given BaseTile.

        Args:
            tile:   The BaseTile to display.
            size:   The tile size in pixels.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._tile     = tile
        self._size     = size
        self._selected = False

        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._repaint()

    def _repaint(self) -> None:
        """
        Asks the tile to render itself and displays the result.
        Shows a gray placeholder if rendering returns None.
        """
        pixmap = self._tile.render(self._size)
        if pixmap is not None:
            self.setPixmap(
                pixmap.scaled(
                    self._size, self._size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            # Reset stylesheet first, then apply selection if needed.
            self.setStyleSheet("")
        else:
            self.setText("Loading...")
            self.setStyleSheet("background-color: #CCCCCC; color: #666666;")

        # Apply selection border on top of whatever style is set.
        if self._selected:
            self.setStyleSheet(
                "border: 3px solid #2E5F8A; background-color: #EBF1F8;"
            )

    def update_tile(self, tile: BaseTile, size: int) -> None:
        """
        Replaces the tile content and re-renders.
        Used for tile recycling during virtual scrolling.

        Args:
            tile: The new BaseTile to display.
            size: The new tile size.
        """
        self._tile = tile
        self._size = size
        self.setFixedSize(size, size)
        self._repaint()

    def set_selected(self, selected: bool) -> None:
        """
        Updates the visual selection state of this tile.

        Args:
            selected: True to show as selected, False to deselect.
        """
        self._selected = selected
        self._repaint()

    def invalidate(self) -> None:
        """
        Clears the tile's cache and re-renders.
        Called when row_updated signal arrives for this tile's row.
        """
        if hasattr(self._tile, "invalidate_cache"):
            self._tile.invalidate_cache()
        self._repaint()

    def mousePressEvent(self, event) -> None:
        """Handles single click — emits clicked signal with modifier keys."""
        from PySide6.QtCore import Qt
        modifiers = event.modifiers()
        ctrl_held  = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        shift_held = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        self.clicked.emit(self._tile.get_row_ids(), ctrl_held, shift_held)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        """Handles double click — emits double_clicked signal."""
        self.double_clicked.emit(self._tile.get_row_ids())
        super().mouseDoubleClickEvent(event)


class GalleryWidget(QWidget):
    """
    A scrollable grid of TileWidgets driven by a list of row_ids.

    Receives row_ids from AppController via gallery_updated signal.
    Builds tiles using the currently selected visible columns.
    Handles tile selection and communicates selections back to the
    controller.

    TODO (Student A): Implement virtual scrolling so only visible
    tiles are rendered at any time. The current placeholder
    implementation renders all tiles at once, which will be slow
    for large datasets.
    """

    # Signal emitted when the user selects one or more tiles.
    # Carries the list of selected row_ids.
    selection_changed = Signal(list)

    # Signal emitted when the user double-clicks a tile.
    tile_double_clicked = Signal(list)

    def __init__(self, controller, parent=None):
        """
        Creates the GalleryWidget.

        Args:
            controller: The AppController instance.
            parent:     Optional parent widget.
        """
        super().__init__(parent)
        self._controller    = controller
        self._row_ids:      list[str] = []
        self._tile_size:    int       = 150
        self._visible_cols: list[str] = []
        self._selected_ids: set[str]  = set()
        self._tile_widgets: list[TileWidget] = []

        # Main layout.
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)

        # Scroll area containing the grid.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._layout.addWidget(self._scroll)

        # Grid container widget.
        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(4)
        self._scroll.setWidget(self._grid_widget)

    # ── Public API ────────────────────────────────────────────────────

    def set_row_ids(self, row_ids: list[str]) -> None:
        """
        Updates the gallery with a new ordered list of row_ids.
        Rebuilds all tiles. Called when gallery_updated signal arrives.

        Args:
            row_ids: Ordered list of row_ids to display.

        TODO (Student A): Replace this with virtual scrolling.
        Currently rebuilds all tiles at once.
        """
        self._row_ids = row_ids
        self._selected_ids.clear()
        self._rebuild_tiles()

    def set_tile_size(self, size: int) -> None:
        """
        Updates the tile size and repaints all tiles.
        Called when the tile-size slider changes.

        Args:
            size: New tile size in pixels (50-600).
        """
        self._tile_size = size
        self._rebuild_tiles()

    def set_visible_columns(self, column_names: list[str]) -> None:
        """
        Updates which columns are shown in each tile.
        One column -> ImageTile per item.
        Multiple columns -> GridTile per item.

        Args:
            column_names: Ordered list of column names to display.
        """
        self._visible_cols = column_names
        self._rebuild_tiles()

    def on_row_updated(self, row_id: str) -> None:
        """
        Called when a row's data has changed (e.g. operator result
        arrived). Finds the tile for this row and invalidates it
        so it re-renders with the new value.

        Args:
            row_id: The row whose data changed.
        """
        for tw in self._tile_widgets:
            if row_id in tw._tile.get_row_ids():
                tw.invalidate()

    def on_thumbnail_ready(self, row_id: str) -> None:
        """
        Called when a thumbnail has been generated for row_id.
        Finds the corresponding tile and repaints it.

        Args:
            row_id: The item whose thumbnail is now available.
        """
        self.on_row_updated(row_id)

    def get_selected_row_ids(self) -> list[str]:
        """
        Returns the currently selected row_ids.

        Returns:
            List of selected row_id strings.
        """
        return list(self._selected_ids)

    # ── Internal helpers ──────────────────────────────────────────────

    def _rebuild_tiles(self) -> None:
        """
        Clears the grid and rebuilds all tile widgets from scratch.

        TODO (Student A): Replace with virtual scrolling that only
        creates tiles for the visible viewport area.
        """
        # Clear existing tiles.
        for tw in self._tile_widgets:
            self._grid_layout.removeWidget(tw)
            tw.deleteLater()
        self._tile_widgets.clear()

        if not self._row_ids:
            return

        # Determine how many columns fit in the available width.
        available_width = self.width() or 800
        cols = max(1, available_width // (self._tile_size + 4))

        # Use full_path as default column if none selected.
        columns = self._visible_cols or ["full_path"]

        # Build and place tiles.
        for i, row_id in enumerate(self._row_ids):
            tile = self._make_tile(row_id, columns)
            tw   = TileWidget(tile, self._tile_size)
            tw.clicked.connect(self._on_tile_clicked)
            tw.double_clicked.connect(self._on_tile_double_clicked)

            row = i // cols
            col = i % cols
            self._grid_layout.addWidget(tw, row, col)
            self._tile_widgets.append(tw)

    def _make_tile(self, row_id: str, columns: list[str]) -> BaseTile:
        """
        Creates the appropriate tile type for a row and column list.
        One column -> ImageTile.
        Multiple columns -> GridTile containing ImageTiles.

        Args:
            row_id:  The row to create a tile for.
            columns: The columns to display.

        Returns:
            A BaseTile (either ImageTile or GridTile).
        """
        if len(columns) == 1:
            return ImageTile(row_id, columns[0], self._controller)

        children = [
            ImageTile(row_id, col, self._controller)
            for col in columns
        ]
        return GridTile(children, direction="horizontal")

    def _on_tile_clicked(self, 
                         row_ids: list[str], 
                         ctrl_held: bool = False, 
                         shift_held: bool = False) -> None:
        """
        Handles tile selection.
        - Plain click:  select only this tile, deselect all others.
        - Ctrl+click:   toggle this tile in/out of the selection.
        - Shift+click:  TODO (Student A): select all tiles between the
                        last clicked tile and this one. Requires tracking
                        the index of the last clicked tile in self._last_clicked_index.
                        Then select all row_ids between that index and the
                        current tile's index in self._row_ids.
        """
        if shift_held and not ctrl_held:
            # TODO (Student A): implement range selection.
            # For now, fall through to plain click behaviour.
            pass

        if ctrl_held:
            # Toggle this tile in or out of the existing selection.
            for row_id in row_ids:
                if row_id in self._selected_ids:
                    self._selected_ids.discard(row_id)
                else:
                    self._selected_ids.add(row_id)
        else:
            # Plain click — replace selection with just this tile.
            self._selected_ids.clear()
            for row_id in row_ids:
                self._selected_ids.add(row_id)

        # Update visual selection state on all tiles.
        for tw in self._tile_widgets:
            is_selected = any(
                r in self._selected_ids
                for r in tw._tile.get_row_ids()
            )
            tw.set_selected(is_selected)

        self.selection_changed.emit(list(self._selected_ids))

    def _on_tile_double_clicked(self, row_ids: list[str]) -> None:
        """
        Handles tile double-click. Notifies the controller and emits
        tile_double_clicked signal.

        Args:
            row_ids: The row_ids in the double-clicked tile.
        """
        if row_ids:
            self._controller.select_row(row_ids[0])
        self.tile_double_clicked.emit(row_ids)

    def resizeEvent(self, event) -> None:
        """Rebuilds tiles when the widget is resized (column count changes)."""
        super().resizeEvent(event)
        self._rebuild_tiles()