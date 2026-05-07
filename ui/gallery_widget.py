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
    QLabel,
)
from PySide6.QtCore import Signal, Qt, QEvent

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
        pixmap = self._tile.render(self._size, self._size)
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
        """
        Handles single click — emits clicked signal with modifier keys.

        We call event.accept() so the press doesn't bubble up to the
        gallery's grid background, where the event filter would
        misinterpret it as a click on empty space and clear the
        selection.
        """
        from PySide6.QtCore import Qt
        modifiers = event.modifiers()
        ctrl_held  = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        shift_held = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        self.clicked.emit(self._tile.get_row_ids(), ctrl_held, shift_held)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        """Handles double click — emits double_clicked signal."""
        self.double_clicked.emit(self._tile.get_row_ids())
        event.accept()


class GalleryWidget(QWidget):
    """
    A scrollable grid of TileWidgets driven by a list of row_ids.

    Uses virtual scrolling: only tiles for the rows currently inside
    (or near) the viewport are mounted as widgets. As the user scrolls,
    tiles that leave the viewport are recycled into a free pool and
    later re-bound to newly visible rows. The inner content widget is
    sized to the full grid so the scrollbar reflects all rows even
    though most are not materialised.
    """

    # Signal emitted when the user selects one or more tiles.
    # Carries the list of selected row_ids.
    selection_changed = Signal(list)

    # Signal emitted when the user double-clicks a tile.
    tile_double_clicked = Signal(list)

    _SPACING     = 4   # gap between adjacent tiles, in pixels
    _MARGIN      = 4   # padding around the whole grid, in pixels
    _BUFFER_ROWS = 1   # rows above and below the viewport kept mounted

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

        # Virtual-scrolling state.
        # _mounted maps row_id index -> TileWidget currently shown.
        # _free_pool holds hidden TileWidgets ready to be recycled.
        self._mounted:   dict[int, TileWidget] = {}
        self._free_pool: list[TileWidget]      = []
        self._cols:      int = 1
        # Vertical gap between rows, recomputed in _relayout to flex
        # with the viewport height the same way QGridLayout's column
        # distribution flexes the horizontal gaps with width.
        self._v_gap:     int = self._SPACING

        # Index of the last plain- or ctrl-clicked tile in self._row_ids.
        # Acts as the anchor for shift+click range selection.
        self._last_clicked_index: int | None = None

        # Main layout.
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)

        # Scroll area. setWidgetResizable(True) lets the inner widget
        # match the viewport width, so QGridLayout distributes columns
        # across that width exactly the way the old non-virtual gallery
        # did.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._layout.addWidget(self._scroll)

        # Inner content widget. Layout is:
        #     [ top_spacer ]   <- height stands in for off-screen rows above
        #     [ grid_layout ]  <- real QGridLayout holding only visible tiles
        #     [ bot_spacer ]   <- height stands in for off-screen rows below
        # The spacers preserve scrollbar range and scroll position; the
        # QGridLayout still does the horizontal flow itself.
        self._grid_widget = QWidget()
        self._vbox = QVBoxLayout(self._grid_widget)
        self._vbox.setContentsMargins(0, 0, 0, 0)
        self._vbox.setSpacing(0)

        self._top_spacer = QWidget()
        self._top_spacer.setFixedHeight(0)
        self._vbox.addWidget(self._top_spacer)

        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(self._SPACING)
        self._vbox.addWidget(self._grid_container)

        self._bot_spacer = QWidget()
        self._bot_spacer.setFixedHeight(0)
        self._vbox.addWidget(self._bot_spacer)

        # An expanding stretch below the bottom spacer keeps the grid
        # pinned to the top when the content is shorter than the
        # viewport. With virtual scrolling at scale this is normally
        # not the case, but it matters for small datasets.
        self._vbox.addStretch(1)

        self._scroll.setWidget(self._grid_widget)

        # Clicks that land on the spacers, the grid background, or the
        # scroll-area viewport should clear the current selection —
        # same convention as Windows Explorer and macOS Finder. Tile
        # presses call event.accept() so they never reach this filter.
        self._grid_widget.installEventFilter(self)
        self._grid_container.installEventFilter(self)
        self._scroll.viewport().installEventFilter(self)
        self._scroll.verticalScrollBar().valueChanged.connect(
            self._update_visible_tiles
        )

    # ── Public API ────────────────────────────────────────────────────

    def set_row_ids(self, row_ids: list[str]) -> None:
        """
        Updates the gallery with a new ordered list of row_ids.
        Resets selection and scroll position, then re-mounts the
        visible viewport.

        Args:
            row_ids: Ordered list of row_ids to display.
        """
        self._row_ids = row_ids
        self._selected_ids.clear()
        self._last_clicked_index = None
        self._relayout()
        self._scroll.verticalScrollBar().setValue(0)

    def set_tile_size(self, size: int) -> None:
        """
        Updates the tile size and re-mounts the visible viewport.
        Called when the tile-size slider changes.

        Args:
            size: New tile size in pixels (50-600).
        """
        self._tile_size = size
        self._relayout()

    def set_visible_columns(self, column_names: list[str]) -> None:
        """
        Updates which columns are shown in each tile.
        One column -> ImageTile per item.
        Multiple columns -> GridTile per item.

        Args:
            column_names: Ordered list of column names to display.
        """
        self._visible_cols = column_names
        self._relayout()

    def on_row_updated(self, row_id: str) -> None:
        """
        Called when a row's data has changed (e.g. operator result
        arrived). Finds the mounted tile for this row, if any, and
        invalidates it so it re-renders with the new value. Tiles that
        are not currently mounted will pick up the new value naturally
        the next time they scroll into view.

        Args:
            row_id: The row whose data changed.
        """
        for tw in self._mounted.values():
            if row_id in tw._tile.get_row_ids():
                tw.invalidate()

    def on_thumbnail_ready(self, row_id: str) -> None:
        """
        Called when a thumbnail has been generated for row_id.
        Finds the corresponding mounted tile and repaints it.

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

    def _columns_per_row(self) -> int:
        """How many tiles fit horizontally in the current viewport."""
        viewport_w = self._scroll.viewport().width()
        if viewport_w <= 0:
            viewport_w = self.width() or 800
        # Account for the QGridLayout's default contents margins so we
        # don't over-estimate how many columns fit.
        m = self._grid_layout.contentsMargins()
        usable = viewport_w - m.left() - m.right() + self._SPACING
        return max(1, usable // (self._tile_size + self._SPACING))

    def _relayout(self) -> None:
        """
        Recomputes column count and vertical row spacing, returns every
        currently mounted tile to the free pool, and then re-mounts the
        tiles that fall inside the viewport. Called whenever the data
        set, tile size, visible columns, or viewport size change.
        """
        for tw in self._mounted.values():
            self._grid_layout.removeWidget(tw)
            tw.setParent(None)
            tw.hide()
            self._free_pool.append(tw)
        self._mounted.clear()

        self._cols  = self._columns_per_row()
        self._v_gap = self._vertical_gap()
        # Match QGridLayout's horizontal spacing (4 px between cells)
        # while letting the vertical spacing flex with viewport height.
        self._grid_layout.setHorizontalSpacing(self._SPACING)
        self._grid_layout.setVerticalSpacing(self._v_gap)
        self._update_visible_tiles()

    def _vertical_gap(self) -> int:
        """
        Vertical gap between rows, distributed the same way the QGridLayout
        distributes horizontal slack between columns: take the rows that
        fit in the viewport, and split any leftover height evenly across
        (rows + 1) slots.
        """
        viewport_h = self._scroll.viewport().height()
        if viewport_h <= 0:
            viewport_h = self.height() or 600
        rows_fit = max(1, viewport_h // (self._tile_size + self._SPACING))
        slack    = max(0, viewport_h - rows_fit * self._tile_size)
        return max(self._SPACING, slack // (rows_fit + 1))

    def _update_visible_tiles(self) -> None:
        """
        Computes which row_id indices fall inside (or near) the
        viewport, drops mounted tiles that are no longer needed back
        into the free pool, mounts tiles for newly-visible indices,
        and updates the top/bottom spacer heights so the scrollbar
        range and position stay correct.
        """
        n = len(self._row_ids)
        if n == 0:
            self._top_spacer.setFixedHeight(0)
            self._bot_spacer.setFixedHeight(0)
            return

        total_rows = (n + self._cols - 1) // self._cols
        row_h      = self._tile_size + self._v_gap

        viewport_top = self._scroll.verticalScrollBar().value()
        viewport_h   = self._scroll.viewport().height()

        top_row = max(
            0, ((viewport_top - self._v_gap) // row_h) - self._BUFFER_ROWS
        )
        bottom_row = min(
            total_rows - 1,
            ((viewport_top + viewport_h) // row_h) + self._BUFFER_ROWS,
        )

        first_idx = top_row * self._cols
        last_idx  = min(n - 1, (bottom_row + 1) * self._cols - 1)
        needed    = set(range(first_idx, last_idx + 1))

        if needed != set(self._mounted.keys()):
            # Drop tiles that left the visible window.
            for idx in list(self._mounted.keys()):
                if idx not in needed:
                    tw = self._mounted.pop(idx)
                    self._grid_layout.removeWidget(tw)
                    tw.setParent(None)
                    tw.hide()
                    self._free_pool.append(tw)

            columns = self._visible_cols or ["full_path"]

            # Mount tiles for newly-visible indices, placed at their
            # absolute (row, col) in the QGridLayout. Empty rows above
            # collapse to zero height; the top_spacer below provides
            # the visual offset.
            for idx in needed:
                if idx in self._mounted:
                    continue
                row_id = self._row_ids[idx]
                tile   = self._make_tile(row_id, columns)
                if self._free_pool:
                    tw = self._free_pool.pop()
                    tw.update_tile(tile, self._tile_size)
                    tw.setParent(self._grid_container)
                else:
                    tw = TileWidget(tile, self._tile_size,
                                    parent=self._grid_container)
                    tw.clicked.connect(self._on_tile_clicked)
                    tw.double_clicked.connect(self._on_tile_double_clicked)

                grid_row = idx // self._cols
                grid_col = idx %  self._cols
                self._grid_layout.addWidget(tw, grid_row, grid_col)
                tw.set_selected(row_id in self._selected_ids)
                tw.show()
                self._mounted[idx] = tw

        # Spacer heights stand in for the rows we did not mount, so the
        # scrollbar range matches the full virtual content. The extra
        # v_gap on each side is the very-top and very-bottom padding
        # (the analog of the side padding around the horizontal grid).
        self._top_spacer.setFixedHeight(self._v_gap + top_row * row_h)
        self._bot_spacer.setFixedHeight(
            max(0, (total_rows - bottom_row - 1) * row_h + self._v_gap)
        )

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
        - Plain click:  if the clicked tile is already in the selection,
                        leave the selection alone (only update the
                        anchor) so a follow-up double-click still sees
                        the multi-selection. Clicking an unselected
                        tile replaces the selection with just that one.
        - Ctrl+click:   toggle this tile in/out of the selection.
        - Shift+click:      replace the selection with every tile between
                            the anchor (last plain- or ctrl-clicked tile)
                            and this tile, inclusive. The anchor is not
                            moved, so repeated shift+clicks extend from
                            the same anchor. If no anchor exists yet,
                            falls back to plain click.
        - Ctrl+Shift+click: union the anchor-to-target range with the
                            existing selection, preserving prior picks.
                            The anchor stays put.
        """
        if not row_ids:
            return

        # Locate the clicked tile in self._row_ids. Tiles always group
        # by row_id (one row_id per tile for ImageTile, or a list of
        # ImageTiles for the same row_id in GridTile), so the first
        # row_id is enough to identify the tile's position.
        clicked_id = row_ids[0]
        try:
            clicked_idx = self._row_ids.index(clicked_id)
        except ValueError:
            clicked_idx = None

        have_anchor = (self._last_clicked_index is not None
                       and clicked_idx is not None)

        if shift_held and have_anchor:
            lo = min(self._last_clicked_index, clicked_idx)
            hi = max(self._last_clicked_index, clicked_idx)
            range_ids = set(self._row_ids[lo:hi + 1])
            if ctrl_held:
                # Ctrl+Shift — extend the existing selection with the range.
                self._selected_ids |= range_ids
            else:
                # Shift only — replace selection with just the range.
                self._selected_ids = range_ids
            # Anchor stays put so further shift+clicks extend from it.
        elif ctrl_held:
            for row_id in row_ids:
                if row_id in self._selected_ids:
                    self._selected_ids.discard(row_id)
                else:
                    self._selected_ids.add(row_id)
            self._last_clicked_index = clicked_idx
        else:
            # Plain click — if the clicked tile is already part of the
            # current selection, preserve the selection so a follow-up
            # double-click can open every selected item side-by-side.
            # Otherwise, replace the selection with just this tile.
            already_selected = any(
                r in self._selected_ids for r in row_ids
            )
            if not already_selected:
                self._selected_ids = set(row_ids)
            self._last_clicked_index = clicked_idx

        # Update visual selection state on currently mounted tiles.
        # Hidden (recycled) tiles will pick up the right state when
        # they next scroll into view.
        for tw in self._mounted.values():
            is_selected = any(
                r in self._selected_ids
                for r in tw._tile.get_row_ids()
            )
            tw.set_selected(is_selected)

        self.selection_changed.emit(list(self._selected_ids))

    def eventFilter(self, obj, event) -> bool:
        """
        Watches mouse presses on the grid background and the scroll
        viewport. A press on either (which means the user clicked
        between tiles, not on a TileWidget) clears the current
        selection. Tile clicks never reach this filter because
        TileWidget.mousePressEvent accepts them first.
        """
        if event.type() == QEvent.Type.MouseButtonPress and (
            obj is self._grid_widget
            or obj is self._grid_container
            or obj is self._scroll.viewport()
        ):
            if self._selected_ids:
                self._clear_selection()
        return super().eventFilter(obj, event)

    def _clear_selection(self) -> None:
        """
        Empties the selection, repaints affected tiles, and emits
        selection_changed. Anchor is also reset so the next plain-
        or ctrl-click starts a fresh range.
        """
        self._selected_ids.clear()
        self._last_clicked_index = None
        for tw in self._mounted.values():
            tw.set_selected(False)
        self.selection_changed.emit([])

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
        """
        Re-mounts visible tiles on viewport resize. Triggers a full
        relayout when the column count or vertical-gap changes (column
        count flexes the horizontal flow, vertical-gap flexes the row
        spacing); otherwise just refreshes the visible window.
        """
        super().resizeEvent(event)
        new_cols  = self._columns_per_row()
        new_v_gap = self._vertical_gap()
        if new_cols != self._cols or new_v_gap != self._v_gap:
            self._relayout()
        else:
            self._update_visible_tiles()