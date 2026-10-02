"""
ui/result_table_view.py

Layer B of the read-only table view: a QAbstractTableModel over the
controller's current result, and a widget wrapping a QTableView around it.
Layer A (ui/table_pages.py) holds the page cache and the group-label
lookup; this file is only the Qt glue.

What it shows is the same result the galleries show -- the active table
after the current filters, sort and grouping -- as one row per flat-order
position and one column per column of the table, in schema order. In
grouped mode a leftmost "Group" column carries each row's group label.

Rules this file keeps, because the result can have a million rows:

  * The model never holds the result. rowCount is the layout total, and
    a cell's text comes from a cached page of PAGE_SIZE rows, fetched
    synchronously on the main thread through AppController.get_cell_texts
    the first time a cell of that page is painted.
  * Every row is one fixed height. Nothing ever asks Qt to measure rows
    (no resizeRowsToContents) and there are no cell widgets.
  * Column widths are set once per result from a small sample of the first
    rows, never from the whole table.
  * The selection is read from the selection model's ranges, never by
    asking Qt for a list of every selected index (selectedIndexes() or
    selectedRows() would build one object per selected cell).
  * It is read-only, and clicking a header does not sort.

This widget never reports a displayed range to the controller: only the
galleries ask for thumbnails, and the table has none to ask for.
"""

from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ui.table_pages import GroupLabels, TablePageCache

# Header of the extra leftmost column shown in grouped mode.
GROUP_COLUMN_TITLE = "Group"

# One fixed height for every row, in pixels.
ROW_HEIGHT = 22

# Column width bounds and the number of leading rows sampled to pick a
# width, in pixels / rows. Padding covers the cell margins and, for the
# header, the room the title needs.
MIN_COLUMN_WIDTH = 60
MAX_COLUMN_WIDTH = 320
WIDTH_SAMPLE_ROWS = 50
WIDTH_PADDING = 24


class ResultTableModel(QAbstractTableModel):
    """Read-only model over the controller's current result."""

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._cache = TablePageCache()
        self._result_id: str = ""
        self._table_name: str = ""
        self._total: int = 0
        self._grouped: bool = False
        self._groups = GroupLabels(None)
        # Columns of the table, in schema order, without "Group".
        self._data_columns: tuple[str, ...] = ()

    # -- state ---------------------------------------------------------

    def reset_from_controller(self) -> None:
        """Re-reads the layout and the column list and drops every cached
        page. The one way the model learns that the result, the active
        table or the columns changed."""
        self.beginResetModel()
        layout = self._controller.get_result_layout()
        self._result_id = layout.result_id
        self._table_name = layout.table_name
        self._total = layout.total
        self._grouped = layout.groups is not None
        self._groups = GroupLabels(layout.groups)
        self._data_columns = tuple(
            self._controller.get_column_names(layout.table_name)
        )
        self._cache.reset()
        self.endResetModel()

    def on_rows_updated(self, row_ids) -> None:
        """Rows changed in place: drop only the cached pages holding any
        of them and tell the view which row ranges to repaint. No reset,
        so selection and scroll position survive."""
        last_column = self.columnCount() - 1
        if last_column < 0:
            return
        for start, stop in self._cache.invalidate_rows(row_ids):
            self.dataChanged.emit(
                self.index(start, 0), self.index(stop - 1, last_column)
            )

    @property
    def is_grouped(self) -> bool:
        return self._grouped

    # -- QAbstractTableModel -------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return self._total

    def columnCount(self, parent=QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._data_columns) + (1 if self._grouped else 0)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Vertical:
            return str(section + 1)
        if self._grouped:
            if section == 0:
                return GROUP_COLUMN_TITLE
            section -= 1
        if 0 <= section < len(self._data_columns):
            return self._data_columns[section]
        return None

    def flags(self, index) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        # Selectable but never editable.
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole or not index.isValid():
            return None
        row = index.row()
        column = index.column()
        if row >= self._total:
            return None

        if self._grouped:
            if column == 0:
                return self._groups.label_for(row)
            column -= 1

        text = self._cache.lookup(self._result_id, self._data_columns, row, column)
        if text is None:
            # Miss: fetch the page holding this row (main thread), then
            # read again. A page the controller refuses (stale result)
            # stays missing and the cell paints empty until the next
            # reset.
            self._fetch_page(self._cache.page_index_for(row))
            text = self._cache.lookup(
                self._result_id, self._data_columns, row, column
            )
        return text if text is not None else ""

    # -- paging --------------------------------------------------------

    def _fetch_page(self, page_index: int) -> None:
        """Fetches one page through the controller and caches it."""
        start, stop = self._cache.page_range(page_index)
        stop = min(stop, self._total)
        if start >= stop:
            return
        page = self._controller.get_cell_texts(
            self._table_name,
            self._result_id,
            start,
            stop,
            list(self._data_columns),
        )
        if page is None:
            return
        self._cache.put_page(
            self._result_id,
            self._data_columns,
            page_index,
            page.row_ids,
            page.texts,
        )


class ResultTableView(QWidget):
    """The table view as the centre area shows it."""

    # Emitted with the row id of a double-clicked row.
    row_double_clicked = Signal(str)
    # Emitted when the selection changes. Carries nothing: a selection of
    # a million rows must not be turned into a million ids on every click.
    selection_changed = Signal()

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        # Column widths are fitted once per result, and only while the
        # table is on screen; a refresh while hidden just marks them stale.
        self._widths_stale = True

        self._model = ResultTableModel(controller, self)
        self._view = QTableView(self)
        self._view.setModel(self._model)

        # Read-only, row selection, no sorting by header click.
        self._view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._view.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._view.setSortingEnabled(False)
        self._view.setWordWrap(False)
        self._view.setTextElideMode(Qt.TextElideMode.ElideRight)

        # One fixed row height; Qt never measures a row.
        vertical = self._view.verticalHeader()
        vertical.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        vertical.setDefaultSectionSize(ROW_HEIGHT)

        # Interactive column widths set by hand; a header click does
        # nothing (it would otherwise select a whole column).
        horizontal = self._view.horizontalHeader()
        horizontal.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        horizontal.setSectionsClickable(False)
        horizontal.setResizeContentsPrecision(0)
        horizontal.setStretchLastSection(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._view.doubleClicked.connect(self._on_double_clicked)
        self._view.selectionModel().selectionChanged.connect(
            lambda *_: self.selection_changed.emit()
        )

    # -- refresh -------------------------------------------------------

    def refresh(self) -> None:
        """The result, active table or columns changed: reset the model
        and refit the column widths (now if showing, else when shown)."""
        self._model.reset_from_controller()
        self._widths_stale = True
        if self.isVisible():
            self._fit_column_widths()

    def on_rows_updated(self, row_ids) -> None:
        """Rows of the active table changed in place."""
        self._model.on_rows_updated(row_ids)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._widths_stale:
            self._fit_column_widths()

    def _fit_column_widths(self) -> None:
        """Sets every column's width from its header text and the first
        WIDTH_SAMPLE_ROWS rows only, clamped to the width bounds."""
        self._widths_stale = False
        metrics = self._view.fontMetrics()
        sample = min(self._model.rowCount(), WIDTH_SAMPLE_ROWS)
        for column in range(self._model.columnCount()):
            header = self._model.headerData(
                column, Qt.Orientation.Horizontal
            )
            widest = metrics.horizontalAdvance(str(header or ""))
            for row in range(sample):
                text = self._model.data(self._model.index(row, column))
                widest = max(widest, metrics.horizontalAdvance(str(text or "")))
            width = max(MIN_COLUMN_WIDTH, min(MAX_COLUMN_WIDTH, widest + WIDTH_PADDING))
            self._view.setColumnWidth(column, width)

    # -- selection -----------------------------------------------------

    def _selected_row_spans(self) -> list[tuple[int, int]]:
        """The selected rows as sorted, merged half-open [start, stop)
        spans, read from the selection's ranges."""
        spans = sorted(
            (selected.top(), selected.bottom() + 1)
            for selected in self._view.selectionModel().selection()
        )
        merged: list[list[int]] = []
        for start, stop in spans:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        return [(start, stop) for start, stop in merged]

    def selected_count(self) -> int:
        """How many rows are selected, without fetching any row id."""
        return sum(stop - start for start, stop in self._selected_row_spans())

    def get_selected_row_ids(self) -> list[str]:
        """The selected rows' ids, in flat order."""
        row_ids: list[str] = []
        for start, stop in self._selected_row_spans():
            row_ids.extend(self._controller.get_row_ids_in_range(start, stop))
        return row_ids

    def clear_selection(self) -> None:
        self._view.clearSelection()

    # -- double click --------------------------------------------------

    def _on_double_clicked(self, index) -> None:
        if not index.isValid():
            return
        row_ids = self._controller.get_row_ids_in_range(
            index.row(), index.row() + 1
        )
        if row_ids:
            self.row_double_clicked.emit(row_ids[0])

    # -- test seams ----------------------------------------------------

    @property
    def model(self) -> ResultTableModel:
        return self._model

    @property
    def view(self) -> QTableView:
        return self._view
