"""
tests/test_table_view_widget.py

The table view as a user meets it: a "Gallery | Table" switch in
MainWindow's centre area, a read-only virtual table of the same result,
and selection that belongs to whichever view is showing.

Property-style: nothing here depends on a pixel size or a tile count.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from PySide6.QtCore import QItemSelection, QItemSelectionModel, Qt
from PySide6.QtTest import QTest

from column_types.text_format import format_cell_text
from models.notifications import RowsUpdated
from ui.gallery_widget import TileWidget
from ui.main_window import MainWindow


def _window(make_controller, realize_widget, qapp, tmp_path):
    controller, dataset, _ = make_controller(tmp_path, merge_csv=True)
    window = MainWindow(controller)
    realize_widget(window, width=1300, height=800)
    controller.set_filters([])
    qapp.processEvents()
    return window, controller, dataset


def _select_rows(window, first, last):
    model = window._table_view.model
    selection = QItemSelection(model.index(first, 0), model.index(last, 0))
    window._table_view.view.selectionModel().select(
        selection,
        QItemSelectionModel.SelectionFlag.ClearAndSelect
        | QItemSelectionModel.SelectionFlag.Rows,
    )


def test_row_count_equals_the_layout_total(make_controller, realize_widget, qapp, tmp_path):
    window, controller, _ = _window(make_controller, realize_widget, qapp, tmp_path)
    model = window._table_view.model
    assert model.rowCount() == controller.get_result_layout().total > 0

    controller.set_filters([], sort_by="timestamp", ascending=False)
    assert model.rowCount() == controller.get_result_layout().total


def test_columns_are_the_schema_columns_in_order(make_controller, realize_widget, qapp, tmp_path):
    window, controller, _ = _window(make_controller, realize_widget, qapp, tmp_path)
    model = window._table_view.model
    names = controller.get_column_names()
    assert "row_id" not in names
    assert model.columnCount() == len(names)
    headers = [
        model.headerData(c, Qt.Orientation.Horizontal) for c in range(model.columnCount())
    ]
    assert headers == names


def test_a_cell_shows_the_formatters_text(make_controller, realize_widget, qapp, tmp_path):
    window, controller, dataset = _window(make_controller, realize_widget, qapp, tmp_path)
    model = window._table_view.model
    names = controller.get_column_names()
    schema = dataset.schema_for("frames")
    row_ids = controller.get_row_ids_in_range(0, controller.get_result_layout().total)

    for row in (0, 7, len(row_ids) - 1):
        stored = dataset.get_row(row_ids[row], "frames")
        for column, name in enumerate(names):
            expected = format_cell_text(stored[name], schema.spec_for(name).type_tag)
            assert model.data(model.index(row, column)) == expected


def test_grouped_mode_adds_a_leftmost_group_column(make_controller, realize_widget, qapp, tmp_path):
    window, controller, dataset = _window(make_controller, realize_widget, qapp, tmp_path)
    model = window._table_view.model
    flat_columns = model.columnCount()
    assert not model.is_grouped

    controller.set_group_by("condition")
    qapp.processEvents()

    layout = controller.get_result_layout()
    assert layout.groups
    assert model.is_grouped
    assert model.columnCount() == flat_columns + 1
    assert model.headerData(0, Qt.Orientation.Horizontal) == "Group"
    assert model.headerData(1, Qt.Orientation.Horizontal) == controller.get_column_names()[0]
    # Every row's group cell is its section's label, at both ends of each span.
    for section in layout.groups:
        for row in (section.start, section.stop - 1):
            assert model.data(model.index(row, 0)) == section.label

    controller.set_group_by(None)
    qapp.processEvents()
    assert not model.is_grouped
    assert model.columnCount() == flat_columns


def test_the_table_is_read_only_and_header_click_does_not_sort(
    make_controller, realize_widget, qapp, tmp_path
):
    window, _, _ = _window(make_controller, realize_widget, qapp, tmp_path)
    view = window._table_view.view
    model = window._table_view.model
    assert not (model.flags(model.index(0, 0)) & Qt.ItemFlag.ItemIsEditable)
    assert view.editTriggers() == view.EditTrigger.NoEditTriggers
    assert view.isSortingEnabled() is False


def test_double_click_emits_the_row_id(make_controller, realize_widget, qapp, tmp_path):
    window, controller, _ = _window(make_controller, realize_widget, qapp, tmp_path)
    got = []
    window._table_view.row_double_clicked.connect(got.append)

    index = window._table_view.model.index(3, 1)
    window._table_view.view.doubleClicked.emit(index)

    assert got == [controller.get_row_ids_in_range(3, 4)[0]]


def test_double_click_opens_detail_on_that_row(make_controller, realize_widget, qapp, tmp_path):
    window, controller, _ = _window(make_controller, realize_widget, qapp, tmp_path)
    selected = []
    controller.row_selected.connect(lambda metadata: selected.append(metadata["row_id"]))
    window._table_button.click()

    window._table_view.view.doubleClicked.emit(window._table_view.model.index(4, 0))

    assert selected == [controller.get_row_ids_in_range(4, 5)[0]]
    assert window._right_tabs.currentWidget() is window._detail_widget


def test_with_the_table_showing_no_gallery_reports_a_displayed_range(
    make_controller, realize_widget, qapp, tmp_path
):
    window, controller, _ = _window(make_controller, realize_widget, qapp, tmp_path)
    # The gallery is showing: it asks for its tiles.
    assert controller.get_displayed_ranges() != []

    window._table_button.click()
    qapp.processEvents()
    assert controller.get_displayed_ranges() == []

    # A new result while the table is showing re-points the galleries but
    # they must still report nothing.
    controller.set_filters([], sort_by="timestamp", ascending=False)
    qapp.processEvents()
    controller.set_group_by("condition")
    qapp.processEvents()
    assert controller.get_displayed_ranges() == []
    controller.set_group_by(None)
    qapp.processEvents()
    assert controller.get_displayed_ranges() == []

    # Back to the gallery: demand returns.
    window._gallery_button.click()
    qapp.processEvents()
    assert controller.get_displayed_ranges() != []


def test_the_selected_count_label_counts_the_tables_selection(
    make_controller, realize_widget, qapp, tmp_path
):
    window, controller, _ = _window(make_controller, realize_widget, qapp, tmp_path)
    total = controller.get_result_layout().total
    window._table_button.click()
    qapp.processEvents()
    assert window._selection_label.text() == f"{total} items"

    _select_rows(window, 1, 5)
    qapp.processEvents()

    assert window._selection_label.text() == f"5 of {total} selected"
    assert window._collect_selected_row_ids() == controller.get_row_ids_in_range(1, 6)
    # The "visible" scope is the same flat order whichever view is up.
    assert window._collect_visible_row_ids() == controller.get_visible_row_ids()


def test_switching_views_clears_the_selection_in_both(
    make_controller, realize_widget, qapp, tmp_path
):
    window, controller, _ = _window(make_controller, realize_widget, qapp, tmp_path)
    total = controller.get_result_layout().total

    # Select a tile in the gallery.
    tile = window._main_gallery.findChildren(TileWidget)[0]
    QTest.mouseClick(tile, Qt.MouseButton.LeftButton)
    assert window._main_gallery.get_selected_row_ids() != []

    window._table_button.click()
    qapp.processEvents()
    assert window._main_gallery.get_selected_row_ids() == []
    assert window._selection_label.text() == f"{total} items"

    _select_rows(window, 0, 2)
    assert window._table_view.selected_count() == 3

    window._gallery_button.click()
    qapp.processEvents()
    assert window._table_view.selected_count() == 0
    assert window._main_gallery.get_selected_row_ids() == []
    assert window._selection_label.text() == f"{total} items"


def test_rows_updated_repaints_without_a_reset(make_controller, realize_widget, qapp, tmp_path):
    window, controller, dataset = _window(make_controller, realize_widget, qapp, tmp_path)
    window._table_button.click()
    qapp.processEvents()
    model = window._table_view.model
    model.data(model.index(0, 0))  # caches page 0

    resets, changed = [], []
    model.modelReset.connect(lambda: resets.append(1))
    model.dataChanged.connect(lambda top, bottom: changed.append((top.row(), bottom.row())))

    row_id = controller.get_row_ids_in_range(2, 3)[0]
    controller.rows_updated.emit(RowsUpdated("frames", (row_id,)))

    assert resets == []
    assert changed and changed[0][0] <= 2 <= changed[0][1]

    # A notice for some other table changes nothing.
    changed.clear()
    controller.rows_updated.emit(RowsUpdated("other_table", (row_id,)))
    assert changed == []


def test_result_and_table_changes_reset_the_model(make_controller, realize_widget, qapp, tmp_path):
    window, controller, _ = _window(make_controller, realize_widget, qapp, tmp_path)
    model = window._table_view.model
    resets = []
    model.modelReset.connect(lambda: resets.append(1))

    controller.set_filters([], sort_by="timestamp", ascending=False)
    assert resets  # result_changed

    resets.clear()
    controller.columns_updated.emit(controller.get_column_names())
    assert resets  # columns_updated

    resets.clear()
    controller.project_loaded.emit()
    assert resets  # project_loaded

    resets.clear()
    controller.active_table_changed.emit("frames")
    assert resets  # active_table_changed
