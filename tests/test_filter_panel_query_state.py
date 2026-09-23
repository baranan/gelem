"""
tests/test_filter_panel_query_state.py

CC-37: FilterPanel must SHOW the query state AppController is holding for
the active table after refresh_columns() rebuilds its controls -- not
keep a second, independent record of "what is filtered" that only ever
advances from a click and never re-syncs from the controller.

refresh_columns() is called by MainWindow._on_columns_updated() whenever
AppController.columns_updated fires -- on a table switch and on every
load path. This module calls it directly with the same argument
MainWindow would pass (controller.get_column_names()), which is the real
seam under test.

Written from the work-item spec: each test is designed to fail against
the pre-CC-37 FilterPanel, which rebuilt every control to "no filter" on
refresh_columns() and never read anything back from the controller.

Isolated: imports PySide6 directly -- see run_tests.py's WIDGET_MODULES.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QPushButton

from models.query_engine import Filter
from table_display import QueryState
from ui.filter_panel import FilterPanel

# tests/conftest.py's `qapp` fixture supplies the one process-wide
# QApplication these tests need, both to build QWidgets at all and to
# process the deferred deleteLater() calls refresh_columns() issues for
# a table's old controls -- see _refresh(), which every test below calls
# instead of FilterPanel.refresh_columns() directly.

project_root = Path(__file__).parent.parent
TEST_IMAGES  = project_root / "test_images"


def _refresh(panel: FilterPanel, controller, qapp) -> None:
    """
    Calls refresh_columns() the way MainWindow._on_columns_updated() does,
    then flushes the PREVIOUS table's controls -- scheduled for
    destruction with deleteLater(), not destroyed on the spot -- before a
    test inspects the widget tree. A plain processEvents() call was not
    enough to observe this in the offscreen Qt platform this suite runs
    under (tests/conftest.py): a DeferredDelete event is only guaranteed
    delivered by explicitly asking for it. Without this, a stale,
    still-checked button from the table just left is still a child of
    the panel and findChildren() still sees it.
    """
    panel.refresh_columns(controller.get_column_names())
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def _checked_toggle_texts(panel: FilterPanel) -> list[str]:
    """Text of every checked, checkable QPushButton in the panel -- i.e.
    the toggle-filter buttons the researcher would actually see checked.
    Found through Qt's own widget tree (QWidget.findChildren), not a
    private attribute, the same way tests/test_settings_dialog.py inspects
    a dialog's controls."""
    return [
        btn.text()
        for btn in panel.findChildren(QPushButton)
        if btn.isCheckable() and btn.isChecked()
    ]


def _group_by_combo(panel: FilterPanel) -> QComboBox:
    """The panel's single group-by combo. Safe to assume there is exactly
    one QComboBox as long as the active table has no boolean column (a
    boolean filter is also a QComboBox) -- true for every table built in
    this module."""
    combos = panel.findChildren(QComboBox)
    assert len(combos) == 1, (
        f"expected exactly one QComboBox (the group-by selector), found "
        f"{len(combos)} -- a boolean column in the test table would break "
        f"this assumption"
    )
    return combos[0]


def test_filter_panel_shows_restored_filters_and_group_by_after_a_table_switch(
    tmp_path, make_controller, qapp
):
    controller, dataset, _ = make_controller(tmp_path, merge_csv=True)
    dataset.create_table_from_df(
        "table_b", pd.DataFrame({"label": ["x", "y"], "value": [1, 2]})
    )

    filters = [Filter("condition", "isin", ["positive"])]
    controller.set_filters(filters)
    controller.set_group_by("condition")

    panel = FilterPanel(controller)
    _refresh(panel, controller, qapp)

    # Sanity check before the actual switch: the panel reflects table A's
    # state as soon as it is built.
    assert panel.get_active_filters() == filters
    assert _checked_toggle_texts(panel) == ["positive"]
    assert _group_by_combo(panel).currentData() == "condition"

    controller.set_active_table("table_b")
    _refresh(panel, controller, qapp)

    assert panel.get_active_filters() == []
    assert _checked_toggle_texts(panel) == []
    assert _group_by_combo(panel).currentData() is None

    controller.set_active_table("frames")
    _refresh(panel, controller, qapp)

    assert panel.get_active_filters() == filters
    assert _checked_toggle_texts(panel) == ["positive"]
    assert _group_by_combo(panel).currentData() == "condition"


def test_restoring_query_state_does_not_trigger_an_extra_apply(
    tmp_path, make_controller, qapp
):
    controller, _dataset, _ = make_controller(tmp_path, merge_csv=True)
    controller.set_filters([Filter("condition", "isin", ["positive"])])
    controller.set_group_by("condition")

    panel = FilterPanel(controller)

    filters_emitted:  list[list[Filter]] = []
    group_by_emitted: list[object]       = []
    panel.filters_changed.connect(filters_emitted.append)
    panel.group_by_changed.connect(group_by_emitted.append)

    _refresh(panel, controller, qapp)

    assert filters_emitted == [], (
        "restoring the remembered filters re-emitted filters_changed, "
        "which would trigger a second, redundant query on top of the one "
        "that already produced the state being restored"
    )
    assert group_by_emitted == [], (
        "restoring the remembered group-by re-emitted group_by_changed"
    )


def test_numeric_range_restore_re_caps_the_spin_bounds(
    tmp_path, make_controller, qapp
):
    controller, dataset, _ = make_controller(tmp_path)
    dataset.create_table_from_df(
        "numeric_table", pd.DataFrame({"value": [0.0, 3.0, 7.0, 10.0]})
    )

    controller.set_active_table("numeric_table")
    controller.set_filters([Filter("value", "between", [5.0, 8.0])])

    panel = FilterPanel(controller)
    _refresh(panel, controller, qapp)

    # Round-trip through another table and back, the same as a
    # researcher switching away and returning.
    controller.set_active_table("frames")
    _refresh(panel, controller, qapp)
    controller.set_active_table("numeric_table")

    filters_emitted: list[list[Filter]] = []
    panel.filters_changed.connect(filters_emitted.append)

    _refresh(panel, controller, qapp)

    min_spin, max_spin = panel.findChildren(QDoubleSpinBox)
    assert min_spin.value() == 5.0
    assert max_spin.value() == 8.0
    assert min_spin.maximum() == 8.0, (
        "min_spin's own Qt maximum was not re-capped to the restored hi -- "
        "the very next edit on it could cross max_spin and silently zero "
        "the query"
    )
    assert max_spin.minimum() == 5.0, (
        "max_spin's own Qt minimum was not re-capped to the restored lo"
    )
    assert filters_emitted == [], (
        "restoring the numeric filter's bounds re-emitted filters_changed"
    )


def test_group_height_slider_disabled_when_restored_group_by_is_absent_from_the_combo(
    tmp_path, make_controller, qapp, monkeypatch
):
    controller, _dataset, _ = make_controller(tmp_path, merge_csv=True)
    panel = FilterPanel(controller)

    # A state naming a column that will not be among the columns
    # refresh_columns() is given -- exactly the situation the combo's
    # own fallback-to-"None" branch exists for. Built by stubbing
    # get_query_state() rather than through a real table switch: every
    # current app path already keeps get_query_state() consistent with
    # the target table's columns, so this exercises the panel's own
    # defensive handling of that fallback on its own, independent of
    # whether anything upstream can produce it today.
    monkeypatch.setattr(
        controller, "get_query_state",
        lambda: QueryState(group_by="not_a_real_column"),
    )

    panel.refresh_columns(controller.get_column_names())

    combo = _group_by_combo(panel)
    assert combo.currentData() is None

    assert panel._group_height_slider.isEnabled() is False, (
        "the group-height slider was enabled from the raw (unmatched) "
        "group_by value instead of the combo's own resolved selection"
    )
    assert panel._group_height_label.isEnabled() is False
