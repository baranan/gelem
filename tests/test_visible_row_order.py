"""
tests/test_visible_row_order.py

Tests for P0.4 (docs/media_architecture.md section 6.1): the controller
owns one flat ordered query result plus group boundaries; the gallery
holds no row ids and is given an index range into that order; a result
carries an id and a stale viewport report is dropped.

Written from the work-item specification, not from the implementation.
Each test is designed to fail if the rule it guards is violated -- in
particular, several fail against the pre-P0.4 code where the order lived
in GalleryWidget._row_ids and save_filtered_as_table() re-ran the query
without randomise/seed.

Run with:
    python -m pytest tests/test_visible_row_order.py
"""

from __future__ import annotations

import dataclasses
import sys
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import pytest

from PySide6.QtWidgets import QApplication

from models.dataset import Dataset
from models.query_engine import QueryEngine, Filter
from operators.base import BaseOperator
from models.query_result import ResultLayout, GroupSection

# The controller factory lives in tests/conftest.py (make_controller
# fixture) -- it was duplicated here and in tests/test_gallery_seam.py.


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    if QApplication.instance() is None:
        QApplication(sys.argv)


class _RecordingOperator(BaseOperator):
    """create_columns operator that records the row_id order it is given."""

    name                 = "record_order"
    create_columns_label = "Record order"
    output_columns       = [("probe", "numeric")]
    requires_image       = False

    def __init__(self):
        super().__init__()
        self.seen: list[str] = []

    def create_columns(self, row_id, image, metadata):
        self.seen.append(str(row_id))
        return {"probe": 1.0}


def _run_operator_and_wait(controller, op_registry, row_ids, monkeypatch):
    """Runs record_order over row_ids and blocks until the worker is done."""
    created: list[threading.Thread] = []
    RealThread = threading.Thread

    class _TrackingThread(RealThread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(threading, "Thread", _TrackingThread)
    controller.run_create_columns("record_order", list(row_ids))
    monkeypatch.undo()

    assert created, "run_create_columns() did not start a worker thread"
    created[-1].join(timeout=5)
    assert not created[-1].is_alive(), "worker thread did not finish in time"
    controller._drain_queues()


# ---------------------------------------------------------------------------
# 1. One order: visible list, operator input, and saved table all agree,
#    even with randomise on. Fails on pre-P0.4 code.
# ---------------------------------------------------------------------------

def test_randomised_order_is_the_same_everywhere(make_controller, tmp_path, monkeypatch):
    controller, dataset, op_registry = make_controller(tmp_path)
    op = _RecordingOperator()
    op_registry.register(op)

    # Randomise on, fixed seed.
    controller.set_filters([], randomise=True, seed=1234)

    visible = controller.get_visible_row_ids()
    natural = list(dataset.read_only_view("frames")["row_id"])
    assert visible != natural, (
        "randomise=True did not actually shuffle -- test cannot detect the bug"
    )

    # The rows a "Visible" operator run receives.
    _run_operator_and_wait(controller, op_registry, visible, monkeypatch)
    assert op.seen == visible, (
        "operator received rows in a different order from get_visible_row_ids()"
    )

    # The rows save_filtered_as_table() stores.
    controller.save_filtered_as_table("saved")
    saved = list(dataset.get_table("saved")["row_id"])
    assert saved == visible, (
        "save_filtered_as_table() stored a different order from the one on screen"
    )


# ---------------------------------------------------------------------------
# 2. save_filtered_as_table() runs no second query and copies no table.
# ---------------------------------------------------------------------------

def test_save_filtered_set_does_not_requery_or_copy(make_controller, tmp_path, monkeypatch):
    controller, dataset, _ = make_controller(tmp_path)
    controller.set_filters([], randomise=True, seed=7)

    counts = {"apply": 0, "get_table": 0}
    original_apply     = QueryEngine.apply
    original_get_table = Dataset.get_table

    def _spy_apply(self, *args, **kwargs):
        counts["apply"] += 1
        return original_apply(self, *args, **kwargs)

    def _spy_get_table(self, *args, **kwargs):
        counts["get_table"] += 1
        return original_get_table(self, *args, **kwargs)

    monkeypatch.setattr(QueryEngine, "apply", _spy_apply)
    monkeypatch.setattr(Dataset, "get_table", _spy_get_table)

    controller.save_filtered_as_table("saved")

    assert counts["apply"] == 0, (
        f"save_filtered_as_table() called QueryEngine.apply() {counts['apply']} "
        f"times; it must reuse the flat order the controller already owns"
    )
    assert counts["get_table"] == 0, (
        f"save_filtered_as_table() called Dataset.get_table() {counts['get_table']} "
        f"times; it must not copy the whole table"
    )


# ---------------------------------------------------------------------------
# 3. In grouped mode the GroupSections tile the flat order exactly.
# ---------------------------------------------------------------------------

def test_group_sections_tile_the_flat_order(make_controller, tmp_path):
    controller, dataset, _ = make_controller(tmp_path, merge_csv=True)
    controller.set_group_by("condition")

    layout = controller.get_result_layout()
    flat   = controller.get_visible_row_ids()
    sections = layout.groups

    assert sections is not None and len(sections) > 1, (
        "expected several groups for the 'condition' column"
    )

    # Sections tile [0, total) with no gap and no overlap.
    assert sections[0].start == 0
    assert sections[-1].stop == layout.total == len(flat)
    for a, b in zip(sections, sections[1:]):
        assert a.stop == b.start, "group sections overlap or leave a gap"

    # Each section slices the flat order exactly: every row in the slice
    # belongs to that group, and the lengths sum to total.
    view = dataset.read_only_view("frames").set_index("row_id")
    total_len = 0
    for section in sections:
        slice_ids = controller.get_row_ids_in_range(section.start, section.stop)
        assert slice_ids == list(flat[section.start:section.stop])
        for rid in slice_ids:
            assert str(view.loc[rid, "condition"]) == section.label
        total_len += (section.stop - section.start)
    assert total_len == layout.total


# ---------------------------------------------------------------------------
# 4. groups is None in flat mode, a tuple in grouped mode -- including
#    the zero-group case. None and () are different states.
# ---------------------------------------------------------------------------

def test_groups_none_vs_tuple(make_controller, tmp_path):
    controller, _, _ = make_controller(tmp_path, merge_csv=True)

    controller.set_group_by(None)
    assert controller.get_result_layout().groups is None

    controller.set_group_by("condition")
    groups = controller.get_result_layout().groups
    assert isinstance(groups, tuple) and len(groups) > 0

    # Grouped, but a filter removes every row: grouped mode with zero
    # groups. Still a tuple, not None.
    controller.set_filters([Filter("condition", "eq", "__no_such_value__")])
    empty_groups = controller.get_result_layout().groups
    assert empty_groups == ()
    assert empty_groups is not None


# ---------------------------------------------------------------------------
# 5. Displayed-range bookkeeping: two keys, clearing one, stale id, refresh.
# ---------------------------------------------------------------------------

def test_displayed_range_tracking(make_controller, tmp_path):
    controller, _, _ = make_controller(tmp_path)
    controller.set_filters([])
    rid = controller.get_result_layout().result_id

    controller.report_displayed_range("b", 20, 30, rid)
    controller.report_displayed_range("a", 0, 10, rid)
    assert controller.get_displayed_ranges() == [(0, 10), (20, 30)], (
        "get_displayed_ranges() must return every stored range, sorted by start"
    )

    controller.clear_displayed_range("a")
    assert controller.get_displayed_ranges() == [(20, 30)]

    # A report naming a superseded result is ignored.
    controller.report_displayed_range("c", 0, 5, "not-the-current-id")
    assert controller.get_displayed_ranges() == [(20, 30)]

    # A refresh clears every stored range -- they point into an order
    # that no longer exists.
    controller.set_filters([])
    assert controller.get_displayed_ranges() == []


# ---------------------------------------------------------------------------
# 6. ResultLayout exposes no row ids.
# ---------------------------------------------------------------------------

def test_result_layout_carries_no_row_ids(make_controller, tmp_path):
    controller, _, _ = make_controller(tmp_path, merge_csv=True)
    controller.set_group_by("condition")
    layout = controller.get_result_layout()

    assert isinstance(layout, ResultLayout)
    field_names = {f.name for f in dataclasses.fields(layout)}
    assert field_names == {"result_id", "table_name", "total", "groups"}, (
        f"ResultLayout gained or lost a field: {field_names}"
    )
    assert not hasattr(layout, "row_ids")

    for section in layout.groups:
        section_fields = {f.name for f in dataclasses.fields(section)}
        assert section_fields == {"label", "start", "stop"}, (
            f"GroupSection gained or lost a field: {section_fields}"
        )


# ---------------------------------------------------------------------------
# 6b. A failed query does not leave the previous result standing.
# ---------------------------------------------------------------------------

def test_failed_refresh_publishes_an_empty_result_for_the_active_table(make_controller, 
    tmp_path, monkeypatch
):
    controller, _, _ = make_controller(tmp_path)
    controller.set_filters([])
    assert controller.get_visible_row_ids(), "sanity: started with a real result"

    def _boom(self, *args, **kwargs):
        raise RuntimeError("query blew up")

    monkeypatch.setattr(QueryEngine, "apply", _boom)
    controller.set_filters([])

    assert controller.get_visible_row_ids() == [], (
        "a failed refresh must not leave the previous order visible"
    )
    layout = controller.get_result_layout()
    assert layout.table_name == "frames", (
        "the empty result must name the current active table, so a later "
        "operator run does not feed stale row ids against it"
    )
    assert layout.total == 0 and layout.groups is None
    assert controller.get_displayed_ranges() == []


# ---------------------------------------------------------------------------
# 7. None vs [] for visible columns. CLAUDE.md tags this [NOW]; this is
#    the promised guardrail test.
# ---------------------------------------------------------------------------

def test_visible_columns_none_versus_empty(make_controller, tmp_path):
    controller, _, _ = make_controller(tmp_path)

    # Before any choice: no preference, and the effective list falls back
    # to the default media column.
    assert controller.has_visible_columns_preference() is False
    assert controller.get_effective_visible_columns() == ["full_path"]

    # An explicit empty choice: preference set, effective list empty.
    controller.set_visible_columns([])
    assert controller.has_visible_columns_preference() is True
    assert controller.get_effective_visible_columns() == []

    # Clearing the preference returns to the fallback.
    controller.clear_visible_columns_preference()
    assert controller.has_visible_columns_preference() is False
    assert controller.get_effective_visible_columns() == ["full_path"]
