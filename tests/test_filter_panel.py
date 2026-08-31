"""
tests/test_filter_panel.py

Tests for FilterPanel's categorical-toggle behaviour.

These tests pin down the multi-select 'isin' contract: clicking several
toggle buttons on the same column should accumulate into one
Filter(column, "isin", [...]) — not replace each other and not turn into
an exclusive single-select.

Run with:
    pytest tests/test_filter_panel.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from PySide6.QtWidgets import QApplication, QPushButton

from models.query_engine import Filter
from ui.filter_panel import FilterPanel


@pytest.fixture(scope="module")
def qapp():
    """Single QApplication for the whole module — Qt allows only one."""
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


class _StubController:
    """Minimal controller stub — FilterPanel only needs get_group_values."""

    def __init__(self, values_by_column: dict[str, list]):
        self._values = values_by_column

    def get_group_values(self, column: str) -> list:
        return list(self._values.get(column, []))


def _emitted_filters(panel: FilterPanel) -> list:
    """Capture filters_changed payloads into a list the test can inspect."""
    captured: list = []
    panel.filters_changed.connect(captured.append)
    return captured


# ── _on_text_toggle: direct unit tests ────────────────────────────────────


def test_first_toggle_emits_single_value_isin(qapp):
    panel = FilterPanel(controller=_StubController({}))
    captured = _emitted_filters(panel)

    panel._on_text_toggle("species", "cat", checked=True)

    assert len(captured) == 1
    (filters,) = captured
    assert filters == [Filter("species", "isin", ["cat"])]


def test_second_toggle_accumulates_into_union(qapp):
    panel = FilterPanel(controller=_StubController({}))
    captured = _emitted_filters(panel)

    panel._on_text_toggle("species", "cat", checked=True)
    panel._on_text_toggle("species", "dog", checked=True)

    # Latest emission carries the union, sorted deterministically.
    assert captured[-1] == [Filter("species", "isin", ["cat", "dog"])]


def test_unchecking_one_value_keeps_the_rest(qapp):
    panel = FilterPanel(controller=_StubController({}))
    captured = _emitted_filters(panel)

    panel._on_text_toggle("species", "cat", checked=True)
    panel._on_text_toggle("species", "dog", checked=True)
    panel._on_text_toggle("species", "cat", checked=False)

    assert captured[-1] == [Filter("species", "isin", ["dog"])]


def test_unchecking_last_value_drops_filter(qapp):
    panel = FilterPanel(controller=_StubController({}))
    captured = _emitted_filters(panel)

    panel._on_text_toggle("species", "cat", checked=True)
    panel._on_text_toggle("species", "cat", checked=False)

    assert captured[-1] == []


def test_toggles_on_different_columns_are_independent(qapp):
    panel = FilterPanel(controller=_StubController({}))
    captured = _emitted_filters(panel)

    panel._on_text_toggle("species", "cat", checked=True)
    panel._on_text_toggle("colour", "red", checked=True)

    assert captured[-1] == [
        Filter("species", "isin", ["cat"]),
        Filter("colour", "isin", ["red"]),
    ]


# ── _add_text_filter: button wiring is non-exclusive ──────────────────────


def test_buttons_built_by_add_text_filter_are_non_exclusive(qapp):
    """
    Regression guard: the buttons _add_text_filter creates must allow
    multiple to be checked at once. If a future refactor wraps them in
    a QButtonGroup with exclusive=True (or replaces them with radio
    buttons), this test fails immediately.
    """
    panel = FilterPanel(
        controller=_StubController({"species": ["cat", "dog", "bird"]})
    )
    captured = _emitted_filters(panel)

    panel._add_text_filter("species")

    buttons = panel.findChildren(QPushButton)
    species_buttons = {b.text(): b for b in buttons if b.text() in {"cat", "dog", "bird"}}
    assert set(species_buttons) == {"cat", "dog", "bird"}

    # Click two buttons; both should stay checked, union should be emitted.
    species_buttons["cat"].click()
    species_buttons["dog"].click()

    assert species_buttons["cat"].isChecked()
    assert species_buttons["dog"].isChecked()
    assert captured[-1] == [Filter("species", "isin", ["cat", "dog"])]
