"""
tests/test_visible_set_and_project_reset.py

Covers two related pieces of controller/UI lifecycle behaviour:

  * #36 — the visible row set lives behind AppController.
      get_visible_row_ids() returns the rows currently on screen, in
      display order (flattened + de-duplicated in grouped mode), and
      MainWindow no longer reconstructs it from GalleryWidget internals.

  * #15 — loading a new project resets per-project state.
      _reset_project_state() clears the cached visible set, drops the
      previous project's registered column types, and emits project_reset;
      the three load_* entry points all funnel through it.

The behavioural tests drive a real AppController with lightweight fakes
for the data components (and the real ColumnTypeRegistry). The guard
tests are AST/source-inspection so they run without Qt.

Run with:
    pytest tests/test_visible_set_and_project_reset.py
"""

import ast
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from controller import AppController
from column_types.registry import ColumnTypeRegistry


# ── Qt fixture (AppController is a QObject with a QTimer) ──────────────────
@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


# ── Lightweight fakes for the data components ─────────────────────────────
class _FakeQuery:
    """apply() returns a fixed flat list; apply_grouped() returns a dict
    whose groups deliberately share a row_id ("2") so the flatten can be
    proven to de-duplicate while preserving first-seen order."""

    FLAT = ["1", "2", "3"]
    GROUPED = {"a": ["1", "2"], "b": ["2", "3"]}

    def apply(self, df, **kwargs):
        return list(self.FLAT)

    def apply_grouped(self, df, **kwargs):
        return {k: list(v) for k, v in self.GROUPED.items()}

    def get_group_values(self, df, column):
        return []


class _FakeDataset:
    def __init__(self):
        self.registry = None

    def set_registry(self, registry):
        self.registry = registry

    def get_table(self, name):
        return pd.DataFrame(
            {"row_id": ["1", "2", "3"], "full_path": ["", "", ""]}
        )

    def list_tables(self):
        return ["frames"]

    def load_folder(self, path):
        pass

    def load_csv_as_primary(self, path, image_column=None):
        pass

    def load(self, path):
        pass


class _FakeStore:
    def __init__(self):
        self.on_thumbnail_ready = None
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def request_thumbnail(self, row_id, path):
        pass

    def load_index(self, path):
        pass


class _FakeOpRegistry:
    pass


def _make_controller():
    """Builds an AppController wired to fakes and a real registry primed
    with the built-in type tags."""
    registry = ColumnTypeRegistry()
    registry.setup_defaults(artifact_store=object())  # store is only captured
    controller = AppController(
        _FakeDataset(), _FakeQuery(), _FakeStore(), registry, _FakeOpRegistry()
    )
    return controller, registry


# ── #36: get_visible_row_ids ──────────────────────────────────────────────
def test_visible_row_ids_flat(qapp):
    controller, _ = _make_controller()
    controller.set_filters([])  # flat refresh
    assert controller.get_visible_row_ids() == ["1", "2", "3"]


def test_visible_row_ids_grouped_is_flattened_and_deduped(qapp):
    controller, _ = _make_controller()
    controller.set_group_by("some_column")  # grouped refresh
    # "2" appears in both groups but only once in the visible set, and the
    # order follows group order then within-group order.
    assert controller.get_visible_row_ids() == ["1", "2", "3"]


def test_visible_row_ids_returns_a_copy(qapp):
    controller, _ = _make_controller()
    controller.set_filters([])
    snapshot = controller.get_visible_row_ids()
    snapshot.append("999")
    # Mutating the returned list must not corrupt the controller's cache.
    assert controller.get_visible_row_ids() == ["1", "2", "3"]


# ── #15: project reset ─────────────────────────────────────────────────────
def test_reset_clears_visible_registry_and_emits(qapp):
    controller, registry = _make_controller()

    # Simulate a live previous project: a visible set and a visual column.
    controller.set_filters([])
    registry.register_by_tag("full_path", "media_path")
    assert controller.get_visible_row_ids() == ["1", "2", "3"]
    assert "full_path" in registry.list_visual_columns()

    fired = []
    controller.project_reset.connect(lambda: fired.append(True))

    controller._reset_project_state()

    assert fired == [True], "project_reset must be emitted on reset"
    assert controller.get_visible_row_ids() == [], "cached visible set must clear"
    assert registry.list_all_columns() == [], "stale column types must clear"
    assert registry.list_visual_columns() == []


def test_registry_clear_columns_keeps_builtin_types():
    """clear_columns() drops column-name registrations but keeps the
    built-in type tags, so registration works again afterwards."""
    registry = ColumnTypeRegistry()
    registry.setup_defaults(artifact_store=object())
    registry.register_by_tag("full_path", "media_path")

    registry.clear_columns()
    assert registry.list_all_columns() == []

    # Built-in 'media_path' tag survived, so we can register anew.
    registry.register_by_tag("avatar_path", "media_path")
    assert registry.list_visual_columns() == ["avatar_path"]


# ── Guard tests (source inspection, no Qt) ─────────────────────────────────
_CONTROLLER_SRC = (project_root / "controller.py").read_text(encoding="utf-8")
_MAIN_WINDOW_SRC = (project_root / "ui" / "main_window.py").read_text(encoding="utf-8")


def _method_source(src: str, name: str):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


@pytest.mark.parametrize(
    "method", ["load_folder", "load_csv_as_primary", "load_project"]
)
def test_load_methods_funnel_through_reset(method):
    """Every project-load entry point must call _reset_project_state so a
    new project can't inherit the previous one's state (#15)."""
    src = _method_source(_CONTROLLER_SRC, method)
    assert src is not None, f"{method} not found in controller.py"
    assert "_reset_project_state" in src, (
        f"{method} must call _reset_project_state() to clear per-project state"
    )


def test_main_window_does_not_read_gallery_row_ids():
    """#36 acceptance: MainWindow must not reconstruct the visible set from
    GalleryWidget internals — it reads controller.get_visible_row_ids()."""
    for bad in ("gallery._row_ids", "g._row_ids", "_main_gallery._row_ids"):
        assert bad not in _MAIN_WINDOW_SRC, (
            f"main_window.py reads {bad}; use "
            f"controller.get_visible_row_ids() instead (issue #36)."
        )
