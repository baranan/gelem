"""
tests/test_unsaved_changes.py

AppController.has_unsaved_changes() (unsaved-work item, decision 1): a
dirty flag built from comparing Dataset.table_versions() against a
baseline snapshot taken at construction and re-taken right after every
successful save_project() / load_project() -- nowhere else, so a table
changed through any other public path (a fresh load_folder(),
save_filtered_as_table(), ...) reads as unsaved.

Written from the work-item specification, not from the implementation.
Real Dataset and AppController, no widgets and no QApplication -- see
docs/review/unsaved-work-survey.md section 3 for why table_versions() is
a valid dirty-flag basis and why the baseline must be re-taken after
load(), not computed once at startup.

Run with:
    python -m pytest tests/test_unsaved_changes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.dataset import Dataset
from models.query_engine import QueryEngine
from artifacts.artifact_store import ArtifactStore
from column_types.registry import ColumnTypeRegistry
from operators.operator_registry import OperatorRegistry
from controller import AppController
from media.resolver import MediaResolver


def _make_controller(tmp_path: Path) -> AppController:
    """A real AppController over a fresh, empty Dataset -- no folder or
    CSV loaded yet, unlike tests/conftest.py's make_controller fixture,
    so has_unsaved_changes() can be exercised from a genuinely fresh
    session."""
    store = ArtifactStore(tmp_path / "artifacts", resolver=MediaResolver(max_open_decoders=4))
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)

    dataset = Dataset()
    op_registry = OperatorRegistry()
    return AppController(dataset, QueryEngine(), store, registry, op_registry, resolver=MediaResolver(max_open_decoders=4))


def test_fresh_controller_has_no_unsaved_changes(tmp_path):
    controller = _make_controller(tmp_path)
    assert controller.has_unsaved_changes() is False


def test_loading_tables_into_a_fresh_session_counts_as_unsaved(tmp_path):
    controller = _make_controller(tmp_path)

    # An empty folder still goes through Dataset.load_folder()'s accept
    # path (a placeholder table when no media files are found), which is
    # enough to bump table_versions() away from the construction-time
    # baseline.
    empty_folder = tmp_path / "media"
    empty_folder.mkdir()
    controller.load_folder(empty_folder)

    assert controller.has_unsaved_changes() is True


def test_save_project_clears_unsaved_changes(tmp_path):
    controller = _make_controller(tmp_path)
    empty_folder = tmp_path / "media"
    empty_folder.mkdir()
    controller.load_folder(empty_folder)
    assert controller.has_unsaved_changes() is True

    saved_to = tmp_path / "saved_project"
    result = controller.save_project(saved_to)

    assert result is True
    assert controller.has_unsaved_changes() is False


def test_a_table_change_after_save_counts_as_unsaved_again(tmp_path):
    controller = _make_controller(tmp_path)
    empty_folder = tmp_path / "media"
    empty_folder.mkdir()
    controller.load_folder(empty_folder)

    saved_to = tmp_path / "saved_project"
    assert controller.save_project(saved_to) is True
    assert controller.has_unsaved_changes() is False

    # A table change through a public controller path following the
    # save. load_folder() already populated self._result (empty though
    # it is), so save_filtered_as_table() can create a new table from it
    # without a query first.
    controller.save_filtered_as_table("subset")

    assert controller.has_unsaved_changes() is True


def test_load_project_clears_unsaved_changes(tmp_path):
    controller = _make_controller(tmp_path)
    empty_folder = tmp_path / "media"
    empty_folder.mkdir()
    controller.load_folder(empty_folder)

    saved_to = tmp_path / "saved_project"
    assert controller.save_project(saved_to) is True

    # Dirty it again so load_project() is the thing that actually clears
    # it, not a leftover False from the save above.
    controller.save_filtered_as_table("subset")
    assert controller.has_unsaved_changes() is True

    controller.load_project(saved_to)

    assert controller.has_unsaved_changes() is False
