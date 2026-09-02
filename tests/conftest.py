"""
tests/conftest.py

Three jobs, all of which have to happen before any test module is
imported, so they run in the early block below rather than in fixtures:

  1. Force Qt's "offscreen" platform plugin. pytest imports every
     conftest.py before it collects the test modules beside it, so
     setting QT_QPA_PLATFORM here is early enough that no test -- not
     even one that does `import PySide6` at module scope -- ever pops a
     real window. setdefault() means an explicit QT_QPA_PLATFORM in the
     environment (a developer who wants to watch) still wins.

  2. Put the project root on sys.path. The individual test modules also
     do this for when they are run one at a time; doing it here as well
     is harmless and keeps a bare `pytest` run working from any
     directory.

  3. Flip Dataset._DEFAULT_STRICT_SCHEMA to True for the whole run
     (P1.8b-2). It must be a class-level flip done at import time, not a
     fixture: test modules build Dataset() directly, Dataset.__init__
     runs its own _accept_table("frames", ...) before any fixture could
     reach the instance, and tests/test_dataset.py runs 64 assertions
     through module-level run_test(...) calls at import -- a fixture, set
     up only at the first test's setup, would miss all of that. Because
     conftest.py is imported before its sibling test modules are
     collected, this genuinely runs first. No restore: the process exits
     when pytest is done, exactly as for the os.environ and sys.path
     mutations above. A test that needs the lenient path sets
     `ds.strict_schema = False` on its own instance, which wins over the
     class default.

It also provides the shared Qt fixtures used by the widget-level tests
(currently tests/test_gallery_seam.py). tests/test_visible_row_order.py
and tests/test_result_delivery.py build their own QApplication with the
same QApplication.instance() guard; the fixtures here reuse whatever
instance exists, so the two approaches coexist.
"""

from __future__ import annotations

import os

# MUST run before the first `import PySide6` anywhere in the test run.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# MUST run before any test module is imported -- see job 3 above. The import
# sits here, in the ordering-sensitive block, rather than with the imports at
# the top of the file, so it is obvious it belongs to job 3; models.dataset
# only pulls in pandas/numpy, already dependencies of the suite.
from models.dataset import Dataset as _Dataset

_Dataset._DEFAULT_STRICT_SCHEMA = True

from PySide6.QtWidgets import QApplication

TEST_IMAGES  = Path(__file__).parent.parent / "test_images"
METADATA_CSV = TEST_IMAGES / "metadata.csv"


@pytest.fixture
def make_controller():
    """Factory: a real AppController over the 20-row test_images table.

    Was copied verbatim into tests/test_gallery_seam.py and
    tests/test_visible_row_order.py -- two places computing one thing.
    Returns ``(controller, dataset, op_registry)``; pass ``merge_csv=True``
    to also merge metadata.csv on file_name.
    """
    from models.dataset import Dataset
    from models.query_engine import QueryEngine
    from artifacts.artifact_store import ArtifactStore
    from column_types.registry import ColumnTypeRegistry
    from operators.operator_registry import OperatorRegistry
    from controller import AppController

    def _make(tmp_path, *, merge_csv: bool = False):
        store    = ArtifactStore(tmp_path / "artifacts")
        registry = ColumnTypeRegistry()
        registry.setup_defaults(store)

        dataset = Dataset()
        dataset.set_registry(registry)
        dataset.load_folder(TEST_IMAGES)
        if merge_csv:
            dataset.confirm_merge(
                dataset.merge_csv(METADATA_CSV, join_on="file_name")
            )

        op_registry = OperatorRegistry()
        controller  = AppController(
            dataset, QueryEngine(), store, registry, op_registry
        )
        return controller, dataset, op_registry

    return _make


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """
    The one process-wide QApplication. Qt permits exactly one, so this
    reuses an existing instance if another fixture or module already
    built it.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def realize_widget(qapp):
    """
    Returns a helper that shows a widget at a fixed size and lets Qt
    actually apply the layout before the test inspects it.

    Why the processEvents() calls: Qt computes geometry lazily.
    resize() and show() only *schedule* a layout/resize pass -- they do
    not run it. Until the event loop delivers those queued events, child
    widgets keep their previous sizes, scrollbar ranges are stale, and
    the scroll-area viewport reports the wrong dimensions. The
    widget-level tests here inspect exactly those values, so the helper
    pumps the queue. Two passes on purpose: the first delivers the
    show/resize events, and the second delivers whatever those handlers
    scheduled in turn (a GalleryWidget resize re-runs _relayout, which
    mounts tiles, which schedules more events).
    """

    def _realize(widget, *, width: int = 900, height: int = 700):
        widget.resize(width, height)
        widget.show()
        qapp.processEvents()
        qapp.processEvents()
        return widget

    return _realize
