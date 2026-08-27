"""
tests/conftest.py

Two jobs, both of which have to happen before any test module is
imported:

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

from PySide6.QtWidgets import QApplication


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
