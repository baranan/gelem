"""
tests/test_run_tests_grouping.py

Guards run_tests.py's WIDGET_MODULES list against drift.

run_tests.py splits the suite so that every test module which shows a Qt
widget runs in its own pytest process (see that file's docstring and
docs/known_defects.md). The rule for what belongs in that isolated set is
purely textual: a tests/test_*.py module counts as a widget module if its
source contains any of "PySide6", "realize_widget", or ".show(" -- the
first catches a module that imports the Qt binding directly, the other two
catch a module that only realizes a widget through the shared fixtures in
tests/conftest.py. conftest.py itself is excluded.

This test re-derives that set by scanning the tests directory itself, so
adding a new Qt test module without also adding it to WIDGET_MODULES fails
here rather than silently letting the new module rejoin the fragile
combined group.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# A tests/test_*.py module whose text contains any of these belongs in the
# isolated widget group. Plain string literals -- this file's own name is
# skipped in the scan below, so naming the markers here does not make this
# module match its own rule.
WIDGET_MARKERS = ("PySide6", "realize_widget", ".show(")

# This test module has to name the markers to describe the rule, so exclude
# it from the scan by its own filename rather than obfuscating the markers.
SELF_FILENAME = Path(__file__).name


def _load_run_tests():
    """Import run_tests.py from the repo root by file path.

    It lives at the repo root, not on the tests package path, so a plain
    `import run_tests` is not guaranteed to work under every pytest rootdir.
    """
    module_path = REPO_ROOT / "run_tests.py"
    spec = importlib.util.spec_from_file_location("run_tests", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _modules_touching_widgets() -> set[str]:
    """Scan tests/ and return the filenames of every test_*.py module whose
    text contains any widget marker, excluding conftest.py and this file.
    """
    found: set[str] = set()
    for path in TESTS_DIR.glob("test_*.py"):
        if path.name == "conftest.py" or path.name == SELF_FILENAME:
            continue
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in WIDGET_MARKERS):
            found.add(path.name)
    return found


def test_widget_list_matches_modules_that_touch_widgets():
    run_tests = _load_run_tests()

    declared = set(run_tests.WIDGET_MODULES)
    expected = _modules_touching_widgets()

    missing = expected - declared
    extra = declared - expected

    assert not missing, (
        "test modules realize a Qt widget but are absent from "
        f"run_tests.WIDGET_MODULES: {sorted(missing)}"
    )
    assert not extra, (
        "run_tests.WIDGET_MODULES names modules that do not realize a Qt "
        f"widget: {sorted(extra)}"
    )


def test_widget_list_has_no_duplicates_and_all_files_exist():
    run_tests = _load_run_tests()

    names = list(run_tests.WIDGET_MODULES)
    assert len(names) == len(set(names)), "WIDGET_MODULES has duplicate entries"

    for name in names:
        assert (TESTS_DIR / name).is_file(), f"{name} does not exist in tests/"
