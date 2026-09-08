"""
tests/test_run_tests_grouping.py

Guards run_tests.py's module-grouping rule.

run_tests.py splits the suite so that every test module which shows a Qt
widget runs in its own pytest process (see that file's docstring and
docs/known_defects.md). Two mechanisms decide which modules those are:

  1. an explicit "# run-tests: isolate -- <reason>" or
     "# run-tests: combined -- <reason>" comment token, which decides
     whatever the substring heuristic would say; and
  2. failing a token, the substring heuristic: the module's source contains
     any of "PySide6", "realize_widget", or ".show(".

Both mechanisms live in run_tests.py. This test exercises them, and checks
that the explicitly written WIDGET_MODULES list has not drifted from what the
rule actually derives -- once through an independent re-derivation here, once
through run_tests.py's own scan.

This module necessarily contains the marker substrings to describe the rule,
so run_tests.py excludes it from the directory scan by name
(run_tests.GROUPING_SELF_EXCLUDED).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# An independent copy of the heuristic markers, used only by the re-derivation
# below. run_tests.py has its own copy; this one exists so a regression in
# run_tests.py's heuristic is caught rather than mirrored on both sides of the
# drift check.
_INDEPENDENT_MARKERS = ("PySide6", "realize_widget", ".show(")


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


def _independently_derived_isolated(run_tests) -> set[str]:
    """Re-derive the isolated set here from the real tests/ directory.

    Reuses only run_tests.declared_isolation_token -- STEP 2 of the work item
    requires a single token parser -- and re-implements the substring
    heuristic locally, so a regression in run_tests.py's own heuristic still
    fails a drift check instead of being mirrored on both sides.
    """
    isolated: set[str] = set()
    for path in TESTS_DIR.glob("test_*.py"):
        if path.name in run_tests.GROUPING_SELF_EXCLUDED:
            continue
        text = path.read_text(encoding="utf-8")
        token = run_tests.declared_isolation_token(text, path.name)
        if token is not None:
            if token == "isolate":
                isolated.add(path.name)
            continue
        if any(marker in text for marker in _INDEPENDENT_MARKERS):
            isolated.add(path.name)
    return isolated


# ---------------------------------------------------------------------------
# Existing guards: WIDGET_MODULES stays consistent with what the rule derives.
# ---------------------------------------------------------------------------
def test_widget_list_matches_modules_that_touch_widgets():
    run_tests = _load_run_tests()

    declared = set(run_tests.WIDGET_MODULES)
    expected = _independently_derived_isolated(run_tests)

    missing = expected - declared
    extra = declared - expected

    assert not missing, (
        "test modules are isolated by the rule but absent from "
        f"run_tests.WIDGET_MODULES: {sorted(missing)}"
    )
    assert not extra, (
        "run_tests.WIDGET_MODULES names modules the rule does not isolate: "
        f"{sorted(extra)}"
    )

    # run_tests.py's own scan must agree with the independent re-derivation;
    # if it does not, its classifier has regressed.
    assert run_tests.scan_isolated_modules() == expected


def test_widget_list_has_no_duplicates_and_all_files_exist():
    run_tests = _load_run_tests()

    names = list(run_tests.WIDGET_MODULES)
    assert len(names) == len(set(names)), "WIDGET_MODULES has duplicate entries"

    for name in names:
        assert (TESTS_DIR / name).is_file(), f"{name} does not exist in tests/"


# ---------------------------------------------------------------------------
# The declaration token wins over the substring heuristic.
# ---------------------------------------------------------------------------
def test_isolate_token_classifies_as_isolated_without_any_substring():
    run_tests = _load_run_tests()

    # No "PySide6", "realize_widget" or ".show(" anywhere in this text.
    source = (
        "# run-tests: isolate -- loads Qt transitively via a helper import\n"
        "def test_thing():\n"
        "    assert True\n"
    )

    assert run_tests.module_runs_isolated(source, "test_fake.py") is True


def test_combined_token_classifies_as_combined_even_with_pyside6_present():
    run_tests = _load_run_tests()

    source = (
        "# run-tests: combined -- names PySide6 in a source-scan assertion only\n"
        'FORBIDDEN_BINDING = "PySide6"\n'
        "def test_thing():\n"
        "    assert FORBIDDEN_BINDING not in open(__file__).read()\n"
    )

    assert run_tests.module_runs_isolated(source, "test_fake.py") is False


# ---------------------------------------------------------------------------
# No token: the substring heuristic decides, both ways.
# ---------------------------------------------------------------------------
def test_no_token_falls_back_to_the_substring_heuristic():
    run_tests = _load_run_tests()

    with_widget = "w = build_window()\nw.show()\n"
    without_widget = "value = 1 + 1\nassert value == 2\n"

    assert run_tests.module_runs_isolated(with_widget, "a.py") is True
    assert run_tests.module_runs_isolated(without_widget, "b.py") is False


# ---------------------------------------------------------------------------
# Malformed declarations are refused, each error naming the module.
# ---------------------------------------------------------------------------
def test_a_module_carrying_both_tokens_raises():
    run_tests = _load_run_tests()

    source = (
        "# run-tests: isolate -- one reason\n"
        "# run-tests: combined -- a different reason\n"
    )

    with pytest.raises(run_tests.RunTestsDeclarationError) as excinfo:
        run_tests.declared_isolation_token(source, "test_conflict.py")
    assert "test_conflict.py" in str(excinfo.value)


def test_a_token_with_an_empty_reason_raises():
    run_tests = _load_run_tests()

    source = "# run-tests: isolate --    \n"

    with pytest.raises(run_tests.RunTestsDeclarationError) as excinfo:
        run_tests.declared_isolation_token(source, "test_blank.py")
    assert "test_blank.py" in str(excinfo.value)


def test_an_unknown_token_value_raises():
    run_tests = _load_run_tests()

    source = "# run-tests: sometimes -- because the weather is nice\n"

    with pytest.raises(run_tests.RunTestsDeclarationError) as excinfo:
        run_tests.declared_isolation_token(source, "test_unknown.py")
    assert "test_unknown.py" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The real tests/ directory: every token is well formed, and run_tests.py's
# own scan matches the explicitly written WIDGET_MODULES.
# ---------------------------------------------------------------------------
def test_real_tests_directory_tokens_are_valid_and_scan_matches_widget_modules():
    run_tests = _load_run_tests()

    walked = 0
    for path in TESTS_DIR.glob("test_*.py"):
        walked += 1
        if path.name in run_tests.GROUPING_SELF_EXCLUDED:
            continue
        text = path.read_text(encoding="utf-8")
        # declared_isolation_token raises on a blank reason, an unknown token
        # or a duplicate declaration, so calling it on every module is the
        # "every token is well formed" check.
        run_tests.declared_isolation_token(text, path.name)

    # A directory walk that found nothing would pass vacuously.
    assert walked > 0, "no test_*.py modules were walked"

    assert run_tests.scan_isolated_modules() == set(run_tests.WIDGET_MODULES)
