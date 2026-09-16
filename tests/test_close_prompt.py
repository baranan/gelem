"""
tests/test_close_prompt.py

Layer A only for ui/close_prompt.py's decide_close_prompt() (unsaved-work
item, decision 2 and 3): given (has_unsaved, run_live), which of the
three close prompts is chosen. No wording assertions -- only that the
right prompt (and, where the work item specifies it, the right buttons
and default) comes back for each of the three cases:

  - nothing unsaved and no live run: close without asking.
  - a live run: "Close anyway" / "Cancel", default Cancel.
  - unsaved, no live run: "Save" / "Discard" / "Cancel", default Cancel.

Also asserts that a live run wins even when has_unsaved is also True,
since decide_close_prompt() checks it first (its own docstring says why).

Qt-free by construction: this module never imports PySide6, and a
guardrail below fails if that ever stops being true.

Written from the work-item specification, not from the implementation.

Run with:
    python -m pytest tests/test_close_prompt.py

# run-tests: combined -- names Qt bindings in a source-scan assertion but never imports them
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ui.close_prompt import (
    BUTTON_CANCEL,
    BUTTON_CLOSE_ANYWAY,
    BUTTON_DISCARD,
    BUTTON_SAVE,
    CLOSE_WITHOUT_ASKING,
    LIVE_RUN,
    UNSAVED,
    decide_close_prompt,
)

_CLOSE_PROMPT_FILE = PROJECT_ROOT / "ui" / "close_prompt.py"


def test_nothing_unsaved_and_no_live_run_closes_without_asking():
    prompt = decide_close_prompt(has_unsaved=False, run_live=False)
    assert prompt.kind == CLOSE_WITHOUT_ASKING
    assert prompt.buttons == ()


def test_a_live_run_offers_close_anyway_or_cancel_default_cancel():
    prompt = decide_close_prompt(has_unsaved=False, run_live=True)
    assert prompt.kind == LIVE_RUN
    assert prompt.buttons == (BUTTON_CLOSE_ANYWAY, BUTTON_CANCEL)
    assert prompt.default_button == BUTTON_CANCEL


def test_unsaved_with_no_live_run_offers_save_discard_cancel_default_cancel():
    prompt = decide_close_prompt(has_unsaved=True, run_live=False)
    assert prompt.kind == UNSAVED
    assert prompt.buttons == (BUTTON_SAVE, BUTTON_DISCARD, BUTTON_CANCEL)
    assert prompt.default_button == BUTTON_CANCEL


def test_a_live_run_wins_over_unsaved_changes():
    # Both conditions true at once -- the live-run prompt is checked
    # first and must be the one shown, not the unsaved one.
    prompt = decide_close_prompt(has_unsaved=True, run_live=True)
    assert prompt.kind == LIVE_RUN


def test_close_prompt_module_imports_no_qt():
    # AST walk, not a text search, so a docstring or comment mentioning a
    # Qt binding does not trip this -- the same technique
    # tests/test_operators_are_qt_free.py uses for operators/.
    tree = ast.parse(_CLOSE_PROMPT_FILE.read_text(encoding="utf-8"))
    qt_distributions = ("PySide6", "PySide2", "PyQt5", "PyQt6", "shiboken6")

    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits.extend(
                alias.name
                for alias in node.names
                if any(
                    alias.name == dist or alias.name.startswith(dist + ".")
                    for dist in qt_distributions
                )
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module and any(
                node.module == dist or node.module.startswith(dist + ".")
                for dist in qt_distributions
            ):
                hits.append(node.module)

    assert not hits, (
        f"ui/close_prompt.py (Layer A) must import no Qt binding; found: {hits}"
    )
