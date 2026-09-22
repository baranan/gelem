"""
tests/test_table_display.py

CC-23: table_display.py's two pieces, tested directly and in isolation --
TableDisplayState (per-table remembered visible-column choice) and
resolve_default_visible_columns (the fallback when no choice has been
made). No Qt, no pandas: table_display.py is a plain top-level module
(the same reasoning as table_names.py) and this file proves it stays
that way, both by never importing either itself and by an AST guard
over the module's own import statements.

Written from the work-item specification (CLAUDE.md's working rule),
not from the implementation.

# run-tests: combined -- names "PySide6" only inside a source-scan assertion, never imports it
"""

import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from table_display import TableDisplayState, resolve_default_visible_columns


# ---------------------------------------------------------------------------
# Import boundary: table_display.py imports nothing but the standard
# library. A guard, not a restated comment -- CLAUDE.md's testing section:
# "every architectural rule must become a failing test."
# ---------------------------------------------------------------------------

_STDLIB_ALLOWED_TOP_LEVEL_MODULES = {"__future__"}


def test_table_display_module_imports_only_stdlib():
    source = (REPO / "table_display.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="table_display.py")

    banned_substrings = ("qt", "pyside", "pandas", "numpy", "pil", "cv2")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top not in ("PySide6", "pandas", "numpy", "PIL", "cv2"), (
                    f"table_display.py imports {alias.name!r}; it must stay "
                    "standard-library only"
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0]
            if top in _STDLIB_ALLOWED_TOP_LEVEL_MODULES:
                continue
            lowered = top.lower()
            assert not any(bad in lowered for bad in banned_substrings), (
                f"table_display.py imports from {module!r}; it must stay "
                "standard-library only"
            )
            # No import from this repo at all -- table_names.py's own
            # rule for the same reason (see table_display.py's docstring).
            assert top not in ("models", "controller", "ui", "operators",
                                "artifacts", "column_types", "media"), (
                f"table_display.py imports from {module!r}, a repo module; "
                "it must import nothing from this repo"
            )


# ---------------------------------------------------------------------------
# TableDisplayState
# ---------------------------------------------------------------------------

def test_recall_before_any_choice_is_none():
    state = TableDisplayState()
    assert state.recall("frames") is None


def test_remember_then_recall_returns_the_same_columns():
    state = TableDisplayState()
    state.remember("frames", ["full_path", "label"])
    assert state.recall("frames") == ["full_path", "label"]


def test_remember_an_explicit_empty_list_is_distinct_from_no_choice():
    # An explicit "the researcher unchecked every column" must round-trip
    # as [], never collapse to the same None a table with no choice at
    # all would report.
    state = TableDisplayState()
    state.remember("frames", [])
    assert state.recall("frames") == []
    assert state.recall("frames") is not None


def test_choices_are_kept_separately_per_table():
    state = TableDisplayState()
    state.remember("frames", ["full_path"])
    state.remember("clips", ["clip"])
    assert state.recall("frames") == ["full_path"]
    assert state.recall("clips") == ["clip"]
    assert state.recall("other_table") is None


def test_remember_none_forgets_that_tables_choice():
    state = TableDisplayState()
    state.remember("frames", ["full_path"])
    state.remember("frames", None)
    assert state.recall("frames") is None


def test_remember_stores_a_copy_not_the_callers_list():
    state = TableDisplayState()
    columns = ["full_path"]
    state.remember("frames", columns)
    columns.append("label")
    assert state.recall("frames") == ["full_path"], (
        "mutating the caller's list after remember() must not change "
        "what was remembered"
    )


def test_recall_returns_a_copy_not_internal_state():
    state = TableDisplayState()
    state.remember("frames", ["full_path"])
    result = state.recall("frames")
    result.append("label")
    assert state.recall("frames") == ["full_path"], (
        "mutating what recall() returned must not change what was "
        "remembered"
    )


def test_forget_all_clears_every_table():
    state = TableDisplayState()
    state.remember("frames", ["full_path"])
    state.remember("clips", [])
    state.forget_all()
    assert state.recall("frames") is None
    assert state.recall("clips") is None


# ---------------------------------------------------------------------------
# resolve_default_visible_columns
# ---------------------------------------------------------------------------

def test_default_name_present_is_chosen_over_everything_else():
    result = resolve_default_visible_columns(
        ["label", "full_path", "clip"], "full_path"
    )
    assert result == ["full_path"]


def test_default_name_absent_falls_back_to_first_visual_column():
    result = resolve_default_visible_columns(["clip", "label"], "full_path")
    assert result == ["clip"]


def test_default_name_absent_respects_the_given_column_order():
    result = resolve_default_visible_columns(["label", "clip"], "full_path")
    assert result == ["label"]


def test_no_visual_columns_at_all_returns_empty_list():
    result = resolve_default_visible_columns([], "full_path")
    assert result == []
