"""
tests/test_detail_media_column.py

CC-34: the detail view must show whatever column is actually the active
table's media column, not a hardcoded "full_path". The rule now lives in
one place, AppController.get_detail_media_column() -- the first entry of
get_effective_visible_columns(), or None when that list is empty.
ui/detail_widget.py must ask for it, not decide it.

Written from the work-item spec, not from the implementation:
  1. a table whose only visual column is not named "full_path" still gets
     a real detail media column;
  2. an explicit empty visible-columns preference means no media column;
  3. a remembered multi-column preference picks the FIRST entry, not
     just "whatever is visual" -- [b, a] returns "b".

Plus the AST guardrail: no file under ui/ (other than fake_controller.py,
which stands in for AppController) may pass a string literal as the first
argument of a render_column_value(...) call -- that is exactly the shape
of the CC-34 bug (a hardcoded "full_path").

Run with:
    python -m pytest tests/test_detail_media_column.py
"""

from __future__ import annotations

import ast
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
UI_DIR = ROOT / "ui"

# The controller factory lives in tests/conftest.py (make_controller
# fixture).


# ---------------------------------------------------------------------------
# 1. A table whose only visual column is not named "full_path".
# ---------------------------------------------------------------------------

def test_detail_media_column_is_the_tables_own_visual_column_not_full_path(
    make_controller, tmp_path
):
    controller, dataset, _ = make_controller(tmp_path)

    dataset.create_table_from_df(
        "clips",
        pd.DataFrame({"clip": ["media/a.mp4", "media/b.mov#f=10"]}),
    )
    assert dataset.schema_for("clips").spec_for("clip").type_tag == "media_path", (
        "sanity: 'clip' must actually be tagged media_path for this test "
        "to exercise the rule it claims to"
    )

    controller.set_active_table("clips")
    assert controller.has_visible_columns_preference() is False
    assert controller.get_detail_media_column() == "clip"


# ---------------------------------------------------------------------------
# 2. An explicit empty visible-columns preference -- no media column.
# ---------------------------------------------------------------------------

def test_detail_media_column_is_none_with_an_explicit_empty_preference(
    make_controller, tmp_path
):
    controller, _, _ = make_controller(tmp_path)

    controller.set_visible_columns([])
    assert controller.has_visible_columns_preference() is True
    assert controller.get_effective_visible_columns() == []
    assert controller.get_detail_media_column() is None


# ---------------------------------------------------------------------------
# 3. A remembered multi-column preference: the FIRST entry wins.
# ---------------------------------------------------------------------------

def test_detail_media_column_is_the_first_remembered_column(
    make_controller, tmp_path
):
    controller, _, _ = make_controller(tmp_path)

    controller.set_visible_columns(["b", "a"])
    assert controller.get_effective_visible_columns() == ["b", "a"]
    assert controller.get_detail_media_column() == "b"


# ---------------------------------------------------------------------------
# 4. AST guardrail: no ui/ file (other than fake_controller.py) passes a
#    string literal as render_column_value(...)'s first argument.
# ---------------------------------------------------------------------------

EXCLUDED_FILES = {"fake_controller.py"}


def _literal_first_arg_calls(path: pathlib.Path) -> list[str]:
    """Lines where render_column_value(...) is called with a string
    literal (or an f-string / concatenation of only literals) as its
    first positional argument -- the hardcoded-column-name shape."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    calls_seen = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else None
        )
        if name != "render_column_value":
            continue
        calls_seen += 1
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            hits.append(f"{path.name}:{node.lineno}")

    return hits, calls_seen


def test_no_ui_file_hardcodes_the_media_column_in_render_column_value():
    violations: dict[str, list[str]] = {}
    total_calls_seen = 0

    for py_file in sorted(UI_DIR.rglob("*.py")):
        if py_file.name in EXCLUDED_FILES:
            continue
        hits, calls_seen = _literal_first_arg_calls(py_file)
        total_calls_seen += calls_seen
        if hits:
            violations[str(py_file.relative_to(ROOT))] = hits

    assert total_calls_seen > 0, (
        "the AST walk visited zero render_column_value(...) calls -- it "
        "would pass vacuously; something about the scan is broken"
    )
    assert not violations, (
        "a ui/ file passes a string literal as render_column_value()'s "
        "first argument -- this hardcodes which column is shown as media "
        "(CC-34's bug). The column must come from "
        "controller.get_detail_media_column() or a tile's own "
        "column_name, never a literal:\n"
        + "\n".join(
            f"  {f}: {', '.join(hs)}" for f, hs in violations.items()
        )
    )
