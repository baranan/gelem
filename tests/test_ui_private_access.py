"""
tests/test_ui_private_access.py

AST guardrail for P0.4's design decision that the ordered query result
belongs to the controller, not to a widget. No file under ui/ may read a
private attribute (a name starting with a single underscore) off anything
other than `self` or `cls`.

This is the test that fails if the row order -- or any other component's
private state -- ever goes back into a widget. It is why TileWidget got a
public get_row_ids(): GalleryWidget used to read TileWidget._tile.

A closed allowlist names the sites P1.13 still owns and has not yet
migrated. Everything else is a failure. Extend the allowlist only when a
rule elsewhere says a site is deliberately deferred; never to silence a
new violation.

Two looseness notes, recorded rather than fixed:
  * the allowlist is keyed by attribute *name*, not by file and line, so
    a new `foo._dataset` anywhere under `ui/` would pass. `CLAUDE.md`'s
    closed-list discipline is file-and-line; this test is looser.
  * `shared_widgets/` is not scanned -- only `ui/`.

The second test here (no widget re-introduces `_row_ids` state) guards
the exact regression `docs/media_architecture.md` section 6.1 names:
the ordered query result belongs to the controller, and a widget that
rebuilt its own `self._row_ids` from `get_row_ids_in_range()` would slip
past every other check in this file and in tests/test_visible_row_order.py.

Style modelled on tests/test_architecture_imports.py.

Run with:
    python -m pytest tests/test_ui_private_access.py
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
UI_DIR = ROOT / "ui"

# fake_controller.py stands in for AppController, not a widget, so it is
# excluded here exactly as it is from the import guardrail.
EXCLUDED_FILES = {"fake_controller.py"}

# Private attribute names that a UI file is still allowed to read off
# another object, because migrating them is explicitly someone else's
# work item (P1.13). This list is closed.
#   _op_registry, _dataset, _active_table -- ui/main_window.py reaching
#       into AppController, tracked under CLAUDE.md's "UI never touches
#       private controller attributes" rule.
#   _group_by -- ui/main_window.py reading the group-by parameter back
#       off an operator instance, where operators/base.py stores it with
#       setattr.
ALLOWLIST = {"_op_registry", "_dataset", "_active_table", "_group_by"}

_GETATTR_BUILTINS = {"getattr", "setattr", "hasattr"}


def _is_private(name: str) -> bool:
    """True for a single-underscore private name, not a dunder or a
    name-mangled __name (both start with '__')."""
    return name.startswith("_") and not name.startswith("__")


def _is_self_or_cls(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id in {"self", "cls"}


def _violations_in_file(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []

    for node in ast.walk(tree):
        # Direct attribute access: <something>._private
        if isinstance(node, ast.Attribute) and _is_private(node.attr):
            if _is_self_or_cls(node.value):
                continue
            if node.attr in ALLOWLIST:
                continue
            hits.append(f"{path.name}:{node.lineno}  reads .{node.attr}")

        # Reflective access: getattr/setattr/hasattr(<something>, "_private")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _GETATTR_BUILTINS
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and _is_private(node.args[1].value)
        ):
            attr_name = node.args[1].value
            if _is_self_or_cls(node.args[0]):
                continue
            if attr_name in ALLOWLIST:
                continue
            hits.append(
                f"{path.name}:{node.lineno}  {node.func.id}(..., {attr_name!r})"
            )

    return hits


def _row_ids_assignments_in_file(path: pathlib.Path) -> list[str]:
    """Lines where the file assigns to an attribute named `_row_ids`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []

    def _targets(node):
        if isinstance(node, ast.Assign):
            return node.targets
        if isinstance(node, ast.AnnAssign):
            return [node.target]
        return []

    for node in ast.walk(tree):
        for target in _targets(node):
            if isinstance(target, ast.Attribute) and target.attr == "_row_ids":
                hits.append(f"{path.name}:{target.lineno}")
    return hits


def test_ui_files_do_not_read_foreign_private_attributes():
    violations: dict[str, list[str]] = {}
    for py_file in sorted(UI_DIR.rglob("*.py")):
        if py_file.name in EXCLUDED_FILES:
            continue
        hits = _violations_in_file(py_file)
        if hits:
            violations[str(py_file.relative_to(ROOT))] = hits

    assert not violations, (
        "UI files read private attributes off objects other than self/cls "
        "(and not on the closed P1.13 allowlist):\n"
        + "\n".join(
            f"  {f}:\n" + "\n".join(f"    {h}" for h in hs)
            for f, hs in violations.items()
        )
    )


def test_no_widget_reintroduces_row_ids_state():
    # P0.4 (docs/media_architecture.md section 6.1): the ordered query
    # result belongs to the controller, not to a widget. A gallery is
    # given an absolute index range into that order and fetches the ids
    # it needs; it must not rebuild its own list. This guards the exact
    # regression the document names -- `GalleryWidget._row_ids` coming
    # back, filled from get_row_ids_in_range() -- which every other
    # check in this file and in test_visible_row_order.py would miss.
    #
    # fake_controller.py is excluded exactly as above: it stands in for
    # AppController, so its own _row_ids (the full row set, like the
    # controller's) is legitimate.
    violations: dict[str, list[str]] = {}
    for py_file in sorted(UI_DIR.rglob("*.py")):
        if py_file.name in EXCLUDED_FILES:
            continue
        hits = _row_ids_assignments_in_file(py_file)
        if hits:
            violations[str(py_file.relative_to(ROOT))] = hits

    assert not violations, (
        "a UI widget assigns `_row_ids` -- the ordered query result "
        "belongs to the controller (docs/media_architecture.md section "
        "6.1), and a widget must not hold its own copy of the row "
        "order:\n"
        + "\n".join(
            f"  {f}: {', '.join(hs)}" for f, hs in violations.items()
        )
    )
