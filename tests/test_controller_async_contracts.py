"""
tests/test_controller_async_contracts.py

Checks the async queue contract in AppController:

  - _on_item_complete carries operation_id and table_name (not just row_id)
  - _drain_queues uses the table_name from the payload, not self._active_table
  - The enqueued tuple includes all four fields

These are source-inspection tests (AST-based), so they run without Qt or
any real data and catch contract drift immediately.

Run with: pytest tests/test_controller_async_contracts.py
"""

import ast
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CONTROLLER = ROOT / "controller.py"


def _method_source(method_name: str) -> str | None:
    source = CONTROLLER.read_text(encoding="utf-8")
    tree   = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return ast.get_source_segment(source, node)
    return None


def _method_arg_names(method_name: str) -> list[str] | None:
    tree = ast.parse(CONTROLLER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return [a.arg for a in node.args.args]
    return None


def test_on_item_complete_carries_operation_id():
    """_on_item_complete must accept operation_id for future stale-result detection."""
    args = _method_arg_names("_on_item_complete")
    assert args is not None, "_on_item_complete not found in controller.py"
    assert "operation_id" in args, (
        f"_on_item_complete is missing 'operation_id' — got args: {args}"
    )


def test_on_item_complete_carries_table_name():
    """_on_item_complete must accept table_name so results land in the right table."""
    args = _method_arg_names("_on_item_complete")
    assert args is not None, "_on_item_complete not found in controller.py"
    assert "table_name" in args, (
        f"_on_item_complete is missing 'table_name' — got args: {args}"
    )


def test_on_item_complete_enqueues_all_four_fields():
    """_on_item_complete must enqueue (operation_id, table_name, row_id, result)."""
    src = _method_source("_on_item_complete")
    assert src is not None, "_on_item_complete not found in controller.py"
    for field in ("operation_id", "table_name", "row_id", "result"):
        assert field in src, (
            f"_on_item_complete does not reference '{field}' in its append — "
            f"the full 4-tuple must be enqueued"
        )


def test_drain_queues_uses_payload_table_not_active_table():
    """_drain_queues must apply updates using table_name from the payload, not self._active_table.

    If _drain_queues reads self._active_table, results written while the user has
    switched tables will land in the wrong table.
    """
    src = _method_source("_drain_queues")
    assert src is not None, "_drain_queues not found in controller.py"
    assert "_active_table" not in src, (
        "_drain_queues references self._active_table — "
        "use the table_name field from the result payload instead"
    )
