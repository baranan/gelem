"""
tests/test_operator_registry_boundaries.py

Verifies that OperatorRegistry respects component boundaries:

  - No access to private Dataset attributes (dataset._registry etc.)
  - run_create_columns receives a pre-snapshotted DataFrame plus ordered
    row_ids and a table_name, not a live Dataset
  - The background worker never reads from Dataset directly

These are source-inspection tests (AST-based), so they run without Qt or
real data. They will catch architectural drift the moment a student copies
the old pattern.

P0.2a note: run_create_columns used to take work_items: list[dict], one
pre-built dict per row. That shape is exactly what P0.2a (see CLAUDE.md /
docs/media_architecture.md §6.1 item 4) replaces, because building 530,000
dicts before a run starts is the defect being fixed. The test that used to
assert "work_items" is present was asserting the old shape, so it is
updated here to assert the new one (snapshot DataFrame + row_ids +
table_name) rather than kept passing vacuously under a repurposed name.

Run with: pytest tests/test_operator_registry_boundaries.py
"""

import ast
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

REGISTRY_FILE = ROOT / "operators" / "operator_registry.py"


def _source() -> str:
    return REGISTRY_FILE.read_text(encoding="utf-8")


def _method_arg_names(method_name: str) -> list[str] | None:
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return [a.arg for a in node.args.args]
    return None


def test_no_private_dataset_attribute_access():
    """OperatorRegistry must not reach into any private Dataset attribute.

    Accessing dataset._registry (or any _private attribute on dataset) bypasses
    the defined interface. Column registration must flow through AppController.
    """
    src = _source()
    assert "._registry" not in src, (
        "operator_registry.py accesses ._registry — "
        "column registration must go through AppController.run_create_columns()"
    )


def test_run_create_columns_does_not_accept_dataset():
    """run_create_columns must receive a pre-snapshotted DataFrame, not a
    raw Dataset instance.

    Passing Dataset into the registry gives the worker thread access to live
    model state. AppController must snapshot rows on the main thread first.
    """
    args = _method_arg_names("run_create_columns")
    assert args is not None, "run_create_columns not found in operator_registry.py"
    assert "dataset" not in args, (
        "run_create_columns still accepts 'dataset' — "
        "replace with a snapshot DataFrame (pre-snapshotted by AppController)"
    )


def test_run_create_columns_accepts_snapshot_row_ids_and_table_name():
    """run_create_columns must accept a pre-snapshotted DataFrame plus
    the ordered row_ids and table_name as its row input (P0.2a)."""
    args = _method_arg_names("run_create_columns")
    assert args is not None, "run_create_columns not found in operator_registry.py"
    for required in ("snapshot", "row_ids", "table_name"):
        assert required in args, (
            f"run_create_columns is missing '{required}' — "
            f"the worker should receive one pre-snapshotted DataFrame "
            f"plus ordered row_ids and table_name, not per-row dicts "
            f"built in advance"
        )


def test_worker_does_not_call_dataset_get_row():
    """The background worker must not read from Dataset.

    Row data must be snapshotted by AppController before the thread starts.
    Any call to dataset.get_row() inside the worker re-introduces the
    live-state coupling this architecture is designed to prevent.
    """
    src = _source()
    assert "dataset.get_row" not in src, (
        "_run_create_columns_worker calls dataset.get_row() — "
        "use the pre-snapshotted DataFrame passed to run_create_columns instead"
    )
