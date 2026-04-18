"""
tests/test_operator_registry_boundaries.py

Verifies that OperatorRegistry respects component boundaries:

  - No access to private Dataset attributes (dataset._registry etc.)
  - run_create_columns receives pre-snapshotted work_items, not a live Dataset
  - The background worker never reads from Dataset directly

These are source-inspection tests (AST-based), so they run without Qt or
real data. They will catch architectural drift the moment a student copies
the old pattern.

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
    """run_create_columns must receive work_items, not a raw Dataset instance.

    Passing Dataset into the registry gives the worker thread access to live
    model state. AppController must snapshot rows on the main thread first.
    """
    args = _method_arg_names("run_create_columns")
    assert args is not None, "run_create_columns not found in operator_registry.py"
    assert "dataset" not in args, (
        "run_create_columns still accepts 'dataset' — "
        "replace with work_items (pre-snapshotted by AppController)"
    )


def test_run_create_columns_accepts_work_items():
    """run_create_columns must accept work_items as its row input."""
    args = _method_arg_names("run_create_columns")
    assert args is not None, "run_create_columns not found in operator_registry.py"
    assert "work_items" in args, (
        "run_create_columns does not have a 'work_items' parameter — "
        "the worker should receive pre-snapshotted dicts, not row_ids + dataset"
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
        "use the 'row_data' field from work_items instead"
    )
