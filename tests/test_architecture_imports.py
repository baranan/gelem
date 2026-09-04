"""
tests/test_architecture_imports.py

Architectural import boundary checks.
These tests verify that components do not import from layers they should not
depend on. Run with: pytest tests/test_architecture_imports.py

Rules enforced:
  - column_types/renderers.py must not import from ui.*
  - operators/operator_registry.py must not import from ui.*
  - UI tile files must not import data-processing libraries at module level
    (pandas, PIL, cv2, mediapipe, numpy belong in operators/models, not UI)
  - models/dataset.py must not import from column_types.* (P1.8d-2b-1:
    Dataset holds no reference to ColumnTypeRegistry)
"""

import ast
import sys
import pathlib
import pytest

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _imported_modules(path: pathlib.Path) -> list[str]:
    """Return all module names imported by a Python file (top-level imports)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_renderers_do_not_import_ui():
    """column_types/renderers.py must not import from the UI layer."""
    modules = _imported_modules(ROOT / "column_types" / "renderers.py")
    ui_imports = [m for m in modules if m.startswith("ui")]
    assert not ui_imports, f"renderers.py imports UI modules: {ui_imports}"


def test_operator_registry_does_not_import_ui():
    """operators/operator_registry.py must not import from the UI layer."""
    modules = _imported_modules(ROOT / "operators" / "operator_registry.py")
    ui_imports = [m for m in modules if m.startswith("ui")]
    assert not ui_imports, f"operator_registry.py imports UI modules: {ui_imports}"


def test_ui_tiles_do_not_import_data_libs():
    """UI tile files must not import data-processing libraries at module level."""
    forbidden = {"pandas", "PIL", "cv2", "mediapipe", "numpy"}
    violations: dict[str, list[str]] = {}
    for py_file in (ROOT / "ui" / "tiles").glob("*.py"):
        hits = [
            m for m in _imported_modules(py_file)
            if any(m == f or m.startswith(f + ".") for f in forbidden)
        ]
        if hits:
            violations[py_file.name] = hits
    assert not violations, f"UI tile files import data libs: {violations}"


def test_dataset_does_not_import_column_types():
    """models/dataset.py must not import from column_types.* -- P1.8d-2b-1
    removed Dataset's reference to ColumnTypeRegistry entirely; a column's
    display type tag lives on the table's own TableSchema instead."""
    modules = _imported_modules(ROOT / "models" / "dataset.py")
    registry_imports = [
        m for m in modules if m == "column_types" or m.startswith("column_types.")
    ]
    assert not registry_imports, (
        f"models/dataset.py imports column_types: {registry_imports}"
    )


def test_ui_widgets_do_not_import_data_libs():
    """Top-level UI widget files must not import data-processing libraries."""
    forbidden = {"pandas", "PIL", "cv2", "mediapipe", "numpy"}
    violations: dict[str, list[str]] = {}
    for py_file in (ROOT / "ui").glob("*.py"):
        if py_file.name == "fake_controller.py":
            continue
        hits = [
            m for m in _imported_modules(py_file)
            if any(m == f or m.startswith(f + ".") for f in forbidden)
        ]
        if hits:
            violations[py_file.name] = hits
    assert not violations, f"UI widget files import data libs: {violations}"
