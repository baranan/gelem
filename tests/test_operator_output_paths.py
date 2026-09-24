"""
tests/test_operator_output_paths.py

Guardrail behind P1.9a: an operator writes files only under
run.paths.outputs_dir (operators/CLAUDE.md, "Write only to run.paths"),
never into a directory it chose itself.

Before this item, four operators (plot_advanced, plot,
mean_face, blendshape_avatar) each captured a construction-time
self._output_dir -- a repo-relative gelem_project/ folder, or a system
Temp folder built from tempfile.gettempdir() -- and wrote there regardless
of which project was open (docs/review/p1.9-survey.md).

This test parses every module under operators/ with ast -- not a text
search, which a comment or docstring mentioning one of these names would
trip -- and asserts none of them:
  - references an attribute named _output_dir (the retired pattern),
  - imports tempfile, or references gettempdir,
  - references cwd (Path.cwd()), or
  - references getcwd (os.getcwd()).

It also asserts the walk actually found modules, so a broken glob cannot
let the check pass by inspecting nothing.

Run with:
    python -m pytest tests/test_operator_output_paths.py
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
OPERATORS_DIR = ROOT / "operators"

# Every name whose appearance anywhere in an operator module means it is
# reaching for a scratch directory instead of run.paths.outputs_dir.
_FORBIDDEN_NAMES = ("_output_dir", "tempfile", "gettempdir", "cwd", "getcwd")


def _operator_modules() -> list[pathlib.Path]:
    """Every *.py file directly under operators/."""
    return sorted(OPERATORS_DIR.glob("*.py"))


def _forbidden_references(path: pathlib.Path) -> list[str]:
    """Return every forbidden name referenced anywhere in one module's AST
    -- as an attribute access (self._output_dir, Path.cwd, os.getcwd), a
    bare name, or an import -- so a nested `import tempfile` inside a
    method body is caught the same as a module-level one.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_NAMES:
            hits.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            hits.append(node.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _FORBIDDEN_NAMES:
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module in _FORBIDDEN_NAMES:
                hits.append(node.module)
            for alias in node.names:
                if alias.name in _FORBIDDEN_NAMES:
                    hits.append(alias.name)

    return hits


def test_the_walk_finds_operator_modules():
    # A guardrail that inspects nothing passes vacuously. Pin a floor well
    # below today's count so adding or removing one operator does not make
    # this test brittle, but an empty glob does fail it.
    modules = _operator_modules()
    assert len(modules) >= 8, (
        f"expected to find several operator modules under {OPERATORS_DIR}, "
        f"found {len(modules)}: {[m.name for m in modules]}"
    )


def test_no_operator_module_references_a_scratch_directory():
    violations: dict[str, list[str]] = {}
    for path in _operator_modules():
        hits = _forbidden_references(path)
        if hits:
            violations[path.name] = hits

    assert not violations, (
        "operator modules must write only under run.paths.outputs_dir "
        "(operators/CLAUDE.md, \"Write only to run.paths\") -- these "
        "reference a retired scratch-directory pattern instead:\n"
        + "\n".join(
            f"  {name}: {', '.join(hits)}" for name, hits in violations.items()
        )
    )
