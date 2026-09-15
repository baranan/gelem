"""
tests/test_project_paths.py

models/project_paths.py -- P1.9a. Written from the work-item rules, not
from the implementation:

  - ProjectPaths is a Qt-free, frozen value object with at least root,
    artifacts_dir, outputs_dir and is_workspace.
  - build_project_paths is pure: for a workspace and for a saved folder,
    every directory it computes is inside root.
  - create_workspace creates one new, uniquely named folder under a
    caller-supplied workspaces_root (never a hardcoded constant), with the
    root already present on disk, and is_workspace=True.
  - default_workspaces_root never points at the OS temp directory.

Run with:
    python -m pytest tests/test_project_paths.py
"""

from __future__ import annotations

from pathlib import Path

from models.project_paths import (
    ProjectPaths,
    build_project_paths,
    create_workspace,
    default_workspaces_root,
)


def _assert_every_directory_is_inside_root(paths: ProjectPaths) -> None:
    root = paths.root.resolve()
    for directory in (paths.artifacts_dir, paths.outputs_dir):
        resolved = directory.resolve()
        assert resolved == root or root in resolved.parents, (
            f"{directory} is not inside {paths.root}"
        )


# ---------------------------------------------------------------------------
# build_project_paths: pure, every directory inside root -- both cases.
# ---------------------------------------------------------------------------
def test_workspace_paths_are_all_inside_root(tmp_path):
    root = tmp_path / "workspaces" / "20260101_000000_abcd1234"
    paths = build_project_paths(root, is_workspace=True)

    assert paths.root == root
    assert paths.is_workspace is True
    _assert_every_directory_is_inside_root(paths)


def test_saved_folder_paths_are_all_inside_root(tmp_path):
    root = tmp_path / "my_study"
    paths = build_project_paths(root, is_workspace=False)

    assert paths.root == root
    assert paths.is_workspace is False
    _assert_every_directory_is_inside_root(paths)


def test_builder_touches_no_filesystem(tmp_path):
    # Pure: naming a root that does not exist must not create it.
    root = tmp_path / "never_created"
    build_project_paths(root, is_workspace=True)
    assert not root.exists()


def test_artifacts_and_outputs_dirs_are_distinct():
    paths = build_project_paths(Path("/somewhere/project"), is_workspace=False)
    assert paths.artifacts_dir != paths.outputs_dir


def test_project_paths_is_frozen():
    paths = build_project_paths(Path("/somewhere/project"), is_workspace=False)
    try:
        paths.root = Path("/elsewhere")
    except Exception:
        pass
    else:
        raise AssertionError("ProjectPaths must be frozen")


# ---------------------------------------------------------------------------
# create_workspace: one new, uniquely named folder per call, root created.
# ---------------------------------------------------------------------------
def test_create_workspace_creates_root_on_disk(tmp_path):
    workspaces_root = tmp_path / "workspaces"
    paths = create_workspace(workspaces_root)

    assert paths.is_workspace is True
    assert paths.root.exists()
    assert workspaces_root in paths.root.parents


def test_create_workspace_is_unique_per_call(tmp_path):
    workspaces_root = tmp_path / "workspaces"
    first = create_workspace(workspaces_root)
    second = create_workspace(workspaces_root)
    assert first.root != second.root
    assert first.root.exists()
    assert second.root.exists()


def test_create_workspace_takes_root_as_an_argument_not_a_constant(tmp_path):
    # workspaces_root must be a genuine argument -- a caller (a test, or a
    # future setting) can point it anywhere, including under tmp_path.
    custom_root = tmp_path / "custom_location"
    paths = create_workspace(custom_root)
    assert custom_root in paths.root.parents


# ---------------------------------------------------------------------------
# default_workspaces_root: never the OS temp directory.
# ---------------------------------------------------------------------------
def test_default_workspaces_root_is_not_the_temp_directory():
    import tempfile

    root = default_workspaces_root()
    temp_root = Path(tempfile.gettempdir()).resolve()
    resolved = root.resolve()
    assert resolved != temp_root
    assert temp_root not in resolved.parents
