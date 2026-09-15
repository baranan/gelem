"""
tests/test_controller_project_paths.py

P1.9a: AppController holds the current ProjectPaths and replaces it
wholesale on save_project() and load_project(), at the same point it
already re-roots ArtifactStore (docs/media_architecture.md section 4.7).

_project_root (media-cell resolution) is a DIFFERENT piece of state with
its own rule, unchanged by this item: load_project() moves it because
Dataset.load() absolutises every relative media cell against the new
root, but save_project() deliberately does not -- Dataset.save() only
rewrites the ON-DISK copy, leaving the in-memory cells relative to
whatever root they already resolve against, so moving _project_root at
save time would break resolution of any cell that is still relative. See
the comment in AppController.save_project(). This file tests both facts:
save_project() moves ProjectPaths but leaves _project_root alone;
load_project() moves both.

Uses the make_controller fixture (tests/conftest.py) -- a real Dataset
over test_images, no Qt beyond what ArtifactStore/ColumnTypeRegistry
already need.

Run with:
    python -m pytest tests/test_controller_project_paths.py
"""

from __future__ import annotations

from pathlib import Path


def test_save_project_updates_project_paths_but_not_project_root(
    tmp_path, make_controller
):
    controller, _dataset, _op_registry = make_controller(tmp_path / "start")
    root_before_save = controller._project_root

    new_folder = tmp_path / "saved_here"
    controller.save_project(new_folder)

    assert controller._project_paths.root == new_folder
    assert controller._project_paths.is_workspace is False
    # _project_root answers a different question (media-cell resolution)
    # and save_project() deliberately does not move it -- see the module
    # docstring above and the comment in AppController.save_project().
    assert controller._project_root == root_before_save


def test_load_project_updates_both_project_paths_and_project_root(
    tmp_path, make_controller
):
    controller, _dataset, _op_registry = make_controller(tmp_path / "start")

    # A second, distinct folder with a real saved project to load back --
    # save_project() is the simplest way to produce one.
    other_folder = tmp_path / "loaded_from_here"
    controller.save_project(other_folder)

    # Move ProjectPaths and _project_root somewhere else first, so the
    # assertions below can only pass if load_project() actually moved
    # them -- not because they already happened to be right.
    controller.save_project(tmp_path / "somewhere_else")
    assert controller._project_paths.root != other_folder
    assert controller._project_root != other_folder

    controller.load_project(other_folder)

    assert controller._project_paths.root == other_folder
    assert controller._project_paths.is_workspace is False
    assert controller._project_root == Path(other_folder)


def test_project_paths_outputs_and_artifacts_dirs_are_under_the_new_root(
    tmp_path, make_controller
):
    controller, _dataset, _op_registry = make_controller(tmp_path / "start")

    new_folder = tmp_path / "saved_here"
    controller.save_project(new_folder)

    paths = controller._project_paths
    assert new_folder in paths.artifacts_dir.parents
    assert new_folder in paths.outputs_dir.parents
