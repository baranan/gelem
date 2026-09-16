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


def test_save_project_refuses_while_a_run_is_live(tmp_path, make_controller):
    """P1.9b-1: Save and Save As are refused outright while any operator
    run is live -- both because the output-copy rewrite would give that
    live run a false "data changed" notice (models/dataset.py's
    _commit_prepared bumps a table's write-ticket version on every
    accepted write, unconditionally), and because a live run keeps
    writing new files under the OLD outputs_dir for as long as it runs,
    so a copy planned now would miss them regardless.

    _register_run() is the existing test helper for a live run with no
    real worker thread (see tests/test_result_delivery.py) -- it only
    inserts a plain dict into AppController._live_runs, exactly what
    is_save_blocked() / save_project() check.
    """
    controller, _dataset, _op_registry = make_controller(tmp_path / "start")

    # A real, non-None ProjectPaths first, via an ordinary unblocked save --
    # so the refusal below is checked against real state, not against the
    # fixture's initial None.
    first_folder = tmp_path / "saved_first"
    controller.save_project(first_folder)
    paths_before = controller._project_paths
    assert paths_before is not None
    assert paths_before.root == first_folder

    controller._register_run("op-1", "Probe", "frames")
    assert controller.is_save_blocked() is True

    new_folder = tmp_path / "saved_here"
    controller.save_project(new_folder)

    # Refused before anything was planned, copied or written: no output
    # folder was even created, and ProjectPaths still names the FIRST save.
    assert not new_folder.exists()
    assert controller._project_paths == paths_before
    assert controller._project_paths.root == first_folder


def test_save_project_rolls_back_cell_rewrite_when_dataset_save_fails(
    tmp_path, make_controller, monkeypatch
):
    """P1.9b-1 follow-up: if Dataset.save() raises AFTER the copy and the
    in-memory cell rewrite have already succeeded, the rewrite must be
    rolled back.

    Without the rollback, a rewritten cell would be left pointing at
    project_path/outputs/... while self._project_paths still names the
    OLD outputs_dir (the swap only happens after Dataset.save() returns).
    A LATER successful save plans its copy against
    self._project_paths.outputs_dir, so that cell would never again be
    recognised as an operator output and could never be brought along --
    permanently stranded outside the reach of any future save.
    """
    controller, dataset, _op_registry = make_controller(tmp_path / "start")

    # A real ProjectPaths with a real outputs_dir, via an ordinary save.
    first_folder = tmp_path / "saved_first"
    controller.save_project(first_folder)
    old_outputs_dir = controller._project_paths.outputs_dir

    # A file shaped like a real operator output under the CURRENT
    # outputs_dir, referenced by one row's media cell -- exactly what
    # plan_output_copy needs to produce a real, non-empty plan.
    op_output = old_outputs_dir / "frames" / "frame_000.jpg"
    op_output.parent.mkdir(parents=True, exist_ok=True)
    op_output.write_bytes(b"fake operator output")

    target_row = dataset.get_table("frames")["row_id"].iloc[0]
    dataset.apply_row_updates(
        "frames", {target_row: {"full_path": str(op_output)}}
    )
    cell_before = dataset.get_row(target_row, "frames")["full_path"]

    monkeypatch.setattr(
        dataset,
        "save",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    second_folder = tmp_path / "saved_second"
    controller.save_project(second_folder)

    # The copy itself ran to completion before save() was ever called
    # (execute_output_copy never deletes anything, and this failure is
    # deliberately left on disk -- see save_project()'s docstring) --
    # but Dataset must show no trace of the attempt.
    assert (second_folder / "outputs" / "frames" / "frame_000.jpg").is_file()
    assert dataset.get_row(target_row, "frames")["full_path"] == cell_before
    assert controller._project_paths.root == first_folder


def test_save_project_rolls_back_rewrite_that_raised_partway_through_the_table_loop(
    tmp_path, make_controller, monkeypatch
):
    """P1.9b-1 follow-up: if Dataset.rewrite_media_cell_paths() itself
    raises after rewriting one table but before reaching the next, the
    already-rewritten table must be rolled back too -- for the same
    reason as the Dataset.save() failure above: self._project_paths has
    not moved, so a table left rewritten would be stranded outside the
    reach of any later save.

    "frames" is rewritten first (Dataset._tables preserves insertion
    order, and "frames" always exists before "segments" is added below),
    then "segments"' own apply_row_updates() call is forced to raise --
    modelling one table succeeding and the next failing, rather than one
    table failing halfway through its own row loop (apply_row_updates()
    already rolls that back internally before re-raising, per its own
    docstring).
    """
    controller, dataset, _op_registry = make_controller(tmp_path / "start")

    first_folder = tmp_path / "saved_first"
    controller.save_project(first_folder)
    old_outputs_dir = controller._project_paths.outputs_dir

    op_output = old_outputs_dir / "frames" / "frame_000.jpg"
    op_output.parent.mkdir(parents=True, exist_ok=True)
    op_output.write_bytes(b"fake operator output")

    target_row = dataset.get_table("frames")["row_id"].iloc[0]
    dataset.apply_row_updates(
        "frames", {target_row: {"full_path": str(op_output)}}
    )
    # A second table sharing the same rewritten cell -- create_table_
    # from_rows() keeps row_ids (CLAUDE.md: not unique ACROSS tables) and
    # copies every column, including the media cell just set above.
    dataset.create_table_from_rows(
        "segments", [target_row], source_table="frames"
    )

    cells_before = {
        "frames": dataset.get_row(target_row, "frames")["full_path"],
        "segments": dataset.get_row(target_row, "segments")["full_path"],
    }

    real_apply_row_updates = dataset.apply_row_updates

    def _raise_for_segments(table_name, updates, **kwargs):
        if table_name == "segments":
            raise RuntimeError("boom")
        return real_apply_row_updates(table_name, updates, **kwargs)

    monkeypatch.setattr(dataset, "apply_row_updates", _raise_for_segments)

    second_folder = tmp_path / "saved_second"
    controller.save_project(second_folder)

    assert dataset.get_row(target_row, "frames")["full_path"] == cells_before["frames"]
    assert dataset.get_row(target_row, "segments")["full_path"] == cells_before["segments"]
    assert controller._project_paths.root == first_folder
