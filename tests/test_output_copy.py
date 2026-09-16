"""
tests/test_output_copy.py

P1.9b-1 -- the copy-on-save plan and executor for operator output files.

docs/architecture.md section 2 (ProjectPaths) names the gap: an operator
writes under `run.paths.outputs_dir`, which points at a workspace folder
until the project is first saved, or at whatever folder held it before a
Save As. `models/output_copy.py::plan_output_copy` decides which of those
files need to move into the newly chosen project folder;
`execute_output_copy` does the actual copying.

Written from the work-item spec, not from the implementation. New file on
purpose: tests/test_dataset.py runs everything in it twice.

Run with:
    python -m pytest tests/test_output_copy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.output_copy import (
    CopyConflict,
    CopyEntry,
    OutputCopyPlan,
    execute_output_copy,
    plan_output_copy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def test_plan_selects_only_files_under_old_outputs_dir(tmp_path):
    old_outputs = tmp_path / "workspace" / "outputs"
    new_outputs = tmp_path / "project" / "outputs"

    op_output = old_outputs / "frames" / "run_1" / "frame_000.jpg"
    _write(op_output, b"op output bytes")

    source_media = tmp_path / "source_footage" / "clip.mp4"
    _write(source_media, b"researcher's own video")

    cells = {
        "frames": [
            str(op_output),
            str(source_media),
        ]
    }

    plan = plan_output_copy(cells, old_outputs, new_outputs)

    assert len(plan.entries) == 1
    assert plan.entries[0].src == op_output
    assert plan.entries[0].dst == new_outputs / "frames" / "run_1" / "frame_000.jpg"
    assert plan.entries[0].size_bytes == len(b"op output bytes")
    assert plan.total_bytes == len(b"op output bytes")
    assert plan.conflicts == ()


def test_plan_excludes_source_media_entirely(tmp_path):
    # A cell that never lies under old_outputs_dir contributes nothing to
    # the plan, whatever its extension or apparent media-ness.
    old_outputs = tmp_path / "workspace" / "outputs"
    new_outputs = tmp_path / "project" / "outputs"
    old_outputs.mkdir(parents=True)
    new_outputs.mkdir(parents=True)

    source_media = tmp_path / "source_footage" / "clip.mp4"
    _write(source_media, b"researcher's own video")

    cells = {"frames": [str(source_media)]}

    plan = plan_output_copy(cells, old_outputs, new_outputs)

    assert plan.entries == ()
    assert plan.total_bytes == 0
    assert plan.conflicts == ()


def test_plan_is_empty_when_old_and_new_outputs_dir_are_equal(tmp_path):
    outputs = tmp_path / "project" / "outputs"
    op_output = outputs / "frames" / "frame_000.jpg"
    _write(op_output, b"op output bytes")

    cells = {"frames": [str(op_output)]}

    plan = plan_output_copy(cells, outputs, outputs)

    assert plan.entries == ()
    assert plan.total_bytes == 0
    assert plan.conflicts == ()


def test_plan_skips_a_destination_that_already_matches_by_size(tmp_path):
    old_outputs = tmp_path / "workspace" / "outputs"
    new_outputs = tmp_path / "project" / "outputs"

    op_output = old_outputs / "frames" / "frame_000.jpg"
    _write(op_output, b"same bytes")

    already_there = new_outputs / "frames" / "frame_000.jpg"
    _write(already_there, b"same bytes")  # identical size

    cells = {"frames": [str(op_output)]}

    plan = plan_output_copy(cells, old_outputs, new_outputs)

    assert plan.entries == ()
    assert plan.total_bytes == 0
    assert plan.conflicts == ()


def test_plan_reports_a_conflict_when_destination_size_differs(tmp_path):
    old_outputs = tmp_path / "workspace" / "outputs"
    new_outputs = tmp_path / "project" / "outputs"

    op_output = old_outputs / "frames" / "frame_000.jpg"
    _write(op_output, b"the current bytes, 20 long")

    colliding = new_outputs / "frames" / "frame_000.jpg"
    _write(colliding, b"different content")  # different size

    cells = {"frames": [str(op_output)]}

    plan = plan_output_copy(cells, old_outputs, new_outputs)

    assert plan.entries == ()
    assert plan.total_bytes == 0
    assert len(plan.conflicts) == 1
    conflict = plan.conflicts[0]
    assert conflict.src == op_output
    assert conflict.dst == colliding
    assert conflict.src_size_bytes == len(b"the current bytes, 20 long")
    assert conflict.dst_size_bytes == len(b"different content")


def test_plan_deduplicates_two_cells_that_share_a_file_with_different_fragments(
    tmp_path,
):
    # Two rows can each hold an address into the SAME operator-output
    # video with a different time fragment. The path portion -- not the
    # fragment -- decides which file gets copied, so this must produce
    # exactly one entry, not two, and the fragment must not leak into the
    # destination path.
    old_outputs = tmp_path / "workspace" / "outputs"
    new_outputs = tmp_path / "project" / "outputs"

    op_output = old_outputs / "frames" / "clip.mp4"
    _write(op_output, b"video bytes")

    cell_a = str(op_output) + "#t=1.000000-2.000000"
    cell_b = str(op_output) + "#f=42"

    cells = {"frames": [cell_a], "segments": [cell_b]}

    plan = plan_output_copy(cells, old_outputs, new_outputs)

    assert len(plan.entries) == 1
    entry = plan.entries[0]
    assert entry.src == op_output
    assert entry.dst == new_outputs / "frames" / "clip.mp4"
    assert "#" not in str(entry.dst)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def test_execute_copies_every_entry_and_creates_folders(tmp_path):
    src = tmp_path / "old" / "outputs" / "frames" / "run_1" / "frame_000.jpg"
    dst = tmp_path / "new" / "outputs" / "frames" / "run_1" / "frame_000.jpg"
    _write(src, b"jpeg bytes")

    plan = OutputCopyPlan(
        entries=(CopyEntry(src=src, dst=dst, size_bytes=len(b"jpeg bytes")),),
        total_bytes=len(b"jpeg bytes"),
        conflicts=(),
    )

    execute_output_copy(plan)

    assert dst.is_file()
    assert dst.read_bytes() == b"jpeg bytes"
    # The source is untouched -- this is a copy, not a move.
    assert src.is_file()
    assert src.read_bytes() == b"jpeg bytes"


def test_execute_never_deletes_anything(tmp_path):
    src = tmp_path / "old" / "outputs" / "a.jpg"
    dst = tmp_path / "new" / "outputs" / "a.jpg"
    _write(src, b"a")

    # A file that has nothing to do with this plan, sitting in the
    # destination folder already. execute_output_copy must leave it
    # exactly alone -- it never deletes, whatever it finds on disk.
    unrelated = tmp_path / "new" / "outputs" / "unrelated.jpg"
    _write(unrelated, b"untouched")

    plan = OutputCopyPlan(
        entries=(CopyEntry(src=src, dst=dst, size_bytes=1),),
        total_bytes=1,
        conflicts=(),
    )

    execute_output_copy(plan)

    assert dst.is_file()
    assert unrelated.is_file()
    assert unrelated.read_bytes() == b"untouched"
