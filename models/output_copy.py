"""
models/output_copy.py

P1.9b-1: the copy-on-save plan for operator output files.

docs/architecture.md section 2 (ProjectPaths) names the gap this closes: an
operator writes under `run.paths.outputs_dir`, which points at a workspace
folder until the project is first saved (or at whatever folder held it
before a Save As). `Dataset.save()` only rewrites a media cell's stored
string when the cell's path already lies inside the project folder
(models/dataset.py's `_rewrite_media_cell`) -- it never moves the file on
disk. So a file an operator wrote before the save that is about to happen
would otherwise stay outside the project forever, and any relative-path
rewriting on that cell would be relative to a file that never moved.

Two halves, split the same way `artifacts/cache_sweep.py::plan_sweep` splits
from `ArtifactStore.reconcile_and_evict()`:

  * `plan_output_copy` DECIDES: which files need copying, where each one
    goes, and which destinations already exist with different content
    (a conflict the save must refuse over). It reads the filesystem --
    file sizes on both sides -- but writes nothing.
  * `execute_output_copy` ACTS: copies the files `plan_output_copy` named
    and creates the folders they land in. It never deletes anything --
    the researcher's own output files are not this module's cache to
    evict, unlike the derived-JPEG cache section 4.7 describes.

Both are Qt-free and hold no reference to Dataset or AppController.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import Iterable, Mapping

from media.media_address import MediaAddressError
from media.media_address import parse as parse_address


@dataclasses.dataclass(frozen=True)
class CopyEntry:
    """One file the plan wants copied, src -> dst, both absolute paths."""

    src: Path
    dst: Path
    size_bytes: int


@dataclasses.dataclass(frozen=True)
class CopyConflict:
    """A destination that already exists and does not match the source.

    Item 3 of the work item: same size at the destination is treated as
    "already copied, correctly" and silently skipped (see plan_output_copy);
    only a SIZE MISMATCH becomes a conflict. A byte-for-byte compare would
    be more certain but would mean reading every candidate file twice on
    every save, which is not worth it for a same-machine copy.
    """

    src: Path
    dst: Path
    src_size_bytes: int
    dst_size_bytes: int


@dataclasses.dataclass(frozen=True)
class OutputCopyPlan:
    """What execute_output_copy() must carry out, and what save_project()
    must refuse over.

    entries and conflicts are both already in a stable, deterministic order
    (sorted by destination path) so a test -- and a future UI listing --
    does not depend on dict/set iteration order.
    """

    entries: tuple  # tuple[CopyEntry, ...]
    total_bytes: int
    conflicts: tuple  # tuple[CopyConflict, ...]


def _candidate_paths(
    tables_media_cells: Mapping[str, Iterable[str]], old_root: Path
) -> dict:
    """The distinct source-file paths named by any media cell in any table,
    restricted to those lying under old_root, each mapped to its path
    relative to old_root.

    A media cell is an address (docs/media_architecture.md section 3.2),
    so this parses every cell through media_address.parse() and looks only
    at the PATH portion -- the fragment (#f=, #t=, #r=, a stream selector)
    never affects which file a cell names, and two cells that share a file
    but carry different fragments collapse to the one file here, exactly
    as the "fragment preserved" test expects. A cell that will not parse,
    or whose path is not under old_root -- researcher source media, or a
    path some earlier project state already moved -- is silently excluded;
    this function is a filter, not a validator of every media cell in the
    project.
    """
    found: dict = {}
    for cells in tables_media_cells.values():
        for cell in cells:
            if not cell:
                continue
            try:
                addr = parse_address(cell)
            except MediaAddressError:
                continue
            cell_path = Path(addr.path)
            try:
                relative = cell_path.relative_to(old_root)
            except ValueError:
                continue
            found[cell_path] = relative
    return found


def plan_output_copy(
    tables_media_cells: Mapping[str, Iterable[str]],
    old_outputs_dir,
    new_outputs_dir,
) -> OutputCopyPlan:
    """Decide which operator-output files to copy from old_outputs_dir to
    new_outputs_dir, on the way to a save.

    Args:
        tables_media_cells: {table_name: iterable of raw media cell string
            values} across every stored table -- every value of every
            column save() itself treats as a media path ('full_path' plus
            every column the table's schema tags 'media_path'). Blank
            cells need not be filtered out first; a cell that is not a
            non-blank string is simply not a valid address and is
            excluded the same way an unparseable one is.
        old_outputs_dir: this project's CURRENT ProjectPaths.outputs_dir.
        new_outputs_dir: the outputs_dir the project would have after the
            save this plan is for.

    Selection (item 2 of the work item): a candidate file must be named by
    at least one media cell AND lie under old_outputs_dir. Researcher
    source media -- anything outside old_outputs_dir -- is never a
    candidate, whatever its extension.

    Destination (item 2): each candidate is copied to the SAME path
    relative to new_outputs_dir that it holds relative to old_outputs_dir.

    Collision (item 3): if the destination already exists --
      * same size  -> already there; skipped, not re-copied.
      * different size -> a conflict. Every conflict is collected and
        returned; plan_output_copy raises nothing itself. The caller
        (AppController.save_project()) is the one that refuses the save
        when plan.conflicts is non-empty, BEFORE calling
        execute_output_copy -- so a colliding save copies nothing and
        writes no parquet.

    old_outputs_dir == new_outputs_dir (item 2, a plain save with no
    folder change) returns an empty plan without even reading
    tables_media_cells: there is nothing to move.

    A candidate whose source file no longer exists on disk (deleted by
    the researcher since the operator wrote it) is silently skipped --
    there is nothing to copy, and that is not this planner's problem to
    raise on; the cell will simply fail to resolve at display time,
    exactly as it would without this planner ever having run.
    """
    old_root = Path(old_outputs_dir)
    new_root = Path(new_outputs_dir)

    if old_root == new_root:
        return OutputCopyPlan(entries=(), total_bytes=0, conflicts=())

    candidates = _candidate_paths(tables_media_cells, old_root)

    entries = []
    conflicts = []
    total_bytes = 0
    for cell_path, relative in sorted(
        candidates.items(), key=lambda pair: pair[1].as_posix()
    ):
        if not cell_path.is_file():
            continue
        size = cell_path.stat().st_size
        dst = new_root / relative
        if dst.exists():
            dst_size = dst.stat().st_size
            if dst_size == size:
                continue
            conflicts.append(
                CopyConflict(
                    src=cell_path,
                    dst=dst,
                    src_size_bytes=size,
                    dst_size_bytes=dst_size,
                )
            )
            continue
        entries.append(CopyEntry(src=cell_path, dst=dst, size_bytes=size))
        total_bytes += size

    return OutputCopyPlan(
        entries=tuple(entries),
        total_bytes=total_bytes,
        conflicts=tuple(conflicts),
    )


def execute_output_copy(plan: OutputCopyPlan) -> None:
    """Copy every entry in plan.entries to its destination, creating
    destination folders as needed. Never deletes or overwrites anything
    plan_output_copy did not already name as an entry -- a conflict
    (plan.conflicts) is the caller's to refuse the save over BEFORE this
    is ever called; this function does not look at plan.conflicts at all.

    Runs synchronously and does no cancellation or progress reporting --
    see the "main thread only" comment at this function's call site in
    controller.py.
    """
    for entry in plan.entries:
        entry.dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry.src, entry.dst)
