"""
artifacts/cache_sweep.py

The PURE planning half of the artifact cache directory sweep
(P0.5b-2ii-b2).

`plan_sweep` performs no I/O, imports nothing that touches the
filesystem, and holds no reference to `ArtifactStore`. Given the
sweep-owned files found on disk and the set of paths the index currently
names, it decides:

  * which files to delete -- orphans (on disk, not in the index) plus
    whatever the on-disk ceiling forces out;
  * which index paths name a file that is no longer on disk, so their
    entries can be dropped.

`ArtifactStore.reconcile_and_evict()` does the directory walk, the
deletes and the index surgery. This module only decides, so every
ordering and ceiling rule is testable with plain records and no temp
directory.

`docs/media_architecture.md` section 4.7 is the authority for the sweep
as a whole.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class SweepFile:
    """One sweep-owned file found in the artifacts directory.

    `path` is whatever opaque, hashable value the caller uses to name a
    file -- `ArtifactStore` passes a resolved `Path`. `size_bytes` and
    `mtime_ns` come from the `os.scandir` DirEntry stat the caller
    already did during the walk, so building the plan costs no extra
    syscalls.
    """

    path: object
    size_bytes: int
    mtime_ns: int


@dataclasses.dataclass(frozen=True)
class SweepPlan:
    """What `reconcile_and_evict()` must carry out.

    `files_to_delete` is in delete order: orphans first (unless
    `delete_orphans` was False, in which case there are none), then the
    ceiling evictions oldest-first. `paths_missing` are index paths with
    no file on disk -- always populated, whatever `delete_orphans` was.
    `bytes_before` is the total size of every sweep-owned file found;
    `bytes_after` is what remains once the plan is executed -- both are
    for the log line and the tests.
    """

    files_to_delete: tuple
    paths_missing: tuple
    bytes_before: int
    bytes_after: int


def plan_sweep(
    entries: Iterable[SweepFile],
    indexed_paths: set,
    max_bytes: int,
    delete_orphans: bool = True,
) -> SweepPlan:
    """Decide the sweep.

    Args:
        entries:        the sweep-owned files found in the directory.
        indexed_paths:  the set of paths the index currently names.
        max_bytes:      the on-disk ceiling the survivors must fit under,
                        measured after orphan removal.
        delete_orphans: when False, the plan deletes NO orphan -- only
                        ceiling evictions of indexed files. The caller
                        passes False when it cannot trust that an empty
                        or partial index really means "nothing else is
                        cached" (a load whose `artifact_index.json` was
                        missing or unreadable). `paths_missing` is still
                        computed, so the missing-file reconciliation half
                        runs either way.

    Decisions, in this order:

      1. Orphans -- a file on disk that the index does not name. When
         `delete_orphans` is True they always go: they are unreachable,
         the on-disk name is a one-way hash of the ArtifactKey, so
         nothing can ever look them up again. When it is False they are
         left on disk untouched.
      2. Index paths with no file on disk are recorded in
         `paths_missing`. Their entries must be dropped -- otherwise
         `is_cached()` reports True for a picture that is not there,
         demand-driven display queues no request, and the tile stays a
         permanent grey placeholder (the grey-tile defect).
      3. If the files that survived step 1 still exceed `max_bytes`,
         evict them oldest first until the total is under the ceiling.

    "Oldest" is by mtime -- WRITE order, not access order. No per-file
    access time is tracked: the artifact cache is read on the paint
    path, and giving every read a timestamp write there would put a
    write on the hot path for a signal that does not survive a restart
    anyway. So a file written long ago and shown every day is evicted
    before a file written yesterday and never shown again. For a display
    cache that regenerates a miss in one worker pass, that trade is
    deliberate.
    """
    # Materialise once -- `entries` may be a generator and we iterate it
    # several times below.
    found = list(entries)

    # Total size of everything the sweep owns, before any deletion.
    bytes_before = sum(entry.size_bytes for entry in found)

    # Step 1: split found files into orphans (not named by the index)
    # and indexed survivors.
    orphans = [entry for entry in found if entry.path not in indexed_paths]
    survivors = [entry for entry in found if entry.path in indexed_paths]

    # Step 2: index paths with no matching file on disk. Computed
    # regardless of delete_orphans -- reconciliation always runs.
    present_paths = {entry.path for entry in found}
    paths_missing = tuple(
        path for path in indexed_paths if path not in present_paths
    )

    # Running total after step 1. Orphans only leave if delete_orphans.
    if delete_orphans:
        bytes_after = sum(entry.size_bytes for entry in survivors)
    else:
        bytes_after = bytes_before

    # Step 3: enforce the ceiling against the indexed survivors, oldest
    # mtime first. Write order, not access order -- see the docstring.
    ceiling_evictions = []
    if bytes_after > max_bytes:
        for entry in sorted(survivors, key=lambda item: item.mtime_ns):
            if bytes_after <= max_bytes:
                break
            ceiling_evictions.append(entry)
            bytes_after -= entry.size_bytes

    orphan_deletions = orphans if delete_orphans else []
    files_to_delete = tuple(entry.path for entry in orphan_deletions) + tuple(
        entry.path for entry in ceiling_evictions
    )

    return SweepPlan(
        files_to_delete=files_to_delete,
        paths_missing=paths_missing,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
    )
