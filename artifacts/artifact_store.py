"""
artifacts/artifact_store.py

ArtifactStore manages thumbnail and preview images used by the gallery
for fast display. It handles both image and video source files.

For images: thumbnails are generated using PIL.
For videos:  the first frame is extracted using OpenCV, then the same
             PIL-based resizing pipeline is applied.

Identity (P0.5b-1, docs/media_architecture.md section 4.5). A derived
image is identified by an `ArtifactKey` -- canonical media address, source
fingerprint, purpose, resolution, representative-frame policy and renderer
cache version -- NOT by the row that asked for it. The row, table and
column name the UI subscriber waiting for the picture; they are not part
of the key. This is what lets two media columns on one row, and two
tables pointing at one file, behave correctly.

Fingerprint memo (a design decision for P0.5b-1, not spelled out in 4.5).
The fingerprint is part of the key, but a cache lookup on the paint path
must not call `stat()`. So ArtifactStore keeps a memo of
`canonical address -> (size, mtime)`. Nothing in THIS file stats a source
on a cache lookup: the source stat happens only on a worker thread, inside
`_run_job` (via `_stat_fingerprint`). The demand-driven caller
(`AppController.render_column_value` -> `_queue_thumbnail_request`,
P0.5b-3i) does no `exists()` check either -- a missing source is the
worker's business. An address in the memo is one of three states:

  * **absent** -- a lookup misses. Nothing is served.
  * **seeded-unverified** -- put there by `load_index` from the persisted
    (size, mtime). A lookup IS served from it, so a reopened project shows
    its cached pictures at once instead of decoding every visible source
    image on the main thread. The trade is a briefly stale preview if the
    source changed since the save; the next `request_thumbnail` for that
    address re-stats, gets a different fingerprint, and regenerates. For
    display (not analysis) that trade is deliberate.
  * **verified** -- written by `_run_job`'s commit after a fresh stat this
    session. A lookup is served, and a duplicate `request_thumbnail`
    short-circuits synchronously without queuing a worker.

`load_index`'s docstring is the authority on the seeded-unverified case.

Reading and writing the JPEGs themselves goes through `ArtifactCodec`,
which is the boundary `CLAUDE.md`'s media rules name: derived artifacts
are encoded and read back only by the codec. The matching half -- source
media decoded only by the resolver, so nothing else opens an image at all
-- waits on P1.2; until then `_decode_source` still decodes source media
here.

Request queue (P0.5b-2i, docs/media_architecture.md section 4.4). A
request is served off a bounded `WorkerPool` rather than a raw thread per
call. Requests naming the same canonical address before the first
finishes are coalesced -- one decode, every waiting (table, row) a
subscriber. `reset()` bumps a generation counter; a job whose captured
generation is stale commits nothing -- no index entry, no fingerprint-memo
entry, no notification (a JPEG already encoded to disk can linger, with
nothing pointing at it -- P0.5b-2ii). Requests are issued on demand as
tiles paint (P0.5b-3i, `AppController.render_column_value`); there is no
eager whole-table pass. Priority ordering and viewport cancellation are
P0.5b-3ii and have no producer in the repo yet. Disk-cache eviction and
the memory LRU ceiling are P0.5b-2ii.

This file is written centrally (not by a student).
"""

from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
from typing import NamedTuple
import json
import os
import re
import shutil
import threading

import numpy as np
from PIL import Image

from artifacts.artifact_codec import ArtifactCodec, ArtifactCodecError
from artifacts.cache_sweep import SweepFile, plan_sweep
from artifacts.worker_pool import WorkerPool
from media.artifact_key import ArtifactKey, SourceFingerprint

# Import the extension sets from dataset so they stay in sync.
# We only need to know which extensions are videos here.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

THUMBNAIL_SIZE = (150, 150)
PREVIEW_SIZE   = (600, 600)

# The resolution (max side, in pixels) that enters the ArtifactKey for
# each purpose. Derived from the size tuples above rather than restated,
# so the key and the resize stay in step. THUMBNAIL_SIZE / PREVIEW_SIZE
# remain plain module constants in this diff -- turning them into
# machine-independent settings is P0.5b-2, together with worker count.
THUMBNAIL_RESOLUTION = max(THUMBNAIL_SIZE)
PREVIEW_RESOLUTION   = max(PREVIEW_SIZE)

# For now the store always records the first frame of a video (or the
# whole image) and ignores any frame or time specifier in the address.
# The key still carries the policy explicitly, so a later 'midpoint'
# policy (which needs real per-frame timings -- P1.2) produces a
# different key and does not collide with these pictures.
REPRESENTATIVE_FRAME_POLICY = "first"

# The in-memory image-cache ceiling. Machine dependent, so a
# [TARGET -> P0.5b-2ii-c] settings site alongside
# DEFAULT_DISK_CACHE_MAX_BYTES below -- neither is a setting yet.
DEFAULT_CACHE_MAX_BYTES = 500 * 1024 * 1024

# The on-disk artifact cache ceiling, measured against the total size of
# the sweep-owned JPEGs that survive orphan removal (see
# cache_sweep.plan_sweep). Separate number from DEFAULT_CACHE_MAX_BYTES,
# which is memory. 1 GiB. Machine dependent, so a
# [TARGET -> P0.5b-2ii-c] settings site; until then it is this module
# default and a keyword-only ArtifactStore constructor parameter, the
# same treatment worker_count gets.
DEFAULT_DISK_CACHE_MAX_BYTES = 1024 * 1024 * 1024

# A cache file on disk is named "<stable_hash>.jpg". ArtifactKey.
# stable_hash() is a sha256 hex digest truncated to 32 characters, so
# the name is exactly 32 of [0-9a-f] then ".jpg". ONLY files matching
# this at the top level of the artifacts directory are sweep-owned:
# subdirectories, artifact_index.json and anything else are invisible to
# the sweep. That is what makes running the sweep inside a user's
# project folder safe.
_ARTIFACT_FILENAME_RE = re.compile(r"\A[0-9a-f]{32}\.jpg\Z")

# Bumped when the on-disk index layout changes. load_index() discards an
# index written under a different version rather than half-reading it.
# 2 -> 3 (P0.5b-2ii-b1): record "path" is now RELATIVE to the artifacts
# directory, not absolute. A version-2 index is discarded whole -- its
# absolute paths cannot be trusted once the project folder has moved, and
# a project reopened after this change regenerates its pictures on demand.
INDEX_FORMAT_VERSION = 3


class SweepResult(NamedTuple):
    """Counts from one `reconcile_and_evict()` call, for the log line and
    the tests."""

    files_deleted: int          # orphans + ceiling evictions actually unlinked
    index_entries_dropped: int  # keys removed from _index (deleted or missing)
    missing_files_reconciled: int   # index paths that had no file on disk
    delete_failures: int        # planned deletes that raised OSError
    bytes_before: int
    bytes_after: int


class ArtifactStore:
    """
    Stores and retrieves derived visual files for gallery display.

    Supports both image and video source files. For video files,
    thumbnails are generated from the first frame.
    """

    def __init__(
        self,
        artifacts_dir: Path,
        *,
        worker_count: int = 2,
        disk_cache_max_bytes: int = DEFAULT_DISK_CACHE_MAX_BYTES,
    ):
        self._dir = artifacts_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._codec = ArtifactCodec(self._dir)

        # index and fingerprint memo are touched by worker threads and by
        # the main thread, so both live under _lock.
        self._index: dict[ArtifactKey, Path] = {}
        self._fingerprints: dict[str, SourceFingerprint] = {}
        # Whether an empty or partial _index can be trusted to mean
        # "nothing else is cached". True at construction and after
        # reset() (an empty index then IS the truth); set by load_index()
        # -- False when artifact_index.json was missing or unreadable, so
        # reconcile_and_evict() must not delete the directory's JPEGs as
        # orphans. Main-thread only, like _cache.
        self._index_authoritative: bool = True
        # Addresses whose memo fingerprint came from a fresh stat this
        # session (written by _run_job's commit), as opposed to being
        # seeded from a persisted index by load_index (which may be
        # stale). A request for an unverified address always queues a
        # worker so the source is re-stat-ed, rather than short-circuiting
        # on a maybe-stale fingerprint.
        self._verified: set[str] = set()
        self._lock = threading.Lock()

        # Bounded worker pool that runs thumbnail/preview generation off
        # the main thread, replacing the old raw thread-per-request. The
        # worker count is a constructor parameter with a low default, not
        # a module constant -- it is machine dependent (CLAUDE.md's
        # no-machine-dependent-constant rule, [TARGET -> P0.5b-2]). Same
        # precedent as AppController's drain_budget. Keyword-only so the
        # positional ArtifactStore(dir) construction in main.py and the
        # tests keeps working unchanged.
        self._pool = WorkerPool(worker_count=worker_count)

        # Request coalescing and cancellation state, all guarded by
        # _lock.
        #   _inflight maps (generation, canonical address) to the list of
        #     (table_name, row_id) subscribers waiting on that job. One
        #     job runs per key; a later request for the same address
        #     joins the list instead of starting a second job, and every
        #     subscriber on the list is notified when the job finishes.
        #   _generation is bumped by reset(). A job captures it in its
        #     key when enqueued; a job whose captured generation is no
        #     longer current is dropped -- no JPEG, no index entry, no
        #     notification.
        self._inflight: dict[tuple[int, str], list[tuple[str, str]]] = {}
        self._generation: int = 0

        # The in-memory LRU is populated only by get_pixmap(), on the main
        # thread, so it needs no lock. A worker (_run_job) writes the JPEG
        # to disk and the index only; the first paint after generation
        # reads the JPEG back through the codec and fills this cache.
        self._cache: OrderedDict[ArtifactKey, Image.Image] = OrderedDict()
        self._cache_bytes: int = 0
        self._cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES

        # The on-disk ceiling the directory sweep (reconcile_and_evict)
        # enforces. Keyword-only constructor parameter, not a hardcoded
        # constant -- machine dependent, [TARGET -> P0.5b-2ii-c].
        self._disk_cache_max_bytes: int = disk_cache_max_bytes

        self.on_thumbnail_ready = None

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    @staticmethod
    def resolution_for(purpose: str) -> int:
        """The requested resolution (max side, in pixels) that enters the
        ArtifactKey for a purpose.

        This is the ONE definition of the purpose -> resolution mapping in
        the repository. The renderer factory is handed a store instance
        and calls this on it; nothing imports THUMBNAIL_RESOLUTION /
        PREVIEW_RESOLUTION across a module boundary to recompute it.
        """
        return THUMBNAIL_RESOLUTION if purpose == "thumbnail" else PREVIEW_RESOLUTION

    def _key(
        self,
        canonical_address: str,
        fingerprint: SourceFingerprint,
        purpose: str,
        resolution: int,
        policy: str = REPRESENTATIVE_FRAME_POLICY,
    ) -> ArtifactKey:
        return ArtifactKey(
            canonical_address=canonical_address,
            fingerprint=fingerprint,
            purpose=purpose,
            resolution=resolution,
            policy=policy,
        )

    def _complete_key(
        self,
        canonical_address: str,
        purpose: str,
        resolution: int,
        policy: str,
    ) -> ArtifactKey | None:
        """Build the full key for a lookup from the identity fields a
        caller can know without touching the filesystem, completing the
        fingerprint from the memo. Returns None when the address has no
        memo entry -- that is a cache miss, deliberately."""
        with self._lock:
            fingerprint = self._fingerprints.get(canonical_address)
        if fingerprint is None:
            return None
        try:
            return self._key(
                canonical_address, fingerprint, purpose, resolution, policy
            )
        except ValueError:
            # A bad purpose / resolution / policy is a caller error, but a
            # lookup returns None for every other "not available" case, so
            # it returns None here too rather than raising out of get().
            return None

    # ------------------------------------------------------------------
    # Fingerprint memo
    # ------------------------------------------------------------------

    @staticmethod
    def _stat_fingerprint(source_path: Path) -> SourceFingerprint | None:
        """Stat the source file and return its fingerprint, or None if it
        cannot be stat-ed.

        Pure -- touches no store state. `_run_job` stats through this and
        then writes the memo (`_fingerprints` + `_verified`) inside its
        one generation-checked commit, so a job from a torn-down project
        that reset() raced past writes nothing to the next project's memo.
        """
        try:
            stat = Path(source_path).stat()
        except OSError:
            return None
        return SourceFingerprint(size=stat.st_size, mtime_ns=stat.st_mtime_ns)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        canonical_address: str,
        purpose: str,
        resolution: int | None = None,
        policy: str = REPRESENTATIVE_FRAME_POLICY,
    ) -> Path | None:
        """
        Returns the file path of a stored artifact, or None.

        Args:
            canonical_address: The canonical media address the picture is
                               of (media_address.canonical_key form).
            purpose:           'thumbnail' or 'preview'.
            resolution:        Requested max side in pixels. Defaults to
                               the standard resolution for the purpose.
            policy:            Representative-frame policy.
        """
        if resolution is None:
            resolution = self.resolution_for(purpose)
        key = self._complete_key(canonical_address, purpose, resolution, policy)
        if key is None:
            return None
        with self._lock:
            return self._index.get(key, None)

    # There is no public `put()`. The only writer is `_run_job`, which
    # encodes JPEGs into local variables and then writes `_index` inside
    # one generation-checked lock hold, so a job reset() raced past
    # indexes nothing. A future writer (P1.7a's segment-thumbnail batch
    # job) adds what it needs against that same generation gate rather
    # than through an unguarded helper.

    def get_pixmap(
        self,
        canonical_address: str,
        purpose: str,
        resolution: int | None = None,
        policy: str = REPRESENTATIVE_FRAME_POLICY,
    ):
        """
        Returns a PIL Image ready for conversion to QPixmap in the UI,
        using the in-memory LRU cache to avoid redundant disk reads.
        Returns None if the artifact does not exist (including when the
        address has no fingerprint memo entry yet).

        On a memory-cache hit this touches no filesystem at all -- no
        stat, no exists, no open. A memory miss that hits the disk index
        reads the JPEG back through ArtifactCodec.
        """
        if resolution is None:
            resolution = self.resolution_for(purpose)
        key = self._complete_key(canonical_address, purpose, resolution, policy)
        if key is None:
            return None

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        with self._lock:
            path = self._index.get(key, None)
        if path is None:
            return None

        try:
            image = self._codec.read_image(path)
        except (OSError, ValueError, ArtifactCodecError):
            # ArtifactCodecError: a persisted index carried a path that
            # resolves outside this cache root (project moved between
            # machines). Treat as a miss, not a crash.
            return None
        self._add_to_cache(key, image)
        return image

    def request_thumbnail(
        self,
        row_id: str,
        address: str,
        source_path: Path,
        table_name: str,
    ) -> None:
        """
        Queues thumbnail and preview generation for a picture on the
        bounded worker pool. Handles both image and video source files.

        Args:
            row_id:      The row whose tile is waiting -- echoed back
                         through on_thumbnail_ready(table_name, row_id).
                         Identity of the PICTURE is `address`, not this.
            address:     Canonical media address the picture is of.
            source_path: Absolute filesystem path to decode from. Identity
                         and fetch-route are different things: `address`
                         says what the picture is, `source_path` says
                         where to read it.
            table_name:  Table the row belongs to. Echoed back so the
                         controller can tag the ready notification.

        Delivery contract (P0.5b-2i -- recorded here because it is
        written down nowhere else):

          * Every accepted request delivers one
            on_thumbnail_ready(table_name, row_id) callback with ITS OWN
            (table_name, row_id) -- never a different subscriber's.
          * When several requests name the same canonical address before
            the first has finished, ONE job runs and decodes once; every
            waiting request is a subscriber to that job and gets its own
            callback. A request that joins an in-flight job is a
            subscriber, not a discarded request. (Two requests for the
            same (table_name, row_id) therefore deliver two callbacks --
            one per accepted request; the callback is idempotent so the
            caller need not care.)
          * Three cases deliver no callback:
              - the source cannot be decoded (pre-queue behaviour);
              - reset() has bumped the store generation since the
                request was accepted -- the row's project is being torn
                down (pre-queue behaviour);
              - the request was still queued (not yet running) when
                set_wanted_addresses() declared its address no longer
                wanted and dropped it (P0.5b-3ii-a). A later
                request_thumbnail() for that address starts a fresh job
                rather than joining the dropped one.

        Short-circuit: if the address already has both a thumbnail and a
        preview in the index under a fingerprint that was stat-ed THIS
        session ("verified"), the subscriber is notified synchronously on
        the caller's thread and no job is queued. This path performs no
        stat() -- the caller is the main thread. An UNVERIFIED
        (load_index-seeded) fingerprint always falls through to a worker,
        which re-stats the source and regenerates if it changed since the
        project was saved.
        """
        address = str(address)

        # Short-circuit check in ONE lock hold: generation, fingerprint,
        # verified and both-present are read together, so a worker
        # committing a picture cannot land between the reads and leave the
        # decision half-based on stale state. _notify_ready is called
        # AFTER the lock is released -- it calls into a UI callback.
        with self._lock:
            generation = self._generation
            fingerprint = self._fingerprints.get(address)
            verified = address in self._verified
            short_circuit = (
                verified
                and fingerprint is not None
                and self._both_present_locked(address, fingerprint)
            )
        if short_circuit:
            self._notify_ready(table_name, row_id)
            return

        job_key = (generation, address)
        with self._lock:
            if generation != self._generation:
                # reset() ran while we were checking the short-circuit;
                # the row belongs to a project being torn down. Drop it.
                return
            waiting = self._inflight.get(job_key)
            if waiting is not None:
                # A job for this address is already queued or running.
                # Join it as a subscriber rather than starting a second
                # job -- this is the coalescing.
                waiting.append((table_name, row_id))
                return
            self._inflight[job_key] = [(table_name, row_id)]

        # job_key -- (generation, canonical address) -- is also the pool's
        # opaque key, so set_wanted_addresses() can drop this job by
        # address without the pool knowing what an address is.
        self._pool.submit(
            lambda: self._run_job(job_key, Path(source_path)), key=job_key
        )

    def is_cached(
        self,
        canonical_address: str,
        policy: str = REPRESENTATIVE_FRAME_POLICY,
    ) -> bool:
        """True if BOTH derived pictures for this address -- thumbnail and
        preview, at the standard resolutions -- are already in the index.
        An index-only lookup: opens no source file and stats nothing.

        This is the ONE definition of "is this address already cached".
        `render_column_value()` calls it once per painted media tile to
        decide whether to queue a demand request; `request_thumbnail()`
        shares its both-present helper for its own synchronous
        short-circuit.

        Both purposes are checked, not just one. A tile larger than the
        renderer's preview threshold paints from the 'preview' artifact, a
        request regenerates both anyway, and this matches
        `request_thumbnail()`'s both-present short-circuit -- so "either
        one missing" is the right trigger for a demand request.

        A `load_index()`-seeded entry counts as cached here and is served
        as-is. Whether painting re-stats it, and how `request_thumbnail()`
        treats an unverified fingerprint differently, is `load_index()`'s
        docstring to state -- not this one.

        Single lock hold. The two `get()` calls this replaced cost four
        `_lock` acquisitions per tile.
        """
        with self._lock:
            fingerprint = self._fingerprints.get(canonical_address)
            if fingerprint is None:
                return False
            try:
                return self._both_present_locked(
                    canonical_address, fingerprint, policy
                )
            except ValueError:
                # A bad policy is a caller error, but a lookup returns
                # False for every "not available" case, so it does here
                # too rather than raising out of a paint.
                return False

    def _both_present_locked(
        self,
        address: str,
        fingerprint: SourceFingerprint,
        policy: str = REPRESENTATIVE_FRAME_POLICY,
    ) -> bool:
        """True if this address's thumbnail AND preview keys (standard
        resolutions) are both in `_index`.

        CALLER MUST HOLD `_lock`. Shared by `is_cached()` and
        `request_thumbnail()` so neither nests a second lock acquisition
        the way the old `_both_present()` did -- it re-took `_lock` after
        the caller had already released it, making the short-circuit
        non-atomic.
        """
        thumb_key = self._key(
            address, fingerprint, "thumbnail",
            self.resolution_for("thumbnail"), policy,
        )
        preview_key = self._key(
            address, fingerprint, "preview",
            self.resolution_for("preview"), policy,
        )
        return thumb_key in self._index and preview_key in self._index

    def set_wanted_addresses(self, addresses) -> None:
        """Declare the set of canonical media addresses the display
        currently wants pictures for (P0.5b-3ii-a).

        Among jobs that are still QUEUED (not yet running) in the current
        generation:

          * jobs whose address is in `addresses` survive, keeping their
            existing submit order among themselves -- submit order is
            paint order, and the pool does no priority reordering;
          * every other job is dropped -- its worker job is removed and
            its `_inflight` entry deleted, so a later request_thumbnail()
            for that address starts a fresh job instead of joining one
            that will never run. Its subscribers get no callback (see
            request_thumbnail's delivery contract, third case).

        A job that has already started on a worker is never touched: it
        runs to completion and commits normally. This method takes
        addresses only -- never rows, tables or viewport positions. Its
        consumer is AppController, which calls it whenever a gallery
        reports or clears a displayed range (P0.5b-3ii-b).
        """
        wanted = {str(address) for address in addresses}

        # One lock hold covers: reading _inflight, telling the pool which
        # jobs to keep, and deleting the _inflight entries for the jobs
        # the pool confirms it removed. Holding _lock across the pool call
        # blocks request_thumbnail's own _inflight critical section, so it
        # cannot slip a new subscriber onto a job between drop_pending
        # removing it and us deleting its _inflight entry. Lock order is
        # always store-lock then pool-lock (reset() does the same), so no
        # deadlock with a worker: a worker holds the pool lock only to pop
        # a job, never while running one.
        with self._lock:
            generation = self._generation

            # job_key is (generation, address); it is also the pool key.
            keep_keys = [
                job_key
                for job_key in self._inflight
                if job_key[0] == generation and job_key[1] in wanted
            ]

            # Drop the non-wanted jobs. drop_pending returns only the keys
            # it actually removed from the queue -- a job a worker popped
            # at the same moment is not in that list, so we leave its
            # _inflight entry alone and it commits normally. The surviving
            # jobs keep their submit order; the pool does no priority
            # reordering, and submit order is paint order.
            dropped = self._pool.drop_pending(keep_keys)
            for job_key in dropped:
                self._inflight.pop(job_key, None)

    def set_cache_max_bytes(self, max_bytes: int) -> None:
        """
        Sets the maximum memory the cache may use, in bytes.

        Args:
            max_bytes: Maximum cache size in bytes.
        """
        self._cache_max_bytes = max_bytes
        self._evict_if_needed()

    def set_artifacts_dir(self, new_dir: Path) -> None:
        """Re-point the cache directory at `new_dir` and migrate the index.

        MAIN THREAD ONLY.

        Step 1 (under `_lock`): bump the generation, clear the subscriber
        map, drop the pending queue -- exactly what reset() does for
        cancellation, and for the same reason. load_project() calls this
        right after reset() (the second bump is harmless); save_project()
        calls it with NO preceding reset(), and the bump is what stops a
        worker that is mid-decode from committing an index entry with an
        old-root path into the very file save_index() is about to write.
        Once the generation is bumped, no worker will touch `_index`
        again (its commit re-checks the generation under the lock and
        early-returns) and no new job can be enqueued (request_thumbnail
        is main-thread only and we are on the main thread), so the copy
        loop below can run WITHOUT the lock. A running job may still
        leave a stray JPEG in either root -- nothing points at it
        (P0.5b-2ii-b reclaims orphans).

        Step 2 (no lock): copy every index entry whose file lies OUTSIDE
        the new root into the new root under the same filename, building
        the migrated index. Copy, not move: the old root may be the
        shared scratch folder and another project's saved index may still
        name that file. Drop -- do not migrate -- an entry whose file no
        longer exists. This step does the file I/O, so it is kept off the
        lock and BEFORE the swap: if a copy raises (disk full, streaming
        path offline) the store is left untouched -- still bound to the
        old root with the old index -- rather than half-migrated.

        Step 3 (under `_lock`): install the migrated index, re-point
        `_dir`, and rebuild the ArtifactCodec (it resolves its root once
        at construction, so a fresh instance is required for the boundary
        to move).

        On the load path the index is empty when this runs (reset()
        cleared it, load_index() has not run yet), so step 2 is a no-op
        and this call just re-roots the store and the codec. Migration
        matters on the save path, where the index still holds this
        session's scratch-folder entries.
        """
        # Create the new directory up front. mkdir touches nothing the
        # workers read, so it needs no lock.
        new_dir = Path(new_dir)
        new_dir.mkdir(parents=True, exist_ok=True)

        # Fully-resolved root, so the "is this file already inside?"
        # check below compares like with like.
        resolved_root = new_dir.resolve()

        # Step 1: cancel in-flight and queued work, then snapshot the
        # index. After the generation bump nothing but this method writes
        # `_index`, so the snapshot cannot go stale under us.
        with self._lock:
            self._generation += 1
            self._inflight.clear()
            self._pool.clear_pending()
            index_snapshot = dict(self._index)

        # Step 2: build the migrated index off the lock.
        migrated: dict[ArtifactKey, Path] = {}
        for key, path in index_snapshot.items():
            path = Path(path)

            # Already inside the new root: keep the entry, unless its file
            # has vanished, in which case drop it.
            try:
                path.resolve().relative_to(resolved_root)
                if path.exists():
                    migrated[key] = path
                continue
            except ValueError:
                # Outside the new root -- falls through to migration.
                pass

            # The JPEG is gone: drop the entry rather than leave it
            # pointing at a nonexistent path.
            if not path.exists():
                continue

            # Copy the JPEG into the new root under the same name and
            # repoint the entry. copy2 keeps mtime so a later
            # directory-driven eviction sees the real age.
            dest = new_dir / path.name
            if not dest.exists():
                shutil.copy2(path, dest)
            migrated[key] = dest

        # Step 3: install the results and move the boundary.
        with self._lock:
            self._index = migrated
            self._dir = new_dir
            self._codec = ArtifactCodec(new_dir)

    @staticmethod
    def _confirmed_missing(path: Path) -> bool:
        """True only when `path` is *definitely* not on disk (a
        FileNotFoundError from `os.stat`). A permission error, a locked
        file, or an offline network / Google Drive Streaming placeholder
        returns False -- the sweep never drops an index entry it cannot
        positively prove is gone, because that would negate load_index()
        and force a needless regenerate on the next paint.

        Called only for the handful of index paths the directory walk
        did not see, so the common (clean) sweep does no extra stat at
        all."""
        try:
            os.stat(path)
            return False
        except FileNotFoundError:
            return True
        except OSError:
            return False

    def reconcile_and_evict(self) -> SweepResult:
        """Walk the artifacts directory, delete orphaned and over-ceiling
        JPEGs, and drop index entries whose file is gone.

        MAIN THREAD ONLY. Structured like `set_artifacts_dir` and for the
        same reasons -- it bumps the generation so no worker can commit
        into the directory being swept, then does its file I/O off the
        lock, then reinstalls under the lock.

        `docs/media_architecture.md` section 4.7 is the authority. Two
        problems, one directory walk:

          * The on-disk cache is append-only -- a JPEG whose index entry
            is gone (an old-format index discarded on load, a
            RENDERER_CACHE_VERSION bump, a changed fingerprint, reset()
            on never-saved artifacts, a generation-cancelled job that had
            already encoded its file) is unreachable forever, because the
            on-disk name is a one-way hash. Only a directory walk can
            find it; an index-only pass never can.
          * load_index() seeds an index entry without checking the JPEG
            is on disk. An indexed-but-absent entry reopens with
            is_cached() reporting True, so demand-driven display queues
            no request and the tile is a permanent grey placeholder.
            Reconciling the index against the real directory fixes this.

        Step 1 (under _lock): bump the generation and clear the pending
        queue, so no worker can commit an index entry into the directory
        being swept. Both call sites (save_project, load_project) have
        already bumped -- but a stated precondition is not a guarantee,
        and a worker that commits an entry naming a file this method is
        about to delete would reintroduce the very inconsistency it
        removes. The extra bump is harmless: the generation is a
        monotonic cancellation counter, nothing keys off its value.
        Snapshot the index under the same lock hold, before the walk.

        Step 2 (no lock): walk the directory with os.scandir and use each
        DirEntry's cached stat for size and mtime, so the whole ceiling
        costs one pass, not one stat per file. Only <hash>.jpg files at
        the top level are sweep-owned. Every `_index` value is written as
        `self._dir / "<hash>.jpg"` and migrated to `new_dir / name` by
        set_artifacts_dir, so it is always a direct child of the current
        directory: the walk matches index entries to files by BASENAME,
        with no `resolve()` syscall per path. Hand the found files and
        the index's names to the pure `plan_sweep`, then delete the files
        it plans. A per-file delete failure (file locked, path offline)
        is logged and skipped, never raised -- a sweep that cannot delete
        must still reconcile. If the directory itself is unreadable
        (offline mount, transient Drive error) the whole sweep is a
        no-op rather than treat every indexed file as an orphan.

        Step 3 (under _lock): remove from `_index` every key whose file
        was deleted, plus every key whose file the walk did not see AND
        `os.stat` confirms is gone (a permission error or an offline
        placeholder is NOT "gone" -- dropping it would negate
        load_index()). Drop the matching entry from `_fingerprints` and
        `_verified` when no surviving key still uses that address, and
        drop each removed key from the in-memory image cache so
        get_pixmap() cannot keep serving a picture whose JPEG was just
        deleted. Only the removed keys are touched -- the rest of the
        memory cache is left intact.

        Untrustworthy index. When `self._index_authoritative` is False --
        set by load_index() when `artifact_index.json` was missing or
        unreadable -- an empty or partial `_index` cannot be taken to
        mean "nothing else is cached". The sweep then plans NO orphan
        deletions (`plan_sweep(delete_orphans=False)`): reclaiming a
        genuine orphan is not worth the risk of deleting a real cached
        JPEG whose index entry just failed to load, and one save later
        `save_project` would rewrite the index and the next sweep cleans
        up for real. The missing-file reconciliation half still runs --
        dropping an index entry whose file is gone is safe in any index
        state and fixes the grey tile. Ceiling eviction still applies to
        indexed survivors, of which there are none in this state, so it
        is inert.
        """
        # ---- Step 1: freeze the directory, snapshot the index ----
        with self._lock:
            self._generation += 1
            self._inflight.clear()
            self._pool.clear_pending()
            sweep_dir = Path(self._dir)
            index_items = list(self._index.items())

        # Match index entries to files by BASENAME within this one
        # directory -- os.path.abspath is lexical (no syscall), normcase
        # folds Windows case. keys_by_name maps a normalised filename to
        # the keys naming it; the list keeps step 3 total even though two
        # distinct keys never share a stable_hash. An entry whose parent
        # is NOT this directory (should not happen after set_artifacts_dir
        # migrates every entry) is left entirely alone -- never deleted,
        # never dropped.
        sweep_dir_key = os.path.normcase(os.path.abspath(str(sweep_dir)))
        keys_by_name: dict[str, list[ArtifactKey]] = {}
        for key, path in index_items:
            path = Path(path)
            parent_key = os.path.normcase(os.path.abspath(str(path.parent)))
            if parent_key != sweep_dir_key:
                continue
            keys_by_name.setdefault(
                os.path.normcase(path.name), []
            ).append(key)

        # ---- Step 2: walk, plan, delete (no lock) ----
        found: list[SweepFile] = []
        try:
            with os.scandir(sweep_dir) as scanner:
                for dir_entry in scanner:
                    # Sweep-owned == a top-level <hash>.jpg file. A
                    # subdirectory, artifact_index.json, or a stray file
                    # is invisible.
                    if _ARTIFACT_FILENAME_RE.match(dir_entry.name) is None:
                        continue
                    try:
                        if not dir_entry.is_file():
                            continue
                        entry_stat = dir_entry.stat()
                    except OSError:
                        # Cannot classify this one entry. Skip it -- the
                        # confirmed-missing recheck below still protects
                        # any index entry that points here.
                        continue
                    name_key = os.path.normcase(dir_entry.name)
                    found.append(SweepFile(
                        path=name_key,
                        size_bytes=entry_stat.st_size,
                        mtime_ns=entry_stat.st_mtime_ns,
                    ))
        except OSError as error:
            # The directory itself is unreadable right now. Do nothing
            # rather than treat every indexed file as an orphan or as
            # missing -- the sweep runs again on the next save or load.
            print(f"[ArtifactStore] cache sweep skipped, directory "
                  f"unreadable: {error}")
            return SweepResult(0, 0, 0, 0, 0, 0)

        # Read the trust flag once. When the last load could not read the
        # index, an empty _index does not mean "nothing else is cached",
        # so plan no orphan deletions -- see the docstring.
        delete_orphans = self._index_authoritative

        plan = plan_sweep(
            found,
            indexed_paths=set(keys_by_name),
            max_bytes=self._disk_cache_max_bytes,
            delete_orphans=delete_orphans,
        )

        deleted_names: list[str] = []
        delete_failures = 0
        for name in plan.files_to_delete:
            try:
                (sweep_dir / name).unlink()
                deleted_names.append(name)
            except OSError as error:
                # A file we cannot delete (locked, offline) must not stop
                # the reconciliation half below.
                delete_failures += 1
                print(f"[ArtifactStore] cache sweep could not delete "
                      f"{name}: {error}")

        # An index name the walk did not see is only "missing" if a
        # direct stat confirms it. plan.paths_missing is the candidate
        # set; filter it down to the ones really gone.
        confirmed_missing = [
            name for name in plan.paths_missing
            if self._confirmed_missing(sweep_dir / name)
        ]

        # ---- Step 3: reconcile the index (under _lock) ----
        drop_names = set(deleted_names) | set(confirmed_missing)
        entries_dropped = 0
        dropped_keys: list[ArtifactKey] = []
        with self._lock:
            addresses_touched: set[str] = set()
            for name in drop_names:
                for key in keys_by_name.get(name, []):
                    if self._index.pop(key, None) is not None:
                        entries_dropped += 1
                        dropped_keys.append(key)
                        addresses_touched.add(key.canonical_address)

            # Drop the fingerprint memo + verified flag for an address
            # only when no surviving index key still uses it.
            surviving_addresses = {
                key.canonical_address for key in self._index
            }
            for address in addresses_touched:
                if address not in surviving_addresses:
                    self._fingerprints.pop(address, None)
                    self._verified.discard(address)

        # Evict any dropped key that is still in the in-memory image
        # cache, so get_pixmap() cannot keep serving a picture whose
        # backing JPEG the sweep just deleted. _cache is main-thread only
        # (this method is too), so no lock is needed for it.
        for key in dropped_keys:
            image = self._cache.pop(key, None)
            if image is not None:
                self._cache_bytes -= image.width * image.height * 3

        result = SweepResult(
            files_deleted=len(deleted_names),
            index_entries_dropped=entries_dropped,
            missing_files_reconciled=len(confirmed_missing),
            delete_failures=delete_failures,
            bytes_before=plan.bytes_before,
            bytes_after=plan.bytes_after,
        )
        # Only log when the sweep actually changed something -- it runs on
        # every save and load, and a clean no-op should be silent.
        if entries_dropped or result.files_deleted or delete_failures:
            print(f"[ArtifactStore] cache sweep: {result.files_deleted} "
                  f"deleted, {entries_dropped} index entries dropped, "
                  f"{plan.bytes_before} -> {plan.bytes_after} bytes")
        return result

    def _relative_for_index(self, path: Path) -> Path | None:
        """`path` expressed relative to the artifacts directory, or None
        if it lies outside it.

        Tries a plain `relative_to` first -- no filesystem I/O, and the
        common case, since `_run_job` writes `self._dir / "<hash>.jpg"`
        and `set_artifacts_dir` migrates every entry to `self._dir /
        name`. Falls back to comparing both sides `.resolve()`d for a
        path that is inside the directory but not lexically (a symlinked
        or '..'-laden `_dir`). Returns None when the entry names a file
        outside the cache root: `save_index` SKIPS such a record rather
        than writing an absolute path, and nothing is lost because
        `ArtifactCodec` would refuse to read that path back anyway.

        Reads `self._dir` only. Both callers -- `save_index` (over a
        snapshot) and `load_index` -- run it OUTSIDE `_lock`, so the
        `.resolve()` in the fallback branch is never filesystem I/O under
        the lock.

        The lexical branch does no I/O and covers the common case: every
        `_index` entry is `self._dir / "<hash>.jpg"`. A lexical result is
        rejected if it starts with `..` -- `a/b/../../x` is lexically
        "under" `a` but escapes it -- and only then does the resolved
        fallback run. `resolve()` also raises `OSError` on a symlink
        loop; that is "not safely inside" too, so it returns None.

        Same resolve()+relative_to() containment idea as
        `ArtifactCodec._require_inside`; the codec's copy cannot be
        shared (separate object, private root), so if the semantics ever
        change (Windows case folding, UNC) the two move together.
        """
        path = Path(path)
        try:
            rel = path.relative_to(self._dir)
            if ".." not in rel.parts:
                return rel
        except ValueError:
            pass
        try:
            return path.resolve().relative_to(Path(self._dir).resolve())
        except (ValueError, OSError):
            return None

    def save_index(self, project_path: Path) -> None:
        """Saves the artifact index to disk as versioned JSON.

        Each record's "path" is stored RELATIVE to the artifacts
        directory (`self._dir`), not absolute, so a project folder can be
        moved between machines -- e.g. the Google Drive Streaming path
        this project lives on resolves to a different absolute prefix on
        each machine -- and still find its cached JPEGs. `load_index()`
        rebuilds each path as `self._dir / <relative>`.

        A record whose file lies outside the cache root is skipped (see
        `_relative_for_index`); `ArtifactCodec` would refuse to read such
        a path back anyway, so skipping loses nothing.

        The in-memory `_index` keeps ABSOLUTE paths -- only this JSON
        boundary is relative.

        The `_index` is snapshotted under `_lock` and the relative-path
        conversion runs outside it: `_relative_for_index`'s fallback
        branch can do a `.resolve()` syscall, and that must not block a
        worker commit or a paint-path `is_cached()` call.
        """
        with self._lock:
            entries = list(self._index.items())

        records = []
        for key, path in entries:
            relative = self._relative_for_index(path)
            if relative is None:
                # Outside the cache root -- unreadable through the codec
                # anyway. Skip rather than persist an absolute path that
                # load_index() would have to reject.
                continue
            records.append({
                "address":          key.canonical_address,
                "size":             key.fingerprint.size,
                "mtime_ns":         key.fingerprint.mtime_ns,
                "purpose":          key.purpose,
                "resolution":       key.resolution,
                "policy":           key.policy,
                "renderer_version": key.renderer_version,
                "path":             str(relative),
            })
        payload = {"format_version": INDEX_FORMAT_VERSION, "artifacts": records}
        index_path = project_path / "artifact_index.json"
        index_path.write_text(json.dumps(payload, indent=2))

    def load_index(self, project_path: Path) -> bool:
        """Loads the artifact index from disk.

        Sets `self._index_authoritative` AND returns the same bool -- the
        flag is what `reconcile_and_evict()` reads (it can run on the
        save path, one click after a failed load, with no fresh return
        value to consult); the return value is a convenience for
        `load_project()`, which uses it to skip the walk entirely.

          * True  -- it loaded, OR a format-version mismatch discarded it
            whole. In both cases an empty `_index` is the truth (the
            mismatched index's JPEGs are keyed by hashes that cannot be
            reconstructed without it, so they are genuinely unreachable),
            and it is safe to run `reconcile_and_evict()` and reclaim the
            directory's JPEGs.
          * False -- `artifact_index.json` is missing, or exists but
            could not be read or parsed. A real saved project always
            writes an index, so a missing or unreadable one means a
            partial sync, a half-written file, or a transient filesystem
            error on the Google Drive Streaming path -- the index
            contents are UNKNOWN. `_index` is empty only because the
            preceding reset() cleared it, NOT because nothing is cached,
            so `reconcile_and_evict()` must not delete the directory's
            JPEGs as orphans (it still reconciles missing entries).

        An index whose format version does not match is discarded whole,
        not half-read. Callers must reset() first -- load_index() replaces
        the index but is not a substitute for clearing the memory image
        cache.

        Seeds the fingerprint memo from the persisted (size, mtime), so a
        project that was fully thumbnailed reopens with its cache usable
        with no paint-path decode. Those fingerprints are the freshness as
        of the last save, not a fresh stat, so none is marked verified:
        a request_thumbnail() for such an address always spawns a worker
        (rather than short-circuiting), and if the source changed since the
        save the worker's fresh fingerprint no longer matches and it
        regenerates. A get_pixmap() lookup serves the persisted picture in
        the meantime. For display (not analysis) a briefly-stale preview is
        an accepted trade against decoding every visible source image on
        the main thread after every reload.

        Demand-driven display (P0.5b-3i) issues a request only for an
        address the seeded index does NOT cover -- a painted row whose
        picture is missing from the index. A seeded (unverified) entry is
        served as-is on the paint path and is not re-requested on view;
        the re-stat happens only if something else calls request_thumbnail()
        for it (or after a reset()).

        Paths in the JSON are RELATIVE to the artifacts directory
        (P0.5b-2ii-b1). This method rebuilds each as `self._dir /
        record["path"]`, so its precondition is that `self._dir` is
        already the project's artifacts directory when it runs.
        `AppController.load_project()` satisfies this: it calls
        `set_artifacts_dir(project_path / "artifacts")` immediately
        before `load_index(project_path)`. The precondition is documented
        here, not asserted in code. A record whose stored path is
        absolute is skipped -- a version-3 index should never contain
        one, and `self._dir / <absolute>` would silently discard
        `self._dir` and reintroduce exactly the moved-folder bug this
        format removes. A relative path that escapes the artifacts
        directory (a `..`-laden path in a corrupt or hand-edited index)
        is skipped for the same reason save_index skips an outside-root
        entry: the codec cannot read it.
        """
        index_path = project_path / "artifact_index.json"
        if not index_path.exists():
            # A saved project always has an index. Missing one means a
            # partial sync or a lost file, not "nothing is cached" -- the
            # sweep must not delete the directory's JPEGs as orphans.
            self._index_authoritative = False
            return False
        try:
            payload = json.loads(index_path.read_text())
        except (ValueError, OSError):
            # The file is there but unreadable or unparseable. We do NOT
            # know what is cached, so the sweep must not delete orphans.
            self._index_authoritative = False
            return False
        if (
            not isinstance(payload, dict)
            or payload.get("format_version") != INDEX_FORMAT_VERSION
        ):
            # Written under the old (row_id, artifact_type) scheme or an
            # unknown one. Discard rather than mixing schemes -- those
            # JPEGs are genuinely unreachable now, so an empty _index IS
            # the truth and it is correct for the sweep to reclaim them.
            self._index_authoritative = True
            return True

        new_index: dict[ArtifactKey, Path] = {}
        new_fingerprints: dict[str, SourceFingerprint] = {}
        for record in payload.get("artifacts", []):
            try:
                fingerprint = SourceFingerprint(
                    size=record["size"], mtime_ns=record["mtime_ns"]
                )
                key = ArtifactKey(
                    canonical_address=record["address"],
                    fingerprint=fingerprint,
                    purpose=record["purpose"],
                    resolution=record["resolution"],
                    policy=record.get("policy", REPRESENTATIVE_FRAME_POLICY),
                    renderer_version=record["renderer_version"],
                )
                # Inside the try: a record with no "path", or a non-string
                # path (null, a number), is one bad record to skip -- not
                # a reason to abort the whole load.
                record_path = Path(record["path"])
            except (KeyError, TypeError, ValueError):
                continue

            if not record_path.parts:
                # "" or "." -- a degenerate path that would rebuild to
                # the artifacts directory itself. Skip.
                continue
            if record_path.is_absolute():
                # A version-3 index carries relative paths. pathlib's `/`
                # with an absolute right-hand operand DISCARDS the left
                # side, so `self._dir / record_path` would silently be
                # `record_path` itself -- the moved-folder bug this
                # format removes. Skip the record.
                continue
            rebuilt = self._dir / record_path
            if self._relative_for_index(rebuilt) is None:
                # A '..'-laden relative path (corrupt or hand-edited
                # index) that escapes the artifacts directory. Same
                # reasoning as save_index skipping an outside-root entry:
                # ArtifactCodec would refuse to read it, and keeping it
                # would only make is_cached() report a permanent grey
                # tile as cached. Cheap: _relative_for_index does no I/O
                # unless the path actually contains '..'.
                continue
            new_index[key] = rebuilt
            # Last writer wins; every record for one address carries the
            # same fingerprint (they were saved together).
            new_fingerprints[record["address"]] = fingerprint

        with self._lock:
            self._index = new_index
            self._fingerprints = new_fingerprints
            # Every seeded fingerprint is the freshness as of the last
            # save, not a fresh stat -- so none is verified.
            self._verified = set()
        # The index loaded: it is now the authority on what is cached.
        self._index_authoritative = True
        return True

    def reset(self) -> None:
        """Clears the index, the memory cache, the fingerprint memo and
        the in-flight subscriber map, bumps the worker generation, and
        marks the (now empty) index authoritative.

        load_project() must call this BEFORE load_index(): otherwise a new
        project's index lands on top of the previous project's live image
        cache and fingerprint memo, and an old picture can appear under a
        new row (docs/media_architecture.md section 4.5).

        Cancellation (P0.5b-2i). Bumping the generation drops every job
        still queued or running for the previous project. Each job does
        all its I/O into local variables and commits the memo and index
        entries in one lock hold that re-checks the generation first, so
        a job this call races past commits nothing -- no index entry, no
        fingerprint-memo entry, no notification. A JPEG it had already
        encoded to disk can linger (the append-only disk cache is
        P0.5b-2ii); no index entry points at it.
        """
        with self._lock:
            self._generation += 1
            self._index.clear()
            self._fingerprints.clear()
            self._verified.clear()
            self._inflight.clear()
            # An empty index right after reset() IS the truth -- clear
            # any not-authoritative state a previous failed load left, so
            # the flag does not latch across projects.
            self._index_authoritative = True
            # Drop jobs still sitting in the pool queue. Inside the lock so
            # no request_thumbnail can slip a current-generation job onto
            # the queue between the bump and this clear and have it wiped
            # (that would leak its _inflight entry). The generation bump
            # already makes queued jobs no-ops; this just saves the
            # workers pulling each stale closure off the queue first.
            self._pool.clear_pending()
        self._cache.clear()
        self._cache_bytes = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_job(self, job_key: tuple[int, str], source_path: Path) -> None:
        """
        Runs on a worker thread. Generates the thumbnail and preview for
        one canonical address and then notifies every subscriber that
        joined this job.

        Generation gate. `job_key` is `(generation, address)`; the
        generation was the store's when the job was enqueued. reset()
        bumps that counter, so a job whose generation is no longer
        current is cancelled: it adds no index entry, no fingerprint-memo
        entry and sends no notification.

        The gate matters because the job mutates shared store state
        (`_index`, `_fingerprints`, `_verified`, `_inflight`), and reset()
        clears all of it for the next project. So the job does its stat,
        decode and resize without touching that state, encodes the two
        JPEGs to disk, and only then -- in ONE lock hold that first
        re-checks the generation -- writes the memo and index entries and
        takes the subscriber list. A job that reset() raced past is
        guaranteed to add no index entry, no memo entry and to notify
        nobody. It is NOT guaranteed to leave the filesystem untouched:
        a JPEG it encoded before the losing commit stays on disk with no
        index entry pointing at it -- reclaiming that is P0.5b-2ii.

        For image files the source is loaded via PIL; for video files the
        first frame is extracted via OpenCV. Both are source-media
        decodes that P1.2 will route through the resolver -- this diff
        leaves them where they were, behind `_decode_source`.
        """
        generation, address = job_key

        # Cancelled before we even started (job picked up after reset()).
        if self._is_stale(generation):
            self._discard_subscribers(job_key)
            return

        try:
            # One stat: _stat_fingerprint returns None for a missing or
            # unreadable source, which is the same "nothing to do" case as
            # a file that vanished.
            fingerprint = self._stat_fingerprint(source_path)
            if fingerprint is None:
                self._discard_subscribers(job_key)
                return

            thumb_key = self._key(
                address, fingerprint, "thumbnail",
                self.resolution_for("thumbnail"),
            )
            preview_key = self._key(
                address, fingerprint, "preview",
                self.resolution_for("preview"),
            )

            # A concurrent job for the same address (same generation) may
            # already have produced both pictures; if so, generate
            # nothing and just notify from the commit below.
            with self._lock:
                already_have_both = (
                    thumb_key in self._index and preview_key in self._index
                )

            pending_index: dict[ArtifactKey, Path] = {}
            if not already_have_both:
                image = self._decode_source(source_path)
                if image is None:
                    self._discard_subscribers(job_key)
                    return

                # Courtesy check: skip the encode work if reset() has
                # already happened. The commit below is the check that
                # actually guarantees correctness.
                if self._is_stale(generation):
                    self._discard_subscribers(job_key)
                    return

                thumb = image.copy()
                thumb.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
                thumb_dest = self._dir / f"{thumb_key.stable_hash()}.jpg"
                self._codec.write_jpeg(thumb_dest, np.array(thumb, dtype=np.uint8))
                pending_index[thumb_key] = thumb_dest

                preview = image.copy()
                preview.thumbnail(PREVIEW_SIZE, Image.LANCZOS)
                preview_dest = self._dir / f"{preview_key.stable_hash()}.jpg"
                self._codec.write_jpeg(
                    preview_dest, np.array(preview, dtype=np.uint8)
                )
                pending_index[preview_key] = preview_dest

        except Exception as e:
            print(f"[ArtifactStore] Failed to generate thumbnails "
                  f"for {address}: {e}")
            self._discard_subscribers(job_key)
            return

        # Atomic commit: re-check the generation and, only if still
        # current, write the memo and index entries and take the
        # subscriber list -- all under one lock, so reset() cannot land
        # between the check and the writes.
        with self._lock:
            if generation != self._generation:
                self._inflight.pop(job_key, None)
                return
            self._fingerprints[address] = fingerprint
            self._verified.add(address)
            self._index.update(pending_index)
            subscribers = self._inflight.pop(job_key, [])

        print(f"[ArtifactStore] Thumbnail ready for {address}")
        for table_name, row_id in subscribers:
            # reset() may land mid-loop: stop notifying rows of a project
            # that is being torn down. (The residual gap between this
            # check and the callback is nanoseconds, and a spurious
            # repaint is harmless -- the tile looks up its own current
            # key -- but a dropped job should still send nothing.)
            if self._is_stale(generation):
                return
            self._notify_ready(table_name, row_id)

    def _decode_source(self, source_path: Path) -> Image.Image | None:
        """Decode the source media file to one RGB PIL image -- the first
        frame for a video, the whole image otherwise. Returns None if it
        cannot be read.

        The single source-media decode in this file. P1.2 routes it
        through the resolver; until then it is a direct decode, exactly
        as before the worker pool. It is its own method so a test can
        hold a worker inside it while it exercises coalescing and
        cancellation.
        """
        suffix = source_path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            return self._first_frame_as_pil(source_path)
        return Image.open(source_path).convert("RGB")

    def _is_stale(self, generation: int) -> bool:
        """True if reset() has bumped the generation since `generation`
        was captured -- the job must stop without writing or notifying."""
        with self._lock:
            return generation != self._generation

    def _discard_subscribers(self, job_key: tuple[int, str]) -> None:
        """Drop a job's subscriber list without notifying -- used when
        the job is cancelled, fails, or finds the source missing."""
        with self._lock:
            self._inflight.pop(job_key, None)

    def _notify_ready(self, table_name: str, row_id: str) -> None:
        if self.on_thumbnail_ready is not None:
            self.on_thumbnail_ready(table_name, row_id)

    def _first_frame_as_pil(self, video_path: Path) -> Image.Image | None:
        """
        Extracts the first frame of a video file and returns it as a
        PIL Image in RGB mode.

        Uses OpenCV (cv2). Returns None if OpenCV is not installed or
        if the video cannot be read.

        Args:
            video_path: Path to the video file.

        Returns:
            A PIL Image, or None.
        """
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            ok, frame = cap.read()
            cap.release()

            if not ok or frame is None:
                print(f"[ArtifactStore] Could not read first frame "
                      f"from {video_path}")
                return None

            # OpenCV returns BGR — convert to RGB.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)

        except ImportError:
            print("[ArtifactStore] OpenCV (cv2) not installed — "
                  "cannot generate video thumbnail.")
            return None
        except Exception as e:
            print(f"[ArtifactStore] Video frame error for {video_path}: {e}")
            return None

    def _add_to_cache(
        self,
        key: ArtifactKey,
        image: Image.Image,
    ) -> None:
        """Adds an image to the LRU cache and evicts if over limit."""
        estimated_bytes = image.width * image.height * 3
        self._cache[key] = image
        self._cache.move_to_end(key)
        self._cache_bytes += estimated_bytes
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Removes least recently used entries until within memory limit."""
        while self._cache_bytes > self._cache_max_bytes and self._cache:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= evicted.width * evicted.height * 3
