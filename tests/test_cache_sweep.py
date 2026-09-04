"""
tests/test_cache_sweep.py

P0.5b-2ii-b2 -- artifact cache eviction and index/directory reconciliation.

Two halves of one mechanism, both answered by one directory walk:

  (a) The on-disk artifact cache is append-only. A JPEG whose index entry
      is gone is unreachable forever, because the on-disk name is a
      one-way hash. Only a directory walk can find it.
  (b) `load_index()` seeds an index entry without checking the JPEG is on
      disk, so an indexed-but-absent entry reopens with `is_cached()`
      reporting True and a permanent grey tile. The same walk reconciles
      the index against the real directory.

`artifacts/cache_sweep.py::plan_sweep` is the pure planner;
`ArtifactStore.reconcile_and_evict()` does the I/O; `AppController`
calls it on every save and load.

Written from the work-item spec, not from the implementation. New file on
purpose: tests/test_dataset.py runs everything in it twice. Concurrency
uses threading.Event, copying tests/test_artifact_cache_location.py.

Run with:
    python -m pytest tests/test_cache_sweep.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from PIL import Image

from artifacts.cache_sweep import SweepFile, SweepPlan, plan_sweep
from artifacts.artifact_store import ArtifactStore, _ARTIFACT_FILENAME_RE
from media.artifact_key import ArtifactKey, SourceFingerprint
from media.media_address import resolve_source

from models.dataset import Dataset
from models.query_engine import QueryEngine
from column_types.registry import ColumnTypeRegistry
from operators.operator_registry import OperatorRegistry
from controller import AppController


# ---------------------------------------------------------------------------
# Helpers -- same style as tests/test_artifact_cache_location.py
# ---------------------------------------------------------------------------

def _solid_png(path: Path, colour: tuple[int, int, int]) -> None:
    Image.new("RGB", (240, 240), colour).save(path)


def _address_of(path: Path, root: Path) -> tuple[str, str]:
    return resolve_source(str(path), str(root))


def _wait_for_thumbnail(store: ArtifactStore) -> threading.Event:
    """Chain a ready-callback that sets a fresh Event. Call .wait(timeout)
    then .clear() before the next request."""
    event = threading.Event()
    previous = store.on_thumbnail_ready

    def _on_ready(table_name, row_id):
        if previous is not None:
            previous(table_name, row_id)
        event.set()

    store.on_thumbnail_ready = _on_ready
    return event


def _build_controller(tmp_path, scratch: str = "scratch"):
    """A real AppController whose store starts over a scratch folder that
    is NOT inside any project folder."""
    store = ArtifactStore(tmp_path / scratch)
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)
    dataset = Dataset()
    op_registry = OperatorRegistry()
    controller = AppController(
        dataset, QueryEngine(), store, registry, op_registry
    )
    return controller, dataset, store


def _render_first_row(controller):
    row_id = controller.get_all_row_ids()[0]
    row = controller.get_row(row_id)
    return controller.render_column_value(
        "full_path", row["full_path"], 150, "thumbnail",
        {"row_id": row_id, "column_name": "full_path"},
    )


def _hash_name(char: str = "a") -> str:
    """A filename that matches the sweep's <32 hex chars>.jpg pattern.
    Pass a distinct hex char per file to keep names unique."""
    return char * 32 + ".jpg"


def _jpg_names(directory: Path) -> set[str]:
    return {p.name for p in directory.glob("*.jpg")}


def _seed_thumb_key(store, address, *, on_disk, nbytes=64, mtime_ns=None):
    """Add a single 'thumbnail' index entry for `address`, its file
    present or not. Returns (key, path)."""
    fingerprint = SourceFingerprint(size=1, mtime_ns=1)
    key = ArtifactKey(
        address, fingerprint, "thumbnail", store.resolution_for("thumbnail")
    )
    path = store._dir / f"{key.stable_hash()}.jpg"
    if on_disk:
        path.write_bytes(b"\x00" * nbytes)
        if mtime_ns is not None:
            os.utime(path, ns=(mtime_ns, mtime_ns))
    store._index[key] = path
    store._fingerprints[address] = fingerprint
    return key, path


def _seed_address_fully_indexed(store, address, *, on_disk):
    """Add BOTH the thumbnail and preview index entries for `address` at
    one shared fingerprint, so is_cached() reports True."""
    fingerprint = SourceFingerprint(size=10, mtime_ns=20)
    store._fingerprints[address] = fingerprint
    made = []
    for purpose in ("thumbnail", "preview"):
        key = ArtifactKey(
            address, fingerprint, purpose, store.resolution_for(purpose)
        )
        path = store._dir / f"{key.stable_hash()}.jpg"
        if on_disk:
            path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 120)
        store._index[key] = path
        made.append((key, path))
    return made


# ===========================================================================
# 1. plan_sweep is pure: plain records, no filesystem touched at all.
# ===========================================================================

def test_plan_sweep_is_pure():
    entries = [
        SweepFile(path="A", size_bytes=100, mtime_ns=10),
        SweepFile(path="B", size_bytes=200, mtime_ns=20),
        SweepFile(path="C", size_bytes=50, mtime_ns=5),   # orphan
    ]
    indexed_paths = {"A", "B", "D"}   # D is named by the index, no file

    plan = plan_sweep(entries, indexed_paths=indexed_paths, max_bytes=1000)

    assert isinstance(plan, SweepPlan)
    assert plan.files_to_delete == ("C",), "the orphan should be the only delete"
    assert set(plan.paths_missing) == {"D"}
    assert plan.bytes_before == 350
    assert plan.bytes_after == 300   # A + B survive


def test_plan_sweep_ceiling_evicts_oldest_mtime_first():
    entries = [
        SweepFile(path="new", size_bytes=100, mtime_ns=300),
        SweepFile(path="old", size_bytes=100, mtime_ns=100),
        SweepFile(path="mid", size_bytes=100, mtime_ns=200),
    ]
    indexed_paths = {"new", "old", "mid"}

    plan = plan_sweep(entries, indexed_paths=indexed_paths, max_bytes=150)

    # 300 bytes, ceiling 150: drop 'old' (-> 200), then 'mid' (-> 100).
    assert plan.files_to_delete == ("old", "mid")
    assert plan.paths_missing == ()
    assert plan.bytes_after == 100


def test_plan_sweep_orphans_go_before_the_ceiling_is_checked():
    entries = [
        SweepFile(path="orphan", size_bytes=500, mtime_ns=1),
        SweepFile(path="kept", size_bytes=90, mtime_ns=2),
    ]
    indexed_paths = {"kept"}

    plan = plan_sweep(entries, indexed_paths=indexed_paths, max_bytes=100)

    # The orphan alone frees enough room; 'kept' stays.
    assert plan.files_to_delete == ("orphan",)
    assert plan.bytes_after == 90


def test_plan_sweep_delete_orphans_false_keeps_orphans_but_still_reports_missing():
    entries = [
        SweepFile(path="orphan1", size_bytes=100, mtime_ns=1),
        SweepFile(path="orphan2", size_bytes=100, mtime_ns=2),
        SweepFile(path="indexed", size_bytes=100, mtime_ns=3),
    ]
    indexed_paths = {"indexed", "gone"}   # 'gone' has no file on disk

    plan = plan_sweep(
        entries, indexed_paths=indexed_paths, max_bytes=10_000,
        delete_orphans=False,
    )

    # No orphan is planned for deletion...
    assert plan.files_to_delete == ()
    # ...but the missing-file reconciliation half still runs.
    assert set(plan.paths_missing) == {"gone"}
    # Nothing was removed, so bytes_after == bytes_before.
    assert plan.bytes_before == 300
    assert plan.bytes_after == 300


# ===========================================================================
# 2. An orphan JPEG (on disk, no index entry) is deleted.
# ===========================================================================

def test_orphan_jpeg_is_deleted(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    orphan = store._dir / _hash_name("b")
    orphan.write_bytes(b"\x00" * 4096)

    result = store.reconcile_and_evict()

    assert not orphan.exists(), "an orphan JPEG survived the sweep"
    assert result.files_deleted == 1


# ===========================================================================
# 3. A file that does not match the stable-hash pattern is never deleted,
#    even when the directory is over the ceiling.
# ===========================================================================

def test_non_hash_named_files_are_never_swept(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", disk_cache_max_bytes=0)

    index_json = store._dir / "artifact_index.json"
    index_json.write_bytes(b"x" * 10_000)
    readme = store._dir / "README.txt"
    readme.write_bytes(b"y" * 10_000)
    wrong_length = store._dir / ("a" * 31 + ".jpg")   # 31 chars, not 32
    wrong_length.write_bytes(b"z" * 10_000)
    non_hex = store._dir / ("z" * 32 + ".jpg")          # right length, not hex
    non_hex.write_bytes(b"u" * 10_000)
    matching_dir = store._dir / ("c" * 32 + ".jpg")
    matching_dir.mkdir()                               # matches the name, is a dir

    orphan = store._dir / _hash_name("b")           # this one SHOULD go
    orphan.write_bytes(b"o" * 10_000)

    store.reconcile_and_evict()

    assert index_json.exists()
    assert readme.exists()
    assert wrong_length.exists()
    assert non_hex.exists()
    assert matching_dir.is_dir()
    assert not orphan.exists(), "the genuine orphan should still be evicted"


# ===========================================================================
# 4. An indexed entry whose file is absent is dropped, and is_cached()
#    for that address goes True -> False across the sweep. This is the
#    grey-tile defect -- assert it directly.
# ===========================================================================

def test_indexed_but_absent_entry_flips_is_cached_false(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    address = "C:/never/on/disk.png"
    made = _seed_address_fully_indexed(store, address, on_disk=False)

    assert store.is_cached(address) is True, (
        "precondition: a seeded index entry reports cached even with no file"
    )

    store.reconcile_and_evict()

    assert store.is_cached(address) is False, (
        "is_cached still True after the sweep -- the grey tile is permanent"
    )
    for key, _path in made:
        assert key not in store._index, "a missing-file entry survived the sweep"


# ===========================================================================
# 5. After the sweep, a request for that address queues a job again.
# ===========================================================================

def test_after_sweep_a_request_queues_a_job_again(tmp_path):
    source = tmp_path / "s.png"
    _solid_png(source, (200, 10, 10))

    store = ArtifactStore(tmp_path / "artifacts")
    event = _wait_for_thumbnail(store)
    addr, src = _address_of(source, tmp_path)

    store.request_thumbnail("r", addr, Path(src), "frames")
    assert event.wait(timeout=30), "initial thumbnail never generated"
    assert store.is_cached(addr) is True

    # Delete the cached JPEGs out from under the index.
    for jpg in store._dir.glob("*.jpg"):
        jpg.unlink()
    assert store.is_cached(addr) is True, "index still seeded before the sweep"

    store.reconcile_and_evict()
    assert store.is_cached(addr) is False, "sweep did not drop the dead entries"

    # A fresh request must now queue a worker job.
    submitted: list = []
    real_submit = store._pool.submit

    def _counting_submit(fn, key=None):
        submitted.append(key)
        return real_submit(fn, key=key)

    store._pool.submit = _counting_submit
    event.clear()
    store.request_thumbnail("r", addr, Path(src), "frames")
    assert event.wait(timeout=30), "no callback after re-requesting a swept address"
    assert submitted, "request_thumbnail short-circuited instead of queuing a job"


# ===========================================================================
# 6. Over the ceiling, the oldest-mtime files go first and the survivors
#    are under the ceiling.
# ===========================================================================

def test_reconcile_evicts_oldest_mtime_first(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", disk_cache_max_bytes=250)

    made = []
    # Deliberately not in mtime order, so "oldest" cannot be "first added".
    for index, mtime_ns in enumerate((300_000_000, 100_000_000, 200_000_000)):
        address = f"C:/m/{index}.png"
        key, path = _seed_thumb_key(
            store, address, on_disk=True, nbytes=100, mtime_ns=mtime_ns
        )
        made.append((mtime_ns, key, path))

    # 3 x 100 = 300 bytes on disk, ceiling 250 -> exactly one must go.
    result = store.reconcile_and_evict()

    oldest_mtime, oldest_key, oldest_path = min(made, key=lambda item: item[0])
    assert not oldest_path.exists(), "the oldest-mtime file was not evicted"
    assert oldest_key not in store._index

    surviving = [path for (_m, key, path) in made if key in store._index]
    assert len(surviving) == 2
    total = sum(path.stat().st_size for path in surviving)
    assert total <= 250, f"survivors still over the ceiling: {total}"
    assert result.files_deleted == 1


# ===========================================================================
# 7. Every deleted file's index entry is gone: after the sweep no key in
#    _index names a path that no longer exists.
# ===========================================================================

def test_no_surviving_index_key_names_a_missing_file(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", disk_cache_max_bytes=150)

    # One real indexed file (small, stays), one indexed-but-absent, and a
    # big orphan the ceiling would force out anyway.
    _real_key, real_path = _seed_thumb_key(
        store, "C:/r/real.png", on_disk=True, nbytes=50
    )
    absent_key, _absent_path = _seed_thumb_key(
        store, "C:/a/absent.png", on_disk=False
    )
    orphan = store._dir / _hash_name("f")
    orphan.write_bytes(b"\x00" * 999)

    store.reconcile_and_evict()

    assert absent_key not in store._index
    assert real_path.exists()
    for key, path in list(store._index.items()):
        assert Path(path).exists(), f"{key} still names a gone path: {path}"


# ===========================================================================
# 8. A delete that raises OSError does not propagate and does not stop the
#    reconciliation half.
# ===========================================================================

def test_delete_oserror_does_not_propagate_or_block_reconcile(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")

    orphan = store._dir / _hash_name("b")
    orphan.write_bytes(b"\x00" * 32)
    absent_key, _absent_path = _seed_thumb_key(
        store, "C:/x/y.png", on_disk=False
    )

    def _boom(self, *args, **kwargs):
        raise OSError("cannot delete -- file is locked")

    monkeypatch.setattr(Path, "unlink", _boom)

    result = store.reconcile_and_evict()   # must not raise

    assert result.delete_failures >= 1, "a failed delete was not counted"
    assert orphan.exists(), "the undeletable orphan is still on disk, as expected"
    # The reconciliation half still ran despite the delete failure.
    assert absent_key not in store._index, (
        "a delete failure stopped the index reconciliation"
    )


# ===========================================================================
# 9. Saving a project writes an index that names no deleted file -- the
#    save_project ordering check (sweep runs BEFORE save_index).
# ===========================================================================

def test_saving_a_project_writes_an_index_naming_no_deleted_file(qapp, tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    _solid_png(media / "shot.png", (20, 200, 40))

    controller, _, store = _build_controller(tmp_path)
    event = _wait_for_thumbnail(store)
    controller.load_folder(media)
    _render_first_row(controller)   # cache miss -> queues the demand request
    assert event.wait(timeout=30), "thumbnail never generated"

    project = tmp_path / "proj"
    # Pre-create the artifacts folder holding an orphan JPEG that no
    # index record will ever name. set_artifacts_dir copies the real
    # (differently named) artifacts in beside it; the sweep must delete
    # this one before save_index writes the records.
    (project / "artifacts").mkdir(parents=True)
    orphan_name = _hash_name("d")
    orphan = project / "artifacts" / orphan_name
    orphan.write_bytes(b"\xff" * 8192)

    controller.save_project(project)

    assert not orphan.exists(), "save_project did not sweep before save_index"

    payload = json.loads((project / "artifact_index.json").read_text())
    records = payload["artifacts"]
    assert records, "nothing in the saved index to check"
    stored_names = {record["path"] for record in records}
    assert orphan_name not in stored_names
    for record in records:
        entry = project / "artifacts" / record["path"]
        assert entry.exists(), f"saved index names a file that is gone: {entry}"


# ===========================================================================
# 10. A clean directory (every indexed file present, under the ceiling) is
#     left completely untouched -- the sweep is not destructive by default.
# ===========================================================================

def test_clean_directory_is_untouched(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    made = _seed_address_fully_indexed(store, "C:/ok/pic.png", on_disk=True)

    result = store.reconcile_and_evict()

    assert result.files_deleted == 0
    assert result.index_entries_dropped == 0
    for key, path in made:
        assert key in store._index
        assert path.exists()
    assert store.is_cached("C:/ok/pic.png") is True


# ===========================================================================
# 11. load_index() reports whether the on-disk index is authoritative, so
#     load_project() knows when it is safe to run the destructive sweep.
# ===========================================================================

def test_load_index_return_value_signals_authoritativeness(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")

    # No index file: a saved project always has one, so this is a lost
    # or unsynced file, not "nothing cached" -> NOT safe to sweep.
    assert store.load_index(tmp_path) is False

    # The file exists but is unparseable: contents unknown -> NOT safe.
    (tmp_path / "artifact_index.json").write_text("}{ not json at all")
    assert store.load_index(tmp_path) is False

    # A format-version mismatch is a deliberate whole-index discard ->
    # those JPEGs are genuinely unreachable, so sweeping is correct.
    (tmp_path / "artifact_index.json").write_text(
        json.dumps({"format_version": 1, "artifacts": []})
    )
    assert store.load_index(tmp_path) is True


# ===========================================================================
# 12. An unreadable artifact_index.json on load does NOT wipe the on-disk
#     cache -- a recoverable error must not become permanent data loss.
# ===========================================================================

def test_unreadable_index_on_load_does_not_wipe_the_cache(qapp, tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    _solid_png(media / "shot.png", (40, 90, 200))

    controller, _, store = _build_controller(tmp_path)
    event = _wait_for_thumbnail(store)
    controller.load_folder(media)
    _render_first_row(controller)
    assert event.wait(timeout=30), "thumbnail never generated"

    project = tmp_path / "proj"
    controller.save_project(project)

    jpgs_before = _jpg_names(project / "artifacts")
    assert jpgs_before
    good_index = (project / "artifact_index.json").read_text()

    # Corrupt the index so load_index() cannot parse it.
    (project / "artifact_index.json").write_text("{ this is not json")
    controller.load_project(project)

    assert _jpg_names(project / "artifacts") == jpgs_before, (
        "an unreadable index triggered a cache wipe"
    )

    # Restore the index and reopen: the cache is intact and usable,
    # proving nothing was lost.
    (project / "artifact_index.json").write_text(good_index)
    controller.load_project(project)
    addr, _src = resolve_source(str(media / "shot.png"), str(project))
    assert store.is_cached(addr) is True


# ===========================================================================
# 13. The save one click after a missing-index load must NOT wipe the
#     cache. load_index() leaves _index empty; save_project sweeps
#     unconditionally; every JPEG would be an orphan. The trust flag
#     stops the deletion.
# ===========================================================================

def _prepare_saved_project(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    _solid_png(media / "shot.png", (40, 90, 200))

    controller, _, store = _build_controller(tmp_path)
    event = _wait_for_thumbnail(store)
    controller.load_folder(media)
    _render_first_row(controller)
    assert event.wait(timeout=30), "thumbnail never generated"

    project = tmp_path / "proj"
    controller.save_project(project)
    return controller, store, project, media


def test_save_after_missing_index_load_does_not_wipe_the_cache(qapp, tmp_path):
    controller, store, project, media = _prepare_saved_project(tmp_path)

    jpgs_before = _jpg_names(project / "artifacts")
    assert jpgs_before
    good_index = (project / "artifact_index.json").read_text()

    # The index file vanishes (partial sync / lost file).
    (project / "artifact_index.json").unlink()
    controller.load_project(project)          # sweep skipped on load
    assert store._index_authoritative is False

    # One click later: the user saves. save_project sweeps unconditionally.
    controller.save_project(project)

    assert _jpg_names(project / "artifacts") == jpgs_before, (
        "save after a missing-index load deleted the cached JPEGs as orphans"
    )

    # The JPEGs are still the real ones: restore the original index and
    # they serve.
    (project / "artifact_index.json").write_text(good_index)
    controller.load_project(project)
    addr, _src = resolve_source(str(media / "shot.png"), str(project))
    assert store.is_cached(addr) is True


# ===========================================================================
# 14. Same, for an unparseable artifact_index.json.
# ===========================================================================

def test_save_after_unparseable_index_load_does_not_wipe_the_cache(qapp, tmp_path):
    controller, store, project, media = _prepare_saved_project(tmp_path)

    jpgs_before = _jpg_names(project / "artifacts")
    assert jpgs_before

    (project / "artifact_index.json").write_text("}{ not json")
    controller.load_project(project)
    assert store._index_authoritative is False

    controller.save_project(project)

    assert _jpg_names(project / "artifacts") == jpgs_before, (
        "save after an unparseable-index load deleted the cached JPEGs"
    )


# ===========================================================================
# 15. The trust flag does not latch: reset() restores orphan deletion.
# ===========================================================================

def test_index_authoritative_flag_does_not_latch_across_reset(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")

    # A failed load (no index file) drops the flag.
    assert store.load_index(tmp_path) is False
    assert store._index_authoritative is False

    store.reset()
    assert store._index_authoritative is True, "reset() did not clear the flag"

    # A sweep now deletes orphans again.
    orphan = store._dir / _hash_name("e")
    orphan.write_bytes(b"\x00" * 128)
    store.reconcile_and_evict()
    assert not orphan.exists(), "reset() did not un-latch orphan deletion"


# ===========================================================================
# 16. The sweep's filename pattern must keep matching what stable_hash()
#     actually produces, so a change to the hash truncation cannot
#     silently make every cache file invisible to the sweep.
# ===========================================================================

def test_artifact_filename_re_matches_real_stable_hash():
    fingerprint = SourceFingerprint(size=123, mtime_ns=456)
    for purpose, resolution in (("thumbnail", 150), ("preview", 600)):
        key = ArtifactKey("C:/some/where.png", fingerprint, purpose, resolution)
        name = f"{key.stable_hash()}.jpg"
        assert _ARTIFACT_FILENAME_RE.match(name), (
            f"the sweep pattern no longer matches stable_hash() output: {name}"
        )
