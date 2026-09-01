"""
tests/test_artifact_cache_location.py

P0.5b-2ii-a -- the artifact cache directory is bound to the project
folder on save and load.

`ArtifactStore.set_artifacts_dir(new_dir)` re-points the cache directory,
rebuilds the ArtifactCodec so its containment boundary follows the new
root, and migrates the index: an indexed JPEG outside the new root is
COPIED in and its entry repointed; an entry whose file is gone is
dropped. `AppController.save_project` / `load_project` call it with
`project_path / "artifacts"` so a saved project keeps its thumbnails and
reopens without regenerating them.

P0.5b-2ii-b1 -- the saved `artifact_index.json` stores each path RELATIVE
to the artifacts directory, so a project folder can move between machines
(e.g. a Google Drive Streaming path) without the cache being lost. Same
item folds `AppController._artifact_is_cached` and
`ArtifactStore._both_present` into one public `ArtifactStore.is_cached`.

Written from the work-item spec, not from the implementation. New file on
purpose: tests/test_dataset.py runs everything in it twice.

Run with:
    python -m pytest tests/test_artifact_cache_location.py
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from PIL import Image

from artifacts.artifact_store import (
    ArtifactStore,
    DEFAULT_THUMBNAIL_MAX_SIDE,
    DEFAULT_PREVIEW_MAX_SIDE,
)

# Thumbnail / preview resolutions are now instance state, derived from a
# store's configured sizes. A default-constructed store uses these; the
# by-hand ArtifactKeys below must match store.resolution_for(...).
THUMBNAIL_RESOLUTION = DEFAULT_THUMBNAIL_MAX_SIDE
PREVIEW_RESOLUTION = DEFAULT_PREVIEW_MAX_SIDE
from artifacts.artifact_codec import ArtifactCodecError
from media.artifact_key import ArtifactKey, SourceFingerprint
from media.media_address import resolve_source

from models.dataset import Dataset
from models.query_engine import QueryEngine
from column_types.registry import ColumnTypeRegistry
from operators.operator_registry import OperatorRegistry
from controller import AppController


# ---------------------------------------------------------------------------
# Helpers -- same style as tests/test_artifact_identity.py
# ---------------------------------------------------------------------------

def _solid_png(path: Path, colour: tuple[int, int, int]) -> None:
    Image.new("RGB", (240, 240), colour).save(path)


def _address_of(path: Path, root: Path) -> tuple[str, str]:
    """(canonical address, absolute source path) for a media file."""
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


def _centre_colour(pixmap):
    image = pixmap.toImage()
    return image.pixelColor(image.width() // 2, image.height() // 2)


def _build_controller(tmp_path, scratch: str = "scratch"):
    """A real AppController whose store starts over a scratch folder that
    is NOT inside any project folder -- so a save/load has real migration
    work to do."""
    store = ArtifactStore(tmp_path / scratch)
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)
    dataset = Dataset()
    dataset.set_registry(registry)
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


def _jpg_names(directory: Path) -> set[str]:
    return {p.name for p in directory.glob("*.jpg")}


# ===========================================================================
# 1. After set_artifacts_dir, a newly generated artifact is written under
#    the new directory and nowhere else.
# ===========================================================================

def test_new_artifact_lands_only_in_the_new_directory(tmp_path):
    source = tmp_path / "s.png"
    _solid_png(source, (200, 10, 10))

    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"

    store = ArtifactStore(old_dir)
    store.set_artifacts_dir(new_dir)

    event = _wait_for_thumbnail(store)
    addr, src = _address_of(source, tmp_path)
    store.request_thumbnail("r", addr, Path(src), "frames")
    assert event.wait(timeout=30), "thumbnail was never generated"

    assert len(_jpg_names(new_dir)) == 2, (
        "expected thumbnail + preview under the new directory"
    )
    assert _jpg_names(old_dir) == set(), (
        "a JPEG was written under the old directory after the rebind"
    )
    # And the index points into the new directory.
    thumb = store.get(addr, "thumbnail")
    assert thumb is not None
    assert Path(thumb).resolve().parent == new_dir.resolve()


# ===========================================================================
# 2. An indexed JPEG in the old directory is COPIED under the new directory
#    by migration, and the index entry names the new path. Copy, not move:
#    the old file is still there.
# ===========================================================================

def test_migration_copies_indexed_jpeg_and_repoints_the_entry(tmp_path):
    source = tmp_path / "s.png"
    _solid_png(source, (10, 180, 40))

    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"

    store = ArtifactStore(old_dir)
    event = _wait_for_thumbnail(store)
    addr, src = _address_of(source, tmp_path)
    store.request_thumbnail("r", addr, Path(src), "frames")
    assert event.wait(timeout=30)

    old_thumb = Path(store.get(addr, "thumbnail"))
    old_preview = Path(store.get(addr, "preview"))
    assert old_thumb.parent == old_dir
    old_names = {old_thumb.name, old_preview.name}

    store.set_artifacts_dir(new_dir)

    new_thumb = Path(store.get(addr, "thumbnail"))
    new_preview = Path(store.get(addr, "preview"))
    # The index now names files inside the new directory.
    assert new_thumb.parent.resolve() == new_dir.resolve()
    assert new_preview.parent.resolve() == new_dir.resolve()
    # Same filenames, copied across.
    assert {new_thumb.name, new_preview.name} == old_names
    assert new_thumb.exists() and new_preview.exists()
    # Copy, not move: the originals are still in the old directory.
    assert old_thumb.exists() and old_preview.exists()


# ===========================================================================
# 3. An index entry whose file is missing is DROPPED by migration, not
#    left pointing at a nonexistent path.
# ===========================================================================

def test_migration_drops_an_entry_whose_file_is_gone(tmp_path):
    source = tmp_path / "s.png"
    _solid_png(source, (30, 30, 200))

    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"

    store = ArtifactStore(old_dir)
    event = _wait_for_thumbnail(store)
    addr, src = _address_of(source, tmp_path)
    store.request_thumbnail("r", addr, Path(src), "frames")
    assert event.wait(timeout=30)

    # Delete the cached JPEGs out from under the index.
    for jpg in old_dir.glob("*.jpg"):
        jpg.unlink()

    store.set_artifacts_dir(new_dir)

    # The stale entries are gone -- not migrated to a dead path.
    assert store.get(addr, "thumbnail") is None
    assert store.get(addr, "preview") is None
    assert _jpg_names(new_dir) == set()


# ===========================================================================
# 4. After save_project, every path in the written artifact_index.json is
#    inside project_path / "artifacts".
# ===========================================================================

def test_saved_index_paths_are_all_inside_the_project_folder(qapp, tmp_path):
    folder = tmp_path / "media"
    folder.mkdir()
    _solid_png(folder / "shot.png", (220, 40, 40))

    controller, _, store = _build_controller(tmp_path)
    event = _wait_for_thumbnail(store)

    controller.load_folder(folder)
    _render_first_row(controller)  # cache miss -> queues the demand request
    assert event.wait(timeout=30), "thumbnail never generated"

    project = tmp_path / "proj"
    controller.save_project(project)

    index_path = project / "artifact_index.json"
    payload = json.loads(index_path.read_text())
    records = payload["artifacts"]
    assert records, "the saved index has no entries to check"

    # P0.5b-2ii-b1: paths are stored relative to the artifacts directory.
    # Each must resolve, under that directory, to a real file inside it.
    artifacts_root = (project / "artifacts").resolve()
    for record in records:
        stored = Path(record["path"])
        assert not stored.is_absolute(), (
            f"saved index entry is absolute: {stored}"
        )
        entry = (artifacts_root / stored).resolve()
        assert entry.parent == artifacts_root, (
            f"saved index entry resolves outside the project folder: {entry}"
        )
        assert entry.exists(), f"saved index entry does not exist: {entry}"


# ===========================================================================
# 5. Loading a project saved this way serves its pictures without queueing
#    any worker job. Assert on the pool, not on wall-clock time.
# ===========================================================================

def test_reloading_a_saved_project_queues_no_worker_job(qapp, tmp_path):
    folder = tmp_path / "media"
    folder.mkdir()
    _solid_png(folder / "shot.png", (40, 200, 60))

    controller, _, store = _build_controller(tmp_path)
    event = _wait_for_thumbnail(store)

    controller.load_folder(folder)
    _render_first_row(controller)
    assert event.wait(timeout=30), "thumbnail never generated"

    project = tmp_path / "proj"
    controller.save_project(project)

    # Count every submission to the pool from here on.
    submitted: list = []
    real_submit = store._pool.submit

    def _counting_submit(fn, key=None):
        submitted.append(key)
        return real_submit(fn, key=key)

    store._pool.submit = _counting_submit

    controller.load_project(project)
    pixmap = _render_first_row(controller)

    assert pixmap is not None and not pixmap.isNull()
    colour = _centre_colour(pixmap)
    assert colour.green() > colour.red(), (
        f"reloaded project did not render its own picture: {colour.getRgb()}"
    )
    assert submitted == [], (
        f"reloading a fully-thumbnailed project queued worker jobs: {submitted}"
    )


# ===========================================================================
# 6. Loading project A then project B leaves the store bound to B's folder,
#    and B's folder holds none of A's files.
# ===========================================================================

def test_load_a_then_b_binds_to_b_and_keeps_a_out(qapp, tmp_path):
    folder_a = tmp_path / "fa"
    folder_b = tmp_path / "fb"
    folder_a.mkdir()
    folder_b.mkdir()
    _solid_png(folder_a / "ra.png", (230, 20, 20))
    _solid_png(folder_b / "rb.png", (20, 20, 230))

    project_a = tmp_path / "proj_a"
    project_b = tmp_path / "proj_b"

    controller, _, store = _build_controller(tmp_path)
    event = _wait_for_thumbnail(store)

    controller.load_folder(folder_a)
    _render_first_row(controller)
    assert event.wait(timeout=30), "folder A thumbnail never generated"
    controller.save_project(project_a)
    # Capture A's real artifacts now, while project_a/artifacts holds
    # exactly them -- a later re-import can leave unrelated orphans here.
    a_names = _jpg_names(project_a / "artifacts")
    assert a_names

    event.clear()
    controller.load_folder(folder_b)
    _render_first_row(controller)
    assert event.wait(timeout=30), "folder B thumbnail never generated"
    controller.save_project(project_b)
    assert _jpg_names(project_b / "artifacts")

    controller.load_project(project_a)
    controller.load_project(project_b)

    # Bound to B's folder.
    assert store._dir.resolve() == (project_b / "artifacts").resolve()
    # B's folder holds none of A's content-addressed files.
    assert _jpg_names(project_b / "artifacts").isdisjoint(a_names), (
        "project B's artifacts folder contains a file from project A"
    )
    # And A's picture is not served any more.
    a_addr, _ = resolve_source(str(folder_a / "ra.png"), str(project_a))
    assert store.get(a_addr, "thumbnail") is None


# ===========================================================================
# 7. The codec boundary MOVES with the directory: after a rebind the codec
#    refuses a path under the OLD root and accepts one under the new root.
# ===========================================================================

def test_codec_boundary_moves_with_the_directory(tmp_path):
    source = tmp_path / "s.png"
    _solid_png(source, (180, 180, 20))

    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"

    store = ArtifactStore(old_dir)
    event = _wait_for_thumbnail(store)
    addr, src = _address_of(source, tmp_path)
    store.request_thumbnail("r", addr, Path(src), "frames")
    assert event.wait(timeout=30)

    old_thumb = Path(store.get(addr, "thumbnail"))
    assert old_thumb.parent == old_dir

    store.set_artifacts_dir(new_dir)

    # The old file still exists on disk, but the rebuilt codec refuses it.
    assert old_thumb.exists()
    with pytest.raises(ArtifactCodecError):
        store._codec.read_image(old_thumb)

    # A migrated file under the new root reads back fine.
    new_thumb = Path(store.get(addr, "thumbnail"))
    assert new_thumb.parent.resolve() == new_dir.resolve()
    assert store._codec.read_image(new_thumb).size[0] > 0


# ===========================================================================
# 8. A worker job that is mid-decode when set_artifacts_dir runs commits
#    NOTHING -- no index entry (which, on the save path, save_index would
#    otherwise serialize with an old-root path), no memo entry, no
#    notification. set_artifacts_dir bumps the generation exactly as
#    reset() does, so it is safe even with no preceding reset().
# ===========================================================================

def test_running_job_commits_nothing_across_a_rebind(tmp_path):
    source = tmp_path / "s.png"
    _solid_png(source, (200, 0, 0))
    addr, src = _address_of(source, tmp_path)

    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"

    in_decode = threading.Event()
    proceed = threading.Event()
    job_done = threading.Event()

    class Gated(ArtifactStore):
        def _decode_source(self, source_path):
            in_decode.set()
            assert proceed.wait(timeout=10)
            return super()._decode_source(source_path)

        def _run_job(self, *args, **kwargs):
            try:
                return super()._run_job(*args, **kwargs)
            finally:
                job_done.set()

    store = Gated(old_dir, worker_count=1)
    notified: list = []
    store.on_thumbnail_ready = lambda t, r: notified.append((t, r))

    store.request_thumbnail("r", addr, Path(src), "frames")
    assert in_decode.wait(timeout=5), "the job never started"

    # Rebind while the worker is blocked inside _decode_source.
    store.set_artifacts_dir(new_dir)
    proceed.set()
    assert job_done.wait(timeout=10), "the gated job never finished"

    # The stale job committed nothing.
    assert store.get(addr, "thumbnail") is None
    assert store._index == {}, "a job that raced the rebind left an index entry"
    assert store._fingerprints == {}
    assert store._verified == set()
    assert notified == [], "a job that raced the rebind sent a notification"


# ===========================================================================
# P0.5b-2ii-b1 -- relative index paths
# ===========================================================================

# ===========================================================================
# 9. Round trip across a MOVED project folder: save, move the whole folder
#    to a path it has never seen, load from there. Both is_cached() and
#    get_pixmap() must serve the cached picture. This is the defect the
#    item exists to fix, so it is the first test.
# ===========================================================================

def test_thumbnail_cache_survives_moving_the_project_folder(qapp, tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    shot = media / "shot.png"
    _solid_png(shot, (210, 40, 40))

    controller, _, store = _build_controller(tmp_path)
    event = _wait_for_thumbnail(store)
    controller.load_folder(media)
    _render_first_row(controller)  # cache miss -> queues the demand request
    assert event.wait(timeout=30), "thumbnail never generated"

    original = tmp_path / "proj"
    controller.save_project(original)

    # Move the entire project folder somewhere it has never lived.
    moved = tmp_path / "relocated" / "proj"
    moved.parent.mkdir()
    shutil.move(str(original), str(moved))

    controller.load_project(moved)

    addr, _src = resolve_source(str(shot), str(moved))
    assert store.is_cached(addr), (
        "a project loaded from a moved folder reports its cache as missing"
    )
    image = store.get_pixmap(addr, "thumbnail")
    assert image is not None, (
        "get_pixmap served nothing for a moved project's cached thumbnail"
    )


# ===========================================================================
# 10. save_index writes no absolute path.
# ===========================================================================

def test_saved_index_paths_are_relative(qapp, tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    _solid_png(media / "shot.png", (40, 210, 40))

    controller, _, store = _build_controller(tmp_path)
    event = _wait_for_thumbnail(store)
    controller.load_folder(media)
    _render_first_row(controller)
    assert event.wait(timeout=30), "thumbnail never generated"

    project = tmp_path / "proj"
    controller.save_project(project)

    payload = json.loads((project / "artifact_index.json").read_text())
    records = payload["artifacts"]
    assert records, "nothing in the saved index to check"
    for record in records:
        assert not Path(record["path"]).is_absolute(), (
            f"saved index stored an absolute path: {record['path']!r}"
        )


# ===========================================================================
# 11. A record whose stored "path" is absolute is skipped by load_index --
#     per record, not a whole-file discard: its relative sibling loads.
# ===========================================================================

def test_load_index_skips_a_record_with_an_absolute_path(tmp_path):
    source = tmp_path / "s.png"
    _solid_png(source, (0, 0, 210))

    store = ArtifactStore(tmp_path / "artifacts")
    event = _wait_for_thumbnail(store)
    addr, src = _address_of(source, tmp_path)
    store.request_thumbnail("r", addr, Path(src), "frames")
    assert event.wait(timeout=30)
    store.save_index(tmp_path)

    index_path = tmp_path / "artifact_index.json"
    payload = json.loads(index_path.read_text())
    # Rewrite ONLY the thumbnail record's path to an absolute string;
    # leave the preview record relative.
    made_absolute = None
    for record in payload["artifacts"]:
        if record["purpose"] == "thumbnail":
            record["path"] = str(
                (tmp_path / "artifacts" / record["path"]).resolve()
            )
            made_absolute = record["path"]
    assert made_absolute is not None and Path(made_absolute).is_absolute()
    index_path.write_text(json.dumps(payload))

    store2 = ArtifactStore(tmp_path / "artifacts")
    store2.load_index(tmp_path)

    assert store2.get(addr, "thumbnail") is None, (
        "load_index kept a record whose path was absolute"
    )
    assert store2.get(addr, "preview") is not None, (
        "load_index discarded the relative sibling record too"
    )


# ===========================================================================
# 12. An index file written with format_version 2 is discarded whole.
# ===========================================================================

def test_version_2_index_is_discarded_whole(tmp_path):
    source = tmp_path / "s.png"
    _solid_png(source, (0, 210, 0))

    store = ArtifactStore(tmp_path / "artifacts")
    event = _wait_for_thumbnail(store)
    addr, src = _address_of(source, tmp_path)
    store.request_thumbnail("r", addr, Path(src), "frames")
    assert event.wait(timeout=30)
    store.save_index(tmp_path)

    index_path = tmp_path / "artifact_index.json"
    payload = json.loads(index_path.read_text())
    assert payload["format_version"] == 3
    payload["format_version"] = 2
    index_path.write_text(json.dumps(payload))

    store2 = ArtifactStore(tmp_path / "artifacts")
    store2.load_index(tmp_path)

    assert store2._index == {}, "a version-2 index was not discarded whole"
    assert store2.get(addr, "thumbnail") is None
    assert store2.get(addr, "preview") is None


# ===========================================================================
# 13. is_cached is False when only the thumbnail key is in the index and
#     the preview key is not -- "either missing" is the trigger.
# ===========================================================================

def test_is_cached_needs_both_thumbnail_and_preview(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    address = "C:/x/y.png"
    fingerprint = SourceFingerprint(size=10, mtime_ns=20)

    # Seed the memo and ONLY the thumbnail key.
    thumb_key = ArtifactKey(address, fingerprint, "thumbnail", THUMBNAIL_RESOLUTION)
    store._fingerprints[address] = fingerprint
    store._index[thumb_key] = tmp_path / "artifacts" / "a.jpg"

    assert store.is_cached(address) is False, (
        "is_cached returned True with the preview key absent"
    )

    # Adding the preview key flips it to True.
    preview_key = ArtifactKey(address, fingerprint, "preview", PREVIEW_RESOLUTION)
    store._index[preview_key] = tmp_path / "artifacts" / "b.jpg"
    assert store.is_cached(address) is True


# ===========================================================================
# 14. is_cached is False for an address with no fingerprint-memo entry.
# ===========================================================================

def test_is_cached_false_without_a_fingerprint_memo(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    assert store.is_cached("C:/never/seen.png") is False


# ===========================================================================
# 15. The verified/unverified distinction survives: an address seeded by
#     load_index (unverified) still queues a worker on request_thumbnail
#     rather than short-circuiting, even though is_cached() is True for it.
# ===========================================================================

def test_seeded_address_still_queues_a_worker(tmp_path):
    source = tmp_path / "s.png"
    _solid_png(source, (210, 0, 0))

    store = ArtifactStore(tmp_path / "artifacts")
    event = _wait_for_thumbnail(store)
    addr, src = _address_of(source, tmp_path)
    store.request_thumbnail("r", addr, Path(src), "frames")
    assert event.wait(timeout=30)
    store.save_index(tmp_path)

    # Reopen: the address is seeded (unverified) from the saved index.
    store2 = ArtifactStore(tmp_path / "artifacts")
    store2.load_index(tmp_path)
    assert store2.is_cached(addr) is True, "seeded entry should serve on paint"

    # Count pool submissions across the next request.
    submitted: list = []
    real_submit = store2._pool.submit

    def _counting_submit(fn, key=None):
        submitted.append(key)
        return real_submit(fn, key=key)

    store2._pool.submit = _counting_submit

    event2 = _wait_for_thumbnail(store2)
    store2.request_thumbnail("r", addr, Path(src), "frames")
    assert event2.wait(timeout=30), "seeded address never delivered a callback"
    assert submitted, (
        "request_thumbnail short-circuited on an unverified (seeded) fingerprint"
    )


# ===========================================================================
# 16. load_index skips a relative record whose '..'-laden path escapes the
#     artifacts directory -- the same outside-root rule save_index applies,
#     so is_cached() cannot report an unreadable tile as cached.
# ===========================================================================

def test_load_index_skips_a_relative_path_that_escapes_the_cache_root(tmp_path):
    source = tmp_path / "s.png"
    _solid_png(source, (0, 130, 130))

    store = ArtifactStore(tmp_path / "artifacts")
    event = _wait_for_thumbnail(store)
    addr, src = _address_of(source, tmp_path)
    store.request_thumbnail("r", addr, Path(src), "frames")
    assert event.wait(timeout=30)
    store.save_index(tmp_path)

    index_path = tmp_path / "artifact_index.json"
    payload = json.loads(index_path.read_text())
    # Point the thumbnail record at a path that climbs out of the cache
    # dir; leave the preview record alone.
    for record in payload["artifacts"]:
        if record["purpose"] == "thumbnail":
            record["path"] = "../../elsewhere/x.jpg"
    index_path.write_text(json.dumps(payload))

    store2 = ArtifactStore(tmp_path / "artifacts")
    store2.load_index(tmp_path)

    assert store2.get(addr, "thumbnail") is None, (
        "load_index kept a record whose relative path escapes the cache root"
    )
    assert store2.get(addr, "preview") is not None, (
        "load_index discarded the in-root sibling record too"
    )
    assert store2.is_cached(addr) is False, (
        "is_cached reports True when one artifact escaped the cache root"
    )


# ===========================================================================
# 17. load_index skips a degenerate empty/"." path -- it would rebuild to
#     the artifacts directory itself, which get_pixmap cannot read but
#     is_cached would otherwise count as present.
# ===========================================================================

def test_load_index_skips_a_degenerate_empty_path(tmp_path):
    source = tmp_path / "s.png"
    _solid_png(source, (90, 90, 90))

    store = ArtifactStore(tmp_path / "artifacts")
    event = _wait_for_thumbnail(store)
    addr, src = _address_of(source, tmp_path)
    store.request_thumbnail("r", addr, Path(src), "frames")
    assert event.wait(timeout=30)
    store.save_index(tmp_path)

    index_path = tmp_path / "artifact_index.json"
    payload = json.loads(index_path.read_text())
    for record in payload["artifacts"]:
        if record["purpose"] == "thumbnail":
            record["path"] = ""
    index_path.write_text(json.dumps(payload))

    store2 = ArtifactStore(tmp_path / "artifacts")
    store2.load_index(tmp_path)

    assert store2.get(addr, "thumbnail") is None
    assert store2.get(addr, "preview") is not None
    assert store2.is_cached(addr) is False


# ===========================================================================
# 18. One malformed record (no "path" key, or a non-string path) is
#     skipped, not fatal: the rest of the index still loads.
# ===========================================================================

def test_load_index_skips_a_record_with_no_usable_path(tmp_path):
    source = tmp_path / "s.png"
    _solid_png(source, (120, 60, 30))

    store = ArtifactStore(tmp_path / "artifacts")
    event = _wait_for_thumbnail(store)
    addr, src = _address_of(source, tmp_path)
    store.request_thumbnail("r", addr, Path(src), "frames")
    assert event.wait(timeout=30)
    store.save_index(tmp_path)

    index_path = tmp_path / "artifact_index.json"
    payload = json.loads(index_path.read_text())
    for record in payload["artifacts"]:
        if record["purpose"] == "thumbnail":
            del record["path"]          # missing key
        else:
            record["path"] = None       # non-string
    index_path.write_text(json.dumps(payload))

    store2 = ArtifactStore(tmp_path / "artifacts")
    store2.load_index(tmp_path)  # must not raise

    # Both records were unusable, so nothing loaded -- but the call
    # returned normally rather than aborting load_project.
    assert store2._index == {}
    assert store2.get(addr, "thumbnail") is None


# ===========================================================================
# 19. The producer (_run_job) and the consumer (is_cached / _both_present_
#     locked) must agree on the resolution that enters the key. Both now
#     route through resolution_for(); this pins that the key _run_job
#     actually writes carries resolution_for(purpose), so a future change
#     to resolution_for cannot silently desynchronise them.
# ===========================================================================

def test_run_job_key_resolution_matches_resolution_for(tmp_path):
    source = tmp_path / "s.png"
    _solid_png(source, (70, 140, 210))

    store = ArtifactStore(tmp_path / "artifacts")
    event = _wait_for_thumbnail(store)
    addr, src = _address_of(source, tmp_path)
    store.request_thumbnail("r", addr, Path(src), "frames")
    assert event.wait(timeout=30), "thumbnail was never generated"

    # _run_job committed one key per purpose into the index. Each key's
    # resolution must be exactly what resolution_for() returns for its
    # purpose -- the same call is_cached() makes to look them up.
    by_purpose = {key.purpose: key for key in store._index}
    assert set(by_purpose) == {"thumbnail", "preview"}, (
        f"expected a thumbnail and a preview key, got {sorted(by_purpose)}"
    )
    for purpose, key in by_purpose.items():
        assert key.resolution == store.resolution_for(purpose), (
            f"_run_job wrote a {purpose} key at resolution {key.resolution}, "
            f"but resolution_for({purpose!r}) is {store.resolution_for(purpose)} "
            f"-- producer and is_cached() are desynchronised"
        )

    # And the round trip through is_cached() agrees.
    assert store.is_cached(addr) is True
