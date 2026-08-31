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

Written from the work-item spec, not from the implementation. New file on
purpose: tests/test_dataset.py runs everything in it twice.

Run with:
    python -m pytest tests/test_artifact_cache_location.py
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from PIL import Image

from artifacts.artifact_store import ArtifactStore
from artifacts.artifact_codec import ArtifactCodecError
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

    artifacts_root = (project / "artifacts").resolve()
    for record in records:
        entry = Path(record["path"]).resolve()
        assert entry.parent == artifacts_root, (
            f"saved index entry is outside the project folder: {entry}"
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
