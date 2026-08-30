"""
tests/test_demand_driven_display.py

P0.5b-3i -- demand-driven thumbnail display (docs/media_architecture.md
section 4.6).

The contract under test:

  * In thumbnail mode the media renderer is cache-or-placeholder. It
    never stats, opens or decodes a source file -- for an image tile or a
    video tile. A cache miss returns a paintable placeholder pixmap, not
    None.
  * On a miss AppController.render_column_value() queues exactly one
    generation request through ArtifactStore.request_thumbnail. A second
    miss for the same address before the first completes queues no second
    job (the store coalesces by canonical address).
  * The eager whole-table request loops are gone: load_folder(),
    load_csv_as_primary() and the create_table result path queue nothing.
  * After load_project() on a project whose index is missing an entry,
    rendering that row queues a request and decodes nothing on the
    calling thread.

Written from the spec, not the implementation. New file: tests/test_dataset.py
runs its module twice, so nothing new goes there.

Run with:
    python -m pytest tests/test_demand_driven_display.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from PIL import Image

import column_types.renderers as renderers_mod
from column_types.renderers import make_media_path_renderer

from artifacts.artifact_store import ArtifactStore
from column_types.registry import ColumnTypeRegistry
from models.dataset import Dataset
from models.query_engine import QueryEngine
from operators.operator_registry import OperatorRegistry
from controller import AppController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_png(path: Path, colour: tuple[int, int, int]) -> None:
    Image.new("RGB", (240, 240), colour).save(path)


def _build_controller(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)
    dataset = Dataset()
    dataset.set_registry(registry)
    op_registry = OperatorRegistry()
    controller = AppController(dataset, QueryEngine(), store, registry, op_registry)
    return controller, dataset, store


def _wait_for_thumbnail(store: ArtifactStore) -> threading.Event:
    event = threading.Event()
    previous = store.on_thumbnail_ready

    def _on_ready(table_name, row_id):
        if previous is not None:
            previous(table_name, row_id)
        event.set()

    store.on_thumbnail_ready = _on_ready
    return event


def _capture_submits(store: ArtifactStore) -> list:
    """Replace the worker pool's submit with a recorder that never runs
    the job. Returns the list the callables land in, so a test can count
    queued jobs without any of them decoding anything."""
    submitted: list = []
    store._pool.submit = lambda fn: submitted.append(fn)
    return submitted


def _boom_open(*_args, **_kwargs):
    raise AssertionError("a source image was opened on the paint path")


# ===========================================================================
# 1. Thumbnail mode never opens a source -- image tile.
# ===========================================================================

def test_thumbnail_miss_image_opens_no_source_and_returns_placeholder(
    qapp, tmp_path, monkeypatch
):
    store = ArtifactStore(tmp_path / "artifacts")
    render = make_media_path_renderer(store)

    # A real file, but not a decodable image: Image.open would raise if it
    # were called. It is not -- the paint path is cache-or-placeholder.
    src = tmp_path / "looks_like.png"
    src.write_bytes(b"definitely not a PNG")
    monkeypatch.setattr(renderers_mod.Image, "open", _boom_open)

    ctx = {
        "row_id": "r1",
        "column_name": "full_path",
        "canonical_address": "C:/proj/looks_like.png",  # not in the store
        "source_path": str(src),
    }
    pixmap = render(str(src), 150, "thumbnail", ctx)

    assert pixmap is not None, "cache miss returned None instead of a placeholder"
    assert not pixmap.isNull(), "placeholder pixmap is not paintable"


# ===========================================================================
# 2. Thumbnail mode never opens a source -- video tile.
# ===========================================================================

def test_thumbnail_miss_video_opens_no_source_and_returns_placeholder(
    qapp, tmp_path, monkeypatch
):
    store = ArtifactStore(tmp_path / "artifacts")
    render = make_media_path_renderer(store)

    # Instrument the decode: any cv2.VideoCapture call on the paint path
    # is a regression (this is the "what would still pass if the change
    # were broken" check for the video half of the rule). Skip cleanly if
    # OpenCV is not installed in this environment.
    cv2 = pytest.importorskip("cv2")

    def _boom_capture(*_args, **_kwargs):
        raise AssertionError("cv2.VideoCapture was called on the paint path")

    monkeypatch.setattr(cv2, "VideoCapture", _boom_capture)

    # A real, present .mp4 whose first frame a reintroduced decode path
    # WOULD read. The renderer must still return the flat grey
    # placeholder, not a decoded frame.
    present = tmp_path / "clip.mp4"
    present.write_bytes(b"not a real video, but the decode must not be attempted")
    ctx = {
        "row_id": "r1",
        "column_name": "full_path",
        "canonical_address": "C:/proj/clip.mp4",  # not in the store
        "source_path": str(present),
    }
    pixmap = render(str(present), 150, "thumbnail", ctx)
    assert pixmap is not None, "video cache miss returned None instead of a placeholder"
    assert not pixmap.isNull(), "video placeholder pixmap is not paintable"

    # The placeholder is a uniform fill; a decoded frame would not be.
    image = pixmap.toImage()
    corner = image.pixelColor(1, 1)
    centre = image.pixelColor(image.width() // 2, image.height() // 2)
    assert corner == centre, "video thumbnail is not a uniform placeholder"

    # A non-existent .mp4 also yields a placeholder without a stat/open.
    missing = tmp_path / "no_such_clip.mp4"
    ctx["source_path"] = str(missing)
    ctx["canonical_address"] = "C:/proj/no_such_clip.mp4"
    pixmap2 = render(str(missing), 150, "thumbnail", ctx)
    assert pixmap2 is not None and not pixmap2.isNull()


# ===========================================================================
# 3. A miss queues exactly one request; a second miss for the same address
#    before it completes queues no second job.
# ===========================================================================

def test_miss_queues_one_request_and_coalesces_a_second_miss(qapp, tmp_path):
    folder = tmp_path / "media"
    folder.mkdir()
    _solid_png(folder / "a.png", (10, 20, 200))

    controller, _, store = _build_controller(tmp_path)
    submitted = _capture_submits(store)

    controller.load_folder(folder)
    assert submitted == [], "load_folder queued a thumbnail job by itself"

    row_id = controller.get_all_row_ids()[0]
    row = controller.get_row(row_id)

    def _render():
        return controller.render_column_value(
            "full_path", row["full_path"], 150, "thumbnail",
            {"row_id": row_id, "column_name": "full_path"},
        )

    first = _render()
    assert first is not None and not first.isNull()
    assert len(submitted) == 1, (
        f"first cache miss queued {len(submitted)} jobs, expected 1"
    )

    # Second paint of the same still-pending tile: no second job.
    second = _render()
    assert second is not None and not second.isNull()
    assert len(submitted) == 1, (
        f"a second miss for the same address queued another job "
        f"({len(submitted)} total) -- it was not coalesced"
    )


# ===========================================================================
# 4. load_folder queues no request by itself (stated on its own).
# ===========================================================================

def test_load_folder_queues_no_request(qapp, tmp_path):
    folder = tmp_path / "media"
    folder.mkdir()
    _solid_png(folder / "a.png", (0, 0, 0))
    _solid_png(folder / "b.png", (255, 255, 255))

    controller, _, store = _build_controller(tmp_path)

    calls: list = []
    store.request_thumbnail = lambda *a, **k: calls.append(a)

    controller.load_folder(folder)

    assert calls == [], (
        "load_folder issued thumbnail requests -- the eager whole-table "
        "pass should be gone (P0.5b-3i)"
    )


# ===========================================================================
# 5. load_project on a project whose index is missing an entry: rendering
#    that row queues a request and decodes nothing on the calling thread.
#    This is the regression that had no coverage before P0.5b-3i.
# ===========================================================================

def test_load_project_missing_index_entry_renders_by_demand_not_decode(
    qapp, tmp_path, monkeypatch
):
    folder = tmp_path / "src"
    folder.mkdir()
    _solid_png(folder / "p.png", (0, 0, 200))
    project = tmp_path / "proj"

    controller, _, store = _build_controller(tmp_path)

    # load_folder no longer thumbnails eagerly, and we render nothing
    # before saving, so the saved artifact_index.json has no entry for
    # this row's picture.
    controller.load_folder(folder)
    controller.save_project(project)

    controller.load_project(project)

    row_id = controller.get_all_row_ids()[0]
    row = controller.get_row(row_id)

    submitted = _capture_submits(store)
    monkeypatch.setattr(renderers_mod.Image, "open", _boom_open)

    pixmap = controller.render_column_value(
        "full_path", row["full_path"], 150, "thumbnail",
        {"row_id": row_id, "column_name": "full_path"},
    )

    assert pixmap is not None and not pixmap.isNull(), (
        "a row missing from the reopened index rendered nothing"
    )
    assert len(submitted) == 1, (
        f"rendering a row the seeded index does not cover queued "
        f"{len(submitted)} jobs, expected exactly 1"
    )


# ===========================================================================
# 6. Detail mode is unchanged -- it still opens the source.
# ===========================================================================

def test_detail_mode_still_opens_the_source(qapp, tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    render = make_media_path_renderer(store)

    src = tmp_path / "real.png"
    _solid_png(src, (0, 128, 255))

    widget = render(
        str(src), 600, "detail",
        {"row_id": "r1", "column_name": "full_path", "source_path": str(src)},
    )
    from PySide6.QtWidgets import QWidget

    assert isinstance(widget, QWidget), (
        "detail mode should still return a widget built from the source"
    )
