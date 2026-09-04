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

import os
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
from media.media_address import MediaAddressError
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
    # WorkerPool.submit takes an optional opaque `key` (P0.5b-3ii-a); the
    # recorder must accept it even though this helper ignores it.
    store._pool.submit = lambda fn, key=None: submitted.append(fn)
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


# ===========================================================================
# 7. Viewport -> ArtifactStore.set_wanted_addresses (P0.5b-3ii-b).
#
#    AppController is the consumer of set_wanted_addresses(). It calls it
#    whenever a gallery reports or clears a displayed range, with the
#    canonical addresses of exactly the rows in the union of the reported
#    ranges. The controller does no stat/open/decode to build that set.
# ===========================================================================

def _folder_of_pngs(tmp_path, count):
    folder = tmp_path / "media"
    folder.mkdir()
    for i in range(count):
        _solid_png(folder / f"{i:02d}.png", (i * 7 % 256, 20, 30))
    return folder


def _capture_wanted(store):
    """Record every set_wanted_addresses() call as a set of str."""
    calls: list = []
    store.set_wanted_addresses = lambda addrs: calls.append({str(a) for a in addrs})
    return calls


def _addresses_of_positions(controller, positions):
    """The canonical addresses of the rows at the given flat-order
    positions, resolved the same way the controller resolves a cell."""
    ids = controller.get_visible_row_ids()
    wanted = set()
    for pos in positions:
        cell = controller.get_row(ids[pos])["full_path"]
        address, _src = controller._resolve_media_cell(cell)
        wanted.add(str(address))
    return wanted


def test_reported_range_publishes_exactly_that_range_s_addresses(qapp, tmp_path):
    folder = _folder_of_pngs(tmp_path, 5)
    controller, _, store = _build_controller(tmp_path)
    controller.load_folder(folder)
    calls = _capture_wanted(store)

    layout = controller.get_result_layout()
    controller.report_displayed_range("vp", 1, 4, layout.result_id)

    assert calls, "reporting a displayed range did not call set_wanted_addresses"
    assert calls[-1] == _addresses_of_positions(controller, [1, 2, 3])


def test_unparseable_media_cell_is_skipped_not_raised(qapp, tmp_path):
    folder = _folder_of_pngs(tmp_path, 3)
    controller, _, store = _build_controller(tmp_path)
    controller.load_folder(folder)
    calls = _capture_wanted(store)

    # Make the first row's cell look like a value that is not a media
    # address (a literal '#' with no '=' after it). The controller must
    # skip it, exactly as render_column_value() does, not propagate the
    # MediaAddressError.
    real_resolve = controller._resolve_media_cell
    ids = controller.get_visible_row_ids()
    bad_cell = controller.get_row(ids[0])["full_path"]

    def flaky_resolve(value):
        if value == bad_cell:
            raise MediaAddressError("not an address")
        return real_resolve(value)

    controller._resolve_media_cell = flaky_resolve

    layout = controller.get_result_layout()
    controller.report_displayed_range("vp", 0, 3, layout.result_id)

    # No raise, and the surviving two rows are still published.
    assert calls[-1] == _addresses_of_positions(controller, [1, 2])


def test_clearing_the_last_displayed_range_publishes_an_empty_set(qapp, tmp_path):
    folder = _folder_of_pngs(tmp_path, 4)
    controller, _, store = _build_controller(tmp_path)
    controller.load_folder(folder)
    calls = _capture_wanted(store)

    layout = controller.get_result_layout()
    controller.report_displayed_range("vp", 0, 4, layout.result_id)
    assert calls[-1] != set(), "a non-empty range should want some addresses"

    controller.clear_displayed_range("vp")
    assert calls[-1] == set(), (
        "clearing the last displayed range should leave nothing wanted"
    )


def test_two_viewports_union_their_ranges_rather_than_overwrite(qapp, tmp_path):
    folder = _folder_of_pngs(tmp_path, 8)
    controller, _, store = _build_controller(tmp_path)
    controller.load_folder(folder)
    calls = _capture_wanted(store)

    layout = controller.get_result_layout()
    controller.report_displayed_range("vp1", 0, 2, layout.result_id)
    controller.report_displayed_range("vp2", 5, 8, layout.result_id)

    # The second report does not replace the first: the published set is
    # the union of both galleries' rows.
    assert calls[-1] == _addresses_of_positions(controller, [0, 1, 5, 6, 7])


def test_update_wanted_addresses_touches_no_filesystem(qapp, tmp_path, monkeypatch):
    folder = _folder_of_pngs(tmp_path, 4)
    controller, _, store = _build_controller(tmp_path)
    controller.load_folder(folder)
    calls = _capture_wanted(store)

    layout = controller.get_result_layout()
    # Compute the expectation BEFORE the filesystem is sealed off -- the
    # assertion at the end only compares sets.
    expected = _addresses_of_positions(controller, [0, 1, 2, 3])

    # The helper's architectural claim (docs/media_architecture.md 4.4,
    # and its own docstring) is that it resolves cells with pure path
    # arithmetic and never stats, opens or decodes anything. Seal off
    # every filesystem entry point it could plausibly reach and assert the
    # report still goes through. Path.stat / Path.exists / builtins.open
    # are only the pathlib-and-open surface; os.stat, os.path.exists,
    # Path.is_file and Path.resolve are separate C-level entry points that
    # hit the filesystem without routing through those three (os.path.exists
    # calls os.stat directly, Path.is_file calls os.stat via the accessor,
    # and Path.resolve calls os.path.realpath), so a leak through any of
    # them would slip past the original seal.
    def _boom(*_args, **_kwargs):
        raise AssertionError("_update_wanted_addresses touched the filesystem")

    monkeypatch.setattr(Path, "stat", _boom)
    monkeypatch.setattr(Path, "exists", _boom)
    monkeypatch.setattr("builtins.open", _boom)
    monkeypatch.setattr(os, "stat", _boom)
    monkeypatch.setattr("os.path.exists", _boom)
    monkeypatch.setattr(Path, "is_file", _boom)
    monkeypatch.setattr(Path, "resolve", _boom)

    controller.report_displayed_range("vp", 0, 4, layout.result_id)

    assert calls[-1] == expected


# ===========================================================================
# 8. The thumbnail-ready notification names the tile's OWN table.
#
#    render_column_value() queues the demand request under the table the
#    calling tile names in its context dict, falling back to the
#    controller's active table only when the caller supplies none. The
#    attribution is the caller's to make, not read back off controller
#    state.
# ===========================================================================

def _run_the_one_submitted_job(submitted: list) -> None:
    """Run the single job _capture_submits recorded, on the calling
    thread. _run_job does a real decode + encode and then notifies every
    subscriber through store.on_thumbnail_ready."""
    assert len(submitted) == 1, (
        f"expected exactly one queued job, got {len(submitted)}"
    )
    submitted[0]()


def _queued_subscribers(store: ArtifactStore) -> list:
    """Every (table_name, row_id) pair sitting in the store's in-flight
    map -- read before the job runs and pops its own entry."""
    return [pair for subs in store._inflight.values() for pair in subs]


def test_ready_notification_names_the_context_table_not_the_active_table(
    qapp, tmp_path
):
    folder = tmp_path / "media"
    folder.mkdir()
    _solid_png(folder / "a.png", (10, 20, 200))

    controller, _, store = _build_controller(tmp_path)
    controller.load_folder(folder)

    active = controller.get_active_table()
    assert active != "other", (
        "test needs the tile's table to differ from the active table"
    )

    row_id = controller.get_all_row_ids()[0]
    row = controller.get_row(row_id)

    submitted = _capture_submits(store)
    controller.render_column_value(
        "full_path", row["full_path"], 150, "thumbnail",
        {"row_id": row_id, "column_name": "full_path", "table_name": "other"},
    )

    # The request is queued under "other" -- the table the tile is
    # showing -- not under the controller's active table.
    assert _queued_subscribers(store) == [("other", row_id)], (
        f"request queued under {_queued_subscribers(store)}, expected "
        f"[('other', {row_id!r})]"
    )

    # And the ThumbnailsReady the finished job produces names "other".
    ready: list = []
    controller.thumbnails_ready.connect(ready.append)
    _run_the_one_submitted_job(submitted)
    controller._drain_thumbnails()

    assert [payload.table_name for payload in ready] == ["other"], (
        f"ThumbnailsReady named {[p.table_name for p in ready]}, "
        f"expected ['other']"
    )
    assert ready[0].row_ids == (row_id,)


def test_ready_notification_falls_back_to_active_table_without_context(
    qapp, tmp_path
):
    folder = tmp_path / "media"
    folder.mkdir()
    _solid_png(folder / "a.png", (0, 90, 30))

    controller, _, store = _build_controller(tmp_path)
    controller.load_folder(folder)
    active = controller.get_active_table()

    row_id = controller.get_all_row_ids()[0]
    row = controller.get_row(row_id)

    submitted = _capture_submits(store)
    # No "table_name" key in the context: the fallback must still route
    # the request to the active table.
    controller.render_column_value(
        "full_path", row["full_path"], 150, "thumbnail",
        {"row_id": row_id, "column_name": "full_path"},
    )

    assert _queued_subscribers(store) == [(active, row_id)], (
        f"with no table_name in the context the request should queue "
        f"under the active table {active!r}, got "
        f"{_queued_subscribers(store)}"
    )

    ready: list = []
    controller.thumbnails_ready.connect(ready.append)
    _run_the_one_submitted_job(submitted)
    controller._drain_thumbnails()
    assert [payload.table_name for payload in ready] == [active]


# ===========================================================================
# 9. Wiring: a real UI tile puts table_name in the context it builds.
#
#    This uses a LIVE widget, not an AST check. Constructing ImageTile and
#    calling its real render() exercises the exact context dict the tile
#    hands the controller; an AST check would only prove the string
#    "table_name" appears somewhere in the file. The stub controller
#    records the context it is handed. A test that built the real
#    AppController would prove nothing about the UI -- the controller's
#    fallback would fill the table name in either case.
# ===========================================================================

def test_image_tile_passes_its_table_name_in_the_render_context(qapp):
    from ui.tiles.image_tile import ImageTile

    class _RecordingController:
        """Minimal stand-in: ImageTile.render() calls exactly these
        three methods."""

        def __init__(self):
            self.seen_context = None

        def get_active_table(self):
            return "faces"

        def get_row(self, row_id, table_name=None):
            return {"full_path": "C:/proj/x.png"}

        def render_column_value(
            self, column_name, value, size, mode="thumbnail", context=None
        ):
            self.seen_context = context
            return None

    controller = _RecordingController()
    tile = ImageTile("r1", "full_path", controller)
    tile.render(120, 120)

    assert controller.seen_context is not None, (
        "ImageTile.render() did not call render_column_value with a context"
    )
    assert controller.seen_context.get("table_name") == "faces", (
        "ImageTile did not pass its active table name in the render context"
    )
    assert controller.seen_context.get("row_id") == "r1"


# ===========================================================================
# 10. P1.8d-2a: render_column_value resolves the column's type tag from the
#     schema of the table named in the context dict, and falls back to the
#     ACTIVE table's schema when that table has no schema.
#
#     The deciding line in AppController.render_column_value is:
#         if tag is None:
#             tag = self._schema_tag_for(column_name, self._active_table)
#     Named review check 2.
# ===========================================================================

def test_render_context_table_without_schema_falls_back_to_active_schema(
    qapp, tmp_path
):
    folder = tmp_path / "media"
    folder.mkdir()
    _solid_png(folder / "a.png", (10, 20, 200))

    controller, _, store = _build_controller(tmp_path)
    controller.load_folder(folder)

    # "ghost" is a table that was never accepted: schema_for("ghost") is
    # None. full_path IS media_path in the active table's schema, so the
    # fallback line above must still resolve the tag -- and a demand
    # thumbnail request is queued for the cache miss.
    assert controller._dataset.schema_for("ghost") is None

    row_id = controller.get_all_row_ids()[0]
    row = controller.get_row(row_id)

    submitted = _capture_submits(store)
    pixmap = controller.render_column_value(
        "full_path", row["full_path"], 150, "thumbnail",
        {"row_id": row_id, "column_name": "full_path", "table_name": "ghost"},
    )

    assert pixmap is not None and not pixmap.isNull(), (
        "a media cell rendered as an Unknown placeholder -- the tag was "
        "lost instead of falling back to the active table's schema"
    )
    assert len(submitted) == 1, (
        f"expected the media tag to resolve via the active-table fallback "
        f"and queue one request, got {len(submitted)} queued jobs"
    )
    # The request is still attributed to the tile's own (bogus) table --
    # that attribution is the caller's, unchanged by the tag fallback.
    assert _queued_subscribers(store) == [("ghost", row_id)]


def test_render_unknown_column_is_a_placeholder_never_an_exception(qapp, tmp_path):
    """A column no schema names renders exactly as an unregistered column
    did before P1.8d-2a: a placeholder, not a raise."""
    folder = tmp_path / "media"
    folder.mkdir()
    _solid_png(folder / "a.png", (1, 2, 3))

    controller, _, _store = _build_controller(tmp_path)
    controller.load_folder(folder)

    row_id = controller.get_all_row_ids()[0]
    pixmap = controller.render_column_value(
        "no_such_column", "whatever", 150, "thumbnail",
        {"row_id": row_id, "column_name": "no_such_column"},
    )
    assert pixmap is not None and not pixmap.isNull()
