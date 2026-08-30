"""
tests/test_gallery_seam.py

P0.5a -- the gallery seam.

P0.4 gave the controller ownership of the visible row order. A gallery
is handed an absolute half-open range [start, stop) into that flat
order via set_range(), fetches the ids it paints with
get_row_ids_in_range(), and reports the window it has actually mounted
back as displayed_range_changed(start, stop, result_id) in ABSOLUTE
flat-order indices.

P0.5 builds demand-driven thumbnail rendering directly on top of that
reported range -- it prioritises and cancels work by what the gallery
says is on screen. Nothing drove a real GalleryWidget and checked that
report, so a dropped group offset or an off-by-one would have fed P0.5
the wrong rows with every existing test still green. These tests close
that gap.

They are written to assert properties that hold whatever tile size or
viewport height Qt hands us on a given machine -- never a specific tile
count, mounted-row count, pixel offset or scrollbar value. Where a test
needs to scroll it drives the real vertical scrollbar (to a fraction of
its own maximum), it does not guess a pixel position.

Run with:
    python -m pytest tests/test_gallery_seam.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

from PySide6.QtCore import QEvent, QPoint, QRect
from PySide6.QtWidgets import QScrollArea, QGridLayout

from models.dataset import Dataset
from models.query_engine import QueryEngine
from artifacts.artifact_store import ArtifactStore
from column_types.registry import ColumnTypeRegistry
from operators.operator_registry import OperatorRegistry
from controller import AppController
from ui.gallery_widget import GalleryWidget, TileWidget
from ui.main_window import MainWindow

# The controller factory is the make_controller fixture in
# tests/conftest.py -- it used to be copied into this file and
# tests/test_visible_row_order.py.


def _any_group_with_offset(layout):
    """The first group section whose slice does not start at 0."""
    return next(s for s in layout.groups if s.start > 0)


def _largest_group_with_offset(layout):
    """The biggest group section whose slice does not start at 0 -- used
    by the scroll tests so there is enough content to actually scroll."""
    return max(
        (s for s in layout.groups if s.start > 0),
        key=lambda s: s.stop - s.start,
    )


# ---------------------------------------------------------------------------
# Reading what a real gallery painted, without touching its private state
# ---------------------------------------------------------------------------

def _painted_tiles_in_flat_order(gallery):
    """
    Returns the TileWidgets currently mounted in the gallery, ordered
    the way they sit in the grid (row-major) -- which is flat order,
    because _update_visible_tiles() places local index `idx` at grid
    cell (idx // cols, idx % cols).

    Reads only the QGridLayout (located via findChild, not by attribute
    name) and TileWidget.get_row_ids(), both public.
    """
    grid = gallery.findChild(QGridLayout)
    entries = []
    for i in range(grid.count()):
        widget = grid.itemAt(i).widget()
        if isinstance(widget, TileWidget):
            row, col, _rspan, _cspan = grid.getItemPosition(i)
            entries.append((row, col, widget))
    entries.sort(key=lambda e: (e[0], e[1]))
    return [w for _r, _c, w in entries]


def _painted_ids(gallery):
    ids = []
    for tile in _painted_tiles_in_flat_order(gallery):
        ids.extend(tile.get_row_ids())
    return ids


def _tile_intersects_viewport(tile, viewport):
    """True if any part of `tile` currently falls inside `viewport`."""
    top_left = tile.mapTo(viewport, QPoint(0, 0))
    return viewport.rect().intersects(QRect(top_left, tile.size()))


def _spy(gallery):
    """A plain list that collects (start, stop, result_id) reports."""
    reports: list[tuple[int, int, str]] = []
    gallery.displayed_range_changed.connect(
        lambda start, stop, rid: reports.append((start, stop, rid))
    )
    return reports


def _fresh_gallery(controller, reports_first=True):
    """
    A GalleryWidget wired to `controller`, with one visual column
    chosen so it actually mounts tiles, and a report spy attached
    before any range is set.
    """
    gallery = GalleryWidget(controller)
    reports = _spy(gallery) if reports_first else None
    gallery.set_visible_columns(["full_path"])
    return gallery, reports


# ===========================================================================
# PART 2 -- the reported range
# ===========================================================================

def test_flat_scrolled_to_top_reports_start_zero(make_controller, realize_widget, tmp_path):
    """1. Flat mode, scrolled to top: reported start is 0."""
    controller, _, _ = make_controller(tmp_path)
    controller.set_filters([])
    layout = controller.get_result_layout()

    gallery, reports = _fresh_gallery(controller)
    gallery.set_range(0, layout.total, layout.result_id)
    realize_widget(gallery, width=520, height=360)

    assert reports
    assert reports[-1][0] == 0


def test_grouped_scrolled_to_top_reports_absolute_start(make_controller, realize_widget, tmp_path):
    """
    2. Grouped mode, scrolled to top: a gallery given set_range(S, E)
    with S > 0 reports start == S, not 0.

    This is the central test of the item. A gallery that reported
    group-local indices instead of absolute ones would pass every
    other test in the repo and silently feed P0.5 the wrong rows.
    """
    controller, _, _ = make_controller(tmp_path, merge_csv=True)
    controller.set_group_by("condition")
    layout = controller.get_result_layout()
    section = _any_group_with_offset(layout)
    assert section.start > 0  # precondition for the test to mean anything

    gallery, reports = _fresh_gallery(controller)
    gallery.set_range(section.start, section.stop, layout.result_id)
    realize_widget(gallery, width=520, height=360)

    assert reports
    assert reports[-1][0] == section.start
    assert reports[-1][0] != 0


def test_scrolled_to_bottom_reports_slice_stop(make_controller, realize_widget, qapp, tmp_path):
    """
    3. Scrolled to the bottom: the reported stop equals the slice's
    stop E. Driven by setting the vertical scrollbar to its maximum.
    """
    controller, _, _ = make_controller(tmp_path)
    controller.set_filters([])
    layout = controller.get_result_layout()
    end = layout.total

    gallery, reports = _fresh_gallery(controller)
    gallery.set_range(0, end, layout.result_id)
    # A deliberately small viewport so 20 rows overflow and there is a
    # real bottom to scroll to.
    realize_widget(gallery, width=360, height=260)

    scrollbar = gallery.findChild(QScrollArea).verticalScrollBar()
    scrollbar.setValue(scrollbar.maximum())
    qapp.processEvents()

    assert reports[-1][1] == end


def test_report_is_always_inside_the_slice(make_controller, realize_widget, qapp, tmp_path):
    """
    4. For several scroll positions, S <= start <= stop <= E.
    """
    controller, _, _ = make_controller(tmp_path, merge_csv=True)
    controller.set_group_by("condition")
    layout = controller.get_result_layout()
    section = _largest_group_with_offset(layout)
    start_s, end_e = section.start, section.stop

    gallery, reports = _fresh_gallery(controller)
    gallery.set_range(start_s, end_e, layout.result_id)
    realize_widget(gallery, width=360, height=240)

    scrollbar = gallery.findChild(QScrollArea).verticalScrollBar()
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        scrollbar.setValue(int(scrollbar.maximum() * fraction))
        qapp.processEvents()
        start, stop, _rid = reports[-1]
        assert start_s <= start <= stop <= end_e


def test_painted_ids_match_reported_range_flat(make_controller, realize_widget, qapp, tmp_path):
    """
    5. (flat) The row ids held by the mounted tiles equal
    controller.get_visible_row_ids()[a:b] for the reported [a, b).
    No geometry involved.
    """
    controller, _, _ = make_controller(tmp_path)
    controller.set_filters([])
    layout = controller.get_result_layout()

    gallery, reports = _fresh_gallery(controller)
    gallery.set_range(0, layout.total, layout.result_id)
    realize_widget(gallery, width=360, height=260)

    scrollbar = gallery.findChild(QScrollArea).verticalScrollBar()
    for fraction in (0.0, 0.5, 1.0):
        scrollbar.setValue(int(scrollbar.maximum() * fraction))
        qapp.processEvents()
        start, stop, _rid = reports[-1]
        assert _painted_ids(gallery) == controller.get_visible_row_ids()[start:stop]


def test_painted_ids_match_reported_range_grouped(make_controller, realize_widget, qapp, tmp_path):
    """
    5. (grouped, non-zero slice start) Same assertion as above but for
    a group gallery whose slice starts partway through the flat order.
    If the gallery drops the group offset anywhere -- in the fetch or
    in the report -- the painted ids would be
    get_visible_row_ids()[0:count] and this fails.
    """
    controller, _, _ = make_controller(tmp_path, merge_csv=True)
    controller.set_group_by("condition")
    layout = controller.get_result_layout()
    section = _largest_group_with_offset(layout)
    assert section.start > 0

    gallery, reports = _fresh_gallery(controller)
    gallery.set_range(section.start, section.stop, layout.result_id)
    realize_widget(gallery, width=360, height=240)

    scrollbar = gallery.findChild(QScrollArea).verticalScrollBar()
    for fraction in (0.0, 0.5, 1.0):
        scrollbar.setValue(int(scrollbar.maximum() * fraction))
        qapp.processEvents()
        start, stop, _rid = reports[-1]
        assert start >= section.start
        assert stop <= section.stop
        assert _painted_ids(gallery) == controller.get_visible_row_ids()[start:stop]


def test_reported_window_is_a_superset_of_the_strictly_visible_tiles(make_controller, 
    realize_widget, qapp, tmp_path
):
    """
    6. GalleryWidget._BUFFER_ROWS keeps one tile-row mounted above and
    below the strictly-visible band, so the mounted/reported window is
    a strict superset of what the eye sees. Documented deliberately so
    P0.5 does not stack its own prefetch margin on top of a margin it
    did not know was there. Asserts the relationship, not the number.
    """
    controller, _, _ = make_controller(tmp_path)
    controller.set_filters([])
    layout = controller.get_result_layout()

    gallery, reports = _fresh_gallery(controller)
    gallery.set_range(0, layout.total, layout.result_id)
    realize_widget(gallery, width=360, height=240)

    scrollbar = gallery.findChild(QScrollArea).verticalScrollBar()
    assert scrollbar.maximum() > 0, "need vertical overflow for this test"
    # A middle position: neither the top nor the bottom clamp applies,
    # so a buffer row is mounted on each side.
    scrollbar.setValue(scrollbar.maximum() // 2)
    qapp.processEvents()

    tiles = _painted_tiles_in_flat_order(gallery)
    viewport = gallery.findChild(QScrollArea).viewport()
    visible_positions = [
        i for i, tile in enumerate(tiles)
        if _tile_intersects_viewport(tile, viewport)
    ]
    assert visible_positions, "expected some tiles inside the viewport"
    # Mounted tiles exist both before the first strictly-visible tile
    # and after the last one.
    assert visible_positions[0] > 0
    assert visible_positions[-1] < len(tiles) - 1

    # And the report covers exactly the mounted set.
    start, stop, _rid = reports[-1]
    assert stop - start == len(_painted_ids(gallery))


def test_superseded_result_id_is_not_reported_under_the_new_one(make_controller, 
    realize_widget, qapp, tmp_path
):
    """
    7. After set_range with a new result_id, the next report carries
    the new id -- the gallery stamps its own reports.
    """
    controller, _, _ = make_controller(tmp_path)
    controller.set_filters([])
    layout_a = controller.get_result_layout()

    gallery, reports = _fresh_gallery(controller)
    gallery.set_range(0, layout_a.total, layout_a.result_id)
    realize_widget(gallery, width=360, height=240)

    controller.set_filters([])  # a fresh query -> a fresh result_id
    layout_b = controller.get_result_layout()
    assert layout_b.result_id != layout_a.result_id

    reports.clear()
    gallery.set_range(0, layout_b.total, layout_b.result_id)
    qapp.processEvents()

    assert reports
    assert all(rid == layout_b.result_id for _s, _e, rid in reports)


# ===========================================================================
# PART 3, DEFECT A -- flat gallery honours a columns change made while grouped
# ===========================================================================

def _sole_flat_gallery(window, qapp):
    """
    The flat main gallery. After grouping is cleared the per-group
    section widgets are deleteLater()'d; force the deferred deletions
    through so exactly one GalleryWidget -- the flat one -- remains.
    """
    qapp.processEvents()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    galleries = window.findChildren(GalleryWidget)
    assert len(galleries) == 1, f"expected one flat gallery, found {len(galleries)}"
    return galleries[0]


def test_defect_a_flat_gallery_honours_empty_columns_after_ungrouping(make_controller, 
    realize_widget, qapp, tmp_path
):
    """
    Turn grouping on, uncheck every visible column, turn grouping off.
    The flat gallery must show the "no visual column" placeholder --
    not fall back to its stale local choice and paint full_path tiles.
    """
    controller, _, _ = make_controller(tmp_path, merge_csv=True)
    window = MainWindow(controller)
    realize_widget(window, width=1100, height=800)

    controller.set_filters([])  # an initial flat result
    controller.set_group_by("condition")  # -> grouped view
    qapp.processEvents()

    # The researcher unchecks every column while grouping is on. This is
    # exactly the slot the Columns combo's selection_changed is wired to.
    window._apply_visible_columns([])

    controller.set_group_by(None)  # back to the flat view
    gallery = _sole_flat_gallery(window, qapp)

    assert gallery.findChildren(TileWidget) == [], (
        "flat gallery painted tiles after the researcher chose zero "
        "visible columns while grouped -- it kept its stale local "
        "_visible_cols instead of the controller's preference"
    )


def test_defect_a_no_preference_ungrouping_does_not_hand_gallery_empty_list(make_controller, 
    realize_widget, qapp, tmp_path
):
    """
    check #4: with NO visible-columns preference, clearing grouping must
    not hand the flat gallery an empty list. If it did, the gallery
    would show the placeholder; instead it must fall back to full_path
    and paint tiles.
    """
    controller, _, _ = make_controller(tmp_path, merge_csv=True)
    window = MainWindow(controller)
    realize_widget(window, width=1100, height=800)

    assert controller.has_visible_columns_preference() is False

    controller.set_filters([])
    controller.set_group_by("condition")
    qapp.processEvents()
    controller.set_group_by(None)
    gallery = _sole_flat_gallery(window, qapp)

    assert controller.has_visible_columns_preference() is False
    assert gallery.findChildren(TileWidget), (
        "flat gallery showed the placeholder after ungrouping even "
        "though the researcher set no preference -- it was handed [] "
        "where it should have been left to fall back to full_path"
    )


def test_defect_a_preference_reset_while_grouped_reaches_the_flat_gallery(make_controller, 
    realize_widget, qapp, tmp_path
):
    """
    The other direction of Defect A: the flat gallery has an explicit
    "no columns" choice, then the preference is reset (as a table switch
    or project reset does) while grouping is on. clear_visible_columns_
    preference() is emitted with self._galleries holding only the group
    galleries, so the flat gallery never hears it. On ungrouping it must
    still be resynced -- back to the full_path fallback, not stuck on
    the stale empty choice.
    """
    controller, _, _ = make_controller(tmp_path, merge_csv=True)
    window = MainWindow(controller)
    realize_widget(window, width=1100, height=800)

    controller.set_filters([])
    window._apply_visible_columns([])                 # explicit "no columns"
    controller.set_group_by("condition")             # -> grouped
    qapp.processEvents()
    controller.clear_visible_columns_preference()     # reset while grouped
    controller.set_group_by(None)                     # -> flat
    gallery = _sole_flat_gallery(window, qapp)

    assert controller.has_visible_columns_preference() is False
    assert gallery.findChildren(TileWidget), (
        "flat gallery stayed on its stale empty choice after the "
        "preference was reset while grouped"
    )


# ===========================================================================
# PART 3, DEFECT B -- the no-visual-column relayout reports a zero-width range
# ===========================================================================

def test_defect_b_no_visual_column_reports_zero_width_range_flat(make_controller, 
    realize_widget, tmp_path
):
    """
    A gallery whose visible-column choice resolves to no visual column
    reports a zero-width range at its own slice start (0 in flat mode),
    the same thing the empty case in _update_visible_tiles does.
    """
    controller, _, _ = make_controller(tmp_path)
    controller.set_filters([])
    layout = controller.get_result_layout()

    gallery = GalleryWidget(controller)
    reports = _spy(gallery)
    gallery.set_visible_columns([])  # explicit "no visual column"
    gallery.set_range(0, layout.total, layout.result_id)
    realize_widget(gallery, width=420, height=300)

    assert reports, "no-visual-column gallery never reported its range"
    assert reports[-1] == (0, 0, layout.result_id)


def test_defect_b_no_visual_column_reports_zero_width_range_grouped(make_controller, 
    realize_widget, tmp_path
):
    """
    Same, but for a group gallery whose slice starts at S > 0: the
    zero-width range is reported at S, not at 0.
    """
    controller, _, _ = make_controller(tmp_path, merge_csv=True)
    controller.set_group_by("condition")
    layout = controller.get_result_layout()
    section = _any_group_with_offset(layout)
    assert section.start > 0

    gallery = GalleryWidget(controller)
    reports = _spy(gallery)
    gallery.set_visible_columns([])
    gallery.set_range(section.start, section.stop, layout.result_id)
    realize_widget(gallery, width=420, height=300)

    assert reports, "no-visual-column gallery never reported its range"
    assert reports[-1] == (section.start, section.start, layout.result_id)


# ===========================================================================
# PART 4 -- guardrail for P0.5b-1 (ArtifactStore key collision).
# ===========================================================================
#
# Was a known defect ("A row with several media columns shares one cached
# image"): ArtifactStore keyed its index, memory cache and filenames on
# (row_id, artifact_type), so a row with full_path and avatar_path showed
# the same picture in both tiles. P0.5b-1 keys derived artifacts by media
# address instead (docs/media_architecture.md 4.5), and this test -- which
# goes through PUBLIC controller API only (render_column_value() and the
# thumbnail-request path reached by load_folder()) -- now passes.


def _solid_png(path: Path, colour: tuple[int, int, int]) -> None:
    from PIL import Image

    Image.new("RGB", (240, 240), colour).save(path)


# P0.5b-1 landed: derived artifacts are keyed by media address, not by
# (row_id, artifact_type) (docs/media_architecture.md 4.5), so the two
# media columns on one row no longer collide. This test was
# xfail(strict, raises=AssertionError) until then; the marker is gone.
# Setup checks still raise RuntimeError rather than AssertionError so the
# only AssertionError the test can produce is the real colour comparison.
#
# NOTE (see the work-item handoff): this test passes EVEN IF the cache is
# broken, because avatar_path has no cached artifact and _render_image
# falls back to Image.open(blue.png). It proves the fallback, not the
# keying. test_artifact_identity.py::test_second_media_column_gets_its_
# own_cached_artifact is what actually guards the fix.
def test_two_media_columns_on_one_row_render_different_pictures(qapp, tmp_path):
    # qapp: QPixmap (built below via render_column_value) requires a
    # live QApplication, and Qt aborts the whole process with qFatal
    # rather than raising if none exists. Depend on the fixture so the
    # test builds one itself instead of relying on some earlier test in
    # the run having created it.
    # A one-image project whose image is solid red.
    folder = tmp_path / "media"
    folder.mkdir()
    red_path = folder / "red.png"
    _solid_png(red_path, (255, 0, 0))

    # A distinct solid-blue image that a merged CSV column points at.
    blue_path = tmp_path / "blue.png"
    _solid_png(blue_path, (0, 0, 255))

    store = ArtifactStore(tmp_path / "artifacts")
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)
    dataset = Dataset()
    dataset.set_registry(registry)
    op_registry = OperatorRegistry()
    controller = AppController(dataset, QueryEngine(), store, registry, op_registry)

    # Wait for thumbnail generation deterministically: chain the
    # store's ready callback and block on an Event, not a fixed sleep.
    ready = threading.Event()
    previous_cb = store.on_thumbnail_ready

    def _on_ready(table_name, row_id):
        if previous_cb is not None:
            previous_cb(table_name, row_id)
        ready.set()

    store.on_thumbnail_ready = _on_ready

    # Public thumbnail-request path: load_folder() queues a thumbnail
    # for full_path (the red image).
    controller.load_folder(folder)
    # Setup check -> RuntimeError, not AssertionError: see the marker.
    if not ready.wait(timeout=30):
        raise RuntimeError(
            "thumbnail generation for the loaded image did not finish "
            "within 30s"
        )

    # Merge a second media column, avatar_path, pointing at the blue
    # image. Its .png values make ColumnTypeRegistry infer 'media_path',
    # so it renders through the same path as full_path.
    csv_path = tmp_path / "avatars.csv"
    csv_path.write_text(f"file_name,avatar_path\nred.png,{blue_path.as_posix()}\n")
    dataset.confirm_merge(dataset.merge_csv(csv_path, join_on="file_name"))

    row_id = controller.get_all_row_ids()[0]
    row = controller.get_row(row_id)

    def _render(column_name):
        pixmap = controller.render_column_value(
            column_name,
            row[column_name],
            150,
            "thumbnail",
            {"row_id": row_id, "column_name": column_name},
        )
        # Setup check -> RuntimeError, not AssertionError: see the marker.
        if pixmap is None or pixmap.isNull():
            raise RuntimeError(
                f"render_column_value returned nothing for {column_name}"
            )
        image = pixmap.toImage()
        return image.pixelColor(image.width() // 2, image.height() // 2)

    full_path_colour = _render("full_path")
    avatar_path_colour = _render("avatar_path")

    assert full_path_colour != avatar_path_colour, (
        "full_path and avatar_path rendered the same picture for one row -- "
        "the two media columns collided. Their identities differ only by "
        "canonical media address, so this is the end-to-end check that "
        "render_column_value carries the right address per column. The "
        "cache-level guard is "
        "test_artifact_identity.py::test_second_media_column_gets_its_own_"
        "cached_artifact."
    )
