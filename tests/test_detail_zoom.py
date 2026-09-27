"""
tests/test_detail_zoom.py

The detail view's zoom behaviour, written from the settled spec, not from
the implementation:

  (a) a zoom factor set while viewing one picture gives fit-times-factor
      on a different picture of a different pixel size, from the same
      table;
  (b) with no factor, a view opens (or resets to) plain fit; the Fit
      button clears the active table's remembered factor;
  (c) a factor remembered for one table is not applied when a different
      table's picture is shown;
  (d) a programmatic fit or reset never emits the zoom signal -- only a
      wheel zoom does.
  (e) show_pixmap's deferred fit does not fire against a view whose
      underlying C++ object has already been destroyed.

Zoom is view state: it lives in ZoomableImageView and DetailWidget only,
is not saved with the project, and is not read from or written to
Dataset, AppController or TableDisplayState.

Run with:
    python -m pytest tests/test_detail_zoom.py
"""

from __future__ import annotations

import pandas as pd
import pytest
import shiboken6
from PIL import Image

from column_types.renderers import _pil_to_pixmap
from shared_widgets.zoomable_image_view import ZoomableImageView
from ui.detail_widget import DetailWidget

# The controller factory and Qt fixtures live in tests/conftest.py
# (make_controller, qapp, realize_widget).


class _FakeAngleDelta:
    """Stands in for QWheelEvent.angleDelta()'s return value."""

    def __init__(self, y: int) -> None:
        self._y = y

    def y(self) -> int:
        return self._y


class _FakeWheelEvent:
    """A minimal stand-in for QWheelEvent.

    ZoomableImageView.wheelEvent only ever calls event.angleDelta().y(),
    so building a real Qt wheel event (with its own device-specific
    constructor arguments) is unnecessary to drive a zoom step in a test.
    """

    def __init__(self, delta_y: int) -> None:
        self._delta = _FakeAngleDelta(delta_y)

    def angleDelta(self) -> _FakeAngleDelta:
        return self._delta


def _solid_pixmap(width: int, height: int, color=(120, 60, 200)):
    return _pil_to_pixmap(Image.new("RGB", (width, height), color))


def _make_two_table_dataset(make_controller, tmp_path):
    """
    Builds a controller with two media tables:
      'photos'   -- two rows, img_a (600x400) and img_b (150x100).
      'photos_b' -- one row, the same img_b file, so its plain-fit scale
                    is directly comparable to img_b's scale under
                    'photos' without needing a second geometric formula.
    Returns (controller, dataset, row ids by table/name).
    """
    controller, dataset, _ = make_controller(tmp_path)

    img_a = tmp_path / "img_a_600x400.png"
    img_b = tmp_path / "img_b_150x100.png"
    Image.new("RGB", (600, 400), (200, 60, 60)).save(img_a)
    Image.new("RGB", (150, 100), (60, 200, 60)).save(img_b)

    dataset.create_table_from_df(
        "photos", pd.DataFrame({"media": [str(img_a), str(img_b)]})
    )
    assert dataset.schema_for("photos").spec_for("media").type_tag == "media_path", (
        "sanity: 'media' must actually be tagged media_path for this test "
        "to exercise the rule it claims to"
    )
    dataset.create_table_from_df(
        "photos_b", pd.DataFrame({"media": [str(img_b)]})
    )

    photos_ids = dataset.get_table("photos")["row_id"].astype(str).tolist()
    photos_b_ids = dataset.get_table("photos_b")["row_id"].astype(str).tolist()

    return controller, dataset, {
        "photos_a": photos_ids[0],
        "photos_b_row": photos_ids[1],
        "photos_b_table_row": photos_b_ids[0],
    }


def _show_and_settle(detail: DetailWidget, row_ids: list[str], qapp) -> None:
    detail.show_rows(row_ids)
    qapp.processEvents()
    qapp.processEvents()


# ---------------------------------------------------------------------------
# (d) A programmatic fit or reset never emits the zoom signal.
# ---------------------------------------------------------------------------

def test_programmatic_fit_and_reset_do_not_emit_zoom_signal(qapp):
    view = ZoomableImageView()
    view.resize(400, 300)
    view.show()
    qapp.processEvents()
    qapp.processEvents()

    events: list[float] = []
    view.zoom_factor_changed.connect(events.append)

    view.show_pixmap(_solid_pixmap(600, 400))
    qapp.processEvents()
    qapp.processEvents()
    assert events == [], (
        "show_pixmap's deferred fit must not emit the zoom signal"
    )

    view.fit_to_view()
    view.reset_view()
    assert events == [], (
        "a direct fit_to_view()/reset_view() call must not emit the zoom "
        "signal"
    )

    view.wheelEvent(_FakeWheelEvent(120))
    assert len(events) == 1, (
        "a wheel zoom step is the one case that must emit the signal"
    )


# ---------------------------------------------------------------------------
# (e) show_pixmap's deferred fit is tied to the view's own lifetime.
# ---------------------------------------------------------------------------

def test_deferred_fit_does_not_fire_against_a_deleted_view(qapp):
    view = ZoomableImageView()
    view.resize(400, 300)
    view.show()
    qapp.processEvents()
    qapp.processEvents()

    view.show_pixmap(_solid_pixmap(600, 400))

    # Destroy the C++ object before either of show_pixmap's two deferred
    # turns has run. deleteLater() alone does not reliably reach actual
    # destruction within a handful of processEvents() calls in this
    # offscreen test setup, so force it directly.
    shiboken6.delete(view)

    # Without a context object, Qt still invokes the pending callable
    # against the deleted view and PySide raises RuntimeError
    # ("Internal C++ object already deleted") straight out of
    # processEvents() -- catch that here rather than let it escape and
    # fail the test with a less legible traceback.
    errors: list[BaseException] = []
    for _ in range(6):
        try:
            qapp.processEvents()
        except Exception as exc:  # pragma: no cover -- only hit pre-fix
            errors.append(exc)

    assert errors == [], (
        f"a deferred fit fired against the already-deleted view: {errors!r}"
    )


# ---------------------------------------------------------------------------
# (b), first half -- reset with no factor is plain fit.
# ---------------------------------------------------------------------------

def test_reset_with_no_factor_is_plain_fit(qapp):
    plain = ZoomableImageView()
    plain.resize(400, 300)
    plain.show()
    qapp.processEvents()
    qapp.processEvents()
    plain.show_pixmap(_solid_pixmap(600, 400))
    qapp.processEvents()
    qapp.processEvents()
    plain_scale = plain.transform().m11()

    reset_only = ZoomableImageView()
    reset_only.resize(400, 300)
    reset_only.show()
    qapp.processEvents()
    qapp.processEvents()
    reset_only.show_pixmap(_solid_pixmap(600, 400))
    qapp.processEvents()
    qapp.processEvents()
    reset_only.set_preferred_factor(None)
    reset_only.reset_view()

    assert reset_only.transform().m11() == pytest.approx(plain_scale)


# ---------------------------------------------------------------------------
# (a) and (c) -- factor carries across pictures within a table, and does
# not cross to another table.
# ---------------------------------------------------------------------------

def test_zoom_factor_carries_to_a_different_sized_picture_in_the_same_table_only(
    make_controller, realize_widget, qapp, tmp_path
):
    controller, dataset, rows = _make_two_table_dataset(make_controller, tmp_path)

    controller.set_active_table("photos")
    detail = DetailWidget(controller)
    realize_widget(detail, width=500, height=400)

    # Baseline: img_b on its own, before any factor exists for 'photos' --
    # plain fit.
    _show_and_settle(detail, [rows["photos_b_row"]], qapp)
    assert "photos" not in detail._table_zoom_factors
    baseline_view = detail._current_zoom_views[0]
    plain_fit_scale = baseline_view.transform().m11()

    # img_a, zoomed in twice with the wheel -- capture the factor the
    # signal reports.
    _show_and_settle(detail, [rows["photos_a"]], qapp)
    view_a = detail._current_zoom_views[0]
    factors: list[float] = []
    view_a.zoom_factor_changed.connect(factors.append)
    view_a.wheelEvent(_FakeWheelEvent(120))
    view_a.wheelEvent(_FakeWheelEvent(120))
    assert len(factors) == 2
    captured_factor = factors[-1]
    assert captured_factor != pytest.approx(1.0)
    assert detail._table_zoom_factors["photos"] == pytest.approx(captured_factor)

    # (a) Switch to img_b, a different pixel size, same table -- must
    # open at fit(img_b) x captured_factor, clamped.
    _show_and_settle(detail, [rows["photos_b_row"]], qapp)
    view_b = detail._current_zoom_views[0]
    expected = plain_fit_scale * captured_factor
    expected = max(
        ZoomableImageView._MIN_SCALE, min(ZoomableImageView._MAX_SCALE, expected)
    )
    assert view_b.transform().m11() == pytest.approx(expected, rel=1e-3)

    # (c) A different table showing the SAME image (img_b) must open at
    # plain fit, not fit x 'photos'' factor -- its own dict entry is
    # absent, and the resulting scale matches the untouched baseline
    # rather than the factor-scaled one.
    controller.set_active_table("photos_b")
    _show_and_settle(detail, [rows["photos_b_table_row"]], qapp)
    assert "photos_b" not in detail._table_zoom_factors
    view_c = detail._current_zoom_views[0]
    assert view_c.transform().m11() == pytest.approx(plain_fit_scale, rel=1e-3)
    # The other table's remembered factor must still be intact.
    assert detail._table_zoom_factors["photos"] == pytest.approx(captured_factor)


# ---------------------------------------------------------------------------
# (b), second half -- the Fit button clears the active table's factor.
# ---------------------------------------------------------------------------

def test_fit_button_resets_view_and_forgets_the_tables_factor(
    make_controller, realize_widget, qapp, tmp_path
):
    controller, dataset, rows = _make_two_table_dataset(make_controller, tmp_path)

    controller.set_active_table("photos")
    detail = DetailWidget(controller)
    realize_widget(detail, width=500, height=400)

    _show_and_settle(detail, [rows["photos_b_row"]], qapp)
    plain_fit_scale = detail._current_zoom_views[0].transform().m11()

    _show_and_settle(detail, [rows["photos_a"]], qapp)
    view_a = detail._current_zoom_views[0]
    assert detail._fit_btn.isEnabled(), (
        "the Fit button must be enabled once a ZoomableImageView is shown"
    )
    view_a.wheelEvent(_FakeWheelEvent(120))
    assert "photos" in detail._table_zoom_factors

    _show_and_settle(detail, [rows["photos_b_row"]], qapp)
    zoomed_scale = detail._current_zoom_views[0].transform().m11()
    assert zoomed_scale != pytest.approx(plain_fit_scale), (
        "sanity: the remembered factor must actually change the scale, "
        "otherwise clicking Fit proves nothing"
    )

    events: list[float] = []
    detail._current_zoom_views[0].zoom_factor_changed.connect(events.append)
    detail._fit_btn.click()
    qapp.processEvents()

    assert "photos" not in detail._table_zoom_factors
    assert events == [], "the Fit button's reset must not emit the zoom signal"
    assert detail._current_zoom_views[0].transform().m11() == pytest.approx(
        plain_fit_scale, rel=1e-3
    )

    # The next picture from this table opens at plain fit again.
    _show_and_settle(detail, [rows["photos_a"]], qapp)
    assert "photos" not in detail._table_zoom_factors


def test_fit_button_disabled_with_no_zoomable_view_shown(
    make_controller, realize_widget, qapp, tmp_path
):
    controller, dataset, _ = make_controller(tmp_path)
    detail = DetailWidget(controller)
    realize_widget(detail, width=500, height=400)

    assert not detail._fit_btn.isEnabled()

    detail.show_result({"operator_name": "no image result"})
    qapp.processEvents()
    assert not detail._fit_btn.isEnabled(), (
        "a result with no artifact_path shows no ZoomableImageView"
    )


# ---------------------------------------------------------------------------
# Multi-item comparison panels also honour the table's remembered factor.
# ---------------------------------------------------------------------------

def test_multi_item_panels_open_at_the_tables_remembered_factor(
    make_controller, realize_widget, qapp, tmp_path
):
    controller, dataset, rows = _make_two_table_dataset(make_controller, tmp_path)

    controller.set_active_table("photos")
    detail = DetailWidget(controller)
    realize_widget(detail, width=700, height=400)

    _show_and_settle(detail, [rows["photos_a"]], qapp)
    view_a = detail._current_zoom_views[0]
    factors: list[float] = []
    view_a.zoom_factor_changed.connect(factors.append)
    view_a.wheelEvent(_FakeWheelEvent(120))
    captured_factor = factors[-1]

    _show_and_settle(detail, [rows["photos_a"], rows["photos_b_row"]], qapp)
    assert len(detail._current_zoom_views) == 2
    for view in detail._current_zoom_views:
        assert view._preferred_factor == pytest.approx(captured_factor)
