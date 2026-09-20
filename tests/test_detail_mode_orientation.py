"""
tests/test_detail_mode_orientation.py

P1.2c-2 review round 4, item 4: column_types/renderers.py's detail-mode
_render_image() switched from a plain QPixmap(path) construction to
Qt's QImageReader with setAutoTransform(True), so a photo's EXIF
orientation is applied the same way analysis already gets it from the
resolver (decision 6, docs/media_architecture.md section 3.6). No test
exercised this behaviour before now.

This calls the same render() function DetailWidget calls
(column_types/renderers.py's make_media_path_renderer, "detail" mode),
without ever showing a widget -- render()'s detail-mode path
(_render_image) does not require the returned ZoomableImageView to be
shown to run correctly; tests/test_demand_driven_display.py's
test_detail_mode_still_opens_the_source already calls it this way,
checking only that a QWidget comes back. This test goes one step
further and reads the actual QPixmap that was displayed back off the
widget (ZoomableImageView.current_pixmap()) to check its real,
orientation-corrected dimensions -- real behaviour, not which class
name appears in the source.

Run with:
    python -m pytest tests/test_detail_mode_orientation.py
"""

from __future__ import annotations

from PIL import Image

from column_types.renderers import make_media_path_renderer


def test_detail_mode_applies_exif_orientation(qapp, tmp_path):
    # A non-square fixture: orientation 6 (EXIF "rotate 90 CW to display
    # upright") swaps width and height, so a dimension swap alone -- no
    # pixel inspection needed -- proves the tag was actually applied.
    width, height = 40, 24
    image = Image.new("RGB", (width, height), (200, 100, 50))
    jpeg_path = tmp_path / "photo.jpg"
    exif = image.getexif()
    exif[274] = 6  # Orientation tag: a common real-camera value.
    image.save(jpeg_path, format="JPEG", quality=100, exif=exif)

    render = make_media_path_renderer(None)
    widget = render(
        str(jpeg_path), 600, "detail",
        {
            "row_id": "r1", "column_name": "full_path",
            "source_path": str(jpeg_path),
        },
    )

    assert widget is not None, "detail-mode render() returned None"
    pixmap = widget.current_pixmap()
    assert pixmap is not None, "no pixmap was shown on the returned widget"

    # The stored file is width x height (40 x 24); orientation 6's
    # upright display is height x width (24 x 40). Would still pass if
    # EXIF orientation were ignored, as a plain QPixmap(path) does? No --
    # the pixmap would then read back as the file's stored 40 x 24.
    assert (pixmap.width(), pixmap.height()) == (height, width), (
        f"expected the EXIF-upright {(height, width)}, got "
        f"{(pixmap.width(), pixmap.height())}"
    )
