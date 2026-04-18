"""
tests/test_renderer.py

Tests the column type renderers by rendering sample values and saving
the output as PNG files. Run this to verify your render functions
produce correct visual output without needing the full application.

Usage:
    python tests/test_renderer.py

Output images are saved to tests/renderer_output/
Student A should inspect these images to verify their renderers work.

What this tests:
    - media_path renderer (thumbnail mode) for images
    - media_path renderer (thumbnail mode) for videos (if any exist)
    - media_path renderer (detail mode) — checks a QWidget is returned
    - numeric renderer (thumbnail and detail modes)
    - text renderer (thumbnail and detail modes)
    - boolean_flag renderer
    - placeholder for unknown column
    - placeholder for None value
"""

import sys
from pathlib import Path

# Add project root to path.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tempfile
from PySide6.QtWidgets import QApplication, QWidget

app = QApplication(sys.argv)

from artifacts.artifact_store import ArtifactStore
from column_types.registry import ColumnTypeRegistry

# Set up registry with a temporary artifact store.
store    = ArtifactStore(Path(tempfile.gettempdir()) / "gelem_test_artifacts")
registry = ColumnTypeRegistry()
registry.setup_defaults(store)

# Output folder for rendered thumbnail images.
output_dir = project_root / "tests" / "renderer_output"
output_dir.mkdir(exist_ok=True)

SIZE = 200  # Render at 200x200 pixels for easy inspection.

_passed = 0
_failed = 0


def test_thumbnail(column_name: str, value, filename: str) -> None:
    """
    Renders a value in thumbnail mode and saves it as a PNG file.
    A QPixmap is expected.
    """
    global _passed, _failed
    pixmap = registry.render(column_name, value, SIZE, mode="thumbnail")
    if pixmap is not None:
        path = output_dir / filename
        pixmap.save(str(path), "PNG")
        print(f"  PASS  {filename}")
        _passed += 1
    else:
        print(f"  FAIL  {filename} — render returned None")
        _failed += 1


def test_detail(column_name: str, value, label: str) -> None:
    """
    Renders a value in detail mode and checks that a QWidget is returned.
    Does not save to disk (widgets can't be saved as PNG directly).
    """
    global _passed, _failed
    widget = registry.render(column_name, value, SIZE, mode="detail")
    if isinstance(widget, QWidget):
        print(f"  PASS  {label} (detail mode — QWidget returned)")
        _passed += 1
    else:
        print(f"  FAIL  {label} — detail render returned {type(widget)}, expected QWidget")
        _failed += 1


print(f"Rendering thumbnails to {output_dir}")
print()

# ── media_path renderer — images ──────────────────────────────────────────

print("── media_path (images) ─────────────────────────────────────────")

test_images = list((project_root / "test_images").glob("*.jpg"))
test_images += list((project_root / "test_images").glob("*.png"))

if test_images:
    # Register full_path as media_path so the renderer is found.
    registry.register_by_tag("full_path", "media_path")
    test_thumbnail("full_path", str(test_images[0]), "test_image_thumbnail.png")
    test_detail("full_path", str(test_images[0]), "image detail mode")
else:
    print("  SKIP  image tests — no .jpg/.png files found in test_images/")

# ── media_path renderer — videos ──────────────────────────────────────────

print()
print("── media_path (videos) ─────────────────────────────────────────")

test_videos = list((project_root / "test_images").glob("*.mp4"))
test_videos += list((project_root / "test_images").glob("*.mov"))

if test_videos:
    test_thumbnail("full_path", str(test_videos[0]), "test_video_thumbnail.png")
    test_detail("full_path", str(test_videos[0]), "video detail mode")
else:
    print("  SKIP  video tests — no .mp4/.mov files found in test_images/")
    print("        Add a short video to test_images/ to test video rendering.")

# ── numeric renderer ──────────────────────────────────────────────────────

print()
print("── numeric ─────────────────────────────────────────────────────")

registry.register_by_tag("some_number",  "numeric")
registry.register_by_tag("some_integer", "numeric")

test_thumbnail("some_number",  3.14159, "test_numeric_float.png")
test_thumbnail("some_integer", 42,      "test_numeric_int.png")
test_detail("some_number", 3.14159, "numeric detail mode")

# ── text renderer ─────────────────────────────────────────────────────────

print()
print("── text ────────────────────────────────────────────────────────")

registry.register_by_tag("condition",  "text")
registry.register_by_tag("long_label", "text")

test_thumbnail("condition",  "positive", "test_text_short.png")
test_thumbnail("long_label", "A very long label that might overflow", "test_text_long.png")
test_detail("condition", "positive", "text detail mode")

# ── boolean_flag renderer ─────────────────────────────────────────────────

print()
print("── boolean_flag ────────────────────────────────────────────────")

registry.register_by_tag("is_valid",   "boolean_flag")

test_thumbnail("is_valid", True,  "test_boolean_true.png")
test_thumbnail("is_valid", False, "test_boolean_false.png")
test_detail("is_valid", True, "boolean_flag detail mode")

# ── placeholder cases ─────────────────────────────────────────────────────

print()
print("── placeholders ────────────────────────────────────────────────")

# Unknown column — not registered with registry.
test_thumbnail("unknown_column", "some_value", "test_unknown_column.png")

# None value — registered column but value is None.
test_thumbnail("full_path", None, "test_none_value.png")

# ── ZoomableImageView import (regression after move to shared_widgets) ────────

print()
print("── ZoomableImageView ───────────────────────────────────────────")

try:
    from shared_widgets.zoomable_image_view import ZoomableImageView
    view = ZoomableImageView()
    print("  PASS  ZoomableImageView imports and instantiates from shared_widgets")
    _passed += 1
except Exception as e:
    print(f"  FAIL  ZoomableImageView — {e}")
    _failed += 1

# ── render with optional context parameter ────────────────────────────────────

print()
print("── render with context parameter ───────────────────────────────")

try:
    ctx = {"row_id": "test_row_001", "column_name": "some_number"}
    result = registry.render("some_number", 3.14, SIZE, mode="thumbnail", context=ctx)
    if result is not None:
        print("  PASS  registry.render() accepts context= without error")
        _passed += 1
    else:
        print("  FAIL  registry.render() with context returned None")
        _failed += 1
except TypeError as e:
    print(f"  FAIL  registry.render() does not accept context= kwarg: {e}")
    _failed += 1

if test_images:
    try:
        ctx = {"row_id": "test_row_001", "column_name": "full_path"}
        result = registry.render(
            "full_path", str(test_images[0]), SIZE,
            mode="thumbnail", context=ctx,
        )
        if result is not None:
            print("  PASS  media_path renderer accepts context= in thumbnail mode")
            _passed += 1
        else:
            print("  FAIL  media_path renderer with context returned None")
            _failed += 1
    except TypeError as e:
        print(f"  FAIL  media_path renderer does not accept context= kwarg: {e}")
        _failed += 1

# ── summary ───────────────────────────────────────────────────────────────

print()
print("─" * 60)
print(f"Results: {_passed} passed, {_failed} failed")
if _failed == 0:
    print("All renderer tests passed.")
    print(f"Open the images in {output_dir} to inspect the visual output.")
else:
    print(f"{_failed} test(s) failed — see details above.")
print("─" * 60)
