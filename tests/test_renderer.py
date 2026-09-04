"""
tests/test_renderer.py

Tests the column type renderers by rendering sample values and saving
the output as PNG files. Run this to verify render functions produce
correct visual output without needing the full application.

This is a standalone manual-check script, not a pytest test module --
it is named test_*.py only because it lives alongside the automated
suite. check_thumbnail() and check_detail() are prefixed "check_", not
"test_", so pytest does not try to collect them as test functions.

Usage:
    python tests/test_renderer.py

Output images are saved to tests/renderer_output/
Inspect them by eye to verify the renderers work.

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


def check_thumbnail(tag, value, filename: str) -> None:
    """
    Renders a value in thumbnail mode and saves it as a PNG file.
    A QPixmap is expected.

    P1.8d-2b-3: the registry maps a type tag to a renderer, so this
    passes a tag ('media_path', 'numeric', ..., or None for the
    unknown-column case) rather than a column name.
    """
    global _passed, _failed
    pixmap = registry.render_by_tag(tag, value, SIZE, mode="thumbnail", label=tag)
    if pixmap is not None:
        path = output_dir / filename
        pixmap.save(str(path), "PNG")
        print(f"  PASS  {filename}")
        _passed += 1
    else:
        print(f"  FAIL  {filename} — render returned None")
        _failed += 1


def check_detail(tag, value, label: str) -> None:
    """
    Renders a value in detail mode and checks that a QWidget is returned.
    Does not save to disk (widgets can't be saved as PNG directly).
    """
    global _passed, _failed
    widget = registry.render_by_tag(tag, value, SIZE, mode="detail", label=tag)
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
    check_thumbnail("media_path", str(test_images[0]), "test_image_thumbnail.png")
    check_detail("media_path", str(test_images[0]), "image detail mode")
else:
    print("  SKIP  image tests — no .jpg/.png files found in test_images/")

# ── media_path renderer — videos ──────────────────────────────────────────

print()
print("── media_path (videos) ─────────────────────────────────────────")

test_videos = list((project_root / "test_images").glob("*.mp4"))
test_videos += list((project_root / "test_images").glob("*.mov"))

if test_videos:
    check_thumbnail("media_path", str(test_videos[0]), "test_video_thumbnail.png")
    check_detail("media_path", str(test_videos[0]), "video detail mode")
else:
    print("  SKIP  video tests — no .mp4/.mov files found in test_images/")
    print("        Add a short video to test_images/ to test video rendering.")

# ── numeric renderer ──────────────────────────────────────────────────────

print()
print("── numeric ─────────────────────────────────────────────────────")

check_thumbnail("numeric",  3.14159, "test_numeric_float.png")
check_thumbnail("numeric", 42,      "test_numeric_int.png")
check_detail("numeric", 3.14159, "numeric detail mode")

# ── text renderer ─────────────────────────────────────────────────────────

print()
print("── text ────────────────────────────────────────────────────────")

check_thumbnail("text",  "positive", "test_text_short.png")
check_thumbnail("text", "A very long label that might overflow", "test_text_long.png")
check_detail("text", "positive", "text detail mode")

# ── boolean_flag renderer ─────────────────────────────────────────────────

print()
print("── boolean_flag ────────────────────────────────────────────────")

check_thumbnail("boolean_flag", True,  "test_boolean_true.png")
check_thumbnail("boolean_flag", False, "test_boolean_false.png")
check_detail("boolean_flag", True, "boolean_flag detail mode")

# ── placeholder cases ─────────────────────────────────────────────────────

print()
print("── placeholders ────────────────────────────────────────────────")

# Unknown tag — no renderer registered for it.
check_thumbnail(None, "some_value", "test_unknown_column.png")

# None value — known tag but value is None.
check_thumbnail("media_path", None, "test_none_value.png")

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
    result = registry.render_by_tag("numeric", 3.14, SIZE, mode="thumbnail", context=ctx)
    if result is not None:
        print("  PASS  registry.render_by_tag() accepts context= without error")
        _passed += 1
    else:
        print("  FAIL  registry.render_by_tag() with context returned None")
        _failed += 1
except TypeError as e:
    print(f"  FAIL  registry.render_by_tag() does not accept context= kwarg: {e}")
    _failed += 1

if test_images:
    try:
        ctx = {"row_id": "test_row_001", "column_name": "full_path"}
        result = registry.render_by_tag(
            "media_path", str(test_images[0]), SIZE,
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
