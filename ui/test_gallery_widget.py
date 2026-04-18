"""
tests/test_gallery_widget.py

A standalone script for testing GalleryWidget in isolation.
Run with:
    python tests/test_gallery_widget.py

This script creates a minimal Qt window containing just the
GalleryWidget and a FakeController. It is much faster to iterate
on UI changes this way than running the full application every time.

---- HOW TO ADAPT THIS FOR OTHER WIDGETS ----

To test a different widget, follow these steps:

1. Change the import:
      from ui.gallery_widget import GalleryWidget
   to whichever widget you want to test, e.g.:
      from ui.filter_panel import FilterPanel

2. Change the widget instantiation:
      widget = GalleryWidget(controller)
   to:
      widget = FilterPanel(controller)

3. If your widget needs to respond to controller signals,
   connect them here. For example, FilterPanel needs columns:
      controller.columns_updated.connect(widget.refresh_columns)

4. Adjust the window size if needed:
      widget.resize(300, 800)  # FilterPanel is narrow and tall

---- CONNECTING SIGNALS ----

FakeController emits the same signals as the real AppController.
After calling controller.start(), it emits:
    - columns_updated   -> connect to filter panel, column selector
    - gallery_updated   -> connect to gallery widget
    - thumbnail_ready   -> connect to gallery widget

If your widget listens for signals, connect them before
calling controller.start(). See the examples below.

---- EXAMPLE: testing FilterPanel ----

    from ui.filter_panel import FilterPanel
    widget = FilterPanel(controller)
    controller.columns_updated.connect(widget.refresh_columns)
    widget.resize(280, 700)
    widget.show()

---- EXAMPLE: testing DetailWidget ----

    from ui.detail_widget import DetailWidget
    widget = DetailWidget(controller)
    controller.row_selected.connect(
        lambda meta: widget.show_rows([meta["row_id"]])
    )
    widget.resize(600, 700)
    widget.show()
    # After start(), click a tile in the gallery to trigger row_selected.

"""

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

# Add the project root to the Python path so imports work correctly.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ui.fake_controller import FakeController
from ui.gallery_widget import GalleryWidget


def main():
    app = QApplication(sys.argv)

    # Create FakeController pointing at the test images folder.
    test_folder = project_root / "test_images"
    if not test_folder.exists():
        print(f"Warning: test_images folder not found at {test_folder}")
        print("The gallery will show placeholder tiles instead of real images.")
        test_folder = project_root

    controller = FakeController(test_folder)

    # Create the widget you want to test.
    # Change this line to test a different widget.
    widget = GalleryWidget(controller)

    # Connect any signals your widget needs.
    # GalleryWidget needs gallery_updated and thumbnail_ready.
    controller.gallery_updated.connect(widget.set_row_ids)
    controller.thumbnail_ready.connect(widget.on_thumbnail_ready)
    controller.row_updated.connect(widget.on_row_updated)

    # Set a reasonable window size and show the widget.
    widget.resize(900, 600)
    widget.setWindowTitle("Widget test — GalleryWidget")
    widget.show()

    # Start FakeController after the widget is visible.
    # The 100ms delay ensures all signal connections are established
    # before the controller starts emitting.
    QTimer.singleShot(100, controller.start)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()