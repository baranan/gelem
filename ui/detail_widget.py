"""
ui/detail_widget.py

DetailWidget shows a full-size view of one or more items when the
researcher double-clicks a tile in the gallery.

The content displayed depends on the column type of full_path:
    - Images: a ZoomableImageView with zoom and pan.
    - Videos: a video player with play/pause controls.
    - Other column types: whatever QWidget the renderer returns.

DetailWidget never loads files directly or checks file extensions.
It calls controller.render_column_value(mode='detail') and displays
whatever widget comes back. This keeps all media-type logic inside
the renderer where it belongs.

Student A is responsible for implementing this class.
"""

from __future__ import annotations
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGraphicsView, QGraphicsScene,
    QFileDialog, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QWheelEvent
from shared_widgets.zoomable_image_view import ZoomableImageView

class DetailWidget(QWidget):
    """
    Shows full-size content for a selected row, plus metadata.

    On double-click, MainWindow calls show_rows([row_id]). DetailWidget
    asks the controller to render the 'full_path' column in 'detail'
    mode, and displays whatever widget comes back — a ZoomableImageView
    for images, a video player for videos, or a placeholder for anything
    unrecognised.

    Supports:
        - Single item view (one media widget + metadata table)
        - Multi-item view (TODO: several items side by side)
        - Set-bound result view (mean face or other group outputs)

    TODO (Student A): Implement side-by-side multi-item layout.
    """

    def __init__(self, controller, parent=None):
        """
        Creates the DetailWidget.

        Args:
            controller: The AppController instance.
            parent:     Optional parent widget.
        """
        super().__init__(parent)
        self._controller      = controller
        self._current_pixmap  = None   # For Save as PNG (images only).
        self._media_widget    = None   # The currently displayed QWidget.
        self._meta_table_sized = False  # First-populate splitter resize.

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar with label, Save button, and Close button.
        toolbar = QHBoxLayout()
        self._label = QLabel("No item selected")
        self._label.setStyleSheet("font-weight: bold;")
        toolbar.addWidget(self._label)
        toolbar.addStretch()

        self._save_btn = QPushButton("Save as PNG")
        self._save_btn.clicked.connect(self._save_as_png)
        self._save_btn.setEnabled(False)
        self._save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        toolbar.addWidget(self._save_btn)

        self._close_btn = QPushButton("\u2715")  # multiplication-x glyph
        self._close_btn.setToolTip("Close")
        self._close_btn.setFixedWidth(28)
        self._close_btn.clicked.connect(self._clear)
        self._close_btn.setEnabled(False)
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        toolbar.addWidget(self._close_btn)

        layout.addLayout(toolbar)

        # Container for the media area. The placeholder lives inside it
        # initially; once a row is shown, the placeholder is hidden and
        # the rendered media widget is added alongside it.
        self._media_area    = QWidget()
        self._media_layout  = QVBoxLayout(self._media_area)
        self._media_layout.setContentsMargins(0, 0, 0, 0)

        self._placeholder = QLabel("Double-click a tile to view it here.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: #888888; font-size: 13px;"
        )
        self._media_layout.addWidget(self._placeholder, stretch=1)

        # Metadata table (shown below the media widget). Two columns —
        # property name on the left, value on the right.
        self._meta_table = QTableWidget(0, 2)
        self._meta_table.setHorizontalHeaderLabels(["Property", "Value"])
        self._meta_table.verticalHeader().setVisible(False)
        self._meta_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._meta_table.setAlternatingRowColors(True)
        self._meta_table.setSelectionMode(
            QTableWidget.SelectionMode.NoSelection
        )
        self._meta_table.setShowGrid(False)
        header = self._meta_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        # Vertical splitter between media area and metadata table so the
        # researcher can resize either to taste.
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.addWidget(self._media_area)
        self._splitter.addWidget(self._meta_table)
        self._splitter.setStretchFactor(0, 1)   # media area expands
        self._splitter.setStretchFactor(1, 0)   # table sized to handle
        self._splitter.setSizes([600, 200])     # initial proportions
        self._splitter.setChildrenCollapsible(False)
        layout.addWidget(self._splitter, stretch=1)

    def show_rows(self, row_ids: list[str]) -> None:
        """
        Shows the content for the given row_ids.

        Calls controller.render_column_value(mode='detail') to get the
        appropriate widget for the row's full_path column. Displays
        that widget in the detail area.

        For a single row_id, shows one media widget plus metadata.
        For multiple row_ids, shows the first item (side-by-side layout
        is a TODO for Student A).

        Args:
            row_ids: List of row_ids to display.

        TODO (Student A): Implement side-by-side layout for multiple
        row_ids — create one media widget per row_id and arrange them
        in a QSplitter or QHBoxLayout.
        """
        if not row_ids:
            return

        row_id   = row_ids[0]
        metadata = self._controller.get_row(row_id)

        # Ask the controller to render full_path in detail mode.
        # The renderer decides whether to return a ZoomableImageView,
        # a video player, or a placeholder — DetailWidget does not care.
        full_path = metadata.get("full_path", "")
        widget    = self._controller.render_column_value(
            "full_path", full_path, size=600, mode="detail",
            context={"row_id": row_id, "column_name": "full_path"},
        )

        self._swap_media_widget(widget)

        # Update the title label with the file name.
        self._label.setText(metadata.get("file_name", row_id))

        # Populate the metadata table. We hide full_path and row_id —
        # full_path is shown as media above, row_id is internal.
        self._populate_meta_table(metadata)

        # Store pixmap if the widget is a ZoomableImageView (for Save PNG).
        if isinstance(widget, ZoomableImageView) and widget._pixmap_item:
            self._current_pixmap = widget._pixmap_item.pixmap()
        else:
            self._current_pixmap = None

        # Something is shown — enable the toolbar buttons.
        self._save_btn.setEnabled(self._current_pixmap is not None)
        self._close_btn.setEnabled(True)

    def show_result(self, result: dict) -> None:
        """
        Shows a set-bound operator result (e.g. mean face image).
        Called when display_result_ready signal arrives.

        Uses the same render_column_value pathway if result contains
        an artifact_path — treats it as a media_path column value.

        Args:
            result: Dict from the operator. May contain 'artifact_path'.

        TODO (Student A): Display summary statistics from result dict
        alongside the media widget.
        """
        artifact_path = result.get("artifact_path", "")
        if artifact_path:
            widget = self._controller.render_column_value(
                "full_path", artifact_path, size=600, mode="detail"
            )
            self._swap_media_widget(widget)
            self._label.setText("Group result")
            if isinstance(widget, ZoomableImageView) and widget._pixmap_item:
                self._current_pixmap = widget._pixmap_item.pixmap()
            else:
                self._current_pixmap = None
            self._save_btn.setEnabled(self._current_pixmap is not None)
            self._close_btn.setEnabled(True)

    # ── Internal helpers ──────────────────────────────────────────────

    def _clear(self) -> None:
        """
        Resets the detail view to its empty state — hides any media
        widget, restores the placeholder, empties the metadata table,
        and disables the toolbar buttons.
        """
        self._swap_media_widget(None)
        self._meta_table.setRowCount(0)
        self._label.setText("No item selected")
        self._current_pixmap = None
        self._save_btn.setEnabled(False)
        self._close_btn.setEnabled(False)

    def _swap_media_widget(self, new_widget: QWidget | None) -> None:
        """
        Replaces the current media widget with a new one inside the
        media area of the splitter.

        Args:
            new_widget: The widget to display, or None to fall back to
                        the placeholder.
        """
        if self._media_widget is not None:
            self._media_layout.removeWidget(self._media_widget)
            self._media_widget.deleteLater()
            self._media_widget = None

        if new_widget is not None:
            self._placeholder.hide()
            self._media_layout.addWidget(new_widget, stretch=1)
            self._media_widget = new_widget
        else:
            self._placeholder.show()

    def _populate_meta_table(self, metadata: dict) -> None:
        """
        Fills the metadata table with one row per property in metadata,
        skipping fields that aren't useful to show ('full_path' is the
        media itself, 'row_id' is internal).

        Floats are formatted with 4 significant digits to keep the
        column compact; everything else is rendered with str().
        """
        rows = [
            (k, v) for k, v in metadata.items()
            if k not in ("full_path", "row_id")
        ]

        self._meta_table.setRowCount(len(rows))
        for row_idx, (key, value) in enumerate(rows):
            key_item = QTableWidgetItem(str(key))
            key_item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )

            text = f"{value:.4g}" if isinstance(value, float) else str(value)
            val_item = QTableWidgetItem(text)
            val_item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )

            self._meta_table.setItem(row_idx, 0, key_item)
            self._meta_table.setItem(row_idx, 1, val_item)

        self._meta_table.resizeRowsToContents()

        # On the first populate, resize the splitter so the entire table
        # is visible without scrolling. Subsequent populates leave the
        # researcher's manual splitter position alone.
        if not self._meta_table_sized and rows:
            self._fit_splitter_to_table()
            self._meta_table_sized = True

    def _fit_splitter_to_table(self) -> None:
        """
        Adjusts the splitter so the metadata table is tall enough to show
        every row without a vertical scrollbar — capped so the media area
        never shrinks below half the splitter's height.
        """
        header_h = self._meta_table.horizontalHeader().height()
        rows_h   = sum(
            self._meta_table.rowHeight(r)
            for r in range(self._meta_table.rowCount())
        )
        frame    = 2 * self._meta_table.frameWidth()
        desired  = header_h + rows_h + frame + 6  # small buffer

        total = sum(self._splitter.sizes()) or self._splitter.height()
        if total <= 0:
            return

        # Don't let the table eat more than ~60% of the splitter.
        max_table = int(total * 0.6)
        table_h   = min(desired, max_table)
        media_h   = max(total - table_h, 1)
        self._splitter.setSizes([media_h, table_h])

    def _save_as_png(self) -> None:
        """
        Saves the currently displayed image as a PNG file.
        Only available when the current item is an image.
        """
        if self._current_pixmap is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save image as PNG",
            "",
            "PNG files (*.png)",
        )
        if path:
            self._current_pixmap.save(path, "PNG")
