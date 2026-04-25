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
    QFileDialog, QSplitter
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
        - Multi-item view (several items side by side in a QSplitter)
        - Set-bound result view (mean face or other group outputs)

    TODO (Student A): Implement metadata table display.
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar with label and Save button.
        toolbar = QHBoxLayout()
        self._label = QLabel("No item selected")
        self._label.setStyleSheet("font-weight: bold;")
        toolbar.addWidget(self._label)
        toolbar.addStretch()

        save_btn = QPushButton("Save as PNG")
        save_btn.clicked.connect(self._save_as_png)
        toolbar.addWidget(save_btn)
        layout.addLayout(toolbar)

        # Placeholder shown before any item is selected.
        self._placeholder = QLabel("Double-click a tile to view it here.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: #888888; font-size: 13px;"
        )
        layout.addWidget(self._placeholder, stretch=1)

        # Metadata label (shown below the media widget).
        self._meta_label = QLabel("")
        self._meta_label.setWordWrap(True)
        self._meta_label.setStyleSheet(
            "font-size: 11px; color: #444444; padding: 4px;"
        )
        layout.addWidget(self._meta_label)

        # Keep a reference to the layout so we can swap media widgets.
        self._content_layout = layout

    def show_rows(self, row_ids: list[str]) -> None:
        """
        Shows the content for the given row_ids.

        For a single row_id, shows one media widget on top with the
        metadata text below it. For multiple row_ids, shows one panel
        per row_id arranged left-to-right in a horizontal QSplitter, so
        the researcher can resize each panel and compare items side by
        side. Each panel has its own media widget, file-name header,
        and metadata text.

        Args:
            row_ids: List of row_ids to display.
        """
        if not row_ids:
            return

        if len(row_ids) == 1:
            self._show_single(row_ids[0])
        else:
            self._show_multi(row_ids)

    def _show_single(self, row_id: str) -> None:
        """
        Renders a single row in the existing one-column layout.
        Updates the title, media widget, metadata label, and the cached
        pixmap used by Save as PNG.
        """
        metadata = self._controller.get_row(row_id)

        full_path = metadata.get("full_path", "")
        widget    = self._controller.render_column_value(
            "full_path", full_path, size=600, mode="detail",
            context={"row_id": row_id, "column_name": "full_path"},
        )

        self._swap_media_widget(widget)

        # Restore the global metadata label in case we just came back
        # from a multi-item view that hid it.
        self._meta_label.show()

        self._label.setText(metadata.get("file_name", row_id))
        self._meta_label.setText(self._format_meta_text(metadata))

        # Store pixmap if the widget is a ZoomableImageView (for Save PNG).
        if isinstance(widget, ZoomableImageView) and widget._pixmap_item:
            self._current_pixmap = widget._pixmap_item.pixmap()
        else:
            self._current_pixmap = None

    def _show_multi(self, row_ids: list[str]) -> None:
        """
        Renders a side-by-side comparison of several rows in a
        horizontal QSplitter. Each panel gets its own media widget,
        a file-name header, and a small metadata label so panels are
        independently readable.

        Save as PNG is disabled in multi-item mode — no single pixmap
        represents the view.
        """
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        for row_id in row_ids:
            panel = self._build_item_panel(row_id)
            splitter.addWidget(panel)

        # Equal split between panels by default.
        splitter.setSizes([1] * len(row_ids))

        self._swap_media_widget(splitter)

        # Hide the global metadata label — each panel has its own.
        self._meta_label.hide()

        self._label.setText(f"{len(row_ids)} items")

        # No single pixmap is meaningful when several items are shown.
        self._current_pixmap = None

    def _build_item_panel(self, row_id: str) -> QWidget:
        """
        Builds one column of the multi-item view: file-name header,
        rendered media widget, and a compact metadata label.

        Args:
            row_id: The row to render.

        Returns:
            A QWidget ready to be added to the side-by-side splitter.
        """
        metadata  = self._controller.get_row(row_id)
        full_path = metadata.get("full_path", "")
        media     = self._controller.render_column_value(
            "full_path", full_path, size=400, mode="detail",
            context={"row_id": row_id, "column_name": "full_path"},
        )

        panel  = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        title = QLabel(metadata.get("file_name", row_id))
        title.setStyleSheet("font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        if media is not None:
            layout.addWidget(media, stretch=1)

        meta = QLabel(self._format_meta_text(metadata))
        meta.setWordWrap(True)
        meta.setStyleSheet(
            "font-size: 11px; color: #444444; padding: 4px;"
        )
        layout.addWidget(meta)

        return panel

    def _format_meta_text(self, metadata: dict) -> str:
        """
        Renders metadata as `key: value` lines, skipping the columns
        that aren't useful in the detail view (full_path is the media
        itself, row_id is internal). Floats are formatted with 4
        significant digits to keep the panel narrow.
        """
        lines: list[str] = []
        for key, value in metadata.items():
            if key in ("full_path", "row_id"):
                continue
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4g}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

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

    # ── Internal helpers ──────────────────────────────────────────────

    def _swap_media_widget(self, new_widget: QWidget | None) -> None:
        """
        Replaces the current media widget with a new one.
        Removes the old widget from the layout and adds the new one.

        Args:
            new_widget: The widget to display, or None for placeholder.
        """
        # Remove the old media widget if present.
        if self._media_widget is not None:
            self._content_layout.removeWidget(self._media_widget)
            self._media_widget.deleteLater()
            self._media_widget = None

        # Hide or show the placeholder.
        if new_widget is not None:
            self._placeholder.hide()
            # Insert the new widget above the metadata label.
            # The metadata label is always the last item; insert before it.
            insert_index = self._content_layout.count() - 1
            self._content_layout.insertWidget(insert_index, new_widget, stretch=1)
            self._media_widget = new_widget
        else:
            self._placeholder.show()

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
