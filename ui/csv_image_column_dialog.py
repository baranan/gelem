"""
ui/csv_image_column_dialog.py

A small dialog the researcher sees when starting a new project
from a CSV file. It reads the CSV's header row, lets them pick
which column (if any) holds the image file paths, and previews
the first value from that column so they can confirm they chose
correctly.

We read the header with the stdlib `csv` module — no pandas — so
this stays inside the UI import boundary defined in
ARCHITECTURE_RULES.md.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton
)


# Sentinel that means "this CSV does not have an image column".
NO_IMAGE_COLUMN_LABEL = "(none — CSV has no image paths)"


class CsvImageColumnDialog(QDialog):
    """
    Asks the researcher which CSV column holds image file paths.

    Behaviour:
        - The combo lists every column name from the CSV's header
          row, plus a "(none)" entry at the top.
        - When the selection changes, the dialog shows a one-line
          preview of the first non-empty value from that column so
          the researcher can sanity-check their pick.
        - If the CSV cannot be read, the dialog still opens with
          the "(none)" option only and a small error label.

    After exec() returns Accepted, read `image_column` — it is None
    when the researcher chose "(none)", otherwise the column name.
    """

    def __init__(self, csv_path: Path, parent=None):
        """
        Args:
            csv_path: Path to the CSV the researcher just chose.
            parent:   Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("New project from CSV — image column")
        self.setMinimumWidth(420)

        # Public result. None means "no image column".
        self.image_column: str | None = None

        # Cached header / first-row data so preview lookups don't
        # re-read the file on every selection change.
        self._columns:   list[str] = []
        self._first_row: dict[str, str] = {}
        self._read_error: str | None   = None
        self._load_csv_header(csv_path)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        layout.addWidget(QLabel(
            "Pick the CSV column that contains image file paths.\n"
            "Choose \"(none)\" if your CSV has no images."
        ))

        if self._read_error:
            err = QLabel(f"Could not read CSV: {self._read_error}")
            err.setStyleSheet("color: #B00020; font-size: 11px;")
            layout.addWidget(err)

        self._combo = QComboBox()
        self._combo.addItem(NO_IMAGE_COLUMN_LABEL, userData=None)
        for col in self._columns:
            self._combo.addItem(col, userData=col)
        self._combo.currentIndexChanged.connect(self._refresh_preview)
        layout.addWidget(self._combo)

        # Auto-pick a sensible default if the CSV has an obvious
        # image-path column. The researcher can still change it.
        self._guess_default_column()

        # Preview label — shows the first value from the chosen
        # column so the researcher can confirm.
        self._preview = QLabel("")
        self._preview.setWordWrap(True)
        self._preview.setStyleSheet(
            "color: #444444; font-size: 11px; padding: 4px;"
        )
        layout.addWidget(self._preview)
        self._refresh_preview()

        # Buttons.
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("Open project")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_accept)
        btn_layout.addWidget(ok_btn)

        layout.addLayout(btn_layout)

    # ── Internal helpers ──────────────────────────────────────────────

    def _load_csv_header(self, csv_path: Path) -> None:
        """
        Reads the header row plus the first data row (for previews).
        Stores any error message on self._read_error so the dialog
        can surface it without crashing.
        """
        try:
            with csv_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    self._read_error = "CSV is empty."
                    return

                self._columns = [c.strip() for c in header if c.strip()]

                first = next(reader, None)
                if first is not None:
                    # Map column name -> first row's value for previewing.
                    for i, col in enumerate(header):
                        if i < len(first):
                            self._first_row[col.strip()] = first[i]
        except Exception as exc:
            self._read_error = str(exc)

    def _guess_default_column(self) -> None:
        """
        Picks the first column whose name suggests it holds image
        paths (file_name, image_path, full_path, ...) and selects
        it in the combo. Falls back to "(none)" if nothing matches.
        """
        hints = (
            "full_path", "image_path", "img_path", "path",
            "image", "file_name", "filename", "file",
        )
        lowered = {c.lower(): c for c in self._columns}
        for hint in hints:
            if hint in lowered:
                idx = self._combo.findData(lowered[hint])
                if idx >= 0:
                    self._combo.setCurrentIndex(idx)
                    return

    def _refresh_preview(self) -> None:
        """
        Updates the preview label with the first non-empty value
        from the currently selected column, or a hint when "(none)"
        is selected.
        """
        column = self._combo.currentData()
        if column is None:
            self._preview.setText(
                "No image column — the project will load metadata only."
            )
            return

        sample = self._first_row.get(column, "").strip()
        if sample:
            self._preview.setText(f"First value: {sample}")
        else:
            self._preview.setText(
                "First value: (empty — pick another column "
                "if this isn't the right one)"
            )

    def _on_accept(self) -> None:
        """Stores the chosen column on the dialog and accepts."""
        self.image_column = self._combo.currentData()
        self.accept()
