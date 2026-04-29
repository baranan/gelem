"""
ui/merge_report_dialog.py

A diagnostics dialog the researcher sees after starting a CSV merge,
before any data is written. Replaces the old plain Yes/No QMessageBox.

The dialog reads attributes off a MergeReport object — total counts
plus four diagnostic lists (unmatched files, unmatched CSV rows,
duplicate keys on each side). It does NOT import MergeReport itself,
so ui/ stays inside the import boundary in ARCHITECTURE_RULES.md.
The dialog uses duck-typing — anything with the expected attribute
names will work.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTabWidget, QListWidget,
    QListWidgetItem, QWidget, QFrame
)
from PySide6.QtCore import Qt


# Maximum number of items shown per diagnostic list before truncation.
# A merge with thousands of unmatched rows would otherwise hang the
# dialog for a few seconds; the researcher can still see the count.
_LIST_PREVIEW_LIMIT = 500


class MergeReportDialog(QDialog):
    """
    Shows the result of a dry-run merge so the researcher can inspect
    what would happen before committing the changes.

    Layout:
        - Counts grid: total CSV rows, total image files, matched rows,
          and four issue counts. Each issue count is colour-coded so
          problems jump out at a glance.
        - Tabbed list of the actual problem rows (unmatched files,
          unmatched CSV rows, duplicate keys on either side). Tabs
          for empty issues are hidden so the dialog stays compact.
        - Proceed / Cancel buttons. The accepted attribute is True
          after exec() returns Accepted; False otherwise.
    """

    def __init__(self, report, parent=None):
        """
        Args:
            report: A MergeReport-like object with attributes
                    total_csv_rows, total_image_files, matched_rows,
                    unmatched_files, unmatched_csv_rows,
                    duplicate_keys_files, duplicate_keys_csv.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Merge diagnostics")
        self.setMinimumSize(560, 460)
        self.accepted_merge: bool = False

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────
        header = QLabel("Review the merge before applying it")
        header.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #4A90D9;"
        )
        layout.addWidget(header)

        # ── Counts grid ──────────────────────────────────────────────
        layout.addWidget(self._build_counts_grid(report))

        # Visual separator between counts and per-issue lists.
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # ── Tabbed issue lists ───────────────────────────────────────
        tabs = self._build_issue_tabs(report)
        if tabs is not None:
            layout.addWidget(tabs, stretch=1)
        else:
            ok_label = QLabel(
                "No issues found — every CSV row matches an image file "
                "and every image file matches a CSV row."
            )
            ok_label.setStyleSheet("color: #43A047; font-size: 11px;")
            ok_label.setWordWrap(True)
            layout.addWidget(ok_label, stretch=1)

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        proceed_btn = QPushButton("Proceed with merge")
        proceed_btn.setDefault(True)
        proceed_btn.clicked.connect(self._on_proceed)
        btn_row.addWidget(proceed_btn)

        layout.addLayout(btn_row)

    # ── Section builders ──────────────────────────────────────────────

    def _build_counts_grid(self, report) -> QWidget:
        """
        Builds the counts grid at the top of the dialog. Three top
        counts on the first row (CSV rows, image files, matched), and
        the four issue counts on the second row, colour-coded.
        """
        wrap   = QWidget()
        grid   = QGridLayout(wrap)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(4)

        # Row 0 — totals.
        grid.addWidget(self._stat("CSV rows",
                                  report.total_csv_rows), 0, 0)
        grid.addWidget(self._stat("Image files",
                                  report.total_image_files), 0, 1)
        grid.addWidget(self._stat("Matched",
                                  report.matched_rows,
                                  good=report.matched_rows > 0), 0, 2)

        # Row 1 — issue counts. Bad-news counts are red when non-zero.
        unmatched_files_n  = len(report.unmatched_files)
        unmatched_csv_n    = len(report.unmatched_csv_rows)
        dup_files_n        = len(report.duplicate_keys_files)
        dup_csv_n          = len(report.duplicate_keys_csv)

        grid.addWidget(self._stat("Files w/o match",
                                  unmatched_files_n,
                                  bad=unmatched_files_n > 0), 1, 0)
        grid.addWidget(self._stat("CSV rows w/o file",
                                  unmatched_csv_n,
                                  bad=unmatched_csv_n > 0), 1, 1)
        grid.addWidget(self._stat("Duplicate file keys",
                                  dup_files_n,
                                  bad=dup_files_n > 0), 1, 2)
        grid.addWidget(self._stat("Duplicate CSV keys",
                                  dup_csv_n,
                                  bad=dup_csv_n > 0), 1, 3)

        return wrap

    def _build_issue_tabs(self, report) -> QTabWidget | None:
        """
        Adds one tab per non-empty issue list. Returns None when all
        four lists are empty so the caller can show a single "no
        issues" line instead of an empty tab widget.
        """
        sources = [
            ("Files without a CSV match", report.unmatched_files),
            ("CSV rows without a file",   report.unmatched_csv_rows),
            ("Duplicate keys (files)",    report.duplicate_keys_files),
            ("Duplicate keys (CSV)",      report.duplicate_keys_csv),
        ]

        non_empty = [(t, items) for t, items in sources if items]
        if not non_empty:
            return None

        tabs = QTabWidget()
        for title, items in non_empty:
            tabs.addTab(self._build_issue_list(items),
                        f"{title} ({len(items)})")
        return tabs

    # ── Small helpers ─────────────────────────────────────────────────

    def _stat(
        self,
        label: str,
        value: int,
        bad: bool = False,
        good: bool = False,
    ) -> QWidget:
        """
        Builds a small "<value>\n<label>" widget. `bad=True` colours
        the value red when non-zero; `good=True` colours it green.
        """
        wrap   = QWidget()
        col    = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        v = QLabel(str(value))
        # Only override the palette colour when we have a real signal
        # (bad/good with non-zero value). Otherwise let Qt pick the
        # right foreground for the active theme — hardcoded #222222
        # is invisible on dark themes.
        if bad and value > 0:
            v.setStyleSheet(
                "font-weight: bold; font-size: 16px; color: #E53935;"
            )
        elif good and value > 0:
            v.setStyleSheet(
                "font-weight: bold; font-size: 16px; color: #43A047;"
            )
        else:
            v.setStyleSheet("font-weight: bold; font-size: 16px;")
        col.addWidget(v)

        cap = QLabel(label)
        cap.setStyleSheet("font-size: 11px;")
        col.addWidget(cap)

        return wrap

    def _build_issue_list(self, items: list) -> QListWidget:
        """
        Renders one diagnostic list as a QListWidget. Truncates to
        _LIST_PREVIEW_LIMIT entries and appends a marker row when the
        actual list is longer, so massive merges don't lock the UI.
        """
        widget = QListWidget()
        widget.setUniformItemSizes(True)

        shown = items[:_LIST_PREVIEW_LIMIT]
        for it in shown:
            widget.addItem(QListWidgetItem(str(it)))

        if len(items) > _LIST_PREVIEW_LIMIT:
            extra = len(items) - _LIST_PREVIEW_LIMIT
            more  = QListWidgetItem(f"… and {extra} more (not shown)")
            more.setForeground(Qt.GlobalColor.gray)
            widget.addItem(more)

        return widget

    def _on_proceed(self) -> None:
        """Marks the dialog as accepted and closes it."""
        self.accepted_merge = True
        self.accept()
