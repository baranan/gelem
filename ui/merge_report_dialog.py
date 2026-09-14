"""
ui/merge_report_dialog.py

A diagnostics dialog the researcher sees after starting a CSV merge,
before any data is written. Replaces the old plain Yes/No QMessageBox.

The dialog reads attributes off a MergeReport object — total counts
plus diagnostic lists (unmatched target rows, unmatched CSV rows,
duplicate keys on each side, the keys that would have expanded the
target table, and an advisory decimal-key warning). It does NOT import
MergeReport itself, so ui/ stays inside the import boundary in
ARCHITECTURE_RULES.md. The dialog uses duck-typing — anything with the
expected attribute names will work.
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
        - Counts grid: total CSV rows, total target rows, matched rows,
          and four issue counts. Each issue count is colour-coded so
          problems jump out at a glance.
        - Tabbed list of the actual problem rows (rows that would expand
          the table, unmatched target rows, unmatched CSV rows,
          duplicate keys on either side, and an advisory decimal-key
          warning). Tabs for empty issues are hidden so the dialog stays
          compact.
        - Proceed / Cancel buttons. The accepted attribute is True
          after exec() returns Accepted; False otherwise. Proceed is
          disabled only when the merge was refused (would_expand) --
          there is nothing pending to commit then. The decimal-key
          warning is advisory and never disables Proceed.
    """

    def __init__(self, report, parent=None):
        """
        Args:
            report: A MergeReport-like object with attributes
                    total_csv_rows, total_target_rows, matched_rows,
                    unmatched_target_rows, unmatched_csv_rows,
                    duplicate_keys_target, duplicate_keys_csv,
                    would_expand, float_key_warning.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Merge diagnostics")
        self.setMinimumSize(560, 460)
        self.accepted_merge: bool = False

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        refused = bool(report.would_expand)

        # ── Header ────────────────────────────────────────────────────
        if refused:
            header = QLabel(
                f"Merge refused — {len(report.would_expand)} row(s) would "
                f"expand the table"
            )
            header.setStyleSheet(
                "font-weight: bold; font-size: 13px; color: #E53935;"
            )
        else:
            header = QLabel("Review the merge before applying it")
            header.setStyleSheet(
                "font-weight: bold; font-size: 13px; color: #4A90D9;"
            )
        layout.addWidget(header)

        if refused:
            explain = QLabel(
                "Some rows in the CSV share a key that also matches one row "
                "of the target table. Merging them would turn that one row "
                "into several (row expansion), which this merge does not do "
                "yet — it is planned as a separate feature. No changes have "
                "been made."
            )
            explain.setWordWrap(True)
            explain.setStyleSheet("font-size: 11px;")
            layout.addWidget(explain)

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
                "No issues found — every CSV row matches a target row "
                "and every target row matches a CSV row."
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

        # A refused merge has nothing pending to commit (Dataset.merge_csv
        # left _pending_df unset), so proceeding would be a silent no-op.
        # Disabling the button here says so up front instead of letting the
        # researcher click through into nothing happening.
        proceed_btn = QPushButton("Proceed with merge")
        proceed_btn.setDefault(not refused)
        proceed_btn.setEnabled(not refused)
        proceed_btn.clicked.connect(self._on_proceed)
        btn_row.addWidget(proceed_btn)

        layout.addLayout(btn_row)

    # ── Section builders ──────────────────────────────────────────────

    def _build_counts_grid(self, report) -> QWidget:
        """
        Builds the counts grid at the top of the dialog. Three top
        counts on the first row (CSV rows, target rows, matched), and
        the four issue counts on the second row, colour-coded.
        """
        wrap   = QWidget()
        grid   = QGridLayout(wrap)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(4)

        # Row 0 — totals.
        grid.addWidget(self._stat("CSV rows",
                                  report.total_csv_rows), 0, 0)
        grid.addWidget(self._stat("Target rows",
                                  report.total_target_rows), 0, 1)
        grid.addWidget(self._stat("Matched",
                                  report.matched_rows,
                                  good=report.matched_rows > 0), 0, 2)

        # Row 1 — issue counts. Bad-news counts are red when non-zero.
        unmatched_target_n = len(report.unmatched_target_rows)
        unmatched_csv_n    = len(report.unmatched_csv_rows)
        dup_target_n       = len(report.duplicate_keys_target)
        dup_csv_n          = len(report.duplicate_keys_csv)

        grid.addWidget(self._stat("Target rows w/o match",
                                  unmatched_target_n,
                                  bad=unmatched_target_n > 0), 1, 0)
        grid.addWidget(self._stat("CSV rows w/o match",
                                  unmatched_csv_n,
                                  bad=unmatched_csv_n > 0), 1, 1)
        grid.addWidget(self._stat("Duplicate target keys",
                                  dup_target_n,
                                  bad=dup_target_n > 0), 1, 2)
        grid.addWidget(self._stat("Duplicate CSV keys",
                                  dup_csv_n,
                                  bad=dup_csv_n > 0), 1, 3)

        return wrap

    def _build_issue_tabs(self, report) -> QTabWidget | None:
        """
        Adds one tab per non-empty issue list. Returns None when every
        list is empty so the caller can show a single "no issues" line
        instead of an empty tab widget.
        """
        sources = [
            ("Would expand the table",        report.would_expand),
            ("Target rows without a CSV match", report.unmatched_target_rows),
            ("CSV rows without a target match", report.unmatched_csv_rows),
            ("Duplicate keys (target)",         report.duplicate_keys_target),
            ("Duplicate keys (CSV)",            report.duplicate_keys_csv),
            # Advisory, not a refusal -- shown the same way (a tab with the
            # message as its one row) but never affects whether Proceed is
            # enabled; see the `refused` computation in __init__.
            ("Decimal key warning", [report.float_key_warning] if report.float_key_warning else []),
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
