"""
ui/merge_report_dialog.py

A diagnostics dialog the researcher sees after starting a CSV merge,
before any data is written. Replaces the old plain Yes/No QMessageBox.

The dialog reads attributes off a MergeReport object — total counts
plus diagnostic lists (unmatched target rows, unmatched CSV rows,
duplicate keys on each side, the CSV key values that would expand the
target table, and an advisory decimal-key warning), plus -- when the
merge would expand the target table -- the expansion offer fields
(expand_table_name, expand_row_count, expand_carried_columns). It does
NOT import MergeReport itself, so ui/ stays inside the import boundary
in CLAUDE.md. The dialog uses duck-typing — anything with the expected
attribute names will work.

The file is in two layers, the same split ui/settings_dialog.py and
ui/parameter_dialog.py use:

  * Layer A -- module-level functions taking a report and returning plain
    strings/lists, with no Qt. All the wording lives here, so the tests
    can exercise it without a QApplication.

  * Layer B -- class MergeReportDialog(QDialog), thin glue that calls
    Layer A and lays the results out in widgets.
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


# ---------------------------------------------------------------------------
# Layer A -- wording and arithmetic, no Qt.
# ---------------------------------------------------------------------------

def is_expand_offer(report) -> bool:
    """True when this merge would expand the target table -- P1.5b's offer
    to create a new table, rather than the ordinary in-place join."""
    return bool(report.would_expand)


def header_text(report) -> str:
    """The dialog's headline. P1.5a's red 'Merge refused' wording is gone:
    an expanding merge is now an offer, not an error, so it gets the same
    neutral phrasing as the ordinary review header, just naming what will
    be created."""
    if is_expand_offer(report):
        return (
            f"This will create a new table — "
            f"{report.expand_row_count} row(s), one per CSV row"
        )
    return "Review the merge before applying it"


def explain_text(report) -> str:
    """The paragraph under the header. Empty string means no paragraph is
    shown -- the ordinary in-place merge doesn't need one; its counts grid
    and issue tabs already say everything."""
    if not is_expand_offer(report):
        return ""
    return (
        f"{len(report.would_expand)} key value(s) in the CSV each match "
        f"one row of '{report.target_table}' more than once, so joining "
        f"them in place would turn that one row into several. Proceeding "
        f"will instead create a new table, '{report.expand_table_name}', "
        f"with one row per CSV row. '{report.target_table}' itself will "
        f"not be changed."
    )


def carried_columns_text(report) -> str:
    """Describes which columns of target_table the new table would carry.
    Only meaningful when is_expand_offer(report) is true."""
    if not report.expand_carried_columns:
        return f"No columns are carried from '{report.target_table}'."
    cols = ", ".join(report.expand_carried_columns)
    return f"Carried from '{report.target_table}': {cols}"


def proceed_button_text(report) -> str:
    """Label for the affirmative button -- names the new table when this
    is an expansion offer, so the researcher sees exactly what Proceed
    will do."""
    if is_expand_offer(report):
        return f"Create '{report.expand_table_name}'"
    return "Proceed with merge"


def issue_tab_sources(report) -> list[tuple[str, list]]:
    """The (title, items) pairs _build_issue_tabs renders one tab per
    non-empty entry of. Centralised here so the tab wording is tested the
    same way as every other string in this module.

    The would_expand tab's title no longer calls these keys a problem --
    P1.5b turned the refusal they used to cause into the expansion offer
    described elsewhere in the dialog -- but the list itself (which CSV
    key values triggered it) is still worth showing.
    """
    return [
        ("CSV keys matching one target row more than once", report.would_expand),
        ("Target rows without a CSV match", report.unmatched_target_rows),
        ("CSV rows without a target match", report.unmatched_csv_rows),
        ("Duplicate keys (target)", report.duplicate_keys_target),
        ("Duplicate keys (CSV)", report.duplicate_keys_csv),
        # Advisory, not a refusal -- shown the same way (a tab with the
        # message as its one row) but never affects whether Proceed is
        # enabled.
        ("Decimal key warning", [report.float_key_warning] if report.float_key_warning else []),
    ]


# ---------------------------------------------------------------------------
# Layer B -- thin Qt glue.
# ---------------------------------------------------------------------------

class MergeReportDialog(QDialog):
    """
    Shows the result of a dry-run merge so the researcher can inspect
    what would happen before committing the changes.

    Layout:
        - Header and, for an expansion offer, an explanatory paragraph
          and a line naming the columns that would be carried.
        - Counts grid: total CSV rows, total target rows, matched rows,
          and four issue counts. Each issue count is colour-coded so
          problems jump out at a glance.
        - Tabbed list of the actual rows behind each count (see
          issue_tab_sources). Tabs for empty issues are hidden so the
          dialog stays compact.
        - Proceed / Cancel buttons. The accepted_merge attribute is True
          after exec() returns Accepted; False otherwise. Proceed is
          always enabled: merge_csv() always leaves something for
          confirm_merge() to commit now, whether that is the in-place
          join or the new-table offer. The decimal-key warning is
          advisory and never disables Proceed.
    """

    def __init__(self, report, parent=None):
        """
        Args:
            report: A MergeReport-like object -- see the module docstring
                    for its attributes.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Merge diagnostics")
        self.setMinimumSize(560, 460)
        self.accepted_merge: bool = False

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        expand_offer = is_expand_offer(report)

        # ── Header ────────────────────────────────────────────────────
        header = QLabel(header_text(report))
        header.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #4A90D9;"
        )
        layout.addWidget(header)

        if expand_offer:
            explain = QLabel(explain_text(report))
            explain.setWordWrap(True)
            explain.setStyleSheet("font-size: 11px;")
            layout.addWidget(explain)

            carried = QLabel(carried_columns_text(report))
            carried.setWordWrap(True)
            carried.setStyleSheet("font-size: 11px; font-style: italic;")
            layout.addWidget(carried)

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

        proceed_btn = QPushButton(proceed_button_text(report))
        proceed_btn.setDefault(True)
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
        non_empty = [(t, items) for t, items in issue_tab_sources(report) if items]
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
