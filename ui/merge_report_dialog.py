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
    QLabel, QLineEdit, QPushButton, QTabWidget, QListWidget,
    QListWidgetItem, QWidget, QFrame
)
from PySide6.QtCore import Qt

# table-name-validation, round 4: both reused exactly as they are, never
# a second comparison, from the one shared Qt-free home (table_names.py)
# -- round 3 reached into controller.py for resolve_table_name and into
# ui/parameter_dialog.py for validate_new_table_name; now every consumer
# (this module, ui/parameter_dialog.py's operator parameter form,
# controller.py's own create_table_from_df naming logic) imports both
# from the same place, so there is exactly one place that decides "is
# this name taken" and this dialog's red text can never disagree with
# either the operator dialog's or the backstop
# Dataset.create_table_from_df / Dataset.confirm_merge itself applies.
from table_names import resolve_table_name, validate_new_table_name


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


def explain_text(report, resolved_name: str | None = None) -> str:
    """The paragraph under the header. Empty string means no paragraph is
    shown -- the ordinary in-place merge doesn't need one; its counts grid
    and issue tabs already say everything.

    resolved_name (table-name-validation round 4) names the new table
    when given, overriding report.expand_table_name -- Layer B no longer
    writes the resolved default (or the researcher's choice) onto
    `report` at all (see MergeReportDialog), so this is how the
    paragraph agrees with what the name box actually shows. None (every
    caller before this round) falls back to report.expand_table_name,
    unchanged.
    """
    if not is_expand_offer(report):
        return ""
    name = resolved_name if resolved_name is not None else report.expand_table_name
    return (
        f"{len(report.would_expand)} key value(s) in the CSV each match "
        f"one row of '{report.target_table}' more than once, so joining "
        f"them in place would turn that one row into several. Proceeding "
        f"will instead create a new table, '{name}', "
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


def proceed_button_text(report, chosen_name: str | None = None) -> str:
    """Label for the affirmative button -- names the new table when this
    is an expansion offer, so the researcher sees exactly what Proceed
    will do.

    chosen_name (table-name-validation round 4) overrides
    report.expand_table_name for an expand offer, the same override
    explain_text takes and for the same reason: Layer B no longer writes
    to `report`, so this is how the button's label follows the name box's
    CURRENT text on every edit rather than merge_csv()'s original
    suggestion. None (every caller before this round) falls back to
    report.expand_table_name, unchanged.

    A blank or whitespace-only name (round 6) falls back to the plain
    "Create the new table" instead of quoting an empty string -- Proceed
    stays disabled either way (expand_table_name_message refuses a blank
    name on its own), so this only fixes what the disabled button's
    label reads, not whether it is clickable.
    """
    if is_expand_offer(report):
        name = chosen_name if chosen_name is not None else report.expand_table_name
        if name is None or name.strip() == "":
            return "Create the new table"
        return f"Create '{name}'"
    return "Proceed with merge"


def resolved_default_expand_table_name(report, existing) -> str:
    """The expand-name box's opening value (table-name-validation,
    round 3): merge_csv()'s own suggestion (report.expand_table_name,
    always the raw f"{target}_expanded" -- models/dataset.py's
    _default_expand_table_name, never resolved there, since
    Dataset.confirm_merge()'s own refusal on a taken name is the
    documented backstop and must stay reachable) resolved against
    `existing` through the exact same resolve_table_name
    ui/parameter_dialog.py's operator-parameter dialog uses (both from
    table_names.py as of round 4), so the box never opens already
    colliding.

    Only meaningful when is_expand_offer(report) is true -- the caller
    is expected to check that first, the same way every other Layer A
    function here that reads an expand_* field does.
    """
    return resolve_table_name(report.expand_table_name, existing)


def expand_table_name_message(name: str, existing) -> str | None:
    """The researcher-facing message for the merge dialog's editable
    expand-table-name field's CURRENT text, or None if it is fine to
    proceed with.

    A blank or whitespace-only name is refused HERE, unlike
    validate_new_table_name (which defers that case to
    ui/parameter_dialog.py's own required-field mechanism, a QMessageBox
    shown on OK): this dialog has no equivalent required-field check
    elsewhere, so without this a blank field would let Proceed try to
    create a table literally named "".

    Every other case is validate_new_table_name itself, reused exactly
    as it is -- never a second comparison. See the module's import for
    why that keeps this dialog's red text unable to disagree with either
    the operator parameter dialog's or Dataset's own backstop refusal.
    """
    if name.strip() == "":
        return "Please enter a name for the new table."
    return validate_new_table_name(name, existing)


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
        - Header and, for an expansion offer, an explanatory paragraph,
          a line naming the columns that would be carried, and an
          editable "New table name" field (table-name-validation round
          3) defaulted to the resolved, currently-free name
          (resolved_default_expand_table_name). Editing it live-checks
          against existing_table_names (expand_table_name_message) and
          shows a red message under the field on a taken or blank name.
        - Counts grid: total CSV rows, total target rows, matched rows,
          and four issue counts. Each issue count is colour-coded so
          problems jump out at a glance.
        - Tabbed list of the actual rows behind each count (see
          issue_tab_sources). Tabs for empty issues are hidden so the
          dialog stays compact.
        - Proceed / Cancel buttons. The accepted_merge attribute is True
          after exec() returns Accepted; False otherwise. For the
          ORDINARY in-place merge, Proceed is always enabled: merge_csv()
          always leaves something for confirm_merge() to commit, and the
          decimal-key warning is advisory and never disables it. For an
          EXPANSION offer, Proceed is additionally gated on the name
          field reading as neither taken nor blank -- a courtesy only:
          Dataset.confirm_merge() keeps its own refusal on a taken name
          as the real guarantee, since another run can still claim a
          name while this dialog stays open.
    """

    def __init__(self, report, parent=None, existing_table_names=()):
        """
        Args:
            report: A MergeReport-like object -- see the module docstring
                    for its attributes.
            parent: Parent widget.
            existing_table_names: The project's current table names (fix
                round, table-name-validation round 3) -- handed straight
                in by whoever builds this dialog (AppController.get_table_names(),
                the same route ui/parameter_dialog.py's operator parameter
                form uses), never fetched here: this widget must never
                reach Dataset. Only consulted when the merge is an
                expansion offer -- an ordinary in-place merge gets no
                name box at all, since the target table's name is not in
                question.
        """
        super().__init__(parent)
        self.setWindowTitle("Merge diagnostics")
        self.setMinimumSize(560, 460)
        self.accepted_merge: bool = False
        self._report = report
        self._existing_table_names = tuple(existing_table_names)
        # Filled in below only for an expansion offer.
        self._name_box = None
        self._name_error_label = None
        self._proceed_btn = None

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
            # table-name-validation, round 4: resolve merge_csv()'s raw
            # suggestion against the project's current tables BEFORE
            # building anything below that would otherwise read
            # report.expand_table_name -- the explanatory paragraph, the
            # name box's own opening text and the Proceed button label
            # all then agree with each other from the first paint. Kept
            # as a LOCAL variable, never written back onto `report`:
            # this dialog no longer mutates the report it was given at
            # all (round 3 did; a reviewer asked for the researcher's
            # choice to travel as an explicit argument instead -- see
            # chosen_expand_table_name below and AppController.confirm_merge()).
            # Dataset.confirm_merge()'s own refusal (models/dataset.py)
            # is untouched and stays the backstop for whatever name is
            # actually passed to it -- another run can still claim a
            # name while this dialog is open.
            resolved_name = resolved_default_expand_table_name(
                report, self._existing_table_names
            )

            explain = QLabel(explain_text(report, resolved_name))
            explain.setWordWrap(True)
            explain.setStyleSheet("font-size: 11px;")
            layout.addWidget(explain)

            carried = QLabel(carried_columns_text(report))
            carried.setWordWrap(True)
            carried.setStyleSheet("font-size: 11px; font-style: italic;")
            layout.addWidget(carried)

            # ── New table name ───────────────────────────────────────
            layout.addWidget(QLabel("New table name"))
            self._name_box = QLineEdit(resolved_name)
            layout.addWidget(self._name_box)
            # Red, hidden-until-needed -- the same treatment and colour
            # ui/parameter_dialog.py's NewTableNameParameter field uses.
            self._name_error_label = QLabel()
            self._name_error_label.setWordWrap(True)
            self._name_error_label.setStyleSheet("color: #B00020;")
            self._name_error_label.setVisible(False)
            layout.addWidget(self._name_error_label)
            self._name_box.textChanged.connect(self._on_name_changed)

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

        proceed_btn = QPushButton(
            proceed_button_text(report, resolved_name if expand_offer else None)
        )
        proceed_btn.setDefault(True)
        proceed_btn.clicked.connect(self._on_proceed)
        btn_row.addWidget(proceed_btn)
        self._proceed_btn = proceed_btn

        layout.addLayout(btn_row)

        # Render the box's opening state through the same path every
        # later edit uses, rather than trusting the widgets built above
        # to already agree with it -- one code path for "what does the
        # name field's current text mean", always.
        if expand_offer:
            self._on_name_changed()

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

    def _on_name_changed(self, *_args) -> None:
        """The expand-table-name field changed (or the dialog just
        finished building) -- re-render the red message and the Proceed
        button/label from the box's CURRENT text.

        table-name-validation round 4: no longer writes anywhere onto
        `self._report` -- this dialog never mutates the MergeReport it
        was given, full stop. The researcher's choice lives only in this
        dialog's own widget state; ui/main_window.py reads it back
        through chosen_expand_table_name (below) once the dialog is
        accepted, and passes it as an explicit argument to
        AppController.confirm_merge() rather than the report carrying it.

        Only ever connected for an expansion offer -- see __init__.
        """
        text = self._name_box.text()
        message = expand_table_name_message(text, self._existing_table_names)
        if message is None:
            self._name_error_label.clear()
            self._name_error_label.setVisible(False)
        else:
            self._name_error_label.setText(message)
            self._name_error_label.setVisible(True)
        self._proceed_btn.setEnabled(message is None)
        self._proceed_btn.setText(
            proceed_button_text(self._report, text.strip())
        )

    @property
    def chosen_expand_table_name(self) -> str | None:
        """The researcher's current choice for the expansion offer's new
        table name, read live off the name box -- None when this merge
        is not an expansion offer, since no box exists (table-name-
        validation round 4).

        ui/main_window.py reads this after the dialog is accepted and
        passes it straight to AppController.confirm_merge()'s
        expand_table_name argument. Computed on every access rather than
        cached, so there is exactly one place ("what is in the box right
        now, stripped") this can ever disagree with -- the same
        stripping collect_parameters and validate_new_table_name apply
        elsewhere, so the name Dataset actually receives matches what
        was validated here.
        """
        if self._name_box is None:
            return None
        return self._name_box.text().strip()

    def _on_proceed(self) -> None:
        """Marks the dialog as accepted and closes it."""
        self.accepted_merge = True
        self.accept()
