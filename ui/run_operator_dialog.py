"""
ui/run_operator_dialog.py

A dialog that appears whenever the researcher runs an operator.
Lets the researcher choose the scope of the operation:
    - Selected rows only
    - Currently visible (filtered) rows
    - All rows in the active table

After choosing scope, the dialog closes and MainWindow shows the
operator's own parameter dialog (if it has one) before running.

Student A is responsible for implementing the full version of this
dialog. The current version is a functional placeholder.
"""

from __future__ import annotations
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QRadioButton, QPushButton, QButtonGroup,
    QGroupBox, QFrame
)
from PySide6.QtCore import Qt


class RunOperatorDialog(QDialog):
    """
    Scope selection dialog shown before running any operator.

    After exec() returns Accepted, read self.chosen_row_ids to get
    the list of row_ids the operator should run on.
    """

    def __init__(
        self,
        operator_name: str,
        selected_ids: list[str],
        visible_ids: list[str],
        all_ids: list[str],
        parent=None,
    ):
        """
        Creates the dialog.

        Args:
            operator_name: Name of the operator being run.
            selected_ids:  Currently selected row_ids.
            visible_ids:   Currently visible (filtered) row_ids.
            all_ids:       All row_ids in the active table.
            parent:        Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle(f"Run — {operator_name}")
        self.setMinimumWidth(380)
        self.chosen_row_ids: list[str] = []

        self._selected_ids = selected_ids
        self._visible_ids  = visible_ids
        self._all_ids      = all_ids

        self._build_ui(operator_name)

    def _build_ui(self, operator_name: str) -> None:
        """Builds the dialog layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header.
        header = QLabel(f"<b>{operator_name}</b>")
        header.setStyleSheet("font-size: 14px;")
        layout.addWidget(header)

        # Divider.
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #CCCCCC;")
        layout.addWidget(line)

        # Scope group.
        scope_group = QGroupBox("Run on:")
        scope_layout = QVBoxLayout(scope_group)
        self._button_group = QButtonGroup(self)

        n_selected = len(self._selected_ids)
        n_visible  = len(self._visible_ids)
        n_all      = len(self._all_ids)

        # Row counts per button id, so Run can check the chosen scope.
        self._counts = {0: n_selected, 1: n_visible, 2: n_all}

        # Selected rows option.
        self._radio_selected = QRadioButton(
            f"Selected rows ({n_selected})"
        )
        self._radio_selected.setEnabled(n_selected > 0)
        self._button_group.addButton(self._radio_selected, 0)
        scope_layout.addWidget(self._radio_selected)

        # Visible rows option.
        self._radio_visible = QRadioButton(
            f"Visible rows — current filter ({n_visible})"
        )
        self._radio_visible.setEnabled(n_visible > 0)
        self._button_group.addButton(self._radio_visible, 1)
        scope_layout.addWidget(self._radio_visible)

        # All rows option.
        self._radio_all = QRadioButton(
            f"All rows in active table ({n_all})"
        )
        self._radio_all.setEnabled(n_all > 0)
        self._button_group.addButton(self._radio_all, 2)
        scope_layout.addWidget(self._radio_all)

        layout.addWidget(scope_group)

        # Shown when the table is empty, so Run being greyed out is clear.
        self._warning = QLabel(
            "⚠ No data to run on. Upload a CSV, images or videos first."
        )
        self._warning.setWordWrap(True)
        self._warning.setStyleSheet(
            "color: #B25000; font-size: 11px; padding: 2px;"
        )
        self._warning.setVisible(n_all == 0)
        layout.addWidget(self._warning)

        # Default to the first scope that has rows.
        if n_selected > 0:
            self._radio_selected.setChecked(True)
        elif n_visible > 0 and n_visible < n_all:
            self._radio_visible.setChecked(True)
        elif n_all > 0:
            self._radio_all.setChecked(True)
        # else: nothing is checked and Run stays disabled.

        # Buttons.
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self._run_btn = QPushButton("Run")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._on_run)
        btn_layout.addWidget(self._run_btn)

        layout.addLayout(btn_layout)

        # Disable Run unless the chosen scope has rows.
        self._button_group.buttonToggled.connect(
            lambda *_: self._update_run_enabled()
        )
        self._update_run_enabled()

    def _update_run_enabled(self) -> None:
        """Enable Run only when the chosen scope has rows."""
        checked_id = self._button_group.checkedId()
        has_rows   = checked_id != -1 and self._counts.get(checked_id, 0) > 0
        self._run_btn.setEnabled(has_rows)

    def _on_run(self) -> None:
        """Stores the chosen row_ids and accepts the dialog."""
        checked_id = self._button_group.checkedId()
        if checked_id == 0:
            self.chosen_row_ids = self._selected_ids
        elif checked_id == 1:
            self.chosen_row_ids = self._visible_ids
        else:
            self.chosen_row_ids = self._all_ids
        self.accept()