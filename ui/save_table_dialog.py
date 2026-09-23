"""
ui/save_table_dialog.py

A simple dialog asking the researcher to name a new table
when saving a filtered subset as a permanent table.

Student A is responsible for implementing a more polished version.

CC-29: the name field gets the same live, red, in-place collision
check every other new-table-name field has (ui/parameter_dialog.py's
NewTableNameParameter field, ui/merge_report_dialog.py's expand-name
field) -- reusing table_names.validate_new_table_name exactly, never a
second comparison. This is only ever a courtesy: another run can still
claim the name while this dialog stays open, so Dataset's own refusal
at store time (create_table_from_rows, through
AppController.save_filtered_as_table) remains the actual guarantee.
"""

from __future__ import annotations
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton
)

from table_names import validate_new_table_name


class SaveTableDialog(QDialog):
    """
    Asks the researcher for a name for the new table.
    After exec() returns Accepted, read self.table_name.
    """

    def __init__(self, n_rows: int, parent=None, existing_table_names=()):
        """
        Args:
            n_rows: Number of rows that will be saved.
            parent: Parent widget.
            existing_table_names: The project's current table names
                (CC-29) -- handed straight in by whoever builds this
                dialog (AppController.get_table_names(), the same route
                ui/parameter_dialog.py and ui/merge_report_dialog.py
                use), never fetched here: this widget must never reach
                Dataset. Defaults to (), so a caller that does not pass
                it gets a field that never flags a collision inline --
                the store-time refusal in Dataset is unaffected either
                way.
        """
        super().__init__(parent)
        self.setWindowTitle("Save filtered set as new table")
        self.setMinimumWidth(320)
        self.table_name: str = ""
        self._existing_table_names = tuple(existing_table_names)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel(
            f"Save {n_rows} currently visible rows as a new table."
        ))
        layout.addWidget(QLabel("Table name:"))

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. positive_condition")
        layout.addWidget(self._name_edit)

        # Red, hidden-until-needed -- the same treatment and colour
        # every other new-table-name field uses.
        self._name_error_label = QLabel()
        self._name_error_label.setWordWrap(True)
        self._name_error_label.setStyleSheet("color: #B00020;")
        self._name_error_label.setVisible(False)
        layout.addWidget(self._name_error_label)
        self._name_edit.textChanged.connect(self._on_name_changed)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setDefault(True)
        self._save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(self._save_btn)

        layout.addLayout(btn_layout)

        # Render the opening state (an already-taken default text is not
        # possible here -- the field starts blank -- but this keeps one
        # code path for "what does the current text mean", same as
        # ui/merge_report_dialog.py).
        self._on_name_changed()

    def _on_name_changed(self, *_args) -> None:
        """Live "this name is already taken" feedback (CC-29), reusing
        validate_new_table_name exactly -- never a second comparison.
        A blank field is not flagged here (validate_new_table_name
        returns None for it); Save simply does nothing on a blank name,
        the same behaviour this dialog always had.
        """
        message = validate_new_table_name(
            self._name_edit.text(), self._existing_table_names
        )
        if message is None:
            self._name_error_label.clear()
            self._name_error_label.setVisible(False)
        else:
            self._name_error_label.setText(message)
            self._name_error_label.setVisible(True)
        self._save_btn.setEnabled(message is None)

    def _on_save(self) -> None:
        # CC-30: only .strip(), matching ui/parameter_dialog.py and
        # ui/merge_report_dialog.py -- no space-to-underscore rewrite. The
        # live check above validates this exact stripped text, so the
        # rewrite used to let a name that read as free (e.g. "my table")
        # submit as a DIFFERENT, possibly taken, name ("my_table"); Save
        # is disabled while validate_new_table_name flags this text, so
        # deleting the rewrite is what makes that guarantee hold here too.
        name = self._name_edit.text().strip()
        if not name:
            return
        self.table_name = name
        self.accept()