"""
ui/save_table_dialog.py

A simple dialog asking the researcher to name a new table
when saving a filtered subset as a permanent table.

Student A is responsible for implementing a more polished version.
"""

from __future__ import annotations
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton
)


class SaveTableDialog(QDialog):
    """
    Asks the researcher for a name for the new table.
    After exec() returns Accepted, read self.table_name.
    """

    def __init__(self, n_rows: int, parent=None):
        """
        Args:
            n_rows: Number of rows that will be saved.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Save filtered set as new table")
        self.setMinimumWidth(320)
        self.table_name: str = ""

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel(
            f"Save {n_rows} currently visible rows as a new table."
        ))
        layout.addWidget(QLabel("Table name:"))

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. positive_condition")
        layout.addWidget(self._name_edit)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

    def _on_save(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            return
        # Replace spaces with underscores for safety.
        self.table_name = name.replace(" ", "_")
        self.accept()