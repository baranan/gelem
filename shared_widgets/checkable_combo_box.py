"""
shared_widgets/checkable_combo_box.py

A QComboBox whose dropdown items are checkable, so several options can
be selected at once while the control keeps the exact look of a normal
combo box.

This widget knows nothing about the application's data: callers pass in
plain string labels and read back the checked labels. It is therefore
safe to reuse anywhere a multi-select combo is needed.
"""

from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox, QStyle, QStyleOptionComboBox, QStylePainter,
)


class CheckableComboBox(QComboBox):
    """
    A combo box that lets the user check any number of its items.

    The closed control shows a comma-separated summary of the checked
    items, or a placeholder when nothing is checked. The popup stays
    open while items are toggled and closes when the user clicks away.
    """

    # Emitted with the list of checked labels whenever the selection
    # changes through user interaction.
    selection_changed = Signal(list)

    # Emitted just before the popup opens, so callers can repopulate the
    # items first (mirrors QMenu.aboutToShow).
    about_to_show = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._placeholder = ""
        self.setModel(QStandardItemModel(self))

        # Toggle an item's check state when its row is pressed; hidePopup
        # is then suppressed once so the dropdown stays open.
        self.view().pressed.connect(self._on_item_pressed)
        self.model().itemChanged.connect(self._on_model_changed)
        self._keep_open = False

    # ── Public API ────────────────────────────────────────────────────

    def set_placeholder(self, text: str) -> None:
        """Sets the text shown when no item is checked."""
        self._placeholder = text
        self.update()

    def set_items(self, labels: list[str], checked: list[str] | None = None) -> None:
        """
        Replaces the dropdown contents.

        Args:
            labels:  Ordered list of option labels.
            checked: Labels that should start checked (default: none).

        Repopulating does not emit selection_changed.
        """
        checked_set = set(checked or [])
        self.model().blockSignals(True)
        self.model().clear()
        for label in labels:
            item = QStandardItem(label)
            item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
            )
            item.setData(
                Qt.CheckState.Checked if label in checked_set
                else Qt.CheckState.Unchecked,
                Qt.ItemDataRole.CheckStateRole,
            )
            self.model().appendRow(item)
        self.model().blockSignals(False)
        self.update()

    def checked_items(self) -> list[str]:
        """Returns the labels of all checked items, in display order."""
        result = []
        for row in range(self.model().rowCount()):
            item = self.model().item(row)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                result.append(item.text())
        return result

    # ── Internal behaviour ────────────────────────────────────────────

    def showPopup(self) -> None:
        self.about_to_show.emit()
        super().showPopup()

    def hidePopup(self) -> None:
        # Keep the popup open right after a toggle so several items can be
        # checked in one interaction.
        if self._keep_open:
            self._keep_open = False
            return
        super().hidePopup()

    def _on_item_pressed(self, index) -> None:
        item = self.model().itemFromIndex(index)
        if item is None:
            return
        new_state = (
            Qt.CheckState.Unchecked
            if item.checkState() == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )
        self._keep_open = True
        item.setCheckState(new_state)

    def _on_model_changed(self, _item) -> None:
        self.update()
        self.selection_changed.emit(self.checked_items())

    def _display_text(self) -> str:
        checked = self.checked_items()
        return ", ".join(checked) if checked else self._placeholder

    def paintEvent(self, event) -> None:
        # Paint as a standard combo box but with our multi-select summary
        # as the visible text, so it matches sibling QComboBoxes exactly.
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = self._display_text()
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)
