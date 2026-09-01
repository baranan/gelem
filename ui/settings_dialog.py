"""
ui/settings_dialog.py

SettingsDialog -- the researcher-facing editor for the machine-tunable
values described in docs/architecture.md section 9.

The dialog edits through the controller only: AppController.get_settings_fields()
returns one plain-data SettingField per value, and AppController.apply_settings()
takes a PARTIAL mapping (field name -> value in the field's native unit) of just
the fields that moved. Everything below the controller -- SettingsGateway,
SettingsStore, GelemSettings -- is invisible here. This file must not import
settings/, and a guardrail in tests/test_settings.py fails if it does.

The file is in two layers:

  * Layer A -- module-level functions taking and returning plain data, with no
    Qt. All the arithmetic and all the wording live here, so the tests can
    exercise them without a QApplication.

  * Layer B -- class SettingsDialog(QDialog), thin glue that reads the spin
    boxes, calls Layer A, and talks to the controller.

Byte-valued settings (the two cache ceilings) are shown and edited in MiB,
because a researcher thinks in "500 MiB", not "524288000". The spin box holds a
MiB number; Layer A converts in both directions.
"""

from __future__ import annotations

import math

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


# ---------------------------------------------------------------------------
# Layer A -- plain-data logic. No Qt in this section.
# ---------------------------------------------------------------------------

# One MiB, the unit the two byte-valued settings are shown in.
_MIB = 1024 * 1024

# The three button labels on the restart-confirmation box. Named here so the
# dialog and the tests cannot drift.
BUTTON_SAVE_AND_QUIT = "Save and quit Gelem"
BUTTON_SAVE_AND_KEEP = "Save and keep working"
BUTTON_CANCEL = "Cancel"

# Heading of the box that reports settings the store adjusted after a save.
ADJUSTED_HEADING = "Gelem adjusted some settings"

# The single sentence shown when there are no editable settings (--fake-data).
NO_SETTINGS_SENTENCE = "Settings are not available in this mode."


def mib_spin_bounds(minimum_bytes: int, maximum_bytes: int) -> tuple[int, int]:
    """Return the inclusive MiB spin-box bounds for a byte range.

    The lower bound is rounded UP and the upper bound is rounded DOWN, so
    that every MiB value the spin box can hold converts back (via
    display_mib_to_bytes) to a byte count that is still inside
    [minimum_bytes, maximum_bytes]. Rounding the other way would let the
    lowest or highest spin position fall outside the real range.
    """
    lower_mib = math.ceil(minimum_bytes / _MIB)
    upper_mib = math.floor(maximum_bytes / _MIB)
    return (lower_mib, upper_mib)


def bytes_to_display_mib(
    value_bytes: int, spin_minimum: int, spin_maximum: int
) -> int:
    """Convert a byte count to the MiB number to show in the spin box.

    Rounds to the nearest MiB, then clamps into [spin_minimum, spin_maximum]
    so a stored value slightly outside the displayable range still lands on
    a valid spin position.
    """
    nearest_mib = round(value_bytes / _MIB)
    if nearest_mib < spin_minimum:
        return spin_minimum
    if nearest_mib > spin_maximum:
        return spin_maximum
    return nearest_mib


def display_mib_to_bytes(value_mib: int) -> int:
    """Convert a MiB spin-box number back to a byte count."""
    return value_mib * _MIB


def changed_values(initial: dict, current: dict) -> dict:
    """Return only the entries of `current` whose value differs from `initial`.

    Both mappings are field name -> submitted value in the field's NATIVE
    unit (bytes for the ceilings, not MiB). The result is the partial
    mapping handed to AppController.apply_settings().
    """
    result: dict = {}
    for name, current_value in current.items():
        # A field absent from `initial`, or present with a different value,
        # counts as changed.
        if name not in initial or initial[name] != current_value:
            result[name] = current_value
    return result


def restart_required_names(fields, changed: dict) -> list[str]:
    """Return the labels of the changed fields whose restart_required is True.

    The flag is read straight off each field -- which fields need a restart
    is never hardcoded here. Order follows `fields`.
    """
    labels: list[str] = []
    for field in fields:
        if field.name in changed and field.restart_required:
            labels.append(field.label)
    return labels


def _join_with_and(labels: list[str]) -> str:
    """Join labels as plain English: "A", "A and B", "A, B and C"."""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def confirmation_text(fields, changed: dict) -> str:
    """Build the plain-English body of the restart-confirmation box.

    It always names which changed settings need a restart. It ADDS the
    picture-regeneration warning only when a pixels-unit field (the
    thumbnail or preview size) is among the changed names -- identified by
    field.unit == "pixels", never by matching the field name.
    """
    sentences: list[str] = []

    # Which changed settings only take effect on restart. Gelem never
    # restarts itself, so the wording tells the researcher to quit and
    # start it again rather than implying an automatic restart.
    restart_labels = restart_required_names(fields, changed)
    if restart_labels:
        sentences.append(
            f"Changing {_join_with_and(restart_labels)} takes effect the "
            f"next time Gelem starts. Gelem does not restart itself, so "
            f"quit and start it again."
        )

    # A changed thumbnail or preview size invalidates every derived picture:
    # the resolution is part of the artifact key, so the old files can no
    # longer be found and every tile regenerates from source.
    pixels_changed = [
        field
        for field in fields
        if field.name in changed and field.unit == "pixels"
    ]
    if pixels_changed:
        sentences.append(
            "Every thumbnail and preview already generated becomes "
            "unreachable and will be regenerated from the source files."
        )

    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Layer B -- the Qt dialog. Thin glue over Layer A and the controller.
# ---------------------------------------------------------------------------

# Spin-box suffixes by unit. "bytes" is handled separately (shown in MiB);
# the suffix is decided from the unit string, never from the field name.
_SUFFIX_BY_UNIT = {
    "pixels": " px",
    "count": "",
}


class SettingsDialog(QDialog):
    """Editor for the machine-tunable settings.

    Construct with the controller and a parent; call exec(). On OK the
    dialog submits only the fields the researcher actually changed.
    """

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self._controller = controller
        self.setWindowTitle("Settings")

        # Ask the controller once for the editable fields.
        fields = controller.get_settings_fields()

        # --fake-data mode has no settings store, so the list is empty.
        # Show one sentence and a Close button, and build nothing else.
        self._empty = not fields
        if self._empty:
            self._build_empty()
            return

        # name -> the SettingField, for lookups on OK.
        self._fields = list(fields)
        # name -> its QSpinBox.
        self._spin_boxes: dict = {}
        # name -> its starting value in NATIVE units, recorded at build time.
        self._initial_native: dict = {}

        self._build_fields()

    # ── Building ─────────────────────────────────────────────────────────

    def _build_empty(self) -> None:
        """The no-settings layout: one sentence and a Close button."""
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(NO_SETTINGS_SENTENCE))

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)

        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

    def _build_fields(self) -> None:
        """One row per field, in the order the controller gave them."""
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        for field in self._fields:
            # Label and spin box on one line.
            row = QHBoxLayout()
            row.addWidget(QLabel(field.label))
            row.addStretch()

            spin = QSpinBox()

            if field.unit == "bytes":
                # Shown and edited in MiB.
                lower_mib, upper_mib = mib_spin_bounds(
                    field.minimum, field.maximum
                )
                spin.setRange(lower_mib, upper_mib)
                spin.setSuffix(" MiB")
                start_mib = bytes_to_display_mib(
                    field.current_value, lower_mib, upper_mib
                )
                spin.setValue(start_mib)
                # Record the starting value in native units, but as the
                # byte count the DISPLAYED MiB number stands for -- not the
                # raw stored value. The spin box can only ever submit whole
                # MiB, so if the stored value were not an exact MiB multiple
                # an untouched field would otherwise read as "changed" and
                # trigger an unwanted eviction or sweep.
                starting_native = display_mib_to_bytes(start_mib)
            else:
                # Shown in the native unit; suffix from the unit string.
                spin.setRange(field.minimum, field.maximum)
                spin.setSuffix(_SUFFIX_BY_UNIT.get(field.unit, ""))
                spin.setValue(field.current_value)
                starting_native = field.current_value

            row.addWidget(spin)
            layout.addLayout(row)

            # Help text below the row, word-wrapped.
            help_label = QLabel(field.help_text)
            help_label.setWordWrap(True)
            layout.addWidget(help_label)

            # Record the spin box and the starting value in native units.
            self._spin_boxes[field.name] = spin
            self._initial_native[field.name] = starting_native

        # OK / Cancel.
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ── Reading the spin boxes ───────────────────────────────────────────

    def _current_native(self) -> dict:
        """Field name -> current spin-box value, in NATIVE units."""
        native: dict = {}
        for field in self._fields:
            spin = self._spin_boxes[field.name]
            if field.unit == "bytes":
                native[field.name] = display_mib_to_bytes(spin.value())
            else:
                native[field.name] = spin.value()
        return native

    def _refresh_from_controller(self) -> None:
        """Re-read every field from the controller and reset the spin boxes.

        The stored value can differ from what was typed -- the store clamps
        out-of-range numbers and applies the preview-not-smaller-than-thumbnail
        rule -- so the researcher must see what was actually kept.
        """
        for field in self._controller.get_settings_fields():
            spin = self._spin_boxes.get(field.name)
            if spin is None:
                continue
            if field.unit == "bytes":
                lower_mib, upper_mib = mib_spin_bounds(
                    field.minimum, field.maximum
                )
                spin.setValue(
                    bytes_to_display_mib(
                        field.current_value, lower_mib, upper_mib
                    )
                )
            else:
                spin.setValue(field.current_value)

    # ── OK ───────────────────────────────────────────────────────────────

    def _on_ok(self) -> None:
        """Submit only the changed fields, confirming a restart if needed."""
        changed = changed_values(self._initial_native, self._current_native())

        # Nothing moved -- close with no controller call at all.
        if not changed:
            self.accept()
            return

        # If any changed field needs a restart, confirm first. The box has
        # exactly three buttons; Cancel returns to the dialog untouched.
        quit_after_save = False
        if restart_required_names(self._fields, changed):
            choice = self._ask_restart(changed)
            if choice == "cancel":
                return
            quit_after_save = choice == "quit"

        # Persist just the changed entries, then show what the store kept.
        messages = self._controller.apply_settings(changed)
        self._refresh_from_controller()

        # The returned strings are not all errors -- one may report cached
        # picture files a sweep deleted. Show them as plain messages, and
        # show them BEFORE quitting if quit was chosen.
        if messages:
            box = QMessageBox(self)
            box.setWindowTitle(ADJUSTED_HEADING)
            box.setText("\n".join(messages))
            box.exec()

        if quit_after_save:
            QApplication.instance().quit()

        self.accept()

    def _ask_restart(self, changed: dict) -> str:
        """Show the three-button restart box. Returns "quit", "keep" or
        "cancel"."""
        box = QMessageBox(self)
        box.setWindowTitle("Takes effect after restart")
        box.setText(confirmation_text(self._fields, changed))

        quit_button = box.addButton(
            BUTTON_SAVE_AND_QUIT, QMessageBox.ButtonRole.AcceptRole
        )
        keep_button = box.addButton(
            BUTTON_SAVE_AND_KEEP, QMessageBox.ButtonRole.AcceptRole
        )
        cancel_button = box.addButton(
            BUTTON_CANCEL, QMessageBox.ButtonRole.RejectRole
        )
        # "Save and quit Gelem" is added first and would otherwise be the
        # default that Enter activates -- make the least destructive choice
        # the default instead.
        box.setDefaultButton(keep_button)
        box.exec()

        clicked = box.clickedButton()
        if clicked is quit_button:
            return "quit"
        if clicked is keep_button:
            return "keep"
        return "cancel"
