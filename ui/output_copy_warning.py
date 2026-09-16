"""
ui/output_copy_warning.py

The researcher-facing warning shown before a large copy-on-save
(docs/architecture.md section 2, P1.9b-2). AppController.plan_output_copy()
(P1.9b-1) already decides WHAT would be copied; this file only decides
whether to ask about it and what to say.

Two layers, the same split ui/settings_dialog.py and
ui/merge_report_dialog.py use:

  * Layer A -- module-level functions taking plain numbers and returning
    plain strings/bools, with no Qt. All the wording and arithmetic live
    here, so the tests can exercise them without a QApplication.

  * Layer B -- a thin QMessageBox-based function that shows Layer A's
    wording and reports which button the researcher clicked.
"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox


# ---------------------------------------------------------------------------
# Layer A -- wording and arithmetic, no Qt.
# ---------------------------------------------------------------------------

_MB = 1024 * 1024
_GB = 1024 * _MB


def format_size_mb_gb(total_bytes: int) -> str:
    """Human-readable size, one decimal place, MB below 1 GB and GB at or
    above it.

    The switch is decided on the ROUNDED megabyte figure, not the raw byte
    count: a value that rounds to "1024.0 MB" is shown as "1.0 GB" instead,
    so the researcher never sees a four-digit MB number that looks like it
    should have already become a GB one.
    """
    megabytes = total_bytes / _MB
    if round(megabytes, 1) >= 1024.0:
        return f"{total_bytes / _GB:.1f} GB"
    return f"{megabytes:.1f} MB"


def should_warn(total_bytes: int, threshold_bytes: int) -> bool:
    """True when a copy this large should be called out before it runs.

    Strictly greater than the threshold -- a copy exactly at the
    threshold is still the researcher's configured "fine, don't ask"
    boundary.
    """
    return total_bytes > threshold_bytes


def warning_message(file_count: int, total_bytes: int) -> str:
    """The confirmation box's body text: what will be copied, its size,
    and the warning that the window will be unresponsive while it runs
    (save_project() does the copy synchronously on the main thread --
    see docs/architecture.md section 2 and controller.py's save_project()
    comment)."""
    noun = "file" if file_count == 1 else "files"
    size_text = format_size_mb_gb(total_bytes)
    return (
        f"Saving will copy {file_count} output {noun} ({size_text}) into "
        f"the project folder. Gelem's window will be unresponsive until "
        f"the copy finishes. Continue?"
    )


# ---------------------------------------------------------------------------
# Layer B -- thin Qt glue.
# ---------------------------------------------------------------------------

def confirm_output_copy(parent, file_count: int, total_bytes: int) -> bool:
    """Shows the Continue/Cancel confirmation box. Returns True only if
    the researcher clicked Continue; False for Cancel or for the box
    being dismissed any other way.

    Cancel is the default button -- the least disruptive choice, the same
    convention ui/settings_dialog.py's restart box uses.
    """
    box = QMessageBox(parent)
    box.setWindowTitle("Large output copy")
    box.setText(warning_message(file_count, total_bytes))

    continue_button = box.addButton(
        "Continue", QMessageBox.ButtonRole.AcceptRole
    )
    cancel_button = box.addButton(
        "Cancel", QMessageBox.ButtonRole.RejectRole
    )
    box.setDefaultButton(cancel_button)
    box.exec()

    return box.clickedButton() is continue_button
