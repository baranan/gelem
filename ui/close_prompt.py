"""
ui/close_prompt.py

Layer A for the close-time Save/Discard/Cancel decision (unsaved-work
item). Given whether the project has unsaved changes and whether an
operator run is still live, decide_close_prompt() returns which of three
prompts MainWindow.closeEvent should show, and that prompt's
researcher-facing wording: title, body text, the buttons in display
order, and which one is the default.

This module is plain data only -- no Qt import anywhere in it -- so the
decision and the wording can be tested without a QApplication, the same
split ui/settings_dialog.py uses for its Layer A. Layer B, the QMessageBox
that actually shows one of these prompts and reads back which button was
clicked, lives in ui/main_window.py: MainWindow already imports Qt for
everything else closeEvent does (cancelling live runs, reusing the
existing save flow), so there is nothing Layer B needs that a second file
would gain.
"""

from __future__ import annotations

from dataclasses import dataclass

# The three prompts decide_close_prompt() can return, named by what
# closeEvent should do with the answer. CLOSE_WITHOUT_ASKING means: close
# immediately, no dialog shown at all.
CLOSE_WITHOUT_ASKING = "close_without_asking"
LIVE_RUN = "live_run"
UNSAVED = "unsaved"

# Button labels, named here so MainWindow and the tests cannot drift.
BUTTON_CLOSE_ANYWAY = "Close anyway"
BUTTON_SAVE = "Save"
BUTTON_DISCARD = "Discard"
BUTTON_CANCEL = "Cancel"

LIVE_RUN_TITLE = "An operator is still running"
LIVE_RUN_TEXT = (
    "An operator is still running. Closing Gelem now stops it and loses "
    "any work it has not finished."
)

UNSAVED_TITLE = "Unsaved changes"
UNSAVED_TEXT = (
    "This project has unsaved changes. Do you want to save before "
    "closing?"
)


@dataclass(frozen=True)
class ClosePrompt:
    """What MainWindow.closeEvent should show, as plain data.

    kind is one of CLOSE_WITHOUT_ASKING, LIVE_RUN, UNSAVED -- MainWindow
    switches on it to decide what each button leads to. buttons is in
    display order; default_button names one of its entries (or is None
    for CLOSE_WITHOUT_ASKING, which shows no box at all).
    """

    kind: str
    title: str
    text: str
    buttons: tuple[str, ...]
    default_button: str | None


def decide_close_prompt(has_unsaved: bool, run_live: bool) -> ClosePrompt:
    """The close-time decision (unsaved-work item, decision 3).

    Checked in order, and a live run always wins regardless of
    has_unsaved: closing while a run is live stops it and loses whatever
    it has not finished, a risk that exists whether or not Dataset
    already holds saved-vs-unsaved data, and CLAUDE.md's "Long-running
    work" rule already treats a live run as the more urgent condition
    (is_save_blocked() refuses even an explicit Save while one is live).
    Only once no run is live does has_unsaved decide between asking to
    save and closing outright.
    """
    if run_live:
        return ClosePrompt(
            kind=LIVE_RUN,
            title=LIVE_RUN_TITLE,
            text=LIVE_RUN_TEXT,
            buttons=(BUTTON_CLOSE_ANYWAY, BUTTON_CANCEL),
            default_button=BUTTON_CANCEL,
        )
    if has_unsaved:
        return ClosePrompt(
            kind=UNSAVED,
            title=UNSAVED_TITLE,
            text=UNSAVED_TEXT,
            buttons=(BUTTON_SAVE, BUTTON_DISCARD, BUTTON_CANCEL),
            default_button=BUTTON_CANCEL,
        )
    return ClosePrompt(
        kind=CLOSE_WITHOUT_ASKING,
        title="",
        text="",
        buttons=(),
        default_button=None,
    )
