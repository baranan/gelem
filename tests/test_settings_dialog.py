"""
tests/test_settings_dialog.py

The settings dialog (P0.5b-2ii-c2b2).

Layer A -- the plain-data functions in ui/settings_dialog.py -- carries all
the arithmetic and all the wording, and is what these tests mostly exercise:
it needs no QApplication. A handful of Layer B tests use the shared `qapp`
fixture to prove the dialog builds the right shape and submits only what
moved.

Written from the work-item specification, not from the implementation.

Run with:
    python -m pytest tests/test_settings_dialog.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from settings.settings import (
    PICTURE_MEMORY_MAX_BYTES_RANGE,
    PICTURE_DISK_MAX_BYTES_RANGE,
)
from settings.settings_gateway import SettingField, SettingsGateway
from settings.settings_store import SettingsStore

from ui.settings_dialog import (
    ADJUSTED_HEADING,
    BUTTON_CANCEL,
    BUTTON_SAVE_AND_KEEP,
    BUTTON_SAVE_AND_QUIT,
    NO_SETTINGS_SENTENCE,
    bytes_to_display_mib,
    changed_values,
    confirmation_text,
    display_mib_to_bytes,
    mib_spin_bounds,
    restart_required_names,
)

_MIB = 1024 * 1024


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------

class _DictBackend:
    """The two-method key/value backend a SettingsStore needs, in memory."""

    def __init__(self):
        self._values: dict[str, str] = {}

    def get(self, key: str):
        return self._values.get(key)

    def set(self, key: str, value: str) -> None:
        self._values[key] = value


class _FakeController:
    """Mirrors just the two settings methods MainWindow's dialog calls."""

    def __init__(self, fields):
        self._fields = list(fields)
        self.applied: list[dict] = []

    def get_settings_fields(self):
        return list(self._fields)

    def apply_settings(self, values: dict) -> list[str]:
        self.applied.append(dict(values))
        return []


def _real_fields() -> list[SettingField]:
    """The five fields exactly as they ship, read through the gateway."""
    store = SettingsStore(_DictBackend())
    return SettingsGateway(store).describe_fields()


def _field(
    name: str,
    *,
    unit: str,
    restart_required: bool,
    label: str | None = None,
    minimum: int = 0,
    maximum: int = 1000,
    current_value: int = 10,
) -> SettingField:
    return SettingField(
        name=name,
        label=label or name,
        help_text=f"help for {name}",
        minimum=minimum,
        maximum=maximum,
        unit=unit,
        restart_required=restart_required,
        current_value=current_value,
    )


# ===========================================================================
# Layer A -- arithmetic
# ===========================================================================

def test_mib_bounds_round_trip_memory_range():
    minimum_bytes, maximum_bytes = PICTURE_MEMORY_MAX_BYTES_RANGE
    low_mib, high_mib = mib_spin_bounds(minimum_bytes, maximum_bytes)

    # Sane ordering.
    assert low_mib <= high_mib

    # EVERY value the spin box can hold must convert back to a byte count
    # inside the real range -- check both extremes explicitly.
    assert minimum_bytes <= display_mib_to_bytes(low_mib) <= maximum_bytes
    assert minimum_bytes <= display_mib_to_bytes(high_mib) <= maximum_bytes


def test_mib_bounds_round_trip_disk_range():
    minimum_bytes, maximum_bytes = PICTURE_DISK_MAX_BYTES_RANGE
    low_mib, high_mib = mib_spin_bounds(minimum_bytes, maximum_bytes)

    assert low_mib <= high_mib
    assert minimum_bytes <= display_mib_to_bytes(low_mib) <= maximum_bytes
    assert minimum_bytes <= display_mib_to_bytes(high_mib) <= maximum_bytes


def test_mib_bounds_use_ceiling_and_floor():
    # 1 MiB + 1 byte .. 3 MiB - 1 byte  ->  ceil(1.00..) = 2, floor(2.99..) = 2
    low_mib, high_mib = mib_spin_bounds(_MIB + 1, 3 * _MIB - 1)
    assert (low_mib, high_mib) == (2, 2)


def test_bytes_to_display_mib_rounds_to_nearest():
    # 500 MiB exactly.
    assert bytes_to_display_mib(500 * _MIB, 1, 100000) == 500
    # 1.4 MiB rounds down to 1, 1.6 MiB rounds up to 2.
    assert bytes_to_display_mib(int(1.4 * _MIB), 1, 100000) == 1
    assert bytes_to_display_mib(int(1.6 * _MIB), 1, 100000) == 2


def test_bytes_to_display_mib_clamps_into_spin_bounds():
    # Below the spin minimum and above the spin maximum both clamp.
    assert bytes_to_display_mib(1 * _MIB, 16, 65536) == 16
    assert bytes_to_display_mib(999999 * _MIB, 16, 65536) == 65536


def test_display_mib_to_bytes_is_exact_multiple():
    assert display_mib_to_bytes(1) == _MIB
    assert display_mib_to_bytes(500) == 500 * _MIB


# ===========================================================================
# Layer A -- changed_values
# ===========================================================================

def test_changed_values_returns_only_what_moved():
    initial = {"a": 1, "b": 2, "c": 3}
    current = {"a": 1, "b": 20, "c": 3}
    assert changed_values(initial, current) == {"b": 20}


def test_changed_values_empty_when_nothing_moved():
    same = {"a": 1, "b": 2}
    assert changed_values(same, dict(same)) == {}


def test_changed_values_reports_a_new_key():
    assert changed_values({"a": 1}, {"a": 1, "b": 5}) == {"b": 5}


# ===========================================================================
# Layer A -- restart_required_names
# ===========================================================================

def test_restart_required_names_read_off_the_flag():
    fields = [
        _field("mem", unit="bytes", restart_required=False, label="Memory"),
        _field("workers", unit="count", restart_required=True, label="Workers"),
        _field("thumb", unit="pixels", restart_required=True, label="Thumb"),
    ]
    changed = {"mem": 1, "workers": 4, "thumb": 200}

    names = restart_required_names(fields, changed)

    # A test that walks a list asserts the list is non-empty first.
    assert names
    assert names == ["Workers", "Thumb"]
    # The non-restart field is not named.
    assert "Memory" not in names


def test_restart_required_names_empty_when_only_immediate_field_moved():
    fields = [
        _field("mem", unit="bytes", restart_required=False, label="Memory"),
        _field("workers", unit="count", restart_required=True, label="Workers"),
    ]
    assert restart_required_names(fields, {"mem": 1}) == []


# ===========================================================================
# Layer A -- confirmation_text
# ===========================================================================

def test_confirmation_text_names_the_restart_settings():
    fields = [
        _field("workers", unit="count", restart_required=True, label="Workers"),
    ]
    text = confirmation_text(fields, {"workers": 4})
    # Names the changed setting, and tells the researcher to quit and
    # relaunch -- Gelem does not restart itself.
    assert "Workers" in text
    assert "takes effect the next time Gelem starts" in text
    assert "Gelem does not restart itself, so quit and start it again" in text


def test_confirmation_text_warns_of_regeneration_when_a_pixels_field_changed():
    fields = [
        _field("thumb", unit="pixels", restart_required=True, label="Thumb"),
        _field("preview", unit="pixels", restart_required=True, label="Preview"),
    ]
    text = confirmation_text(fields, {"thumb": 200})
    assert "regenerated from the source files" in text
    assert "unreachable" in text


def test_confirmation_text_omits_regeneration_warning_for_bytes_only_change():
    fields = [
        _field("mem", unit="bytes", restart_required=False, label="Memory"),
        _field("workers", unit="count", restart_required=True, label="Workers"),
    ]
    # Only the worker count (a "count" field) changed -- no pixels field.
    text = confirmation_text(fields, {"workers": 4})
    assert "regenerated from the source files" not in text
    assert "unreachable" not in text


def test_researcher_facing_strings_are_exactly_as_specified():
    # These are read as English by a researcher; pin the wording.
    assert NO_SETTINGS_SENTENCE == "Settings are not available in this mode."
    assert BUTTON_SAVE_AND_QUIT == "Save and quit Gelem"
    assert BUTTON_SAVE_AND_KEEP == "Save and keep working"
    assert BUTTON_CANCEL == "Cancel"
    assert ADJUSTED_HEADING == "Gelem adjusted some settings"


def test_confirmation_text_uses_unit_not_name_to_spot_the_size_fields():
    # A pixels field whose name matches neither "thumbnail_max_side" nor
    # "preview_max_side" must still trigger the warning: the code keys on
    # field.unit == "pixels".
    fields = [
        _field("some_other_pixels", unit="pixels", restart_required=True),
    ]
    text = confirmation_text(fields, {"some_other_pixels": 42})
    assert "regenerated from the source files" in text


# ===========================================================================
# Layer B -- the dialog builds the right shape
# ===========================================================================

def test_empty_field_list_shows_one_sentence_and_no_spin_boxes(qapp):
    from PySide6.QtWidgets import QLabel, QSpinBox
    from ui.settings_dialog import SettingsDialog

    dialog = SettingsDialog(_FakeController([]))

    labels = [w.text() for w in dialog.findChildren(QLabel)]
    assert NO_SETTINGS_SENTENCE in labels
    # Nothing editable was built.
    assert dialog.findChildren(QSpinBox) == []


def test_real_fields_render_one_spin_box_each(qapp):
    from PySide6.QtWidgets import QSpinBox
    from ui.settings_dialog import SettingsDialog

    fields = _real_fields()
    assert fields  # walking this list below -- prove it is non-empty first

    dialog = SettingsDialog(_FakeController(fields))
    spins = dialog.findChildren(QSpinBox)
    assert len(spins) == len(fields)

    # The two byte-valued fields are shown in MiB.
    mib_spins = [s for s in spins if s.suffix() == " MiB"]
    byte_fields = [f for f in fields if f.unit == "bytes"]
    assert len(mib_spins) == len(byte_fields) == 2


def _silence_modal_boxes(monkeypatch):
    """Stop any QMessageBox.exec() blocking a headless test, and answer the
    restart confirmation "keep" without a dialog.

    Returns a list that records each _ask_restart call (its `changed`
    argument), so a test can assert whether the confirmation box was
    offered at all.
    """
    from PySide6.QtWidgets import QMessageBox
    from ui.settings_dialog import SettingsDialog

    ask_restart_calls: list = []

    def _fake_ask_restart(self, changed):
        ask_restart_calls.append(changed)
        return "keep"

    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(SettingsDialog, "_ask_restart", _fake_ask_restart)
    return ask_restart_calls


def test_ok_with_nothing_changed_calls_no_controller_method(qapp, monkeypatch):
    from ui.settings_dialog import SettingsDialog

    _silence_modal_boxes(monkeypatch)
    controller = _FakeController(_real_fields())
    dialog = SettingsDialog(controller)

    dialog._on_ok()

    assert controller.applied == []


def test_ok_submits_only_the_changed_field_in_native_units(qapp, monkeypatch):
    from ui.settings_dialog import SettingsDialog

    _silence_modal_boxes(monkeypatch)
    fields = _real_fields()
    controller = _FakeController(fields)
    dialog = SettingsDialog(controller)

    # Move only the memory ceiling (an immediate-effect field, so no restart
    # confirmation box is shown). Its spin box is in MiB.
    mem_field = next(f for f in fields if f.name == "picture_memory_max_bytes")
    mem_spin = dialog._spin_boxes["picture_memory_max_bytes"]
    new_mib = mem_spin.value() + 64
    mem_spin.setValue(new_mib)

    dialog._on_ok()

    assert len(controller.applied) == 1
    submitted = controller.applied[0]
    # Only the one field, and in bytes not MiB.
    assert list(submitted.keys()) == ["picture_memory_max_bytes"]
    assert submitted["picture_memory_max_bytes"] == new_mib * _MIB


def test_immediate_only_change_shows_no_restart_confirmation(qapp, monkeypatch):
    # Moving only an immediate-effect field (the memory ceiling) must save
    # without ever offering the quit/relaunch confirmation box.
    from ui.settings_dialog import SettingsDialog

    ask_restart_calls = _silence_modal_boxes(monkeypatch)
    fields = _real_fields()
    controller = _FakeController(fields)
    dialog = SettingsDialog(controller)

    mem_spin = dialog._spin_boxes["picture_memory_max_bytes"]
    mem_spin.setValue(mem_spin.value() + 64)

    dialog._on_ok()

    # The box was never offered ...
    assert ask_restart_calls == []
    # ... and the change was still persisted.
    assert len(controller.applied) == 1


def test_non_mib_aligned_stored_ceiling_is_not_reported_as_changed(
    qapp, monkeypatch
):
    # A stored byte value that is not a whole number of MiB must not read as
    # "changed" when the researcher never touches its spin box -- otherwise
    # opening the dialog and clicking OK would fire an eviction/sweep.
    from ui.settings_dialog import SettingsDialog

    _silence_modal_boxes(monkeypatch)
    fields = [
        _field(
            "picture_disk_max_bytes",
            unit="bytes",
            restart_required=False,
            label="Disk",
            minimum=64 * _MIB,
            maximum=1024 * _MIB,
            current_value=512 * _MIB + 123,
        ),
    ]
    controller = _FakeController(fields)
    dialog = SettingsDialog(controller)

    dialog._on_ok()

    assert controller.applied == []


# ===========================================================================
# AST guardrail -- the File menu wires a QAction to a SettingsDialog handler
# ===========================================================================

def _main_window_tree() -> ast.Module:
    source = (PROJECT_ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    return ast.parse(source)


def _functions_that_construct(tree: ast.Module, class_name: str) -> set[str]:
    """Names of functions whose body contains a call to `class_name(...)`."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == class_name
                ):
                    names.add(node.name)
    return names


def _handlers_connected_to_triggered_in(tree: ast.Module, fn_name: str) -> set[str]:
    """`self.<name>` handlers passed to any `.triggered.connect(...)` call
    inside the named function."""
    handlers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "connect"
                    and isinstance(sub.func.value, ast.Attribute)
                    and sub.func.value.attr == "triggered"
                ):
                    for arg in sub.args:
                        if (
                            isinstance(arg, ast.Attribute)
                            and isinstance(arg.value, ast.Name)
                            and arg.value.id == "self"
                        ):
                            handlers.add(arg.attr)
    return handlers


def test_file_menu_action_is_connected_to_a_settings_dialog_handler():
    tree = _main_window_tree()

    # ui/main_window.py imports SettingsDialog.
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "SettingsDialog" in imported

    constructs = _functions_that_construct(tree, "SettingsDialog")
    assert constructs, "no method in main_window.py constructs SettingsDialog"

    connected = _handlers_connected_to_triggered_in(tree, "_build_menu")
    assert connected, "_build_menu connects no QAction to a self handler"

    # A menu QAction must be wired to a handler that opens the dialog.
    assert constructs & connected, (
        "no File-menu QAction is connected to a handler that constructs "
        f"SettingsDialog (constructors: {constructs}, connected: {connected})"
    )
