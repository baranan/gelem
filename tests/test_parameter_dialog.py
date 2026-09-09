"""
tests/test_parameter_dialog.py

Layer B of the generated parameter dialog (P1.12e-1 built it, P1.12e-2
wired it in) -- the ``ParameterDialog(QDialog)`` in ui/parameter_dialog.py.

Layer A (the Qt-free functions) is covered by tests/test_parameter_form.py.
This file is the widget test P1.12e-1 could not write: it realises the
actual dialog and checks that

  * it builds exactly one widget per declared field, in declaration
    order, of the type that field's kind calls for;
  * reading the widgets back and running them through Layer A --
    which is all ``parameter_values()`` does -- produces the dict
    ``collect_parameters`` would return for those same raw values.

Widget realisation follows tests/test_settings_dialog.py: the shared
session-scoped ``qapp`` fixture from tests/conftest.py provides the one
QApplication (offscreen), and the dialog is constructed but never shown.

Written from the work-item specification, not the implementation.

Run with:
    python -m pytest tests/test_parameter_dialog.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from operators.descriptor import (
    BooleanParameter,
    ChoiceParameter,
    ColumnParameter,
    ExecutionMode,
    InputKind,
    InputSpec,
    ModeDescriptor,
    NumberParameter,
    OutputSpec,
    TextParameter,
)

from operators.descriptor import ParameterSpec

from ui.parameter_dialog import (
    ParameterDialog,
    ParameterFormError,
    build_field_specs,
    collect_parameters,
)

import ui.main_window as main_window_module
from ui.main_window import MainWindow, _UnresolvableInputKind


# ---------------------------------------------------------------------------
# Fixtures: a mode with one parameter of each kind, in a fixed declaration
# order, plus the columns its one ColumnParameter draws from.
# ---------------------------------------------------------------------------

_SOURCE_INPUT = InputSpec(
    name="src", label="Source table", kind=InputKind.ACTIVE_TABLE
)

_COLUMNS_BY_INPUT = {
    "src": (("age", "numeric"), ("name", "text"), ("clip", "media_path")),
}

# Declaration order is: number, text, boolean, choice, column (single),
# column (multi). parameter_values() must key on these names.
_PARAMETERS = (
    NumberParameter(
        name="frame_step", label="Frame step", minimum=1, maximum=100, default=1
    ),
    TextParameter(name="title", label="Title", required=False, default=""),
    BooleanParameter(name="normalise", label="Normalise", default=False),
    ChoiceParameter(
        name="chart_type",
        label="Chart type",
        choices=(("scatter", "Scatter"), ("bar", "Bar")),
        default="scatter",
    ),
    ColumnParameter(name="x", label="X axis", from_input="src"),
    ColumnParameter(
        name="extra",
        label="Extra columns",
        from_input="src",
        allow_multiple=True,
        required=False,
    ),
)


def _mode():
    return ModeDescriptor(
        mode=ExecutionMode.DISPLAY,
        label="Test mode",
        inputs=(_SOURCE_INPUT,),
        parameters=_PARAMETERS,
        output=OutputSpec(is_display_only=True),
    )


# ===========================================================================
# One widget per declared field, in declaration order, of the right type
# ===========================================================================

def test_one_widget_per_field_in_declaration_order(qapp):
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QLineEdit,
        QListWidget,
        QSpinBox,
    )

    dialog = ParameterDialog(_mode(), _COLUMNS_BY_INPUT)

    # _widgets is name -> widget; dicts preserve insertion order, and
    # _build inserts in field-spec order, which is declaration order.
    assert list(dialog._widgets.keys()) == [
        "frame_step",
        "title",
        "normalise",
        "chart_type",
        "x",
        "extra",
    ]

    widgets = dialog._widgets
    assert isinstance(widgets["frame_step"], QSpinBox)
    assert isinstance(widgets["title"], QLineEdit)
    assert isinstance(widgets["normalise"], QCheckBox)
    assert isinstance(widgets["chart_type"], QComboBox)
    assert isinstance(widgets["x"], QComboBox)          # single column
    assert isinstance(widgets["extra"], QListWidget)    # multi column


def test_field_count_equals_declared_parameter_count(qapp):
    dialog = ParameterDialog(_mode(), _COLUMNS_BY_INPUT)
    assert len(dialog._field_specs) == len(_PARAMETERS)


# ===========================================================================
# Reading the widgets back == what Layer A returns for the same raw values
# ===========================================================================

def test_parameter_values_matches_layer_a_for_the_widgets_state(qapp):
    dialog = ParameterDialog(_mode(), _COLUMNS_BY_INPUT)
    w = dialog._widgets

    # Drive every widget to a known state.
    w["frame_step"].setValue(7)
    w["title"].setText("My chart")
    w["normalise"].setChecked(True)
    # Choose "bar" by its item data (the choice VALUE, not the label).
    w["chart_type"].setCurrentIndex(w["chart_type"].findData("bar"))
    # Choose the "age" column by data.
    w["x"].setCurrentIndex(w["x"].findData("age"))
    # Select two rows of the multi-column list.
    for row in range(w["extra"].count()):
        if w["extra"].item(row).text() in ("age", "clip"):
            w["extra"].item(row).setSelected(True)

    # The raw values those widget states represent. This is exactly what
    # ParameterDialog._raw_values() should read back.
    expected_raw = {
        "frame_step": 7,
        "title": "My chart",
        "normalise": True,
        "chart_type": "bar",
        "x": "age",
        # multi-column comes back in row order, not selection order
        "extra": ["age", "clip"],
    }
    assert dialog._raw_values() == expected_raw

    # And parameter_values() must be precisely Layer A applied to those.
    specs = build_field_specs(_mode(), _COLUMNS_BY_INPUT)
    assert dialog.parameter_values() == collect_parameters(specs, expected_raw)
    # Spell the expected result out too, so a change in both halves at once
    # is still caught.
    assert dialog.parameter_values() == {
        "frame_step": 7,
        "title": "My chart",
        "normalise": True,
        "chart_type": "bar",
        "x": "age",
        "extra": ("age", "clip"),
    }


def test_untouched_optional_fields_are_omitted_like_layer_a(qapp):
    # Leave title (optional text) blank and select no extra columns.
    # Everything required still needs a value, so give x a column.
    dialog = ParameterDialog(_mode(), _COLUMNS_BY_INPUT)
    w = dialog._widgets

    w["frame_step"].setValue(3)
    w["x"].setCurrentIndex(w["x"].findData("name"))

    values = dialog.parameter_values()

    # Optional text left blank and optional multi-column left empty are
    # absent from the dict, exactly as collect_parameters specifies.
    assert "title" not in values
    assert "extra" not in values
    # Number and boolean always emit (a spin box / checkbox has no "unset").
    assert values["frame_step"] == 3
    assert values["normalise"] is False
    assert values["chart_type"] == "scatter"  # descriptor default
    assert values["x"] == "name"


# ===========================================================================
# A single-column field's default preselects the widget (P1.12e-3)
# ===========================================================================

def _column_default_mode(*, default):
    """A one-parameter DISPLAY mode: a single-selection column field whose
    ColumnParameter names ``default`` as its preselected column."""
    return ModeDescriptor(
        mode=ExecutionMode.DISPLAY,
        label="Column default test",
        inputs=(_SOURCE_INPUT,),
        parameters=(
            ColumnParameter(
                name="video_column",
                label="Video path column",
                from_input="src",
                default=default,
            ),
        ),
        output=OutputSpec(is_display_only=True),
    )


def test_single_column_widget_starts_on_the_defaulted_column(qapp):
    # "clip" is one of the offered columns, so the combo opens on it rather
    # than on the blank first entry.
    dialog = ParameterDialog(
        _column_default_mode(default="clip"), _COLUMNS_BY_INPUT
    )
    combo = dialog._widgets["video_column"]
    assert combo.currentData() == "clip"
    # parameter_values() therefore already carries it with no interaction.
    assert dialog.parameter_values() == {"video_column": "clip"}


def test_single_column_widget_starts_blank_when_the_default_is_absent(qapp):
    # _COLUMNS_BY_INPUT offers age/name/clip but not full_path, so the
    # default is dropped and the field opens on the blank entry.
    dialog = ParameterDialog(
        _column_default_mode(default="full_path"), _COLUMNS_BY_INPUT
    )
    combo = dialog._widgets["video_column"]
    assert combo.currentData() is None
    # Required and unset -> parameter_values() refuses through Layer A.
    with pytest.raises(ParameterFormError):
        dialog.parameter_values()


# ===========================================================================
# The two MainWindow-side rules P1.12e-2 adds (STEP 5 deletion check):
#   - a mode that declares no parameters shows NO dialog at all and
#     proceeds with {} -- exactly what a None get_parameters_dialog()
#     return used to mean;
#   - an input kind the form cannot resolve (NAMED_TABLE, WHOLE_PROJECT)
#     refuses the run before showing anything, naming the operator and the
#     kind.
#
# Tested at the seam: a bare MainWindow (no __init__, nothing realised)
# with a fake controller, and RunOperatorDialog / ParameterDialog swapped
# for spies. This is the "no widget is realised" approach the removed
# _parameters_from_dialog tests used.
# ===========================================================================

class _FakeScopeDialog:
    """Stands in for RunOperatorDialog: always accepted, one chosen row."""

    def __init__(self, *args, **kwargs):
        self.chosen_row_ids = ["r1"]

    def exec(self):
        return 1


class _ParameterDialogSpy:
    """Records that MainWindow built a parameter dialog, and with what."""

    instances: list = []

    def __init__(self, mode_descriptor, columns_by_input, parent=None):
        self.mode_descriptor = mode_descriptor
        self.columns_by_input = columns_by_input
        _ParameterDialogSpy.instances.append(self)

    def exec(self):
        return 1

    def parameter_values(self):
        return {"picked": "yes"}


class _FakeController:
    """Just the public accessors _show_scope_and_params_dialog calls."""

    def __init__(self, operator):
        self._operator = operator

    def get_all_row_ids(self):
        return ["r1", "r2"]

    def get_operator(self, name):
        return self._operator

    def get_active_table(self):
        return "frames"

    def get_column_names(self, table_name=None):
        return ["age", "clip"]

    def get_column_type(self, column_name, table_name=None):
        class _CT:
            tag = "numeric" if column_name == "age" else "media_path"
        return _CT()


class _Operator:
    def __init__(self, descriptor, name="demo_op"):
        self.name = name
        self.descriptor = descriptor


def _display_mode(*, parameters, inputs):
    return ModeDescriptor(
        mode=ExecutionMode.DISPLAY,
        label="Demo display",
        inputs=tuple(inputs),
        parameters=tuple(parameters),
        output=OutputSpec(is_display_only=True),
    )


def _bare_window(controller, monkeypatch):
    """A MainWindow with no __init__ run: enough state for the method under
    test, and the two dialog classes replaced by spies."""
    _ParameterDialogSpy.instances.clear()
    monkeypatch.setattr(main_window_module, "RunOperatorDialog", _FakeScopeDialog)
    monkeypatch.setattr(main_window_module, "ParameterDialog", _ParameterDialogSpy)

    window = MainWindow.__new__(MainWindow)
    window._controller = controller
    monkeypatch.setattr(window, "_collect_selected_row_ids", lambda: [], raising=False)
    monkeypatch.setattr(window, "_collect_visible_row_ids", lambda: [], raising=False)
    window._on_error_calls = []
    monkeypatch.setattr(
        window, "_on_error", window._on_error_calls.append, raising=False
    )
    return window


def test_a_mode_with_no_parameters_shows_no_dialog_and_returns_empty(qapp, monkeypatch):
    from operators.descriptor import OperatorDescriptor

    descriptor = OperatorDescriptor(
        name="demo_op",
        version="1.0",
        description="no parameters at all",
        modes=(_display_mode(parameters=(), inputs=(_SOURCE_INPUT,)),),
    )
    window = _bare_window(_FakeController(_Operator(descriptor)), monkeypatch)

    result = window._show_scope_and_params_dialog("demo_op", ExecutionMode.DISPLAY)

    # No ParameterDialog was built, and the run proceeds with {}.
    assert _ParameterDialogSpy.instances == []
    assert result == (["r1"], {})


def test_an_unresolvable_input_kind_refuses_before_any_dialog(qapp, monkeypatch):
    from operators.descriptor import OperatorDescriptor

    # A parameter is declared (so we get past the no-parameters branch),
    # and the mode's input is NAMED_TABLE -- which the form cannot resolve.
    named_input = InputSpec(
        name="other", label="Other table", kind=InputKind.NAMED_TABLE
    )
    descriptor = OperatorDescriptor(
        name="demo_op",
        version="1.0",
        description="declares a NAMED_TABLE input",
        modes=(
            _display_mode(
                parameters=(TextParameter(name="note", label="Note"),),
                inputs=(named_input,),
            ),
        ),
    )
    window = _bare_window(_FakeController(_Operator(descriptor)), monkeypatch)

    result = window._show_scope_and_params_dialog("demo_op", ExecutionMode.DISPLAY)

    # Refused: no dialog built, None returned, and the researcher was told
    # which operator and which kind.
    assert _ParameterDialogSpy.instances == []
    assert result is None
    assert len(window._on_error_calls) == 1
    message = window._on_error_calls[0]
    assert "demo_op" in message
    assert "NAMED_TABLE" in message


class _WeirdParameter(ParameterSpec):
    """A ParameterSpec subclass build_field_specs has no branch for -- the
    'declared kind the form cannot render' case the vocabulary is designed
    to grow into."""

    kind = "weird"


class _RealSpecsParameterDialog:
    """A ParameterDialog stand-in that runs the REAL build_field_specs in
    its constructor -- exactly the call that raises ParameterFormError for
    an unrenderable kind -- without needing a QApplication or a real parent
    widget. It lets the seam test drive that failure path.
    """

    def __init__(self, mode_descriptor, columns_by_input, parent=None):
        build_field_specs(mode_descriptor, columns_by_input)


def test_an_unrenderable_parameter_kind_refuses_instead_of_crashing(qapp, monkeypatch):
    from operators.descriptor import OperatorDescriptor

    descriptor = OperatorDescriptor(
        name="demo_op",
        version="1.0",
        description="declares a parameter kind the form cannot render",
        modes=(
            _display_mode(
                parameters=(_WeirdParameter(name="odd", label="Odd one"),),
                inputs=(_SOURCE_INPUT,),
            ),
        ),
    )

    # First: the real build_field_specs really does reject this kind.
    with pytest.raises(ParameterFormError):
        build_field_specs(descriptor.modes[0], {"src": ()})

    monkeypatch.setattr(main_window_module, "RunOperatorDialog", _FakeScopeDialog)
    monkeypatch.setattr(
        main_window_module, "ParameterDialog", _RealSpecsParameterDialog
    )
    window = MainWindow.__new__(MainWindow)
    window._controller = _FakeController(_Operator(descriptor))
    monkeypatch.setattr(window, "_collect_selected_row_ids", lambda: [], raising=False)
    monkeypatch.setattr(window, "_collect_visible_row_ids", lambda: [], raising=False)
    errors: list = []
    monkeypatch.setattr(window, "_on_error", errors.append, raising=False)

    result = window._show_scope_and_params_dialog("demo_op", ExecutionMode.DISPLAY)

    # Refused through _on_error, not raised out of the menu slot.
    assert result is None
    assert len(errors) == 1
    assert "odd" in errors[0]


def test_columns_by_input_raises_for_a_whole_project_input(qapp):
    whole_project = InputSpec(
        name="everything", label="Everything", kind=InputKind.WHOLE_PROJECT
    )
    mode = _display_mode(
        parameters=(TextParameter(name="note", label="Note"),),
        inputs=(whole_project,),
    )
    window = MainWindow.__new__(MainWindow)
    window._controller = _FakeController(_Operator(None))

    with pytest.raises(_UnresolvableInputKind) as excinfo:
        window._columns_by_input("demo_op", mode)
    assert "WHOLE_PROJECT" in str(excinfo.value)
    assert "demo_op" in str(excinfo.value)
