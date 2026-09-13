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
    resolve_form,
)

from operators.form_advice import FormAdvice, FormAdviceError

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

    def __init__(
        self, mode_descriptor, columns_by_input, parent=None, advice_provider=None
    ):
        self.mode_descriptor = mode_descriptor
        self.columns_by_input = columns_by_input
        self.advice_provider = advice_provider
        _ParameterDialogSpy.instances.append(self)

    def exec(self):
        return 1

    def parameter_values(self):
        return {"picked": "yes"}


class _FakeController:
    """Just the public accessors _show_scope_and_params_dialog calls, plus
    (P1.12f-2) the conflict-warning query and the three run_create_*
    entry points -- stubbed and call-recording, not real, so the
    write/read confirm seam can be tested without a real controller."""

    def __init__(self, operator, conflict_warnings=None):
        self._operator = operator
        self._conflict_warnings = list(conflict_warnings or [])
        self.run_create_columns_calls: list = []
        self.run_create_table_calls: list = []
        self.run_create_display_calls: list = []

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

    def get_write_read_conflict_warnings(self, operator_name, mode_name):
        return list(self._conflict_warnings)

    def run_create_columns(self, operator_name, row_ids, parameters):
        self.run_create_columns_calls.append((operator_name, row_ids, parameters))

    def run_create_table(self, operator_name, row_ids, parameters):
        self.run_create_table_calls.append((operator_name, row_ids, parameters))

    def run_create_display(self, operator_name, row_ids, parameters):
        self.run_create_display_calls.append((operator_name, row_ids, parameters))


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


def _columns_mode(*, parameters, inputs):
    from operators.descriptor import OutputColumn

    return ModeDescriptor(
        mode=ExecutionMode.COLUMNS,
        label="Demo columns",
        inputs=tuple(inputs),
        parameters=tuple(parameters),
        output=OutputSpec(columns=(OutputColumn(name="out", type_tag="numeric"),)),
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

    def __init__(
        self, mode_descriptor, columns_by_input, parent=None, advice_provider=None
    ):
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


class _RealAdviceParameterDialog:
    """A ParameterDialog stand-in that runs build_field_specs and the REAL
    resolve_form in its constructor -- exactly the call that raises
    FormAdviceError when advice narrows a multi-select column field --
    without needing a QApplication or a real parent widget. It lets the
    seam test drive that failure path the same way
    _RealSpecsParameterDialog does for ParameterFormError above.
    """

    def __init__(
        self, mode_descriptor, columns_by_input, parent=None, advice_provider=None
    ):
        specs = build_field_specs(mode_descriptor, columns_by_input)
        # The advice a buggy or mistaken refine_form might return: it
        # narrows "extra", a multi-select column field. resolve_form
        # refuses this (see ui/parameter_dialog.py).
        bad_advice = FormAdvice(allowed_choices={"extra": ("age",)})
        resolve_form(specs, bad_advice, {"extra": []})


def test_a_form_advice_error_is_routed_to_the_error_path_not_raised(
    qapp, monkeypatch
):
    from operators.descriptor import OperatorDescriptor

    descriptor = OperatorDescriptor(
        name="demo_op",
        version="1.0",
        description="an operator whose refine_form gives bad advice",
        modes=(
            _display_mode(
                parameters=(
                    ColumnParameter(
                        name="extra",
                        label="Extra columns",
                        from_input="src",
                        allow_multiple=True,
                        required=False,
                    ),
                ),
                inputs=(_SOURCE_INPUT,),
            ),
        ),
    )

    # First: the real resolve_form really does refuse this advice.
    specs = build_field_specs(
        descriptor.modes[0], {"src": (("age", "numeric"),)}
    )
    with pytest.raises(FormAdviceError):
        resolve_form(
            specs,
            FormAdvice(allowed_choices={"extra": ("age",)}),
            {"extra": []},
        )

    monkeypatch.setattr(main_window_module, "RunOperatorDialog", _FakeScopeDialog)
    monkeypatch.setattr(
        main_window_module, "ParameterDialog", _RealAdviceParameterDialog
    )
    window = MainWindow.__new__(MainWindow)
    window._controller = _FakeController(_Operator(descriptor))
    monkeypatch.setattr(window, "_collect_selected_row_ids", lambda: [], raising=False)
    monkeypatch.setattr(window, "_collect_visible_row_ids", lambda: [], raising=False)
    errors: list = []
    monkeypatch.setattr(window, "_on_error", errors.append, raising=False)

    result = window._show_scope_and_params_dialog("demo_op", ExecutionMode.DISPLAY)

    # Refused through _on_error -- the FormAdviceError does not unwind out
    # of this menu-action slot -- and the researcher is told plainly, by
    # operator name, that the form guidance is faulty, not shown a
    # traceback.
    assert result is None
    assert len(errors) == 1
    message = errors[0]
    assert "demo_op" in message
    assert "form guidance" in message
    assert "faulty" in message


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


# ===========================================================================
# P1.12e-4b: the dialog uses an operator's refine_form -- disable, narrow,
# message, block OK -- and never re-enters itself while applying advice.
#
# The advice provider under test is PlotAdvancedOperator.refine_form itself
# (it reads nothing off the instance, so a bare __new__ instance is fine),
# driven through a mode whose two parameters carry plot_advanced's exact
# names and choices.
# ===========================================================================

from operators.plot_advanced import PlotAdvancedOperator


def _plot_advice_provider():
    operator = PlotAdvancedOperator.__new__(PlotAdvancedOperator)
    return operator.refine_form


def _guided_mode(*, default_chart="scatter"):
    """A DISPLAY mode with chart_type + aggregate, the pair plot_advanced's
    refine_form couples."""
    return ModeDescriptor(
        mode=ExecutionMode.DISPLAY,
        label="Guided",
        inputs=(_SOURCE_INPUT,),
        parameters=(
            ChoiceParameter(
                name="chart_type",
                label="Chart type",
                choices=(
                    ("scatter", "scatter"),
                    ("bar", "bar"),
                    ("box", "box"),
                    ("histogram", "histogram"),
                ),
                default=default_chart,
            ),
            ChoiceParameter(
                name="aggregate",
                label="Aggregate",
                choices=tuple(
                    (value, value)
                    for value in ("none", "count", "sum", "mean", "median")
                ),
                default="none",
            ),
        ),
        output=OutputSpec(is_display_only=True),
    )


def _combo_row_for(combo, value):
    return combo.findData(value)


def test_advice_is_applied_before_the_dialog_is_first_shown(qapp):
    # chart_type starts on "box", which makes aggregate inapplicable. The
    # aggregate widget must already be disabled at construction -- exec()
    # is never called.
    dialog = ParameterDialog(
        _guided_mode(default_chart="box"),
        _COLUMNS_BY_INPUT,
        advice_provider=_plot_advice_provider(),
    )
    assert dialog._widgets["aggregate"].isEnabled() is False


def test_a_disabled_field_greys_out_and_keeps_its_value(qapp):
    dialog = ParameterDialog(
        _guided_mode(default_chart="scatter"),
        _COLUMNS_BY_INPUT,
        advice_provider=_plot_advice_provider(),
    )
    aggregate = dialog._widgets["aggregate"]
    # Put a real value in, then move to a chart that disables the field.
    aggregate.setCurrentIndex(_combo_row_for(aggregate, "mean"))
    dialog._widgets["chart_type"].setCurrentIndex(
        _combo_row_for(dialog._widgets["chart_type"], "box")
    )
    assert aggregate.isEnabled() is False
    # The value the researcher chose is untouched.
    assert aggregate.currentData() == "mean"


def test_a_narrowed_dropdown_keeps_a_now_disallowed_selection_and_blocks_ok(qapp):
    dialog = ParameterDialog(
        _guided_mode(default_chart="scatter"),
        _COLUMNS_BY_INPUT,
        advice_provider=_plot_advice_provider(),
    )
    aggregate = dialog._widgets["aggregate"]
    chart_type = dialog._widgets["chart_type"]

    # Choose median, then switch to histogram, which allows only
    # count/sum/mean.
    aggregate.setCurrentIndex(_combo_row_for(aggregate, "median"))
    chart_type.setCurrentIndex(_combo_row_for(chart_type, "histogram"))

    # The selection is preserved even though it is now disallowed...
    assert aggregate.currentData() == "median"
    # ...the median item is disabled (not removed)...
    median_item = aggregate.model().item(_combo_row_for(aggregate, "median"))
    assert median_item.isEnabled() is False
    # ...count/sum/mean stay enabled...
    for allowed in ("count", "sum", "mean"):
        assert aggregate.model().item(_combo_row_for(aggregate, allowed)).isEnabled()
    # ...and OK is blocked while the disallowed value stands.
    assert dialog._ok_button.isEnabled() is False
    # The blocking message is rendered as an error, distinct from a warning.
    label_html = dialog._message_label.text()
    assert "Error" in label_html
    assert "#B00020" in label_html  # the error colour, not the warning amber


def test_ok_re_enables_when_an_allowed_value_is_picked(qapp):
    dialog = ParameterDialog(
        _guided_mode(default_chart="scatter"),
        _COLUMNS_BY_INPUT,
        advice_provider=_plot_advice_provider(),
    )
    aggregate = dialog._widgets["aggregate"]
    chart_type = dialog._widgets["chart_type"]

    aggregate.setCurrentIndex(_combo_row_for(aggregate, "median"))
    chart_type.setCurrentIndex(_combo_row_for(chart_type, "histogram"))
    assert dialog._ok_button.isEnabled() is False

    # Pick an allowed value: OK comes back. histogram+count still warns,
    # but a warning does not block.
    aggregate.setCurrentIndex(_combo_row_for(aggregate, "count"))
    assert dialog._ok_button.isEnabled() is True
    # The warning is shown (the dialog itself is never exec()'d, so check
    # the label was populated and un-hidden rather than isVisible()).
    assert dialog._message_label.isHidden() is False
    assert "Warning" in dialog._message_label.text()
    assert "Y column" in dialog._message_label.text()


class _CountingProvider:
    """Wraps a real provider and counts how many times the dialog asks."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def __call__(self, raw_values):
        self.calls += 1
        return self._inner(raw_values)


@pytest.mark.parametrize(
    "broken",
    [
        pytest.param(lambda raw_values: (_ for _ in ()).throw(RuntimeError("bug")),
                     id="raises"),
        pytest.param(lambda raw_values: None, id="forgets-return"),
        pytest.param(lambda raw_values: {"inapplicable": ("aggregate",)},
                     id="wrong-type"),
    ],
)
def test_a_broken_advice_provider_degrades_to_a_plain_form(qapp, broken):
    # refine_form is operator code called during construction and on every
    # keystroke. A raise, a forgotten return (None), or a wrong return type
    # must all leave the dialog building and working -- as a plain form --
    # not crash the Operators menu.
    dialog = ParameterDialog(
        _guided_mode(default_chart="box"),
        _COLUMNS_BY_INPUT,
        advice_provider=broken,
    )
    # No guidance applied: aggregate is not disabled, OK is not blocked.
    assert dialog._widgets["aggregate"].isEnabled() is True
    assert dialog._ok_button.isEnabled() is True
    # A later change also does not raise out of the slot.
    dialog._widgets["chart_type"].setCurrentIndex(
        _combo_row_for(dialog._widgets["chart_type"], "histogram")
    )
    assert dialog._widgets["aggregate"].isEnabled() is True


def test_advice_is_asked_once_per_field_change(qapp):
    # The construction-time apply is one call; each later field change is
    # exactly one more.
    provider = _CountingProvider(_plot_advice_provider())
    dialog = ParameterDialog(
        _guided_mode(default_chart="scatter"),
        _COLUMNS_BY_INPUT,
        advice_provider=provider,
    )
    assert provider.calls == 1

    aggregate = dialog._widgets["aggregate"]
    aggregate.setCurrentIndex(_combo_row_for(aggregate, "median"))
    calls_before = provider.calls
    dialog._widgets["chart_type"].setCurrentIndex(
        _combo_row_for(dialog._widgets["chart_type"], "histogram")
    )
    assert provider.calls == calls_before + 1


def test_widening_re_enables_items_a_previous_advice_disabled(qapp):
    dialog = ParameterDialog(
        _guided_mode(default_chart="scatter"),
        _COLUMNS_BY_INPUT,
        advice_provider=_plot_advice_provider(),
    )
    aggregate = dialog._widgets["aggregate"]
    chart_type = dialog._widgets["chart_type"]

    # histogram disables median...
    chart_type.setCurrentIndex(_combo_row_for(chart_type, "histogram"))
    assert aggregate.model().item(_combo_row_for(aggregate, "median")).isEnabled() is False

    # ...back to scatter, which restricts nothing: every item enabled again.
    chart_type.setCurrentIndex(_combo_row_for(chart_type, "scatter"))
    for value in ("none", "count", "sum", "mean", "median"):
        assert aggregate.model().item(_combo_row_for(aggregate, value)).isEnabled()
    assert aggregate.isEnabled() is True
    assert dialog._ok_button.isEnabled() is True


def _both_provider(raw_values):
    """`aggregate` is BOTH inapplicable AND narrowed -- the case the
    disable-before-narrow ordering is about."""
    return FormAdvice(
        inapplicable=("aggregate",),
        allowed_choices={"aggregate": ("count",)},
    )


def test_a_disabled_field_is_not_also_narrowed(qapp):
    # STEP 7 (P1.12e-4b) -- disable-before-narrow ordering. When a field is
    # inapplicable the narrowing pass skips it entirely: its dropdown items
    # are left exactly as they were, so nothing about its preserved value
    # is pruned.
    dialog = ParameterDialog(
        _guided_mode(default_chart="scatter"),
        _COLUMNS_BY_INPUT,
        advice_provider=_both_provider,
    )
    aggregate = dialog._widgets["aggregate"]

    assert aggregate.isEnabled() is False
    # `count` is the only allowed value, but because the field is disabled
    # the narrowing never runs -- `median` and `none` stay enabled.
    for value in ("none", "count", "sum", "mean", "median"):
        assert aggregate.model().item(_combo_row_for(aggregate, value)).isEnabled()


# ---------------------------------------------------------------------------
# P1.12e-4b follow-up -- resolve_form refuses to narrow a multi-select
# column field. Narrowing one used to un-pick a now-disallowed column mid-
# render (a value written back, forbidden by operators/form_advice.py) and
# left a stale "not allowed" error showing for one render afterwards. Both
# are gone: the combination is refused outright, at the one place (Layer A)
# that has both the advice and the field declarations. `_mode()` (top of
# this file) already declares a multi-select field, `extra`.
# ---------------------------------------------------------------------------

def test_resolve_form_refuses_to_narrow_a_multi_select_field():
    # Layer A only -- resolve_form touches no Qt, so no qapp is needed.
    specs = build_field_specs(_mode(), _COLUMNS_BY_INPUT)
    advice = FormAdvice(allowed_choices={"extra": ("age", "name")})

    with pytest.raises(FormAdviceError) as excinfo:
        resolve_form(specs, advice, {"extra": ["age"]})

    assert "extra" in str(excinfo.value)


def test_resolve_form_still_narrows_a_single_select_field():
    # The refusal is specific to a multi-select column; an ordinary
    # single-select field (a ChoiceParameter, or a ColumnParameter with
    # allow_multiple=False) is narrowed exactly as before.
    specs = build_field_specs(_mode(), _COLUMNS_BY_INPUT)
    advice = FormAdvice(allowed_choices={"chart_type": ("bar",)})

    resolved = resolve_form(specs, advice, {"chart_type": "scatter"})

    assert resolved.allowed_values == {"chart_type": ("bar",)}
    assert resolved.blocked is True


def test_narrowing_a_multi_select_field_raises_out_of_the_dialog(qapp):
    # The widget layer, not just Layer A: an operator whose refine_form
    # narrows a multi-select column field gets a loud failure -- not the
    # old silent un-pick-and-stale-error. The provider narrows `extra` as
    # soon as it is asked, so this raises at construction, before the
    # dialog is ever shown.
    def _narrows_a_multi_select_field(raw_values):
        return FormAdvice(allowed_choices={"extra": ("age",)})

    with pytest.raises(FormAdviceError):
        ParameterDialog(
            _mode(), _COLUMNS_BY_INPUT,
            advice_provider=_narrows_a_multi_select_field,
        )


# ===========================================================================
# P1.12f-2: the pre-emptive write/read conflict check, at the MainWindow
# seam. controller.py's get_write_read_conflict_warnings() and its pure
# arithmetic are covered by tests/test_result_delivery.py -- these tests
# pin only the UI wiring: a QMessageBox.question() confirm is raised
# (with the safe default) when, and only when, the query returns
# warnings, and declining it starts no run at all.
#
# Written from the work-item specification, not the implementation.
# ===========================================================================

from PySide6.QtWidgets import QMessageBox


def _fake_message_box(reply):
    """A QMessageBox stand-in: records every question() call instead of
    showing a real modal dialog, and always answers with *reply*."""
    calls: list = []

    class _FakeQMessageBox:
        StandardButton = QMessageBox.StandardButton

        @staticmethod
        def question(
            parent, title, text, buttons,
            defaultButton=QMessageBox.StandardButton.NoButton,
        ):
            calls.append(
                {
                    "title": title,
                    "text": text,
                    "buttons": buttons,
                    "defaultButton": defaultButton,
                }
            )
            return reply

    return _FakeQMessageBox, calls


def test_no_conflict_shows_no_dialog(qapp, monkeypatch):
    # Would still pass if violated? No. A version that always asked,
    # even with nothing to warn about, would fail this: calls would be
    # non-empty.
    fake_box, calls = _fake_message_box(QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(main_window_module, "QMessageBox", fake_box)

    window = MainWindow.__new__(MainWindow)
    window._controller = _FakeController(None, conflict_warnings=[])

    result = window._confirm_start_despite_conflicts(
        "demo_op", ExecutionMode.COLUMNS
    )

    assert result is True
    assert calls == []


def test_conflict_shows_dialog_defaulting_to_the_safe_choice(qapp, monkeypatch):
    # The default button must be explicitly No (do not start) -- the spec
    # is explicit that this must not rely on Qt's own default.
    fake_box, calls = _fake_message_box(QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(main_window_module, "QMessageBox", fake_box)

    window = MainWindow.__new__(MainWindow)
    window._controller = _FakeController(
        None, conflict_warnings=['"Other run" is still writing to "frames".']
    )

    result = window._confirm_start_despite_conflicts(
        "demo_op", ExecutionMode.COLUMNS
    )

    assert result is True
    assert len(calls) == 1
    assert calls[0]["defaultButton"] == QMessageBox.StandardButton.No
    assert '"Other run" is still writing to "frames".' in calls[0]["text"]


def test_declining_the_conflict_confirm_returns_false(qapp, monkeypatch):
    fake_box, calls = _fake_message_box(QMessageBox.StandardButton.No)
    monkeypatch.setattr(main_window_module, "QMessageBox", fake_box)

    window = MainWindow.__new__(MainWindow)
    window._controller = _FakeController(
        None, conflict_warnings=["some conflict"]
    )

    result = window._confirm_start_despite_conflicts(
        "demo_op", ExecutionMode.COLUMNS
    )

    assert result is False


def test_no_conflict_run_create_columns_starts_the_run_with_no_dialog(
    qapp, monkeypatch
):
    from operators.descriptor import OperatorDescriptor

    descriptor = OperatorDescriptor(
        name="demo_op",
        version="1.0",
        description="a COLUMNS-mode operator with nothing running",
        modes=(_columns_mode(parameters=(), inputs=(_SOURCE_INPUT,)),),
    )
    fake_box, calls = _fake_message_box(QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(main_window_module, "QMessageBox", fake_box)
    controller = _FakeController(_Operator(descriptor), conflict_warnings=[])
    window = _bare_window(controller, monkeypatch)

    window._on_run_create_columns("demo_op")

    # No warnings -- no dialog was shown at all -- and the run started.
    assert calls == []
    assert controller.run_create_columns_calls == [("demo_op", ["r1"], {})]


def test_declining_a_conflict_starts_no_run_and_nothing_else_happens(
    qapp, monkeypatch
):
    from operators.descriptor import OperatorDescriptor

    descriptor = OperatorDescriptor(
        name="demo_op",
        version="1.0",
        description="a COLUMNS-mode operator that conflicts with a live run",
        modes=(_columns_mode(parameters=(), inputs=(_SOURCE_INPUT,)),),
    )
    fake_box, calls = _fake_message_box(QMessageBox.StandardButton.No)
    monkeypatch.setattr(main_window_module, "QMessageBox", fake_box)
    controller = _FakeController(
        _Operator(descriptor),
        conflict_warnings=['"Other run" is still writing to "frames".'],
    )
    window = _bare_window(controller, monkeypatch)

    window._on_run_create_columns("demo_op")

    # The confirm was shown and declined: no run started, and nothing
    # else happened (no error surfaced either).
    assert len(calls) == 1
    assert controller.run_create_columns_calls == []
    assert window._on_error_calls == []


def test_accepting_a_conflict_starts_the_run(qapp, monkeypatch):
    from operators.descriptor import OperatorDescriptor

    descriptor = OperatorDescriptor(
        name="demo_op",
        version="1.0",
        description="a COLUMNS-mode operator that conflicts with a live run",
        modes=(_columns_mode(parameters=(), inputs=(_SOURCE_INPUT,)),),
    )
    fake_box, calls = _fake_message_box(QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(main_window_module, "QMessageBox", fake_box)
    controller = _FakeController(
        _Operator(descriptor),
        conflict_warnings=['"Other run" is still writing to "frames".'],
    )
    window = _bare_window(controller, monkeypatch)

    window._on_run_create_columns("demo_op")

    assert len(calls) == 1
    assert controller.run_create_columns_calls == [("demo_op", ["r1"], {})]
