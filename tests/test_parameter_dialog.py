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
    NewTableNameParameter,
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

    def get_table_names(self):
        # table-name-validation, round 2: _show_scope_and_params_dialog
        # now calls this unconditionally whenever the mode declares any
        # parameters, to build ParameterDialog's existing_table_names --
        # every test below that reaches that branch needs it to exist.
        # A test that cares about a specific collision overrides this on
        # its own controller instance.
        return ["frames"]

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
        self, mode_descriptor, columns_by_input, parent=None, advice_provider=None,
        existing_table_names=(),
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
        self, mode_descriptor, columns_by_input, parent=None, advice_provider=None,
        existing_table_names=(),
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


# ===========================================================================
# run-indicator-1: the status-bar run indicator, at the MainWindow seam.
#
# controller.format_run_indicator_text() and AppController.get_live_runs()
# are covered by tests/test_result_delivery.py -- these tests pin only the
# widget wiring: the label is hidden when nothing is running, shown with
# the right text when something is, and _on_live_runs_changed /
# _on_operator_progress rebuild it from the controller's current state
# rather than from their own argument alone.
#
# Written from the work-item specification, not the implementation.
# ===========================================================================

from PySide6.QtWidgets import QLabel, QPushButton


class _LiveRunsController:
    """get_live_runs() -- the one thing _refresh_run_indicator() reads
    from the controller -- plus (run-indicator-3) cancel_run(), recording
    every call instead of doing anything real."""

    def __init__(self, live_runs=None):
        self._live_runs = list(live_runs or [])
        self.cancel_run_calls: list[str] = []

    def get_live_runs(self):
        return list(self._live_runs)

    def cancel_run(self, operation_id):
        self.cancel_run_calls.append(operation_id)


def _indicator_window(controller):
    """A MainWindow with no __init__ run, carrying just the run-indicator
    state _build_status_bar() would have created -- the label
    (run-indicator-1/2) and, since run-indicator-3, the Cancel button
    beside it."""
    window = MainWindow.__new__(MainWindow)
    window._controller = controller
    window._run_indicator_label = QLabel()
    window._cancel_run_button = QPushButton()
    window._latest_run_percent = None
    return window


def test_indicator_label_is_hidden_with_no_live_runs(qapp):
    window = _indicator_window(_LiveRunsController([]))

    window._refresh_run_indicator()

    # Would still pass if violated? No. A version that only cleared the
    # text but left the widget visible would fail this: isHidden() would
    # still read False, leaving an empty gap in the status bar.
    assert window._run_indicator_label.isHidden() is True
    assert window._run_indicator_label.text() == ""


def test_indicator_label_is_shown_with_one_live_run(qapp):
    controller = _LiveRunsController(
        [{"label": "Extract blendshapes", "table_name": "frames"}]
    )
    window = _indicator_window(controller)

    window._refresh_run_indicator()

    assert window._run_indicator_label.isHidden() is False
    assert "Extract blendshapes" in window._run_indicator_label.text()
    assert "frames" in window._run_indicator_label.text()


def test_on_live_runs_changed_drops_the_stale_percentage(qapp):
    # A percentage left over from a run that just finished must not be
    # shown next to whatever the live-run set becomes next.
    controller = _LiveRunsController(
        [{"label": "Extract blendshapes", "table_name": "frames"}]
    )
    window = _indicator_window(controller)
    window._latest_run_percent = 90

    window._on_live_runs_changed()

    assert "90" not in window._run_indicator_label.text()
    assert window._latest_run_percent is None


def test_on_operator_progress_updates_the_label_text(qapp):
    controller = _LiveRunsController(
        [{"label": "Extract blendshapes", "table_name": "frames"}]
    )
    window = _indicator_window(controller)

    window._on_operator_progress(33)

    assert window._latest_run_percent == 33
    assert "33" in window._run_indicator_label.text()
    assert window._run_indicator_label.isHidden() is False


def test_indicator_label_hides_again_once_the_run_ends(qapp):
    controller = _LiveRunsController(
        [{"label": "Extract blendshapes", "table_name": "frames"}]
    )
    window = _indicator_window(controller)
    window._on_operator_progress(50)
    assert window._run_indicator_label.isHidden() is False

    # The run finished: the controller's live-run list is now empty by
    # the time live_runs_changed fires (deregister happens first).
    controller._live_runs = []
    window._on_live_runs_changed()

    assert window._run_indicator_label.isHidden() is True
    assert window._run_indicator_label.text() == ""


# ===========================================================================
# run-indicator-2: the free-text run.log() message, at the same MainWindow
# seam. controller.format_run_indicator_text()'s handling of "message" and
# AppController's get_live_runs()/_apply_run_logs() are covered by
# tests/test_result_delivery.py -- these tests pin only that the widget
# shows what get_live_runs() currently reports, the same "pull the
# controller's current state" property the run-indicator-1 tests above pin
# for label/table/percent.
#
# Written from the work-item specification, not the implementation.
# ===========================================================================

def test_indicator_label_shows_a_run_log_message(qapp):
    controller = _LiveRunsController(
        [{
            "label": "Extract frames",
            "table_name": "videos",
            "message": "video 3 of 40: clip.mp4",
        }]
    )
    window = _indicator_window(controller)

    window._refresh_run_indicator()

    assert window._run_indicator_label.isHidden() is False
    assert "video 3 of 40: clip.mp4" in window._run_indicator_label.text()


def test_indicator_label_shows_no_message_text_before_one_arrives(qapp):
    controller = _LiveRunsController(
        [{"label": "Extract frames", "table_name": "videos", "message": None}]
    )
    window = _indicator_window(controller)

    window._refresh_run_indicator()

    # Would still pass if violated? No. A version that rendered "None"
    # into the sentence for an unset message would fail this.
    assert "None" not in window._run_indicator_label.text()
    assert window._run_indicator_label.text() == (
        'Running "Extract frames" on "videos"'
    )


def test_indicator_label_reflects_a_message_update_on_refresh(qapp):
    # Pull-based, like the rest of this section: the widget shows
    # whatever get_live_runs() reports as of the most recent refresh,
    # not a value cached from an earlier one.
    controller = _LiveRunsController(
        [{"label": "Extract blendshapes", "table_name": "frames", "message": None}]
    )
    window = _indicator_window(controller)
    window._refresh_run_indicator()
    assert "no face" not in window._run_indicator_label.text()

    controller._live_runs[0]["message"] = "no face detected in row r7"
    window._refresh_run_indicator()

    assert "no face detected in row r7" in window._run_indicator_label.text()


# ===========================================================================
# run-indicator-3: the Cancel control, at the same MainWindow seam.
#
# controller.py's cancel_run(), numbered_run_choices() and
# format_cancel_message() are covered by tests/test_result_delivery.py --
# these tests pin only the widget wiring: the Cancel button's visibility
# mirrors the run-indicator label's, a single live run is cancelled
# directly with no picker shown, more than one live run shows a pick-one
# dialog built from numbered_run_choices(), and dismissing that dialog
# cancels nothing.
#
# Written from the work-item specification, not the implementation.
# ===========================================================================

from controller import numbered_run_choices as _numbered_run_choices


def _fake_input_dialog(chosen_text, ok):
    """A QInputDialog stand-in: records every getItem() call and always
    returns (chosen_text, ok) instead of showing a real modal dialog."""
    calls: list = []

    class _FakeQInputDialog:
        @staticmethod
        def getItem(parent, title, label, items, current=0, editable=True):
            calls.append(
                {"title": title, "label": label, "items": list(items)}
            )
            return chosen_text, ok

    return _FakeQInputDialog, calls


def _fake_information_box():
    """A QMessageBox stand-in carrying only .information(), the one
    QMessageBox member _on_cancel_run_clicked() calls."""
    calls: list = []

    class _FakeQMessageBox:
        @staticmethod
        def information(parent, title, text):
            calls.append({"title": title, "text": text})

    return _FakeQMessageBox, calls


def test_cancel_button_hidden_with_no_live_runs(qapp):
    window = _indicator_window(_LiveRunsController([]))

    window._refresh_run_indicator()

    assert window._cancel_run_button.isHidden() is True


def test_cancel_button_shown_with_one_live_run(qapp):
    controller = _LiveRunsController(
        [{
            "operation_id": "op-1", "label": "Extract blendshapes",
            "table_name": "frames",
        }]
    )
    window = _indicator_window(controller)

    window._refresh_run_indicator()

    assert window._cancel_run_button.isHidden() is False


def test_clicking_cancel_with_no_live_runs_does_nothing(qapp, monkeypatch):
    controller = _LiveRunsController([])
    window = _indicator_window(controller)
    fake_input_dialog, input_calls = _fake_input_dialog("unused", True)
    fake_message_box, info_calls = _fake_information_box()
    monkeypatch.setattr(main_window_module, "QInputDialog", fake_input_dialog)
    monkeypatch.setattr(main_window_module, "QMessageBox", fake_message_box)

    window._on_cancel_run_clicked()

    assert controller.cancel_run_calls == []
    assert input_calls == []
    assert info_calls == []


def test_clicking_cancel_with_one_live_run_cancels_it_directly(
    qapp, monkeypatch,
):
    controller = _LiveRunsController(
        [{
            "operation_id": "op-1", "label": "Extract blendshapes",
            "table_name": "frames",
        }]
    )
    window = _indicator_window(controller)
    fake_input_dialog, input_calls = _fake_input_dialog("unused", True)
    fake_message_box, info_calls = _fake_information_box()
    monkeypatch.setattr(main_window_module, "QInputDialog", fake_input_dialog)
    monkeypatch.setattr(main_window_module, "QMessageBox", fake_message_box)

    window._on_cancel_run_clicked()

    # Would still pass if violated? No. Showing the picker even with only
    # one live run would leave input_calls non-empty; this asserts it is
    # skipped entirely and the single run is cancelled straight away.
    assert input_calls == []
    assert controller.cancel_run_calls == ["op-1"]
    assert len(info_calls) == 1
    assert "Extract blendshapes" in info_calls[0]["text"]


def test_clicking_cancel_with_two_live_runs_shows_a_numbered_picker(
    qapp, monkeypatch,
):
    controller = _LiveRunsController([
        {
            "operation_id": "op-1", "label": "Extract blendshapes",
            "table_name": "frames",
        },
        {
            "operation_id": "op-2", "label": "Extract blendshapes",
            "table_name": "frames",
        },
    ])
    window = _indicator_window(controller)
    # Two runs of the same operator on the same table read identically
    # without the numbering -- pick the SECOND numbered entry and assert
    # it is op-2, not op-1, that gets cancelled.
    expected_choices = _numbered_run_choices(controller.get_live_runs())
    chosen_text = expected_choices[1][2]
    fake_input_dialog, input_calls = _fake_input_dialog(chosen_text, True)
    fake_message_box, info_calls = _fake_information_box()
    monkeypatch.setattr(main_window_module, "QInputDialog", fake_input_dialog)
    monkeypatch.setattr(main_window_module, "QMessageBox", fake_message_box)

    window._on_cancel_run_clicked()

    assert len(input_calls) == 1
    assert input_calls[0]["items"] == [
        text for _op_id, _label, text in expected_choices
    ]
    assert controller.cancel_run_calls == ["op-2"]
    assert len(info_calls) == 1


def test_dismissing_the_picker_cancels_nothing(qapp, monkeypatch):
    controller = _LiveRunsController([
        {
            "operation_id": "op-1", "label": "Extract blendshapes",
            "table_name": "frames",
        },
        {
            "operation_id": "op-2", "label": "Extract blendshapes",
            "table_name": "frames",
        },
    ])
    window = _indicator_window(controller)
    fake_input_dialog, input_calls = _fake_input_dialog("does not matter", False)
    fake_message_box, info_calls = _fake_information_box()
    monkeypatch.setattr(main_window_module, "QInputDialog", fake_input_dialog)
    monkeypatch.setattr(main_window_module, "QMessageBox", fake_message_box)

    window._on_cancel_run_clicked()

    # Would still pass if violated? No. A version that cancelled the
    # first (or any) run regardless of the dialog's "ok" flag would leave
    # cancel_run_calls non-empty here.
    assert len(input_calls) == 1
    assert controller.cancel_run_calls == []
    assert info_calls == []


# ===========================================================================
# table-name-validation: a taken NewTableNameParameter name disables OK and
# shows a red message under the field, live, on every keystroke -- and
# re-enables OK the moment the name is changed to a free one.
# ===========================================================================

def _new_table_name_mode(*, default="segments"):
    """A TABLE mode with a single NewTableNameParameter field."""
    return ModeDescriptor(
        mode=ExecutionMode.TABLE,
        label="Cut into segments",
        inputs=(_SOURCE_INPUT,),
        parameters=(
            NewTableNameParameter(
                name="output_table", label="New table name", default=default,
            ),
        ),
        output=OutputSpec(creates_table=True),
    )


def test_ok_is_disabled_on_a_taken_name_and_re_enabled_on_a_free_one(qapp):
    dialog = ParameterDialog(
        _new_table_name_mode(),
        _COLUMNS_BY_INPUT,
        existing_table_names=("frames", "segments"),
    )
    field = dialog._widgets["output_table"]

    # The dialog is constructed but never exec()'d/shown() (the pattern
    # every other widget test in this file follows), so a child widget's
    # isVisible() always reads False regardless of what setVisible() was
    # called with -- isHidden() is the flag this code actually set. See
    # test_ok_re_enables_when_an_allowed_value_is_picked above for the
    # same distinction on the shared _message_label.
    error_label = dialog._table_name_error_labels["output_table"]

    # The declared default itself is already taken -- table-name-validation
    # is a live check, not something limited to a hand-typed collision, so
    # this must catch it on construction, before any edit.
    assert dialog._ok_button.isEnabled() is False
    assert error_label.isHidden() is False
    assert "segments" in error_label.text()

    # Editing to another taken name changes nothing about the block.
    field.setText("frames")
    assert dialog._ok_button.isEnabled() is False

    # Would still pass if violated? No. If the live check only ran once,
    # at construction, this edit to a free name would leave OK disabled.
    field.setText("segments_2")
    assert dialog._ok_button.isEnabled() is True
    assert error_label.isHidden() is True

    # And back to a taken name blocks again.
    field.setText("frames")
    assert dialog._ok_button.isEnabled() is False


def test_taken_name_check_never_reaches_refine_form(qapp):
    # An operator must never be given a way to learn which tables exist
    # (CLAUDE.md). The advice_provider is the operator's own refine_form;
    # this proves the table-name collision is decided without ever
    # consulting it -- the provider here declares no opinion at all about
    # output_table, and OK is still blocked on a taken name.
    calls: list[dict] = []

    def _advice_provider(raw_values):
        calls.append(dict(raw_values))
        return FormAdvice()

    dialog = ParameterDialog(
        _new_table_name_mode(default="free_name"),
        _COLUMNS_BY_INPUT,
        advice_provider=_advice_provider,
        existing_table_names=("segments",),
    )
    assert dialog._ok_button.isEnabled() is True

    dialog._widgets["output_table"].setText("segments")
    assert dialog._ok_button.isEnabled() is False

    # The provider was called (it drives the rest of the form), but never
    # told "the name is taken" -- it only ever sees the raw field values,
    # the same thing it always saw, and returned no opinion either way.
    assert calls
    assert all(call.get("output_table") in ("free_name", "segments") for call in calls)


def test_default_existing_table_names_leaves_the_form_unblocked(qapp):
    # No existing_table_names given (the pre-item behaviour): the field
    # never reads as "taken", whatever the default is.
    dialog = ParameterDialog(_new_table_name_mode(), _COLUMNS_BY_INPUT)
    assert dialog._ok_button.isEnabled() is True
    # isHidden(), not isVisible() -- the dialog is never shown, so
    # isVisible() reads False regardless of what setVisible() was called
    # with (see test_ok_is_disabled_on_a_taken_name_and_re_enabled_on_a_free_one).
    assert dialog._table_name_error_labels["output_table"].isHidden() is True


# ===========================================================================
# table-name-validation, round 2: MainWindow._show_scope_and_params_dialog
# actually wires existing_table_names through, and the REAL dialog it
# builds genuinely blocks OK on a taken name -- not just that the argument
# was forwarded.
# ===========================================================================

class _CapturingRealParameterDialog(ParameterDialog):
    """The REAL ParameterDialog -- real widgets, real OK-button and
    per-field label logic -- with exec() short-circuited to "cancelled"
    so the test never enters an actual modal loop, and every instance
    built kept so the test can inspect it afterwards.

    Deliberately NOT a spy that skips construction (like
    _ParameterDialogSpy above): the whole point here is to prove the real
    _render_table_name_validation machinery reacts to what MainWindow
    passed in, which only the real class can show.
    """

    instances: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _CapturingRealParameterDialog.instances.append(self)

    def exec(self):
        return 0


def test_the_real_dialog_built_by_main_window_blocks_ok_on_a_taken_name(
    qapp, monkeypatch,
):
    # Proves the WIRING produces the real BEHAVIOUR: MainWindow calls the
    # controller's get_table_names() and the REAL ParameterDialog it
    # builds from the result genuinely blocks OK when the declared
    # default collides with an existing table name. A test that only
    # checked "the constructor received existing_table_names=[...]"
    # would still pass if _render_table_name_validation were deleted
    # entirely; this would not.
    from operators.descriptor import OperatorDescriptor

    descriptor = OperatorDescriptor(
        name="segment_demo",
        version="1.0",
        description="cuts into segments",
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.TABLE,
                label="Cut into segments",
                inputs=(_SOURCE_INPUT,),
                parameters=(
                    NewTableNameParameter(
                        name="output_table", label="New table name",
                        default="segments",
                    ),
                ),
                output=OutputSpec(creates_table=True),
            ),
        ),
    )
    controller = _FakeController(_Operator(descriptor, name="segment_demo"))
    # "segments" -- the declared default -- is already taken.
    controller.get_table_names = lambda: ["frames", "segments"]

    _CapturingRealParameterDialog.instances.clear()
    monkeypatch.setattr(main_window_module, "RunOperatorDialog", _FakeScopeDialog)
    monkeypatch.setattr(
        main_window_module, "ParameterDialog", _CapturingRealParameterDialog
    )
    # A bare MainWindow.__new__() (the pattern _bare_window uses above)
    # never runs QMainWindow's own __init__, so it is not a valid Qt
    # parent object -- fine for the fake dialog stand-ins, which never
    # touch `parent`, but the REAL ParameterDialog's QDialog.__init__(parent)
    # needs one. QMainWindow.__init__ is run directly instead of
    # MainWindow's own (which would build the whole real UI).
    from PySide6.QtWidgets import QMainWindow
    window = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(window)
    window._controller = controller
    monkeypatch.setattr(window, "_collect_selected_row_ids", lambda: [], raising=False)
    monkeypatch.setattr(window, "_collect_visible_row_ids", lambda: [], raising=False)
    monkeypatch.setattr(window, "_on_error", lambda *a: None, raising=False)

    window._show_scope_and_params_dialog("segment_demo", ExecutionMode.TABLE)

    # Would still pass if violated? No. If MainWindow forgot to pass
    # existing_table_names, or passed the wrong thing (an empty list, the
    # visible table's name instead of the project's), the real dialog
    # would build with OK enabled and no message shown.
    assert len(_CapturingRealParameterDialog.instances) == 1
    dialog = _CapturingRealParameterDialog.instances[0]
    assert dialog._ok_button.isEnabled() is False
    error_label = dialog._table_name_error_labels["output_table"]
    assert error_label.isHidden() is False
    assert "segments" in error_label.text()


def test_the_real_dialog_built_by_main_window_allows_a_free_name(qapp, monkeypatch):
    # The mirror case: nothing collides, so OK opens enabled.
    from operators.descriptor import OperatorDescriptor

    descriptor = OperatorDescriptor(
        name="segment_demo2",
        version="1.0",
        description="cuts into segments",
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.TABLE,
                label="Cut into segments",
                inputs=(_SOURCE_INPUT,),
                parameters=(
                    NewTableNameParameter(
                        name="output_table", label="New table name",
                        default="segments",
                    ),
                ),
                output=OutputSpec(creates_table=True),
            ),
        ),
    )
    controller = _FakeController(_Operator(descriptor, name="segment_demo2"))
    controller.get_table_names = lambda: ["frames"]

    _CapturingRealParameterDialog.instances.clear()
    monkeypatch.setattr(main_window_module, "RunOperatorDialog", _FakeScopeDialog)
    monkeypatch.setattr(
        main_window_module, "ParameterDialog", _CapturingRealParameterDialog
    )
    from PySide6.QtWidgets import QMainWindow
    window = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(window)
    window._controller = controller
    monkeypatch.setattr(window, "_collect_selected_row_ids", lambda: [], raising=False)
    monkeypatch.setattr(window, "_collect_visible_row_ids", lambda: [], raising=False)
    monkeypatch.setattr(window, "_on_error", lambda *a: None, raising=False)

    window._show_scope_and_params_dialog("segment_demo2", ExecutionMode.TABLE)

    dialog = _CapturingRealParameterDialog.instances[0]
    assert dialog._ok_button.isEnabled() is True
    assert dialog._table_name_error_labels["output_table"].isHidden() is True


# ===========================================================================
# table-name-validation round 3: ui/merge_report_dialog.py's expand-table-name
# box gets the same red-text treatment. Placed here, not in
# tests/test_merge_expansion.py, because these tests realise a real
# MergeReportDialog widget -- tests/test_merge_expansion.py stays Qt-free
# (it only calls Layer A functions on a duck-typed report) and this file is
# already isolated in run_tests.py's WIDGET_MODULES for exactly that reason;
# adding a newly-widget-realising module elsewhere would need run_tests.py
# updated too, which is outside this item's allowed files.
# ===========================================================================

class _FakeExpandReport:
    """Duck-typed stand-in for MergeReport -- ui/merge_report_dialog.py
    never imports the real class (see its own module docstring)."""

    def __init__(self, **kwargs):
        self.target_table = "source"
        self.total_csv_rows = 5
        self.total_target_rows = 2
        self.matched_rows = 5
        self.unmatched_target_rows = []
        self.unmatched_csv_rows = []
        self.duplicate_keys_target = []
        self.duplicate_keys_csv = []
        self.would_expand = ["p07"]
        self.float_key_warning = None
        self.expand_table_name = "source_expanded"
        self.expand_row_count = 5
        self.expand_carried_columns = []
        self.__dict__.update(kwargs)


def test_merge_report_dialog_blocks_proceed_on_a_taken_expand_name(qapp):
    from ui.merge_report_dialog import MergeReportDialog

    report = _FakeExpandReport()
    dialog = MergeReportDialog(
        report, existing_table_names=("frames", "source_expanded"),
    )

    # "source_expanded" (the report's own suggestion) is taken, so the box
    # opens on the resolved, genuinely free name -- resolve_table_name can
    # never itself open the box already blocked.
    assert dialog._name_box.text() == "source_expanded_1"
    assert dialog._proceed_btn.isEnabled() is True
    assert dialog._name_error_label.isHidden() is True

    # The researcher types over it with a name that IS taken.
    dialog._name_box.setText("frames")
    assert dialog._proceed_btn.isEnabled() is False
    assert dialog._name_error_label.isHidden() is False
    assert "frames" in dialog._name_error_label.text()

    # Would still pass if violated? No. If the live check only ran once,
    # at construction, this edit to a free name would leave Proceed
    # disabled.
    dialog._name_box.setText("participant_trials")
    assert dialog._proceed_btn.isEnabled() is True
    assert dialog._name_error_label.isHidden() is True

    # A blank name is refused too, even though nothing collides.
    dialog._name_box.setText("   ")
    assert dialog._proceed_btn.isEnabled() is False
    assert dialog._name_error_label.isHidden() is False
    # table-name-validation round 6: the disabled button's own label must
    # not read "Create ''" -- a blank name gets a plain fallback label
    # instead of a quoted empty string.
    assert dialog._proceed_btn.text() == "Create the new table"


def test_merge_report_dialog_has_no_name_box_for_an_ordinary_merge(qapp):
    from ui.merge_report_dialog import MergeReportDialog

    report = _FakeExpandReport(would_expand=[])
    dialog = MergeReportDialog(report, existing_table_names=("frames",))
    assert dialog._name_box is None
    assert dialog._name_error_label is None


def test_a_chosen_expand_name_becomes_the_stored_tables_name(qapp, tmp_path):
    # table-name-validation round 4: proves the WIRING produces the real
    # BEHAVIOUR through the NEW route -- a real Dataset, a real
    # merge_csv()-produced report, a real MergeReportDialog the
    # researcher edits, and Dataset.confirm_merge() actually storing the
    # table under the EXPLICIT expand_table_name argument
    # ui/main_window.py reads off dialog.chosen_expand_table_name --
    # never by way of a mutated report field, which this test also
    # pins by asserting the report's own attribute is untouched.
    import pandas as pd
    from models.dataset import Dataset
    from ui.merge_report_dialog import MergeReportDialog

    ds = Dataset()
    ds._accept_table(
        "source",
        pd.DataFrame({
            "row_id": ["1", "2"],
            "participant_id": ["p07", "p08"],
        }),
        source="test",
    )
    csv_path = tmp_path / "trials.csv"
    pd.DataFrame({
        "participant_id": ["p07", "p07", "p08", "p08"],
        "trial_num": [1, 2, 1, 2],
    }).to_csv(csv_path, index=False)

    report = ds.merge_csv(
        csv_path, target_table="source",
        csv_key="participant_id", target_key="participant_id",
    )
    assert report.would_expand
    assert report.expand_table_name == "source_expanded"

    dialog = MergeReportDialog(
        report, existing_table_names=tuple(ds.list_tables()),
    )
    # Nothing collides yet, so the box opens on the bare default.
    assert dialog._name_box.text() == "source_expanded"

    # The researcher renames it and proceeds.
    dialog._name_box.setText("participant_trials")
    assert dialog._proceed_btn.isEnabled() is True
    dialog._on_proceed()

    # Would still pass if violated? No. This is the whole point of round
    # 4: the dialog must never write to the report it was given. If it
    # still did, this would read "participant_trials" instead.
    assert report.expand_table_name == "source_expanded"

    chosen = dialog.chosen_expand_table_name
    assert chosen == "participant_trials"

    ds.confirm_merge(report, chosen)

    # Would still pass if violated? No -- this is the route the report
    # mutation used to prove, now proven through the explicit argument
    # instead: the test cannot pass by the old route, because nothing
    # about `report` was ever changed.
    assert "participant_trials" in ds.list_tables()
    assert "source_expanded" not in ds.list_tables()
    # The target table itself is unaffected, as confirm_merge already
    # guarantees for every expansion offer.
    assert len(ds.get_table("source")) == 2


# ===========================================================================
# table-name-validation round 6: FakeController.confirm_merge() -- round 5's
# unconditional raise broke the ordinary (non-expanding) merge path in
# --fake-data mode, which worked before that round. Only an expansion offer
# has nothing honest to fake, so only that case raises.
# ===========================================================================

def test_fake_controller_confirm_merge_restores_the_ordinary_path_and_still_refuses_an_expand_offer(
    qapp,
):
    from media.resolver import MediaResolver
    from models.dataset import MergeReport
    from ui.fake_controller import FakeController

    resolver = MediaResolver(max_open_decoders=2)
    controller = FakeController(PROJECT_ROOT / "test_images", resolver=resolver)

    columns_updates: list = []
    controller.columns_updated.connect(columns_updates.append)

    # An ordinary merge (would_expand empty, matching load_csv()'s own
    # fake report) must behave exactly as it did before round 5 -- no
    # exception, the same fake refresh.
    ordinary_report = MergeReport(target_table="frames", would_expand=[])
    controller.confirm_merge(ordinary_report)

    # Would still pass if violated? No. Before this fix, this call always
    # raised NotImplementedError regardless of would_expand.
    assert columns_updates

    # An expansion offer still has nothing honest to fake, and still
    # refuses -- unchanged from round 5, just no longer reached for the
    # ordinary case above.
    expand_report = MergeReport(target_table="frames", would_expand=["p07"])
    with pytest.raises(NotImplementedError, match="does not support merging"):
        controller.confirm_merge(expand_report, "chosen_name")
