"""
tests/test_parameter_form.py

Layer A of the generated parameter dialog (P1.12e-1) -- the module-level,
Qt-free functions in ui/parameter_dialog.py:

  * build_field_specs(mode_descriptor, columns_by_input) -> tuple[FieldSpec]
  * collect_parameters(field_specs, raw_values) -> dict
  * ParameterFormError

These carry every coercion, filter and validation rule and all the
researcher-facing wording, so they are testable with no QApplication. This
file realises no widget and names no widget marker, so run_tests.py keeps
it in the combined group (as tests/test_operator_run_wiring.py already
imports a ui module there). The import of ui.parameter_dialog pulls in the
Qt binding transitively, but nothing here builds or shows a widget.

Written from the work-item specification, not the implementation.

Run with:
    python -m pytest tests/test_parameter_form.py
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

from ui.parameter_dialog import (
    FieldSpec,
    ParameterFormError,
    build_field_specs,
    collect_parameters,
)


# ---------------------------------------------------------------------------
# Helpers -- build a minimal well-formed ModeDescriptor around some
# parameters. DISPLAY mode needs no output columns, which keeps the
# fixtures small.
# ---------------------------------------------------------------------------

_SOURCE_INPUT = InputSpec(
    name="src", label="Source table", kind=InputKind.ACTIVE_TABLE
)
_OTHER_INPUT = InputSpec(
    name="other", label="Other table", kind=InputKind.NAMED_TABLE
)


def _mode(parameters, inputs=()):
    return ModeDescriptor(
        mode=ExecutionMode.DISPLAY,
        label="Test mode",
        inputs=tuple(inputs),
        parameters=tuple(parameters),
        output=OutputSpec(is_display_only=True),
    )


def _one_field(parameter, inputs=(), columns_by_input=None):
    specs = build_field_specs(
        _mode([parameter], inputs=inputs), columns_by_input or {}
    )
    assert len(specs) == 1
    return specs[0]


# ===========================================================================
# build_field_specs -- one FieldSpec per parameter kind
# ===========================================================================

def test_number_parameter_becomes_a_number_field():
    field = _one_field(
        NumberParameter(
            name="frame_step",
            label="Frame step",
            help_text="How many frames to skip",
            minimum=1,
            maximum=100,
            decimals=2,
            default=5,
        )
    )
    assert isinstance(field, FieldSpec)
    assert field.name == "frame_step"
    assert field.kind == "number"
    assert field.label == "Frame step"
    assert field.help_text == "How many frames to skip"
    assert field.required is True
    assert field.default == 5
    assert field.minimum == 1
    assert field.maximum == 100
    assert field.decimals == 2


def test_text_parameter_becomes_a_text_field():
    field = _one_field(
        TextParameter(name="title", label="Chart title", default="Untitled")
    )
    assert field.kind == "text"
    assert field.name == "title"
    assert field.default == "Untitled"


def test_boolean_parameter_becomes_a_boolean_field():
    field = _one_field(
        BooleanParameter(name="normalise", label="Normalise values", default=True)
    )
    assert field.kind == "boolean"
    assert field.default is True


def test_choice_parameter_becomes_a_choice_field_with_its_pairs():
    field = _one_field(
        ChoiceParameter(
            name="chart_type",
            label="Chart type",
            choices=(("scatter", "Scatter"), ("bar", "Bar")),
            default="bar",
        )
    )
    assert field.kind == "choice"
    assert field.choices == (("scatter", "Scatter"), ("bar", "Bar"))
    assert field.default == "bar"


def test_column_parameter_becomes_a_column_field():
    field = _one_field(
        ColumnParameter(
            name="value_column",
            label="Value column",
            from_input="src",
            allow_multiple=True,
        ),
        inputs=[_SOURCE_INPUT],
        columns_by_input={"src": [("age", "numeric"), ("name", "text")]},
    )
    assert field.kind == "column"
    assert field.allow_multiple is True
    assert field.column_names == ("age", "name")


def test_new_table_name_parameter_becomes_a_new_table_name_field():
    field = _one_field(
        NewTableNameParameter(
            name="output_table", label="New table name", default="segments"
        )
    )
    assert field.kind == "new_table_name"
    assert field.name == "output_table"
    assert field.default == "segments"


# ===========================================================================
# build_field_specs -- ColumnParameter column offering and tag filtering
# ===========================================================================

def test_column_parameter_offers_only_its_own_inputs_columns():
    # Two inputs; the parameter names "src". Only "src" columns are offered,
    # even though the mapping also carries "other" columns.
    parameter = ColumnParameter(
        name="col", label="Column", from_input="src"
    )
    field = _one_field(
        parameter,
        inputs=[_SOURCE_INPUT, _OTHER_INPUT],
        columns_by_input={
            "src": [("age", "numeric"), ("name", "text")],
            "other": [("weight", "numeric"), ("height", "numeric")],
        },
    )
    assert field.column_names == ("age", "name")
    assert "weight" not in field.column_names
    assert "height" not in field.column_names


def test_required_tags_filters_the_offered_columns():
    parameter = ColumnParameter(
        name="col",
        label="Numeric column",
        from_input="src",
        required_tags=("numeric",),
    )
    field = _one_field(
        parameter,
        inputs=[_SOURCE_INPUT],
        columns_by_input={
            "src": [
                ("age", "numeric"),
                ("name", "text"),
                ("score", "numeric"),
                ("clip", "media_path"),
            ]
        },
    )
    # Only the two numeric columns survive the filter.
    assert field.column_names == ("age", "score")


def test_empty_required_tags_offers_every_column_of_the_input():
    parameter = ColumnParameter(
        name="col", label="Any column", from_input="src", required_tags=()
    )
    field = _one_field(
        parameter,
        inputs=[_SOURCE_INPUT],
        columns_by_input={
            "src": [("age", "numeric"), ("name", "text"), ("clip", "media_path")]
        },
    )
    assert field.column_names == ("age", "name", "clip")


# ===========================================================================
# build_field_specs -- field order follows declaration order
# ===========================================================================

def test_field_order_matches_the_declaration_order():
    parameters = [
        TextParameter(name="first", label="First"),
        BooleanParameter(name="second", label="Second"),
        NumberParameter(name="third", label="Third"),
        ChoiceParameter(
            name="fourth", label="Fourth", choices=(("x", "X"),)
        ),
    ]
    specs = build_field_specs(_mode(parameters), {})
    assert [spec.name for spec in specs] == [
        "first",
        "second",
        "third",
        "fourth",
    ]


# ===========================================================================
# collect_parameters -- coercion
# ===========================================================================

def test_integer_number_is_coerced_to_int():
    specs = build_field_specs(
        _mode([NumberParameter(name="step", label="Step", decimals=0)]), {}
    )
    # A spin box can hand back a float even for an integer parameter.
    result = collect_parameters(specs, {"step": 3.0})
    assert result == {"step": 3}
    assert isinstance(result["step"], int)


def test_decimal_number_is_coerced_to_float():
    specs = build_field_specs(
        _mode([NumberParameter(name="threshold", label="Threshold", decimals=2)]),
        {},
    )
    result = collect_parameters(specs, {"threshold": 4})
    assert result == {"threshold": 4.0}
    assert isinstance(result["threshold"], float)


def test_boolean_comes_back_as_a_bool():
    specs = build_field_specs(
        _mode([BooleanParameter(name="flag", label="Flag")]), {}
    )
    assert collect_parameters(specs, {"flag": True}) == {"flag": True}


def test_choice_value_passes_straight_through():
    specs = build_field_specs(
        _mode(
            [
                ChoiceParameter(
                    name="kind",
                    label="Kind",
                    choices=(("a", "A"), ("b", "B")),
                )
            ]
        ),
        {},
    )
    assert collect_parameters(specs, {"kind": "b"}) == {"kind": "b"}


# ===========================================================================
# collect_parameters -- multi-column selection
# ===========================================================================

def test_multi_column_selection_comes_back_as_a_tuple():
    parameter = ColumnParameter(
        name="cols", label="Columns", from_input="src", allow_multiple=True
    )
    specs = build_field_specs(
        _mode([parameter], inputs=[_SOURCE_INPUT]),
        {"src": [("a", "numeric"), ("b", "numeric"), ("c", "numeric")]},
    )
    result = collect_parameters(specs, {"cols": ["a", "c"]})
    assert result == {"cols": ("a", "c")}
    assert isinstance(result["cols"], tuple)


# ===========================================================================
# collect_parameters -- required empty raises, optional unset is omitted
# ===========================================================================

def test_required_parameter_left_empty_raises_naming_the_parameter():
    specs = build_field_specs(
        _mode([TextParameter(name="title", label="Chart title")]), {}
    )
    with pytest.raises(ParameterFormError) as excinfo:
        collect_parameters(specs, {"title": "   "})
    # The message names the field the researcher must fill in.
    assert "Chart title" in str(excinfo.value)


def test_required_choice_left_unchosen_raises():
    # The dialog gives a required, no-default choice a blank first entry
    # whose data is None; collect_parameters must treat that as "empty".
    specs = build_field_specs(
        _mode(
            [
                ChoiceParameter(
                    name="chart_type",
                    label="Chart type",
                    choices=(("scatter", "Scatter"), ("bar", "Bar")),
                )
            ]
        ),
        {},
    )
    with pytest.raises(ParameterFormError) as excinfo:
        collect_parameters(specs, {"chart_type": None})
    assert "Chart type" in str(excinfo.value)


def test_required_column_left_unchosen_raises():
    parameter = ColumnParameter(
        name="col", label="Value column", from_input="src"
    )
    specs = build_field_specs(
        _mode([parameter], inputs=[_SOURCE_INPUT]),
        {"src": [("age", "numeric")]},
    )
    with pytest.raises(ParameterFormError) as excinfo:
        collect_parameters(specs, {"col": None})
    assert "Value column" in str(excinfo.value)


def test_optional_parameter_left_unset_is_omitted_entirely():
    specs = build_field_specs(
        _mode(
            [TextParameter(name="note", label="Note", required=False)]
        ),
        {},
    )
    result = collect_parameters(specs, {"note": ""})
    # Not passed as an empty string -- absent from the dict.
    assert result == {}
    assert "note" not in result


def test_optional_multi_column_left_empty_is_omitted():
    parameter = ColumnParameter(
        name="cols",
        label="Extra columns",
        from_input="src",
        allow_multiple=True,
        required=False,
    )
    specs = build_field_specs(
        _mode([parameter], inputs=[_SOURCE_INPUT]),
        {"src": [("a", "numeric")]},
    )
    assert collect_parameters(specs, {"cols": []}) == {}


def test_a_supplied_optional_parameter_is_kept():
    specs = build_field_specs(
        _mode(
            [TextParameter(name="note", label="Note", required=False)]
        ),
        {},
    )
    assert collect_parameters(specs, {"note": " hello "}) == {"note": "hello"}
