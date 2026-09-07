"""
tests/test_operator_descriptor.py

Guards the operator descriptor vocabulary in operators/descriptor.py.

The module is pure data: a set of frozen dataclasses whose __post_init__
methods refuse a malformed descriptor by raising OperatorDescriptorError.
There is one test per validation rule the design lists, each named for the
rule it guards and each asserting the raise. On top of that:

  * an immutability test -- a constructed descriptor cannot be mutated and
    its collections are tuples;
  * a source-scan test -- descriptor.py imports only the standard library,
    nothing from Qt / pandas / numpy / PIL / cv2 / mediapipe and nothing
    from elsewhere in this repo. The scan reads the file text (and parses
    it with ast); it never inspects sys.modules, because by the time this
    test runs another test in the same process may already have imported
    those libraries;
  * a happy-path test that builds a realistic three-mode descriptor and
    reads it back through mode_for().

This module imports no Qt and no pandas, so run_tests.py keeps it in the
combined non-widget group.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from operators.descriptor import (
    BooleanParameter,
    ChoiceParameter,
    ColumnParameter,
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModeDescriptor,
    ModelLifecycle,
    NewTableNameParameter,
    NumberParameter,
    OperatorDescriptor,
    OperatorDescriptorError,
    OutputColumn,
    OutputSpec,
    ParameterSpec,
    TextParameter,
)


# ---------------------------------------------------------------------------
# Builders for the parts a test is not itself exercising, so each test only
# supplies the one malformed value it is about.
# ---------------------------------------------------------------------------

def _ok_columns_output() -> OutputSpec:
    return OutputSpec(columns=(OutputColumn(name="score", type_tag="numeric"),))


def _ok_table_output() -> OutputSpec:
    return OutputSpec(creates_table=True)


def _ok_display_output() -> OutputSpec:
    return OutputSpec(is_display_only=True)


def _ok_input(name: str = "rows") -> InputSpec:
    return InputSpec(name=name, label="Rows", kind=InputKind.ACTIVE_TABLE)


def _ok_columns_mode() -> ModeDescriptor:
    return ModeDescriptor(
        mode=ExecutionMode.COLUMNS,
        label="Compute score per row",
        inputs=(_ok_input(),),
        output=_ok_columns_output(),
    )


# ===========================================================================
# InputSpec
# ===========================================================================

def test_input_spec_rejects_empty_name():
    with pytest.raises(OperatorDescriptorError):
        InputSpec(name="", label="Rows", kind=InputKind.ACTIVE_TABLE)


def test_input_spec_rejects_non_identifier_name():
    with pytest.raises(OperatorDescriptorError):
        InputSpec(name="not a name", label="Rows", kind=InputKind.ACTIVE_TABLE)


def test_identifier_field_rejects_python_keyword():
    # str.isidentifier() is true for reserved words, so the keyword guard
    # is what stops a name like "class" or "None".
    with pytest.raises(OperatorDescriptorError):
        InputSpec(name="class", label="Rows", kind=InputKind.ACTIVE_TABLE)


def test_input_spec_rejects_empty_label():
    with pytest.raises(OperatorDescriptorError):
        InputSpec(name="rows", label="   ", kind=InputKind.ACTIVE_TABLE)


# ===========================================================================
# ParameterSpec base rules (exercised through a concrete subclass)
# ===========================================================================

def test_parameter_rejects_empty_name():
    with pytest.raises(OperatorDescriptorError):
        TextParameter(name="", label="Prefix")


def test_parameter_rejects_non_identifier_name():
    with pytest.raises(OperatorDescriptorError):
        TextParameter(name="1prefix", label="Prefix")


def test_parameter_rejects_empty_label():
    with pytest.raises(OperatorDescriptorError):
        TextParameter(name="prefix", label="")


# ===========================================================================
# NumberParameter
# ===========================================================================

def test_number_parameter_rejects_minimum_above_maximum():
    with pytest.raises(OperatorDescriptorError):
        NumberParameter(name="w", label="Window", minimum=10, maximum=5)


def test_number_parameter_rejects_default_below_minimum():
    with pytest.raises(OperatorDescriptorError):
        NumberParameter(name="w", label="Window", minimum=0, maximum=10, default=-1)


def test_number_parameter_rejects_default_above_maximum():
    with pytest.raises(OperatorDescriptorError):
        NumberParameter(name="w", label="Window", minimum=0, maximum=10, default=99)


def test_number_parameter_rejects_negative_decimals():
    with pytest.raises(OperatorDescriptorError):
        NumberParameter(name="w", label="Window", decimals=-1)


# ===========================================================================
# ChoiceParameter
# ===========================================================================

def test_choice_parameter_rejects_no_choices():
    with pytest.raises(OperatorDescriptorError):
        ChoiceParameter(name="mode", label="Mode", choices=())


def test_choice_parameter_rejects_duplicate_choice_values():
    with pytest.raises(OperatorDescriptorError):
        ChoiceParameter(
            name="mode",
            label="Mode",
            choices=(("a", "First"), ("a", "Second")),
        )


def test_choice_parameter_rejects_default_not_among_choices():
    with pytest.raises(OperatorDescriptorError):
        ChoiceParameter(
            name="mode",
            label="Mode",
            choices=(("a", "First"), ("b", "Second")),
            default="c",
        )


def test_choice_parameter_rejects_choice_entry_that_is_not_a_pair():
    # A malformed entry must raise our own error type, not a bare
    # ValueError from tuple unpacking.
    with pytest.raises(OperatorDescriptorError):
        ChoiceParameter(
            name="mode",
            label="Mode",
            choices=(("a", "First"), ("b",)),
        )


# ===========================================================================
# OutputColumn
# ===========================================================================

def test_output_column_rejects_empty_name():
    with pytest.raises(OperatorDescriptorError):
        OutputColumn(name="", type_tag="numeric")


def test_output_column_rejects_empty_type_tag():
    with pytest.raises(OperatorDescriptorError):
        OutputColumn(name="score", type_tag="")


# ===========================================================================
# OutputSpec -- exactly one of the three shapes
# ===========================================================================

def test_output_spec_rejects_no_shape_declared():
    with pytest.raises(OperatorDescriptorError):
        OutputSpec()


def test_output_spec_rejects_columns_and_creates_table_together():
    with pytest.raises(OperatorDescriptorError):
        OutputSpec(
            columns=(OutputColumn(name="score", type_tag="numeric"),),
            creates_table=True,
        )


def test_output_spec_rejects_columns_and_display_together():
    with pytest.raises(OperatorDescriptorError):
        OutputSpec(
            columns=(OutputColumn(name="score", type_tag="numeric"),),
            is_display_only=True,
        )


def test_output_spec_rejects_creates_table_and_display_together():
    with pytest.raises(OperatorDescriptorError):
        OutputSpec(creates_table=True, is_display_only=True)


def test_output_spec_rejects_duplicate_column_names():
    with pytest.raises(OperatorDescriptorError):
        OutputSpec(
            columns=(
                OutputColumn(name="score", type_tag="numeric"),
                OutputColumn(name="score", type_tag="text"),
            )
        )


# ===========================================================================
# ModeDescriptor
# ===========================================================================

def test_mode_rejects_empty_label():
    with pytest.raises(OperatorDescriptorError):
        ModeDescriptor(
            mode=ExecutionMode.COLUMNS,
            label="",
            inputs=(_ok_input(),),
            output=_ok_columns_output(),
        )


def test_mode_rejects_duplicate_input_names():
    with pytest.raises(OperatorDescriptorError):
        ModeDescriptor(
            mode=ExecutionMode.COLUMNS,
            label="Compute score",
            inputs=(_ok_input("rows"), _ok_input("rows")),
            output=_ok_columns_output(),
        )


def test_mode_rejects_duplicate_parameter_names():
    with pytest.raises(OperatorDescriptorError):
        ModeDescriptor(
            mode=ExecutionMode.COLUMNS,
            label="Compute score",
            inputs=(_ok_input(),),
            parameters=(
                TextParameter(name="p", label="One"),
                BooleanParameter(name="p", label="Two"),
            ),
            output=_ok_columns_output(),
        )


def test_mode_rejects_column_parameter_pointing_at_unknown_input():
    with pytest.raises(OperatorDescriptorError):
        ModeDescriptor(
            mode=ExecutionMode.COLUMNS,
            label="Compute score",
            inputs=(_ok_input("rows"),),
            parameters=(
                ColumnParameter(
                    name="which_col", label="Column", from_input="other_table"
                ),
            ),
            output=_ok_columns_output(),
        )


def test_mode_columns_requires_non_empty_output_columns():
    with pytest.raises(OperatorDescriptorError):
        ModeDescriptor(
            mode=ExecutionMode.COLUMNS,
            label="Compute score",
            inputs=(_ok_input(),),
            output=_ok_table_output(),
        )


def test_mode_table_requires_creates_table_output():
    with pytest.raises(OperatorDescriptorError):
        ModeDescriptor(
            mode=ExecutionMode.TABLE,
            label="Aggregate",
            inputs=(_ok_input(),),
            output=_ok_display_output(),
        )


def test_mode_display_requires_is_display_only_output():
    with pytest.raises(OperatorDescriptorError):
        ModeDescriptor(
            mode=ExecutionMode.DISPLAY,
            label="Quick view",
            inputs=(_ok_input(),),
            output=_ok_table_output(),
        )


# ===========================================================================
# ModeDescriptor.media_requirement
# ===========================================================================

def test_mode_media_requirement_defaults_to_metadata():
    # A mode that does not declare a media requirement gets METADATA, so an
    # operator that forgets to declare is handed no media rather than
    # silently paying for a decode.
    mode = ModeDescriptor(
        mode=ExecutionMode.COLUMNS,
        label="Compute score",
        inputs=(_ok_input(),),
        output=_ok_columns_output(),
    )
    assert mode.media_requirement is MediaRequirement.METADATA


def test_mode_media_requirement_accepts_each_member():
    # Every one of the five declared requirements can be set and read back.
    for requirement in (
        MediaRequirement.METADATA,
        MediaRequirement.FRAME,
        MediaRequirement.VIDEO_SPAN,
        MediaRequirement.AUDIO_SPAN,
        MediaRequirement.ADDRESS,
    ):
        mode = ModeDescriptor(
            mode=ExecutionMode.COLUMNS,
            label="Compute score",
            inputs=(_ok_input(),),
            media_requirement=requirement,
            output=_ok_columns_output(),
        )
        assert mode.media_requirement is requirement


def test_media_requirement_values_match_operator_claude_md():
    # The five string values are the contract operators/CLAUDE.md documents.
    assert MediaRequirement.METADATA.value == "metadata"
    assert MediaRequirement.FRAME.value == "frame"
    assert MediaRequirement.VIDEO_SPAN.value == "video_span"
    assert MediaRequirement.AUDIO_SPAN.value == "audio_span"
    assert MediaRequirement.ADDRESS.value == "address"


def test_table_mode_may_declare_address_media_requirement():
    # A TABLE-mode operator that resolves addresses itself (video frame
    # extraction is the real example) legitimately declares ADDRESS. This
    # test fails if someone later adds a mode/media cross-check, which this
    # follow-up explicitly forbids.
    mode = ModeDescriptor(
        mode=ExecutionMode.TABLE,
        label="Extract frames",
        inputs=(_ok_input(),),
        media_requirement=MediaRequirement.ADDRESS,
        output=_ok_table_output(),
    )
    assert mode.mode is ExecutionMode.TABLE
    assert mode.media_requirement is MediaRequirement.ADDRESS


# ===========================================================================
# OperatorDescriptor
# ===========================================================================

def test_operator_rejects_empty_name():
    with pytest.raises(OperatorDescriptorError):
        OperatorDescriptor(
            name="",
            version="1.0",
            description="Does a thing.",
            modes=(_ok_columns_mode(),),
        )


def test_operator_rejects_non_identifier_name():
    with pytest.raises(OperatorDescriptorError):
        OperatorDescriptor(
            name="face metrics",
            version="1.0",
            description="Does a thing.",
            modes=(_ok_columns_mode(),),
        )


def test_operator_rejects_empty_version():
    with pytest.raises(OperatorDescriptorError):
        OperatorDescriptor(
            name="face_metrics",
            version="",
            description="Does a thing.",
            modes=(_ok_columns_mode(),),
        )


def test_operator_rejects_empty_description():
    with pytest.raises(OperatorDescriptorError):
        OperatorDescriptor(
            name="face_metrics",
            version="1.0",
            description="",
            modes=(_ok_columns_mode(),),
        )


def test_operator_rejects_no_modes():
    with pytest.raises(OperatorDescriptorError):
        OperatorDescriptor(
            name="face_metrics",
            version="1.0",
            description="Does a thing.",
            modes=(),
        )


def test_operator_rejects_duplicate_execution_mode():
    with pytest.raises(OperatorDescriptorError):
        OperatorDescriptor(
            name="face_metrics",
            version="1.0",
            description="Does a thing.",
            modes=(_ok_columns_mode(), _ok_columns_mode()),
        )


# ===========================================================================
# Collection fields must be tuples, not lists
# ===========================================================================

def test_choice_parameter_choices_must_be_a_tuple():
    with pytest.raises(OperatorDescriptorError):
        ChoiceParameter(name="c", label="C", choices=[("a", "A")])


def test_column_parameter_required_tags_must_be_a_tuple():
    with pytest.raises(OperatorDescriptorError):
        ColumnParameter(
            name="col", label="Col", from_input="rows", required_tags=["numeric"]
        )


def test_output_spec_columns_must_be_a_tuple():
    with pytest.raises(OperatorDescriptorError):
        OutputSpec(columns=[OutputColumn(name="s", type_tag="numeric")])


def test_mode_inputs_must_be_a_tuple():
    with pytest.raises(OperatorDescriptorError):
        ModeDescriptor(
            mode=ExecutionMode.COLUMNS,
            label="L",
            inputs=[_ok_input()],
            output=_ok_columns_output(),
        )


def test_mode_parameters_must_be_a_tuple():
    with pytest.raises(OperatorDescriptorError):
        ModeDescriptor(
            mode=ExecutionMode.COLUMNS,
            label="L",
            inputs=(_ok_input(),),
            parameters=[TextParameter(name="p", label="P")],
            output=_ok_columns_output(),
        )


def test_operator_modes_must_be_a_tuple():
    with pytest.raises(OperatorDescriptorError):
        OperatorDescriptor(
            name="op",
            version="1.0",
            description="d",
            modes=[_ok_columns_mode()],
        )


# ===========================================================================
# Immutability
# ===========================================================================

def test_descriptor_is_immutable_and_uses_tuples():
    descriptor = OperatorDescriptor(
        name="face_metrics",
        version="1.0",
        description="Computes a face metric per row.",
        modes=(_ok_columns_mode(),),
    )

    # Every collection field is a tuple, not a list.
    assert isinstance(descriptor.modes, tuple)
    assert isinstance(descriptor.modes[0].inputs, tuple)
    assert isinstance(descriptor.modes[0].parameters, tuple)
    assert isinstance(descriptor.modes[0].output.columns, tuple)

    # Assigning to a field of a frozen dataclass raises.
    with pytest.raises(Exception):
        descriptor.name = "renamed"
    with pytest.raises(Exception):
        descriptor.modes[0].output.creates_table = True


# ===========================================================================
# Source scan -- stdlib only, nothing from this repo
# ===========================================================================

def test_descriptor_module_imports_only_stdlib_and_nothing_from_repo():
    # Assemble the Qt binding name from parts so this test file's own text
    # does not contain the literal marker run_tests.py scans for when it
    # decides which test modules touch a widget.
    qt_binding = "PySide" + "6"

    module_path = (
        Path(__file__).resolve().parent.parent / "operators" / "descriptor.py"
    )
    source_text = module_path.read_text(encoding="utf-8")

    # 1. Raw substring check for the forbidden third-party libraries.
    forbidden_libs = (qt_binding, "pandas", "numpy", "PIL", "cv2", "mediapipe")
    for lib in forbidden_libs:
        assert f"import {lib}" not in source_text, (
            f"descriptor.py must not import {lib}"
        )
        assert f"from {lib}" not in source_text, (
            f"descriptor.py must not import from {lib}"
        )

    # 2. Parse the text and inspect every import node. Parsing the text is
    #    not the same as trusting sys.modules -- it reflects exactly what
    #    the file says.
    allowed_top_level = {
        "__future__",
        "dataclasses",
        "enum",
        "keyword",
        "typing",
    }
    repo_packages = {
        "operators",
        "models",
        "column_types",
        "artifacts",
        "media",
        "ui",
        "settings",
        "shared_widgets",
        "controller",
    }

    tree = ast.parse(source_text, filename=str(module_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                assert top in allowed_top_level, (
                    f"descriptor.py imports unexpected module {alias.name!r}"
                )
        elif isinstance(node, ast.ImportFrom):
            # A relative import (level > 0) is by definition from this repo.
            assert node.level == 0, (
                "descriptor.py must not use a relative import"
            )
            top = (node.module or "").split(".")[0]
            assert top not in repo_packages, (
                f"descriptor.py imports from repo package {node.module!r}"
            )
            assert top in allowed_top_level, (
                f"descriptor.py imports from unexpected module {node.module!r}"
            )


# ===========================================================================
# Happy path
# ===========================================================================

def test_realistic_three_mode_descriptor_round_trips():
    # A COLUMNS mode: read the active table, let the researcher pick which
    # column holds the face image, add one numeric column per row.
    columns_mode = ModeDescriptor(
        mode=ExecutionMode.COLUMNS,
        label="Smile intensity per row",
        inputs=(
            InputSpec(
                name="rows",
                label="Rows to process",
                kind=InputKind.ACTIVE_TABLE,
            ),
        ),
        parameters=(
            ColumnParameter(
                name="face_column",
                label="Face image column",
                from_input="rows",
                required_tags=("media_path",),
            ),
            NumberParameter(
                name="window_ms",
                label="Averaging window (ms)",
                minimum=0,
                maximum=5000,
                decimals=0,
                default=500,
            ),
        ),
        output=OutputSpec(
            columns=(OutputColumn(name="smile_intensity", type_tag="numeric"),)
        ),
        model_lifecycle=ModelLifecycle.PER_WORKER,
        # A non-default media requirement: this mode needs one decoded frame
        # per row.
        media_requirement=MediaRequirement.FRAME,
    )

    # A TABLE mode: aggregate the active table into a new named table.
    table_mode = ModeDescriptor(
        mode=ExecutionMode.TABLE,
        label="Mean smile per condition",
        inputs=(
            InputSpec(
                name="source",
                label="Source table",
                kind=InputKind.ACTIVE_TABLE,
            ),
        ),
        parameters=(
            ChoiceParameter(
                name="stat",
                label="Statistic",
                choices=(("mean", "Mean"), ("median", "Median")),
                default="mean",
            ),
            NewTableNameParameter(
                name="new_table_name",
                label="New table name",
                default="smile_by_condition",
            ),
        ),
        output=OutputSpec(creates_table=True),
        deterministic=True,
        cacheable=True,
    )

    # A DISPLAY mode: summarise the selected rows into the Results panel.
    display_mode = ModeDescriptor(
        mode=ExecutionMode.DISPLAY,
        label="Smile summary (quick view)",
        inputs=(
            InputSpec(
                name="selected",
                label="Selected rows",
                kind=InputKind.ACTIVE_TABLE,
            ),
        ),
        output=OutputSpec(is_display_only=True),
        cacheable=False,
    )

    descriptor = OperatorDescriptor(
        name="smile_metrics",
        version="2.1",
        description="Measures smile intensity from the face in each row.",
        modes=(columns_mode, table_mode, display_mode),
    )

    # Read it back.
    assert descriptor.name == "smile_metrics"
    assert descriptor.version == "2.1"
    assert len(descriptor.modes) == 3

    assert descriptor.mode_for(ExecutionMode.COLUMNS) is columns_mode
    assert descriptor.mode_for(ExecutionMode.TABLE) is table_mode
    assert descriptor.mode_for(ExecutionMode.DISPLAY) is display_mode

    table = descriptor.mode_for(ExecutionMode.TABLE)
    assert table.label == "Mean smile per condition"
    assert table.output.creates_table is True

    columns = descriptor.mode_for(ExecutionMode.COLUMNS)
    assert columns.output.columns[0].name == "smile_intensity"
    assert columns.parameters[0].from_input == "rows"
    assert columns.parameters[0].kind == "column"
    assert columns.media_requirement is MediaRequirement.FRAME

    # The TABLE and DISPLAY modes did not declare one, so they default.
    assert table_mode.media_requirement is MediaRequirement.METADATA
    assert display_mode.media_requirement is MediaRequirement.METADATA

    # mode_for() returns None for a mode the operator does not offer.
    display_only = OperatorDescriptor(
        name="viewer",
        version="1.0",
        description="Display only.",
        modes=(display_mode,),
    )
    assert display_only.mode_for(ExecutionMode.COLUMNS) is None
    assert display_only.mode_for(ExecutionMode.TABLE) is None


# ===========================================================================
# A base ParameterSpec still enforces the shared rules
# ===========================================================================

def test_parameter_spec_base_enforces_identifier_rule():
    # ParameterSpec itself is the base class; the shared checks live on it.
    with pytest.raises(OperatorDescriptorError):
        ParameterSpec(name="has space", label="Label")
