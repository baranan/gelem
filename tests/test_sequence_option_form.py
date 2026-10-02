"""
tests/test_sequence_option_form.py

A mode's ``sequence_option`` names a boolean (``enabled_by``) that turns
sequence handling on, and optionally a column (``group_by``) that only has
meaning while it is on. The generated parameter form must therefore keep
the ``group_by`` field disabled while its boolean is off, and enabled while
it is on -- live, as the checkbox toggles -- without the operator's own
``refine_form`` having to say so.

Layer A (``build_field_specs`` / ``resolve_form``, Qt-free) decides it;
Layer B (``ParameterDialog``) only renders it.

Written from the work-item specification, not the implementation.

Run with:
    python -m pytest tests/test_sequence_option_form.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from operators.descriptor import (
    BooleanParameter,
    ColumnParameter,
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModeDescriptor,
    ModelLifecycle,
    OutputColumn,
    OutputSpec,
    SequenceOption,
)
from operators.form_advice import FormAdvice
from ui.parameter_dialog import (
    ParameterDialog,
    build_field_specs,
    collect_parameters,
    resolve_form,
)

_COLUMNS_BY_INPUT = {
    "active_table": (("trial", "text"), ("clip", "media_path")),
}


def _mode(*, with_group_by: bool = True, required_group: bool = False):
    return ModeDescriptor(
        mode=ExecutionMode.COLUMNS,
        label="Seq mode",
        inputs=(
            InputSpec(
                name="active_table", label="Active table",
                kind=InputKind.ACTIVE_TABLE,
            ),
        ),
        media_requirement=MediaRequirement.FRAME,
        parameters=(
            BooleanParameter(
                name="use_seq", label="Use sequence",
                required=False, default=False,
            ),
            ColumnParameter(
                name="group_col", label="Group by",
                from_input="active_table", required=required_group,
            ),
            BooleanParameter(
                name="unrelated", label="Unrelated",
                required=False, default=False,
            ),
        ),
        output=OutputSpec(columns=(OutputColumn(name="out", type_tag="numeric"),)),
        model_lifecycle=ModelLifecycle.PER_WORKER,
        sequence_option=SequenceOption(
            enabled_by="use_seq",
            group_by="group_col" if with_group_by else None,
        ),
    )


# ---- Layer A -------------------------------------------------------------

def test_group_by_field_spec_records_which_boolean_enables_it():
    specs = {s.name: s for s in build_field_specs(_mode(), _COLUMNS_BY_INPUT)}
    assert specs["group_col"].enabled_by == "use_seq"
    # No other field is tied to the boolean.
    assert specs["use_seq"].enabled_by is None
    assert specs["unrelated"].enabled_by is None


def test_group_by_is_disabled_while_boolean_is_off_and_enabled_when_on():
    specs = build_field_specs(_mode(), _COLUMNS_BY_INPUT)
    off = resolve_form(specs, FormAdvice(), {"use_seq": False, "group_col": None})
    on = resolve_form(specs, FormAdvice(), {"use_seq": True, "group_col": None})
    assert "group_col" in off.disabled_fields
    assert "group_col" not in on.disabled_fields
    # A missing raw value counts as off.
    missing = resolve_form(specs, FormAdvice(), {})
    assert "group_col" in missing.disabled_fields


def test_mode_without_group_by_disables_nothing():
    specs = build_field_specs(_mode(with_group_by=False), _COLUMNS_BY_INPUT)
    resolved = resolve_form(specs, FormAdvice(), {"use_seq": False})
    assert resolved.disabled_fields == ()


def test_disabled_group_by_never_blocks_ok_and_keeps_its_value():
    specs = build_field_specs(_mode(required_group=True), _COLUMNS_BY_INPUT)
    advice = FormAdvice(allowed_choices={"group_col": ("clip",)})
    raw = {"use_seq": False, "group_col": "trial"}
    resolved = resolve_form(specs, advice, raw)
    assert "group_col" in resolved.disabled_fields
    assert resolved.blocked is False
    # The value is still collected.
    assert collect_parameters(specs, raw)["group_col"] == "trial"


# ---- Layer B -------------------------------------------------------------

def test_dialog_group_by_widget_follows_checkbox_live(qapp):
    from PySide6.QtWidgets import QCheckBox, QComboBox

    dialog = ParameterDialog(_mode(), _COLUMNS_BY_INPUT)
    checkbox = dialog._widgets["use_seq"]
    group = dialog._widgets["group_col"]
    assert isinstance(checkbox, QCheckBox)
    assert isinstance(group, QComboBox)
    assert group.isEnabled() is False
    checkbox.setChecked(True)
    assert group.isEnabled() is True
    checkbox.setChecked(False)
    assert group.isEnabled() is False
