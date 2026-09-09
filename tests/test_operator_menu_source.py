"""
tests/test_operator_menu_source.py

P1.12d-3: the Operators menu's facts -- which operators appear in which
section, and the label each shows -- now come from ONE place, the
operator's ``descriptor``. The legacy ``*_label`` / ``output_columns`` /
``display_label`` attributes that used to hold the same facts in parallel
are gone.

These tests pin that single source:
  - OperatorRegistry.list_operators_for_mode(mode) lists exactly the
    registered operators whose descriptor declares ``mode``, in
    registration order (which is operators_config.yaml's order, hence the
    menu order), each with that mode's ModeDescriptor label;
  - an operator declaring two modes appears in both mode lists, once each;
  - the label is read off the descriptor, never off a legacy attribute;
  - BaseOperator no longer carries any of the five deleted names.

Combined (non-widget) group: constructs no Qt widget and deliberately
names none of run_tests.py's widget markers.
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from operators.base import BaseOperator
from operators.operator_registry import OperatorRegistry
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    ModeDescriptor,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
)


# ---------------------------------------------------------------------------
# Small helpers for building descriptor doubles. One mode each, minimal
# but well-formed.
# ---------------------------------------------------------------------------

def _active_table_input():
    return (
        InputSpec(
            name="active_table",
            label="Active table",
            kind=InputKind.ACTIVE_TABLE,
        ),
    )


def _columns_mode(label):
    return ModeDescriptor(
        mode=ExecutionMode.COLUMNS,
        label=label,
        inputs=_active_table_input(),
        output=OutputSpec(
            columns=(OutputColumn(name="out", type_tag="numeric"),)
        ),
    )


def _table_mode(label):
    return ModeDescriptor(
        mode=ExecutionMode.TABLE,
        label=label,
        inputs=_active_table_input(),
        output=OutputSpec(creates_table=True),
    )


def _display_mode(label):
    return ModeDescriptor(
        mode=ExecutionMode.DISPLAY,
        label=label,
        inputs=_active_table_input(),
        output=OutputSpec(is_display_only=True),
    )


def _operator(name, *modes):
    """A BaseOperator subclass instance carrying just a name and a
    descriptor with the given modes."""

    class _Double(BaseOperator):
        pass

    _Double.name = name
    _Double.descriptor = OperatorDescriptor(
        name=name,
        version="1.0",
        description=f"Test double: {name}.",
        modes=tuple(modes),
    )
    return _Double()


# ---------------------------------------------------------------------------
# list_operators_for_mode: exactly the declaring operators, in
# registration order, with the descriptor's labels.
# ---------------------------------------------------------------------------

def test_list_for_mode_is_exactly_the_declaring_operators_in_registration_order():
    """Would still pass if list_operators_for_mode sorted, filtered by a
    stale attribute, or dropped the label? No -- the registration order
    here is not alphabetical, only two of the three operators declare
    COLUMNS, and each tuple's second element is asserted."""
    reg = OperatorRegistry()
    reg.register(_operator("gamma", _columns_mode("Gamma columns")))
    reg.register(_operator("alpha", _columns_mode("Alpha columns")))
    reg.register(_operator("beta", _display_mode("Beta view")))

    assert reg.list_operators_for_mode(ExecutionMode.COLUMNS) == [
        ("gamma", "Gamma columns"),
        ("alpha", "Alpha columns"),
    ]
    assert reg.list_operators_for_mode(ExecutionMode.DISPLAY) == [
        ("beta", "Beta view"),
    ]
    assert reg.list_operators_for_mode(ExecutionMode.TABLE) == []


def test_operator_with_two_modes_appears_in_both_lists_once_each():
    """Would still pass if a two-mode operator were listed once, or
    twice in one list? No -- 'twofer' must show up in TABLE and in
    DISPLAY, exactly once in each, with that mode's own label."""
    reg = OperatorRegistry()
    reg.register(_operator("solo", _columns_mode("Solo columns")))
    reg.register(
        _operator(
            "twofer",
            _table_mode("Twofer table"),
            _display_mode("Twofer view"),
        )
    )

    assert reg.list_operators_for_mode(ExecutionMode.TABLE) == [
        ("twofer", "Twofer table"),
    ]
    assert reg.list_operators_for_mode(ExecutionMode.DISPLAY) == [
        ("twofer", "Twofer view"),
    ]
    assert reg.list_operators_for_mode(ExecutionMode.COLUMNS) == [
        ("solo", "Solo columns"),
    ]


def test_label_comes_from_the_descriptor_not_a_legacy_style_attribute():
    """A stray ``create_columns_label`` on the instance must be ignored:
    the descriptor's ModeDescriptor label is the only source."""
    reg = OperatorRegistry()
    op = _operator("labelled", _columns_mode("From the descriptor"))
    op.create_columns_label = "From a legacy attribute"  # must not be read
    reg.register(op)

    assert reg.list_operators_for_mode(ExecutionMode.COLUMNS) == [
        ("labelled", "From the descriptor"),
    ]


def test_operator_without_a_descriptor_contributes_nothing():
    reg = OperatorRegistry()

    class _NoDescriptor(BaseOperator):
        name = "no_descriptor"

    reg.register(_NoDescriptor())
    reg.register(_operator("has_descriptor", _columns_mode("Real columns")))

    assert reg.list_operators_for_mode(ExecutionMode.COLUMNS) == [
        ("has_descriptor", "Real columns"),
    ]


# ---------------------------------------------------------------------------
# The legacy names are gone from BaseOperator.
# ---------------------------------------------------------------------------

def test_base_operator_has_none_of_the_five_deleted_names():
    """Asserted with hasattr, not a source scan: a re-introduced class
    attribute OR property would both be caught."""
    for attr in (
        "create_columns_label",
        "create_table_label",
        "create_display_label",
        "output_columns",
        "display_label",
    ):
        assert not hasattr(BaseOperator, attr), (
            f"BaseOperator still carries {attr!r} -- P1.12d-3 removed it"
        )
