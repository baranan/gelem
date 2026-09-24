"""
tests/test_controller_operator_listing.py

P1.13: AppController.list_operators_for_mode() is a thin pass-through onto
OperatorRegistry.list_operators_for_mode() -- the seam ui/main_window.py's
Operators-menu builder now goes through instead of reaching into
AppController._op_registry directly (CLAUDE.md, "UI" -- "UI never touches
private controller attributes").

Uses the make_controller fixture (tests/conftest.py), which hands back a
real but empty OperatorRegistry; this test registers one stub operator
declaring all three ExecutionModes so the pass-through can be checked for
each of them against the registry's own answer.

Run with:
    python -m pytest tests/test_controller_operator_listing.py
"""

from __future__ import annotations

from operators.base import BaseOperator
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    ModeDescriptor,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
)


class _StubOperator(BaseOperator):
    name = "p1_13_stub"

    descriptor = OperatorDescriptor(
        name="p1_13_stub",
        version="1.0",
        description="Test double declaring all three execution modes.",
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.COLUMNS,
                label="Stub columns",
                inputs=(
                    InputSpec(
                        name="active_table", label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                parameters=(),
                output=OutputSpec(
                    columns=(OutputColumn(name="stub_score", type_tag="numeric"),)
                ),
            ),
            ModeDescriptor(
                mode=ExecutionMode.TABLE,
                label="Stub table",
                inputs=(
                    InputSpec(
                        name="active_table", label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                parameters=(),
                output=OutputSpec(creates_table=True),
            ),
            ModeDescriptor(
                mode=ExecutionMode.DISPLAY,
                label="Stub display",
                inputs=(
                    InputSpec(
                        name="active_table", label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                parameters=(),
                output=OutputSpec(is_display_only=True),
            ),
        ),
    )


def test_list_operators_for_mode_matches_the_registry_for_every_mode(
    tmp_path, make_controller
):
    controller, _dataset, op_registry = make_controller(tmp_path)
    op_registry.register(_StubOperator())

    for mode in (ExecutionMode.COLUMNS, ExecutionMode.TABLE, ExecutionMode.DISPLAY):
        assert controller.list_operators_for_mode(mode) == (
            op_registry.list_operators_for_mode(mode)
        )
        # Not vacuously equal on two empty lists -- the stub actually shows up.
        assert ("p1_13_stub", f"Stub {mode.name.lower()}") in (
            controller.list_operators_for_mode(mode)
        )
