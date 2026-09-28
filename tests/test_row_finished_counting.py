"""
tests/test_row_finished_counting.py

operators/operator_registry.py's on_row_finished callback (the run-time-
estimate item): every row that reaches a terminal outcome -- success, or
a refusal before create_columns() was ever called (MissingMedia,
UnparseableMedia, WholeVideoRow, an unreadable image) -- must be counted,
not just the rows the existing on_item_complete/emitted channel already
covers.

Exercises the real OperatorRegistry.run_create_columns() over a small
snapshot with a mix of a real image path and a blank media cell.

Written from the work-item specification, not the implementation.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from operators.base import BaseOperator
from operators.operator_registry import OperatorRegistry
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModeDescriptor,
    ModelLifecycle,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
)
from operators.run_context import (
    CancellationToken,
    OperatorRun,
    OperatorRunSpec,
    RunData,
)
from media.resolver import MediaResolver

TEST_IMAGES = PROJECT_ROOT / "test_images"
_A_REAL_IMAGE = str(next(TEST_IMAGES.glob("*.jpg")))


class _AlwaysSucceeds(BaseOperator):
    """A FRAME-requirement operator that never raises -- the only row
    failures a run of this operator can produce are the pre-decode
    refusals (MissingMedia here), never an operator exception."""

    name = "always_succeeds"
    descriptor = OperatorDescriptor(
        name="always_succeeds",
        version="1.0",
        description="Test double: always returns a fixed score.",
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.COLUMNS,
                label="Always succeeds",
                inputs=(
                    InputSpec(
                        name="active_table", label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                media_requirement=MediaRequirement.FRAME,
                model_lifecycle=ModelLifecycle.NONE,
                output=OutputSpec(
                    columns=(OutputColumn(name="score", type_tag="numeric"),),
                ),
            ),
        ),
    )

    def create_columns(self, row_id, media, metadata, run):
        return {"score": 1}


def _build_run(resolver: MediaResolver) -> OperatorRun:
    mode_descriptor = _AlwaysSucceeds.descriptor.mode_for(ExecutionMode.COLUMNS)
    spec = OperatorRunSpec(
        operation_id="op-1",
        operator_name="always_succeeds",
        mode=ExecutionMode.COLUMNS,
        mode_descriptor=mode_descriptor,
        parameters={},
        target_table="frames",
    )
    return OperatorRun(
        spec=spec,
        data=RunData(tables={}, projects={}),
        paths=object(),
        resolver=resolver,
        _token=CancellationToken(),
    )


def test_a_row_refused_before_create_columns_still_counts_as_finished(
    monkeypatch,
):
    """Reversal check: if on_row_finished stopped being called for a row
    refused before create_columns() was ever called, this test fails --
    the finished count would then undercount by the number of such rows,
    exactly the gap the work item's spec calls out ("rows that ended in
    a row error count as finished").
    """
    op_registry = OperatorRegistry()
    op_registry.register(_AlwaysSucceeds())
    resolver = MediaResolver(max_open_decoders=4)
    run = _build_run(resolver)

    row_ids = ["r0", "r1", "r2", "r3"]
    # r1 and r3 have no media to read at all -- MissingMedia, a refusal
    # that never reaches create_columns() or on_item_complete.
    snapshot = pd.DataFrame({
        "media": [_A_REAL_IMAGE, "", _A_REAL_IMAGE, ""],
    })

    finished_calls: list[str] = []
    results: dict[str, dict] = {}
    row_errors: list[tuple] = []
    completions: list[int] = []
    created: list[threading.Thread] = []
    real_thread = threading.Thread

    class _Tracked(real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    monkeypatch.setattr(threading, "Thread", _Tracked)
    started = op_registry.run_create_columns(
        "always_succeeds", snapshot, row_ids, "frames", run,
        operation_id="op-1",
        on_item_complete=lambda _oid, _t, rid, result: results.__setitem__(
            rid, result
        ),
        on_progress=None,
        on_complete=lambda _oid, _name, emitted, **_kw: completions.append(
            emitted
        ),
        on_setup_error=None,
        on_row_errors=lambda _oid, _label, errors: row_errors.extend(errors),
        media_column="media",
        on_row_finished=lambda oid: finished_calls.append(oid),
    )

    assert started, "run_create_columns did not start a worker"
    for t in created:
        t.join(timeout=30)
        assert not t.is_alive(), "worker thread did not finish in time"

    # Two rows succeeded through the ordinary channel...
    assert len(results) == 2
    assert completions == [2]
    # ...but ALL FOUR rows reached a terminal outcome, including the two
    # MissingMedia refusals that never touch on_item_complete/emitted.
    assert len(finished_calls) == 4
    assert finished_calls.count("op-1") == 4
    assert len(row_errors) == 2
    assert {kind for _rid, kind, _msg in row_errors} == {"MissingMedia"}
