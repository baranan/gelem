"""
tests/test_segment_operator.py

Standalone test for SegmentOperator (P1.6b).

Run as a script:
    python tests/test_segment_operator.py

Or under pytest:
    pytest tests/test_segment_operator.py

Written from the P1.6b work item and docs/architecture.md §4.2, not from
the implementation:
  - the carried columns appear in carry-set order before the new columns,
    and a column absent from the carry set does not appear;
  - a source value that already carries a fragment produces an address
    built from its path portion, not from the whole value;
  - segment_index restarts at 0 for each distinct source path and follows
    segment_start;
  - segment_duration equals end minus start;
  - a row with end <= start and a row with a null boundary are both
    dropped while the other rows survive;
  - the returned frame is not the input frame, and the input frame is
    unchanged;
  - the column named by media_column is excluded from the carried
    columns even when the carry set names it (operators/CLAUDE.md,
    "Carry the lineage columns" -- a carry decision the operator makes
    for itself, corrected into this file after the first pass wrongly
    treated it as out of scope);
  - a row with a null media value is dropped while the other rows
    survive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from operators.segment import SegmentOperator
from operators.descriptor import ExecutionMode
from operators.run_context import (
    CancellationToken,
    OperatorRun,
    OperatorRunSpec,
    RunData,
    TableSnapshot,
)


def _run(op, *, media_column, start_column, end_column, carry_columns):
    """A minimal OperatorRun for a direct create_table() call. Parameters
    are validated against the operator's OWN descriptor (as every other
    operator test in this repo does), and carry_columns is put on the
    'active_table' input's TableSnapshot exactly the way
    AppController._build_operator_run does for a real run (P1.6a) -- the
    operator only ever reads it off there, never off Dataset."""
    mode_descriptor = op.descriptor.mode_for(ExecutionMode.TABLE)
    spec = OperatorRunSpec(
        operation_id="test-run",
        operator_name=op.name,
        mode=ExecutionMode.TABLE,
        mode_descriptor=mode_descriptor,
        parameters={
            "media_column": media_column,
            "start_column": start_column,
            "end_column": end_column,
            "output_table": "segments",
        },
        target_table="",
    )
    snapshot = TableSnapshot(
        table_name="source",
        frame=None,
        version=1,
        carry_columns=tuple(carry_columns),
    )
    return OperatorRun(
        spec=spec,
        data=RunData(tables={"active_table": snapshot}, projects={}),
        paths=None,
        _token=CancellationToken(),
    )


def _make_df():
    # Row 0: p1.mp4, start=5-end=10   -> kept, second in the p1.mp4 group.
    # Row 1: p1.mp4#f=999, start=1-end=3 -> kept, ALREADY carries a
    #        fragment; the address must be built from its path portion
    #        alone ("p1.mp4"), not from the literal cell value.
    # Row 2: p1.mp4, start=None-end=4  -> dropped (null boundary).
    # Row 3: p2.mp4, start=0-end=1     -> kept, only member of its group.
    # Row 4: p1.mp4, start=2-end=1     -> dropped (end <= start).
    return pd.DataFrame(
        {
            "row_id": ["r0", "r1", "r2", "r3", "r4"],
            "session_id": ["s0", "s0", "s0", "s1", "s0"],
            "participant_id": ["p1", "p1", "p1", "p2", "p1"],
            "not_carried": ["x", "x", "x", "x", "x"],
            "media": [
                "videos/p1.mp4",
                "videos/p1.mp4#f=999",
                "videos/p1.mp4",
                "videos/p2.mp4",
                "videos/p1.mp4",
            ],
            "start_s": [5.0, 1.0, None, 0.0, 2.0],
            "end_s": [10.0, 3.0, 4.0, 1.0, 1.0],
        }
    )


def _carry_columns():
    # Deliberately NOT in df's own column order (session_id before
    # participant_id), and includes a name df does not have at all
    # ("missing_col") -- both must be respected literally by the operator.
    return ("session_id", "participant_id", "missing_col")


def test_carried_columns_in_carry_set_order_and_absent_column_excluded():
    op = SegmentOperator()
    df = _make_df()
    run = _run(
        op,
        media_column="media",
        start_column="start_s",
        end_column="end_s",
        carry_columns=_carry_columns(),
    )
    result = op.create_table(df, run)

    columns = list(result.columns)
    assert columns[:2] == ["session_id", "participant_id"], (
        f"expected the two present carry-set columns first, in carry-set "
        f"order; got {columns}"
    )
    assert "missing_col" not in columns, (
        "a carry-set name absent from df must not appear in the output"
    )
    assert "not_carried" not in columns, (
        "a df column absent from the carry set must not appear in the "
        "output"
    )
    assert "row_id" not in columns, "no row_id in a create_table() result"


def test_fragment_already_present_is_stripped_to_its_path_portion():
    op = SegmentOperator()
    df = _make_df()
    run = _run(
        op,
        media_column="media",
        start_column="start_s",
        end_column="end_s",
        carry_columns=_carry_columns(),
    )
    result = op.create_table(df, run)

    # Row 1 (media = "videos/p1.mp4#f=999", start=1, end=3): the address
    # must be built from the PATH PORTION only.
    matching = result[
        (result["segment_start"] == 1.0) & (result["segment_end"] == 3.0)
    ]
    assert len(matching) == 1
    assert matching.iloc[0]["segment_address"] == (
        "videos/p1.mp4#t=1.000000-3.000000"
    ), (
        "expected the address built from the path portion alone, not a "
        "literal copy of the already-fragmented source cell"
    )


def test_segment_index_restarts_per_source_path_ordered_by_start():
    op = SegmentOperator()
    df = _make_df()
    run = _run(
        op,
        media_column="media",
        start_column="start_s",
        end_column="end_s",
        carry_columns=_carry_columns(),
    )
    result = op.create_table(df, run)

    # Surviving p1.mp4 rows: start=1 (row 1) and start=5 (row 0).
    # Ordered by segment_start, row 1 (start=1) must be index 0 and row 0
    # (start=5) must be index 1 -- even though row 0 appears first in df.
    p1_rows = result[result["segment_address"].str.startswith("videos/p1.mp4")]
    p1_rows = p1_rows.sort_values("segment_start")
    assert list(p1_rows["segment_index"]) == [0, 1], (
        f"expected segment_index 0, 1 ordered by segment_start; got "
        f"{list(p1_rows['segment_index'])}"
    )

    # The single surviving p2.mp4 row restarts its own count at 0.
    p2_rows = result[result["segment_address"].str.startswith("videos/p2.mp4")]
    assert list(p2_rows["segment_index"]) == [0], (
        "a distinct source path must restart segment_index at 0"
    )


def test_segment_duration_equals_end_minus_start():
    op = SegmentOperator()
    df = _make_df()
    run = _run(
        op,
        media_column="media",
        start_column="start_s",
        end_column="end_s",
        carry_columns=_carry_columns(),
    )
    result = op.create_table(df, run)

    for _, row in result.iterrows():
        assert row["segment_duration"] == row["segment_end"] - row["segment_start"]


def test_invalid_rows_are_dropped_and_others_survive():
    op = SegmentOperator()
    df = _make_df()
    run = _run(
        op,
        media_column="media",
        start_column="start_s",
        end_column="end_s",
        carry_columns=_carry_columns(),
    )
    result = op.create_table(df, run)

    # Five input rows; row 2 (null start) and row 4 (end <= start) are
    # dropped, leaving three.
    assert len(result) == 3
    assert (result["segment_end"] > result["segment_start"]).all()
    starts = sorted(result["segment_start"].tolist())
    assert starts == [0.0, 1.0, 5.0]


def test_media_column_is_excluded_from_output_even_when_in_carry_set():
    op = SegmentOperator()
    df = _make_df()
    # Deliberately include "media" (this run's media_column) in the carry
    # set, alongside a normal identifier. A segment row already carries
    # the source path inside segment_address, so a literal second copy in
    # a carried "media" column would just repeat it for no use
    # (operators/CLAUDE.md, "Carry the lineage columns", the full_path
    # example) -- the operator must drop it regardless of what the carry
    # set says.
    carry_columns = ("session_id", "media", "participant_id")
    run = _run(
        op,
        media_column="media",
        start_column="start_s",
        end_column="end_s",
        carry_columns=carry_columns,
    )
    result = op.create_table(df, run)

    assert "media" not in result.columns, (
        "media_column must be excluded from the carried columns even "
        "when the carry set names it"
    )
    assert list(result.columns[:2]) == ["session_id", "participant_id"], (
        "the remaining carry-set columns must still appear, in carry-set "
        f"order, with media_column simply removed; got "
        f"{list(result.columns)}"
    )


def test_row_with_null_media_value_is_dropped_while_others_survive():
    op = SegmentOperator()
    df = pd.DataFrame(
        {
            "row_id": ["r0", "r1", "r2"],
            "participant_id": ["p1", "p1", "p2"],
            # Row 1 has no media value at all. str(None) reads "None" and
            # the missing-value default str(nan) reads "nan" -- both parse
            # as a plausible bare path unless explicitly caught, which is
            # exactly the defect this test guards against.
            "media": ["videos/p1.mp4", None, "videos/p2.mp4"],
            "start_s": [1.0, 2.0, 0.0],
            "end_s": [2.0, 3.0, 1.0],
        }
    )
    run = _run(
        op,
        media_column="media",
        start_column="start_s",
        end_column="end_s",
        carry_columns=("participant_id",),
    )
    result = op.create_table(df, run)

    assert len(result) == 2, (
        f"expected the null-media row dropped and the other two rows "
        f"(both with a valid media value and time range) kept; got "
        f"{len(result)} row(s)"
    )
    assert sorted(result["segment_start"].tolist()) == [0.0, 1.0]
    assert not result["segment_address"].str.contains("nan", case=False).any(), (
        "a null media value must never reach a stored address as the "
        "literal text 'nan'"
    )


def test_returned_frame_is_new_and_input_is_unchanged():
    op = SegmentOperator()
    df = _make_df()
    snapshot_before = df.copy(deep=True)
    run = _run(
        op,
        media_column="media",
        start_column="start_s",
        end_column="end_s",
        carry_columns=_carry_columns(),
    )
    result = op.create_table(df, run)

    assert result is not df
    pd.testing.assert_frame_equal(df, snapshot_before)


if __name__ == "__main__":
    test_carried_columns_in_carry_set_order_and_absent_column_excluded()
    test_fragment_already_present_is_stripped_to_its_path_portion()
    test_segment_index_restarts_per_source_path_ordered_by_start()
    test_segment_duration_equals_end_minus_start()
    test_invalid_rows_are_dropped_and_others_survive()
    test_media_column_is_excluded_from_output_even_when_in_carry_set()
    test_row_with_null_media_value_is_dropped_while_others_survive()
    test_returned_frame_is_new_and_input_is_unchanged()

    op = SegmentOperator()
    df = _make_df()
    run = _run(
        op,
        media_column="media",
        start_column="start_s",
        end_column="end_s",
        carry_columns=_carry_columns(),
    )
    sample = op.create_table(df, run)
    print("\nSample result:")
    print(sample)
    print("\nAll SegmentOperator tests passed.")
