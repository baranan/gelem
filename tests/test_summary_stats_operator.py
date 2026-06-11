"""
tests/test_summary_stats_operator.py

Standalone test for SummaryStatsOperator.

Run as a script:
    python tests/test_summary_stats_operator.py

Or under pytest:
    pytest tests/test_summary_stats_operator.py

Verifies the v5 spec for SummaryStatsOperator:
  - Returns mean / sd / min / max / median / n for each numeric column.
  - Sample SD (ddof=1), matching JASP/SPSS.
  - row_id and text columns are excluded by default.
  - self._columns restricts the result to the listed columns.
  - The input DataFrame is not mutated.
  - Empty / all-NaN columns are skipped.
"""

from __future__ import annotations
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from operators.summary_stats import SummaryStatsOperator


def _make_df():
    return pd.DataFrame({
        "row_id": [1, 2, 3, 4, 5],
        "reaction_time": [1.0, 2.0, 3.0, 4.0, 5.0],   # mean=3, sd≈1.5811, n=5
        "score": [10.0, 20.0, 30.0, float("nan"), 40.0],  # mean≈25, n=4
        "condition": ["A", "B", "A", "B", "A"],       # text — must be excluded
    })


def test_default_includes_all_numeric_except_row_id():
    op = SummaryStatsOperator()
    result = op.create_display(_make_df())

    assert result["operator_name"] == "summary_stats"
    assert result["n_rows"] == 5
    assert set(result["summary"].keys()) == {"reaction_time", "score"}, \
        f"unexpected columns: {set(result['summary'].keys())}"

    rt = result["summary"]["reaction_time"]
    assert rt["n"] == 5
    assert rt["mean"] == 3.0
    assert math.isclose(rt["sd"], 1.5811388300841898, rel_tol=1e-9), \
        f"sd should be sample SD (ddof=1); got {rt['sd']}"
    assert rt["min"] == 1.0
    assert rt["max"] == 5.0
    assert rt["median"] == 3.0

    sc = result["summary"]["score"]
    assert sc["n"] == 4, "NaN should be dropped before counting"
    assert sc["mean"] == 25.0


def test_columns_override():
    op = SummaryStatsOperator(columns=["score"])
    result = op.create_display(_make_df())

    assert set(result["summary"].keys()) == {"score"}, \
        "self._columns should restrict the result"


def test_input_dataframe_not_mutated():
    df = _make_df()
    snapshot = df.copy()
    op = SummaryStatsOperator()
    op.create_display(df)
    pd.testing.assert_frame_equal(df, snapshot)


def test_all_nan_column_is_skipped():
    df = pd.DataFrame({
        "row_id": [1, 2, 3],
        "valid":  [1.0, 2.0, 3.0],
        "empty":  [float("nan"), float("nan"), float("nan")],
    })
    result = SummaryStatsOperator().create_display(df)
    assert set(result["summary"].keys()) == {"valid"}


if __name__ == "__main__":
    test_default_includes_all_numeric_except_row_id()
    test_columns_override()
    test_input_dataframe_not_mutated()
    test_all_nan_column_is_skipped()

    # Print a sample for human inspection.
    op = SummaryStatsOperator()
    sample = op.create_display(_make_df())
    print("\nSample result:")
    for col, stats in sample["summary"].items():
        print(f"  {col}: {stats}")
    print("\nAll SummaryStatsOperator tests passed.")
