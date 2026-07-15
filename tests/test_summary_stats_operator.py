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


# ── Warning cases ───────────────────────────────────────────────────────
# A blank result must never be silent — the operator explains why.

def test_no_numeric_columns_warns():
    # Only row_id and a text column: nothing to summarise.
    df = pd.DataFrame({
        "row_id":    [1, 2, 3],
        "condition": ["A", "B", "A"],
    })
    result = SummaryStatsOperator().create_display(df)
    assert result["summary"] == {}
    assert len(result["warnings"]) == 1
    assert "no numeric columns" in result["warnings"][0].lower()
    # A single warning surfaces as plain interpretation text (no bullet).
    assert result["interpretation"] == result["warnings"][0]


def test_empty_selection_warns():
    df = pd.DataFrame({"row_id": [], "score": []})
    result = SummaryStatsOperator().create_display(df)
    assert result["summary"] == {}
    assert any("no rows" in w.lower() for w in result["warnings"])


def test_all_columns_nan_warns():
    df = pd.DataFrame({
        "row_id": [1, 2, 3],
        "empty":  [float("nan"), float("nan"), float("nan")],
    })
    result = SummaryStatsOperator().create_display(df)
    assert result["summary"] == {}
    assert any("missing values" in w.lower() for w in result["warnings"])


def test_partial_nan_column_warns_but_still_computes():
    df = pd.DataFrame({
        "row_id": [1, 2, 3],
        "valid":  [1.0, 2.0, 3.0],
        "empty":  [float("nan"), float("nan"), float("nan")],
    })
    result = SummaryStatsOperator().create_display(df)
    assert set(result["summary"].keys()) == {"valid"}
    assert any("empty" in w for w in result["warnings"])


def test_missing_requested_column_warns():
    df = pd.DataFrame({"row_id": [1, 2], "score": [10.0, 20.0]})
    result = SummaryStatsOperator(columns=["score", "ghost"]).create_display(df)
    assert set(result["summary"].keys()) == {"score"}
    assert any("ghost" in w for w in result["warnings"])


def test_non_numeric_requested_column_warns_no_crash():
    # Text column requested by name must be reported, not crash the run.
    df = pd.DataFrame({"row_id": [1, 2], "label": ["x", "y"]})
    result = SummaryStatsOperator(columns=["label"]).create_display(df)
    assert result["summary"] == {}
    assert any("not numeric" in w.lower() for w in result["warnings"])


def test_multiple_warnings_render_as_bullets():
    df = pd.DataFrame({"row_id": [1, 2], "label": ["x", "y"]})
    result = SummaryStatsOperator(
        columns=["label", "ghost"]
    ).create_display(df)
    assert len(result["warnings"]) >= 2
    # Several warnings render one-per-line as a bulleted list.
    assert result["interpretation"].startswith("• ")
    assert result["interpretation"].count("• ") == len(result["warnings"])


def test_no_warnings_when_all_good():
    result = SummaryStatsOperator().create_display(_make_df())
    assert result["warnings"] == []
    assert "interpretation" not in result


def test_object_dtype_numeric_column_is_recognised():
    # Blendshapes applied row-by-row land as object dtype (created as None,
    # then floats written in). They must still be summarised, not skipped.
    df = pd.DataFrame({"row_id": ["a", "b", "c"]})
    df["bs_jawOpen"] = None
    for rid, val in [("a", 0.12), ("b", 0.44), ("c", None)]:  # c = no face
        df.loc[df["row_id"] == rid, "bs_jawOpen"] = val
    assert df["bs_jawOpen"].dtype == object, "test premise: column is object dtype"

    result = SummaryStatsOperator().create_display(df)
    assert "bs_jawOpen" in result["summary"], \
        "object-dtype blendshape column should be recognised as numeric"
    stats = result["summary"]["bs_jawOpen"]
    assert stats["n"] == 2                    # None dropped
    assert math.isclose(stats["mean"], 0.28)  # (0.12 + 0.44) / 2
    assert result["warnings"] == []


def test_text_object_column_still_excluded():
    # A genuine text column must NOT be coerced into the stats.
    df = pd.DataFrame({
        "row_id":    ["a", "b"],
        "condition": ["A", "B"],
    })
    result = SummaryStatsOperator().create_display(df)
    assert result["summary"] == {}
    assert any("no numeric columns" in w.lower() for w in result["warnings"])


def test_datetime_column_is_excluded():
    # Regression: pd.to_numeric turns datetimes into nanosecond integers.
    # A datetime column must NOT be summarised (matches the original
    # select_dtypes("number") behaviour); the real numeric column still is.
    df = pd.DataFrame({
        "row_id": ["a", "b"],
        "when":   pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "score":  [1.0, 2.0],
    })
    result = SummaryStatsOperator().create_display(df)
    assert set(result["summary"].keys()) == {"score"}, \
        "datetime column must not be treated as numeric"


def test_single_none_dropped_per_column():
    # One missing value in a column must drop only that row for that column;
    # other columns keep their own n. No warning — the column has data.
    df = pd.DataFrame({
        "row_id": ["a", "b", "c"],
        "score":  [10.0, None, 30.0],   # one missing -> n=2
        "other":  [1.0, 2.0, 3.0],      # complete    -> n=3
    })
    result = SummaryStatsOperator().create_display(df)
    assert result["summary"]["score"]["n"] == 2
    assert result["summary"]["score"]["mean"] == 20.0
    assert result["summary"]["other"]["n"] == 3
    assert result["warnings"] == []


def test_bool_column_is_excluded():
    # Booleans are not summarised (matches select_dtypes("number")).
    df = pd.DataFrame({
        "row_id": ["a", "b"],
        "flag":   [True, False],
        "score":  [1.0, 2.0],
    })
    result = SummaryStatsOperator().create_display(df)
    assert set(result["summary"].keys()) == {"score"}


def test_categorical_column_is_excluded():
    df = pd.DataFrame({
        "row_id": ["a", "b"],
        "cat":    pd.Categorical(["x", "y"]),
        "score":  [1.0, 2.0],
    })
    result = SummaryStatsOperator().create_display(df)
    assert set(result["summary"].keys()) == {"score"}


def test_all_bool_object_column_is_excluded():
    # Bools inside an object column must be excluded, exactly like a whole
    # bool column — not counted as 1/0. All-bool -> skipped, no warning.
    df = pd.DataFrame({"row_id": ["a", "b", "c"]})
    df["flag"] = pd.Series([True, False, True], dtype=object)
    result = SummaryStatsOperator().create_display(df)
    assert "flag" not in result["summary"]
    assert any("no numeric columns" in w.lower() for w in result["warnings"])


def test_bool_mixed_with_numbers_in_object_is_skipped_and_warned():
    # True/False alongside real numbers -> treated as mixed, skipped + warned
    # (bools are non-numeric here, so they can't average in as 1/0).
    df = pd.DataFrame({"row_id": ["a", "b", "c"]})
    df["c"]     = pd.Series([True, 1, 2], dtype=object)
    df["score"] = [1.0, 2.0, 3.0]
    result = SummaryStatsOperator().create_display(df)
    assert set(result["summary"].keys()) == {"score"}
    assert any(
        "mixed" in w.lower() and "c" in w for w in result["warnings"]
    ), f"expected a mixed-column warning, got {result['warnings']}"


def test_mixed_text_and_number_column_warns_and_skips():
    # A numeric column dirtied by stray text is skipped, but flagged — not
    # silently dropped, and not partially summarised.
    df = pd.DataFrame({
        "row_id": ["a", "b", "c"],
        "dirty":  [0.1, "oops", 0.3],   # object dtype, mixed
        "score":  [1.0, 2.0, 3.0],
    })
    result = SummaryStatsOperator().create_display(df)
    assert set(result["summary"].keys()) == {"score"}, \
        "a mixed column must not be summarised"
    assert any(
        "mixed" in w.lower() and "dirty" in w
        for w in result["warnings"]
    ), f"expected a mixed-column warning, got {result['warnings']}"


def test_int_and_float_mix_is_numeric():
    # Ints and floats together are fine — pandas/coercion yields floats.
    # Covers both a native mixed-numeric column and an object one.
    df = pd.DataFrame({"row_id": ["a", "b", "c"]})
    df["native"] = [1, 2.5, 3]          # -> float64
    df["obj"]    = pd.Series([1, 2.5, 3], dtype=object)
    result = SummaryStatsOperator().create_display(df)
    assert set(result["summary"].keys()) == {"native", "obj"}
    for col in ("native", "obj"):
        s = result["summary"][col]
        assert s["n"] == 3
        assert math.isclose(s["mean"], (1 + 2.5 + 3) / 3)
    assert result["warnings"] == []


if __name__ == "__main__":
    test_default_includes_all_numeric_except_row_id()
    test_columns_override()
    test_input_dataframe_not_mutated()
    test_all_nan_column_is_skipped()
    test_no_numeric_columns_warns()
    test_empty_selection_warns()
    test_all_columns_nan_warns()
    test_partial_nan_column_warns_but_still_computes()
    test_missing_requested_column_warns()
    test_non_numeric_requested_column_warns_no_crash()
    test_multiple_warnings_render_as_bullets()
    test_no_warnings_when_all_good()
    test_object_dtype_numeric_column_is_recognised()
    test_text_object_column_still_excluded()
    test_datetime_column_is_excluded()
    test_single_none_dropped_per_column()
    test_bool_column_is_excluded()
    test_categorical_column_is_excluded()
    test_all_bool_object_column_is_excluded()
    test_bool_mixed_with_numbers_in_object_is_skipped_and_warned()
    test_mixed_text_and_number_column_warns_and_skips()
    test_int_and_float_mix_is_numeric()

    # Print a sample for human inspection.
    op = SummaryStatsOperator()
    sample = op.create_display(_make_df())
    print("\nSample result:")
    for col, stats in sample["summary"].items():
        print(f"  {col}: {stats}")
    print("\nAll SummaryStatsOperator tests passed.")
