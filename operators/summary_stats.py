"""
operators/summary_stats.py

SummaryStatsOperator computes descriptive statistics over a selection
of rows and displays the result in the Results panel.

This is a display operator — it implements create_display() and returns
a summary dict. The result is shown in ResultsPanel, not stored as
table columns.

Student C is responsible for implementing this operator.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from operators.base import BaseOperator


class SummaryStatsOperator(BaseOperator):
    """
    Computes mean, SD, min, max, and median for all numeric columns
    across the selected rows.

    Result is shown in ResultsPanel, not stored as table columns.
    """

    name = "summary_stats"
    create_display_label = "Summary statistics"
    output_columns       = []
    requires_image       = False

    def __init__(self, columns: list[str] | None = None):
        """
        Creates the operator.

        Args:
            columns: List of column names to compute statistics for.
                     If None, computes for all numeric columns.
        """
        self._columns = columns

    def create_display(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """
        Computes summary statistics for the rows in df.

        Args:
            df: The selected rows as a DataFrame. Treated as read-only.

        Returns:
            Dict with keys:
                'operator_name': 'summary_stats'
                'n_rows':        int number of rows analysed.
                'summary':       nested dict of
                                 {column: {stat: value}}
                                 where stats are mean, sd, min,
                                 max, median, n.
                'warnings':      list of str — human-readable notes about
                                 anything that could not be summarised
                                 (no numeric columns, empty selection,
                                 all-NaN columns, mixed numeric/text
                                 columns, missing/non-numeric requested
                                 columns). Empty when all is well.
                'interpretation': str — the warnings formatted for display
                                 (a single sentence for one warning, a
                                 bulleted list for several). Omitted when
                                 there are no warnings.

        Notes:
            - Sample SD is computed with ddof=1 (matches JASP/SPSS).
            - Works on a defensive copy of df; the input is never mutated.
            - A blank result (no numeric data) is never silent: the
              researcher always gets a warning explaining why.
        """
        df = df.copy()
        warnings: list[str] = []
        try:
            # Don't trust dtype alone: blendshapes and other operator
            # columns can arrive as object dtype (update_row creates them as
            # None, so later floats never upgrade to float64).
            if self._columns:
                present = [c for c in self._columns if c in df.columns]
                missing = [c for c in self._columns if c not in df.columns]
                if missing:
                    warnings.append(
                        "Requested column(s) not found and skipped: "
                        + ", ".join(missing) + "."
                    )
                candidates = present
            else:
                candidates = [c for c in df.columns if c != "row_id"]

            numeric_values: dict[str, pd.Series] = {}
            non_numeric: list[str] = []
            mixed: list[str] = []
            for col in candidates:
                kind, series = self._classify_numeric_columns(df[col])
                if kind == "numeric":
                    numeric_values[col] = series
                elif kind == "mixed":
                    mixed.append(col)
                else:
                    non_numeric.append(col)
            numeric_cols = list(numeric_values.keys())

            # Flag dirty columns rather than drop them silently.
            if mixed:
                warnings.append(
                    "Column(s) skipped — mixed numeric and non-numeric "
                    "values: " + ", ".join(mixed) + "."
                )
            if self._columns and non_numeric:
                warnings.append(
                    "Requested column(s) are not numeric and skipped: "
                    + ", ".join(non_numeric) + "."
                )

            # Compute per column; track ones with no usable values.
            summary = {}
            empty_cols: list[str] = []
            for col in numeric_cols:
                values = numeric_values[col].dropna()
                if len(values) == 0:
                    empty_cols.append(col)
                    continue
                summary[col] = {
                    "mean":   float(values.mean()),
                    "sd":     float(values.std(ddof=1)),
                    "min":    float(values.min()),
                    "max":    float(values.max()),
                    "median": float(values.median()),
                    "n":      int(len(values)),
                }

            # Explain an empty or partial result.
            if len(df) == 0:
                warnings.append(
                    "No rows are selected, so there is nothing to summarise."
                )
            elif not numeric_cols:
                if self._columns:
                    warnings.append(
                        "None of the requested columns are numeric, so no "
                        "statistics could be computed."
                    )
                else:
                    warnings.append(
                        "The selection has no numeric columns to compute "
                        "statistics from."
                    )
            elif not summary:
                warnings.append(
                    "All candidate columns are empty (only missing values), "
                    "so no statistics could be computed."
                )
            elif empty_cols:
                warnings.append(
                    "Column(s) skipped (only missing values): "
                    + ", ".join(empty_cols) + "."
                )

            return self._result(len(df), summary, warnings)

        except Exception as e:
            print(f"[SummaryStatsOperator] Error: {e}")
            warnings.append(f"Could not compute statistics: {e}")
            return self._result(len(df), {}, warnings)

    @staticmethod
    def _classify_numeric_columns(
        series: pd.Series,
    ) -> tuple[str, pd.Series | None]:
        """
        Classify a column as ("numeric", values), ("mixed", None), or
        ("other", None).

        numeric: int/float, or object/string fully coercible to numbers
                 (recovers blendshapes stored as object dtype).
        mixed:   object/string mixing numbers and non-numbers; skipped,
                 caller warns.
        other:   text, datetime, bool, categorical, or all-missing object.

        Bools and datetimes count as non-numeric on purpose: bool matches
        select_dtypes("number"), and to_numeric would turn datetimes into
        nanosecond integers.
        """
        if pd.api.types.is_bool_dtype(series):
            return ("other", None)
        if pd.api.types.is_numeric_dtype(series):
            return ("numeric", series)

        # Only object/string columns are coerced; datetime, timedelta,
        # categorical, etc. are left out.
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            return ("other", None)

        non_null = series.dropna()
        if len(non_null) == 0:
            # All missing — can't tell the type.
            return ("other", None)

        # Count bools as non-numeric so True/False don't sneak in as 1/0.
        is_bool = non_null.map(lambda v: isinstance(v, (bool, np.bool_)))
        numeric_mask = (
            pd.to_numeric(non_null, errors="coerce").notna() & ~is_bool
        )

        if numeric_mask.all():
            # All-true only if no bool slipped in, so coercion is safe.
            return ("numeric", pd.to_numeric(series, errors="coerce"))
        if numeric_mask.any():
            return ("mixed", None)
        return ("other", None)

    @staticmethod
    def _result(n_rows: int, summary: dict, warnings: list[str]) -> dict:
        """
        Build the result dict. Warnings go out as a list and as an
        'interpretation' string (one sentence, or bullets if several) that
        the Results panel already renders.
        """
        result = {
            "operator_name": "summary_stats",
            "n_rows":        n_rows,
            "summary":       summary,
            "warnings":      warnings,
        }
        if len(warnings) == 1:
            result["interpretation"] = warnings[0]
        elif warnings:
            result["interpretation"] = "\n".join(f"• {w}" for w in warnings)
        return result