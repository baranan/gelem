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
            df: The selected rows as a DataFrame. Read-only.

        Returns:
            Dict with keys:
                'operator_name': 'summary_stats'
                'n_rows':        int number of rows analysed.
                'summary':       nested dict of
                                 {column: {stat: value}}
                                 where stats are mean, sd, min,
                                 max, median, n.

        TODO (Student C): Implement this method.

        Suggested approach:
            1. Determine which columns to analyse:
               - If self._columns is set, use those.
               - Otherwise use df.select_dtypes(include='number').columns
               - Exclude 'row_id' from numeric columns.
            2. For each column:
               values = df[col].dropna()
               if len(values) == 0: skip
               summary[col] = {
                   'mean':   float(values.mean()),
                   'sd':     float(values.std()),
                   'min':    float(values.min()),
                   'max':    float(values.max()),
                   'median': float(values.median()),
                   'n':      int(len(values)),
               }
            3. Return the result dict.
        """
        # PLACEHOLDER: returns empty summary.
        try:
            if self._columns:
                numeric_cols = [
                    c for c in self._columns
                    if c in df.columns
                ]
            else:
                numeric_cols = list(
                    df.select_dtypes(include="number").columns
                )
                numeric_cols = [
                    c for c in numeric_cols
                    if c != "row_id"
                ]

            summary = {}
            for col in numeric_cols:
                values = df[col].dropna()
                if len(values) == 0:
                    continue
                summary[col] = {
                    "mean":   float(values.mean()),
                    "sd":     float(values.std()),
                    "min":    float(values.min()),
                    "max":    float(values.max()),
                    "median": float(values.median()),
                    "n":      int(len(values)),
                }

            print(
                f"[SummaryStatsOperator] Computed stats for "
                f"{len(summary)} columns across {len(df)} rows"
            )
            return {
                "operator_name": "summary_stats",
                "n_rows":        len(df),
                "summary":       summary,
            }

        except Exception as e:
            print(f"[SummaryStatsOperator] Error: {e}")
            return {
                "operator_name": "summary_stats",
                "n_rows":        len(df),
                "summary":       {},
            }