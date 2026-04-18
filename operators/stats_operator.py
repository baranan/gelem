"""
operators/stats_operator.py

StatsOperator wraps the pingouin library to provide proper statistical
tests with effect sizes and confidence intervals. Pingouin is designed
for psychology research and produces APA-style output.

Supported tests (chosen via parameter dialog):
    - Independent samples t-test (two groups)
    - Paired samples t-test (within-subjects)
    - One-way ANOVA
    - Repeated measures ANOVA
    - Pearson/Spearman correlation
    - Chi-square test of independence

The result is displayed in the Results panel as a formatted table.
Nothing is stored in the dataset.

If pingouin is not installed, falls back to scipy.stats for basic
t-tests and correlations.

Student C is responsible for implementing this operator.

Dependencies:
    pip install pingouin
    pip install scipy  (fallback if pingouin not available)
"""

from __future__ import annotations
import pandas as pd

from operators.base import BaseOperator


# Statistical tests the researcher can choose from.
TEST_TYPES = [
    "t-test (independent)",
    "t-test (paired)",
    "one-way ANOVA",
    "repeated measures ANOVA",
    "correlation (Pearson)",
    "correlation (Spearman)",
    "chi-square",
]


class StatsOperator(BaseOperator):
    """
    Runs a statistical test on the selected rows using pingouin.

    The researcher chooses the test type, dependent variable,
    and grouping column via the parameter dialog.

    Results include test statistic, p-value, effect size, and
    confidence intervals where applicable.
    """

    name = "stats"
    create_display_label = "Statistical test (pingouin)"
    output_columns       = []
    requires_image       = False

    def __init__(self):
        """Creates the operator with default parameter values."""
        self._test_type:    str        = "t-test (independent)"
        self._dv_column:    str | None = None  # Dependent variable
        self._group_column: str | None = None  # Grouping variable
        self._subject_col:  str | None = None  # Subject ID (for repeated measures)

    def get_parameters_dialog(self, parent=None):
        """
        Shows a dialog asking the researcher to choose:
            - Test type
            - Dependent variable column
            - Grouping column (for t-test, ANOVA)
            - Subject ID column (for paired/repeated measures tests)

        Stores the chosen values as instance attributes.

        TODO (Student C): Implement this dialog.
        Show dropdowns for test type and column selection.
        Grey out irrelevant fields based on the chosen test type
        (e.g. subject column is only needed for paired tests).

        For now returns None (no dialog — uses default values).
        """
        return None

    def create_display(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """
        Runs the chosen statistical test on df and returns results.

        Args:
            df: The selected rows as a DataFrame. Read-only.

        Returns:
            Dict with keys:
                'operator_name': 'stats'
                'test_type':     str name of the test run.
                'summary':       dict of result values (varies by test).
                                 Always includes 'p_value' and
                                 'effect_size' where applicable.
                'table':         list of dicts for tabular display.
                                 Each dict is one row of the results table.
                'interpretation': str plain-language interpretation.

        TODO (Student C): Implement this method.

        Suggested approach using pingouin:
            import pingouin as pg

            if self._test_type == 't-test (independent)':
                # Split df into two groups by self._group_column.
                groups = df[self._group_column].unique()
                if len(groups) != 2:
                    return error dict
                g1 = df[df[self._group_column] == groups[0]][self._dv_column]
                g2 = df[df[self._group_column] == groups[1]][self._dv_column]
                result = pg.ttest(g1, g2)
                # result is a DataFrame with columns:
                # T, dof, alternative, p-val, CI95%, cohen-d, BF10, power

            elif self._test_type == 'one-way ANOVA':
                result = pg.anova(
                    data=df,
                    dv=self._dv_column,
                    between=self._group_column,
                )

            elif self._test_type == 'correlation (Pearson)':
                result = pg.corr(
                    df[self._dv_column],
                    df[self._group_column],
                    method='pearson',
                )

            # Convert result DataFrame to list of dicts for display.
            table = result.round(4).to_dict('records')
        """
        # PLACEHOLDER: returns a fake result.
        print(
            f"[StatsOperator] PLACEHOLDER — "
            f"test={self._test_type}, "
            f"dv={self._dv_column}, "
            f"group={self._group_column}, "
            f"n={len(df)} rows"
        )
        return {
            "operator_name":  "stats",
            "test_type":      self._test_type,
            "summary": {
                "p_value":     0.042,
                "effect_size": 0.61,
                "statistic":   2.34,
            },
            "table": [
                {
                    "Test":        self._test_type,
                    "Statistic":   2.34,
                    "p-value":     0.042,
                    "Effect size": 0.61,
                    "Note":        "PLACEHOLDER — implement create_display()",
                }
            ],
            "interpretation": (
                "PLACEHOLDER result. Implement StatsOperator.create_display() "
                "using pingouin to get real statistical output."
            ),
        }