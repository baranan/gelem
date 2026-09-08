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
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModelLifecycle,
    ModeDescriptor,
    OperatorDescriptor,
    OutputSpec,
)


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

    # ------------------------------------------------------------------
    # Descriptor (P1.12d-1). What create_display() ACTUALLY does today:
    #  - one DISPLAY mode, over the active table's selected rows,
    #    storing nothing;
    #  - NO parameters. get_parameters_dialog() returns None, so the
    #    test type / dependent variable / grouping column that __init__
    #    sets up (self._test_type etc.) are never chosen by the
    #    researcher -- they keep their constructor defaults. See the
    #    P1.12d-1 report, "should have a parameter but does not".
    #  - media_requirement METADATA: works purely from the DataFrame;
    #  - PLACEHOLDER output: returns a hard-coded fake result
    #    (p_value 0.042, effect_size 0.61, statistic 2.34) regardless of
    #    the data. The description says so.
    #  - deterministic: the same constant dict every call.
    # ------------------------------------------------------------------
    descriptor = OperatorDescriptor(
        name="stats",
        version="1.0",
        description=(
            "PLACEHOLDER operator: meant to run a pingouin statistical test "
            "(t-test, ANOVA, correlation, chi-square) on the selected rows "
            "and show an APA-style result table. Today it ignores the data "
            "and returns a fixed fake result (p = 0.042, effect size 0.61)."
        ),
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.DISPLAY,
                label="Statistical test (pingouin)",
                inputs=(
                    InputSpec(
                        name="active_table",
                        label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                media_requirement=MediaRequirement.METADATA,
                parameters=(),
                output=OutputSpec(is_display_only=True),
                model_lifecycle=ModelLifecycle.NONE,
                deterministic=True,
                cacheable=True,
            ),
        ),
    )

    def __init__(self):
        """Creates the operator with default parameter values."""
        self._test_type:    str        = "t-test (independent)"
        self._dv_column:    str | None = None  # Dependent variable
        self._group_column: str | None = None  # Grouping variable
        self._subject_col:  str | None = None  # Subject ID (for repeated measures)

    def get_parameters_dialog(self, parent=None, columns=None):
        """
        Would show a dialog asking the researcher to choose:
            - Test type
            - Dependent variable column
            - Grouping column (for t-test, ANOVA)
            - Subject ID column (for paired/repeated measures tests)

        TODO (Student C): Implement this dialog.
        Show dropdowns for test type and column selection populated
        from the `columns` argument supplied by MainWindow.
        Grey out irrelevant fields based on the chosen test type
        (e.g. subject column is only needed for paired tests). Hand the
        choices back through parameter_values() keyed by parameters this
        operator's descriptor would then declare.

        For now returns None (no dialog): the values keep their
        construction-time defaults (self._test_type etc.).
        """
        return None

    def create_display(
        self,
        df: pd.DataFrame,
        run,
    ) -> dict:
        """
        Runs the chosen statistical test on df and returns results.

        Args:
            df:  The selected rows as a DataFrame. Read-only.
            run: The OperatorRun for this run. This operator declares no
                 parameters yet, so run.parameters is empty and the test
                 configuration still comes from self._test_type /
                 self._dv_column / self._group_column (construction-time
                 defaults). The argument is here for the uniform operator
                 contract.

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