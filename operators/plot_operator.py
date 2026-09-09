"""
operators/plot_operator.py

PlotOperator generates a matplotlib bar chart of selected column values
for each row. The researcher chooses which columns to plot via the
parameter dialog before running.

The output is a PNG image file saved to disk, with the path stored
as a 'plot_path' column value.

For more advanced interactive plots (line charts over time, grouped
by condition with error bars), see PlotAdvancedOperator which wraps
Plotly Express.

Student C is responsible for implementing this operator.

Dependencies:
    pip install matplotlib
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from operators.base import BaseOperator
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModelLifecycle,
    ModeDescriptor,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
)


class PlotOperator(BaseOperator):
    """
    Generates a matplotlib bar chart of selected column values
    for each row.

    The researcher selects which columns to plot via the parameter
    dialog. The result is a small PNG image stored per row.
    """

    name = "plot"

    # ------------------------------------------------------------------
    # Descriptor (P1.12d-1). What create_columns() ACTUALLY does today:
    #  - one COLUMNS mode, over the active table, producing one
    #    media_path column "plot_path". 'plot_path' holds the path to a
    #    PNG file this operator writes to disk (see create_columns), so
    #    it is a media path -- the same tag a folder of images gets, and
    #    the tag the column's TableSchema spec is built from;
    #  - NO parameters. get_parameters_dialog() returns None, so the
    #    "which columns to plot" choice is NOT a parameter today: the
    #    column list is fixed in __init__ (self._columns default). See
    #    the P1.12d-1 report, "should have a parameter but does not".
    #  - media_requirement METADATA: create_columns() reads only metadata
    #    values, decoding nothing, so the runner hands it media=None;
    #  - despite the "# PLACEHOLDER" comment, create_columns() does real
    #    work -- it builds a real matplotlib bar chart from the row's
    #    real metadata values and writes a real PNG -- so the
    #    description does not call it a placeholder;
    #  - deterministic: fixed output filename, chart built from the
    #    row's values.
    # ------------------------------------------------------------------
    descriptor = OperatorDescriptor(
        name="plot",
        version="1.0",
        description=(
            "For each row, draws a horizontal matplotlib bar chart of a "
            "fixed set of numeric columns' values and writes it as a PNG, "
            "storing the file path in the 'plot_path' media column. Missing "
            "values are treated as 0.0."
        ),
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.COLUMNS,
                label="Plot columns (bar chart)",
                inputs=(
                    InputSpec(
                        name="active_table",
                        label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                media_requirement=MediaRequirement.METADATA,
                parameters=(),
                output=OutputSpec(
                    columns=(
                        OutputColumn(name="plot_path", type_tag="media_path"),
                    ),
                ),
                model_lifecycle=ModelLifecycle.NONE,
                deterministic=True,
                cacheable=True,
            ),
        ),
    )

    def __init__(
        self,
        columns: list[str] | None = None,
        output_dir: Path | None = None,
    ):
        """
        Creates the operator.

        Args:
            columns:    List of column names to plot. Set by the
                        parameter dialog before running. Defaults to
                        common blendshape columns if not set.
            output_dir: Where to save plot images.
        """
        import tempfile
        self._columns = columns or [
            "bs_jawOpen",
            "bs_mouthSmileLeft",
            "bs_mouthSmileRight",
            "bs_browInnerUp",
            "bs_eyeBlinkLeft",
            "bs_eyeBlinkRight",
            "bs_cheekPuff",
        ]
        self._output_dir = output_dir or (
            Path(tempfile.gettempdir()) / "gelem_plots"
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def get_parameters_dialog(self, parent=None, columns=None):
        """
        Would show a dialog asking the researcher which columns to plot.

        TODO (Student C): Implement this dialog. It should show a
        list of all numeric columns in the current table and let the
        researcher select which ones to include in the plot, and hand
        them back through parameter_values() keyed by a "columns"
        parameter this operator's descriptor would then declare.
        Use the `columns` argument supplied by MainWindow.

        For now returns None (no dialog): the column list is a fixed
        construction-time default (self._columns), not yet a parameter.
        """
        return None

    def create_columns(
        self,
        row_id: str,
        media: np.ndarray | None,
        metadata: dict,
        run,
    ) -> dict:
        """
        Generates a bar chart for one row showing the values of
        the selected columns.

        Args:
            row_id:   The row being processed.
            media:    The payload the runner decoded for this row, decided
                      by the mode's media_requirement. This operator
                      declares METADATA, so it is None.
            metadata: Contains column values for this row.
            run:      The OperatorRun for this run. This operator declares
                      no parameters yet, so run.parameters is empty and
                      the column list still comes from self._columns (a
                      construction-time default); the argument is here for
                      the uniform operator contract.

        Returns:
            Dict with 'plot_path' pointing to the saved PNG.

        TODO (Student C): Implement this method.

        Suggested approach:
            1. Read values from metadata for self._columns.
               Handle missing values gracefully (use 0.0 if None).
            2. Create a matplotlib figure:
               import matplotlib; matplotlib.use('Agg')
               import matplotlib.pyplot as plt
               fig, ax = plt.subplots(figsize=(4, 3))
            3. Plot a horizontal bar chart:
               ax.barh(self._columns, values)
               ax.set_xlim(0, 1)
               ax.set_title(f'Row {row_id}')
            4. Save and close:
               output_path = self._output_dir / f'{row_id}_plot.png'
               fig.savefig(str(output_path), dpi=72, bbox_inches='tight')
               plt.close(fig)
            5. Return {'plot_path': str(output_path)}
        """
        # PLACEHOLDER: creates a simple bar chart with placeholder values.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(4, 3))
            values = [
                metadata.get(col, 0.0) or 0.0
                for col in self._columns
            ]
            ax.barh(self._columns, values)
            ax.set_xlim(0, 1)
            ax.set_title(f"Row {row_id}")
            ax.set_xlabel("Value")

            output_path = self._output_dir / f"{row_id}_plot.png"
            fig.savefig(str(output_path), dpi=72, bbox_inches="tight")
            plt.close(fig)
            return {"plot_path": str(output_path)}

        except Exception as e:
            print(f"[PlotOperator] Error for {row_id}: {e}")
            return {"plot_path": None}