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


class PlotOperator(BaseOperator):
    """
    Generates a matplotlib bar chart of selected column values
    for each row.

    The researcher selects which columns to plot via the parameter
    dialog. The result is a small PNG image stored per row.
    """

    name = "plot"
    create_columns_label = "Plot columns (bar chart)"
    output_columns = [("plot_path", "plot_image")]
    requires_image = False  # Reads column values from metadata.

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

    def get_parameters_dialog(self, parent=None):
        """
        Shows a dialog asking the researcher which columns to plot.
        Stores the chosen columns in self._columns so create_columns()
        can read them.

        TODO (Student C): Implement this dialog. It should show a
        list of all numeric columns in the current table and let the
        researcher select which ones to include in the plot.

        For now returns None (no dialog, uses default columns).
        """
        return None

    def create_columns(
        self,
        row_id: str,
        image: np.ndarray | None,
        metadata: dict,
    ) -> dict:
        """
        Generates a bar chart for one row showing the values of
        the selected columns.

        Args:
            row_id:   The row being processed.
            image:    Not used (requires_image = False).
            metadata: Contains column values for this row.

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