"""
operators/plot_advanced.py

PlotAdvancedOperator generates interactive Plotly Express figures
from the selected rows. Unlike PlotOperator which produces a static
bar chart per row, this operator produces one interactive figure
for a whole selection.

Plotly Express handles on-the-fly aggregation — the researcher does
not need to pre-aggregate the data before plotting. For example:
    - Mean blendshape value per condition with error bars
    - Line chart of blendshape over time, coloured by session
    - Box plot of blendshape distribution per condition

For complex multi-level aggregations (e.g. mean over time-from-trial-
start averaged across both trials and sessions simultaneously), the
researcher should use Dataset.aggregate() first to create an aggregated
table, then run this operator on that table.

The output is an interactive HTML file (for browser viewing) and a
static PNG (for the Results panel). Both paths are returned in the
result dict.

Student C is responsible for implementing this operator.

Dependencies:
    pip install plotly kaleido
    kaleido is required for saving static PNG images from Plotly.
"""

from __future__ import annotations
from pathlib import Path
import tempfile
import pandas as pd

from operators.base import BaseOperator


# Plot types the researcher can choose from in the parameter dialog.
PLOT_TYPES = [
    "bar",          # Bar chart — mean per group
    "line",         # Line chart — values over x column
    "box",          # Box plot — distribution per group
    "violin",       # Violin plot — distribution per group
    "scatter",      # Scatter plot — x vs y
    "histogram",    # Histogram — distribution of one column
]


class PlotAdvancedOperator(BaseOperator):
    """
    Generates an interactive Plotly Express figure from selected rows.

    The researcher chooses x column, y column, colour-by column,
    and plot type via the parameter dialog.

    For bar charts, Plotly automatically computes the mean of y
    per group of x, with standard error bars — no pre-aggregation
    needed.
    """

    name = "plot_advanced"
    create_display_label = "Plot (interactive, Plotly)"
    output_columns       = []
    requires_image       = False

    def __init__(self):
        """Creates the operator with default parameter values."""
        self._x_column:     str | None = None
        self._y_column:     str | None = None
        self._color_column: str | None = None
        self._plot_type:    str        = "bar"
        self._output_dir = Path(tempfile.gettempdir()) / "gelem_plots_advanced"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def get_parameters_dialog(self, parent=None):
        """
        Shows a dialog asking the researcher to choose:
            - Plot type (bar, line, box, violin, scatter, histogram)
            - X column
            - Y column
            - Colour-by column (optional)

        Stores the chosen values as instance attributes so
        create_display() can read them.

        TODO (Student C): Implement this dialog.
        The dialog should read available column names from the
        controller and show them in dropdowns. Use the pattern
        shown in the BaseOperator docstring as a starting point.

        For now returns None (no dialog — uses default values).
        """
        return None

    def create_display(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """
        Generates a Plotly Express figure from the selected rows.

        Args:
            df: The selected rows as a DataFrame. May be the full
                frames table or an aggregated table — Plotly handles
                both.

        Returns:
            Dict with keys:
                'operator_name': 'plot_advanced'
                'artifact_path': str path to a static PNG for display
                                 in the Results panel.
                'plot_html':     str path to an interactive HTML file
                                 for opening in a browser.
                'n_rows':        int number of rows used.

        TODO (Student C): Implement this method.

        Suggested approach:
            1. Read self._x_column, self._y_column, self._color_column,
               self._plot_type (set by parameter dialog).
            2. If any required column is not set or not in df, return
               an error dict with a message.
            3. Create the Plotly figure using plotly.express:
               import plotly.express as px
               if self._plot_type == 'bar':
                   fig = px.bar(df,
                       x=self._x_column,
                       y=self._y_column,
                       color=self._color_column,
                       barmode='group',
                       error_y=True,   # adds standard error bars
                   )
               elif self._plot_type == 'line':
                   fig = px.line(df,
                       x=self._x_column,
                       y=self._y_column,
                       color=self._color_column,
                   )
               # ... etc for other plot types
            4. Save as HTML:
               html_path = self._output_dir / 'plot.html'
               fig.write_html(str(html_path))
            5. Save as PNG (requires kaleido):
               png_path = self._output_dir / 'plot.png'
               fig.write_image(str(png_path))
            6. Return the result dict.
        """
        # PLACEHOLDER: creates a simple gray placeholder image.
        try:
            from PIL import Image
            import numpy as np

            png_path  = self._output_dir / "plot_advanced_placeholder.png"
            html_path = self._output_dir / "plot_advanced_placeholder.html"

            # Create a placeholder PNG.
            arr = np.ones((300, 400, 3), dtype=np.uint8) * 200
            Image.fromarray(arr).save(str(png_path))

            # Create a placeholder HTML file.
            html_path.write_text(
                "<html><body>"
                "<h2>PlotAdvancedOperator placeholder</h2>"
                "<p>Implement create_display() to generate a real plot.</p>"
                "</body></html>"
            )

            print(
                f"[PlotAdvancedOperator] PLACEHOLDER — "
                f"{len(df)} rows, plot_type={self._plot_type}"
            )
            return {
                "operator_name": "plot_advanced",
                "artifact_path": str(png_path),
                "plot_html":     str(html_path),
                "n_rows":        len(df),
            }

        except Exception as e:
            print(f"[PlotAdvancedOperator] Error: {e}")
            return {
                "operator_name": "plot_advanced",
                "artifact_path": None,
                "plot_html":     None,
                "n_rows":        len(df),
            }