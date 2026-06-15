"""
PlotAdvancedOperator
--------------------
A create_display operator that builds one interactive Plotly Express figure
for the selected rows.

The researcher chooses chart type, x/y columns, optional colour and facet
columns, and whether to summarise groups before plotting (aggregate).

Output: a result dict with two keys --
    "artifact_path" : path to a static PNG shown in the Results panel
    "html_path"     : path to an interactive HTML file opened by the
                      "Open interactive version" button in the Results panel
"""

import plotly.express as px

from operators.base import BaseOperator


# These lookup tables translate the menu words the researcher sees into the
# names that pandas and Plotly expect internally.  They are module-level
# constants so the student does not have to look up the exact strings.

# Used in create_display when we do a pandas groupby before plotting.
_AGG_TO_PANDAS = {
    "count":  "count",
    "sum":    "sum",
    "mean":   "mean",
    "median": "median",
}

# Used when chart == "histogram".  Plotly's histfunc does not support median.
_AGG_TO_HISTFUNC = {
    "count": "count",
    "sum":   "sum",
    "mean":  "avg",
}


class PlotAdvancedOperator(BaseOperator):

    name = "plot_advanced"
    # Setting create_display_label makes this operator appear in the
    # Operators menu under "Display results for selection".
    create_display_label = "Plot (interactive, Plotly)"

    def get_parameters_dialog(self):
        """Show a dialog and store the researcher's choices as instance attributes.

        Collect:
            self._chart_type  -- one of: scatter | line | bar | box | violin | histogram
            self._x           -- column name for the horizontal axis
            self._y           -- column name for the vertical axis
            self._color       -- (optional) column to colour marks by group; None if not chosen
            self._facet       -- (optional) column to split into a grid of small plots; None if not chosen
            self._aggregate   -- one of: none | count | sum | mean | median

        Notes for the dialog:
        - Disable the aggregate control when chart_type is "box" or "violin"
          (those chart types always use every row).
        - Offer count / sum / mean for histogram (no median -- Plotly does not
          support median histfunc).
        """
        # TODO (Student C): implement the parameter dialog
        pass

    def create_display(self, df):
        """Build one interactive Plotly figure for the selected rows.

        Parameters
        ----------
        df : pd.DataFrame
            The selected rows, passed in by AppController.
            Do not modify it -- work on a copy.

        Returns
        -------
        dict
            {"artifact_path": str, "html_path": str}
        """
        data = df.copy()  # never modify the DataFrame received from AppController

        x         = self._x
        y         = self._y
        color     = self._color or None   # None is fine -- px ignores it
        facet     = self._facet or None   # None is fine -- px ignores it
        chart     = self._chart_type
        aggregate = self._aggregate

        # ------------------------------------------------------------------
        # Step 1: decide whether to summarise the data before plotting
        # ------------------------------------------------------------------
        #
        # box / violin  → always use every raw row (that is the point of them)
        # histogram     → Plotly summarises internally via histfunc (see Step 2)
        # scatter / line / bar with aggregate == "none"
        #               → use every raw row as-is
        # scatter / line / bar with a real aggregate
        #               → do a pandas groupby here; Plotly cannot aggregate
        #                 these chart types on its own
        #
        # Scope note: groupby uses the exact values of x, which is correct
        # for categorical columns (condition, participant_id, etc.).
        # Numeric binning is out of scope for this version.
        #
        # TODO (Student C): produce plot_df

        # ------------------------------------------------------------------
        # Step 2: build the figure with the matching Plotly Express function
        # ------------------------------------------------------------------
        #
        # Use the common keyword dict to avoid repeating x/y/color/facet_col:
        #
        #   common = dict(x=x, y=y, color=color, facet_col=facet)
        #
        # Then dispatch on chart:
        #   scatter  → px.scatter(plot_df, **common)
        #   line     → px.line(plot_df, **common)
        #   bar      → px.bar(plot_df, **common, barmode="group")
        #   box      → px.box(plot_df, **common)
        #   violin   → px.violin(plot_df, **common, box=True, points="all")
        #   histogram→ px.histogram(plot_df, x=x, y=y, color=color,
        #                           facet_col=facet,
        #                           histfunc=_AGG_TO_HISTFUNC.get(aggregate, "count"))
        #
        # TODO (Student C): produce fig

        # ------------------------------------------------------------------
        # Step 3: save and return
        # ------------------------------------------------------------------
        #
        # Always write into self.output_dir -- do not write next to the source
        # data files.  Use self.run_id to make filenames unique per run.
        #
        #   html_path = self.output_dir / f"plot_{self.run_id}.html"
        #   png_path  = self.output_dir / f"plot_{self.run_id}.png"
        #   fig.write_html(str(html_path))
        #   fig.write_image(str(png_path))   # requires: pip install kaleido
        #
        # Return both paths so the Results panel can show the PNG inline and
        # offer an "Open interactive version" button for the HTML.
        #
        # TODO (Student C): save files and return result dict
        pass
