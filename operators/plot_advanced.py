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

from __future__ import annotations
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px

from operators.base import BaseOperator
from operators.descriptor import (
    ChoiceParameter,
    ColumnParameter,
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModelLifecycle,
    ModeDescriptor,
    OperatorDescriptor,
    OutputSpec,
    TextParameter,
)


CHART_TYPES = ["scatter", "line", "bar", "box", "violin", "histogram"]
AGGREGATES  = ["none", "count", "sum", "mean", "median"]

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
    # The descriptor's DISPLAY ModeDescriptor (below) is what makes this
    # operator appear in the Operators menu's "Show result" section.

    # ------------------------------------------------------------------
    # Descriptor (P1.12d-1). What create_display() ACTUALLY does today:
    #  - one DISPLAY mode, over the active table's selected rows,
    #    storing nothing;
    #  - SEVEN parameters -- title, chart_type, x, y, color, facet,
    #    aggregate -- all genuinely read by create_display(). They arrive
    #    in run.parameters: MainWindow builds the form from these
    #    declarations (ui/parameter_dialog.py), the run spec validates the
    #    collected values against this descriptor, and nothing is read off
    #    the operator instance.
    #  - media_requirement METADATA: works purely from the DataFrame;
    #  - deterministic FALSE / cacheable FALSE: the returned artifact_path
    #    and html_path embed a wall-clock timestamp --
    #        now = datetime.now()
    #        run_id = (now.strftime("%Y.%m.%d_%H.%M.%S") + ...)
    #    so two runs with identical inputs return different paths.
    # ------------------------------------------------------------------
    descriptor = OperatorDescriptor(
        name="plot_advanced",
        version="1.0",
        description=(
            "Builds one interactive Plotly Express figure (scatter, line, "
            "bar, box, violin or histogram) for the selected rows, "
            "optionally colouring and facetting by a column and aggregating "
            "groups first. Writes a static PNG and an interactive HTML file "
            "and shows them in the Results panel; stores nothing."
        ),
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.DISPLAY,
                label="Plot (interactive, Plotly)",
                inputs=(
                    InputSpec(
                        name="active_table",
                        label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                media_requirement=MediaRequirement.METADATA,
                parameters=(
                    TextParameter(
                        name="title",
                        label="Title",
                        help_text="Blank uses the auto title \"{y} by {x}\".",
                        required=False,
                        default="",
                    ),
                    ChoiceParameter(
                        name="chart_type",
                        label="Chart type",
                        choices=tuple(
                            (chart_type, chart_type)
                            for chart_type in CHART_TYPES
                        ),
                        default="scatter",
                    ),
                    ColumnParameter(
                        name="x",
                        label="X axis",
                        from_input="active_table",
                    ),
                    ColumnParameter(
                        name="y",
                        label="Y axis",
                        from_input="active_table",
                    ),
                    ColumnParameter(
                        name="color",
                        label="Colour (optional)",
                        from_input="active_table",
                        required=False,
                    ),
                    ColumnParameter(
                        name="facet",
                        label="Facet (optional)",
                        from_input="active_table",
                        required=False,
                    ),
                    ChoiceParameter(
                        name="aggregate",
                        label="Aggregate",
                        choices=tuple(
                            (aggregate, aggregate) for aggregate in AGGREGATES
                        ),
                        default="none",
                    ),
                ),
                output=OutputSpec(is_display_only=True),
                model_lifecycle=ModelLifecycle.NONE,
                deterministic=False,
                cacheable=False,
            ),
        ),
    )

    def __init__(self, output_dir: Path | None = None):
        # Default to a project-relative folder, not the system Temp
        # directory (same pattern as VideoFramesOperator). main.py can
        # pass an explicit output_dir once Dataset.save()/load() define
        # a real project folder.
        #
        # output_dir is a genuine construction-time value and stays here.
        # The seven run parameters (title, chart_type, x, y, color, facet,
        # aggregate) travel in run.parameters and nothing about them lives
        # on the instance -- two concurrent runs must not share one set of
        # values.
        self._output_dir = output_dir or (
            Path.cwd() / "gelem_project" / "plots"
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # No get_parameters_dialog(). The seven parameters above are declared
    # on the descriptor; MainWindow builds the form from them
    # (ui/parameter_dialog.py) and passes the collected values to the
    # controller in run.parameters. This module contains no Qt.
    #
    # The generated form is plainer than the hand-drawn one it replaced:
    # it has no live coupling between chart_type and the aggregate control
    # and no inline warnings. create_display() already tolerates every
    # combination -- box/violin ignore aggregate, and histogram falls back
    # to "count" for an unsupported histfunc -- so nothing breaks; the
    # researcher just loses the guidance the old dialog showed while open.

    def create_display(self, df, run):
        """Build one interactive Plotly figure for the selected rows.

        Parameters
        ----------
        df : pd.DataFrame
            The selected rows, passed in by AppController.
            Do not modify it -- work on a copy.
        run : OperatorRun
            The run object. Every parameter is read from run.parameters
            (validated against this operator's descriptor before the run
            started); nothing is read off self.

        Returns
        -------
        dict
            {"artifact_path": str, "html_path": str}
        """
        data = df.copy()  # never modify the DataFrame received from AppController

        params    = run.parameters
        x         = params["x"]
        y         = params["y"]
        color     = params.get("color") or None   # None is fine -- px ignores it
        facet     = params.get("facet") or None   # None is fine -- px ignores it
        chart     = params["chart_type"]
        aggregate = params.get("aggregate", "none")

        # ------------------------------------------------------------------
        # Step 1: decide whether to summarise the data before plotting
        # ------------------------------------------------------------------
        #
        # box / violin  -> always use every raw row (that is the point of them)
        # histogram     -> Plotly summarises internally via histfunc (see Step 2)
        # scatter / line / bar with aggregate == "none"
        #               -> use every raw row as-is
        # scatter / line / bar with a real aggregate
        #               -> do a pandas groupby here; Plotly cannot aggregate
        #                  these chart types on its own
        #
        # Scope note: groupby uses the exact values of x, which is correct
        # for categorical columns (condition, participant_id, etc.).
        # Numeric binning is out of scope for this version.
        if (
            chart in ("scatter", "line", "bar")
            and aggregate in _AGG_TO_PANDAS
            and x is not None
            and y is not None
        ):
            group_keys = [k for k in (x, color, facet) if k]
            plot_df = (
                data.groupby(group_keys, as_index=False)[y]
                .agg(_AGG_TO_PANDAS[aggregate])
            )
        else:
            plot_df = data

        # ------------------------------------------------------------------
        # Step 2: build the figure with the matching Plotly Express function
        # ------------------------------------------------------------------
        common = dict(x=x, y=y, color=color, facet_col=facet)

        if chart == "scatter":
            fig = px.scatter(plot_df, **common)
        elif chart == "line":
            fig = px.line(plot_df, **common)
        elif chart == "bar":
            fig = px.bar(plot_df, **common, barmode="group")
        elif chart == "box":
            fig = px.box(plot_df, **common)
        elif chart == "violin":
            fig = px.violin(plot_df, **common, box=True, points="all")
        elif chart == "histogram":
            fig = px.histogram(
                plot_df,
                x=x,
                y=y,
                color=color,
                facet_col=facet,
                histfunc=_AGG_TO_HISTFUNC.get(aggregate, "count"),
            )
        else:
            raise ValueError(f"Unknown chart type: {chart!r}")

        # Title: researcher's text if provided, otherwise "{y} by {x}".
        # x=0.5 + xanchor="center" centres the title above the plot area.
        title = params.get("title") or f"{y} by {x}"
        fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"))

        # ------------------------------------------------------------------
        # Step 3: save and return
        # ------------------------------------------------------------------
        # Unique-per-run filenames so back-to-back plots do not overwrite.
        # The 2-digit centiseconds suffix keeps names unique within one second.
        now = datetime.now()
        run_id = (
            now.strftime("%Y.%m.%d_%H.%M.%S")
            + f".{now.microsecond // 10000:02d}"
        )
        html_path = self._output_dir / f"plot_{run_id}.html"
        png_path  = self._output_dir / f"plot_{run_id}.png"

        fig.write_html(str(html_path))
        fig.write_image(str(png_path))  # requires: pip install kaleido

        return {
            "operator_name": self.name,
            "artifact_path": str(png_path),
            "html_path":     str(html_path),
        }
