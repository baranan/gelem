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
    #    aggregate -- all genuinely read by create_display(). As of
    #    P1.12d-2a they arrive in run.parameters (the dialog returns them
    #    from parameter_values(), the run spec validates them against this
    #    descriptor), never off the operator instance.
    #    The dialog is a real QDialog (not None), so these ARE the
    #    operator's parameters.
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
        # aggregate) used to be stored on self by get_parameters_dialog;
        # as of P1.12d-2a they travel in run.parameters and nothing about
        # them lives on the instance -- two concurrent runs must not share
        # one set of values.
        self._output_dir = output_dir or (
            Path.cwd() / "gelem_project" / "plots"
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def get_parameters_dialog(self, parent=None, columns=None):
        """Show a dialog and hand the researcher's choices back through
        parameter_values(). It stores NOTHING on the operator instance.

        parameter_values() returns a dict keyed by the descriptor's
        parameter names:
            chart_type  -- one of: scatter | line | bar | box | violin | histogram
            x           -- column name for the horizontal axis
            y           -- column name for the vertical axis
            color       -- (optional) column to colour marks by group; None if not chosen
            facet       -- (optional) column to split into a grid of small plots; None if not chosen
            aggregate   -- one of: none | count | sum | mean | median
            title       -- free text; "" means "use the auto title {y} by {x}"

        Notes for the dialog:
        - Populate the x/y/color/facet dropdowns from the `columns` argument
          supplied by MainWindow.
        - Disable the aggregate control when chart_type is "box" or "violin"
          (those chart types always use every row).
        - Offer count / sum / mean for histogram (no median -- Plotly does not
          support median histfunc).
        """
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QGridLayout,
            QLabel,
            QLineEdit,
            QVBoxLayout,
        )

        available = list(columns) if columns else []

        dialog = QDialog(parent)
        dialog.setWindowTitle("Plot (interactive, Plotly)")
        dialog.setMinimumWidth(420)

        layout = QVBoxLayout(dialog)
        # Equal column stretch keeps the label/field split at the middle.
        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        row = 0

        def _add_row(text: str, widget) -> None:
            nonlocal row
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(lbl, row, 0)
            grid.addWidget(widget, row, 1)
            row += 1

        # Plot title -- free text. Leave blank for the auto title
        # "{y} by {x}" computed in create_display().
        title_edit = QLineEdit()
        title_edit.setPlaceholderText("Auto: \"{y} by {x}\"")
        _add_row("Title:", title_edit)

        # Chart-type dropdown. Starts at the descriptor's default ("scatter").
        chart_combo = QComboBox()
        chart_combo.addItems(CHART_TYPES)
        chart_combo.setCurrentText("scatter")
        _add_row("Chart type:", chart_combo)

        # Column dropdowns for x/y/color/facet. Colour and facet accept
        # "(none)" as an explicit "no column" choice.
        def _make_col_combo(current: str | None, allow_none: bool = False) -> QComboBox:
            cb = QComboBox()
            if allow_none:
                cb.addItem("(none)")
            cb.addItems(available)
            if current and current in available:
                cb.setCurrentText(current)
            elif allow_none:
                cb.setCurrentText("(none)")
            return cb

        x_combo     = _make_col_combo(None)
        y_combo     = _make_col_combo(None)
        color_combo = _make_col_combo(None, allow_none=True)
        facet_combo = _make_col_combo(None, allow_none=True)
        _add_row("X axis:",            x_combo)
        _add_row("Y axis:",            y_combo)
        _add_row("Colour (optional):", color_combo)
        _add_row("Facet (optional):",  facet_combo)

        # Aggregate dropdown -- its enabled state and options change with
        # the chart type (see _apply_chart_rules below).
        agg_combo = QComboBox()
        agg_combo.addItems(AGGREGATES)
        agg_combo.setCurrentText("none")
        _add_row("Aggregate:", agg_combo)

        # One label reused for both warnings (bar+none and histogram+count
        # never overlap). retainSizeWhenHidden keeps the row so the dialog
        # does not jump when the text toggles.
        warning = QLabel()
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #B36B00; font-size: 11px;")
        warning.setMinimumHeight(45)
        policy = warning.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        warning.setSizePolicy(policy)
        warning.setVisible(False)
        grid.addWidget(warning, row, 1)
        row += 1

        def _apply_chart_rules(chart: str) -> None:
            """
            Enforce the plan-doc rules on the aggregate control:
                box / violin   -> disabled (always uses every row)
                histogram      -> only count/sum/mean (no median)
                scatter/line/bar -> all five options
            Also shows/hides the warning line.
            """
            if chart in ("box", "violin"):
                agg_combo.setCurrentText("none")
                agg_combo.setEnabled(False)
            elif chart == "histogram":
                current = agg_combo.currentText()
                agg_combo.clear()
                agg_combo.addItems(["count", "sum", "mean"])
                if current in ("count", "sum", "mean"):
                    agg_combo.setCurrentText(current)
                else:
                    agg_combo.setCurrentText("count")
                agg_combo.setEnabled(True)
            else:
                current = agg_combo.currentText()
                agg_combo.clear()
                agg_combo.addItems(AGGREGATES)
                if current in AGGREGATES:
                    agg_combo.setCurrentText(current)
                else:
                    agg_combo.setCurrentText("none")
                agg_combo.setEnabled(True)
            _refresh_warnings()

        def _refresh_warnings() -> None:
            chart = chart_combo.currentText()
            agg   = agg_combo.currentText()
            if chart == "bar" and agg == "none":
                warning.setText(
                    "Bar with aggregate=none sums rows sharing X. "
                    "Pick mean or sum for one bar per group."
                )
                warning.setVisible(True)
            elif chart == "histogram" and agg == "count":
                warning.setText(
                    "Histogram with aggregate=count ignores the Y column. "
                    "Pick sum or mean to use Y."
                )
                warning.setVisible(True)
            else:
                warning.setVisible(False)

        chart_combo.currentTextChanged.connect(_apply_chart_rules)
        agg_combo.currentTextChanged.connect(lambda _: _refresh_warnings())
        _apply_chart_rules(chart_combo.currentText())

        layout.addLayout(grid)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        # OK stays disabled until both x and y are chosen -- otherwise
        # create_display would hand None to Plotly and crash.
        ok_button = buttons.button(QDialogButtonBox.Ok)

        def _refresh_ok_enabled() -> None:
            ok_button.setEnabled(
                bool(x_combo.currentText()) and bool(y_combo.currentText())
            )

        x_combo.currentTextChanged.connect(lambda _: _refresh_ok_enabled())
        y_combo.currentTextChanged.connect(lambda _: _refresh_ok_enabled())
        _refresh_ok_enabled()

        # The dialog hands its answers back through parameter_values(),
        # keyed by the descriptor's parameter names. It stores nothing on
        # the operator instance -- two concurrent runs must not share one
        # set of values. x and y are always real column names here (the OK
        # button stays disabled until both are chosen); color and facet
        # are None when left as "(none)".
        chosen: dict = {}

        def _store():
            color = color_combo.currentText()
            facet = facet_combo.currentText()
            chosen.clear()
            chosen.update(
                {
                    "title":      title_edit.text().strip(),
                    "chart_type": chart_combo.currentText(),
                    "x":          x_combo.currentText(),
                    "y":          y_combo.currentText(),
                    "color":      None if color in ("", "(none)") else color,
                    "facet":      None if facet in ("", "(none)") else facet,
                    "aggregate":  agg_combo.currentText(),
                }
            )

        dialog.accepted.connect(_store)
        dialog.parameter_values = lambda: dict(chosen)
        return dialog

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
