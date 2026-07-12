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
    # Setting create_display_label makes this operator appear in the
    # Operators menu under "Display results for selection".
    create_display_label = "Plot (interactive, Plotly)"

    def __init__(self, output_dir: Path | None = None):
        # Default to a project-relative folder, not the system Temp
        # directory (same pattern as VideoFramesOperator). main.py can
        # pass an explicit output_dir once Dataset.save()/load() define
        # a real project folder.
        self._output_dir = output_dir or (
            Path.cwd() / "gelem_project" / "plots"
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._chart_type: str        = "scatter"
        self._x:          str | None = None
        self._y:          str | None = None
        self._color:      str | None = None
        self._facet:      str | None = None
        self._aggregate:  str        = "none"
        self._title:      str | None = None  # None -> auto ("{y} by {x}")

    def get_parameters_dialog(self, parent=None, columns=None):
        """Show a dialog and store the researcher's choices as instance attributes.

        Collect:
            self._chart_type  -- one of: scatter | line | bar | box | violin | histogram
            self._x           -- column name for the horizontal axis
            self._y           -- column name for the vertical axis
            self._color       -- (optional) column to colour marks by group; None if not chosen
            self._facet       -- (optional) column to split into a grid of small plots; None if not chosen
            self._aggregate   -- one of: none | count | sum | mean | median

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
        if self._title:
            title_edit.setText(self._title)
        title_edit.setPlaceholderText("Auto: \"{y} by {x}\"")
        _add_row("Title:", title_edit)

        # Chart-type dropdown.
        chart_combo = QComboBox()
        chart_combo.addItems(CHART_TYPES)
        chart_combo.setCurrentText(self._chart_type)
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

        x_combo     = _make_col_combo(self._x)
        y_combo     = _make_col_combo(self._y)
        color_combo = _make_col_combo(self._color, allow_none=True)
        facet_combo = _make_col_combo(self._facet, allow_none=True)
        _add_row("X axis:",            x_combo)
        _add_row("Y axis:",            y_combo)
        _add_row("Colour (optional):", color_combo)
        _add_row("Facet (optional):",  facet_combo)

        # Aggregate dropdown -- its enabled state and options change with
        # the chart type (see _apply_chart_rules below).
        agg_combo = QComboBox()
        agg_combo.addItems(AGGREGATES)
        agg_combo.setCurrentText(self._aggregate)
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

        def _store():
            self._chart_type = chart_combo.currentText()
            self._x          = x_combo.currentText() or None
            self._y          = y_combo.currentText() or None
            color            = color_combo.currentText()
            self._color      = None if color in ("", "(none)") else color
            facet            = facet_combo.currentText()
            self._facet      = None if facet in ("", "(none)") else facet
            self._aggregate  = agg_combo.currentText()
            self._title      = title_edit.text().strip() or None

        dialog.accepted.connect(_store)
        return dialog

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
        title = self._title or f"{y} by {x}"
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
