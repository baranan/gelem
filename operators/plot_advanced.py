"""
operators/plot_advanced.py

PlotAdvancedOperator generates interactive Plotly Express figures
from the selected rows.

Workflow:
    1. User picks plot type, X column, Y column, color column from dialog
    2. create_display() checks if any rows are missing blendshape data
    3. If missing → extracts ALL 52 blendshapes for those rows
    4. Then builds the plot

Student C is responsible for implementing this operator.

Dependencies:
    pip install plotly kaleido matplotlib
"""

from __future__ import annotations
from pathlib import Path
import tempfile
import pandas as pd

from operators.base import BaseOperator
from operators.operator_constants import BLENDSHAPE_NAMES


PLOT_TYPES = ["bar", "line", "box", "violin", "scatter", "histogram"]

AVAILABLE_COLUMNS = ["file_name"] + BLENDSHAPE_NAMES


class PlotAdvancedOperator(BaseOperator):
    """
    Generates an interactive Plotly Express figure from selected rows.
    Auto-extracts blendshapes for rows that are missing data.
    """

    name = "plot_advanced"
    create_display_label = "Plot (interactive, Plotly)"
    output_columns = []
    requires_image = False

    def __init__(self):
        self._x_column: str | None = None
        self._y_column: str | None = None
        self._color_column: str | None = None
        self._plot_type: str = "bar"
        self._x_label: str | None = None
        self._y_label: str | None = None
        self._title: str | None = None

        self._output_dir = Path(tempfile.gettempdir()) / "gelem_plots_advanced"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ── Parameter Dialog (main thread) ────────────────────────────────

    def get_parameters_dialog(self, parent=None):
        """
        Shows a popup dialog for the researcher to configure the plot.
        Dropdown options are hardcoded: file_name + all 52 bs_ columns.
        """
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QLabel,
            QPushButton, QComboBox, QFormLayout, QLineEdit,
        )

        dialog = QDialog(parent)
        dialog.setWindowTitle("Plot (interactive, Plotly)")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Configure the plot:"))

        form = QFormLayout()

        type_combo = QComboBox()
        type_combo.addItems(PLOT_TYPES)
        type_combo.setCurrentText(self._plot_type)
        form.addRow("Plot type:", type_combo)

        x_combo = QComboBox()
        x_combo.addItems(AVAILABLE_COLUMNS)
        form.addRow("X axis data:", x_combo)

        x_label_input = QLineEdit()
        x_label_input.setText(x_combo.currentText())
        x_label_input.setPlaceholderText("Label shown on the X axis")
        form.addRow("X axis label:", x_label_input)
        x_combo.currentTextChanged.connect(lambda text: x_label_input.setText(text))

        y_combo = QComboBox()
        y_combo.addItem("(none)")
        y_combo.addItems(AVAILABLE_COLUMNS)
        form.addRow("Y axis data:", y_combo)

        y_label_input = QLineEdit()
        y_label_input.setText("")
        y_label_input.setPlaceholderText("Label shown on the Y axis")
        form.addRow("Y axis label:", y_label_input)
        y_combo.currentTextChanged.connect(
            lambda text: y_label_input.setText("" if text == "(none)" else text)
        )

        title_input = QLineEdit()
        title_input.setPlaceholderText("Chart title (auto-generated if empty)")
        form.addRow("Title:", title_input)

        color_combo = QComboBox()
        color_combo.addItem("(none)")
        color_combo.addItems(AVAILABLE_COLUMNS)
        form.addRow("Split by colour:", color_combo)

        layout.addLayout(form)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button)

        def _store_params():
            self._plot_type = type_combo.currentText()
            self._x_column = x_combo.currentText()

            y_text = y_combo.currentText()
            self._y_column = None if y_text == "(none)" else y_text

            color_text = color_combo.currentText()
            self._color_column = None if color_text == "(none)" else color_text

            self._x_label = x_label_input.text() or self._x_column
            self._y_label = y_label_input.text() or self._y_column
            self._title = title_input.text() or None

        dialog.accepted.connect(_store_params)
        return dialog

    # ── Extraction Logic (background thread) ──────────────────────────

    def _find_rows_to_extract(self, df: pd.DataFrame) -> list:
        """
        Returns DataFrame indices of rows where ANY of the 52 bs_ columns
        is missing (column doesn't exist, or value is None/NaN).
        """
        rows_to_extract = []

        for idx, row in df.iterrows():
            for bs_col in BLENDSHAPE_NAMES:
                if bs_col not in df.columns:
                    rows_to_extract.append(idx)
                    break
                val = row.get(bs_col)
                if val is None or pd.isna(val):
                    rows_to_extract.append(idx)
                    break

        return rows_to_extract

    def _extract_blendshapes(self, df: pd.DataFrame, rows_to_extract: list) -> tuple[pd.DataFrame, dict]:
        """
        Runs BlendshapeOperator on the specified rows to fill ALL 52 bs_ columns.
        Returns (updated DataFrame, row_updates dict for persistence).
        """
        from operators.blendshapes import BlendshapeOperator

        bs_op = BlendshapeOperator()
        row_updates = {}

        print(f"[PlotAdvanced] {len(rows_to_extract)} rows need blendshape extraction.")

        success_count = 0
        fail_count = 0

        for idx in rows_to_extract:
            row = df.loc[idx]
            full_path = row.get("full_path", "")
            row_id = row.get("row_id", str(idx))

            image = bs_op.load_image(full_path)
            if image is None:
                print(f"  {row_id}: FAILED — image not found ({full_path})")
                fail_count += 1
                continue

            result = bs_op.create_columns(row_id, image, {})

            if all(v is None for v in result.values()):
                print(f"  {row_id}: FAILED — no face detected")
                fail_count += 1
            else:
                for col, val in result.items():
                    df.loc[idx, col] = val
                row_updates[row_id] = result
                success_count += 1

        print(f"[PlotAdvanced] Extraction done: {success_count} succeeded, {fail_count} failed.")
        return df, row_updates

    # ── Validation ────────────────────────────────────────────────────

    def _validate_columns(self, df: pd.DataFrame) -> str | None:
        """Returns an error message if chosen columns are invalid, else None."""
        if not self._x_column or self._x_column not in df.columns:
            return f"X column '{self._x_column}' not found in data."

        needs_y = self._plot_type not in ("histogram",)
        if needs_y and (not self._y_column or self._y_column not in df.columns):
            return f"Y column '{self._y_column}' not found in data."

        return None

    # ── Plotting ──────────────────────────────────────────────────────

    def _build_figure(self, df: pd.DataFrame) -> tuple[Path, Path | None]:
        """
        Creates a Plotly HTML figure and a matplotlib static PNG.
        Returns (html_path, png_path or None).
        """
        import plotly.express as px

        color = self._color_column if (
            self._color_column and self._color_column in df.columns
        ) else None

        # Shorten file paths in X column so chart labels are readable.
        if self._x_column == "file_name":
            pass  # already short
        elif self._x_column in df.columns:
            sample_val = str(df[self._x_column].iloc[0]) if len(df) > 0 else ""
            if "/" in sample_val or "\\" in sample_val:
                df[self._x_column] = df[self._x_column].apply(
                    lambda p: Path(str(p)).name if pd.notna(p) else p
                )

        # Build Plotly figure.
        if self._plot_type == "bar":
            fig = px.bar(df, x=self._x_column, y=self._y_column, color=color, barmode="group")
        elif self._plot_type == "line":
            fig = px.line(df, x=self._x_column, y=self._y_column, color=color)
        elif self._plot_type == "box":
            fig = px.box(df, x=self._x_column, y=self._y_column, color=color)
        elif self._plot_type == "violin":
            fig = px.violin(df, x=self._x_column, y=self._y_column, color=color)
        elif self._plot_type == "scatter":
            fig = px.scatter(df, x=self._x_column, y=self._y_column, color=color)
        elif self._plot_type == "histogram":
            fig = px.histogram(df, x=self._x_column, color=color)
        else:
            fig = px.bar(df, x=self._x_column, y=self._y_column, color=color)

        fig.update_layout(template="plotly_white", width=800, height=500)

        # Save HTML.
        html_path = self._output_dir / "plot.html"
        fig.write_html(str(html_path))

        # Save static PNG via matplotlib.
        png_path = self._build_static_png(df, color)

        return html_path, png_path

    def _build_static_png(self, df: pd.DataFrame, color: str | None) -> Path | None:
        """Renders a matplotlib version of the chart and saves as PNG."""
        png_path = self._output_dir / "plot.png"
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(10, 6))

            if self._plot_type == "histogram":
                if color:
                    for group_val in df[color].dropna().unique():
                        subset = df[df[color] == group_val]
                        ax.hist(subset[self._x_column].dropna(), alpha=0.7,
                                label=str(group_val), edgecolor="white")
                    ax.legend(fontsize=10)
                else:
                    ax.hist(df[self._x_column].dropna(), alpha=0.8,
                            edgecolor="white", color="#4C72B0")
                ax.set_xlabel(self._x_label or self._x_column, fontsize=12)
                ax.set_ylabel(self._y_label or "Count", fontsize=12)

            elif self._plot_type == "scatter":
                ax.scatter(df[self._x_column], df[self._y_column],
                           alpha=0.7, s=60, edgecolors="white", linewidth=0.5)
                ax.set_xlabel(self._x_label or self._x_column, fontsize=12)
                ax.set_ylabel(self._y_label or self._y_column, fontsize=12)

            elif self._plot_type in ("box", "violin"):
                groups = sorted(df[self._x_column].dropna().unique())
                data_by_group = [
                    df[df[self._x_column] == g][self._y_column].dropna().values
                    for g in groups
                ]
                if self._plot_type == "box":
                    bp = ax.boxplot(data_by_group, labels=[str(g) for g in groups],
                                    patch_artist=True)
                    for patch in bp["boxes"]:
                        patch.set_facecolor("#4C72B0")
                        patch.set_alpha(0.7)
                else:
                    vp = ax.violinplot(data_by_group, showmedians=True)
                    for body in vp["bodies"]:
                        body.set_alpha(0.7)
                    ax.set_xticks(range(1, len(groups) + 1))
                    ax.set_xticklabels([str(g) for g in groups])
                ax.set_xlabel(self._x_label or self._x_column, fontsize=12)
                ax.set_ylabel(self._y_label or self._y_column, fontsize=12)

            else:
                # Bar or line chart.
                kind = "bar" if self._plot_type == "bar" else "line"
                if color:
                    grouped = df.groupby([self._x_column, color])[self._y_column].mean()
                    if kind == "bar":
                        grouped.unstack().plot(kind="bar", ax=ax, edgecolor="white")
                    else:
                        grouped.unstack().plot(kind="line", ax=ax, marker="o", linewidth=2)
                else:
                    grouped = df.groupby(self._x_column)[self._y_column].mean()
                    if kind == "bar":
                        grouped.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white")
                    else:
                        grouped.plot(kind="line", ax=ax, color="#4C72B0", marker="o", linewidth=2)
                ax.set_xlabel(self._x_label or self._x_column, fontsize=12)
                ax.set_ylabel(self._y_label or self._y_column, fontsize=12)

            default_title = (
                f"{self._plot_type.title()} — "
                f"{self._y_label or self._y_column or self._x_column} by "
                f"{self._x_label or self._x_column}"
            )
            ax.set_title(self._title or default_title, fontsize=14, fontweight="bold")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
            fig.tight_layout()
            fig.savefig(str(png_path), dpi=200, bbox_inches="tight")
            plt.close(fig)
            return png_path
        except Exception as e:
            print(f"[PlotAdvanced] PNG save failed: {e}")
            return None

    # ── Helpers ───────────────────────────────────────────────────────

    def _make_error_result(self, message: str, n_rows: int) -> dict:
        return {
            "operator_name": "plot_advanced",
            "error": message,
            "artifact_path": None,
            "plot_html": None,
            "n_rows": n_rows,
        }

    # ── Main Orchestrator (background thread) ─────────────────────────

    def create_display(self, df: pd.DataFrame) -> dict:
        """
        Orchestrates: check for missing data → extract if needed → plot.
        """
        df = df.copy()

        # Step 1: Find rows that need blendshape extraction.
        rows_to_extract = self._find_rows_to_extract(df)

        # Step 2: Extract if needed.
        row_updates = {}
        if rows_to_extract:
            df, row_updates = self._extract_blendshapes(df, rows_to_extract)
        else:
            print("[PlotAdvanced] All rows already have blendshape data.")

        # Step 3: Force bs_ columns to numeric.
        for col in BLENDSHAPE_NAMES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Step 4: Validate chosen columns exist.
        error = self._validate_columns(df)
        if error:
            return self._make_error_result(error, len(df))

        # Step 5: Build the figure.
        html_path, png_path = self._build_figure(df)

        print(f"[PlotAdvanced] Done. HTML: {html_path}, PNG: {png_path}")
        return {
            "operator_name": "plot_advanced",
            "artifact_path": str(png_path) if png_path else None,
            "plot_html": str(html_path),
            "n_rows": len(df),
            "row_updates": row_updates,
        }
