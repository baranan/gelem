"""
operators/plot_operator.py

PlotOperator generates a matplotlib horizontal bar chart of blendshape
values for a single selected photo.

Workflow:
    1. User selects exactly 1 photo in the gallery
    2. Dialog shows all 52 blendshapes as checkboxes (unchecked by default)
    3. User must check at least 1 blendshape
    4. If blendshape values are missing, they are extracted automatically
    5. Bar chart is created and shown in the Results panel

Student C is responsible for implementing this operator.

Dependencies:
    pip install matplotlib
"""

from __future__ import annotations
from pathlib import Path
import tempfile
import pandas as pd

from operators.base import BaseOperator
from operators.operator_constants import BLENDSHAPE_NAMES


class PlotOperator(BaseOperator):
    """
    Generates a matplotlib horizontal bar chart of blendshape values
    for a single selected photo.

    The researcher selects which blendshapes to plot via the parameter
    dialog. The result is a PNG image shown in the Results panel.
    """

    name = "plot"

    # This makes the operator appear under "create_display" in the menu
    # (produces a visual result in the Results panel, not a new column).
    create_display_label = "Plot columns (bar chart)"

    # No output columns — create_display doesn't add columns to the table.
    output_columns = []

    # We don't need the framework to load the image for us.
    # If blendshapes are missing, we load the image ourselves in _extract_blendshapes().
    requires_image = False

    def __init__(self):
        # Will be populated by the dialog — list of blendshape names the user checked.
        self._columns: list[str] = []
        self._output_dir = Path(tempfile.gettempdir()) / "gelem_plots"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ── Parameter Dialog (main thread) ────────────────────────────────

    def get_parameters_dialog(self, parent=None):
        """
        Shows a popup dialog with checkboxes for all 52 blendshapes.
        All checkboxes start unchecked.
        At least one must be selected — if none are checked when the user
        clicks OK, an error message pops up and the dialog stays open.
        """
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QLabel,
            QPushButton, QCheckBox, QScrollArea, QWidget, QMessageBox,
        )

        dialog = QDialog(parent)
        dialog.setWindowTitle("Plot — choose blendshapes")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select which blendshapes to include in the bar chart:"))

        # Scrollable area with one checkbox per blendshape (all 52).
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        checkboxes = []
        for bs_name in BLENDSHAPE_NAMES:
            checkbox = QCheckBox(bs_name)
            checkbox.setChecked(False)  # All unchecked by default.
            scroll_layout.addWidget(checkbox)
            checkboxes.append(checkbox)

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(400)
        layout.addWidget(scroll)

        ok_button = QPushButton("OK")
        layout.addWidget(ok_button)

        def _on_ok():
            """Validate at least 1 checkbox is selected before closing."""
            selected = [cb.text() for cb in checkboxes if cb.isChecked()]
            if not selected:
                # Show error and keep the dialog open so user can try again.
                QMessageBox.warning(
                    dialog,
                    "No blendshapes selected",
                    "Please select at least one blendshape to plot.",
                )
                return  # Don't close — user goes back to the checkboxes.
            self._columns = selected
            dialog.accept()

        ok_button.clicked.connect(_on_ok)
        return dialog

    # ── Extraction Logic (background thread) ──────────────────────────

    def _needs_extraction(self, row: pd.Series) -> bool:
        """Returns True if the row is missing any blendshape data."""
        for bs_col in BLENDSHAPE_NAMES:
            val = row.get(bs_col)
            if val is None or pd.isna(val):
                return True
        return False

    def _extract_blendshapes(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Extracts all 52 blendshapes for the single row using BlendshapeOperator.
        Same pattern as PlotAdvancedOperator — reuses the existing extraction code.
        Returns (updated DataFrame, row_updates dict for persistence to the table).
        """
        from operators.blendshapes import BlendshapeOperator

        bs_op = BlendshapeOperator()
        row_updates = {}

        idx = df.index[0]
        row = df.loc[idx]
        row_id = row.get("row_id", str(idx))
        full_path = row.get("full_path", "")

        # Load the image file ourselves (since requires_image=False).
        image = bs_op.load_image(full_path)
        if image is None:
            print(f"[PlotOperator] FAILED — image not found ({full_path})")
            return df, row_updates

        # Run blendshape extraction on the image.
        result = bs_op.create_columns(row_id, image, {})

        if all(v is None for v in result.values()):
            print(f"[PlotOperator] FAILED — no face detected for {row_id}")
        else:
            # Write extracted values into the DataFrame.
            for col, val in result.items():
                df.loc[idx, col] = val
            # row_updates tells the controller to persist these values
            # back to the table (same pattern as PlotAdvancedOperator).
            row_updates[row_id] = result
            print(f"[PlotOperator] Extraction succeeded for {row_id}.")

        return df, row_updates

    # ── Plotting ──────────────────────────────────────────────────────

    def _build_chart(self, row: pd.Series, row_id: str) -> Path | None:
        """Creates a horizontal bar chart for the selected blendshapes."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.style.use("seaborn-v0_8-whitegrid")

            columns = self._columns
            values = [float(row.get(col, 0.0) or 0.0) for col in columns]

            # Scale figure height based on number of bars (up to 52).
            n_bars = len(columns)
            fig_height = max(4, n_bars * 0.45)
            fig, ax = plt.subplots(figsize=(8, fig_height))

            bars = ax.barh(columns, values, color="#4C72B0", edgecolor="white")

            # Extra space on the right so the numeric values don't get clipped.
            ax.set_xlim(0, 1.15)
            ax.set_title(f"Blendshapes — {row_id}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Value")

            # Show the numeric value to the right of each bar.
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_width() + 0.03,  # Small gap between bar end and number.
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}",
                    va="center", fontsize=9,
                )

            fig.tight_layout()
            output_path = self._output_dir / f"{row_id}_plot.png"
            fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            return output_path

        except Exception as e:
            print(f"[PlotOperator] Chart error for {row_id}: {e}")
            return None

    # ── Main Orchestrator (background thread) ─────────────────────────

    def create_display(self, df: pd.DataFrame) -> dict:
        """
        Main entry point. Called by the framework with the selected rows.

        Steps:
            1. Validate exactly 1 row selected (error otherwise)
            2. Extract blendshapes if missing
            3. Build the bar chart
            4. Return result dict (with row_updates so values get saved)
        """
        # --- Step 1: Must be exactly 1 photo selected ---
        if len(df) != 1:
            return {
                "operator_name": "plot",
                "error": f"Please select exactly 1 photo (you selected {len(df)}).",
                "artifact_path": None,
                "n_rows": len(df),
            }

        df = df.copy()
        row_updates = {}

        idx = df.index[0]
        row = df.loc[idx]

        # --- Step 2: Extract blendshapes if this row doesn't have them yet ---
        if self._needs_extraction(row):
            df, row_updates = self._extract_blendshapes(df)
            row = df.loc[idx]

        row_id = row.get("row_id", str(idx))

        # Force selected columns to numeric (in case they were stored as strings).
        for col in self._columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        row = df.loc[idx]

        # --- Step 3: Build the bar chart ---
        png_path = self._build_chart(row, row_id)

        # --- Step 4: Return result ---
        # row_updates tells the controller to save extracted blendshape
        # values back to the table so future runs won't need re-extraction.
        return {
            "operator_name": "plot",
            "artifact_path": str(png_path) if png_path else None,
            "n_rows": 1,
            "row_updates": row_updates,
        }
