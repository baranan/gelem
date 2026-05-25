"""
operators/mean_face.py

MeanFaceOperator computes a mean face from a group of frames by
averaging their blendshape values and rendering the result.

It supports two modes:

    create_table:
        Groups rows by a researcher-chosen column (e.g. condition),
        computes mean blendshapes per group, renders one face per
        group, and returns a new DataFrame — one row per group.
        Use this when you want a permanent table of mean faces.

    create_display:
        Takes all selected rows as one group, computes their mean
        blendshapes, renders one mean face, and returns it for
        display in the Results panel without storing anything.
        Use this for a quick one-off mean face of a selection.

Student C is responsible for implementing this operator.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from operators.base import BaseOperator
from operators.operator_constants import BLENDSHAPE_NAMES


class MeanFaceOperator(BaseOperator):
    """
    Computes mean faces from groups of frames.
    Supports both create_table (grouped) and create_display (quick view).
    """

    name = "mean_face"
    create_table_label   = "Mean face table"
    create_display_label = "Mean face (quick view)"
    output_columns       = []
    requires_image       = False

    def __init__(self, output_dir: Path | None = None):
        """
        Creates the operator.

        Args:
            output_dir: Where to save mean face images.
        """
        import tempfile
        self._output_dir = output_dir or (
            Path(tempfile.gettempdir()) / "gelem_mean_faces"
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ── Extraction Logic (reuses PlotAdvancedOperator) ────────────────

    def _ensure_blendshapes(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Checks for missing blendshape data and extracts if needed.
        Returns (updated_df, row_updates_dict).
        """
        from operators.plot_advanced import PlotAdvancedOperator

        extractor = PlotAdvancedOperator()
        rows_to_extract = extractor._find_rows_to_extract(df)

        row_updates = {}
        if rows_to_extract:
            df, row_updates = extractor._extract_blendshapes(df, rows_to_extract)
        else:
            print("[MeanFace] All rows already have blendshape data.")

        for col in BLENDSHAPE_NAMES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df, row_updates

    # ── create_table ─────────────────────────────────────────────────

    def create_table(
        self,
        df: pd.DataFrame,
        group_by: str | list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Computes a mean face per group and returns a new DataFrame
        plus row_updates so extracted blendshapes are persisted.

        Args:
            df:       The active table as a DataFrame.
            group_by: Column to group by (e.g. "condition").
                      If None, treats all rows as one group.

        Returns:
            A tuple of (result_df, row_updates).
            result_df: one row per group. Do not include
            row_id — OperatorRegistry generates those.
            row_updates: dict of row_id -> extracted blendshape values.
        """
        from PIL import Image

        df = df.copy()
        df, row_updates = self._ensure_blendshapes(df)

        # If no grouping, treat all rows as a single group called "all".
        if group_by is None:
            groups = [("all", df)]
        else:
            groups = list(df.groupby(group_by))

        rows = []
        for group_value, group_df in groups:
            # Compute mean of each blendshape for this group.
            row = {"n_frames": len(group_df)}
            if group_by:
                row[group_by] = group_value

            for bs_name in BLENDSHAPE_NAMES:
                if bs_name in group_df.columns:
                    mean_value = group_df[bs_name].dropna().mean()
                    row[f"{bs_name}_mean"] = float(mean_value) if not np.isnan(mean_value) else None
                else:
                    row[f"{bs_name}_mean"] = None

            # TODO: Placeholder image — will be replaced with avatar rendering later.
            output_path = self._output_dir / f"mean_face_{group_value}.jpg"
            Image.new("RGB", (256, 256), color=(160, 160, 160)).save(
                str(output_path), "JPEG"
            )
            row["mean_face_path"] = str(output_path)
            rows.append(row)

        return pd.DataFrame(rows), row_updates

    # ── create_display ───────────────────────────────────────────────

    def create_display(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """
        Computes the mean blendshape values across all selected rows
        and returns the result for display in the Results panel.

        Args:
            df: The selected rows as a DataFrame.

        Returns:
            Dict with mean blendshape values and a placeholder image
            (image rendering will be added when avatar approach is decided).
        """
        from PIL import Image

        df = df.copy()
        df, row_updates = self._ensure_blendshapes(df)

        # Compute the mean of each blendshape across all selected rows.
        mean_blendshapes = {}
        for bs_name in BLENDSHAPE_NAMES:
            if bs_name in df.columns:
                mean_value = df[bs_name].dropna().mean()
                mean_blendshapes[bs_name] = {
                    "mean": float(mean_value) if not np.isnan(mean_value) else None
                }
            else:
                mean_blendshapes[bs_name] = {"mean": None}

        n_frames = len(df)

        # Placeholder image — will be replaced with avatar rendering later.
        output_path = self._output_dir / f"mean_face_{n_frames}_frames.jpg"
        Image.new("RGB", (256, 256), color=(160, 160, 160)).save(
            str(output_path), "JPEG"
        )

        return {
            "artifact_path": str(output_path),
            "operator_name": "mean_face",
            "n_frames":      n_frames,
            "summary":       mean_blendshapes,
            "row_updates":   row_updates,
        }