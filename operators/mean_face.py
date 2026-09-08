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
from operators.blendshapes import BLENDSHAPE_NAMES
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModelLifecycle,
    ModeDescriptor,
    OperatorDescriptor,
    OutputSpec,
)


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

    # ------------------------------------------------------------------
    # Descriptor (P1.12d-1). What the two methods ACTUALLY do today:
    #  - TABLE mode (create_table): builds a new table, one row per
    #    group, with n_frames, the group-by column, and a
    #    "<bs>_mean" column per blendshape;
    #  - DISPLAY mode (create_display): shows one mean face for the
    #    whole selection, storing nothing;
    #  - NO parameters for either mode. get_parameters_dialog() is NOT
    #    overridden, so it returns None. create_table() looks for a
    #    "group_by" value in run.parameters, but the descriptor declares
    #    none, so it is always absent and grouping is unreachable today
    #    (before P1.12d-2a MainWindow read a "_group_by" attribute that
    #    nothing ever set -- same result). P1.12e adds the dialog.
    #    See the P1.12d-1 report, "should have a parameter but does not".
    #  - media_requirement METADATA for both: they work purely from the
    #    DataFrame;
    #  - PLACEHOLDER output: both modes write flat grey JPEGs and report
    #    every blendshape mean as 0.0. The group counting / grouping
    #    itself is real. The description says which parts are placeholder.
    #  - deterministic: fixed filenames, constant grey images, constant
    #    0.0 means.
    # ------------------------------------------------------------------
    descriptor = OperatorDescriptor(
        name="mean_face",
        version="1.0",
        description=(
            "Groups the selected rows (TABLE mode groups by a column, "
            "DISPLAY mode takes the whole selection as one group) and is "
            "meant to average their blendshapes and render a mean face. "
            "PLACEHOLDER today: the grouping and row counts are real, but "
            "every rendered face is a flat grey JPEG and every reported "
            "blendshape mean is 0.0."
        ),
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.TABLE,
                label="Mean face table",
                inputs=(
                    InputSpec(
                        name="active_table",
                        label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                media_requirement=MediaRequirement.METADATA,
                parameters=(),
                output=OutputSpec(creates_table=True),
                model_lifecycle=ModelLifecycle.NONE,
                deterministic=True,
                cacheable=True,
            ),
            ModeDescriptor(
                mode=ExecutionMode.DISPLAY,
                label="Mean face (quick view)",
                inputs=(
                    InputSpec(
                        name="active_table",
                        label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                media_requirement=MediaRequirement.METADATA,
                parameters=(),
                output=OutputSpec(is_display_only=True),
                model_lifecycle=ModelLifecycle.NONE,
                deterministic=True,
                cacheable=True,
            ),
        ),
    )

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

    def create_table(
        self,
        df: pd.DataFrame,
        run,
    ) -> pd.DataFrame:
        """
        Computes a mean face per group and returns a new DataFrame.
        One row per group, containing mean blendshape values and the
        path to the rendered mean face image.

        Args:
            df:  The active table as a DataFrame.
            run: The OperatorRun for this run. A grouping column would
                 arrive as run.parameters["group_by"]; this operator's
                 descriptor declares no parameters yet (P1.12e adds the
                 dialog), so run.parameters is empty and grouping is
                 unreachable -- exactly as it was before P1.12d-2a, when
                 MainWindow read a "_group_by" attribute nothing ever set.

        Returns:
            A new DataFrame with one row per group. Columns include
            the group-by column(s), mean blendshape values, and
            'mean_face_path'. Do not include row_id — OperatorRegistry
            generates new row_ids before storing.

        TODO (Student C): Implement this method.

        Suggested approach:
            1. If group_by is None, treat all rows as one group.
            2. For each group:
               a. Filter df to the group's rows.
               b. Compute mean of each blendshape column.
               c. Render a face with those mean values.
               d. Save the image and record the path.
            3. Build a DataFrame from the group results and return it.
        """
        # PLACEHOLDER: returns one row per group with gray images.
        from PIL import Image

        # A grouping column would come from run.parameters; the descriptor
        # declares none yet, so this is None and every row is one group.
        group_by = run.parameters.get("group_by")

        if group_by is None:
            groups = [("all", df)]
        else:
            groups = list(df.groupby(group_by))

        rows = []
        for group_val, group_df in groups:
            output_path = (
                self._output_dir / f"mean_face_{group_val}.jpg"
            )
            Image.new("RGB", (256, 256), color=(160, 160, 160)).save(
                str(output_path), "JPEG"
            )
            row = {
                "mean_face_path": str(output_path),
                "n_frames":       len(group_df),
            }
            if group_by:
                row[group_by] = group_val
            for bs_name in BLENDSHAPE_NAMES:
                row[f"{bs_name}_mean"] = 0.0
            rows.append(row)

        print(
            f"[MeanFaceOperator] PLACEHOLDER create_table — "
            f"{len(rows)} groups"
        )
        return pd.DataFrame(rows)

    def create_display(
        self,
        df: pd.DataFrame,
        run,
    ) -> dict:
        """
        Computes a single mean face from all rows in df and returns
        a result dict for display in the Results panel.

        Args:
            df:  The selected rows as a DataFrame.
            run: The OperatorRun for this run. This mode declares no
                 parameters, so run.parameters is empty; the argument is
                 here for the uniform operator contract.

        Returns:
            Dict with keys:
                'artifact_path': str path to the rendered mean face.
                'operator_name': 'mean_face'
                'n_frames':      int number of frames averaged.
                'summary':       dict of mean blendshape values.

        TODO (Student C): Implement this method.

        Suggested approach:
            1. For each blendshape in BLENDSHAPE_NAMES, compute the
               mean of that column across all rows in df, skipping NaN.
            2. Render a face deformed by those mean values.
            3. Save the image and return the result dict.
        """
        # PLACEHOLDER: creates a gray image.
        from PIL import Image

        n = len(df)
        output_path = self._output_dir / f"mean_face_{n}_frames.jpg"
        Image.new("RGB", (256, 256), color=(160, 160, 160)).save(
            str(output_path), "JPEG"
        )
        print(f"[MeanFaceOperator] PLACEHOLDER create_display — {n} frames")
        return {
            "artifact_path": str(output_path),
            "operator_name": "mean_face",
            "n_frames":      n,
            "summary":       {bs_name: 0.0 for bs_name in BLENDSHAPE_NAMES},
        }