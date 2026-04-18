"""
operators/base.py

Defines the BaseOperator class that all Gelem operators must inherit from.

An operator is a self-contained analysis plugin. It reads data and
produces one of three kinds of output:

    create_columns():
        Processes one row at a time. Returns a dict of new column
        values for that row. Used for per-frame analysis such as
        blendshape extraction. Called by OperatorRegistry in a
        background thread, once per row.

        Example: BlendshapeOperator reads one face image and returns
        52 blendshape values that become new columns in that row.

    create_table():
        Processes a whole DataFrame at once. Returns a new DataFrame
        representing an aggregated or derived table. Used for group-
        level computations such as computing a mean face per condition.

        Example: MeanFaceOperator groups rows by condition, computes
        mean blendshapes per group, renders one face per group, and
        returns a DataFrame with one row per condition.

    create_display():
        Processes a DataFrame (one row or many). Returns a dict
        describing something to show in the Results panel — an image,
        a statistics table, a plot. The result is never stored in any
        table.

        Example: SummaryStatsOperator computes mean, SD, min, max for
        selected rows and returns a dict shown in the Results panel.

An operator may implement one, two, or all three of these methods.
The Operators menu is built automatically from whichever labels are
set — if a label is None, that method is not shown in the menu.

THREADING RULES:
    - create_columns() runs in a background thread.
      Do NOT call any Qt functions inside create_columns().
      Do NOT call Dataset.update_row() inside create_columns().
      Just return the result dict.

    - create_table() and create_display() receive a plain DataFrame.
      They run in a background thread called by OperatorRegistry.
      Do NOT call any Qt functions inside these methods.
      Do NOT modify the input DataFrame — always work on a copy.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


class BaseOperator:
    """
    Abstract base class for all Gelem operators.

    To create a new operator:
        1. Create a new .py file in the operators/ folder.
        2. Define a class that inherits from BaseOperator.
        3. Set the name attribute.
        4. Set one or more label attributes for the methods you implement.
        5. Implement the corresponding create_*() methods.
        6. If your operator produces a new visual column type, add a
           renderer in column_types/renderers.py and register it in
           column_types/registry.py setup_defaults().
        7. Add your operator to operators_config.yaml.
        8. Register it in main.py create_app().

    Example — a simple per-row operator:

        class MyOperator(BaseOperator):
            name = "my_operator"
            create_columns_label = "Compute my score"
            output_columns = [("my_score", "numeric")]

            def create_columns(self, row_id, image, metadata):
                score = compute_something(image)
                return {"my_score": score}

    Example — an operator that supports all three modes:

        class MeanFaceOperator(BaseOperator):
            name = "mean_face"
            create_table_label   = "Mean face table"
            create_display_label = "Mean face (quick view)"

            def create_table(self, df, group_by):
                ...

            def create_display(self, df):
                ...
    """

    name: str = "unnamed"
    """
    Short identifier for this operator. Must be unique across all
    operators. Used as the key in operators_config.yaml and as the
    internal identifier in OperatorRegistry.
    Example: "blendshapes", "mean_face", "plot"
    """

    # ── Menu labels ───────────────────────────────────────────────────
    # Set these to a non-None string to make the corresponding method
    # appear in the Operators menu. Leave as None to hide it.

    create_columns_label: str | None = None
    """
    Label shown in the Operators menu for the create_columns() action.
    Example: "Extract blendshapes"
    None means this operator does not implement create_columns().
    """

    create_table_label: str | None = None
    """
    Label shown in the Operators menu for the create_table() action.
    Example: "Mean face table"
    None means this operator does not implement create_table().
    """

    create_display_label: str | None = None
    """
    Label shown in the Operators menu for the create_display() action.
    Example: "Mean face (quick view)"
    None means this operator does not implement create_display().
    """

    # ── Output columns (for create_columns only) ──────────────────────

    output_columns: list = []
    """
    List of (column_name, column_type_tag) pairs describing what
    this operator adds to the table when create_columns() is used.
    Not used by create_table() or create_display().

    Example:
        output_columns = [
            ("bs_jawOpen",      "numeric"),
            ("bs_mouthSmile_L", "numeric"),
            ("avatar_path",     "avatar_path"),
        ]
    """

    requires_image: bool = True
    """
    Whether create_columns() needs a full-resolution image to run.
    Set to False for operators that work purely from metadata columns.
    Only relevant for create_columns() — create_table() and
    create_display() receive a DataFrame and never load images.

    Example: BlendshapeAvatarOperator reads blendshape values from
    metadata and renders an avatar without needing the original image,
    so it sets requires_image = False.
    """

    # ── Core methods ──────────────────────────────────────────────────

    def create_columns(
        self,
        row_id: str,
        image: np.ndarray | None,
        metadata: dict,
    ) -> dict:
        """
        Processes one row and returns new column values for that row.
        Called by OperatorRegistry once per row in a background thread.

        This is how new columns get added to the table progressively.
        OperatorRegistry calls this once per row, and after each call
        AppController applies the returned dict to Dataset.update_row()
        on the main thread. The gallery tile for that row repaints
        immediately with the new values.

        Args:
            row_id:   The unique identifier of the row being processed.
            image:    The full-resolution image as a numpy array of
                      shape (height, width, 3), dtype uint8, RGB order.
                      None if requires_image is False.
            metadata: Dict of all existing column values for this row.
                      Read-only — do not modify this dict.

        Returns:
            A dict mapping column names to new values.
            Keys must match the column names in output_columns.
            Example: {"bs_jawOpen": 0.42, "bs_mouthSmile_L": 0.18}

            For visual columns, the value should be a string file path
            to an image the operator has already saved to disk.

        Raises:
            NotImplementedError: If the operator does not implement
                                 per-row processing.
        """
        raise NotImplementedError(
            f"Operator '{self.name}' does not implement create_columns()."
        )

    def create_table(
        self,
        df: pd.DataFrame,
        group_by: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Processes a DataFrame and returns a new DataFrame representing
        an aggregated or derived table. The result is stored as a new
        named table in Dataset.

        Called by OperatorRegistry in a background thread with the
        currently active table as df. The researcher specifies group_by
        via the parameter dialog.

        Args:
            df:       The active table as a DataFrame. Read-only —
                      work on a copy: df = df.copy()
            group_by: Column name or list of column names to group by,
                      as chosen by the researcher in the parameter
                      dialog. None if no grouping is needed.

        Returns:
            A new DataFrame. Must not contain a row_id column —
            OperatorRegistry generates new row_ids before storing.
            Must not modify the input df.

        Example return value for MeanFaceOperator grouped by condition:
            pd.DataFrame({
                "condition":       ["positive", "negative", "neutral"],
                "mean_face_path":  ["/tmp/pos.jpg", "/tmp/neg.jpg", "/tmp/neu.jpg"],
                "bs_jawOpen_mean": [0.42, 0.31, 0.38],
                ...
            })

        Raises:
            NotImplementedError: If the operator does not implement
                                 table creation.
        """
        raise NotImplementedError(
            f"Operator '{self.name}' does not implement create_table()."
        )

    def create_display(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """
        Processes a DataFrame and returns a result dict to display
        in the Results panel. The result is never stored in any table.

        Called by OperatorRegistry in a background thread. df contains
        the rows the researcher selected or filtered, as chosen in the
        scope dialog.

        Args:
            df: The selected rows as a DataFrame. Read-only.
                May be a single row (one-row DataFrame) or many rows.

        Returns:
            A dict describing the result. Common keys:
                "operator_name":  str name of this operator.
                "artifact_path":  str path to a generated image file.
                "summary":        dict of statistics or other data.
                "plot_html":      str path to an interactive HTML plot.

        Raises:
            NotImplementedError: If the operator does not implement
                                 display creation.
        """
        raise NotImplementedError(
            f"Operator '{self.name}' does not implement create_display()."
        )

    # ── Parameter dialog ──────────────────────────────────────────────

    def get_parameters_dialog(self, parent=None):
        """
        Returns a QDialog for collecting operator-specific parameters,
        or None if this operator needs no parameters.

        If this method returns a dialog, MainWindow will show it after
        the researcher chooses the run scope, before the operator starts.
        The dialog should store chosen parameters as instance attributes
        on self so that create_columns(), create_table(), and
        create_display() can read them.

        For create_table() operators, this dialog should ask which
        column to group by and what to name the new table.

        For plot operators, this dialog should ask which columns to
        plot, what kind of plot, and whether to group by a column.

        Args:
            parent: The parent widget for the dialog.

        Returns:
            A QDialog instance, or None.

        Example (in a subclass):
            def get_parameters_dialog(self, parent=None):
                from PySide6.QtWidgets import (
                    QDialog, QVBoxLayout, QComboBox,
                    QLabel, QPushButton
                )
                dialog = QDialog(parent)
                dialog.setWindowTitle("Parameters")
                layout = QVBoxLayout(dialog)
                layout.addWidget(QLabel("Group by:"))
                self._group_combo = QComboBox()
                self._group_combo.addItems(["condition", "session_id"])
                layout.addWidget(self._group_combo)
                btn = QPushButton("OK")
                btn.clicked.connect(dialog.accept)
                layout.addWidget(btn)
                # Store the chosen value so create_table() can read it.
                dialog.accepted.connect(
                    lambda: setattr(
                        self, '_group_by',
                        self._group_combo.currentText()
                    )
                )
                return dialog
        """
        return None

    # ── Convenience methods ───────────────────────────────────────────

    def load_image(self, full_path) -> np.ndarray | None:
        """
        Loads an image file and returns it as a numpy array (RGB uint8).
        Convenience method so operators do not need to import PIL.

        Args:
            full_path: Path to the image file (str or Path).

        Returns:
            numpy array of shape (height, width, 3), dtype uint8.
            None if the file does not exist or cannot be loaded.
        """
        try:
            from PIL import Image
            path = Path(str(full_path))
            if not path.exists():
                return None
            with Image.open(path) as img:
                return np.array(img.convert("RGB"), dtype=np.uint8)
        except Exception as e:
            print(f"[{self.name}] load_image error for {full_path}: {e}")
            return None

    def save_image(
        self,
        image: np.ndarray,
        output_path: Path,
        quality: int = 85,
    ) -> Path:
        """
        Saves a numpy array as a JPEG file.
        Convenience method for operators that produce image outputs.

        Args:
            image:       numpy array of shape (height, width, 3), uint8.
            output_path: Where to save the file.
            quality:     JPEG quality 1-95. Default 85.

        Returns:
            The output_path that was written.
        """
        from PIL import Image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_image = Image.fromarray(image)
        pil_image.save(output_path, "JPEG", quality=quality)
        return output_path

    def __repr__(self) -> str:
        labels = []
        if self.create_columns_label:
            labels.append(f"columns='{self.create_columns_label}'")
        if self.create_table_label:
            labels.append(f"table='{self.create_table_label}'")
        if self.create_display_label:
            labels.append(f"display='{self.create_display_label}'")
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"{', '.join(labels)})"
        )