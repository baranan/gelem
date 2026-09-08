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

# operators/descriptor.py is standard-library only and imports nothing from
# this package, so a plain runtime import here is not circular. We import it
# at runtime (not merely under TYPE_CHECKING) so the annotation below stays
# resolvable and every operator module can build its descriptor from the
# shared vocabulary.
from operators.descriptor import OperatorDescriptor


class OperatorSetupError(Exception):
    """
    Raised by an operator when a one-time setup step has not been
    completed yet (e.g. a required model file has not been downloaded).
    The runner aborts the run on the first occurrence and surfaces the
    message to the user via AppController.error_occurred.
    """


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
        7. Add an entry for your operator to operators_config.yaml
           (its position there sets the Operators menu order).
        8. Add a matching factory to OPERATOR_FACTORIES in
           operators/operator_config.py. A disagreement between the two
           raises OperatorConfigError at startup.

    Example — a simple per-row operator:

        class MyOperator(BaseOperator):
            name = "my_operator"
            create_columns_label = "Compute my score"
            output_columns = [("my_score", "numeric")]

            def create_columns(self, row_id, image, metadata, run):
                score = compute_something(image)
                return {"my_score": score}

    Example — an operator that supports two modes:

        class MeanFaceOperator(BaseOperator):
            name = "mean_face"
            create_table_label   = "Mean face table"
            create_display_label = "Mean face (quick view)"

            def create_table(self, df, run):
                ...

            def create_display(self, df, run):
                ...

    Every create_*() method takes a final `run` argument -- an
    OperatorRun (operators/run_context.py). Per-run values reach the
    operator through `run.parameters` (a read-only mapping keyed by the
    operator's declared descriptor parameter names) and NEVER as
    attributes on `self`: operator instances are singletons, so a value
    stored on `self` would be shared by two concurrent runs.
    """

    name: str = "unnamed"
    """
    Short identifier for this operator. Must be unique across all
    operators. Used as the key in operators_config.yaml and as the
    internal identifier in OperatorRegistry.
    Example: "blendshapes", "mean_face", "plot"
    """

    descriptor: OperatorDescriptor | None = None
    """
    The operator's pure-data self-description: its modes, the inputs and
    parameters each mode takes, and what each mode produces (see
    operators/descriptor.py for the vocabulary).

    Some fields DESCRIBE the operator as it already behaves (the labels,
    the output columns) and must match what the code does today; others
    state a REQUIREMENT the runner must satisfy for the operator to be
    correct (model_lifecycle, media_requirement) and may declare a
    stricter contract than the current single-worker code happens to
    need.

    As of P1.12d-1 every concrete operator sets this, and a consistency
    test (tests/test_operator_descriptors_match.py) pins each descriptor
    against the operator's existing name / *_label / output_columns
    attributes. As of P1.12d-2a AppController consumes the descriptor at
    run time: every run looks up the mode's ModeDescriptor to build the
    OperatorRun, and a run will not start for an operator that carries no
    descriptor (or none for the requested mode). The Operators menu and
    parameter dialog still build from the legacy *_label / parameters
    attributes; wiring those to the descriptor is later P1.12d work.
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
            ("avatar_path",     "media_path"),
        ]
    """

    # ── What media a create_columns run is handed ─────────────────────
    #
    # Whether create_columns() is handed a decoded frame is declared on
    # the operator's descriptor, not by a boolean here. The COLUMNS
    # ModeDescriptor carries a `media_requirement` (operators/descriptor.py
    # -> MediaRequirement): FRAME means the runner decodes one frame and
    # passes it as `image`; METADATA and ADDRESS mean the runner decodes
    # nothing and passes `image=None` (ADDRESS is for an operator that
    # resolves the media itself from its metadata). VIDEO_SPAN and
    # AUDIO_SPAN are refused before the run starts -- the per-row runner
    # cannot supply a span. See operators/operator_registry.py
    # (_run_create_columns_worker) and AppController.run_create_columns.

    # ── Identity ──────────────────────────────────────────────────────

    @property
    def display_label(self) -> str:
        """
        The operator's user-facing name for dialogs and error messages.

        One operator can expose up to three menu actions with three
        labels; this picks whichever is set, falling back to the
        internal name. It lives here so that OperatorRegistry can pass a
        ready label into its error callbacks and the controller never
        rebuilds this chain on a worker thread just to phrase an error
        (a CLAUDE.md threading rule).
        """
        return (
            self.create_columns_label
            or self.create_table_label
            or self.create_display_label
            or self.name
        )

    # ── Core methods ──────────────────────────────────────────────────

    def create_columns(
        self,
        row_id: str,
        image: np.ndarray | None,
        metadata: dict,
        run,
    ) -> dict:
        """
        Processes one row and returns new column values for that row.
        Called by OperatorRegistry once per row in a background thread.

        This is how new columns get added to the table progressively.
        OperatorRegistry calls this once per row; AppController collects
        a timer tick's worth of returned dicts and applies them with one
        Dataset.apply_row_updates() call per table on the main thread,
        then repaints the affected gallery tiles.

        Args:
            row_id:   The unique identifier of the row being processed.
            image:    The full-resolution image as a numpy array of
                      shape (height, width, 3), dtype uint8, RGB order.
                      None unless this operator's COLUMNS ModeDescriptor
                      declares media_requirement = FRAME.
            metadata: Dict of all existing column values for this row.
                      Read-only — do not modify this dict.
            run:      The OperatorRun for this run (operators/run_context.py).
                      Read per-run values from run.parameters (a read-only
                      mapping keyed by this operator's declared descriptor
                      parameter names). Never store or read parameters on
                      self -- operator instances are shared between runs.
                      run also carries run.data (frozen input snapshots),
                      run.paths (project directories) and run.cancelled()
                      (checked between units of work); those are dormant
                      for a per-row operator today. A run.data table holds
                      only the rows this run was selected to process, not
                      the whole stored table -- its name and version just
                      identify where those rows came from.

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
        run,
    ) -> pd.DataFrame:
        """
        Processes a DataFrame and returns a new DataFrame representing
        an aggregated or derived table. The result is stored as a new
        named table in Dataset.

        Called by OperatorRegistry in a background thread with the
        currently active table as df.

        Args:
            df:   The active table as a DataFrame. Read-only —
                  work on a copy: df = df.copy()
            run:  The OperatorRun for this run (operators/run_context.py).
                  Read every per-run value from run.parameters (a
                  read-only mapping keyed by this operator's declared
                  descriptor parameter names) -- e.g. a group-by column
                  the researcher chose in the dialog. Never read
                  parameters off self.

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
        run,
    ) -> dict:
        """
        Processes a DataFrame and returns a result dict to display
        in the Results panel. The result is never stored in any table.

        Called by OperatorRegistry in a background thread. df contains
        the rows the researcher selected or filtered, as chosen in the
        scope dialog.

        Args:
            df:  The selected rows as a DataFrame. Read-only.
                 May be a single row (one-row DataFrame) or many rows.
            run: The OperatorRun for this run (operators/run_context.py).
                 Read every per-run value from run.parameters (a
                 read-only mapping keyed by this operator's declared
                 descriptor parameter names). Never read parameters off
                 self.

        Returns:
            A dict describing the result. Common keys:
                "operator_name":  str name of this operator.
                "artifact_path":  str path to a generated image file.
                "summary":        dict of statistics or other data.
                "html_path":      str path to an interactive HTML plot.

        Raises:
            NotImplementedError: If the operator does not implement
                                 display creation.
        """
        raise NotImplementedError(
            f"Operator '{self.name}' does not implement create_display()."
        )

    # ── Parameter dialog ──────────────────────────────────────────────

    def get_parameters_dialog(self, parent=None, columns=None):
        """
        Returns a QDialog for collecting operator-specific parameters,
        or None if this operator needs no parameters.

        If this method returns a dialog, MainWindow shows it after the
        researcher chooses the run scope, before the operator starts.
        The dialog MUST expose a ``parameter_values() -> dict`` method
        that returns the chosen values keyed by this operator's declared
        descriptor parameter names. MainWindow calls it after the dialog
        is accepted and passes the dict to the controller, which builds
        the run's OperatorRunSpec from it. The dialog must NOT store
        anything on the operator instance: instances are shared between
        runs, so a value on ``self`` would be clobbered by a second
        concurrent run -- exactly the failure P1.12d-2a removes.

        A dialog returned without a ``parameter_values`` method is a bug:
        MainWindow raises rather than silently running with no parameters.

        Args:
            parent:  The parent widget for the dialog.
            columns: List of column names in the active table, supplied
                     by MainWindow. Operators that need to populate
                     column dropdowns should read from this list rather
                     than reaching into the controller themselves.
                     None when called outside MainWindow (e.g. tests).

        Returns:
            A QDialog instance exposing parameter_values(), or None.

        Example (in a subclass):
            def get_parameters_dialog(self, parent=None, columns=None):
                from PySide6.QtWidgets import (
                    QDialog, QVBoxLayout, QComboBox,
                    QLabel, QPushButton
                )
                dialog = QDialog(parent)
                dialog.setWindowTitle("Parameters")
                layout = QVBoxLayout(dialog)
                layout.addWidget(QLabel("Group by:"))
                group_combo = QComboBox()
                group_combo.addItems(columns or [])
                layout.addWidget(group_combo)
                btn = QPushButton("OK")
                btn.clicked.connect(dialog.accept)
                layout.addWidget(btn)
                # Hand the chosen values back by NAME through
                # parameter_values(). Every key must be a parameter this
                # operator's descriptor declares. Nothing is stored on self.
                chosen = {}
                def _store():
                    chosen["group_by"] = group_combo.currentText()
                dialog.accepted.connect(_store)
                dialog.parameter_values = lambda: dict(chosen)
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