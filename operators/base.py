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
The Operators menu is built automatically from the operator's
`descriptor` (operators/descriptor.py): one menu entry per declared
execution mode, carrying that mode's `label`.

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

    To create a new operator (see operators/CLAUDE.md for the full
    guide, which is the authority):
        1. Create a new .py file in the operators/ folder.
        2. Define a class that inherits from BaseOperator.
        3. Set the name attribute.
        4. Build the `descriptor` (operators/descriptor.py): one
           ModeDescriptor per method you implement, each carrying the
           menu label and, for a COLUMNS mode, its output columns.
        5. Implement the corresponding create_*() methods.
        6. If your operator produces a new visual column type, add a
           renderer in column_types/renderers.py and register it in
           column_types/registry.py setup_defaults().
        7. Add an entry for your operator to operators_config.yaml
           (its position there sets the Operators menu order).
        8. Add a matching factory to OPERATOR_FACTORIES in
           operators/operator_config.py. A disagreement between the two
           raises OperatorConfigError at startup.

    Example -- a simple per-row operator:

        class MyOperator(BaseOperator):
            name = "my_operator"
            descriptor = OperatorDescriptor(
                name="my_operator",
                version="1.0",
                description="...",
                modes=(
                    ModeDescriptor(
                        mode=ExecutionMode.COLUMNS,
                        label="Compute my score",
                        inputs=(...),
                        output=OutputSpec(columns=(
                            OutputColumn(name="my_score", type_tag="numeric"),
                        )),
                    ),
                ),
            )

            def create_columns(self, row_id, media, metadata, run):
                score = compute_something(media)
                return {"my_score": score}

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

    Some fields DESCRIBE the operator as it already behaves (the mode
    labels, the output columns) and must match what the code does today;
    others state a REQUIREMENT the runner must satisfy for the operator
    to be correct (model_lifecycle, media_requirement) and may declare a
    stricter contract than the current single-worker code happens to
    need.

    As of P1.12d-3 the descriptor is the SINGLE source of these facts.
    The Operators menu builds from it (one entry per declared mode,
    carrying that mode's `label`), AppController builds each run's
    OperatorRun from the mode's ModeDescriptor and its per-row column
    tags from the mode's OutputSpec, and a run will not start for an
    operator that carries no descriptor (or none for the requested
    mode). The legacy `*_label` / `output_columns` / `display_label`
    attributes that used to hold these facts in parallel are gone.
    """

    # ── What media a create_columns run is handed ─────────────────────
    #
    # Whether create_columns() is handed a decoded frame is declared on
    # the operator's descriptor, not by a boolean here. The COLUMNS
    # ModeDescriptor carries a `media_requirement` (operators/descriptor.py
    # -> MediaRequirement): FRAME means the runner decodes one frame and
    # passes it as `media`; METADATA and ADDRESS mean the runner decodes
    # nothing and passes `media=None` (ADDRESS is for an operator that
    # resolves the media itself from its metadata). VIDEO_SPAN and
    # AUDIO_SPAN are refused before the run starts -- the per-row runner
    # cannot supply a span. See operators/operator_registry.py
    # (_run_create_columns_worker) and AppController.run_create_columns.

    # ── Core methods ──────────────────────────────────────────────────

    def create_columns(
        self,
        row_id: str,
        media: np.ndarray | None,
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
            media:    The payload the runner decoded for this row, decided
                      by the mode's media_requirement. It is None for
                      METADATA and ADDRESS. For FRAME it is the
                      full-resolution frame as a numpy array of shape
                      (height, width, 3), dtype uint8, RGB order.
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
            Keys must match the column names the COLUMNS mode declares
            in its descriptor's OutputSpec.
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

    # ── Model factory ─────────────────────────────────────────────────

    def build_model(self):
        """
        FACTORY for this operator's model -- NOT an instance accessor.

        The runner calls this as often as the mode's declared
        ``model_lifecycle`` requires (operators/descriptor.py ->
        ``ModelLifecycle``): once per worker for ``PER_WORKER``, once per
        application for ``SHARED``, and never for ``NONE``. Whatever it
        returns is handed to the execution method as ``run.model``.

        An operator that keeps the result on ``self`` -- or builds a model
        anywhere other than here -- has defeated the declaration: an
        instance is a singleton, so a model on ``self`` is shared by two
        concurrent runs whatever lifecycle the descriptor names. Read
        ``run.model`` and nothing else.

        Returns ``None`` by default, which is correct for an operator that
        needs no model.

        Raise ``OperatorSetupError`` from here if a one-time prerequisite
        is missing -- typically an undownloaded model file. The runner
        aborts the run through its ``on_setup_error`` callback and
        processes no rows. See operators/CLAUDE.md -> "Where a model
        lives".
        """
        return None

    # ── Parameter dialog ──────────────────────────────────────────────
    #
    # There is no get_parameters_dialog(). An operator declares its
    # parameters as ParameterSpec entries on its ModeDescriptor (see
    # operators/descriptor.py); MainWindow builds the form from those
    # declarations (ui/parameter_dialog.py) and passes the collected
    # values to the controller in run.parameters. An operator module
    # contains no Qt -- that is guarded by
    # tests/test_operators_are_qt_free.py.

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
        # Describe the operator by the modes its descriptor declares --
        # e.g. "columns='Extract blendshapes'". An operator with no
        # descriptor is not runnable, but repr() must still not raise.
        if self.descriptor is not None:
            modes = ", ".join(
                f"{mode_descriptor.mode.name.lower()}="
                f"{mode_descriptor.label!r}"
                for mode_descriptor in self.descriptor.modes
            )
        else:
            modes = "no descriptor"
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"{modes})"
        )