"""
operators/operator_registry.py

OperatorRegistry manages all analysis plugins. It knows which operators
are available, runs them, and emits result payloads to AppController
for main-thread application.

It is the only component allowed to call operator code.

The three operator modes and how they are run:

    create_columns:
        Runs in a background thread, once per row_id, over a single
        pre-snapshotted DataFrame (AppController takes one
        Dataset.snapshot_rows() copy before the thread starts). After
        each row completes, calls
        on_item_complete(operation_id, table_name, row_id, result_dict).
        AppController collects a timer tick's worth of those results and
        applies them with one Dataset.apply_row_updates() call per
        table on the main thread, then repaints the affected tiles.

    create_table:
        Runs in a background thread with the full DataFrame. Returns a
        new DataFrame. AppController stores it as a new named table via
        Dataset.create_table_from_df(name, df).

    create_display:
        Runs in a background thread with the selected rows as a
        DataFrame. Returns a result dict. AppController passes it to
        ResultsPanel for display.

Threading model:
    All three modes run in background threads.
    Callbacks (on_item_complete, on_complete etc.) are called from the
    background thread. AppController routes them to the main thread via
    QTimer.singleShot.

This file is written centrally (not by a student).
"""

from __future__ import annotations
from pathlib import Path
import dataclasses
import threading
import pandas as pd

from operators.base import BaseOperator, OperatorSetupError
from operators.descriptor import ExecutionMode, MediaRequirement, ModelLifecycle


class OperatorRegistry:
    """
    Manages and runs analysis operators.

    Usage:
        registry = OperatorRegistry()
        registry.register(BlendshapeOperator())

        # Run create_columns on a list of rows. snapshot holds exactly
        # these rows, in this order -- AppController builds it, and the
        # OperatorRun (run), with one Dataset.snapshot_rows() call before
        # calling this method.
        registry.run_create_columns(
            "blendshapes", snapshot, row_ids, table_name, run,
            on_item_complete=callback,
            on_progress=progress_callback,
            on_complete=done_callback,
        )

        # Run create_table on the active DataFrame. Any group-by column is
        # in run.parameters, not a separate argument:
        registry.run_create_table(
            "mean_face", df,
            operation_id=operation_id,
            run=run,
            on_complete=done_callback,
        )

        # Run create_display on selected rows:
        registry.run_create_display(
            "summary_stats", df,
            operation_id=operation_id,
            run=run,
            on_complete=done_callback,
        )

    Every mode is handed one OperatorRun (operators/run_context.py),
    which the registry passes straight through as the final argument to
    the operator's execution method. Per-run parameter values live in
    run.parameters; the registry does not read them.
    """

    def __init__(self):
        # Maps operator name -> BaseOperator instance.
        self._operators: dict[str, BaseOperator] = {}

        # SHARED-lifecycle models: one instance per operator name, built
        # lazily by the runner on the first run that needs it and reused by
        # every run after. The lock makes two runs that start at the same
        # moment build at most one instance between them -- it is held
        # across the build() call on purpose, so the second run waits for
        # the first rather than racing it. SHARED models are rare and must
        # be demonstrably thread-safe, so the coarse lock costs nothing in
        # practice. NONE / PER_WORKER / PER_SEQUENCE never touch this.
        #
        # This cache lives for the process: it is NOT cleared when the
        # researcher opens a different project. That is correct only while
        # SHARED means what the descriptor says it means -- an immutable,
        # project-independent object (a lookup table, config, a pure
        # function). The first operator that actually declares SHARED must
        # confirm that, or teach AppController.load_project to clear this.
        # Nothing declares SHARED today.
        self._shared_models: dict[str, object] = {}
        self._shared_models_lock = threading.Lock()

    def _shared_model_for(self, operator: BaseOperator) -> object:
        """
        The one application-wide model instance for a SHARED-lifecycle
        operator, built on first use under ``_shared_models_lock`` so two
        runs starting at once cannot both build.

        Any ``OperatorSetupError`` raised by ``operator.build_model()``
        propagates to the caller (the worker), which aborts the run
        through ``on_setup_error``; nothing is cached, so a later run
        retries the build.
        """
        with self._shared_models_lock:
            if operator.name not in self._shared_models:
                self._shared_models[operator.name] = operator.build_model()
            return self._shared_models[operator.name]

    def register(self, operator: BaseOperator) -> None:
        """
        Registers an operator instance by its name.

        Args:
            operator: An instance of a BaseOperator subclass.
        """
        self._operators[operator.name] = operator
        print(f"[OperatorRegistry] Registered: {operator.name}")

    def list_operators(self) -> list[str]:
        """
        Returns the names of all registered operators.

        Returns:
            List of operator name strings.
        """
        return list(self._operators.keys())

    def get(self, operator_name: str) -> BaseOperator | None:
        """
        Returns the operator with the given name, or None.

        Args:
            operator_name: Name of the operator to retrieve.

        Returns:
            The BaseOperator instance, or None if not found.
        """
        return self._operators.get(operator_name, None)

    def list_operators_for_mode(
        self, mode: ExecutionMode
    ) -> list[tuple[str, str]]:
        """
        Returns the operators whose descriptor declares ``mode``, in
        registration order (which is operators_config.yaml's order, so
        this is the Operators-menu order for that section).

        Args:
            mode: The ExecutionMode to list -- COLUMNS, TABLE or DISPLAY.

        Returns:
            List of (operator_name, label) tuples, where ``label`` is the
            ModeDescriptor's label for that mode. An operator that
            declares two modes appears in the list for each, once.
            An operator with no descriptor contributes nothing.
        """
        listed: list[tuple[str, str]] = []
        for op in self._operators.values():
            if op.descriptor is None:
                continue
            mode_descriptor = op.descriptor.mode_for(mode)
            if mode_descriptor is not None:
                listed.append((op.name, mode_descriptor.label))
        return listed

    # ── run_create_columns ────────────────────────────────────────────

    def run_create_columns(
        self,
        operator_name: str,
        snapshot: pd.DataFrame,
        row_ids: list[str],
        table_name: str,
        run,
        operation_id: str = "",
        on_item_complete=None,
        on_progress=None,
        on_complete=None,
        on_setup_error=None,
        on_row_errors=None,
    ) -> bool:
        """
        Runs create_columns() over an ordered group of rows in a
        background thread. AppController takes one snapshot of the
        selected rows (Dataset.snapshot_rows) on the main thread before
        calling this method, so the worker never reads from Dataset
        directly and never receives one dict per row built in advance.

        snapshot holds exactly the rows named by row_ids, in the same
        order, as a single DataFrame. The worker builds each row's
        metadata dict from it as it reaches that row. AppController built
        this DataFrame, not the worker, so per CLAUDE.md's data-ownership
        rule it is not "a worker's own DataFrame" -- the worker must
        treat it as read-only.

        For each completed row, calls
        on_item_complete(operation_id, table_name, row_id, result).
        AppController batches a tick's results and routes them to
        Dataset.apply_row_updates() on the main thread.

        Args:
            operator_name: Name of the operator to run.
            snapshot:      DataFrame with one row per entry in row_ids,
                           in the same order. Read-only.
            row_ids:       Ordered row_ids matching snapshot's rows.
            table_name:    Table the rows belong to.
            run:           The OperatorRun for this run
                           (operators/run_context.py), built and validated
                           by AppController. Passed straight through as the
                           final argument to operator.create_columns();
                           the registry does not read run.parameters.
            operation_id:  Unique ID for this run. Travels through every
                           callback below so AppController can reject a
                           result from a run that is no longer live.
            on_item_complete: Called after each row completes.
                              Signature: (operation_id, table_name,
                                          row_id, result)
                              Called from background thread.
            on_progress:      Called with progress percentage (0-100).
            on_complete:      Called when all rows are done.
                              Signature: (operation_id: str,
                                          operator_name: str,
                                          emitted: int)
                              `emitted` is the number of per-row results
                              handed to on_item_complete.
            on_setup_error:   Called if the operator raises
                              OperatorSetupError on any row. The run is
                              aborted and remaining rows are skipped.
                              Signature: (operation_id: str,
                                          label: str, message: str)
                              `label` is the COLUMNS mode's descriptor
                              label, computed by the worker.
            on_row_errors:    Called once at the end of the run if any rows
                              raised an unexpected exception. Lets the
                              controller surface a single end-of-run
                              summary to the user so unexpected failures
                              are visibly distinct from the normal "no
                              face detected" case.
                              Signature: (operation_id: str,
                                          label: str,
                                          errors: list[tuple[str, str, str]])
                              Each tuple is (row_id, exc_type_name, message).

        Raises:
            ValueError: If snapshot does not have exactly one row per
                        entry in row_ids. The worker pairs the two by
                        position (snapshot.iloc[i] for row_ids[i]), so a
                        length mismatch would silently attribute one
                        row's data to a different row_id for every row
                        from that point on -- wrong output, not a
                        missing row, and with no error. Dataset.
                        snapshot_rows() is the primary defence (it raises
                        if it cannot find a requested row_id, so it
                        cannot itself hand back a short frame); this is a
                        second check against any other caller of this
                        method.
        """
        # Returns True if a worker thread was started, False if the run
        # could not begin (unknown operator or wrong mode). AppController
        # registers the run in _live_runs before calling this and
        # deregisters it on a False return, so a run that never starts
        # never lingers as "live".
        operator = self._operators.get(operator_name)
        if operator is None:
            print(f"[OperatorRegistry] Unknown operator: {operator_name}")
            return False

        if (
            operator.descriptor is None
            or operator.descriptor.mode_for(ExecutionMode.COLUMNS) is None
        ):
            print(
                f"[OperatorRegistry] Operator '{operator_name}' "
                f"does not implement create_columns()."
            )
            return False

        if len(snapshot) != len(row_ids):
            raise ValueError(
                f"run_create_columns: snapshot has {len(snapshot)} rows "
                f"but row_ids has {len(row_ids)} -- the worker pairs them "
                f"by position, so they must be the same length."
            )

        thread = threading.Thread(
            target=self._run_create_columns_worker,
            args=(
                operator, snapshot, row_ids, table_name, run, operation_id,
                on_item_complete, on_progress, on_complete,
                on_setup_error, on_row_errors,
            ),
            daemon=True,
        )
        thread.start()
        return True

    def _run_create_columns_worker(
        self,
        operator: BaseOperator,
        snapshot: pd.DataFrame,
        row_ids: list[str],
        table_name: str,
        run,
        operation_id: str,
        on_item_complete,
        on_progress,
        on_complete,
        on_setup_error,
        on_row_errors,
    ) -> None:
        """
        Worker that runs create_columns() in the background thread.
        Builds each row's metadata dict from the pre-snapshotted
        DataFrame as it reaches that row — never reads Dataset, and
        never receives 530,000 dicts built in advance.
        """
        total = len(row_ids)
        row_errors: list[tuple[str, str, str]] = []
        # The user-facing label for this run's error callbacks. It is the
        # COLUMNS mode's descriptor label -- computed here, on the worker,
        # from the run object it was handed, so the controller never
        # rebuilds it on this thread just to phrase an error (a CLAUDE.md
        # threading rule). run.spec.mode_descriptor is the COLUMNS
        # ModeDescriptor: AppController.run_create_columns built the run
        # for ExecutionMode.COLUMNS.
        label = run.spec.mode_descriptor.label
        # How many per-row results we hand to on_item_complete. The
        # controller holds this run's completion back until it has
        # applied this many, so the completion never races ahead of the
        # last results into the bounded main-thread drain.
        emitted = 0

        # What the runner decodes for each row is decided by this run's
        # declared media_requirement (operators/descriptor.py ->
        # MediaRequirement), read off the run's mode descriptor -- not by a
        # boolean flag on the operator. The five declared values and what
        # each one means for this per-row COLUMNS runner:
        #
        #   FRAME    -- the operator needs one decoded frame. The runner
        #               decodes the row's image below, exactly as before,
        #               and skips the row if it cannot be loaded.
        #   METADATA -- the operator works purely from the row's ordinary
        #               columns. The runner decodes nothing and hands it
        #               media=None.
        #   ADDRESS  -- the operator resolves the media itself from its
        #               metadata (the video frame-extraction operator is
        #               the real case). The runner decodes nothing here
        #               either and hands it media=None.
        #   VIDEO_SPAN / AUDIO_SPAN -- an ordered span of video or audio.
        #               This per-row runner cannot produce one: it sees a
        #               single row at a time and has no decoder. A run that
        #               declares either is REFUSED BEFORE IT STARTS, in
        #               AppController.run_create_columns, naming the
        #               operator and the requirement -- it must never reach
        #               this worker. Approximating a span with one frame,
        #               or silently handing None, is exactly the
        #               wrong-number failure P1.12d exists to remove, so if
        #               one somehow reaches here we surface it as a setup
        #               error and process no rows rather than guess.
        media_requirement = run.spec.mode_descriptor.media_requirement
        if media_requirement in (
            MediaRequirement.VIDEO_SPAN,
            MediaRequirement.AUDIO_SPAN,
        ):
            if on_setup_error is not None:
                on_setup_error(
                    operation_id,
                    label,
                    f"needs {media_requirement.name} media, which the "
                    f"per-row runner cannot supply. AppController should "
                    f"have refused this run before it started.",
                )
            if on_complete is not None:
                on_complete(operation_id, operator.name, emitted)
            return

        # FRAME is the only requirement that makes the runner decode.
        needs_frame = media_requirement is MediaRequirement.FRAME

        # Build this run's model BEFORE the row loop, honouring the mode's
        # declared model_lifecycle (operators/descriptor.py ->
        # ModelLifecycle). build_model() is a FACTORY (operators/base.py):
        #
        #   NONE         -- no model. run.model stays None.
        #   PER_WORKER   -- build once here, in this worker, and hand the
        #                   same instance to every row of this run. This
        #                   per-row runner starts one worker thread per
        #                   run, so "per worker" is "per run" today; two
        #                   concurrent runs get two workers and therefore
        #                   two isolated instances.
        #   SHARED       -- one instance for the whole application, built
        #                   and cached on the registry under a lock so two
        #                   runs starting at once cannot both build.
        #   PER_SEQUENCE -- one isolated instance per clip-run, reset at
        #                   the sequence boundary. This per-row runner has
        #                   no sequence concept and cannot honour it;
        #                   AppController.run_create_columns refuses such a
        #                   run before it starts. The check below is a
        #                   defensive guard for a run that somehow reaches
        #                   here anyway -- same shape as the VIDEO_SPAN /
        #                   AUDIO_SPAN guard above.
        #
        # If build_model() raises OperatorSetupError -- e.g. a model file
        # was never downloaded -- the run aborts through on_setup_error
        # having processed NO rows. That is stricter than the old lazy
        # load inside create_columns, which only raised on the first row
        # whose media happened to decode: a run where every row failed to
        # decode used to report success having processed zero rows and
        # never mention the missing model.
        model_lifecycle = run.spec.mode_descriptor.model_lifecycle
        if model_lifecycle is ModelLifecycle.PER_SEQUENCE:
            if on_setup_error is not None:
                on_setup_error(
                    operation_id,
                    label,
                    f"declares model_lifecycle {model_lifecycle.name}, which "
                    f"the per-row runner cannot honour -- it has no sequence "
                    f"boundary to reset at. AppController should have refused "
                    f"this run before it started.",
                )
            if on_complete is not None:
                on_complete(operation_id, operator.name, emitted)
            return

        try:
            if model_lifecycle is ModelLifecycle.PER_WORKER:
                run = dataclasses.replace(run, model=operator.build_model())
            elif model_lifecycle is ModelLifecycle.SHARED:
                run = dataclasses.replace(
                    run, model=self._shared_model_for(operator)
                )
            # NONE: leave run.model as it is (None).
        except Exception as e:
            # Any failure to build the model aborts the run before the row
            # loop, through the same on_setup_error path a raised
            # OperatorSetupError takes: the run cannot proceed without its
            # model. OperatorSetupError is the expected case (an
            # undownloaded model file) and gets its exact message; anything
            # else (a corrupt model file making the library raise
            # RuntimeError, say) is wrapped so the message still names the
            # operator and the cause. Either way on_setup_error THEN
            # on_complete are called and the worker returns, so the run is
            # torn down and deregistered rather than left live by a dead
            # worker thread.
            if isinstance(e, OperatorSetupError):
                message = str(e)
            else:
                message = (
                    f"could not build its model: "
                    f"{type(e).__name__}: {e}"
                )
            print(
                f"[OperatorRegistry] Model build failed for "
                f"'{operator.name}': {type(e).__name__}: {e}"
            )
            if on_setup_error is not None:
                on_setup_error(operation_id, label, message)
            if on_complete is not None:
                on_complete(operation_id, operator.name, emitted)
            return

        for i, row_id in enumerate(row_ids):
            metadata = snapshot.iloc[i].to_dict()
            try:
                full_path = metadata.get("full_path", "")

                # FRAME: decode one frame and skip the row if it will not
                # load. METADATA / ADDRESS: hand the operator None.
                if needs_frame:
                    media = operator.load_image(full_path)
                    if media is None:
                        print(
                            f"[OperatorRegistry] Could not load image "
                            f"for {row_id}: {full_path}"
                        )
                        continue
                else:
                    media = None

                result = operator.create_columns(row_id, media, metadata, run)

                if on_item_complete is not None:
                    on_item_complete(operation_id, table_name, row_id, result)
                    emitted += 1

            except NotImplementedError:
                print(
                    f"[OperatorRegistry] Operator '{operator.name}' "
                    f"does not implement create_columns()."
                )
                break
            except OperatorSetupError as e:
                # Setup-level failure (e.g. required model file missing).
                # Abort the run rather than spamming the same error per row.
                print(
                    f"[OperatorRegistry] Setup error in '{operator.name}': {e}"
                )
                if on_setup_error is not None:
                    # Pass the operator's own display label so the
                    # controller never rebuilds it from a worker thread.
                    on_setup_error(operation_id, label, str(e))
                break
            except Exception as e:
                # Unexpected per-row failure (mediapipe crash, bug, malformed
                # image, etc.). Mark the row as missing for consistency with
                # the operator's no-face path, and remember it so we can
                # surface a single summary at the end of the run.
                print(
                    f"[OperatorRegistry] Unexpected error in '{operator.name}' "
                    f"on {row_id}: {type(e).__name__}: {e}"
                )
                row_errors.append((row_id, type(e).__name__, str(e)))
                if on_item_complete is not None:
                    all_none = {
                        column.name: None
                        for column
                        in run.spec.mode_descriptor.output.columns
                    }
                    on_item_complete(operation_id, table_name, row_id, all_none)
                    emitted += 1

            if on_progress is not None:
                percent = int((i + 1) / total * 100)
                on_progress(percent)

        if row_errors and on_row_errors is not None:
            on_row_errors(operation_id, label, row_errors)

        if on_complete is not None:
            on_complete(operation_id, operator.name, emitted)

    # ── run_create_table ──────────────────────────────────────────────

    def run_create_table(
        self,
        operator_name: str,
        df: pd.DataFrame,
        operation_id: str,
        run,
        on_complete=None,
        on_error=None,
    ) -> bool:
        """
        Runs create_table() in a background thread.

        The operator receives the full DataFrame and the OperatorRun and
        returns a new DataFrame. AppController stores it as a new named
        table via Dataset.create_table_from_df().

        Args:
            operator_name: Name of the operator to run.
            df:            The active table as a DataFrame.
            operation_id:  Unique ID for this run, echoed back through
                           on_complete / on_error so AppController can
                           reject a result whose run is no longer live.
            run:           The OperatorRun for this run
                           (operators/run_context.py). Passed straight
                           through as the final argument to
                           operator.create_table(). A group-by column, if
                           the operator declares one, is in run.parameters
                           -- there is no separate group_by argument.
            on_complete:   Called when done.
                           Signature: (operation_id: str,
                                       operator_name: str,
                                       result_df: pd.DataFrame)
                           Called from background thread — AppController
                           routes to main thread.
            on_error:      Called if create_table raises an exception.
                           Signature: (operation_id: str,
                                       operator_name: str, message: str)
                           Called from background thread.
        """
        # Returns True if a worker was started, False otherwise -- see
        # run_create_columns for why AppController needs to know.
        operator = self._operators.get(operator_name)
        if operator is None:
            print(f"[OperatorRegistry] Unknown operator: {operator_name}")
            return False

        if (
            operator.descriptor is None
            or operator.descriptor.mode_for(ExecutionMode.TABLE) is None
        ):
            print(
                f"[OperatorRegistry] Operator '{operator_name}' "
                f"does not implement create_table()."
            )
            return False

        thread = threading.Thread(
            target=self._run_create_table_worker,
            args=(operator, df, operation_id, run, on_complete, on_error),
            daemon=True,
        )
        thread.start()
        return True

    def _run_create_table_worker(
        self,
        operator: BaseOperator,
        df: pd.DataFrame,
        operation_id,
        run,
        on_complete,
        on_error,
    ) -> None:
        """Worker that runs create_table() in the background thread."""
        try:
            result_df = operator.create_table(df, run)
            if on_complete is not None:
                on_complete(operation_id, operator.name, result_df)
        except NotImplementedError:
            # Report it as an error so AppController deregisters the run
            # it registered before starting this worker -- a silent
            # return would leave it in _live_runs forever.
            print(
                f"[OperatorRegistry] Operator '{operator.name}' "
                f"does not implement create_table()."
            )
            if on_error is not None:
                on_error(
                    operation_id, operator.name,
                    f"Operator '{operator.name}' does not implement "
                    f"create_table().",
                )
        except Exception as e:
            print(
                f"[OperatorRegistry] Error in create_table "
                f"for '{operator.name}': {e}"
            )
            if on_error is not None:
                on_error(operation_id, operator.name, str(e))

    # ── run_create_display ────────────────────────────────────────────

    def run_create_display(
        self,
        operator_name: str,
        df: pd.DataFrame,
        operation_id: str,
        run,
        on_complete=None,
        on_error=None,
    ) -> bool:
        """
        Runs create_display() in a background thread.

        The operator receives the selected rows as a DataFrame and the
        OperatorRun and returns a result dict. AppController passes this
        to ResultsPanel for display.

        Args:
            operator_name: Name of the operator to run.
            df:            The selected rows as a DataFrame.
            operation_id:  Unique ID for this run, echoed back through
                           on_complete / on_error so AppController can
                           reject a result whose run is no longer live.
            run:           The OperatorRun for this run
                           (operators/run_context.py). Passed straight
                           through as the final argument to
                           operator.create_display().
            on_complete:   Called when done.
                           Signature: (operation_id: str,
                                       operator_name: str,
                                       result: dict)
                           Called from background thread.
            on_error:      Called if create_display raises an exception.
                           Signature: (operation_id: str,
                                       operator_name: str, message: str)
                           Called from background thread.
        """
        # Returns True if a worker was started, False otherwise -- see
        # run_create_columns for why AppController needs to know.
        operator = self._operators.get(operator_name)
        if operator is None:
            print(f"[OperatorRegistry] Unknown operator: {operator_name}")
            return False

        if (
            operator.descriptor is None
            or operator.descriptor.mode_for(ExecutionMode.DISPLAY) is None
        ):
            print(
                f"[OperatorRegistry] Operator '{operator_name}' "
                f"does not implement create_display()."
            )
            return False

        thread = threading.Thread(
            target=self._run_create_display_worker,
            args=(operator, df, operation_id, run, on_complete, on_error),
            daemon=True,
        )
        thread.start()
        return True

    def _run_create_display_worker(
        self,
        operator: BaseOperator,
        df: pd.DataFrame,
        operation_id,
        run,
        on_complete,
        on_error,
    ) -> None:
        """Worker that runs create_display() in the background thread."""
        try:
            result = operator.create_display(df, run)
            if on_complete is not None:
                on_complete(operation_id, operator.name, result)
        except NotImplementedError:
            # See _run_create_table_worker: report it so the registered
            # run is deregistered rather than leaking.
            print(
                f"[OperatorRegistry] Operator '{operator.name}' "
                f"does not implement create_display()."
            )
            if on_error is not None:
                on_error(
                    operation_id, operator.name,
                    f"Operator '{operator.name}' does not implement "
                    f"create_display().",
                )
        except Exception as e:
            print(
                f"[OperatorRegistry] Error in create_display "
                f"for '{operator.name}': {e}"
            )
            if on_error is not None:
                on_error(operation_id, operator.name, str(e))