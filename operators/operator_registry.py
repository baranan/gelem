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
import threading
import pandas as pd

from operators.base import BaseOperator, OperatorSetupError


class OperatorRegistry:
    """
    Manages and runs analysis operators.

    Usage:
        registry = OperatorRegistry()
        registry.register(BlendshapeOperator())

        # Run create_columns on a list of rows. snapshot holds exactly
        # these rows, in this order -- AppController builds it with one
        # Dataset.snapshot_rows() call before calling this method.
        registry.run_create_columns(
            "blendshapes", snapshot, row_ids, table_name,
            on_item_complete=callback,
            on_progress=progress_callback,
            on_complete=done_callback,
        )

        # Run create_table on the active DataFrame:
        registry.run_create_table(
            "mean_face", df, group_by="condition",
            operation_id=operation_id,
            on_complete=done_callback,
        )

        # Run create_display on selected rows:
        registry.run_create_display(
            "summary_stats", df,
            operation_id=operation_id,
            on_complete=done_callback,
        )
    """

    def __init__(self):
        # Maps operator name -> BaseOperator instance.
        self._operators: dict[str, BaseOperator] = {}

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

    def list_create_columns_operators(self) -> list[tuple[str, str]]:
        """
        Returns all operators that implement create_columns().

        Returns:
            List of (operator_name, label) tuples for operators
            whose create_columns_label is not None.
        """
        return [
            (op.name, op.create_columns_label)
            for op in self._operators.values()
            if op.create_columns_label is not None
        ]

    def list_create_table_operators(self) -> list[tuple[str, str]]:
        """
        Returns all operators that implement create_table().

        Returns:
            List of (operator_name, label) tuples for operators
            whose create_table_label is not None.
        """
        return [
            (op.name, op.create_table_label)
            for op in self._operators.values()
            if op.create_table_label is not None
        ]

    def list_create_display_operators(self) -> list[tuple[str, str]]:
        """
        Returns all operators that implement create_display().

        Returns:
            List of (operator_name, label) tuples for operators
            whose create_display_label is not None.
        """
        return [
            (op.name, op.create_display_label)
            for op in self._operators.values()
            if op.create_display_label is not None
        ]

    # ── run_create_columns ────────────────────────────────────────────

    def run_create_columns(
        self,
        operator_name: str,
        snapshot: pd.DataFrame,
        row_ids: list[str],
        table_name: str,
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
                                          display_label: str, message: str)
            on_row_errors:    Called once at the end of the run if any rows
                              raised an unexpected exception. Lets the
                              controller surface a single end-of-run
                              summary to the user so unexpected failures
                              are visibly distinct from the normal "no
                              face detected" case.
                              Signature: (operation_id: str,
                                          display_label: str,
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

        if operator.create_columns_label is None:
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
                operator, snapshot, row_ids, table_name, operation_id,
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
        # How many per-row results we hand to on_item_complete. The
        # controller holds this run's completion back until it has
        # applied this many, so the completion never races ahead of the
        # last results into the bounded main-thread drain.
        emitted = 0

        for i, row_id in enumerate(row_ids):
            metadata = snapshot.iloc[i].to_dict()
            try:
                full_path = metadata.get("full_path", "")

                # Load image only if the operator requires it.
                if operator.requires_image:
                    image = operator.load_image(full_path)
                    if image is None:
                        print(
                            f"[OperatorRegistry] Could not load image "
                            f"for {row_id}: {full_path}"
                        )
                        continue
                else:
                    image = None

                result = operator.create_columns(row_id, image, metadata)

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
                    on_setup_error(operation_id, operator.display_label, str(e))
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
                    all_none = {name: None for name, _ in operator.output_columns}
                    on_item_complete(operation_id, table_name, row_id, all_none)
                    emitted += 1

            if on_progress is not None:
                percent = int((i + 1) / total * 100)
                on_progress(percent)

        if row_errors and on_row_errors is not None:
            on_row_errors(operation_id, operator.display_label, row_errors)

        if on_complete is not None:
            on_complete(operation_id, operator.name, emitted)

    # ── run_create_table ──────────────────────────────────────────────

    def run_create_table(
        self,
        operator_name: str,
        df: pd.DataFrame,
        group_by: str | list[str] | None,
        operation_id: str,
        on_complete=None,
        on_error=None,
    ) -> bool:
        """
        Runs create_table() in a background thread.

        The operator receives the full DataFrame and returns a new
        DataFrame. AppController stores it as a new named table via
        Dataset.create_table_from_df().

        Args:
            operator_name: Name of the operator to run.
            df:            The active table as a DataFrame.
            group_by:      Column or columns to group by, as chosen
                           by the researcher in the parameter dialog.
            operation_id:  Unique ID for this run, echoed back through
                           on_complete / on_error so AppController can
                           reject a result whose run is no longer live.
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

        if operator.create_table_label is None:
            print(
                f"[OperatorRegistry] Operator '{operator_name}' "
                f"does not implement create_table()."
            )
            return False

        thread = threading.Thread(
            target=self._run_create_table_worker,
            args=(operator, df, group_by, operation_id, on_complete, on_error),
            daemon=True,
        )
        thread.start()
        return True

    def _run_create_table_worker(
        self,
        operator: BaseOperator,
        df: pd.DataFrame,
        group_by,
        operation_id,
        on_complete,
        on_error,
    ) -> None:
        """Worker that runs create_table() in the background thread."""
        try:
            result_df = operator.create_table(df, group_by)
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
        on_complete=None,
        on_error=None,
    ) -> bool:
        """
        Runs create_display() in a background thread.

        The operator receives the selected rows as a DataFrame and
        returns a result dict. AppController passes this to
        ResultsPanel for display.

        Args:
            operator_name: Name of the operator to run.
            df:            The selected rows as a DataFrame.
            operation_id:  Unique ID for this run, echoed back through
                           on_complete / on_error so AppController can
                           reject a result whose run is no longer live.
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

        if operator.create_display_label is None:
            print(
                f"[OperatorRegistry] Operator '{operator_name}' "
                f"does not implement create_display()."
            )
            return False

        thread = threading.Thread(
            target=self._run_create_display_worker,
            args=(operator, df, operation_id, on_complete, on_error),
            daemon=True,
        )
        thread.start()
        return True

    def _run_create_display_worker(
        self,
        operator: BaseOperator,
        df: pd.DataFrame,
        operation_id,
        on_complete,
        on_error,
    ) -> None:
        """Worker that runs create_display() in the background thread."""
        try:
            result = operator.create_display(df)
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