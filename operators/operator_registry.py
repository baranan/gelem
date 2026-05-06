"""
operators/operator_registry.py

OperatorRegistry manages all analysis plugins. It knows which operators
are available, runs them, and emits result payloads to AppController
for main-thread application.

It is the only component allowed to call operator code.

The three operator modes and how they are run:

    create_columns:
        Runs in a background thread, once per row_id. After each row
        completes, calls on_item_complete(row_id, result_dict).
        AppController applies the dict to Dataset.update_row() on the
        main thread. The gallery tile repaints immediately.

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

from operators.base import BaseOperator


class OperatorRegistry:
    """
    Manages and runs analysis operators.

    Usage:
        registry = OperatorRegistry()
        registry.register(BlendshapeOperator())

        # Run create_columns on a list of rows:
        registry.run_create_columns(
            "blendshapes", row_ids, dataset,
            on_item_complete=callback,
            on_progress=progress_callback,
            on_complete=done_callback,
        )

        # Run create_table on the active DataFrame:
        registry.run_create_table(
            "mean_face", df, group_by="condition",
            on_complete=done_callback,
        )

        # Run create_display on selected rows:
        registry.run_create_display(
            "summary_stats", df,
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
        work_items: list[dict],
        operation_id: str = "",
        on_item_complete=None,
        on_progress=None,
        on_complete=None,
    ) -> None:
        """
        Runs create_columns() on pre-snapshotted work items in a background
        thread. AppController snapshots row data on the main thread before
        calling this method, so the worker never reads from Dataset directly.

        Each work item is a dict:
            {
                "row_id":     str,
                "table_name": str,
                "row_data":   dict,   # snapshot of the row at launch time
            }

        For each completed item, calls
        on_item_complete(operation_id, table_name, row_id, result).
        AppController routes this to Dataset.update_row() on the main thread.

        Args:
            operator_name:    Name of the operator to run.
            work_items:       Pre-snapshotted list of row inputs.
            operation_id:     Unique ID for this run (reserved for
                              future stale-result detection).
            on_item_complete: Called after each row completes.
                              Signature: (operation_id, table_name,
                                          row_id, result)
                              Called from background thread.
            on_progress:      Called with progress percentage (0-100).
            on_complete:      Called when all rows are done.
                              Signature: (operator_name: str)
        """
        operator = self._operators.get(operator_name)
        if operator is None:
            print(f"[OperatorRegistry] Unknown operator: {operator_name}")
            return

        if operator.create_columns_label is None:
            print(
                f"[OperatorRegistry] Operator '{operator_name}' "
                f"does not implement create_columns()."
            )
            return

        thread = threading.Thread(
            target=self._run_create_columns_worker,
            args=(
                operator, work_items, operation_id,
                on_item_complete, on_progress, on_complete,
            ),
            daemon=True,
        )
        thread.start()

    def _run_create_columns_worker(
        self,
        operator: BaseOperator,
        work_items: list[dict],
        operation_id: str,
        on_item_complete,
        on_progress,
        on_complete,
    ) -> None:
        """
        Worker that runs create_columns() in the background thread.
        Processes only the pre-snapshotted work items — never reads Dataset.
        """
        total = len(work_items)

        for i, item in enumerate(work_items):
            row_id     = item["row_id"]
            table_name = item["table_name"]
            metadata   = item["row_data"]
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
                        # Return None for every output column so the row
                        # still appears in the table and progress updates.
                        result = {col: None for col, _ in operator.output_columns}
                        if on_item_complete is not None:
                            on_item_complete(operation_id, table_name, row_id, result)
                        if on_progress is not None:
                            percent = int((i + 1) / total * 100)
                            on_progress(percent)
                        continue
                else:
                    image = None

                result = operator.create_columns(row_id, image, metadata)

                if on_item_complete is not None:
                    on_item_complete(operation_id, table_name, row_id, result)

            except NotImplementedError:
                print(
                    f"[OperatorRegistry] Operator '{operator.name}' "
                    f"does not implement create_columns()."
                )
                break
            except Exception as e:
                print(
                    f"[OperatorRegistry] Error in create_columns "
                    f"for '{operator.name}' on {row_id}: {e}"
                )

            if on_progress is not None:
                percent = int((i + 1) / total * 100)
                on_progress(percent)

        if on_complete is not None:
            on_complete(operator.name)

    # ── run_create_table ──────────────────────────────────────────────

    def run_create_table(
        self,
        operator_name: str,
        df: pd.DataFrame,
        group_by: str | list[str] | None,
        on_complete=None,
    ) -> None:
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
            on_complete:   Called when done.
                           Signature: (operator_name: str,
                                       result_df: pd.DataFrame)
                           Called from background thread — AppController
                           routes to main thread.
        """
        operator = self._operators.get(operator_name)
        if operator is None:
            print(f"[OperatorRegistry] Unknown operator: {operator_name}")
            return

        if operator.create_table_label is None:
            print(
                f"[OperatorRegistry] Operator '{operator_name}' "
                f"does not implement create_table()."
            )
            return

        thread = threading.Thread(
            target=self._run_create_table_worker,
            args=(operator, df, group_by, on_complete),
            daemon=True,
        )
        thread.start()

    def _run_create_table_worker(
        self,
        operator: BaseOperator,
        df: pd.DataFrame,
        group_by,
        on_complete,
    ) -> None:
        """Worker that runs create_table() in the background thread."""
        try:
            result_df = operator.create_table(df, group_by)
            if on_complete is not None:
                on_complete(operator.name, result_df)
        except NotImplementedError:
            print(
                f"[OperatorRegistry] Operator '{operator.name}' "
                f"does not implement create_table()."
            )
        except Exception as e:
            print(
                f"[OperatorRegistry] Error in create_table "
                f"for '{operator.name}': {e}"
            )

    # ── run_create_display ────────────────────────────────────────────

    def run_create_display(
        self,
        operator_name: str,
        df: pd.DataFrame,
        on_complete=None,
    ) -> None:
        """
        Runs create_display() in a background thread.

        The operator receives the selected rows as a DataFrame and
        returns a result dict. AppController passes this to
        ResultsPanel for display.

        Args:
            operator_name: Name of the operator to run.
            df:            The selected rows as a DataFrame.
            on_complete:   Called when done.
                           Signature: (operator_name: str,
                                       result: dict)
                           Called from background thread.
        """
        operator = self._operators.get(operator_name)
        if operator is None:
            print(f"[OperatorRegistry] Unknown operator: {operator_name}")
            return

        if operator.create_display_label is None:
            print(
                f"[OperatorRegistry] Operator '{operator_name}' "
                f"does not implement create_display()."
            )
            return

        thread = threading.Thread(
            target=self._run_create_display_worker,
            args=(operator, df, on_complete),
            daemon=True,
        )
        thread.start()

    def _run_create_display_worker(
        self,
        operator: BaseOperator,
        df: pd.DataFrame,
        on_complete,
    ) -> None:
        """Worker that runs create_display() in the background thread."""
        try:
            result = operator.create_display(df)
            if on_complete is not None:
                on_complete(operator.name, result)
        except NotImplementedError:
            print(
                f"[OperatorRegistry] Operator '{operator.name}' "
                f"does not implement create_display()."
            )
        except Exception as e:
            print(
                f"[OperatorRegistry] Error in create_display "
                f"for '{operator.name}': {e}"
            )