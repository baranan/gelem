"""
controller.py

AppController is the wiring layer between the UI and the rest of the
application. It receives events from the UI, calls the appropriate
components, and sends results back to the UI via Qt signals.

AppController contains no data logic and no display logic. If you find
business logic accumulating here, it belongs in one of the other
components instead.

Threading model:
    AppController lives on the main thread.
    Operator results arrive from background threads via callbacks.
    AppController routes all callbacks to the main thread using a
    queue drained by a QTimer every 50ms.

This file is written centrally (not by a student).
"""

from __future__ import annotations
from pathlib import Path
import uuid
import pandas as pd

from PySide6.QtCore import QObject, Signal, QTimer


class AppController(QObject):
    """
    Wires together Dataset, QueryEngine, ArtifactStore,
    ColumnTypeRegistry, and OperatorRegistry in response to UI events.

    All signals are emitted on the main thread. All Dataset mutations
    happen on the main thread.

    Signals:
        gallery_updated:         Flat ordered list of row_ids.
        grouped_gallery_updated: Dict of group_value -> list[row_ids].
        row_selected:            Metadata dict for the selected row.
        columns_updated:         List of all registered column names.
        tables_updated:          List of all table names in the project.
        thumbnail_ready:         row_id whose thumbnail is now available.
        row_updated:             row_id whose data has changed.
        operator_progress:       Integer 0-100 progress percentage.
        operator_complete:       Name of the operator that finished.
        merge_report_ready:      MergeReport object for display.
        error_occurred:          Human-readable error message string.
        display_result_ready:    Result dict from a create_display
                                 operator, for ResultsPanel.
        table_created:           Name of a newly created table.
    """

    gallery_updated          = Signal(list)
    grouped_gallery_updated  = Signal(dict)
    row_selected             = Signal(dict)
    columns_updated          = Signal(list)
    tables_updated           = Signal(list)
    thumbnail_ready          = Signal(str)
    row_updated              = Signal(str)
    operator_progress        = Signal(int)
    operator_complete        = Signal(str)
    merge_report_ready       = Signal(object)
    error_occurred           = Signal(str)
    display_result_ready     = Signal(dict)
    table_created            = Signal(str)

    def __init__(
        self,
        dataset,
        query_engine,
        artifact_store,
        registry,
        operator_registry,
    ):
        super().__init__()

        self._dataset          = dataset
        self._query            = query_engine
        self._store            = artifact_store
        self._registry         = registry
        self._op_registry      = operator_registry

        self._dataset.set_registry(registry)
        self._store.on_thumbnail_ready = self._on_thumbnail_ready

        self._thumbnail_queue:   list[str]   = []
        self._item_result_queue: list[tuple] = []
        self._complete_queue:    list[tuple] = []
        self._progress_queue:    list[int]   = []

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._drain_queues)
        self._timer.start()

        self._active_table:   str        = "frames"
        self._active_filters: list       = []
        self._sort_by:        str | None = None
        self._ascending:      bool       = True
        self._randomise:      bool       = False
        self._seed:           int | None = None
        self._group_by:       str | None = None
        self._visible_cols:   list[str]  = []

    # ── Queue draining (main thread) ──────────────────────────────────

    def _drain_queues(self) -> None:
        """Called on the main thread every 50ms. Drains all queues."""
        while self._thumbnail_queue:
            row_id = self._thumbnail_queue.pop(0)
            self.thumbnail_ready.emit(row_id)

        while self._item_result_queue:
            operation_id, table_name, row_id, result = self._item_result_queue.pop(0)
            self._dataset.update_row(row_id, result, table_name)
            self.row_updated.emit(row_id)

        while self._progress_queue:
            percent = self._progress_queue.pop(0)
            self.operator_progress.emit(percent)

        while self._complete_queue:
            operator_name, payload = self._complete_queue.pop(0)
            self._on_operator_complete(operator_name, payload)

    # ── Background thread callbacks ───────────────────────────────────

    def _on_thumbnail_ready(self, row_id: str) -> None:
        self._thumbnail_queue.append(row_id)

    def _on_item_complete(self, operation_id: str, table_name: str, row_id: str, result: dict) -> None:
        self._item_result_queue.append((operation_id, table_name, row_id, result))

    def _on_progress(self, percent: int) -> None:
        self._progress_queue.append(percent)

    def _on_create_columns_complete(self, operator_name: str) -> None:
        self._complete_queue.append(("create_columns", operator_name))

    def _on_create_table_complete(
        self,
        operator_name: str,
        result_df: pd.DataFrame,
    ) -> None:
        self._complete_queue.append(("create_table", (operator_name, result_df)))

    def _on_create_display_complete(
        self,
        operator_name: str,
        result: dict,
    ) -> None:
        self._complete_queue.append(("create_display", (operator_name, result)))

    def _on_operator_complete(self, mode: str, payload) -> None:
        """
        Called on the main thread when any operator finishes.
        Routes the result to the appropriate destination.
        """
        if mode == "create_columns":
            operator_name = payload
            self.operator_complete.emit(operator_name)
            self.columns_updated.emit(self._registry.list_all_columns())
            self._refresh_gallery()

        elif mode == "create_table":
            operator_name, result_df = payload
            table_name = f"{operator_name}_result"
            try:
                self._dataset.create_table_from_df(table_name, result_df)
                self.tables_updated.emit(self._dataset.list_tables())
                self.table_created.emit(table_name)
                self.operator_complete.emit(operator_name)
            except Exception as e:
                self.error_occurred.emit(
                    f"Failed to store table from '{operator_name}': {e}"
                )

        elif mode == "create_display":
            operator_name, result = payload
            result["operator_name"] = operator_name
            self.display_result_ready.emit(result)
            self.operator_complete.emit(operator_name)

    # ── Gallery refresh ───────────────────────────────────────────────

    def _refresh_gallery(self) -> None:
        """Re-runs the current query and emits the gallery signal."""
        try:
            df = self._dataset.get_table(self._active_table)

            if self._group_by:
                grouped = self._query.apply_grouped(
                    df,
                    group_by=self._group_by,
                    filters=self._active_filters,
                    sort_by=self._sort_by,
                    ascending=self._ascending,
                    randomise=self._randomise,
                    seed=self._seed,
                )
                self.grouped_gallery_updated.emit(grouped)
            else:
                row_ids = self._query.apply(
                    df,
                    filters=self._active_filters,
                    sort_by=self._sort_by,
                    ascending=self._ascending,
                    randomise=self._randomise,
                    seed=self._seed,
                )
                self.gallery_updated.emit(row_ids)

        except Exception as e:
            self.error_occurred.emit(f"Gallery refresh error: {e}")

    # ── Public API ────────────────────────────────────────────────────

    def load_folder(self, folder_path: Path) -> None:
        """
        Loads a folder of media files into the dataset and starts
        thumbnail generation for all items.

        Args:
            folder_path: Path to the folder containing media files.
        """
        try:
            self._store.reset()
            self._active_filters = []
            self._group_by       = None
            self._visible_cols   = []

            self._dataset.load_folder(folder_path)
            df = self._dataset.get_table("frames")

            for _, row in df.iterrows():
                self._store.request_thumbnail(
                    row["row_id"],
                    Path(row["full_path"]),
                )

            self.columns_updated.emit(self._registry.list_all_columns())
            self.tables_updated.emit(self._dataset.list_tables())
            self._refresh_gallery()

        except Exception as e:
            self.error_occurred.emit(f"Failed to load folder: {e}")

    def load_csv_as_primary(
        self,
        csv_path: Path,
        image_column: str | None = None,
    ) -> None:
        """
        Loads a CSV file as the primary data source without images.

        Args:
            csv_path:     Path to the CSV file.
            image_column: Optional column containing media file paths.
        """
        try:
            self._store.reset()
            self._active_filters = []
            self._group_by       = None
            self._visible_cols   = []

            self._dataset.load_csv_as_primary(csv_path, image_column)
            df = self._dataset.get_table("frames")

            for _, row in df.iterrows():
                full_path = row.get("full_path", "")
                if full_path and Path(full_path).exists():
                    self._store.request_thumbnail(
                        row["row_id"],
                        Path(full_path),
                    )

            self.columns_updated.emit(self._registry.list_all_columns())
            self.tables_updated.emit(self._dataset.list_tables())
            self._refresh_gallery()

        except Exception as e:
            self.error_occurred.emit(f"Failed to load CSV: {e}")

    def load_csv(
        self,
        csv_path: Path,
        join_on: str,
        preprocess: dict | None = None,
    ) -> None:
        """
        Starts the CSV merge workflow.

        Args:
            csv_path:   Path to the CSV file.
            join_on:    Column name in the CSV to join on.
            preprocess: Optional preprocessing rules.
        """
        try:
            report = self._dataset.merge_csv(csv_path, join_on, preprocess)
            self.merge_report_ready.emit(report)
        except Exception as e:
            self.error_occurred.emit(f"Failed to read CSV: {e}")

    def confirm_merge(self, report) -> None:
        """
        Commits a CSV merge after the researcher reviews the report.

        Args:
            report: The MergeReport returned by merge_csv().
        """
        try:
            self._dataset.confirm_merge(report)
            self.columns_updated.emit(self._registry.list_all_columns())
            self._refresh_gallery()
        except Exception as e:
            self.error_occurred.emit(f"Failed to confirm merge: {e}")

    def set_filters(
        self,
        filters: list,
        sort_by: str | None = None,
        ascending: bool = True,
        randomise: bool = False,
        seed: int | None = None,
    ) -> None:
        """
        Updates the current filter and sort state and refreshes the gallery.

        Args:
            filters:   List of Filter objects.
            sort_by:   Column to sort by, or None.
            ascending: Sort direction.
            randomise: If True, shuffle results.
            seed:      Random seed for reproducibility.
        """
        self._active_filters = filters or []
        self._sort_by        = sort_by
        self._ascending      = ascending
        self._randomise      = randomise
        self._seed           = seed
        self._refresh_gallery()

    def set_group_by(self, column_name: str | None) -> None:
        """
        Sets or clears the group-by column.

        Args:
            column_name: Column to group by, or None to clear.
        """
        self._group_by = column_name
        self._refresh_gallery()

    def set_visible_columns(self, column_names: list[str]) -> None:
        """
        Sets which columns the gallery displays in each tile.

        Args:
            column_names: Ordered list of column names to display.
        """
        self._visible_cols = column_names
        self._refresh_gallery()

    def get_visible_columns(self) -> list[str]:
        """Returns the currently selected visible columns."""
        return list(self._visible_cols)

    def select_row(self, row_id: str) -> None:
        """
        Retrieves full metadata for a row and emits row_selected.

        Args:
            row_id: The row the user clicked on.
        """
        try:
            metadata = self._dataset.get_row(row_id, self._active_table)
            self.row_selected.emit(metadata)
        except Exception as e:
            self.error_occurred.emit(f"Failed to select row: {e}")

    def run_create_columns(
        self,
        operator_name: str,
        row_ids: list[str],
    ) -> None:
        """
        Runs create_columns() on a list of rows in a background thread.

        Args:
            operator_name: Name of the operator to run.
            row_ids:       Rows to process.
        """
        try:
            operator = self._op_registry.get(operator_name)
            if operator is not None:
                for col_name, col_type in operator.output_columns:
                    try:
                        self._registry.register_by_tag(col_name, col_type)
                    except KeyError as e:
                        print(f"[Controller] Warning: {e}")
            operation_id = str(uuid.uuid4())
            table_name   = self._active_table
            work_items   = [
                {
                    "row_id":     row_id,
                    "table_name": table_name,
                    "row_data":   self._dataset.get_row(row_id, table_name),
                }
                for row_id in row_ids
            ]
            self._op_registry.run_create_columns(
                operator_name,
                work_items,
                operation_id=operation_id,
                on_item_complete=self._on_item_complete,
                on_progress=self._on_progress,
                on_complete=self._on_create_columns_complete,
            )
        except Exception as e:
            self.error_occurred.emit(
                f"Failed to start create_columns operator: {e}"
            )

    def run_create_table(
        self,
        operator_name: str,
        row_ids: list[str],
        group_by: str | list[str] | None = None,
    ) -> None:
        """
        Runs create_table() in a background thread.

        Args:
            operator_name: Name of the operator to run.
            row_ids:       Rows to include in the DataFrame.
            group_by:      Column or columns to group by.
        """
        try:
            df          = self._dataset.get_table(self._active_table)
            selected_df = df[df["row_id"].isin(row_ids)].copy()

            self._op_registry.run_create_table(
                operator_name,
                selected_df,
                group_by,
                on_complete=self._on_create_table_complete,
            )
        except Exception as e:
            self.error_occurred.emit(
                f"Failed to start create_table operator: {e}"
            )

    def run_create_display(
        self,
        operator_name: str,
        row_ids: list[str],
    ) -> None:
        """
        Runs create_display() in a background thread.

        Args:
            operator_name: Name of the operator to run.
            row_ids:       Rows to include in the DataFrame.
        """
        try:
            df          = self._dataset.get_table(self._active_table)
            selected_df = df[df["row_id"].isin(row_ids)].copy()

            self._op_registry.run_create_display(
                operator_name,
                selected_df,
                on_complete=self._on_create_display_complete,
            )
        except Exception as e:
            self.error_occurred.emit(
                f"Failed to start create_display operator: {e}"
            )

    def add_computed_column(
        self,
        name: str,
        expression: str,
        col_type: str = "numeric",
    ) -> None:
        """
        Adds a computed column to the active table.

        Args:
            name:       Name of the new column.
            expression: Pandas eval-compatible expression.
            col_type:   Column type tag.
        """
        try:
            self._dataset.add_computed_column(
                name, expression, col_type, self._active_table
            )
            self.columns_updated.emit(self._registry.list_all_columns())
            self._refresh_gallery()
        except Exception as e:
            self.error_occurred.emit(f"Failed to add column: {e}")

    def aggregate(
        self,
        name: str,
        group_by: str | list[str],
        aggregations: dict,
    ) -> None:
        """
        Creates a new aggregated table from the active table.

        Args:
            name:         Name for the new table.
            group_by:     Column or columns to group by.
            aggregations: Dict of column names to aggregation functions.
        """
        try:
            self._dataset.aggregate(
                name,
                source_table=self._active_table,
                group_by=group_by,
                aggregations=aggregations,
            )
            self.tables_updated.emit(self._dataset.list_tables())
        except Exception as e:
            self.error_occurred.emit(f"Failed to aggregate: {e}")

    def set_active_table(self, name: str) -> None:
        """
        Switches the active table.

        Args:
            name: Table name to activate.
        """
        try:
            self._dataset.get_table(name)
            self._active_table   = name
            self._active_filters = []
            self._group_by       = None
            self._visible_cols   = []
            self.columns_updated.emit(self._registry.list_all_columns())
            self._refresh_gallery()
        except KeyError as e:
            self.error_occurred.emit(f"Table not found: {e}")

    def save_filtered_as_table(self, name: str) -> None:
        """
        Creates a new permanent table from the currently visible rows.

        Args:
            name: Name for the new table.
        """
        try:
            visible_row_ids = self._query.apply(
                self._dataset.get_table(self._active_table),
                filters=self._active_filters,
                sort_by=self._sort_by,
                ascending=self._ascending,
            )
            self._dataset.create_table_from_rows(
                name,
                visible_row_ids,
                source_table=self._active_table,
            )
            self.tables_updated.emit(self._dataset.list_tables())
        except Exception as e:
            self.error_occurred.emit(f"Failed to save filtered set: {e}")

    def export_csv(
        self,
        path: Path,
        row_ids: list[str] | None = None,
    ) -> None:
        """
        Exports the active table (or a subset) to CSV.

        Args:
            path:    Destination file path.
            row_ids: If provided, export only these rows.
        """
        try:
            df = self._dataset.get_table(self._active_table)
            if row_ids is not None:
                df = df[df["row_id"].isin(row_ids)]
            df.to_csv(path, index=False)
        except Exception as e:
            self.error_occurred.emit(f"Failed to export CSV: {e}")

    def save_project(self, project_path: Path) -> None:
        """
        Saves the current project to disk.

        Args:
            project_path: Path to the project folder.
        """
        try:
            self._dataset.save(project_path)
            self._store.save_index(project_path)
        except Exception as e:
            self.error_occurred.emit(f"Failed to save project: {e}")

    def load_project(self, project_path: Path) -> None:
        """
        Loads a previously saved project from disk.

        Args:
            project_path: Path to an existing project folder.
        """
        try:
            self._dataset.load(project_path)
            self._store.load_index(project_path)
            self.tables_updated.emit(self._dataset.list_tables())
            self.columns_updated.emit(self._registry.list_all_columns())
            self._refresh_gallery()
        except Exception as e:
            self.error_occurred.emit(f"Failed to load project: {e}")

    # ── Convenience getters for the UI ────────────────────────────────

    def get_table_names(self) -> list[str]:
        """Returns all table names in the current project."""
        return self._dataset.list_tables()

    def get_column_names(self) -> list[str]:
        """Returns all registered column names."""
        return self._registry.list_all_columns()

    def get_visual_column_names(self) -> list[str]:
        """Returns column names that produce visual output in tiles."""
        return self._registry.list_visual_columns()

    def get_group_values(self, column: str) -> list:
        """
        Returns sorted unique values in a column of the active table.

        Args:
            column: Column name to inspect.

        Returns:
            Sorted list of unique values.
        """
        try:
            df = self._dataset.get_table(self._active_table)
            return self._query.get_group_values(df, column)
        except Exception:
            return []

    def get_artifact_pixmap(self, row_id: str, artifact_type: str):
        """
        Returns a PIL Image for the given artifact, or None.

        Args:
            row_id:        The item whose artifact to retrieve.
            artifact_type: 'thumbnail' or 'preview'.
        """
        return self._store.get_pixmap(row_id, artifact_type)

    def get_row(self, row_id: str, table_name: str = "frames") -> dict:
        """
        Returns all column values for one row as a plain dictionary.

        Args:
            row_id:     The row to retrieve.
            table_name: The table containing the row (default: 'frames').

        Returns:
            Dict of column name to value. Empty dict if not found.
        """
        return self._dataset.get_row(row_id, table_name)

    def render_column_value(
        self,
        column_name: str,
        value,
        size: int,
        mode: str = "thumbnail",
        context: dict | None = None,
    ):
        """
        Renders a column value using ColumnTypeRegistry.

        In 'thumbnail' mode, returns a QPixmap for display in a gallery
        tile. In 'detail' mode, returns a QWidget for display in
        DetailWidget.

        The UI always calls this method rather than importing from
        column_types directly — this keeps the component boundary clean.

        Args:
            column_name: The column to render.
            value:       The cell value.
            size:        Target size in pixels.
            mode:        'thumbnail' (default) or 'detail'.
            context:     Optional dict with row-level metadata, e.g.
                         {'row_id': ..., 'column_name': ...}.

        Returns:
            A QPixmap (thumbnail mode), QWidget (detail mode), or None.
        """
        return self._registry.render(column_name, value, size, mode, context)
