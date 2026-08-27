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
    Each worker-bound callback only puts a result onto a queue (or, for
    progress, overwrites a single latest value under a lock). The
    QTimer-driven drain on the main thread is the only place those
    results are turned into Dataset writes and Qt signals, and it
    processes a bounded number of items per tick.

This file is written centrally (not by a student).
"""

from __future__ import annotations
from pathlib import Path
import queue
import threading
import uuid
import pandas as pd

from PySide6.QtCore import QObject, Signal, QTimer

from models.query_result import QueryResult, ResultLayout, GroupSection
from models.notifications import RowsUpdated, ThumbnailsReady

DEFAULT_MEDIA_COLUMN_NAME = "full_path"

class AppController(QObject):
    """
    Wires together Dataset, QueryEngine, ArtifactStore,
    ColumnTypeRegistry, and OperatorRegistry in response to UI events.

    All signals are emitted on the main thread. All Dataset mutations
    happen on the main thread.

    Signals:
        result_changed:          ResultLayout describing the new ordered
                                 query result: how many rows, and where
                                 the group boundaries fall. Carries no
                                 row ids -- the UI fetches those from the
                                 controller as it paints.
        row_selected:            Metadata dict for the selected row.
        columns_updated:         List of all registered column names.
        tables_updated:          List of all table names in the project.
        thumbnails_ready:        ThumbnailsReady payload -- a table name
                                 and the tuple of row_ids whose
                                 thumbnail is now available.
        rows_updated:            RowsUpdated payload -- a table name and
                                 the tuple of row_ids whose data changed
                                 in this drain tick.
        operator_progress:       Integer 0-100 progress percentage.
        operator_complete:       Name of the operator that finished.
        merge_report_ready:      MergeReport object for display.
        error_occurred:          Human-readable error message string.
        display_result_ready:    Result dict from a create_display
                                 operator, for ResultsPanel.
        table_created:           Name of a newly created table.
    """

    result_changed           = Signal(object)
    row_selected             = Signal(dict)
    columns_updated          = Signal(list)
    tables_updated           = Signal(list)
    active_table_changed     = Signal(str)
    thumbnails_ready         = Signal(object)
    rows_updated             = Signal(object)
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
        drain_budget: int = 200,
    ):
        super().__init__()

        self._dataset          = dataset
        self._query            = query_engine
        self._store            = artifact_store
        self._registry         = registry
        self._op_registry      = operator_registry

        self._dataset.set_registry(registry)
        self._store.on_thumbnail_ready = self._on_thumbnail_ready

        # Result queues. Each is drained by at most self._drain_budget
        # items per timer tick (see _drain_queues), so a large operator
        # run cannot empty thousands of results into one main-thread
        # tick. Worker threads only ever put onto these; nothing reads
        # controller state from a worker.
        #
        # queue.SimpleQueue is used rather than a list so there is no
        # list.pop(0) (linear in queue length) anywhere on the path.
        self._thumbnail_queue:   queue.SimpleQueue = queue.SimpleQueue()
        self._item_result_queue: queue.SimpleQueue = queue.SimpleQueue()
        # Completions, setup errors, per-row-error summaries and
        # create_table/display errors all share one queue so they are
        # processed in the order they happened, on the main thread.
        self._complete_queue:    queue.SimpleQueue = queue.SimpleQueue()

        # Progress is not a queue. A run can emit hundreds of progress
        # values between two ticks and only the last one matters, so the
        # worker overwrites a single latest value under a lock and the
        # drain emits operator_progress at most once per tick. This
        # coalesces at the source and is bounded by construction.
        self._progress_lock = threading.Lock()
        self._latest_progress: int | None = None

        # How many items to take from each queue per tick. The same
        # budget is applied to each queue independently. A constructor
        # parameter rather than a module constant so a test can drive a
        # small deterministic budget and a future setting can raise it;
        # CLAUDE.md's "no machine-dependent constant" rule is
        # [TARGET -> P0.5], and this at least does not add a new
        # violation.
        self._drain_budget: int = drain_budget

        # Live operator runs, keyed by operation_id. Each value carries
        # what the completion path needs on the main thread: the
        # operator's display label, the table it targets, the count of
        # per-row results already applied, and the list of row_ids
        # apply_row_updates() could not place.
        #
        #   - Registered in run_create_columns / run_create_table /
        #     run_create_display when the run starts.
        #   - Deregistered when its completion or error is processed by
        #     the drain -- for create_columns, only once every per-row
        #     result it emitted has been applied.
        #   - The whole dict is cleared by load_folder(),
        #     load_csv_as_primary() and load_project() -- and by nothing
        #     else. In particular NOT by set_active_table(): a result
        #     carries its own table_name and belongs in that table
        #     whatever is on screen.
        #
        # Supersession is deliberately not tracked here. Two runs over
        # the same rows may overlap and last-write-wins. Detecting "this
        # value was superseded by a newer run" needs per-row per-column
        # versioning, which is over-engineering; the place to prevent it
        # is refusing to start the second run, which is P1.12's
        # cancellation work. Removing an entry from this dict is exactly
        # the shape a future cancellation will use.
        self._live_runs: dict[str, dict] = {}

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._drain_queues)
        self._timer.start()

        # The ordered query result the controller owns (P0.4). The
        # gallery holds no row ids of its own -- it is given an index
        # range into this order and fetches the ids it needs to paint.
        #   _result          -- the current QueryResult, or None before
        #                       the first query.
        #   _result_index    -- row_id -> position in _result.row_ids,
        #                       rebuilt whenever _result is rebuilt.
        #   _displayed_ranges-- viewport_key -> (start, stop) half-open
        #                       absolute index range that a gallery says
        #                       it is currently showing. P0.5 reads this
        #                       to prioritise and cancel renders.
        self._result:           QueryResult | None      = None
        self._result_index:     dict[str, int]          = {}
        self._displayed_ranges: dict[str, tuple[int, int]] = {}

        self._active_table:   str        = "frames"
        self._active_filters: list       = []
        self._sort_by:        str | None = None
        self._ascending:      bool       = True
        self._randomise:      bool       = False
        self._seed:           int | None = None
        self._group_by:       str | None = None
        # None means the researcher has not made a choice yet; an empty
        # list means the researcher explicitly unchecked every column.
        self._visible_cols:   list[str] | None = None

    # ── Run registry ─────────────────────────────────────────────────

    def _register_run(
        self,
        operation_id: str,
        label: str,
        table_name: str,
    ) -> None:
        """Records a started operator run as live. See _live_runs."""
        self._live_runs[operation_id] = {
            "label":       label,
            "table_name":  table_name,
            "applied":     0,
            "unplaceable": [],
        }

    def _deregister_run(self, operation_id: str) -> None:
        """Drops a run from the live set. Idempotent."""
        self._live_runs.pop(operation_id, None)

    # ── Queue draining (main thread) ──────────────────────────────────

    def _drain_queues(self) -> None:
        """
        Called on the main thread every 50ms. Orchestrates the bounded
        drain of each result queue and the single coalesced progress
        emission. Each helper takes at most self._drain_budget items
        from its own queue, so a large run is spread over several ticks
        instead of stalling one.
        """
        self._drain_thumbnails()
        self._drain_item_results()
        self._emit_progress_if_changed()
        self._drain_completions()

    def _drain_thumbnails(self) -> None:
        """
        Emits one ThumbnailsReady per table for up to _drain_budget
        queued thumbnails, so the gallery makes one pass over its
        mounted tiles per table rather than one per thumbnail.
        """
        by_table: dict[str, list[str]] = {}
        for _ in range(self._drain_budget):
            try:
                table_name, row_id = self._thumbnail_queue.get_nowait()
            except queue.Empty:
                break
            by_table.setdefault(table_name, []).append(row_id)

        for table_name, row_ids in by_table.items():
            self.thumbnails_ready.emit(
                ThumbnailsReady(table_name=table_name, row_ids=tuple(row_ids))
            )

    def _drain_item_results(self) -> None:
        """
        Applies up to _drain_budget per-row operator results. Results
        from a run that is no longer live are dropped silently -- the
        rows they targeted no longer mean anything after a project
        reload, and a per-row drop is not worth a dialog.

        A tick's surviving results are grouped by table and applied with
        one Dataset.apply_row_updates() call per table, followed by one
        rows_updated emission per table. Row ids that could not be
        placed are attributed back to the run that produced them, for
        the completion path to report.
        """
        batches: dict[str, dict[str, dict]] = {}
        row_owner: dict[tuple[str, str], str] = {}

        for _ in range(self._drain_budget):
            try:
                operation_id, table_name, row_id, result = \
                    self._item_result_queue.get_nowait()
            except queue.Empty:
                break
            run = self._live_runs.get(operation_id)
            if run is None:
                # Result from a dead run -- drop silently.
                continue
            # Count every live result taken off the queue, placed or
            # not. The create_columns completion is held back until this
            # reaches the number of results the worker said it emitted,
            # so a fast concurrent run cannot make this run's completion
            # wait on the shared queue as a whole.
            run["applied"] += 1
            batches.setdefault(table_name, {})[row_id] = result
            row_owner[(table_name, row_id)] = operation_id

        for table_name, updates in batches.items():
            unplaceable = self._dataset.apply_row_updates(table_name, updates)
            unplaceable_set = set(unplaceable)
            for row_id in unplaceable_set:
                operation_id = row_owner.get((table_name, row_id))
                run = self._live_runs.get(operation_id)
                if run is not None:
                    run["unplaceable"].append(row_id)

            placed = tuple(
                rid for rid in updates if rid not in unplaceable_set
            )
            if placed:
                self.rows_updated.emit(
                    RowsUpdated(table_name=table_name, row_ids=placed)
                )

    def _emit_progress_if_changed(self) -> None:
        """
        Emits operator_progress once if a worker pushed a new progress
        value since the last tick. The value is coalesced at the source
        (the worker just overwrites it), so this is bounded by
        construction and needs no budget.
        """
        with self._progress_lock:
            percent = self._latest_progress
            self._latest_progress = None
        if percent is not None:
            self.operator_progress.emit(percent)

    def _drain_completions(self) -> None:
        """
        Processes up to _drain_budget completion-queue items.

        A create_columns completion is deferred while its run still has
        per-row results working through the bounded item drain
        (run["applied"] < the count the worker emitted). Deregistering
        the run before its last results land would make those results
        look like they came from a dead run and be dropped. The wait is
        on this run's own outstanding count, not on the shared queue
        being empty, so a second fast run cannot starve it.
        """
        deferred: list[tuple] = []
        for _ in range(self._drain_budget):
            try:
                mode, payload = self._complete_queue.get_nowait()
            except queue.Empty:
                break
            if mode == "create_columns":
                operation_id, _operator_name, emitted = payload
                run = self._live_runs.get(operation_id)
                if run is not None and run["applied"] < emitted:
                    deferred.append((mode, payload))
                    continue
            self._on_operator_complete(mode, payload)

        for item in deferred:
            self._complete_queue.put(item)

    # ── Background thread callbacks ───────────────────────────────────
    # Each of these runs on a worker thread and does nothing but put a
    # result onto a queue (or overwrite the latest progress value). None
    # reads controller or component state.

    def _on_thumbnail_ready(self, table_name: str, row_id: str) -> None:
        self._thumbnail_queue.put((table_name, row_id))

    def _on_item_complete(
        self,
        operation_id: str,
        table_name: str,
        row_id: str,
        result: dict,
    ) -> None:
        self._item_result_queue.put((operation_id, table_name, row_id, result))

    def _on_progress(self, percent: int) -> None:
        with self._progress_lock:
            self._latest_progress = percent

    def _on_create_columns_complete(
        self,
        operation_id: str,
        operator_name: str,
        emitted: int,
    ) -> None:
        # `emitted` is how many per-row results the worker handed to
        # on_item_complete. The drain holds this completion back until it
        # has applied that many for this run.
        self._complete_queue.put(
            ("create_columns", (operation_id, operator_name, emitted))
        )

    def _on_operator_setup_error(
        self,
        operation_id: str,
        label: str,
        message: str,
    ) -> None:
        # The label is computed by the operator itself
        # (BaseOperator.display_label) and passed in, so this callback
        # reads no component state from the worker thread.
        self._complete_queue.put(
            ("setup_error", (operation_id, label, message))
        )

    def _on_operator_row_errors(
        self,
        operation_id: str,
        label: str,
        errors: list[tuple[str, str, str]],
    ) -> None:
        # As above: the label travels with the error, so nothing is
        # looked up from a worker thread.
        self._complete_queue.put(
            ("row_errors", (operation_id, label, errors))
        )

    def _on_create_table_complete(
        self,
        operation_id: str,
        operator_name: str,
        result_df: pd.DataFrame,
    ) -> None:
        self._complete_queue.put(
            ("create_table", (operation_id, operator_name, result_df))
        )

    def _on_create_display_complete(
        self,
        operation_id: str,
        operator_name: str,
        result: dict,
    ) -> None:
        self._complete_queue.put(
            ("create_display", (operation_id, operator_name, result))
        )

    def _on_operator_error(
        self,
        operation_id: str,
        operator_name: str,
        message: str,
    ) -> None:
        """Background-thread callback for operator errors. Marshals
        the error onto the main thread via _complete_queue."""
        self._complete_queue.put(
            ("error", (operation_id, operator_name, message))
        )

    def _on_operator_complete(self, mode: str, payload) -> None:
        """
        Called on the main thread from the completion-queue drain.
        Routes the result to the appropriate destination and, for the
        modes that own a live run, deregisters it.
        """
        if mode == "create_columns":
            operation_id, operator_name, _emitted = payload
            run = self._live_runs.get(operation_id)
            if run is None:
                # The project was replaced while this run was in flight.
                # Its per-row results were already dropped silently by
                # the item drain (a per-row drop after a reload is not
                # worth a dialog). Still emit operator_complete -- like
                # every other mode does for a dead run -- so a progress
                # or "running" UI clears; just skip the columns refresh,
                # which would be work for a run that changed nothing.
                self.operator_complete.emit(operator_name)
                return
            if run["unplaceable"]:
                n = len(run["unplaceable"])
                self.error_occurred.emit(
                    f'"{run["label"]}" finished, but {n} result row(s) '
                    f'could not be matched to a row in table '
                    f'"{run["table_name"]}" and were discarded.'
                )
            self._deregister_run(operation_id)
            self.operator_complete.emit(operator_name)
            self.columns_updated.emit(self._registry.list_all_columns())
            self._refresh_result()

        elif mode == "setup_error":
            # A run that aborted before finishing. Rows processed before
            # the abort still produced results that are queued behind
            # this message; the create_columns completion that follows
            # owns deregistration once they have landed, so this branch
            # only surfaces the message.
            _operation_id, label, message = payload
            self.error_occurred.emit(
                f'Cannot run operator "{label}"\n\n{message}'
            )

        elif mode == "row_errors":
            # An additional end-of-run summary. Deregistration and the
            # unplaceable report belong to the create_columns completion
            # that still follows, so this branch does not deregister.
            _operation_id, label, errors = payload
            counts: dict[str, int] = {}
            first_msg: dict[str, str] = {}
            for _row_id, exc_type, msg in errors:
                counts[exc_type] = counts.get(exc_type, 0) + 1
                first_msg.setdefault(exc_type, msg)
            lines = [
                f'  - {t} (x{counts[t]}) - "{first_msg[t]}"'
                for t in sorted(counts, key=lambda k: -counts[k])
            ]
            self.error_occurred.emit(
                f'"{label}" finished, but {len(errors)} row(s) hit '
                f"unexpected errors.\n\nError types seen:\n"
                + "\n".join(lines)
                + "\n\nThe affected rows have no values for the new columns."
            )

        elif mode == "create_table":
            operation_id, operator_name, result_df = payload
            was_live = operation_id in self._live_runs
            self._deregister_run(operation_id)
            if not was_live:
                # Minutes of compute that arrived after the project
                # changed. Do not store it; say so rather than dropping
                # it silently.
                self.error_occurred.emit(
                    f'The "{operator_name}" table result arrived after the '
                    f"project changed and was discarded."
                )
                return
            table_name = f"{operator_name}_result"
            try:
                self._dataset.create_table_from_df(table_name, result_df)
                stored_df = self._dataset.get_table(table_name)
                for _, row in stored_df.iterrows():
                    full_path = row.get("full_path", "")
                    if full_path and Path(str(full_path)).exists():
                        self._store.request_thumbnail(
                            row["row_id"],
                            Path(str(full_path)),
                            table_name,
                        )
                self.tables_updated.emit(self._dataset.list_tables())
                self.table_created.emit(table_name)
                self.operator_complete.emit(operator_name)
            except Exception as e:
                self.error_occurred.emit(
                    f"Failed to store table from '{operator_name}': {e}"
                )

        elif mode == "create_display":
            operation_id, operator_name, result = payload
            was_live = operation_id in self._live_runs
            self._deregister_run(operation_id)
            if not was_live:
                self.error_occurred.emit(
                    f'The "{operator_name}" result arrived after the '
                    f"project changed and was discarded."
                )
                return
            result["operator_name"] = operator_name
            self.display_result_ready.emit(result)
            self.operator_complete.emit(operator_name)

        elif mode == "error":
            operation_id, operator_name, message = payload
            self._deregister_run(operation_id)
            self.error_occurred.emit(message)
            self.operator_complete.emit(operator_name)

    # ── Result refresh ───────────────────────────────────────────────

    def _refresh_result(self) -> None:
        """
        Re-runs the current query, stores the result as the single
        source of truth for row order, and emits result_changed with a
        row-id-free ResultLayout.

        The query itself is unchanged from before P0.4 -- same
        QueryEngine call, same arguments. What changed is that the
        result is now kept (as a QueryResult) rather than emitted and
        thrown away, and grouped mode builds one flat order plus group
        boundaries instead of a dict the UI has to flatten itself.
        """
        try:
            # Read-only: QueryEngine never mutates what it is handed
            # ([NOW] rule), so this does not need Dataset's own copy.
            df = self._dataset.read_only_view(self._active_table)

            if self._group_by:
                # Grouped mode: run the grouped query, then build the
                # flat order by concatenating the groups in the order
                # apply_grouped() returns them, recording each group's
                # [start, stop) span as we go. We never re-run apply()
                # to get the flat order -- that would be two
                # computations of the same thing with different
                # arguments, the exact defect P0.4 removes.
                grouped = self._query.apply_grouped(
                    df,
                    group_by=self._group_by,
                    filters=self._active_filters,
                    sort_by=self._sort_by,
                    ascending=self._ascending,
                    randomise=self._randomise,
                    seed=self._seed,
                )
                flat_order: list[str] = []
                sections: list[GroupSection] = []
                for label, ids in grouped.items():
                    start = len(flat_order)
                    flat_order.extend(ids)
                    stop = len(flat_order)
                    sections.append(GroupSection(label=str(label), start=start, stop=stop))
                row_ids = flat_order
                groups: tuple[GroupSection, ...] | None = tuple(sections)
            else:
                # Flat mode: the query result is already the flat order.
                row_ids = self._query.apply(
                    df,
                    filters=self._active_filters,
                    sort_by=self._sort_by,
                    ascending=self._ascending,
                    randomise=self._randomise,
                    seed=self._seed,
                )
                groups = None

            # Store the new result with a fresh id, rebuild the
            # row_id -> index lookup, and drop any displayed ranges --
            # they point into an order that no longer exists.
            result_id = str(uuid.uuid4())
            self._result = QueryResult(
                result_id=result_id,
                table_name=self._active_table,
                row_ids=tuple(row_ids),
                groups=groups,
            )
            self._result_index = {rid: i for i, rid in enumerate(row_ids)}
            self._displayed_ranges = {}

            self.result_changed.emit(self._result.layout())

        except Exception as e:
            # The query failed. Do not leave the previous result
            # standing: after a failed set_active_table() its
            # table_name would disagree with _active_table, and a
            # "Visible" operator run would then feed old-table row ids
            # against the new table. Publish an empty result for the
            # current table instead, so the gallery clears and the
            # error shows next to it.
            self._result = QueryResult(
                result_id=str(uuid.uuid4()),
                table_name=self._active_table,
                row_ids=(),
                groups=None,
            )
            self._result_index = {}
            self._displayed_ranges = {}
            self.result_changed.emit(self._result.layout())
            self.error_occurred.emit(f"Could not compute the visible rows: {e}")

    # ── Ordered-result accessors (the P0.4 seam) ─────────────────────

    def get_result_layout(self) -> ResultLayout:
        """
        Returns the current ResultLayout. Before the first query this is
        an empty flat layout so the UI always has something to lay out.
        """
        if self._result is None:
            return ResultLayout(
                result_id="",
                table_name=self._active_table,
                total=0,
                groups=None,
            )
        return self._result.layout()

    def get_visible_row_ids(self) -> list[str]:
        """
        Returns the whole flat order -- every row that matches the
        current filters, in display order. In grouped mode this is the
        groups concatenated in their on-screen sequence.
        """
        if self._result is None:
            return []
        return list(self._result.row_ids)

    def get_row_ids_in_range(self, start: int, stop: int) -> list[str]:
        """
        Returns the row ids at flat-order positions [start, stop).

        Out-of-range indices are clamped rather than raising: a gallery
        can legitimately ask for a range that a just-arrived smaller
        result no longer covers.
        """
        if self._result is None:
            return []
        n = len(self._result.row_ids)
        lo = max(0, min(start, n))
        hi = max(lo, min(stop, n))
        return list(self._result.row_ids[lo:hi])

    def get_result_index(self, row_id: str) -> int | None:
        """Returns row_id's position in the flat order, or None."""
        return self._result_index.get(row_id)

    def order_by_result(self, row_ids: list[str]) -> list[str]:
        """
        Puts an arbitrary collection of row ids into the current
        result's order. Ids not in the result are silently dropped.

        Sorts the given ids by their flat-order position rather than
        scanning the whole flat order, so this stays cheap on a
        530k-row result -- it is called once per gallery on every
        selection change.
        """
        if self._result is None:
            return []
        placed = [
            (self._result_index[rid], rid)
            for rid in row_ids
            if rid in self._result_index
        ]
        placed.sort(key=lambda pair: pair[0])
        return [rid for _, rid in placed]

    def report_displayed_range(
        self,
        viewport_key: str,
        start: int,
        stop: int,
        result_id: str,
    ) -> None:
        """
        Records that the gallery keyed by viewport_key is currently
        showing flat-order positions [start, stop).

        A call whose result_id does not match the current result is
        ignored -- it describes an order that no longer exists. This is
        the same staleness discipline P0.2b applies to operator results.
        """
        if self._result is None or result_id != self._result.result_id:
            return
        self._displayed_ranges[viewport_key] = (start, stop)

    def clear_displayed_range(self, viewport_key: str) -> None:
        """Forgets the displayed range for viewport_key, if any."""
        self._displayed_ranges.pop(viewport_key, None)

    def get_displayed_ranges(self) -> list[tuple[int, int]]:
        """
        Returns every reported displayed range, sorted by start.

        P0.4 only makes this information available. Prefetch margins,
        priorities and cancellation are P0.5's work and are not built
        here.
        """
        return sorted(self._displayed_ranges.values(), key=lambda r: r[0])

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
            # The dataset is being replaced: every in-flight operator run
            # now targets rows that will not exist. Drop them so a late
            # result cannot write onto the new folder's rows.
            self._live_runs.clear()
            self._active_filters = []
            self._group_by       = None
            self._visible_cols   = None

            self._dataset.load_folder(folder_path)
            df = self._dataset.get_table("frames")

            for _, row in df.iterrows():
                self._store.request_thumbnail(
                    row["row_id"],
                    Path(row["full_path"]),
                    "frames",
                )

            self.columns_updated.emit(self._registry.list_all_columns())
            self.tables_updated.emit(self._dataset.list_tables())
            self._refresh_result()

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
            # See load_folder(): the dataset is being replaced, so no
            # in-flight run's results are valid any more.
            self._live_runs.clear()
            self._active_filters = []
            self._group_by       = None
            self._visible_cols   = None

            self._dataset.load_csv_as_primary(csv_path, image_column)
            df = self._dataset.get_table("frames")

            for _, row in df.iterrows():
                full_path = row.get("full_path", "")
                if full_path and Path(full_path).exists():
                    self._store.request_thumbnail(
                        row["row_id"],
                        Path(full_path),
                        "frames",
                    )

            self.columns_updated.emit(self._registry.list_all_columns())
            self.tables_updated.emit(self._dataset.list_tables())
            self._refresh_result()

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
            self._refresh_result()
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
        self._refresh_result()

    def set_group_by(self, column_name: str | None) -> None:
        """
        Sets or clears the group-by column.

        Args:
            column_name: Column to group by, or None to clear.
        """
        self._group_by = column_name
        self._refresh_result()

    def set_visible_columns(self, column_names: list[str]) -> None:
        """
        Sets which columns the gallery displays in each tile. An empty
        list records that the researcher explicitly unchecked every
        column; pass through clear_visible_columns_preference() instead
        to return to the unset/default state.

        Args:
            column_names: Ordered list of column names to display.
        """
        self._visible_cols = column_names
        self._refresh_result()

    def clear_visible_columns_preference(self) -> None:
        """Clears the visible-column preference back to its unset state."""
        self._visible_cols = None
        self._refresh_result()

    def get_effective_visible_columns(self) -> list[str]:
        """
        Returns the columns that should actually be displayed right now --
        the single source of truth for both the gallery and the Columns
        checkbox, so they can never show two different states.

        Returns:
            - The stored preference, if one has been explicitly set
            (including an explicit empty list).
            - Otherwise, [DEFAULT_MEDIA_COLUMN_NAME] if that column exists
            and is visual in the current table.
            - Otherwise, an empty list.
        """
        if self.has_visible_columns_preference():
            return list(self._visible_cols)
        if DEFAULT_MEDIA_COLUMN_NAME in self._registry.list_visual_columns():
            return [DEFAULT_MEDIA_COLUMN_NAME]
        return []

    def has_visible_columns_preference(self) -> bool:
        """
        True iff the researcher has explicitly set the visible-column
        list (including unchecking everything). False before any choice
        has been made or after a project reset.
        """
        return self._visible_cols is not None

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
            if operator is None or operator.create_columns_label is None:
                self.error_occurred.emit(
                    f'Operator "{operator_name}" cannot add columns.'
                )
                return
            for col_name, col_type in operator.output_columns:
                try:
                    self._registry.register_by_tag(col_name, col_type)
                except KeyError as e:
                    print(f"[Controller] Warning: {e}")
            operation_id = str(uuid.uuid4())
            table_name   = self._active_table
            # One snapshot of exactly the selected rows, taken once here
            # on the main thread -- not one Dataset.get_row() call (and
            # one full-table copy) per row.
            snapshot = self._dataset.snapshot_rows(table_name, row_ids)
            self._register_run(operation_id, operator.display_label, table_name)
            try:
                started = self._op_registry.run_create_columns(
                    operator_name,
                    snapshot,
                    row_ids,
                    table_name,
                    operation_id=operation_id,
                    on_item_complete=self._on_item_complete,
                    on_progress=self._on_progress,
                    on_complete=self._on_create_columns_complete,
                    on_setup_error=self._on_operator_setup_error,
                    on_row_errors=self._on_operator_row_errors,
                )
            except Exception:
                # The run never started, so no callback will ever
                # deregister it. Drop the entry here.
                self._deregister_run(operation_id)
                raise
            if not started:
                # Registry declined the run (unknown operator / wrong
                # mode) without raising and without a callback. Same
                # cleanup.
                self._deregister_run(operation_id)
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
            operator = self._op_registry.get(operator_name)
            if operator is None or operator.create_table_label is None:
                self.error_occurred.emit(
                    f'Operator "{operator_name}" cannot create a table.'
                )
                return
            operation_id = str(uuid.uuid4())
            table_name   = self._active_table
            selected_df  = self._dataset.snapshot_rows(table_name, row_ids)
            self._register_run(operation_id, operator.display_label, table_name)
            try:
                started = self._op_registry.run_create_table(
                    operator_name,
                    selected_df,
                    group_by,
                    operation_id=operation_id,
                    on_complete=self._on_create_table_complete,
                    on_error=self._on_operator_error,
                )
            except Exception:
                self._deregister_run(operation_id)
                raise
            if not started:
                self._deregister_run(operation_id)
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
            operator = self._op_registry.get(operator_name)
            if operator is None or operator.create_display_label is None:
                self.error_occurred.emit(
                    f'Operator "{operator_name}" cannot show a result.'
                )
                return
            operation_id = str(uuid.uuid4())
            table_name   = self._active_table
            selected_df  = self._dataset.snapshot_rows(table_name, row_ids)
            self._register_run(operation_id, operator.display_label, table_name)
            try:
                started = self._op_registry.run_create_display(
                    operator_name,
                    selected_df,
                    operation_id=operation_id,
                    on_complete=self._on_create_display_complete,
                    on_error=self._on_operator_error,
                )
            except Exception:
                self._deregister_run(operation_id)
                raise
            if not started:
                self._deregister_run(operation_id)
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
            self._refresh_result()
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

        Deliberately does NOT touch self._live_runs: an in-flight run
        carries its own table_name and its results belong in that table
        whatever is on screen. Staleness is keyed on run liveness, never
        on the active table.

        Args:
            name: Table name to activate.
        """
        try:
            self._dataset.get_table(name)
            self._active_table   = name
            self._active_filters = []
            self._group_by       = None
            self._visible_cols   = None
            self.active_table_changed.emit(name)
            self.columns_updated.emit(self._registry.list_all_columns())
            self._refresh_result()
        except KeyError as e:
            self.error_occurred.emit(f"Table not found: {e}")

    def save_filtered_as_table(self, name: str) -> None:
        """
        Creates a new permanent table from the currently visible rows,
        in exactly the order they are on screen.

        Before P0.4 this re-ran QueryEngine.apply() with the filters and
        sort but without randomise and seed, so with randomise on it
        saved a different order from the one displayed, and it copied the
        whole table to do it. Now it uses the flat order the controller
        already owns -- no second query, no full-table copy.

        Args:
            name: Name for the new table.
        """
        if self._result is None:
            self.error_occurred.emit(
                "Failed to save filtered set: there is no active result yet."
            )
            return
        try:
            self._dataset.create_table_from_rows(
                name,
                list(self._result.row_ids),
                source_table=self._result.table_name,
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
            df = self._dataset.snapshot_rows(self._active_table, row_ids)
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
            # The dataset is being replaced: drop every in-flight run so
            # a late result cannot land in the newly loaded project.
            self._live_runs.clear()
            self._dataset.load(project_path)
            self._store.load_index(project_path)
            self.tables_updated.emit(self._dataset.list_tables())
            self.columns_updated.emit(self._registry.list_all_columns())
            self._refresh_result()
        except Exception as e:
            self.error_occurred.emit(f"Failed to load project: {e}")

    # ── Convenience getters for the UI ────────────────────────────────

    def get_table_names(self) -> list[str]:
        """Returns all table names in the current project."""
        return self._dataset.list_tables()

    def get_active_table(self) -> str:
        """
        Returns the name of the currently active table.

        MainWindow uses this to decide whether a rows_updated /
        thumbnails_ready notification is for the table on screen. That
        "is it visible?" check lives in MainWindow, once -- not in the
        controller and not in each gallery.
        """
        return self._active_table

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
            df = self._dataset.read_only_view(self._active_table)
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

    def get_row(self, row_id: str, table_name: str | None = None) -> dict:
        """
        Returns all column values for one row as a plain dictionary.

        Args:
            row_id:     The row to retrieve.
            table_name: The table containing the row. Defaults to the
                        currently active table.

        Returns:
            Dict of column name to value. Empty dict if not found.
        """
        if table_name is None:
            table_name = self._active_table
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

    def get_column_type(self, column_name: str):
        """
        Returns the column-type object registered for *column_name*.

        Gives the UI read access to column type metadata without reaching
        into _registry directly.

        Args:
            column_name: The column whose type to look up.

        Returns:
            The registered column-type object, or None if not found.
        """
        return self._registry.get(column_name)

    def get_all_row_ids(self, table_name: str | None = None) -> list[str]:
        """
        Returns every row_id in the active table (or *table_name* if given).

        Provides a clean accessor so the UI does not need to reach into
        _dataset directly.

        Args:
            table_name: Table to read from.  Defaults to the active table.

        Returns:
            Ordered list of row_id strings.
        """
        name = table_name if table_name is not None else self._active_table
        return list(self._dataset.read_only_view(name)["row_id"])

    def get_operator(self, operator_name: str):
        """
        Returns the operator registered under *operator_name*.

        Gives the UI access to operator objects without reaching into
        _op_registry directly.

        Args:
            operator_name: Name of the operator to retrieve.

        Returns:
            The operator object, or None if not found.
        """
        return self._op_registry.get(operator_name)
