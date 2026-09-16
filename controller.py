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
from models.project_paths import ProjectPaths, build_project_paths
from models.output_copy import OutputCopyPlan
from models.output_copy import execute_output_copy as _execute_output_copy
from models.output_copy import plan_output_copy as _plan_output_copy
from media.media_address import from_path as _media_address_from_path
from media.media_address import resolve_source, MediaAddressError
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    MediaRequirement,
    ModelLifecycle,
)
from operators.run_context import (
    CancellationToken,
    OperatorRun,
    OperatorRunError,
    OperatorRunSpec,
    RunData,
    TableSnapshot,
)

DEFAULT_MEDIA_COLUMN_NAME = "full_path"


# ---------------------------------------------------------------------------
# P1.12f-2: the pre-emptive write/read conflict check.
#
# Pure, Qt-free arithmetic plus the researcher-facing sentences it produces
# -- no widget, no dialog, importable and testable on its own. AppController
# assembles the plain-data arguments from self._live_runs and calls this;
# ui/main_window.py calls the ONE public AppController method that wraps it
# and is the only place a dialog is raised.
# ---------------------------------------------------------------------------

def write_read_conflict_warnings(
    live_runs: list[dict],
    new_run_label: str,
    new_run_reads: set[str],
    new_run_writes: set[str],
) -> list[str]:
    """The plain-English warning sentences for starting a run that reads
    new_run_reads and writes new_run_writes, against every already-running
    operator in live_runs.

    live_runs is plain data, one dict per in-flight run:
        {"label": str, "reads": set[str], "writes": set[str]}
    ("reads"/"writes" may be any iterable of table names; they are used
    as sets here.)

    Two different situations are checked, and each gets its own sentence
    per conflicting table, because each is a different way the result of
    one run can end up wrong:

      1. The about-to-start run would READ a table a live run is still
         WRITING -- its own result could be computed from data that run
         has not finished producing.
      2. The about-to-start run would WRITE a table a live run is still
         READING -- that live run's result may end up computed from data
         that changed while it was still working.

    Returns an empty list when neither situation applies to any live run
    -- the caller shows no dialog at all in that case.
    """
    warnings: list[str] = []
    for live_run in live_runs:
        live_label = live_run["label"]
        live_reads = set(live_run["reads"])
        live_writes = set(live_run["writes"])

        # Situation 1: the new run would read data the live run has not
        # finished writing yet.
        for table in sorted(new_run_reads & live_writes):
            warnings.append(
                f'"{live_label}" is still adding data to the table '
                f'"{table}". If you start "{new_run_label}" now, it may '
                f'read that table before "{live_label}" has finished, so '
                f"its result could be based on incomplete data. Waiting "
                f'for "{live_label}" to finish first would avoid this.'
            )

        # Situation 2: the new run would change data the live run is
        # still reading, so the live run's own result would no longer
        # match what it read.
        for table in sorted(new_run_writes & live_reads):
            warnings.append(
                f'"{live_label}" is still reading the table "{table}". If '
                f'you start "{new_run_label}" now, it will change that '
                f'table while "{live_label}" is still using it, so the '
                f'result "{live_label}" produces may not match what is on '
                f'screen by the time it finishes. Waiting for '
                f'"{live_label}" to finish first would avoid this.'
            )

    return warnings


# ---------------------------------------------------------------------------
# run-indicator-1: the status-bar sentence describing what is running.
#
# Pure, Qt-free arithmetic plus the researcher-facing wording -- no widget,
# importable and testable with no Qt and no controller. ui/main_window.py
# calls this with the plain data from AppController.get_live_runs() and the
# latest value it has seen from the operator_progress signal.
# ---------------------------------------------------------------------------

# run-indicator-2: a run.log() message is researcher-facing free text and
# could in principle be arbitrarily long. Truncated here, in the Qt-free
# sentence function, rather than in the controller or the operator -- the
# STORED message (AppController._latest_logs / run["message"]) is never
# cut, only what a one-line status bar shows. 100 characters is comfortably
# longer than the examples in the work item ("clip 3 of 40", "no face found
# in 12 frames so far") while still leaving room for the operator label and
# percentage on one line.
_MAX_INDICATOR_MESSAGE_CHARS = 100


def _truncated_message(message: str | None) -> str | None:
    """message, unchanged if it fits or is None/empty; otherwise cut to
    _MAX_INDICATOR_MESSAGE_CHARS - 3 characters plus a trailing "..."."""
    if not message:
        return message
    if len(message) <= _MAX_INDICATOR_MESSAGE_CHARS:
        return message
    return message[: _MAX_INDICATOR_MESSAGE_CHARS - 3] + "..."


def format_run_indicator_text(
    live_runs: list[dict],
    percent: int | None,
) -> str:
    """The sentence the run-indicator status-bar widget should show, or
    "" when nothing is running -- the empty string is what tells
    ui/main_window.py to hide the widget.

    live_runs is plain data, one dict per in-flight run, read from the
    "label", "table_name" and "message" keys -- exactly the shape
    AppController.get_live_runs() returns, though this function does not
    look at that dict's "operation_id" (run-indicator-1-fix): a run's
    identity has nothing to say about the sentence describing it.
    "message" (run-indicator-2) is the run's latest run.log() text, or
    None/absent if the operator has not called it yet -- a caller such as
    tests/test_result_delivery.py's older cases may omit the key entirely,
    which reads the same as None.

    percent is the latest value reported by AppController's
    operator_progress signal (0-100), or None if no progress tick has
    arrived since the live-run set last changed.

    AppController coalesces progress into ONE latest value for the WHOLE
    APPLICATION, not one per run (see operator_progress on AppController).
    So a percentage can only be shown honestly when there is exactly one
    live run -- with two or more, the shared number cannot be attributed
    to either one, so no percentage is shown at all, only how many
    operators are running and their labels.

    A run.log() message has no such ambiguity: it is stored per
    operation_id (see AppController._latest_logs), so it is shown next to
    every run it belongs to, whether there is one live run or several.
    """
    if not live_runs:
        return ""

    if len(live_runs) == 1:
        run = live_runs[0]
        text = f'Running "{run["label"]}" on "{run["table_name"]}"'
        if percent is not None:
            text += f" -- {percent}%"
        message = _truncated_message(run.get("message"))
        if message:
            text += f" -- {message}"
        return text

    parts = []
    for run in live_runs:
        part = f'"{run["label"]}"'
        message = _truncated_message(run.get("message"))
        if message:
            part += f" ({message})"
        parts.append(part)
    return f"{len(live_runs)} operators running: {', '.join(parts)}"


# ---------------------------------------------------------------------------
# run-indicator-3: the Cancel control. Pure, Qt-free wording and numbering --
# ui/main_window.py builds the pick-one dialog and the confirmation from
# these, rather than composing either itself, so a future change to the
# wording or the numbering rule touches one place.
# ---------------------------------------------------------------------------

def numbered_run_choices(live_runs: list[dict]) -> list[tuple[str, str, str]]:
    """(operation_id, label, display_text) for every live run, numbered in
    the order they started.

    live_runs must already be in start order -- exactly what
    AppController.get_live_runs() returns, because _live_runs is a plain
    dict keyed by operation_id, populated by _register_run and only ever
    popped (never reinserted) by _deregister_run, so its iteration order is
    insertion order, i.e. the order runs started.

    The numbering exists because two runs of the same operator on the same
    table read identically without it -- "Extract blendshapes on frames"
    twice tells the researcher nothing about which is which. With this,
    a Cancel control showing more than one live run can offer "1. ... on
    ..." and "2. ... on ..." instead.
    """
    return [
        (
            run["operation_id"],
            run["label"],
            f'{i}. "{run["label"]}" on "{run["table_name"]}"',
        )
        for i, run in enumerate(live_runs, start=1)
    ]


def format_cancel_message(label: str) -> str:
    """The plain sentence shown to the researcher once they have requested
    cancellation of the run named label (run-indicator-3).

    Worded the same for every mode rather than naming COLUMNS / TABLE /
    DISPLAY, because it must be an honest, plain description of BOTH: a
    COLUMNS run keeps every row it already wrote and stops before starting
    the next one (AppController.cancel_run(), the per-row runner checks
    between rows); a TABLE or DISPLAY run cannot be interrupted
    mid-computation and instead has its finished result discarded when it
    arrives, rather than stored or shown. "Nothing already written is
    undone" and "nothing still in progress will be saved" are both true
    sentences under either outcome.
    """
    return (
        f'Cancelling "{label}". Nothing already written to the table is '
        f"undone. Any work still in progress when it stops will not be "
        f"saved."
    )


# ---------------------------------------------------------------------------
# P1.9b-1: saving is refused outright while any operator run is live.
#
# Pure, Qt-free wording -- no widget, importable and testable on its own.
# Two independent reasons both land on the same refusal, so one sentence
# covers both rather than naming which applies:
#   1. The copy-on-save step rewrites in-memory media cells through
#      Dataset's normal accept path, which bumps that table's write-ticket
#      version (models/dataset.py's _commit_prepared bumps on every
#      accepted write, unconditionally). A live run reading that table
#      would see a foreign write land under it and report a false "data
#      changed" notice when it finishes (controller.py's
#      _superseded_input_tables) -- nothing about the analysis data
#      actually changed, only where an output FILE lives on disk.
#   2. A live run keeps writing new files under the OLD outputs_dir for
#      as long as it runs (operators read run.paths once per run, at
#      start -- see operators/CLAUDE.md). A copy planned and executed
#      before that run finishes would miss every file it writes after the
#      plan was built, leaving them outside the project regardless.
# ---------------------------------------------------------------------------

def format_save_blocked_message() -> str:
    """The plain sentence shown to the researcher when Save or Save As is
    refused because an operator run is still live. See the module comment
    above for why both halves of the refusal are covered by one sentence."""
    return (
        "Gelem cannot save while an operation is running. Wait for it to "
        "finish, or stop it, then save again."
    )


def format_output_copy_conflict_message(
    conflicts: tuple,
) -> str:
    """The plain sentence shown to the researcher when the copy-on-save
    plan (models/output_copy.py::OutputCopyPlan) refuses the save because
    one or more output files already exist at the destination with a
    different size than the source -- item 3 of the P1.9b-1 work item:
    the save must be refused before anything is copied or any parquet is
    written, never overwritten silently.

    conflicts is plan.conflicts, a tuple of CopyConflict. Names at most
    three destination files so the message stays readable when many
    collide at once; the exact count is always given regardless.
    """
    count = len(conflicts)
    shown = ", ".join(f'"{c.dst.name}"' for c in conflicts[:3])
    if count > 3:
        shown += f", and {count - 3} more"
    noun = "file" if count == 1 else "files"
    return (
        f"Gelem cannot save: {count} output {noun} already exist at the "
        f"destination with different content than the project's own copy "
        f"({shown}). Choose a different folder, or remove the conflicting "
        f"files there, then save again."
    )


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
        live_runs_changed:       The set of in-flight operator runs
                                 changed -- a run started or finished.
                                 Carries no payload; a listener reads the
                                 new set via get_live_runs(). Emitted on
                                 both register and deregister so a run
                                 indicator can appear the moment a run
                                 starts, before its first progress tick.
        operator_log_changed:    A run's run.log() message moved onto its
                                 live-run entry this tick (run-indicator-2).
                                 Carries no payload -- a listener reads the
                                 new text via get_live_runs()'s "message"
                                 field, the same pull-based pattern
                                 live_runs_changed uses. Emitted at most
                                 once per drain tick, and only when at
                                 least one message actually landed on a
                                 still-live run.
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
    live_runs_changed        = Signal()
    operator_log_changed     = Signal()
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
        *,
        settings_gateway=None,
        project_paths: ProjectPaths | None = None,
    ):
        super().__init__()

        self._dataset          = dataset
        self._query            = query_engine
        self._store            = artifact_store
        self._registry         = registry
        self._op_registry      = operator_registry

        # This project's directories (models/project_paths.py::ProjectPaths).
        # Default None so existing test construction sites need no edit;
        # main.py builds the real one (a fresh workspace) and passes it in.
        # It is handed to every OperatorRun as run.paths, and replaced
        # wholesale -- never mutated -- by save_project() and
        # load_project(), at the same point each already re-roots
        # ArtifactStore. Left None, a run still starts, but any of the five
        # operators that write files (video_frames, plot_advanced, plot,
        # mean_face, blendshape_avatar) fails loudly reading run.paths.
        # outputs_dir off None rather than silently writing somewhere
        # unexpected -- main.py always passes a real one, so production is
        # never in that state.
        self._project_paths    = project_paths

        # The plain-data editing face of the machine-tunable settings
        # (settings/settings_gateway.py). Default None so existing test
        # construction sites need no edit; main.py builds a real one and
        # passes it in. The two settings pass-through methods below raise
        # if it is absent. The controller only forwards calls -- it never
        # imports settings/ and holds no GelemSettings or SettingsStore.
        self._settings_gateway = settings_gateway

        # P1.8d-2b-3: Dataset holds no registry reference and there is
        # nothing to wire between them. The controller owns `registry` only
        # to answer "what does this tag render as" -- it maps a type tag to
        # a renderer and nothing else (see column_types/registry.py). A
        # column's type tag comes from that table's TableSchema.
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

        # run-indicator-2: the newest run.log() message per operation_id.
        # Coalesced the same way progress is, but keyed per run instead of
        # a single application-wide value: a worker calls run.log(text),
        # which reaches _on_run_log and overwrites this run's entry under
        # the lock, and the drain moves at most one value per run onto its
        # live-run entry per tick. Not a queue and not a growing list --
        # LATEST WINS, PER RUN -- so a run that logs once per row over
        # 50,000 rows costs the same as one that logs twice.
        self._log_lock = threading.Lock()
        self._latest_logs: dict[str, str] = {}

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

        # The directory media cells are resolved against when turning a
        # stored cell into a canonical artifact-cache address plus an
        # absolute source path. In a loaded project the cells are already
        # absolute (Dataset.load absolutises them), so this only matters
        # for a relative cell; it tracks the project folder once one
        # exists. Kept thin: the arithmetic lives in
        # media.media_address.resolve_source.
        self._project_root:   Path       = Path.cwd()

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

        # The dirty-flag reference point for has_unsaved_changes(): a
        # snapshot of Dataset.table_versions() taken right here at
        # construction, and re-taken right after every successful
        # save_project() / load_project() (never anywhere else -- a
        # table changed through any other public path, including
        # load_folder() and load_csv_as_primary(), is meant to compare
        # unequal). dict equality catches an added OR removed table key
        # as well as a changed version, so a new table counts as unsaved
        # even though its name was never in the baseline.
        self._unsaved_baseline: dict[str, int] = self._dataset.table_versions()

    # ── Run registry ─────────────────────────────────────────────────

    def _register_run(
        self,
        operation_id: str,
        label: str,
        table_name: str,
        column_tags: dict[str, str] | None = None,
        token: CancellationToken | None = None,
        *,
        operator_name: str = "",
        mode_name: str = "",
        target_table: str = "",
        parameters: dict | None = None,
        rows_requested: int = 0,
        inputs: dict[str, dict] | None = None,
    ) -> None:
        """Records a started operator run as live. See _live_runs.

        column_tags is the run's COLUMNS mode descriptor OutputSpec
        columns as a {column_name: type_tag} mapping (P1.12d-3). The item drain reads
        it back and hands it to Dataset.apply_row_updates() so a column the
        run creates is tagged in the table's schema by what the operator
        declared, not by value inference. Empty for create_table /
        create_display runs, which create no per-row columns.

        token is the run's CancellationToken (P1.12d-2a), kept here so
        cancel_run() (P1.12f-3) has something to call .cancel() on by
        operation_id. Nothing reads it back off this dict except
        cancel_run() and _run_was_cancelled() -- the row loop that reacts
        to it is handed the SAME token object through the OperatorRun this
        run was built with, not through _live_runs.

        The keyword-only arguments (P1.12f-1) carry everything
        Dataset.record_operator_run() needs at arrival that is not
        already on the run: the operator and mode identity, the
        declared target table, the parameter values, how many rows were
        requested, and the input table(s) read with their start
        versions. They default to empty/zero so a caller that only wants
        the pre-existing tracking (several tests construct a run this
        way) still works; such a run is simply recorded with blank
        provenance fields.

        "own_versions" starts empty and is filled in by
        _drain_item_results as this run's own per-row results are
        applied -- the last table_version() a table reached because of
        THIS run's writes, so that a COLUMNS run does not read its own
        writes back as if some other run had superseded it (see
        _superseded_input_tables).

        "superseded_latched" (P1.12f-1-fix) starts empty and is a SET of
        table names, filled in by _drain_item_results right BEFORE it
        applies this run's own results to a table: if that table's
        current version already disagrees with the version this run
        itself last caused there, something foreign landed in between
        two of this run's own drain ticks. Once a table name is added it
        is never removed -- a comparison only at arrival would miss
        exactly this case, because this run's own later write becomes
        the new "last thing that happened" and erases the evidence.

        "had_setup_error" / "had_row_errors" start False and are flipped
        by _on_operator_complete's "setup_error" / "row_errors" branches,
        which arrive before this run's own completion and do not
        deregister it.

        "message" (run-indicator-2) starts None and is overwritten by
        _apply_run_logs with this run's newest run.log() text, moved off
        self._latest_logs during the drain -- see get_live_runs().
        """
        self._live_runs[operation_id] = {
            "label":              label,
            "table_name":         table_name,
            "message":            None,
            "applied":            0,
            "unplaceable":        [],
            "column_tags":        dict(column_tags) if column_tags else {},
            "token":              token,
            "operator_name":      operator_name,
            "mode_name":          mode_name,
            "target_table":       target_table,
            "parameters":         dict(parameters) if parameters else {},
            "rows_requested":     rows_requested,
            "inputs":             dict(inputs) if inputs else {},
            "own_versions":       {},
            "superseded_latched": set(),
            "had_setup_error":    False,
            "had_row_errors":     False,
        }
        # A new operation_id is always a new entry (uuid-generated per
        # run), so this always changes the set -- see live_runs_changed.
        self.live_runs_changed.emit()

    def _deregister_run(self, operation_id: str) -> None:
        """Drops a run from the live set. Idempotent.

        Only emits live_runs_changed when a run was actually removed --
        a no-op pop (already-idempotent caller, or a run the failed-start
        cleanup already dropped) is not a change to the set.
        """
        if self._live_runs.pop(operation_id, None) is not None:
            self.live_runs_changed.emit()

    def get_live_runs(self) -> list[dict]:
        """Plain data describing every in-flight operator run, for the
        run-indicator widget (ui/main_window.py) -- never the internal
        dict, the CancellationToken, or any other internal key.

        Returns one
        {"operation_id": str, "label": str, "table_name": str,
         "message": str | None}
        dict per live run, IN THE ORDER THOSE RUNS STARTED.

        That order is a real guarantee, not an accident of dict iteration:
        _live_runs is only ever appended to by _register_run (a fresh
        operation_id every time -- uuid-generated per run) and popped from
        by _deregister_run, never reordered or reinserted, so its
        insertion order -- which Python dicts preserve -- is always the
        order runs started. run-indicator-3's Cancel control relies on
        this: with more than one live run it numbers them "1.", "2." and
        so on (see numbered_run_choices()) so the researcher can tell two
        runs of the same operator on the same table apart, and that
        numbering is only meaningful because this order is start order.

        operation_id is an opaque handle, the same discipline as row_id
        (see the "Row identity and lineage" rules in CLAUDE.md): the UI
        may hold it, pass it back to the controller, and use it to tell
        two live runs apart, but must not parse it, sort by it, or
        construct one itself. It is the same value _register_run was
        called with -- run-indicator-1-fix added it, and run-indicator-3's
        cancel_run(operation_id) is the caller that now names a run with
        it.

        message (run-indicator-2) is this run's newest run.log() text, or
        None if the operator has not called run.log() yet. Unlike
        percent -- which AppController coalesces into one value for the
        whole application (see operator_progress) -- a message is stored
        per operation_id, so each live run carries its own.
        """
        return [
            {
                "operation_id": operation_id,
                "label":        run["label"],
                "table_name":   run["table_name"],
                "message":      run["message"],
            }
            for operation_id, run in self._live_runs.items()
        ]

    def is_save_blocked(self) -> bool:
        """True while Save / Save As must be refused (P1.9b-1): any
        operator run is still live. See the module comment above
        format_save_blocked_message() for why.

        Thin sugar over get_live_runs() -- self._live_runs is exactly the
        "any run live" query already public through that method -- kept
        as its own method so the UI can ask the one question it actually
        has ("can I even open the save dialog right now?") without
        building and discarding the full per-run list, and so a future
        second reason to block saving has one place to add itself.
        Intended to be checked BEFORE a file dialog is even shown, and
        save_project() checks the same condition again itself -- a run
        could go live in the gap between the two.
        """
        return bool(self._live_runs)

    def has_unsaved_changes(self) -> bool:
        """True if any stored table has changed since the last save or
        load -- the dirty flag ui/close_prompt.py's decide_close_prompt()
        reads to decide whether closing needs to ask.

        Compares Dataset.table_versions() (a table_name -> write-ticket
        version dict) against self._unsaved_baseline, a snapshot taken at
        construction and re-taken right after every successful
        save_project() / load_project(). dict equality means an added or
        removed table counts as a change, not just a bumped version, so a
        fresh load_folder() / load_csv_as_primary() into a session that
        started with nothing saved reads as unsaved even though every
        individual table is "new" rather than "changed".

        Deliberately narrower than "everything a researcher might call
        unsaved": filters, sort, group-by and selection are not Dataset
        state (see docs/review/unsaved-work-survey.md section 2) and
        save_project() never persists them, so they are correctly absent
        from this comparison.
        """
        return self._dataset.table_versions() != self._unsaved_baseline

    def cancel_run(self, operation_id: str) -> None:
        """Requests cancellation of the live run named by operation_id
        (run-indicator-3), the opaque handle get_live_runs() hands back.

        Sets that run's CancellationToken and nothing else. What happens
        next depends on the mode, and neither path is driven from here:

          * COLUMNS -- OperatorRegistry._run_create_columns_worker checks
            run.cancelled() BETWEEN rows (never mid-row: an operator's own
            row work is its own business) and stops there, keeping every
            result already handed to on_item_complete.
          * TABLE / DISPLAY -- single-shot; the runner cannot interrupt
            one mid-computation. Its finished result is discarded when it
            arrives instead -- see _on_operator_complete's "create_table"
            / "create_display" branches and _run_was_cancelled().

        Either way _run_outcome() records the run "partial" once it ends.

        Cancelling an operation_id that does not name a live run is a
        no-op, not an error: the run may have finished, or never started,
        between the researcher's click and this call reaching the
        controller.
        """
        run = self._live_runs.get(operation_id)
        if run is None:
            return
        token = run.get("token")
        if token is not None:
            token.cancel()

    def _run_was_cancelled(self, run: dict) -> bool:
        """True if cancel_run() was called for this live-run entry.

        Reads the same CancellationToken the run's OperatorRun carries, so
        this agrees with what run.cancelled() tells the operator/runner --
        there is no separate "cancelled" flag to fall out of sync with it.
        """
        token = run.get("token")
        return token is not None and token.is_cancelled()

    def _attach_run_provenance(
        self,
        operation_id: str,
        *,
        operator_name: str,
        mode_name: str,
        target_table: str,
        parameters: dict,
        rows_requested: int,
        inputs: dict[str, dict],
    ) -> None:
        """Fills in the provenance fields _register_run defaults to empty
        (P1.12f-1), as a step separate from _register_run itself.

        Kept separate rather than folded into _register_run's own
        argument list: tests/test_operator_tag_hints.py monkeypatches
        _register_run with a spy that forwards only its original five
        arguments, so widening that call's signature at the
        run_create_columns call site would break a test outside this
        item's permitted files. Called immediately after _register_run,
        before the worker starts, so there is no window where a live run
        carries the empty defaults while it could actually complete.
        """
        run = self._live_runs.get(operation_id)
        if run is None:
            return
        run["operator_name"]  = operator_name
        run["mode_name"]      = mode_name
        run["target_table"]   = target_table
        run["parameters"]     = dict(parameters)
        run["rows_requested"] = rows_requested
        run["inputs"]         = dict(inputs)

    def _run_inputs_snapshot(self, run: OperatorRun) -> dict[str, dict]:
        """{declared input name: {"table": name, "version": n}} for every
        single-table input this run reads (P1.12f-1).

        A whole-project input (RunData.tables() rather than .table()) is
        skipped: run.data.snapshot() raises OperatorRunError for one, and
        nothing builds a project input today (_build_operator_run only
        ever populates RunData.projects as {}), so there is nothing yet
        to record for that case.
        """
        snapshot: dict[str, dict] = {}
        for input_name in run.data.input_names():
            try:
                table_snapshot = run.data.snapshot(input_name)
            except OperatorRunError:
                continue
            snapshot[input_name] = {
                "table":   table_snapshot.table_name,
                "version": table_snapshot.version,
            }
        return snapshot

    def _run_outcome(self, mode: str, run: dict) -> str:
        """One of "complete", "partial" or "failed" for a run ending at
        this _on_operator_complete branch (P1.12f-1; cancellation folded
        in by P1.12f-3, corrected by run-indicator-3-fix).

        "failed" is exactly the "error" mode -- the run never finished.
        Otherwise "partial" if a setup_error or row_errors landed for
        this run, or it left rows unplaceable -- unchanged, and true
        whether or not anyone cancelled.

        RULING (run-indicator-3-fix): outcome describes what the run
        PRODUCED, not what the researcher asked for. Cancelling is not,
        by itself, a reason to call a run "partial" -- a COLUMNS run that
        happened to apply every row it was asked for before the Cancel
        click reached the controller produced a complete result, and
        recording it "partial" would write a false "this is incomplete"
        into a provenance log a researcher may read months later. Whether
        cancellation was even REQUESTED is recorded separately (see
        _finish_run_provenance's cancellation_requested field) so that
        fact is never lost.

        So a cancelled run is "partial" only if it actually produced less
        than it was asked for:

          * COLUMNS -- run["applied"] (the count of per-row results the
            drain actually applied; unplaceable is already ruled out
            above, so nothing here double-counts it) compared against
            run["rows_requested"]. Falls short -> "partial" (the between-
            rows check really did stop it early); reaches it -> "complete"
            even though cancel_run() was called.
            WHAT THIS TEST GETS WRONG: a row silently skipped because its
            image failed to load (BaseOperator.load_image() returning
            None) is not counted in run["applied"] either, and is not a
            setup_error, a row_error, or an unplaceable row -- it leaves
            no trace anywhere else on the run. An UNcancelled run with
            such a skip is already called "complete" today (this rule
            does not newly check applied-vs-requested when there is no
            cancellation). But if the SAME run is also cancelled -- even
            after the loop had already finished attempting every row --
            this test sees applied < rows_requested from the skip alone
            and reports "partial", crediting the shortfall to
            cancellation when the real cause was an unrelated load
            failure the run already had before anyone clicked Cancel.
            Closing that gap needs the load-failure count tracked
            separately on the run, which nothing does today; not fixed
            here.
          * TABLE / DISPLAY -- single-shot; _on_operator_complete's
            "create_table" / "create_display" branches ALWAYS discard the
            result once cancel_run() has reached a still-live run (they
            never store a result and then separately notice it was
            cancelled -- the store-or-discard decision and this outcome
            are computed from the very same _run_was_cancelled() check,
            at the same moment, on the main thread). A discarded result
            produced nothing, so cancelled COLUMNS's "did it fall short"
            test collapses to always-true here and is never reached: a
            cancelled TABLE/DISPLAY run is unconditionally "partial".
            There is consequently no path in this codebase today that
            reaches "complete" with cancellation_requested true for
            either of these two modes -- only "cancelled but too late to
            matter" (cancel_run() finds the run no longer live, so
            _run_was_cancelled() is never even asked) reaches "complete",
            and that case correctly reports cancellation_requested false,
            because the token was never touched.
        """
        if mode == "error":
            return "failed"
        if run["had_setup_error"] or run["had_row_errors"] or run["unplaceable"]:
            return "partial"
        if self._run_was_cancelled(run):
            if mode == "create_columns":
                if run["applied"] < run["rows_requested"]:
                    return "partial"
                return "complete"
            # create_table / create_display: always discarded once
            # cancellation reaches a live run -- see the docstring above.
            return "partial"
        return "complete"

    def _run_input_start_version(self, run: dict, table_name: str) -> int | None:
        """The write-ticket version run["inputs"] recorded for table_name
        (P1.12f-1-fix), or None if table_name is not one of this run's
        declared inputs.

        Used by _drain_item_results' pre-apply latch check, which needs
        this run's ORIGINAL start version for a table it may not have
        written to yet -- run["own_versions"] only gains an entry for a
        table after this run's first write there.
        """
        for info in run["inputs"].values():
            if info["table"] == table_name:
                return info["version"]
        return None

    def _superseded_input_tables(self, run: dict) -> list[str]:
        """Input table names whose data has moved out from under this run
        (P1.12f-1, latch added by P1.12f-1-fix).

        A table counts as superseded by either of two independent checks,
        because each catches a different moment a foreign write can land:

          1. ARRIVAL: its CURRENT write-ticket version differs from the
             version THIS RUN itself last caused there -- read off
             run["own_versions"], updated by _drain_item_results every
             time it applies this run's own results to that table -- or,
             for a table this run never wrote to (TABLE/DISPLAY mode, or
             a COLUMNS run that produced no results for it), from the
             version recorded at run start (run["inputs"]). This is the
             false-positive guard: a COLUMNS run reads and writes the
             SAME table, and its own apply_row_updates() call bumps that
             table's version before this run's completion is processed.
             Comparing against the start version alone would mark almost
             every successful COLUMNS run as superseded by itself.

          2. LATCHED: run["superseded_latched"] already names the table.
             _drain_item_results sets this BEFORE each of this run's own
             applies, by the same comparison as (1) but made at that
             earlier moment. This is what catches a foreign write that
             lands BETWEEN two of this run's own drain ticks: by arrival
             time this run's own later write is the last thing that
             happened to the table, so check (1) alone would see nothing
             wrong -- the foreign write is masked by this run's own
             next commit. The latch is taken at the one moment the
             foreign write is still the last thing that happened, and
             never cleared once set.

        A false staleness note is advisory and cheap; a missed one is a
        wrong number in a paper, so this errs toward over-reporting: OR,
        not AND.
        """
        superseded: list[str] = []
        seen_tables: set[str] = set()
        for info in run["inputs"].values():
            table_name = info["table"]
            if table_name in seen_tables:
                continue
            seen_tables.add(table_name)
            baseline = run["own_versions"].get(table_name, info["version"])
            arrival_moved = self._dataset.table_version(table_name) != baseline
            latched_moved = table_name in run["superseded_latched"]
            if arrival_moved or latched_moved:
                superseded.append(table_name)
        return superseded

    def _finish_run_provenance(self, mode: str, run: dict) -> None:
        """Records this run's provenance entry and, if any input table it
        read has moved under it, emits one plain-English notice (P1.12f-1).

        Called for every run-ending branch of _on_operator_complete WHILE
        the run is still in self._live_runs, and always before
        _deregister_run -- a run no longer live at arrival is not
        recorded (see the "arrived after the project changed" branches,
        which already return before this would be reached).

        cancellation_requested (run-indicator-3-fix) records the FACT
        that cancel_run() was called for this run, independently of
        outcome -- outcome alone can no longer say so, now that a
        cancelled COLUMNS run that finished everything it was asked for
        is recorded "complete". Read straight off the same token
        _run_outcome() itself checks, so the two can never disagree.
        """
        outcome    = self._run_outcome(mode, run)
        superseded = self._superseded_input_tables(run)
        self._dataset.record_operator_run(
            operator_name=run["operator_name"],
            mode=run["mode_name"],
            label=run["label"],
            parameters=run["parameters"],
            target_table=run["target_table"],
            inputs=run["inputs"],
            rows_requested=run["rows_requested"],
            rows_applied=run["applied"] - len(run["unplaceable"]),
            unplaceable_row_ids=run["unplaceable"],
            outcome=outcome,
            superseded_tables=superseded,
            cancellation_requested=self._run_was_cancelled(run),
        )
        if superseded:
            tables_str = ", ".join(f'"{t}"' for t in superseded)
            self.error_occurred.emit(
                f'"{run["label"]}" read {tables_str}, but that data has '
                f"changed since the run started, so this result may not "
                f'match what is on screen. Re-running "{run["label"]}" '
                f"would recompute it against the current data."
            )

    def get_write_read_conflict_warnings(
        self, operator_name: str, mode_name: str,
    ) -> list[str]:
        """The plain-English warning sentences (P1.12f-2) for starting
        operator_name's mode_name mode ("COLUMNS" / "TABLE" / "DISPLAY",
        an ExecutionMode member name, not its .value) against the active
        table right now, given every run already in self._live_runs.

        Empty means no conflict: ui/main_window.py's one caller shows no
        dialog at all in that case. An unknown operator, an operator
        carrying no descriptor, or a descriptor with no entry for
        mode_name all return an empty list rather than raising -- this
        query only ever adds a warning, and the run itself will refuse
        for its own reasons (with its own message) when the caller
        actually tries to start it.

        Only ACTIVE_TABLE inputs exist today (operators/descriptor.py),
        so the about-to-start run's read set is the active table, once
        per declared ACTIVE_TABLE input on this mode. Its write set is
        the active table when the mode's declared output adds columns to
        it (COLUMNS mode; see OutputSpec), empty otherwise -- the same
        rule _build_operator_run uses for target_table. Deriving both
        from the descriptor, rather than comparing mode_name to the
        literal string "COLUMNS", keeps this correct without a rewrite
        once a second input kind exists.
        """
        operator = self._op_registry.get(operator_name)
        if operator is None or operator.descriptor is None:
            return []
        try:
            mode = ExecutionMode[mode_name]
        except KeyError:
            return []
        mode_descriptor = operator.descriptor.mode_for(mode)
        if mode_descriptor is None:
            return []

        active_table = self._active_table
        new_reads = {
            active_table
            for input_spec in mode_descriptor.inputs
            if input_spec.kind is InputKind.ACTIVE_TABLE
        }
        new_writes = {active_table} if mode_descriptor.output.columns else set()

        # Each live run's read set is the tables it declared as inputs at
        # start (run["inputs"], P1.12f-1); its write set is its target
        # table when it has one (empty for TABLE/DISPLAY, see
        # _build_operator_run) -- both already plain data on the run
        # dict, nothing to look up on a descriptor a second time.
        live_runs = [
            {
                "label": run["label"],
                "reads": {info["table"] for info in run["inputs"].values()},
                "writes": {run["target_table"]} if run["target_table"] else set(),
            }
            for run in self._live_runs.values()
        ]

        return write_read_conflict_warnings(
            live_runs, mode_descriptor.label, new_reads, new_writes,
        )

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
        self._apply_run_logs()
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
        # P1.8d-2b-2: the declared output-column tags of every live run that
        # put a row into a given table's batch, unioned. A column the batch
        # creates is then tagged in the schema by what the operator declared
        # rather than by value inference. A later run wins a name collision,
        # the same last-write-wins this drain already applies to values.
        # Built in the drain loop below so this is one pass, not one per
        # table.
        tags_by_table: dict[str, dict[str, str]] = {}
        # P1.12f-1: which live runs put at least one row into this tick's
        # batch for a table, so that once the batch is applied each of
        # those runs can record the version its OWN write just produced
        # (see _superseded_input_tables).
        contributors_by_table: dict[str, set[str]] = {}

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
            contributors_by_table.setdefault(table_name, set()).add(operation_id)
            run_tags = run.get("column_tags")
            if run_tags:
                tags_by_table.setdefault(table_name, {}).update(run_tags)

        for table_name, updates in batches.items():
            # P1.12f-1-fix: latch supersession for every run about to
            # write here, BEFORE this tick's own apply moves the table
            # again. This is the only place a foreign write landing
            # BETWEEN two of this run's own drain ticks is ever visible:
            # once this tick's apply below runs, THIS run's write becomes
            # the table's last commit and a foreign write sandwiched
            # before it is no longer distinguishable at arrival time.
            version_before_apply = self._dataset.table_version(table_name)
            for operation_id in contributors_by_table.get(table_name, ()):
                run = self._live_runs.get(operation_id)
                if run is None:
                    continue
                start_version = self._run_input_start_version(run, table_name)
                if start_version is None:
                    # table_name is not a declared input of this run --
                    # nothing to compare against, so nothing to latch.
                    continue
                baseline = run["own_versions"].get(table_name, start_version)
                if version_before_apply != baseline:
                    run["superseded_latched"].add(table_name)

            column_tags = tags_by_table.get(table_name)
            unplaceable = self._dataset.apply_row_updates(
                table_name, updates, column_tags=column_tags or None
            )
            unplaceable_set = set(unplaceable)
            for row_id in unplaceable_set:
                operation_id = row_owner.get((table_name, row_id))
                run = self._live_runs.get(operation_id)
                if run is not None:
                    run["unplaceable"].append(row_id)

            # This tick's write-ticket version for the table, read once
            # here (same thread, right after the apply that produced it)
            # so every contributing run records the EXACT version its own
            # write caused -- not a version read later, after some other
            # write might have landed too.
            current_version = self._dataset.table_version(table_name)
            for operation_id in contributors_by_table.get(table_name, ()):
                run = self._live_runs.get(operation_id)
                if run is not None:
                    run["own_versions"][table_name] = current_version

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

    def _apply_run_logs(self) -> None:
        """
        Moves each run's newest run.log() message (run-indicator-2) onto
        its live-run entry, at most once per tick per run -- the same
        coalescing _emit_progress_if_changed applies to the single
        application-wide percentage, just keyed per operation_id instead
        of global. A message for a run no longer in self._live_runs
        (already deregistered, or a stray call from a worker that outlived
        a failed start) is dropped here rather than in _on_run_log, so a
        worker thread's log() call never has to know whether its run is
        still live -- the same "drop silently, no dialog" treatment
        _drain_item_results gives a result from a dead run.

        Emits operator_log_changed at most once, and only if at least one
        message actually landed on a still-live run -- an empty tick (no
        operator called run.log() since the last one) emits nothing, the
        same discipline _deregister_run applies to live_runs_changed.
        """
        with self._log_lock:
            latest = self._latest_logs
            self._latest_logs = {}
        changed = False
        for operation_id, text in latest.items():
            run = self._live_runs.get(operation_id)
            if run is not None:
                run["message"] = text
                changed = True
        if changed:
            self.operator_log_changed.emit()

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

    def _on_run_log(self, operation_id: str, text: str) -> None:
        """The run.log() sink (run-indicator-2), wired as OperatorRun's
        _log_fn in _build_operator_run. Runs on the worker thread that
        called run.log(): does nothing but overwrite this run's latest
        message under the lock -- LATEST WINS, PER RUN, same pattern as
        _on_progress. Never raises, whether or not operation_id still
        names a live run: this is a plain dict write, and _apply_run_logs
        is what checks liveness, on the main thread, next tick.
        """
        with self._log_lock:
            self._latest_logs[operation_id] = text

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
        # The label is the run's mode-descriptor label, computed by the
        # worker and passed in, so this callback reads no component state
        # from the worker thread.
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
            self._finish_run_provenance(mode, run)
            self._deregister_run(operation_id)
            self.operator_complete.emit(operator_name)
            self.columns_updated.emit(self.get_column_names())
            self._refresh_result()

        elif mode == "setup_error":
            # A run that aborted before finishing. Rows processed before
            # the abort still produced results that are queued behind
            # this message; the create_columns completion that follows
            # owns deregistration once they have landed, so this branch
            # only surfaces the message. It does record the flag on the
            # still-live run (P1.12f-1), so that completion's provenance
            # entry reports outcome "partial" rather than "complete".
            operation_id, label, message = payload
            run = self._live_runs.get(operation_id)
            if run is not None:
                run["had_setup_error"] = True
            self.error_occurred.emit(
                f'Cannot run operator "{label}"\n\n{message}'
            )

        elif mode == "row_errors":
            # An additional end-of-run summary. Deregistration and the
            # unplaceable report belong to the create_columns completion
            # that still follows, so this branch does not deregister. As
            # with setup_error, it records the flag on the still-live run
            # (P1.12f-1) for that completion's outcome.
            operation_id, label, errors = payload
            run = self._live_runs.get(operation_id)
            if run is not None:
                run["had_row_errors"] = True
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
            run = self._live_runs.get(operation_id)
            was_live = run is not None
            # P1.12f-3: a TABLE run is single-shot -- the runner cannot
            # interrupt it mid-computation, so cancel_run() could not stop
            # this work. It let the run finish and is discarding the
            # result here instead, once it arrives. Checked before
            # _deregister_run pops the entry, but `run` (the dict object)
            # stays valid to read after that -- deregistering only removes
            # it from self._live_runs, not from this local reference.
            cancelled = was_live and self._run_was_cancelled(run)
            if was_live:
                self._finish_run_provenance(mode, run)
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
            if cancelled:
                self.error_occurred.emit(
                    f'"{run["label"]}" was cancelled. A table result '
                    f"cannot be produced partway through, so its finished "
                    f"result was discarded rather than stored."
                )
                self.operator_complete.emit(operator_name)
                return
            table_name = f"{operator_name}_result"
            try:
                self._dataset.create_table_from_df(table_name, result_df)
                # No eager thumbnail pass here (P0.5b-3i): when the
                # researcher switches to this table its tiles paint and
                # render_column_value() queues each request on demand.
                self.tables_updated.emit(self._dataset.list_tables())
                self.table_created.emit(table_name)
                self.operator_complete.emit(operator_name)
            except Exception as e:
                self.error_occurred.emit(
                    f"Failed to store table from '{operator_name}': {e}"
                )

        elif mode == "create_display":
            operation_id, operator_name, result = payload
            run = self._live_runs.get(operation_id)
            was_live = run is not None
            # Same "cannot interrupt, so discard on arrival" limit as
            # create_table above -- see the comment there.
            cancelled = was_live and self._run_was_cancelled(run)
            if was_live:
                self._finish_run_provenance(mode, run)
            self._deregister_run(operation_id)
            if not was_live:
                self.error_occurred.emit(
                    f'The "{operator_name}" result arrived after the '
                    f"project changed and was discarded."
                )
                return
            if cancelled:
                self.error_occurred.emit(
                    f'"{run["label"]}" was cancelled. A result cannot be '
                    f"produced partway through, so its finished result "
                    f"was discarded rather than shown."
                )
                self.operator_complete.emit(operator_name)
                return
            result["operator_name"] = operator_name
            self.display_result_ready.emit(result)
            self.operator_complete.emit(operator_name)

        elif mode == "error":
            operation_id, operator_name, message = payload
            run = self._live_runs.get(operation_id)
            if run is not None:
                self._finish_run_provenance(mode, run)
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
        # A gallery scrolled or mounted a new window: tell the artifact
        # store which addresses are on screen so it can drop queued
        # thumbnail jobs for rows that just left it (P0.5b-3ii-b).
        self._update_wanted_addresses()

    def clear_displayed_range(self, viewport_key: str) -> None:
        """Forgets the displayed range for viewport_key, if any."""
        self._displayed_ranges.pop(viewport_key, None)
        # One fewer gallery on screen. Recompute the wanted set from the
        # ranges that remain -- if this was the last one, the set is empty
        # and every queued job may be dropped (P0.5b-3ii-b).
        self._update_wanted_addresses()

    def get_displayed_ranges(self) -> list[tuple[int, int]]:
        """
        Returns every reported displayed range, sorted by start.

        P0.4 only makes this information available. Prefetch margins,
        priorities and cancellation are P0.5's work and are not built
        here.
        """
        return sorted(self._displayed_ranges.values(), key=lambda r: r[0])

    def _update_wanted_addresses(self) -> None:
        """Tell the ArtifactStore which canonical media addresses the
        galleries are currently showing (P0.5b-3ii-b).

        This is the consumer of ArtifactStore.set_wanted_addresses(): the
        store uses the set to drop queued thumbnail jobs for rows that
        scrolled off screen. It runs on every displayed-range report or
        clear -- a gallery already de-dupes those, so this fires once per
        mounted-window shift, not once per scrolled pixel.

        It must not stat, open or decode anything -- it reads cells that
        are already in memory and turns each into an address with pure
        path arithmetic (media.media_address.resolve_source touches no
        filesystem). The cost is O(visible tiles), the same order as
        painting them.
        """
        # 1. The media columns are the active table's schema columns whose
        #    type tag is "media_path" -- the same test render_column_value()
        #    uses (P1.8d-2a: the schema, not the registry's column-name map,
        #    is the authority for what a named column is). This set does not
        #    change from row to row, so resolve it once here rather than
        #    re-checking every column of every row.
        schema = self._dataset.schema_for(self._active_table)
        if schema is None:
            self._store.set_wanted_addresses(set())
            return
        media_columns = [
            spec.name for spec in schema.columns_with_tag("media_path")
        ]
        if not media_columns:
            self._store.set_wanted_addresses(set())
            return

        # 2. Collect the row ids in every currently-displayed range. Two
        #    galleries showing different slices union here rather than
        #    overwrite -- get_displayed_ranges() returns one entry per
        #    viewport, and a set folds any overlap.
        row_ids: list[str] = []
        for start, stop in self.get_displayed_ranges():
            row_ids.extend(self.get_row_ids_in_range(start, stop))

        # 3. For each row, turn each non-empty media cell into a canonical
        #    artifact-cache address. A cell that does not parse as a media
        #    address is skipped, exactly as render_column_value() skips it
        #    (a file name with a literal '#', pending P1.8).
        wanted: set[str] = set()
        for row_id in row_ids:
            row = self._dataset.get_row(row_id, self._active_table)
            for column_name in media_columns:
                value = row.get(column_name)
                if not isinstance(value, str) or not value:
                    continue
                try:
                    canonical_address, _source_path = self._resolve_media_cell(value)
                except MediaAddressError:
                    continue
                wanted.add(canonical_address)

        # 4. Hand the set to the store. An empty set (every gallery
        #    cleared its range) says nothing on screen wants a picture, so
        #    every queued job may be dropped.
        self._store.set_wanted_addresses(wanted)

    # ── Public API ────────────────────────────────────────────────────

    def load_folder(self, folder_path: Path) -> None:
        """
        Loads a folder of media files into the dataset.

        Thumbnails are generated on demand as tiles paint (P0.5b-3i),
        not in an eager whole-table pass here.

        Args:
            folder_path: Path to the folder containing media files.
        """
        try:
            self._store.reset()
            # The dataset is being replaced: every in-flight operator run
            # now targets rows that will not exist. Drop them so a late
            # result cannot write onto the new folder's rows. Guarded so
            # live_runs_changed fires only on an actual change.
            if self._live_runs:
                self._live_runs.clear()
                self.live_runs_changed.emit()
            self._active_filters = []
            self._group_by       = None
            self._visible_cols   = None
            self._project_root   = Path(folder_path)

            self._dataset.load_folder(folder_path)

            self.columns_updated.emit(self.get_column_names())
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
            # in-flight run's results are valid any more. Guarded so
            # live_runs_changed fires only on an actual change.
            if self._live_runs:
                self._live_runs.clear()
                self.live_runs_changed.emit()
            self._active_filters = []
            self._group_by       = None
            self._visible_cols   = None
            # Relative image paths in the CSV are relative to the CSV's
            # own folder, not the process working directory.
            self._project_root   = Path(csv_path).parent

            self._dataset.load_csv_as_primary(csv_path, image_column)
            # Thumbnails are generated on demand as tiles paint
            # (P0.5b-3i), not in an eager whole-table pass here.

            self.columns_updated.emit(self.get_column_names())
            self.tables_updated.emit(self._dataset.list_tables())
            self._refresh_result()

        except Exception as e:
            self.error_occurred.emit(f"Failed to load CSV: {e}")

    def load_csv(
        self,
        csv_path: Path,
        target_table: str,
        csv_key: str,
        target_key: str,
        preprocess: dict | None = None,
    ) -> None:
        """
        Starts the CSV merge workflow.

        Args:
            csv_path:     Path to the CSV file.
            target_table: Name of the table to merge the CSV onto.
            csv_key:      Column name in the CSV to join on.
            target_key:   Column name in target_table to join on.
            preprocess:   Optional preprocessing rules.
        """
        try:
            report = self._dataset.merge_csv(
                csv_path, target_table, csv_key, target_key, preprocess
            )
            self.merge_report_ready.emit(report)
        except Exception as e:
            self.error_occurred.emit(f"Failed to read CSV: {e}")

    def confirm_merge(self, report) -> None:
        """
        Commits a CSV merge after the researcher reviews the report.

        When report.would_expand is set, Dataset.confirm_merge() creates a
        new table instead of writing to the target table (see its
        docstring), so this emits tables_updated and table_created --
        the same signals an operator's create_table result uses -- rather
        than columns_updated, and does not refresh the current result
        since the active table did not change.

        Args:
            report: The MergeReport returned by merge_csv().
        """
        try:
            self._dataset.confirm_merge(report)
            if report.would_expand:
                self.tables_updated.emit(self._dataset.list_tables())
                self.table_created.emit(report.expand_table_name)
            else:
                self.columns_updated.emit(self.get_column_names())
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
        if DEFAULT_MEDIA_COLUMN_NAME in self.get_visual_column_names():
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

    def _build_operator_run(
        self,
        operator,
        mode: ExecutionMode,
        operation_id: str,
        table_name: str,
        parameters: dict,
        input_frame,
    ):
        """Build the OperatorRun for one run, on the main thread, before
        any worker starts. Returns (run, token).

        The parameters are validated against the operator's descriptor for
        this mode here: OperatorRunSpec raises OperatorRunError on an
        undeclared name, a missing required parameter, an out-of-range
        value or a non-scalar value. The caller catches that and surfaces
        it through error_occurred without starting the run.

        Raises RuntimeError if the operator carries no descriptor, or no
        descriptor for this mode. Every real operator has both as of
        P1.12d-1; an operator that does not is not runnable, and failing
        here -- rather than falling back to a stale instance value -- is
        the whole point of P1.12d-2a.
        """
        if operator.descriptor is None:
            raise RuntimeError(
                f'operator "{operator.name}" has no descriptor and cannot '
                f"run."
            )
        mode_descriptor = operator.descriptor.mode_for(mode)
        if mode_descriptor is None:
            raise RuntimeError(
                f'operator "{operator.name}" descriptor has no '
                f"{mode.name} mode."
            )

        # COLUMNS writes new columns into an existing table and needs its
        # name; TABLE and DISPLAY store nothing into one.
        target_table = table_name if mode is ExecutionMode.COLUMNS else ""

        spec = OperatorRunSpec(
            operation_id=operation_id,
            operator_name=operator.name,
            mode=mode,
            mode_descriptor=mode_descriptor,
            parameters=parameters,
            target_table=target_table,
        )

        # One frozen snapshot for the mode's declared ACTIVE_TABLE input,
        # keyed by the input NAME the descriptor gives it. It wraps the
        # SAME frame the worker already received -- not a second copy --
        # plus the table's name and its current write-ticket version.
        #
        # Note what that frame is: the rows THIS RUN was selected to
        # process, not the whole table. An operator author calling
        # run.data.table("source") gets exactly those selected rows. The
        # table_name and version do not describe the frame's contents;
        # they identify the stored table the rows came from, and the
        # version is what P1.12f compares to detect that the table moved
        # under the run. Widening a declared input to mean the whole
        # table -- and the read-set / write-set distinction that lets an
        # operator ask for more than its selected rows -- is later work.
        tables: dict = {}
        active_input = next(
            (
                input_spec
                for input_spec in mode_descriptor.inputs
                if input_spec.kind is InputKind.ACTIVE_TABLE
            ),
            None,
        )
        if active_input is not None:
            tables[active_input.name] = TableSnapshot(
                table_name=table_name,
                frame=input_frame,
                version=self._dataset.table_version(table_name),
                carry_columns=tuple(
                    self._dataset.columns_to_carry(table_name)
                ),
            )
        run_data = RunData(tables=tables, projects={})

        token = CancellationToken()
        run = OperatorRun(
            spec=spec,
            data=run_data,
            paths=self._project_paths,
            _token=token,
            # The per-row result sink for a COLUMNS run. Wired for the
            # contract P1.12f consumes; no operator calls run.emit() yet.
            _emit_fn=self._on_item_complete,
            # run.log() (run-indicator-2). Wired for every mode -- TABLE
            # and DISPLAY operators may call it too, not just COLUMNS.
            _log_fn=self._on_run_log,
        )
        return run, token

    def run_create_columns(
        self,
        operator_name: str,
        row_ids: list[str],
        parameters: dict | None = None,
    ) -> None:
        """
        Runs create_columns() on a list of rows in a background thread.

        Args:
            operator_name: Name of the operator to run.
            row_ids:       Rows to process.
            parameters:    The run's parameter values, keyed by the
                           operator's declared descriptor parameter names
                           (from the parameter dialog's parameter_values(),
                           via MainWindow). None means "no parameters".
        """
        parameters = dict(parameters or {})
        try:
            operator = self._op_registry.get(operator_name)
            if operator is None:
                self.error_occurred.emit(
                    f'Operator "{operator_name}" cannot add columns.'
                )
                return
            operation_id = str(uuid.uuid4())
            table_name   = self._active_table
            # One snapshot of exactly the selected rows, taken once here
            # on the main thread -- not one Dataset.get_row() call (and
            # one full-table copy) per row. The RunData snapshot below
            # wraps this SAME frame; it is not copied again.
            snapshot = self._dataset.snapshot_rows(table_name, row_ids)

            # Build (and validate) the run before registering it, so a
            # bad parameter set never leaves a live-run entry behind.
            # _build_operator_run also raises (RuntimeError) if the
            # operator carries no descriptor, or none for COLUMNS mode --
            # that is now the only "operator cannot add columns" check.
            try:
                run, token = self._build_operator_run(
                    operator, ExecutionMode.COLUMNS, operation_id,
                    table_name, parameters, snapshot,
                )
            except (OperatorRunError, RuntimeError) as e:
                self.error_occurred.emit(
                    f'Cannot start "{operator.name}": {e}'
                )
                return

            mode_label = run.spec.mode_descriptor.label

            # P1.8d-2b-2 / P1.12d-3: the COLUMNS mode's declared output
            # columns (descriptor OutputSpec) are the per-row column tags.
            # They travel to the target table's TableSchema as ColumnHints
            # on the accept path -- carried on the live run (see
            # _register_run) and handed to Dataset.apply_row_updates() by
            # the item drain. The schema does not police tags, so an
            # unknown tag still reaches it; but such a column renders as a
            # placeholder, so warn once per run about any tag the registry
            # has no renderer for.
            column_tags = {
                column.name: column.type_tag
                for column in run.spec.mode_descriptor.output.columns
            }
            unknown_tags = sorted(
                tag for tag in set(column_tags.values())
                if self._registry.type_for_tag(tag) is None
            )
            if unknown_tags:
                print(
                    f"[Controller] Warning: operator {operator_name!r} "
                    f"declares column type tag(s) {unknown_tags} that "
                    f"ColumnTypeRegistry has no renderer for; those columns "
                    f"will show a placeholder."
                )

            # The per-row runner can hand the operator one decoded frame
            # (media_requirement FRAME) or nothing (METADATA, or ADDRESS --
            # where the operator resolves the media itself from its
            # metadata). It cannot produce an ordered span of video or
            # audio, so a COLUMNS mode that declares VIDEO_SPAN or
            # AUDIO_SPAN is refused here -- before any worker starts and
            # before the run is registered, the same place a bad parameter
            # set is refused. Handing such an operator a single frame, or
            # None, would be the wrong-number failure P1.12d exists to
            # remove.
            media_requirement = run.spec.mode_descriptor.media_requirement
            if media_requirement in (
                MediaRequirement.VIDEO_SPAN,
                MediaRequirement.AUDIO_SPAN,
            ):
                self.error_occurred.emit(
                    f'Cannot start "{mode_label}": it needs '
                    f'{media_requirement.name} media, which the per-row '
                    f'runner cannot supply.'
                )
                return

            # The runner builds this operator's model once per worker
            # (PER_WORKER) or once per application (SHARED), but the
            # per-row runner has no concept of a sequence, so it cannot
            # honour PER_SEQUENCE -- one isolated model instance per
            # clip-run, reset at the sequence boundary. A COLUMNS mode
            # that declares it is refused here, before any worker starts
            # and before the run is registered -- the same place a bad
            # parameter set and a VIDEO_SPAN / AUDIO_SPAN requirement are
            # refused. The worker keeps a defensive guard in case one
            # slips through.
            model_lifecycle = run.spec.mode_descriptor.model_lifecycle
            if model_lifecycle is ModelLifecycle.PER_SEQUENCE:
                self.error_occurred.emit(
                    f'Cannot start "{mode_label}": it declares '
                    f'model_lifecycle {model_lifecycle.name}, which the '
                    f'per-row runner cannot honour.'
                )
                return

            self._register_run(
                operation_id, mode_label, table_name,
                column_tags, token=token,
            )
            self._attach_run_provenance(
                operation_id,
                operator_name=run.spec.operator_name,
                mode_name=run.spec.mode.name,
                target_table=run.spec.target_table,
                parameters=dict(run.spec.parameters),
                rows_requested=len(row_ids),
                inputs=self._run_inputs_snapshot(run),
            )
            try:
                started = self._op_registry.run_create_columns(
                    operator_name,
                    snapshot,
                    row_ids,
                    table_name,
                    run=run,
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
        parameters: dict | None = None,
    ) -> None:
        """
        Runs create_table() in a background thread.

        Args:
            operator_name: Name of the operator to run.
            row_ids:       Rows to include in the DataFrame.
            parameters:    The run's parameter values, keyed by the
                           operator's declared descriptor parameter names.
                           A group-by column, if the operator declares one,
                           arrives here -- not as a separate argument.
        """
        parameters = dict(parameters or {})
        try:
            operator = self._op_registry.get(operator_name)
            if operator is None:
                self.error_occurred.emit(
                    f'Operator "{operator_name}" cannot create a table.'
                )
                return
            operation_id = str(uuid.uuid4())
            table_name   = self._active_table
            selected_df  = self._dataset.snapshot_rows(table_name, row_ids)

            # _build_operator_run raises (RuntimeError) if the operator
            # carries no descriptor, or none for TABLE mode -- that is now
            # the only "operator cannot create a table" check.
            try:
                run, token = self._build_operator_run(
                    operator, ExecutionMode.TABLE, operation_id,
                    table_name, parameters, selected_df,
                )
            except (OperatorRunError, RuntimeError) as e:
                self.error_occurred.emit(
                    f'Cannot start "{operator.name}": {e}'
                )
                return

            self._register_run(
                operation_id, run.spec.mode_descriptor.label, table_name,
                token=token,
            )
            self._attach_run_provenance(
                operation_id,
                operator_name=run.spec.operator_name,
                mode_name=run.spec.mode.name,
                target_table=run.spec.target_table,
                parameters=dict(run.spec.parameters),
                rows_requested=len(row_ids),
                inputs=self._run_inputs_snapshot(run),
            )
            try:
                started = self._op_registry.run_create_table(
                    operator_name,
                    selected_df,
                    operation_id=operation_id,
                    run=run,
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
        parameters: dict | None = None,
    ) -> None:
        """
        Runs create_display() in a background thread.

        Args:
            operator_name: Name of the operator to run.
            row_ids:       Rows to include in the DataFrame.
            parameters:    The run's parameter values, keyed by the
                           operator's declared descriptor parameter names.
        """
        parameters = dict(parameters or {})
        try:
            operator = self._op_registry.get(operator_name)
            if operator is None:
                self.error_occurred.emit(
                    f'Operator "{operator_name}" cannot show a result.'
                )
                return
            operation_id = str(uuid.uuid4())
            table_name   = self._active_table
            selected_df  = self._dataset.snapshot_rows(table_name, row_ids)

            # _build_operator_run raises (RuntimeError) if the operator
            # carries no descriptor, or none for DISPLAY mode -- that is
            # now the only "operator cannot show a result" check.
            try:
                run, token = self._build_operator_run(
                    operator, ExecutionMode.DISPLAY, operation_id,
                    table_name, parameters, selected_df,
                )
            except (OperatorRunError, RuntimeError) as e:
                self.error_occurred.emit(
                    f'Cannot start "{operator.name}": {e}'
                )
                return

            self._register_run(
                operation_id, run.spec.mode_descriptor.label, table_name,
                token=token,
            )
            self._attach_run_provenance(
                operation_id,
                operator_name=run.spec.operator_name,
                mode_name=run.spec.mode.name,
                target_table=run.spec.target_table,
                parameters=dict(run.spec.parameters),
                rows_requested=len(row_ids),
                inputs=self._run_inputs_snapshot(run),
            )
            try:
                started = self._op_registry.run_create_display(
                    operator_name,
                    selected_df,
                    operation_id=operation_id,
                    run=run,
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
            self.columns_updated.emit(self.get_column_names())
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
            self.columns_updated.emit(self.get_column_names())
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

    def plan_output_copy(self, dest_folder: Path) -> OutputCopyPlan:
        """The copy-on-save plan (models/output_copy.py) for saving into
        dest_folder: which operator-output files under the project's
        CURRENT ProjectPaths.outputs_dir would need to be copied into
        dest_folder's own outputs directory, and which destinations
        would collide (item 3 of P1.9b-1).

        Read-only -- plans nothing on disk and changes no Dataset or
        controller state. save_project() calls this itself before
        saving; the researcher-facing warning for a large or colliding
        copy, shown before Save As is even committed to, is P1.9b-2 and
        would call this same method first.

        self._project_paths is None for a controller built without
        project_paths= (several tests construct one this way; main.py
        always passes a real one in production -- see the comment on
        __init__'s project_paths parameter). With no CURRENT outputs_dir
        there is no "old" location for any cell to lie under, so this
        returns an empty plan rather than raising.
        """
        if self._project_paths is None:
            return OutputCopyPlan(entries=(), total_bytes=0, conflicts=())
        new_paths = build_project_paths(Path(dest_folder), is_workspace=False)
        return _plan_output_copy(
            self._dataset.media_cells_by_table(),
            self._project_paths.outputs_dir,
            new_paths.outputs_dir,
        )

    def save_project(self, project_path: Path) -> bool:
        """
        Saves the current project to disk.

        Returns True only if the save actually completed -- False for
        every refusal (a live run, an output-copy conflict) and every
        failure caught below, each of which has already reported itself
        through error_occurred. ui/close_prompt.py's close flow uses this
        to decide whether closing may proceed after "Save": a cancelled
        folder dialog or a declined/failed save must leave the window
        open, which only a real success/failure answer -- not the
        previous bare None -- lets it tell apart.

        Refuses outright while any operator run is live (P1.9b-1),
        checked FIRST -- before the output-copy plan is even built, so
        nothing is copied, no parquet is written, and ProjectPaths is
        not swapped. See format_save_blocked_message() (module level,
        above this class) for why: a live run's result would come back
        falsely marked "data changed" by the in-memory cell rewrite
        below, and a live run keeps writing new files under the OLD
        outputs_dir for as long as it runs, so a copy planned now would
        miss them regardless.

        Otherwise: plans the output copy, refuses on a conflict exactly
        the same way, copies the planned files into project_path's
        outputs folder, repoints the matching in-memory media cells at
        the copies, and only then writes Parquet -- so the paths
        Dataset.save() relativises are already the ones inside
        project_path. If the cell rewrite or Dataset.save() itself then
        raises, the rewrite is rolled back before the error is reported
        (see the comment at that try/except below) -- self._project_paths
        has not moved yet in that case, so a stranded rewritten cell
        would otherwise point somewhere a later save could never find it
        again. A copy failure needs no such rollback: it happens before
        any cell has moved.

        Args:
            project_path: Path to the project folder.
        """
        if self.is_save_blocked():
            self.error_occurred.emit(format_save_blocked_message())
            return False

        try:
            plan = self.plan_output_copy(project_path)
            if plan.conflicts:
                self.error_occurred.emit(
                    format_output_copy_conflict_message(plan.conflicts)
                )
                return False

            # Main thread only (P1.9b-1). AppController lives on the main
            # thread and save_project() is only ever called directly by
            # the UI, never by a worker callback, so this blocks the UI
            # for as long as the copy takes -- the is_save_blocked() check
            # above already ruled out the one source of a CONCURRENT write
            # to the table rewrite_media_cell_paths() is about to touch.
            # If this raises partway, the outer except below catches it
            # and returns before the rewrite and the save run at all --
            # whatever files were already copied are left on disk, not
            # cleaned up (models/output_copy.py's execute_output_copy
            # never deletes anything). Nothing to roll back for a copy
            # failure: no in-memory cell has moved yet.
            _execute_output_copy(plan)

            path_mapping = {
                _media_address_from_path(entry.src).path:
                    _media_address_from_path(entry.dst).path
                for entry in plan.entries
            }
            try:
                self._dataset.rewrite_media_cell_paths(path_mapping)
                self._dataset.save(project_path)
            except Exception:
                # Roll back whatever the rewrite touched -- self.
                # _project_paths has not moved (the swap below only runs
                # after Dataset.save() returns), so leaving a rewritten
                # cell in place here would point it at project_path/
                # outputs/... while self._project_paths still names the
                # OLD outputs_dir. A LATER successful save (to
                # project_path or anywhere else) plans its copy against
                # self._project_paths.outputs_dir, so a cell stranded
                # outside that folder would never be recognised as an
                # operator output again and could never be brought along.
                # This covers BOTH a rewrite that raised partway through
                # its own per-table loop (some tables rewritten, later
                # ones untouched) and a rewrite that fully succeeded but
                # was followed by a save() failure -- inverting the SAME
                # path_mapping is a no-op for any table the forward
                # rewrite never reached, since its cells still hold the
                # OLD values the inverse mapping's KEYS do not match.
                inverse_mapping = {
                    new: old for old, new in path_mapping.items()
                }
                if inverse_mapping:
                    self._dataset.rewrite_media_cell_paths(inverse_mapping)
                raise
            # Bind the artifact cache to this project's folder BEFORE the
            # index is written, so migration finishes first and every path
            # in the saved index names a file inside project_path/artifacts
            # (P0.5b-2ii-a). Main-thread only -- see set_artifacts_dir.
            self._store.set_artifacts_dir(project_path / "artifacts")
            # Re-root ProjectPaths at the same point, so run.paths.
            # outputs_dir for the NEXT operator run lands under the saved
            # project rather than the workspace it started in
            # (P1.9a). build_project_paths is pure -- this only swaps the
            # value the controller holds, same as set_artifacts_dir above.
            self._project_paths = build_project_paths(
                project_path, is_workspace=False
            )
            # Sweep the artifacts directory BEFORE save_index writes the
            # records (P0.5b-2ii-b2). The order is load-bearing:
            # reversed, save_index would serialize records naming files
            # the sweep is about to delete, and the reopened project
            # would report those pictures cached.
            self._store.reconcile_and_evict()
            self._store.save_index(project_path)
            # NOT: self._project_root = project_path. save() does not
            # rewrite the in-memory cells, so the base they resolve
            # against must not move either. load_project() sets the root
            # because load() does absolutise the cells against it. This is
            # unchanged by P1.9a: ProjectPaths (run.paths / the artifacts
            # cache) and _project_root (media cell resolution) answer
            # different questions and re-root on different conditions.
            #
            # Re-baseline the dirty flag now that every table has been
            # written to disk -- see has_unsaved_changes().
            self._unsaved_baseline = self._dataset.table_versions()
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to save project: {e}")
            return False

    def load_project(self, project_path: Path) -> None:
        """
        Loads a previously saved project from disk.

        Args:
            project_path: Path to an existing project folder.
        """
        try:
            # Load the dataset FIRST. Dataset.load() is atomic: on a bad
            # project it raises and leaves the currently open project
            # untouched. Nothing in the controller may move until it has
            # returned successfully -- otherwise a failed load would leave
            # _project_root pointing at the new folder while the dataset
            # still holds the old project, and media would resolve against
            # the wrong root. No controller step needs to run before it:
            # Dataset.load() takes project_path as an argument and reads no
            # controller state.
            self._dataset.load(project_path)
            # The dataset is now the new project: drop every in-flight run so
            # a late result cannot land in it, and re-root media resolution.
            # Guarded so live_runs_changed fires only on an actual change.
            if self._live_runs:
                self._live_runs.clear()
                self.live_runs_changed.emit()
            self._project_root = Path(project_path)
            # reset() BEFORE load_index(): otherwise the new project's
            # index lands on top of the previous project's live image
            # cache and fingerprint memo, and an old picture can appear
            # under a new row (docs/media_architecture.md section 4.5).
            self._store.reset()
            # Bind the artifact cache to this project's folder BEFORE
            # load_index() seeds the persisted paths (P0.5b-2ii-a). The
            # index is empty here (reset() cleared it), so this call just
            # re-roots the store and the codec -- the migration path only
            # does work on save.
            self._store.set_artifacts_dir(project_path / "artifacts")
            # Re-root ProjectPaths at the same point (P1.9a), so the next
            # operator run writes under the newly opened project rather
            # than wherever the previously open project (or the launch
            # workspace) pointed.
            self._project_paths = build_project_paths(
                Path(project_path), is_workspace=False
            )
            # load_index() re-seeds the index AND the fingerprint memo
            # from the persisted (size, mtime) values, so a project that
            # was fully thumbnailed reopens showing its cached pictures
            # with no paint-path decode.
            #
            # load_project() itself issues no thumbnail request -- it only
            # seeds the index and re-runs the query. A row the seeded
            # index does not cover (never thumbnailed before the save)
            # gets a request when its tile paints: render_column_value()
            # sees the cache miss and queues one (P0.5b-3i). A row that IS
            # in the seeded index is served as-is, even if its source has
            # changed since the save -- painting does not re-stat.
            # ArtifactStore.load_index()'s docstring is the authority on
            # what the seeded memo means for freshness; do not restate it
            # here.
            index_is_authoritative = self._store.load_index(project_path)
            # Reconcile the freshly-seeded index against the real
            # artifacts directory and evict orphaned / over-ceiling
            # JPEGs (P0.5b-2ii-b2). load_index() seeds entries from the
            # saved paths without checking the file is present; this
            # drops any entry whose JPEG is missing, so a demand request
            # is queued for that tile instead of it staying a permanent
            # grey placeholder. Skip the sweep when load_index() could
            # not read the index file: _index is empty then only because
            # reset() cleared it, and sweeping would delete every real
            # JPEG as an orphan over a recoverable error.
            if index_is_authoritative:
                self._store.reconcile_and_evict()
            self.tables_updated.emit(self._dataset.list_tables())
            self.columns_updated.emit(self.get_column_names())
            self._refresh_result()
            # Re-baseline the dirty flag against what was just loaded --
            # see has_unsaved_changes(). Dataset.load() mints fresh
            # table_versions() numbers for every table, so this must be
            # read back AFTER load() returns, never computed once at
            # startup (docs/review/unsaved-work-survey.md section 3).
            self._unsaved_baseline = self._dataset.table_versions()
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

    def get_column_names(self, table_name: str | None = None) -> list[str]:
        """
        Returns the column names the given table's TableSchema declares --
        the active table when *table_name* is None.

        P1.8d-2a: the schema, not the registry's column-name map, is the
        authority for what a table's columns are, so two tables that share
        a column name are no longer conflated. A table with no schema
        (only reachable by a test assigning straight into Dataset._tables)
        yields an empty list. row_id is schema-exempt and never appears.
        """
        name = table_name if table_name is not None else self._active_table
        schema = self._dataset.schema_for(name)
        if schema is None:
            return []
        return list(schema.column_names())

    def get_visual_column_names(self, table_name: str | None = None) -> list[str]:
        """
        Returns the given table's columns that render as visual output in
        tiles -- the active table when *table_name* is None.

        A column is visual when the registry maps its schema type tag to a
        ColumnType with visual=True.
        """
        name = table_name if table_name is not None else self._active_table
        schema = self._dataset.schema_for(name)
        if schema is None:
            return []
        visual: list[str] = []
        for spec in schema.columns:
            col_type = self._registry.type_for_tag(spec.type_tag)
            if col_type is not None and col_type.visual:
                visual.append(spec.name)
        return visual

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

    def _queue_thumbnail_request(
        self,
        row_id: str,
        canonical_address: str,
        source_path: str,
        table_name: str,
    ) -> None:
        """Queue a thumbnail/preview generation request for one already
        resolved media cell.

        Called on demand by render_column_value() when a tile paints and
        misses the cache (P0.5b-3i) -- there is no eager whole-table pass
        any more. The caller has already turned the cell into a canonical
        address (the cache key) and an absolute source path (where a
        worker reads the pixels), so this does not re-parse it.

        No exists() check: whether the source can be read is the worker's
        business. ArtifactStore._stat_fingerprint returns None for a
        missing or unreadable file and the job discards its subscribers
        without notifying, exactly as a failed decode does. Keeping the
        check off this path also keeps the paint thread from touching the
        filesystem at all. The trade is that a genuinely broken path is
        re-submitted as a (fast, failing) job each time its tile is
        repainted from scratch. A job for an address that scrolls off
        screen before a worker reaches it is dropped by
        _update_wanted_addresses() (P0.5b-3ii-b).
        """
        self._store.request_thumbnail(
            row_id, canonical_address, Path(source_path), table_name
        )

    def _resolve_media_cell(self, value) -> tuple[str, str]:
        """(canonical artifact-cache address, absolute source path) for a
        media cell, via media.media_address.resolve_source.

        One guarded call so the eager thumbnail sites and render_column_
        value share exactly one definition of "resolve this cell", and so
        renderers need neither a project root nor an address parser.
        Raises MediaAddressError for a value that is not an address.
        """
        return resolve_source(str(value), str(self._project_root))

    def get_artifact_pixmap(
        self,
        address: str,
        purpose: str = "thumbnail",
        resolution: int | None = None,
    ):
        """
        Returns a PIL Image for a derived artifact, or None.

        Identified by the media address the picture is of (P0.5b-1), not
        by a row -- the row, table and column name the UI subscriber, not
        the picture (docs/media_architecture.md section 4.5).

        Args:
            address:    Canonical media address (media_address.canonical_key).
            purpose:    'thumbnail' or 'preview'.
            resolution: Requested max side in pixels; defaults to the
                        standard resolution for the purpose.

        The representative-frame policy is the store's default ('first');
        a real choice arrives with 'midpoint' in P1.2.
        """
        return self._store.get_pixmap(address, purpose, resolution)

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
                         {'row_id': ..., 'column_name': ...,
                         'table_name': ...}. When 'table_name' is given
                         it names the table the calling tile is showing,
                         and a demand thumbnail request is queued under
                         it rather than under the controller's active
                         table.

        Returns:
            A QPixmap (thumbnail mode), QWidget (detail mode), or None.
        """
        ctx = dict(context) if context else {}

        # Resolve the column's display type tag from the schema of the
        # table the calling tile named (P1.8d-2a). Fall back to the active
        # table's schema when the caller named no table, or named a table
        # that has no schema -- so a stale or unknown table_name degrades
        # to the active table rather than losing the tag entirely. A tag
        # that stays None renders exactly as an unregistered column did:
        # render_by_tag returns a placeholder, never an exception.
        context_table = ctx.get("table_name")
        tag = None
        if context_table is not None:
            tag = self._schema_tag_for(column_name, context_table)
        if tag is None:
            tag = self._schema_tag_for(column_name, self._active_table)

        # For a media column, resolve the cell once here and hand the
        # renderer a canonical address (its artifact-cache key) and an
        # absolute source path. The renderer then needs no project root
        # and does no address parsing (docs/media_architecture.md 4.5).
        if (
            tag == "media_path"
            and isinstance(value, str)
            and value
        ):
            try:
                canonical_address, source_path = self._resolve_media_cell(value)
                ctx["canonical_address"] = canonical_address
                ctx["source_path"] = source_path
            except MediaAddressError:
                # The cell does not parse as a media address (e.g. a file
                # name with a literal '#'). In thumbnail mode there is
                # nothing more to do -- the renderer shows a placeholder
                # and no demand request can be keyed. Canonicalising such
                # cells at import is P1.8 (docs/known_defects.md). In
                # detail mode the renderer still treats the raw value as
                # a path.
                pass

            # Demand-driven thumbnail generation (P0.5b-3i). In thumbnail
            # mode the renderer is cache-or-placeholder and never decodes
            # a source, so this is the one place that notices the miss and
            # queues the work. The controller does it, not the renderer,
            # because only the controller knows the row and the table.
            # request_thumbnail() coalesces by address, so a still-pending
            # tile repainted several times adds no extra job. Detail mode
            # opens the source itself and needs no request.
            if (
                mode == "thumbnail"
                and "canonical_address" in ctx
                and ctx.get("row_id") is not None
                and not self._store.is_cached(ctx["canonical_address"])
            ):
                # The caller (a tile) names the table it is showing. Use
                # that, so the request and its ready notification are
                # attributed to the caller's table rather than to
                # whatever this controller's active table is now. Fall
                # back to the active table only when the caller did not
                # supply one.
                request_table = ctx.get("table_name") or self._active_table
                self._queue_thumbnail_request(
                    ctx["row_id"],
                    ctx["canonical_address"],
                    ctx["source_path"],
                    request_table,
                )

        return self._registry.render_by_tag(
            tag, value, size, mode, ctx, label=column_name
        )

    def render_result_image(
        self,
        artifact_path: str,
        size: int,
        mode: str = "detail",
    ):
        """
        Renders an operator's result artifact -- an image file the
        operator wrote -- for the detail panel.

        An operator result is a produced file, not a cell in any table, so
        it is rendered straight through the media type tag and consults no
        table's schema. This is the difference from render_column_value,
        whose tag comes from the row's own table: DetailWidget.show_result
        was calling render_column_value("full_path", ...) as a fiction, and
        P1.8d-2a broke it whenever the active table's schema had no
        'full_path' column (the tag resolved to None and the result image
        became a placeholder).

        The UI hands this a path and gets a widget back; it never learns
        what a tag is.

        A relative artifact_path is resolved against the project root the
        same way a media cell is, so an operator that writes into the
        project folder and returns a relative path still renders -- this
        matches what render_column_value("full_path", ...) did before.
        A result file has no ArtifactStore cache entry, so 'thumbnail'
        mode returns the media renderer's grey placeholder, not a scaled
        picture; the detail panel is the real caller.

        Args:
            artifact_path: Path to the image the operator produced.
            size:          Target size in pixels.
            mode:          'detail' (default) or 'thumbnail'.

        Returns:
            A QWidget (detail mode), a placeholder QPixmap (thumbnail
            mode), or a placeholder if the file cannot be rendered.
        """
        # Resolve a project-relative path to an absolute one, exactly as
        # the media-cell path does. A value that is not a well-formed
        # address (a stray '#') is handed to the renderer unchanged.
        try:
            _canonical_address, source_path = self._resolve_media_cell(
                artifact_path
            )
        except MediaAddressError:
            source_path = str(artifact_path)
        return self._registry.render_by_tag(
            "media_path", source_path, size, mode
        )

    def _schema_tag_for(
        self, column_name: str, table_name: str
    ) -> str | None:
        """The display type tag a table's TableSchema carries for a column,
        or None when the table has no schema or the schema does not name
        the column.

        P1.8d-2a: the schema, not the registry's column-name map, is the
        authority for what a named column is. Shared by render_column_value
        (a per-tile paint-path caller) and get_column_type, so it does one
        pass over the schema's columns -- spec_for raises KeyError for a
        name the schema does not carry -- rather than an `in` scan followed
        by a second lookup.
        """
        schema = self._dataset.schema_for(table_name)
        if schema is None:
            return None
        try:
            return schema.spec_for(column_name).type_tag
        except KeyError:
            return None

    def get_column_type(self, column_name: str, table_name: str | None = None):
        """
        Returns the ColumnType for a column of the given table -- the
        active table when *table_name* is None -- or None.

        P1.8d-2a: the column's type tag is read off that table's
        TableSchema and the registry is asked only what the tag renders
        as. A column the schema does not name returns None, exactly as an
        unregistered column did before.

        Args:
            column_name: The column whose type to look up.
            table_name:  The table containing the column. Defaults to the
                         active table.

        Returns:
            The ColumnType, or None if the table or column is not known.
        """
        name = table_name if table_name is not None else self._active_table
        tag = self._schema_tag_for(column_name, name)
        if tag is None:
            return None
        return self._registry.type_for_tag(tag)

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

    # ── Settings pass-throughs ───────────────────────────────────────
    #
    # The controller does not own settings and does not import settings/.
    # It holds a SettingsGateway (a plain-data face over the store) and
    # forwards these two calls to it. The dialog that will call these is
    # P0.5b-2ii-c2b2; there is no UI here yet.

    def get_settings_fields(self):
        """
        Returns the editable settings as a list of plain-data
        SettingField objects (see settings/settings_gateway.py).

        Raises:
            RuntimeError: if no settings gateway was wired in.
        """
        if self._settings_gateway is None:
            raise RuntimeError(
                "AppController has no settings gateway -- it was constructed "
                "without settings_gateway=, so settings cannot be read."
            )
        return self._settings_gateway.describe_fields()

    def apply_settings(self, values: dict) -> list[str]:
        """
        Validates and persists *values* through the settings gateway,
        then pushes an immediate-effect ceiling into the artifact store
        ONLY when that ceiling's stored value actually changed.

        "Actually changed" is measured against the gateway, not the
        caller's mapping: the value the gateway reports before the save
        is compared with the value it reports after. A caller can submit
        a number the gateway corrects straight back to what was already
        stored -- that pushes nothing, because pushing a ceiling is not
        free (the memory ceiling evicts, the disk ceiling runs a full
        artifacts-directory sweep, both on the main thread).

        When the disk ceiling was pushed and the resulting sweep deleted
        cached picture files, one plain-English sentence naming the count
        is appended to the returned list.

        Args:
            values: field name -> raw value mapping from the settings
                    dialog.

        Returns:
            A list of plain-English messages for the researcher: every
            correction message the gateway produced, plus at most one
            sentence reporting cached picture files the disk-ceiling
            sweep deleted. Empty when nothing was corrected and nothing
            swept.

        Raises:
            RuntimeError: if no settings gateway was wired in.
        """
        if self._settings_gateway is None:
            raise RuntimeError(
                "AppController has no settings gateway -- it was constructed "
                "without settings_gateway=, so settings cannot be applied."
            )

        # Read the ceilings the gateway reports BEFORE the save, so we can
        # tell a real change from a no-op the gateway corrected back.
        before = {
            field.name: field.current_value
            for field in self._settings_gateway.describe_fields()
        }

        problems = self._settings_gateway.save_values(values)

        # Read them again AFTER the save. Any push uses this number -- the
        # corrected one, never the raw request.
        after = {
            field.name: field.current_value
            for field in self._settings_gateway.describe_fields()
        }

        messages = list(problems)

        # Push the memory ceiling only if its stored value moved.
        if (
            before["picture_memory_max_bytes"]
            != after["picture_memory_max_bytes"]
        ):
            self._store.set_memory_cache_max_bytes(
                after["picture_memory_max_bytes"]
            )

        # Push the disk ceiling only if its stored value moved. When the
        # resulting sweep deleted files, report the count to the researcher.
        if (
            before["picture_disk_max_bytes"]
            != after["picture_disk_max_bytes"]
        ):
            sweep = self._store.set_disk_cache_max_bytes(
                after["picture_disk_max_bytes"]
            )
            # The sweep runs on any disk-ceiling change, not only a cut, and
            # its count folds in orphan cleanup as well as ceiling eviction --
            # so the sentence states plainly that files went, without claiming
            # a direction.
            if sweep.files_deleted > 0:
                # Singular/plural: this is researcher-facing text and
                # files_deleted is often 1.
                noun = (
                    "cached picture file"
                    if sweep.files_deleted == 1
                    else "cached picture files"
                )
                messages.append(
                    f"Applying the new disk cache limit deleted "
                    f"{sweep.files_deleted} {noun}."
                )

        return messages

    def get_output_copy_warning_threshold_bytes(self) -> int:
        """The byte threshold above which a Save-As output copy is called
        out to the researcher before it runs (ui/output_copy_warning.py,
        P1.9b-2). Same pass-through discipline as get_settings_fields() /
        apply_settings() above -- the controller does not import
        settings/ and only forwards to the gateway.

        Also editable through the settings dialog (P1.9b-2), as one of
        the fields get_settings_fields() lists -- this direct getter
        stays because the copy-on-save check only wants this one number.

        Raises:
            RuntimeError: if no settings gateway was wired in.
        """
        if self._settings_gateway is None:
            raise RuntimeError(
                "AppController has no settings gateway -- it was constructed "
                "without settings_gateway=, so settings cannot be read."
            )
        return self._settings_gateway.get_output_copy_warning_threshold_bytes()
