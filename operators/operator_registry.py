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
import queue
import threading
import time
import pandas as pd

from operators.base import BaseOperator, OperatorSetupError
from operators.descriptor import (
    ExecutionMode,
    MediaRequirement,
    ModelLifecycle,
    OperatorDescriptorError,
)
from operators.run_context import OperatorRunError
from media.extensions import IMAGE_EXTENSIONS, is_image_path
from media.media_address import MediaAddressError, parse as parse_address
from media.resolver import MediaResolverError
from models.table_schema import ColumnHint, ColumnRole


def _is_bare_video_address(addr) -> bool:
    """True for an address with no #t=/#f= selector at all -- decision 4's
    "the whole file" -- whose path does not look like a still image
    (media.extensions.IMAGE_EXTENSIONS is the one authoritative
    image/video split, shared with media/resolver.py's own dispatch). A
    FRAME operator declares that it reads a single, addressed frame; a
    bare video address does not name one, so the runner refuses the row
    rather than silently defaulting to that video's first frame.
    """
    is_bare = (
        addr.frame is None
        and addr.time_us is None
        and addr.time_range_us is None
    )
    if not is_bare:
        return False
    return Path(addr.path).suffix.lower() not in IMAGE_EXTENSIONS


def _unreadable_media_row_error(row_id: str, full_path, exc: Exception) -> tuple:
    """The row_errors tuple for a still-image or #t= row whose
    resolve_frame() call raised. Shared by the serial per-row loop and
    the parallel path's producer -- one place decides what this failure
    is called and how its message reads, so it cannot silently drop the
    row in one path while reporting it in the other. Previously this row
    was printed and skipped with no row error at all, in both paths: the
    row simply had no result and nothing told the researcher why.
    """
    return (
        row_id, "UnreadableMedia",
        f"could not read the media at {full_path!r}: "
        f"{type(exc).__name__}: {exc}",
    )


def _iter_group_rows(run, addresses, media_by_address_id):
    """(row_id, media, metadata) for one per-source group, in ASCENDING
    ORDINAL order -- the `rows` iterable operators.base.BaseOperator.
    iter_column_updates() declares (P2.1). `addresses` are every #f=
    address of this group, in the order they were classified (not
    necessarily ordinal order); `run.resolver.decode_frames_in_order()`
    does the one sequential decode and re-orders them by presentation
    time. `media_by_address_id` maps `id(address)` (object identity,
    not value equality -- decode_frames_in_order's own docstring: this
    is what tells two rows naming the exact same frame apart) back to
    the (row_id, metadata) that address was classified under, in
    _run_create_columns_worker's per-row loop above.
    """
    for addr, payload in run.resolver.decode_frames_in_order(
        addresses, "analysis"
    ):
        row_id, metadata = media_by_address_id[id(addr)]
        yield row_id, payload.pixels, metadata


def _classify_row_result(operator, row_id: str, media, metadata: dict, run):
    """Call operator.create_columns() for one row and classify what
    happened -- the exception mapping the serial `_deliver` closure and
    the parallel runner's consumer threads both need, factored out so
    there is exactly ONE place that decides what each exception means.
    Pure: no side effect on any shared state, so it is safe to
    call from any thread. Returns one of:

        ("ok", result_dict)
        ("row_error", exc_type_name, message, all_none_dict)
        ("abort_silent", message)        -- NotImplementedError
        ("abort_setup_error", message)   -- OperatorSetupError

    RECORDING what a tag means (appending to row_errors, calling
    on_item_complete, incrementing emitted, reporting progress) is a
    separate concern and must happen only on the run's coordinator
    thread -- see _run_create_columns_worker's `_deliver` and
    `_run_create_columns_parallel`'s `_record`, the two callers of this
    function.
    """
    try:
        result = operator.create_columns(row_id, media, metadata, run)
    except NotImplementedError:
        return (
            "abort_silent",
            f"Operator '{operator.name}' does not implement "
            f"create_columns().",
        )
    except OperatorSetupError as e:
        return ("abort_setup_error", str(e))
    except Exception as e:
        all_none = {
            column.name: None
            for column in run.spec.mode_descriptor.output.columns
        }
        return ("row_error", type(e).__name__, str(e), all_none)
    return ("ok", result)


def _log_run_cost(
    operator_name: str, rows: int, elapsed_seconds: float, worker_count: int
) -> None:
    """The one-line per-run cost summary CLAUDE.md's threading rule
    requires: how many rows, how long, the per-row wall cost, and
    how many worker threads actually did the work (1 for the serial
    path, however it was reached)."""
    ms_per_row = (elapsed_seconds / rows * 1000) if rows else 0.0
    print(
        f"[OperatorRegistry] '{operator_name}': {rows} rows in "
        f"{elapsed_seconds:.3f} s, {ms_per_row:.2f} ms/row wall, "
        f"{worker_count} workers"
    )


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

        Raises:
            OperatorDescriptorError: a TABLE mode's OutputSpec.table_columns
                names a role that is not a models.table_schema.ColumnRole
                member (P1.7-1 fix round). Checked HERE rather than left to
                surface later from hints_for_table_output()'s ColumnRole(...)
                conversion, because registration is the first place both
                operators.descriptor's plain-string vocabulary and
                models.table_schema's ColumnRole are already in scope --
                catching it here fails fast, before any run, rather than
                after a completed TABLE run is silently discarded.
        """
        self._validate_table_column_roles(operator)
        self._operators[operator.name] = operator
        print(f"[OperatorRegistry] Registered: {operator.name}")

    def _validate_table_column_roles(self, operator: BaseOperator) -> None:
        """Raise OperatorDescriptorError if any mode's OutputSpec.table_columns
        declares a role string that names no ColumnRole member. The valid
        set is read off the enum itself (``{r.value for r in ColumnRole}``),
        never hand-written, so it cannot drift from ColumnRole as members
        are added or renamed.
        """
        if operator.descriptor is None:
            return
        valid_roles = {role.value for role in ColumnRole}
        for mode_descriptor in operator.descriptor.modes:
            for column in mode_descriptor.output.table_columns:
                if column.role is not None and column.role not in valid_roles:
                    raise OperatorDescriptorError(
                        f"operator {operator.name!r} declares "
                        f"table_columns[{column.name!r}].role="
                        f"{column.role!r}, which is not a valid "
                        f"ColumnRole; valid roles are "
                        f"{sorted(valid_roles)}"
                    )

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

    def hints_for_table_output(self, operator_name: str) -> dict[str, ColumnHint]:
        """The ColumnHint (P1.7-1) each column a TABLE-mode operator
        declares on its descriptor's OutputSpec.table_columns asks for --
        type_tag, role and/or carry_to_children -- keyed by column name,
        in exactly the shape Dataset.create_table_from_df's ``hints``
        argument expects.

        type_tag is always carried (OutputColumn.type_tag is a required,
        non-empty field, so every declared table_columns entry has an
        opinion about it -- unlike role and carry_to_children, which are
        optional). This mirrors what a COLUMNS mode's own OutputColumn.type_tag
        already does unconditionally (CLAUDE.md's "An operator's declared
        output-column tag reaches its table's schema as a ColumnHint,
        registered or not") -- P1.7-1 fixed a bug here: the first cut of
        this method carried role/carry_to_children but silently dropped
        type_tag, so a column declared media_path was stored as text.

        AppController calls this once a create_table() result is ready to
        store, and hands the result straight to create_table_from_df; the
        operator itself never calls Dataset (CLAUDE.md, "Data ownership"),
        so this translation from descriptor vocabulary (OutputColumn's
        plain-string role) to Dataset's vocabulary (ColumnRole) happens
        here, the one place both are already in scope. ColumnRole(...) is
        never expected to raise here: register() already refused an
        operator whose table_columns names an invalid role, before this
        method can ever be reached for it.

        Returns {} -- "no hints", exactly Dataset's own default -- for an
        unknown operator, one with no descriptor, no TABLE mode, or a
        TABLE mode that declares no table_columns. None of those are
        errors: a TABLE operator that declares none (every one of them,
        before P1.7-1) behaves exactly as before.
        """
        operator = self._operators.get(operator_name)
        if operator is None or operator.descriptor is None:
            return {}
        mode_descriptor = operator.descriptor.mode_for(ExecutionMode.TABLE)
        if mode_descriptor is None:
            return {}

        hints: dict[str, ColumnHint] = {}
        for column in mode_descriptor.output.table_columns:
            hints[column.name] = ColumnHint(
                type_tag=column.type_tag,
                role=ColumnRole(column.role) if column.role is not None else None,
                carry_to_children=column.carry_to_children,
            )
        return hints

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
        media_column: str | None = None,
        worker_count: int = 1,
    ) -> bool:
        """
        Runs create_columns() over an ordered group of rows in a
        background thread. AppController takes one snapshot of the
        selected rows (Dataset.snapshot_rows) on the main thread before
        calling this method, so the worker never reads from Dataset
        directly and never receives one dict per row built in advance.

        worker_count: defaults to 1, which always takes today's
        serial path unchanged -- every existing caller that does not pass
        this stays exactly as it was. A value of 2 or more is honoured
        ONLY when the run is also mode COLUMNS (this method's only mode),
        model_lifecycle PER_WORKER, and the operator does not override
        iter_column_updates; every other combination still runs serial.
        AppController reads the operator_worker_count setting and passes
        it in -- this registry never reads settings itself.

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
            media_column:  The active table's column name to read a FRAME
                           (or ADDRESS) row's media cell from -- decided by
                           AppController (AppController.get_detail_media_
                           column(), CLAUDE.md's media rule), never
                           hardcoded here. None for a run whose
                           media_requirement never reads a media cell
                           (METADATA), or for a direct call that bypasses
                           AppController (a FRAME-requirement run then
                           treats every row as having no media cell, the
                           same outcome an unresolved column name always
                           produced).
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
                              raised an unexpected exception, OR were
                              refused before create_columns() was ever
                              called -- a FRAME row whose media cell is
                              missing/blank ("MissingMedia"), unparseable
                              ("UnparseableMedia"), or a whole video where
                              a single addressed frame was expected
                              ("WholeVideoRow"); see the media_requirement
                              branch below. Lets the controller surface a
                              single end-of-run summary to the user so
                              unexpected failures and deliberate refusals
                              are both visibly distinct from the normal
                              "no face detected" case, which returns None
                              values rather than landing here. Also
                              includes, appended after the above (P1.7-1
                              fix round), every row create_columns()
                              itself reported through
                              run.report_row_error() -- the same channel
                              a TABLE-mode create_table() uses; both
                              sources keep their own report order, merged
                              rather than interleaved.
                              Signature: (operation_id: str,
                                          label: str,
                                          errors: list[tuple[str, str, str]])
                              Each tuple is (row_id, kind, message) --
                              `kind` is an exception type name for a
                              caught exception, or whatever string the
                              operator itself passed to
                              report_row_error().

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
                on_setup_error, on_row_errors, media_column, worker_count,
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
        media_column: str | None = None,
        worker_count: int = 1,
    ) -> None:
        """
        Worker that runs create_columns() in the background thread. This
        thread is the run's COORDINATOR: whether it takes the serial path
        below or delegates to _run_create_columns_parallel, it is
        the only thread that ever touches row_errors, calls
        on_item_complete / on_setup_error / on_row_errors, or reports
        progress. Builds each row's metadata dict from the pre-snapshotted
        DataFrame as it reaches that row — never reads Dataset, and
        never receives 530,000 dicts built in advance.

        A FRAME row whose media names a single frame of a VIDEO (a #f=
        address, decision 8) is not decoded here directly -- it is
        collected into a per-source group and decoded further down, in
        ONE ordered pass per source through
        MediaResolver.decode_frames_in_order(), fed to
        operator.iter_column_updates() (P2.1, operators/CLAUDE.md's
        "sequential pass" rule). Every other FRAME row -- a still image,
        or a #t= time point -- keeps the ORIGINAL per-row resolve_frame()
        path immediately below, unchanged.
        """
        start_time = time.perf_counter()
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
        # How many rows have reached a progress-worthy outcome so far --
        # a successful delivery, or an unexpected create_columns()
        # exception caught and reported (the two cases that, in the
        # per-row loop below, fall through to a progress update; a row
        # refused before decode, or whose decode failed, does not). A
        # row deferred into a per-source group (below) advances this
        # when IT is delivered, not when it is merely classified into a
        # group -- `total` still means "how many rows this run covers",
        # only a deferred row's share of it lands later than its
        # position in row_ids would suggest.
        progress_count = 0
        # Every #f= row on a video, deferred out of the loop below and
        # decoded afterwards in one ordered pass per (path, stream)
        # source -- see the FRAME branch's own note.
        frame_groups: "dict[tuple, list[tuple[str, object, dict]]]" = {}
        # Set by _deliver() returning False (NotImplementedError /
        # OperatorSetupError): the run is aborted, and no row after the
        # one that raised -- in the per-row loop or in a later group --
        # is ever processed, exactly as the original single loop's own
        # `break` stopped it outright.
        aborted = False

        def _report_progress() -> None:
            nonlocal progress_count
            progress_count += 1
            if on_progress is not None:
                on_progress(int(progress_count / total * 100))

        def _deliver(row_id: str, media, metadata: dict) -> bool:
            """Calls operator.create_columns() for one row (through the
            shared _classify_row_result mapping) and RECORDS the
            outcome -- the exact exception handling the per-row loop below
            had inline before P2.1, factored out so the grouped (ordered)
            path can share it. This serial worker thread is its own
            coordinator, so recording happens right here, inline; the
            parallel runner's own _record (below) is the same recording
            logic, on ITS coordinator thread, for an outcome a consumer
            thread computed. Returns False to ABORT THE ENTIRE RUN
            (NotImplementedError, OperatorSetupError), True otherwise --
            whether the row succeeded or the operator raised an
            unexpected exception (reported as a row error, not an
            abort)."""
            nonlocal emitted
            outcome = _classify_row_result(operator, row_id, media, metadata, run)
            kind = outcome[0]
            if kind == "abort_silent":
                print(
                    f"[OperatorRegistry] Operator '{operator.name}' "
                    f"does not implement create_columns()."
                )
                return False
            if kind == "abort_setup_error":
                message = outcome[1]
                # Setup-level failure (e.g. required model file missing).
                # Abort the run rather than spamming the same error per row.
                print(
                    f"[OperatorRegistry] Setup error in '{operator.name}': "
                    f"{message}"
                )
                if on_setup_error is not None:
                    # Pass the operator's own display label so the
                    # controller never rebuilds it from a worker thread.
                    on_setup_error(operation_id, label, message)
                return False
            if kind == "row_error":
                # Unexpected per-row failure (mediapipe crash, bug, malformed
                # image, etc.). Mark the row as missing for consistency with
                # the operator's no-face path, and remember it so we can
                # surface a single summary at the end of the run.
                _, exc_kind, message, all_none = outcome
                print(
                    f"[OperatorRegistry] Unexpected error in '{operator.name}' "
                    f"on {row_id}: {exc_kind}: {message}"
                )
                row_errors.append((row_id, exc_kind, message))
                if on_item_complete is not None:
                    on_item_complete(operation_id, table_name, row_id, all_none)
                    emitted += 1
                return True
            # "ok"
            _, result = outcome
            if on_item_complete is not None:
                on_item_complete(operation_id, table_name, row_id, result)
                emitted += 1
            return True

        # What the runner decodes for each row is decided by this run's
        # declared media_requirement (operators/descriptor.py ->
        # MediaRequirement), read off the run's mode descriptor -- not by a
        # boolean flag on the operator. The five declared values and what
        # each one means for this COLUMNS runner:
        #
        #   FRAME    -- the operator needs one decoded frame. A still image
        #               or a #t= time point is decoded per-row, below,
        #               exactly as before; a #f= address on a video is
        #               grouped by source and decoded in order, further
        #               down (P2.1).
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
                elapsed_seconds = time.perf_counter() - start_time
                _log_run_cost(operator.name, emitted, elapsed_seconds, 1)
                on_complete(
                    operation_id, operator.name, emitted,
                    elapsed_seconds=elapsed_seconds, worker_count=1,
                )
            return

        # FRAME is the only requirement that makes the runner decode.
        needs_frame = media_requirement is MediaRequirement.FRAME

        # Whether this run is eligible for the parallel path,
        # decided BEFORE any model is built -- model_lifecycle PER_WORKER,
        # the operator does not override iter_column_updates() (an
        # overriding operator may carry cross-frame state, which the
        # parallel consumers' independent per-row calls cannot honour),
        # and worker_count >= 2. worker_count == 1 always takes the
        # serial path below, unchanged -- see run_create_columns's
        # docstring. CLAUDE.md's threading rule: only the coordinator
        # thread (this one) ever touches row_errors / on_item_complete /
        # on_setup_error / progress, whichever path is taken.
        model_lifecycle = run.spec.mode_descriptor.model_lifecycle
        uses_default_iter_column_updates = (
            type(operator).iter_column_updates is BaseOperator.iter_column_updates
        )
        if (
            model_lifecycle is ModelLifecycle.PER_WORKER
            and uses_default_iter_column_updates
            and worker_count >= 2
        ):
            self._run_create_columns_parallel(
                operator, snapshot, row_ids, table_name, run, operation_id,
                on_item_complete, on_progress, on_complete, on_setup_error,
                on_row_errors, media_column, needs_frame, label,
                worker_count, start_time,
            )
            return

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
                elapsed_seconds = time.perf_counter() - start_time
                _log_run_cost(operator.name, emitted, elapsed_seconds, 1)
                on_complete(
                    operation_id, operator.name, emitted,
                    elapsed_seconds=elapsed_seconds, worker_count=1,
                )
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
                elapsed_seconds = time.perf_counter() - start_time
                _log_run_cost(operator.name, emitted, elapsed_seconds, 1)
                on_complete(
                    operation_id, operator.name, emitted,
                    elapsed_seconds=elapsed_seconds, worker_count=1,
                )
            return

        for i, row_id in enumerate(row_ids):
            # P1.12f-3: check for cancellation BETWEEN rows, never mid-row
            # -- an operator's own row work is its own business (CLAUDE.md,
            # "Long-running work"). Every result already handed to
            # on_item_complete for an earlier row stays in it; this only
            # stops the loop from starting the next one. `emitted` ends up
            # smaller than `total`, and AppController._run_outcome()
            # separately checks the same token to record this run
            # "partial" once its completion arrives.
            if run.cancelled():
                break
            metadata = snapshot.iloc[i].to_dict()

            # FRAME: decode one frame through the shared resolver and
            # skip the row if it will not load -- or, for a #f= address
            # on a video, defer it into a per-source group, decoded
            # further down in one ordered pass (P2.1). METADATA /
            # ADDRESS: hand the operator None. `media_column` names
            # which of this table's columns holds the media cell --
            # AppController.run_create_columns decides it
            # (AppController.get_detail_media_column(); never a literal
            # "full_path" -- that hardcoding was docs/known_defects.md's
            # "A FRAME-requirement COLUMNS run reads only the full_path
            # metadata key", fixed by this same item) and the cell
            # arrives here ALREADY ABSOLUTE: AppController resolves
            # every FRAME/ADDRESS run's media-tagged columns with the
            # same method the display path uses (_resolve_media_cell)
            # before this worker ever starts -- this registry has no
            # controller access and cannot do that resolution itself,
            # and run.paths (project directories, for output-writing
            # operators) is the WRONG base for a stored media cell:
            # after Save As it re-roots to the new folder while cell
            # resolution deliberately does not (AppController.
            # save_project's note on _project_root). resolve_frame
            # refuses a still-relative address with MediaAddressError,
            # caught below like any other decode failure.
            if needs_frame:
                full_path = metadata.get(media_column, "")

                # A missing/blank cell and an unparseable one are, like
                # the whole-video refusal below, not decode failures --
                # there is no address to even try resolving yet -- but
                # they are just as invisible to the researcher as a
                # bare print() would leave them. Routed through the
                # same row_errors channel, each under its own reason,
                # rather than folded into "WholeVideoRow" (an empty
                # path also satisfies decision 4's "whole file" shape,
                # but telling the researcher their row is missing
                # media is a different, more accurate answer than
                # telling them it names a whole video).
                if not full_path or pd.isna(full_path):
                    row_errors.append((
                        row_id, "MissingMedia",
                        "this row has no media value to read",
                    ))
                    continue
                try:
                    addr = parse_address(full_path)
                except MediaAddressError as e:
                    row_errors.append((
                        row_id, "UnparseableMedia",
                        f"could not parse the media value {full_path!r}: {e}",
                    ))
                    continue
                if _is_bare_video_address(addr):
                    # Not a decode failure -- a deliberate refusal, but
                    # the researcher must still be told: routed through
                    # the same row_errors channel an unexpected
                    # create_columns() exception uses below, so the
                    # end-of-run summary (AppController._on_operator_
                    # complete's "row_errors" branch) reports the count
                    # and this exact reason, and the run's provenance
                    # records outcome "partial" rather than "complete"
                    # -- a print() alone reaches nobody but a developer
                    # watching the console.
                    message = (
                        "this operator reads single frames; this row "
                        "is a whole video"
                    )
                    row_errors.append((row_id, "WholeVideoRow", message))
                    continue

                # A #f=<n> address on a VIDEO is the ordered path's case
                # (P2.1): grouped by source (path, stream) and decoded
                # in one sequential pass below, instead of a seek per
                # row. A still image's own #f=0 (FrameOperator emits one
                # for every image row) and a #t= time point keep the
                # ORIGINAL per-row resolve_frame() path immediately
                # below -- a still has no sequential stream to walk, and
                # a lone time point gains nothing from grouping.
                if addr.frame is not None and not is_image_path(addr.path):
                    group_key = (addr.path, addr.stream)
                    frame_groups.setdefault(group_key, []).append(
                        (row_id, addr, metadata)
                    )
                    continue

                try:
                    media = run.resolver.resolve_frame(
                        addr, "analysis"
                    ).pixels
                except (MediaResolverError, MediaAddressError, OSError) as e:
                    print(
                        f"[OperatorRegistry] Could not load image "
                        f"for {row_id}: {full_path} ({e})"
                    )
                    row_errors.append(
                        _unreadable_media_row_error(row_id, full_path, e)
                    )
                    continue
            else:
                media = None

            if not _deliver(row_id, media, metadata):
                aborted = True
                break
            _report_progress()

        # Whether the operator overrides iter_column_updates() decides how
        # a per-row exception inside a group is handled, not just how the
        # rows are iterated:
        #
        #   NOT overridden (every operator today) -- the runner iterates
        #   the decoded rows itself and calls _deliver() per row, exactly
        #   as the per-row loop above does. A row's own exception is then
        #   _deliver's problem, not the group's: that ONE row gets a row
        #   error and an all-None result, and the NEXT FRAME of the same
        #   source is still attempted -- a review finding on a real
        #   10-minute clip found the group-level catch below turning one
        #   bad frame into every later frame of that source lost, which
        #   is not what the per-row path ever did for the same exception.
        #   This is safe specifically because BaseOperator's own
        #   iter_column_updates carries no state across the rows it
        #   yields (operators/base.py's own docstring) -- there is
        #   nothing an exception could leave corrupted for the next row.
        #
        #   Overridden -- an operator that tracks something across
        #   frames (P2.4) may have left that state unreliable after a
        #   failure partway through its own generator, so the whole
        #   group still ends on any exception, the pre-this-fix
        #   behaviour, kept only for this case. (uses_default_iter_column_
        #   updates was already computed above, before the
        #   parallel-eligibility check, which needs the same value.)

        # The ordered path (P2.1): every #f= row on a video, grouped by
        # source above, decoded here in ONE sequential pass per source
        # through MediaResolver.decode_frames_in_order() -- never one
        # seek per row. Skipped entirely once `aborted` already ended the
        # run above, exactly as the original loop's own `break` would
        # have skipped every row after the one that aborted it.
        if not aborted:
            for (_source_path, _stream), entries in frame_groups.items():
                if run.cancelled():
                    break
                addresses = [addr for (_rid, addr, _md) in entries]
                media_by_address_id = {
                    id(addr): (row_id, metadata)
                    for (row_id, addr, metadata) in entries
                }
                pending_row_ids = {row_id for (row_id, _a, _m) in entries}

                if uses_default_iter_column_updates:
                    try:
                        for row_id, media, metadata in _iter_group_rows(
                            run, addresses, media_by_address_id
                        ):
                            if run.cancelled():
                                break
                            pending_row_ids.discard(row_id)
                            if not _deliver(row_id, media, metadata):
                                aborted = True
                                break
                            _report_progress()
                    except Exception as e:
                        # A DECODE failure partway through this source (a
                        # damaged file, a stale frame-time index) -- an
                        # operator's OWN per-row exception never reaches
                        # here, _deliver() already handled it above and
                        # moved on to the next frame. Reports every row
                        # of THIS group not yet delivered, with one
                        # shared reason, and the run continues with the
                        # next group. Results already delivered for this
                        # group (and every earlier one) are kept.
                        print(
                            f"[OperatorRegistry] Ordered decode failed for "
                            f"'{operator.name}' on {_source_path!r}: "
                            f"{type(e).__name__}: {e}"
                        )
                        for row_id in pending_row_ids:
                            row_errors.append((row_id, type(e).__name__, str(e)))
                        continue
                    if aborted:
                        break

                else:
                    # Overridden iter_column_updates(): unchanged from
                    # before this fix -- any exception, from decode or
                    # from the operator's own cross-frame state, ends
                    # the whole group.
                    try:
                        updates = operator.iter_column_updates(
                            _iter_group_rows(run, addresses, media_by_address_id),
                            run,
                        )
                        for row_id, result in updates:
                            pending_row_ids.discard(row_id)
                            if on_item_complete is not None:
                                on_item_complete(
                                    operation_id, table_name, row_id, result
                                )
                                emitted += 1
                            _report_progress()
                    except NotImplementedError:
                        print(
                            f"[OperatorRegistry] Operator '{operator.name}' "
                            f"does not implement iter_column_updates() (or "
                            f"create_columns())."
                        )
                        aborted = True
                        break
                    except OperatorSetupError as e:
                        print(
                            f"[OperatorRegistry] Setup error in "
                            f"'{operator.name}': {e}"
                        )
                        if on_setup_error is not None:
                            on_setup_error(operation_id, label, str(e))
                        aborted = True
                        break
                    except Exception as e:
                        # A failure anywhere in this group's ordered
                        # decode or processing -- a damaged file, a
                        # stale frame-time index, an unexpected operator
                        # error -- reports every row of THIS group not
                        # yet delivered, with one shared reason
                        # (CLAUDE.md: "one bad row must not kill a run",
                        # extended here to one bad source), and the run
                        # continues with the next group. Results already
                        # delivered for this group (and every earlier
                        # one) are kept. This override case cannot
                        # narrow the failure to one row the way the
                        # non-overridden branch above does -- its own
                        # cross-frame state may no longer be
                        # trustworthy.
                        print(
                            f"[OperatorRegistry] Ordered decode failed for "
                            f"'{operator.name}' on {_source_path!r}: "
                            f"{type(e).__name__}: {e}"
                        )
                        for row_id in pending_row_ids:
                            row_errors.append((row_id, type(e).__name__, str(e)))
                        continue

        # P1.7-1 fix round: run.report_row_error() is available on every
        # OperatorRun, not just a TABLE run's -- a create_columns() that
        # calls it (a reasonable thing to try, since the method makes no
        # mode distinction) must not have that report silently dropped.
        # Drained here and appended AFTER the exceptions this loop caught
        # itself, preserving each source's own report order; nothing
        # before this line is reordered by the merge.
        row_errors = row_errors + run.collected_row_errors()

        if row_errors and on_row_errors is not None:
            on_row_errors(operation_id, label, row_errors)

        if on_complete is not None:
            elapsed_seconds = time.perf_counter() - start_time
            _log_run_cost(operator.name, emitted, elapsed_seconds, 1)
            on_complete(
                operation_id, operator.name, emitted,
                elapsed_seconds=elapsed_seconds, worker_count=1,
            )

    # ── _run_create_columns_parallel ─────────────────────────────────────

    def _run_create_columns_parallel(
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
        media_column: str | None,
        needs_frame: bool,
        label: str,
        worker_count: int,
        start_time: float,
    ) -> None:
        """
        The parallel COLUMNS path. Reached only from
        _run_create_columns_worker's eligibility check (model_lifecycle
        PER_WORKER, the operator does not override iter_column_updates,
        worker_count >= 2) -- every other run takes the serial path
        above, unchanged.

        Shape: ONE producer thread reproduces today's serial per-row
        decode/classification exactly (grouping and ordered decode for a
        #f= row on a video; per-row resolve_frame for a still image or a
        #t= point; media=None for METADATA/ADDRESS), and feeds
        (row_id, media, metadata) into a bounded work queue instead of
        calling _deliver itself. N CONSUMER threads each get their OWN
        PER_WORKER model (built once, before any thread starts) and call
        operator.create_columns() through the same _classify_row_result
        mapping _deliver uses, handing the classified outcome back
        through a results queue. THIS thread -- the coordinator, the
        same thread _run_create_columns_worker is already running on --
        is the only thread that ever appends to row_errors, calls
        on_item_complete / on_setup_error, or reports progress
        (CLAUDE.md's threading rule). Consumers only compute.

        Fallback: if a consumer thread dies from something other than a
        per-row exception (_classify_row_result already turns every
        ordinary create_columns() failure into a row_error or an abort
        outcome, so this is BaseException escaping that mapping, or a bug
        in the consumer loop itself), the coordinator reprocesses that
        row immediately, with its own separately built model, and logs
        one line saying so. If EVERY consumer dies this way (and nothing
        else independently aborted the run), the coordinator takes over
        as the sole remaining consumer until the producer itself stops --
        so every row still gets done, including ones the producer had
        not queued yet at the moment of the last death.

        The run never hangs. Every work_queue.put() the producer makes
        goes through _try_enqueue, which retries on a bounded timeout and
        gives up (stopping the producer entirely) once run.cancelled() or
        abort_event is set -- a plain blocking put() here could wait
        forever on a queue nobody is left to drain, which would in turn
        hang this method's own producer_thread.join() below, so
        on_complete would never fire and the run would stay "live"
        forever. That join is itself bounded as a second line of defence.
        Every consumer polls the same two flags on its own timeout rather
        than blocking on the queue forever either, so nothing here can
        wait on a peer that is never coming back.
        """
        total = len(row_ids)
        row_errors: list[tuple[str, str, str]] = []
        emitted = 0
        progress_count = 0
        aborted = False
        fallback_run_holder: list = []  # 0 or 1 OperatorRun, built lazily

        def _report_progress() -> None:
            nonlocal progress_count
            progress_count += 1
            if on_progress is not None:
                on_progress(int(progress_count / total * 100))

        def _record(row_id: str, outcome: tuple) -> bool:
            """Coordinator-only recording -- see _deliver's docstring
            above for why this is the same logic, just reached from a
            queued outcome instead of a direct call. Returns False to
            abort the run."""
            nonlocal emitted
            kind = outcome[0]
            if kind == "abort_silent":
                print(
                    f"[OperatorRegistry] Operator '{operator.name}' "
                    f"does not implement create_columns()."
                )
                return False
            if kind == "abort_setup_error":
                message = outcome[1]
                print(
                    f"[OperatorRegistry] Setup error in '{operator.name}': "
                    f"{message}"
                )
                if on_setup_error is not None:
                    on_setup_error(operation_id, label, message)
                return False
            if kind == "row_error":
                _, exc_kind, message, all_none = outcome
                print(
                    f"[OperatorRegistry] Unexpected error in '{operator.name}' "
                    f"on {row_id}: {exc_kind}: {message}"
                )
                row_errors.append((row_id, exc_kind, message))
                if on_item_complete is not None:
                    on_item_complete(operation_id, table_name, row_id, all_none)
                    emitted += 1
                _report_progress()
                return True
            # "ok"
            _, result = outcome
            if on_item_complete is not None:
                on_item_complete(operation_id, table_name, row_id, result)
                emitted += 1
            _report_progress()
            return True

        def _get_fallback_run():
            # Built lazily -- only paid for if a consumer actually dies --
            # and only once, reused for every row the fallback processes.
            if not fallback_run_holder:
                fallback_run_holder.append(
                    dataclasses.replace(run, model=operator.build_model())
                )
            return fallback_run_holder[0]

        # Build every consumer's own model BEFORE any thread starts -- the
        # same "abort before any row is processed" guarantee the serial
        # path gives (see its own model-build comment above). Building
        # sequentially, here, on the coordinator, means a build failure
        # needs no thread teardown: nothing else is running yet.
        models = []
        for _ in range(worker_count):
            try:
                models.append(operator.build_model())
            except Exception as e:
                if isinstance(e, OperatorSetupError):
                    message = str(e)
                else:
                    message = f"could not build its model: {type(e).__name__}: {e}"
                print(
                    f"[OperatorRegistry] Model build failed for "
                    f"'{operator.name}': {type(e).__name__}: {e}"
                )
                if on_setup_error is not None:
                    on_setup_error(operation_id, label, message)
                if on_complete is not None:
                    elapsed_seconds = time.perf_counter() - start_time
                    _log_run_cost(operator.name, emitted, elapsed_seconds, worker_count)
                    on_complete(
                        operation_id, operator.name, emitted,
                        elapsed_seconds=elapsed_seconds, worker_count=worker_count,
                    )
                return

        _SENTINEL = object()
        _ENQUEUE_POLL_SECONDS = 0.5
        _JOIN_TIMEOUT_SECONDS = 5.0
        work_queue: "queue.Queue" = queue.Queue(maxsize=2 * worker_count)
        results_queue: "queue.Queue" = queue.Queue()
        abort_event = threading.Event()

        def _try_enqueue(item) -> bool:
            """Put item onto work_queue, retrying on a bounded timeout
            instead of blocking forever. A plain blocking put() here
            could wait on a queue nobody will ever drain again (every
            consumer already stopped for cancellation, an aborting
            outcome, or having died) -- which would leave this producer
            thread stuck forever, which would in turn hang the
            coordinator's own producer_thread.join() below, so
            on_complete would never fire and the run would stay "live"
            forever. Returns False if it gave up without
            enqueueing, in which case the caller stops producing
            entirely -- there is no point trying the next item either.
            """
            while True:
                if run.cancelled() or abort_event.is_set():
                    return False
                try:
                    work_queue.put(item, timeout=_ENQUEUE_POLL_SECONDS)
                    return True
                except queue.Full:
                    continue

        def _producer() -> None:
            frame_groups: "dict[tuple, list[tuple[str, object, dict]]]" = {}
            for i, row_id in enumerate(row_ids):
                if run.cancelled() or abort_event.is_set():
                    return
                metadata = snapshot.iloc[i].to_dict()
                if needs_frame:
                    full_path = metadata.get(media_column, "")
                    if not full_path or pd.isna(full_path):
                        results_queue.put((
                            "producer_row_error", row_id, "MissingMedia",
                            "this row has no media value to read",
                        ))
                        continue
                    try:
                        addr = parse_address(full_path)
                    except MediaAddressError as e:
                        results_queue.put((
                            "producer_row_error", row_id, "UnparseableMedia",
                            f"could not parse the media value {full_path!r}: {e}",
                        ))
                        continue
                    if _is_bare_video_address(addr):
                        results_queue.put((
                            "producer_row_error", row_id, "WholeVideoRow",
                            "this operator reads single frames; this row "
                            "is a whole video",
                        ))
                        continue
                    if addr.frame is not None and not is_image_path(addr.path):
                        group_key = (addr.path, addr.stream)
                        frame_groups.setdefault(group_key, []).append(
                            (row_id, addr, metadata)
                        )
                        continue
                    try:
                        media = run.resolver.resolve_frame(addr, "analysis").pixels
                    except (MediaResolverError, MediaAddressError, OSError) as e:
                        print(
                            f"[OperatorRegistry] Could not load image "
                            f"for {row_id}: {full_path} ({e})"
                        )
                        results_queue.put((
                            "producer_row_error",
                            *_unreadable_media_row_error(row_id, full_path, e),
                        ))
                        continue
                else:
                    media = None
                if not _try_enqueue((row_id, media, metadata)):
                    return

            for (_source_path, _stream), entries in frame_groups.items():
                if run.cancelled() or abort_event.is_set():
                    return
                addresses = [addr for (_rid, addr, _md) in entries]
                media_by_address_id = {
                    id(addr): (row_id, metadata)
                    for (row_id, addr, metadata) in entries
                }
                pending_row_ids = {row_id for (row_id, _a, _m) in entries}
                try:
                    for row_id, media, metadata in _iter_group_rows(
                        run, addresses, media_by_address_id
                    ):
                        if run.cancelled() or abort_event.is_set():
                            return
                        pending_row_ids.discard(row_id)
                        if not _try_enqueue((row_id, media, metadata)):
                            return
                except Exception as e:
                    print(
                        f"[OperatorRegistry] Ordered decode failed for "
                        f"'{operator.name}' on {_source_path!r}: "
                        f"{type(e).__name__}: {e}"
                    )
                    for row_id in pending_row_ids:
                        results_queue.put((
                            "producer_row_error", row_id,
                            type(e).__name__, str(e),
                        ))
                    continue

            if not (run.cancelled() or abort_event.is_set()):
                # Normal completion only: every consumer is still alive
                # and still draining, so this cannot block forever even
                # with the plain, bounded _try_enqueue below giving up
                # early -- there is nothing to give up FROM here, since
                # every real item was already enqueued above. A
                # cancelled/aborted run skips this entirely: consumers
                # notice run.cancelled() / abort_event on their own poll
                # and stop without needing a sentinel.
                for _ in range(worker_count):
                    if not _try_enqueue(_SENTINEL):
                        break

        def _consumer(consumer_run) -> None:
            while True:
                try:
                    item = work_queue.get(timeout=0.5)
                except queue.Empty:
                    if run.cancelled() or abort_event.is_set():
                        results_queue.put(("consumer_finished",))
                        return
                    continue
                if item is _SENTINEL:
                    results_queue.put(("consumer_finished",))
                    return
                if run.cancelled() or abort_event.is_set():
                    results_queue.put(("consumer_finished",))
                    return
                row_id, media, metadata = item
                try:
                    outcome = _classify_row_result(
                        operator, row_id, media, metadata, consumer_run
                    )
                except BaseException as e:
                    results_queue.put(
                        ("consumer_died", row_id, media, metadata, e)
                    )
                    return
                results_queue.put(("row_outcome", row_id, outcome))

        producer_thread = threading.Thread(target=_producer, daemon=True)
        consumer_threads = [
            threading.Thread(
                target=_consumer,
                args=(dataclasses.replace(run, model=models[i]),),
                daemon=True,
            )
            for i in range(worker_count)
        ]
        producer_thread.start()
        for t in consumer_threads:
            t.start()

        live_consumers = worker_count
        died_count = 0
        while live_consumers > 0:
            item = results_queue.get()
            tag = item[0]
            if tag == "consumer_finished":
                live_consumers -= 1
            elif tag == "consumer_died":
                _, row_id, media, metadata, exc = item
                live_consumers -= 1
                died_count += 1
                print(
                    f"[OperatorRegistry] A consumer thread for "
                    f"'{operator.name}' died on row {row_id!r}: "
                    f"{type(exc).__name__}: {exc}; falling back to serial "
                    f"processing on the coordinator for this row and any "
                    f"rows still queued."
                )
                outcome = _classify_row_result(
                    operator, row_id, media, metadata, _get_fallback_run()
                )
                if not _record(row_id, outcome):
                    aborted = True
                    abort_event.set()
            elif tag == "producer_row_error":
                _, row_id, kind, message = item
                row_errors.append((row_id, kind, message))
            else:  # "row_outcome"
                _, row_id, outcome = item
                if not _record(row_id, outcome):
                    aborted = True
                    abort_event.set()

        # Every consumer is now accounted for (finished normally, or
        # died). If NONE survived and nothing else already aborted the
        # run, the producer may still be decoding/classifying rows that
        # nobody is left to consume -- take over as the sole remaining
        # consumer, ourselves, until the producer itself stops:
        # "every consumer dies" must still finish every row, including
        # ones the producer had not queued yet at the moment of death.
        # abort_event is deliberately NOT set before this loop -- setting
        # it would tell the still-running producer to give up early,
        # which is exactly what this branch exists to avoid.
        if died_count == worker_count and worker_count > 0 and not aborted:
            while producer_thread.is_alive() or not work_queue.empty():
                if run.cancelled():
                    break
                try:
                    item = work_queue.get(timeout=_ENQUEUE_POLL_SECONDS)
                except queue.Empty:
                    continue
                if item is _SENTINEL:
                    continue
                row_id, media, metadata = item
                outcome = _classify_row_result(
                    operator, row_id, media, metadata, _get_fallback_run()
                )
                if not _record(row_id, outcome):
                    aborted = True
                    break

        # Whatever the outcome above, the producer must stop now -- a
        # cancelled/aborted run should not keep decoding, and a normally
        # finished or now-fully-drained run has nothing left for it to
        # do. Setting this unconditionally is harmless when the producer
        # has already returned. The join is bounded: a producer that is
        # not a daemon thread's own doom, so a bound here means
        # on_complete still fires (and the run still ends) even in an
        # unanticipated case where the producer does not stop promptly,
        # rather than the coordinator hanging right back where an
        # unbounded join used to.
        abort_event.set()
        producer_thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
        if producer_thread.is_alive():
            print(
                f"[OperatorRegistry] Producer thread for '{operator.name}' "
                f"did not stop within {_JOIN_TIMEOUT_SECONDS}s of being "
                f"told to; continuing without it (it is a daemon thread "
                f"and will not block process exit)."
            )

        # A no-op in the ordinary case: FIFO guarantees no sentinel is
        # ever reached before every real item ahead of it in the queue
        # has been dequeued by some still-living consumer, so once every
        # consumer has reported in without any dying, the queue holds
        # nothing but orphaned extra sentinels, if any. The all-died
        # fallback above already drained everything else. Skipped
        # entirely once cancelled or aborted: a cancelled run must not
        # process more rows, including ones already sitting in the
        # queue.
        if not (run.cancelled() or aborted):
            while True:
                try:
                    item = work_queue.get_nowait()
                except queue.Empty:
                    break
                if item is _SENTINEL:
                    continue
                row_id, media, metadata = item
                outcome = _classify_row_result(
                    operator, row_id, media, metadata, _get_fallback_run()
                )
                if not _record(row_id, outcome):
                    aborted = True

        row_errors = row_errors + run.collected_row_errors()
        if row_errors and on_row_errors is not None:
            on_row_errors(operation_id, label, row_errors)

        if on_complete is not None:
            elapsed_seconds = time.perf_counter() - start_time
            _log_run_cost(operator.name, emitted, elapsed_seconds, worker_count)
            on_complete(
                operation_id, operator.name, emitted,
                elapsed_seconds=elapsed_seconds, worker_count=worker_count,
            )

    # ── run_create_table ──────────────────────────────────────────────

    def run_create_table(
        self,
        operator_name: str,
        df: pd.DataFrame,
        operation_id: str,
        run,
        on_complete=None,
        on_error=None,
        on_row_errors=None,
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
            on_error:      Called if create_table raises an exception, or
                           if it declares an output column (descriptor
                           OutputSpec.table_columns) its returned frame does
                           not actually contain -- P1.7-1 treats that as a
                           run failure, never a silently dropped hint.
                           Signature: (operation_id: str,
                                       operator_name: str, message: str)
                           Called from background thread.
            on_row_errors: P1.7-1. Called once, after create_table()
                           returns and before on_complete, if the operator
                           reported any row through run.report_row_error()
                           -- the exact same channel and callback shape the
                           COLUMNS path already uses (see run_create_columns
                           above); there is no second channel.
                           Signature: (operation_id: str,
                                       label: str,
                                       errors: list[tuple[str, str, str]])
                           Each tuple is (row_id, kind, message).
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
            args=(
                operator, df, operation_id, run, on_complete, on_error,
                on_row_errors,
            ),
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
        on_row_errors,
    ) -> None:
        """Worker that runs create_table() in the background thread.

        Fix round, item 1: run.report_row_error() reports must reach
        on_row_errors whichever way this method ends -- create_table()
        succeeding, raising, or the declared-column check below raising.
        `label` is computed up front, before either can happen, and
        `_deliver_row_errors` is called exactly once, on every exit path,
        immediately before that path's own terminal on_complete/on_error
        call -- never in a `finally`, which would run AFTER the terminal
        call and reorder it behind the wrong message.
        """
        label = run.spec.mode_descriptor.label
        try:
            result_df = operator.create_table(df, run)

            # P1.7-1: a column the operator declared on its OutputSpec.
            # table_columns but did not actually return is an error, not a
            # silent no-op -- without this check, Dataset._prepare_table
            # would simply drop the unmatched hint (it narrows hints to
            # the columns the frame actually has) and the researcher would
            # never learn the declaration and the result disagreed.
            declared_columns = run.spec.mode_descriptor.output.table_columns
            missing = [
                column.name for column in declared_columns
                if column.name not in result_df.columns
            ]
            if missing:
                raise OperatorRunError(
                    f"operator {operator.name!r} declares output column(s) "
                    f"{missing} on its TABLE descriptor, but its "
                    f"create_table() result does not contain them"
                )
        except NotImplementedError:
            # Report it as an error so AppController deregisters the run
            # it registered before starting this worker -- a silent
            # return would leave it in _live_runs forever.
            print(
                f"[OperatorRegistry] Operator '{operator.name}' "
                f"does not implement create_table()."
            )
            self._deliver_row_errors(operation_id, label, run, on_row_errors)
            if on_error is not None:
                on_error(
                    operation_id, operator.name,
                    f"Operator '{operator.name}' does not implement "
                    f"create_table().",
                )
            return
        except Exception as e:
            print(
                f"[OperatorRegistry] Error in create_table "
                f"for '{operator.name}': {e}"
            )
            self._deliver_row_errors(operation_id, label, run, on_row_errors)
            if on_error is not None:
                on_error(operation_id, operator.name, str(e))
            return

        # Success path: same delivery, same "before the terminal call"
        # ordering, exactly once -- this is the only other exit.
        self._deliver_row_errors(operation_id, label, run, on_row_errors)
        if on_complete is not None:
            on_complete(operation_id, operator.name, result_df)

    def _deliver_row_errors(self, operation_id, label, run, on_row_errors) -> None:
        """Read back every row run.report_row_error() collected and hand
        it to on_row_errors, if any were reported and a sink is wired.

        Called exactly once per _run_create_table_worker invocation --
        from every one of its three exit paths, never from more than one
        -- so a report is never delivered twice. run.collected_row_errors()
        itself is side-effect-free (it copies the list rather than
        draining it), so calling this twice would be safe by construction
        even if a future edit ever did; the "exactly once" guarantee here
        is about correctness of INTENT (one on_row_errors call per run),
        not a defence against a double call.
        """
        row_errors = run.collected_row_errors()
        if row_errors and on_row_errors is not None:
            on_row_errors(operation_id, label, row_errors)

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