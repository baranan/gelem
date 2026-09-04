"""
models/dataset.py

The Dataset component is the single source of truth for all item data
in a Gelem project. It owns all tables and is the only component that
may modify them.

All tables are pandas DataFrames stored in a single dictionary.
The frame-level table is named 'frames' by convention, but all tables
are treated identically — any table can be filtered, viewed, exported,
or used as the source for a new aggregation.

Student B is responsible for implementing the real logic in this file.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
import json
import weakref
import pandas as pd

from media.extensions import MEDIA_EXTENSIONS
from media.media_address import (
    MediaAddressError,
    absolutise,
    relativise,
)
from media.media_address import parse as parse_address
from media.media_address import format as format_address
from models.table_schema import (
    ColumnHint,
    SchemaSerialisationError,
    TableSchema,
    check_frame,
    infer_schema,
    normalise_frame,
    schema_from_dict,
)


# The set of media file extensions is declared once in media/extensions.py
# (P1.8d-2) and imported above -- folder scanning here and the schema-layer
# tag rule in models/table_schema.py now read the same list. `f.suffix.lower()
# in MEDIA_EXTENSIONS` works unchanged: it is a frozenset membership test.


# ---------------------------------------------------------------------------
# MergeReport
# ---------------------------------------------------------------------------

@dataclass
class MergeReport:
    """
    Produced by Dataset.merge_csv() before any changes are committed.
    Contains diagnostic information about the quality of the join so
    the researcher can decide whether to proceed.
    """
    total_csv_rows: int = 0
    total_image_files: int = 0
    matched_rows: int = 0
    unmatched_files: list[str] = field(default_factory=list)
    unmatched_csv_rows: list[str] = field(default_factory=list)
    duplicate_keys_files: list[str] = field(default_factory=list)
    duplicate_keys_csv: list[str] = field(default_factory=list)
    one_to_many: list[str] = field(default_factory=list)
    renamed_columns: dict = field(default_factory=dict)
    sample_problems: list[dict] = field(default_factory=list)

    # The joined DataFrame, held privately until confirm_merge() is called.
    _pending_df: pd.DataFrame | None = field(default=None, repr=False)
    _new_columns: list[str] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        """Returns a human-readable summary string for display in the UI."""
        return (
            f"Matched: {self.matched_rows} rows | "
            f"Unmatched files: {len(self.unmatched_files)} | "
            f"Unmatched CSV rows: {len(self.unmatched_csv_rows)} | "
            f"Duplicates: {len(self.duplicate_keys_files)}"
        )


# ---------------------------------------------------------------------------
# ProvenanceLog
# ---------------------------------------------------------------------------

class ProvenanceLog:
    """
    An append-only log of every structural operation performed in a
    Gelem project. Stored as a JSON file inside the project folder.
    """

    def __init__(self):
        self._entries: list[dict] = []

    def record(self, action: str, params: dict) -> None:
        """
        Appends a new entry to the log.

        Args:
            action: Short string identifying the operation.
            params: Dict of parameters used for this operation.
        """
        import datetime
        entry = {
            "action": action,
            "params": params,
            "timestamp": datetime.datetime.now().isoformat(),
            "gelem_version": "0.1.0",
        }
        self._entries.append(entry)

    def export_as_script(self, output_path: Path) -> None:
        """
        Translates the provenance log into a standalone Python script
        that reproduces the full analysis from scratch.

        Args:
            output_path: Path where the .py script should be written.

        TODO (Student B): Implement this.
        """
        # PLACEHOLDER
        script_lines = [
            "# Gelem provenance script",
            "# Generated automatically -- do not edit by hand",
            "",
            "from pathlib import Path",
            "from models.dataset import Dataset",
            "",
            "dataset = Dataset()",
        ]
        for entry in self._entries:
            script_lines.append(
                f"# {entry['action']} at {entry['timestamp']}"
            )
        output_path.write_text("\n".join(script_lines))

    def to_list(self) -> list[dict]:
        """Returns all log entries as a list of dicts."""
        return list(self._entries)

    def replace(self, entries: list[dict]) -> None:
        """Replaces all entries with the given list. Used by Dataset.load()
        to restore the saved log without poking the private _entries field."""
        self._entries = list(entries)


# ---------------------------------------------------------------------------
# Path helpers for save() / load()
# ---------------------------------------------------------------------------

def _is_blank_cell(cell) -> bool:
    """True for a cell that carries no address at all: an empty string, a
    non-string value, or NaN. Such a cell is returned unchanged by both
    rewrite directions below."""
    if not isinstance(cell, str):
        return True
    if cell == "":
        return True
    # A non-empty string is never NaN, but keep the guard explicit and
    # parallel with the old _rel_if_inside / _abs_against null check.
    return bool(pd.isna(cell))


def _rewrite_media_cell(cell, project_root: Path, *, to_stored: bool, working_dir):
    """Rewrite the path portion of one media cell. The single place a media
    cell is parsed during save or load.

    A media cell is an address (docs/media_architecture.md section 3.2), and
    a bare path is a valid address, so plain-path cells are handled by
    construction. On save (to_stored=True):

      1. parse the cell into a MediaAddress (parse() normalises the path
         portion to forward slashes, P0.2c).
      2. absolutise against `working_dir` -- the current working directory,
         read once by the caller and passed in. This preserves save()'s
         historical behaviour of anchoring a relative input to the working
         directory, which test_save_load_relative_source_path_resolves_correctly
         locks. It is passed as an argument, never read here, so
         models/dataset.py stays the only file that touches ambient state --
         the same reason absolutise() takes its base as an argument.
      3. relativise against project_root, so a path inside the project
         folder is stored relative and stays portable (CLAUDE.md's [NOW]
         rule); a path outside it is left absolute by relativise().
      4. format back to the canonical stored string.

    On load (to_stored=False) step 1 is the same, then the path is
    absolutised against project_root and formatted -- no relativise, and
    `working_dir` is unused (pass None).

    The address fragment (#f=, #t=, #r=, stream selector) is preserved
    exactly; only the path portion moves.

    Returns (new_cell, was_unparseable). A blank cell (empty, non-string or
    NaN) is returned unchanged with was_unparseable False. A cell that will
    not parse is returned VERBATIM with was_unparseable True rather than
    raising: one bad cell must not make the whole project unsaveable, and an
    untouched string round-trips exactly. The caller counts the True cases
    into the provenance entry.
    """
    if _is_blank_cell(cell):
        return cell, False
    try:
        addr = parse_address(cell)
    except MediaAddressError:
        return cell, True
    if to_stored:
        addr = absolutise(addr, working_dir)
        addr = relativise(addr, str(project_root))
    else:
        addr = absolutise(addr, str(project_root))
    return format_address(addr), False


def _rewrite_media_column(values, project_root: Path, to_stored: bool):
    """Apply _rewrite_media_cell to every cell in `values`, once each.

    Returns (new_values, unparseable_count). A cell that will not parse as
    an address is passed through unchanged and counted -- never raised on,
    so a single bad cell cannot block save or load. The count is reported in
    the provenance "save" / "load" entry rather than swallowed silently.
    """
    # Read the working directory once per column, not once per cell -- a
    # 530k-row table must not perform 530k getcwd calls during save().
    # Only the to_stored (save) direction anchors relative inputs to it.
    working_dir = str(Path.cwd()) if to_stored else None

    new_values = []
    unparseable = 0
    for cell in values:
        new_cell, was_unparseable = _rewrite_media_cell(
            cell, project_root, to_stored=to_stored, working_dir=working_dir
        )
        new_values.append(new_cell)
        if was_unparseable:
            unparseable += 1
    return new_values, unparseable


# ---------------------------------------------------------------------------
# Schema acceptance
# ---------------------------------------------------------------------------

# row_id is exempt from every TableSchema. docs/architecture.md §4.1 makes it an
# opaque handle that carries no meaning and is unique only within its table; none
# of the three roles in §4.2 (identifier / index / measurement) fits it, and
# adding a fourth role for one bookkeeping column would leak Dataset's internals
# into the vocabulary operators use. The exemption lives here, in Dataset --
# models/table_schema.py stays ignorant of row_id and takes no exemption
# argument.
_SCHEMA_EXEMPT_COLUMNS = frozenset({"row_id"})


def _non_exempt_columns(df: pd.DataFrame) -> list[str]:
    """The frame's columns that a TableSchema describes -- every column except
    the schema-exempt bookkeeping ones, in the frame's own order."""
    return [c for c in df.columns if c not in _SCHEMA_EXEMPT_COLUMNS]


def _cast_empty_frame_to_schema(
    df: pd.DataFrame, schema: TableSchema
) -> pd.DataFrame:
    """Return a copy of a ZERO-ROW frame with every column the schema names cast
    to the dtype the schema declares for it.

    Only meaningful for an empty frame, and only used by load(): pyarrow does
    not preserve an empty column's dtype through a parquet round trip, so a
    reloaded empty text column arrives as float64 and would fail check_frame's
    "crossing text/number/bool" gate -- which rejects on kind before it ever
    looks at values. An empty column casts to any dtype without data loss, so
    doing the cast BEFORE the check restores the saved schema exactly.

    Columns the schema does not name (row_id) are left untouched. Raises
    TypeError or ValueError if a cast is genuinely impossible; the caller then
    falls back to inference for that one table.
    """
    out = df.copy()
    for name in schema.column_names():
        if name in out.columns:
            out[name] = out[name].astype(schema.spec_for(name).dtype)
    return out


class SchemaRejection(ValueError):
    """Raised by Dataset._accept_table when a frame does not conform to its
    table's schema. The frame is not stored, wholly or partly."""


@dataclass
class _PreparedTable:
    """Everything needed to store one table, computed by Dataset._prepare_table
    without mutating any Dataset state. Dataset._commit_prepared is the only
    consumer -- it performs the actual writes. Kept as a plain value object so a
    caller (load()) can build several of these, decide the whole batch is good,
    and only then commit them one after another -- an all-or-nothing load.
    """
    # The table's name.
    table_name: str
    # The frame to store: already a fresh object (a copy when an adjustment was
    # applied, otherwise the frame handed in, which every caller already built
    # or copied itself).
    frame: pd.DataFrame
    # The TableSchema to store for this table.
    schema: TableSchema
    # One plain-English sentence per lossless dtype adjustment, to append to
    # Dataset._schema_messages on commit.
    messages: list[str]
    # A ready-to-record "schema_adjustments" provenance params dict, or None
    # when the frame needed no adjustment.
    provenance_params: dict | None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dataset:
    """
    The single source of truth for all item data in a Gelem project.

    All tables — whether frame-level or aggregated — are stored
    identically as pandas DataFrames in a single dictionary keyed by
    table name. The frame-level table is named 'frames' by convention
    but is not treated differently from any other table.

    Gelem makes no assumptions about column names beyond the required
    internal columns: row_id, full_path, file_name (for the frames table).
    """

    # Required columns that Gelem creates internally for the frames table.
    FRAMES_REQUIRED_COLUMNS = ["row_id", "full_path", "file_name"]

    # Class-level default for the per-instance strict_schema flag. The
    # production default is OFF: an "unexpected" dtype adjustment is applied
    # and recorded, not refused. The pytest suite flips this class default to
    # True for the whole run (tests/conftest.py), so every Dataset built in a
    # test -- including the one this constructor builds below -- is strict.
    _DEFAULT_STRICT_SCHEMA: bool = False

    def __init__(self):
        self.provenance = ProvenanceLog()
        self._id_counter: int = 0

        # row_id -> positional index, one dict per table. Lazily built and
        # self-healing -- see _row_index_for().
        self._row_index: dict[str, dict[str, int]] = {}
        # table_name -> (row count, weakref to the DataFrame) the index
        # above was built from. See _row_index_for() for how this is used.
        self._row_index_stamp: dict[str, tuple[int, weakref.ReferenceType]] = {}

        # One TableSchema per stored table (docs/architecture.md §4.3), decided
        # only in _accept_table. A table assigned straight into _tables by a
        # test has no schema -- schema_for() returns None for it.
        self._schemas: dict[str, TableSchema] = {}
        # Plain-English notes about every lossless dtype adjustment made on
        # accept, drained by take_schema_messages(). A third reporting
        # destination, added instead of widening the return type of nine
        # public methods.
        self._schema_messages: list[str] = []
        # When True, _accept_table also refuses a frame whose only problem is
        # an "unexpected" dtype adjustment (the frame declared a width the
        # schema did not expect), and apply_row_updates re-raises instead of
        # degrading. Read from the class default here, BEFORE the first
        # _accept_table call below, so the constructor's own accept is strict
        # too when the suite has flipped the class default.
        self.strict_schema: bool = type(self)._DEFAULT_STRICT_SCHEMA

        self._tables: dict[str, pd.DataFrame] = {}
        self._accept_table(
            "frames",
            pd.DataFrame(columns=self.FRAMES_REQUIRED_COLUMNS),
            source="__init__",
        )

    def set_registry(self, registry) -> None:
        """Retained as a no-op only so existing callers do not break.

        P1.8d-2b-1: Dataset no longer writes to ColumnTypeRegistry at all.
        After P1.8d-2a nothing reads the registry's column-name map on the
        display path (AppController reads a column's display tag off the
        TableSchema instead), so the map's only remaining writer is the
        operator output-column path in the controller. Dataset holds no
        registry reference and this method deliberately ignores its argument.
        AppController's construction and several test fixtures
        (tests/conftest.py, tests/test_dataset_access_paths.py, ...) still
        call it; keeping the method a no-op means none of them need editing
        in this item.
        """
        # Intentionally does nothing -- see the docstring.
        return

    def _next_id(self) -> str:
        """Generates a new unique row_id string."""
        self._id_counter += 1
        return f"{self._id_counter:06d}"

    # ------------------------------------------------------------------
    # Row-id index (row_id -> positional index, per table)
    # ------------------------------------------------------------------
    #
    # The index is self-healing rather than merely maintained. Several
    # tests in tests/test_dataset.py assign straight to ds._tables[...],
    # bypassing every Dataset method, so an index that trusted its own
    # bookkeeping would go stale behind those writes and silently return
    # rows from the wrong table -- worse than being slow. Instead, every
    # lookup validates the cached index against a cheap stamp,
    # (row count, weakref to the DataFrame), and rebuilds from the live
    # frame whenever the stamp does not match, rather than assuming the
    # cache is still good.
    #
    # The stamp holds a weakref, not id(df). id() is a bare memory
    # address: it does not keep the old frame alive, so once it is freed
    # Python is free to hand that same address to a later, unrelated
    # DataFrame. Two table replacements with no lookup in between, plus a
    # matching row count, would then make a stale stamp compare equal by
    # coincidence -- get_row() silently returning positions from the
    # wrong frame. A weakref cannot be fooled that way: once the frame it
    # pointed to is gone, dereferencing it returns None forever, so the
    # comparison correctly fails rather than accidentally matching a
    # different object at the same address. It also does not keep a
    # replaced table's memory alive the way storing the DataFrame itself
    # would.
    #
    # This also means writers do not need to invalidate anything by hand.
    # apply_row_updates mutates the stored frame in place (new columns and
    # specific cells, never row_id or row count) and, on the common path where
    # the schema accept makes no dtype adjustment, re-stores that same object --
    # so the (row count, weakref) stamp still matches and the cached index is
    # not rebuilt. Every other write path -- add_column, add_computed_column,
    # aggregate, create_table_from_rows, create_table_from_df, confirm_merge,
    # load, a schema accept that did adjust a dtype, and a test's direct
    # ds._tables[...] = ... assignment -- stores a different DataFrame object,
    # so the stamp no longer matches and the very next lookup rebuilds from the
    # live frame rather than trusting stale positions. _set_table() and
    # _reset_tables() below exist so every write goes through one obvious place.

    def _get_stored_table(self, table_name: str) -> pd.DataFrame:
        """Returns the live, stored DataFrame for table_name (no copy).
        Internal only -- callers that need their own copy use get_table()
        or snapshot_rows(); callers that only read use read_only_view()."""
        if table_name not in self._tables:
            raise KeyError(f"Table '{table_name}' does not exist in this project.")
        return self._tables[table_name]

    def _set_table(self, table_name: str, df: pd.DataFrame) -> None:
        """The single place one stored table is written or replaced. Called
        only by _accept_table -- every other write path goes through the schema
        accept path, not here (guarded by tests/test_dataset_schema.py)."""
        self._tables[table_name] = df

    def _reset_tables(self, tables: dict[str, pd.DataFrame]) -> None:
        """The single place the whole _tables dict is replaced, e.g. by a
        fresh load_folder(), load_csv_as_primary(), or load(). Clears the
        schema map and the index caches too -- not required for correctness
        (the stamp check in _row_index_for would catch every one of these
        tables being a new object anyway), but a table dropped by the reset
        (e.g. one that existed only in the previous project) would otherwise
        leave a dead entry sitting in these dicts forever."""
        self._tables = tables
        self._schemas.clear()
        self._row_index.clear()
        self._row_index_stamp.clear()

    # ------------------------------------------------------------------
    # Schema accept path
    # ------------------------------------------------------------------

    def _media_column_hints(self, df: pd.DataFrame) -> dict[str, ColumnHint]:
        """The only hint infer_schema gets: a media type tag for 'full_path'.

        P1.8d-2b-1 removed the ColumnTypeRegistry read that used to widen this
        to every column the registry tagged 'media_path'. That read is no
        longer needed:

          * A column the stored TableSchema already names keeps its ColumnSpec
            unchanged on re-accept -- _prepare_table resolves such a column
            against the authoritative schema and never re-infers it -- so an
            existing media column keeps its 'media_path' tag with no hint.
          * A brand-new column's media tag is now decided by infer_type_tag
            from its values (models/table_schema.py), which recognises a path
            or a media-address fragment.

        'full_path' still needs the hint because on an empty frames table its
        column carries no values for infer_type_tag to read. No role hints --
        that decision has not been taken."""
        return {
            col: ColumnHint(type_tag="media_path")
            for col in df.columns
            if col == "full_path" and col not in _SCHEMA_EXEMPT_COLUMNS
        }

    def _prepare_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        *,
        hints: dict[str, ColumnHint] | None = None,
        schema: TableSchema | None = None,
        consult_stored_schema: bool = True,
        source: str = "",
    ) -> _PreparedTable:
        """All of _accept_table's work up to and including validation and
        normalisation, with NO mutation of Dataset state. Returns a
        _PreparedTable that _commit_prepared then stores. See
        docs/architecture.md §4.3.

        a. row_id (and any other _SCHEMA_EXEMPT_COLUMNS member) is not part of
           any schema.
        b. Build the schema to check against from the FRAME's non-exempt
           columns, in the frame's own order. The ColumnSpec for a column is
           resolved against the AUTHORITATIVE schema -- see below. A column the
           authoritative schema names keeps that ColumnSpec unchanged; a column
           it does not name gets a spec inferred from the frame; a column it
           names but the frame no longer carries is simply dropped (not a
           rejection).
        c./d. check_frame; a non-empty rejection list means the frame is not
           stored -- raise SchemaRejection naming the table, the source and
           every rejection.
        e. Otherwise apply the adjustments and write them back onto a copy of
           the FULL frame, so row_id and the original column order survive.
        f. Return the adjustment sentences and the provenance params so
           _commit_prepared can record them; this method records nothing.

        The AUTHORITATIVE schema is resolved like this:
          * an explicit `schema` argument always wins (load() passes the schema
            it restored from schemas.json);
          * otherwise, only when `consult_stored_schema` is True, the schema
            already stored for this table -- which is the default and is
            byte-for-byte the old behaviour;
          * otherwise None: every column is inferred from the frame alone.
        load() passes `consult_stored_schema=False` because the stored schema,
        if any, belongs to the project being replaced and must never shape the
        project being loaded -- not for a table schemas.json names, and not for
        one it does not.

        P1.8d-2b-1 removed the companion `consult_registry_hints` flag. It
        existed because _media_column_hints() used to read the
        ColumnTypeRegistry, which during a load still holds the outgoing
        project's column tags -- so an incoming column could inherit a
        `media_path` tag from the project being replaced. _media_column_hints()
        now only ever hints 'full_path', which load() already passes in
        `hints` from the incoming project's own restored schema, so there is
        nothing left to suppress.
        """
        schema_columns = _non_exempt_columns(df)

        # A frame with a duplicate column name cannot be described by a
        # TableSchema (or read column-wise). Make it one more refused-frame
        # case with the one exception type, naming table and source.
        if len(schema_columns) != len(set(schema_columns)):
            dupes = sorted(
                {c for c in schema_columns if schema_columns.count(c) > 1}
            )
            raise SchemaRejection(
                f"Dataset refused table {table_name!r} from {source!r}: "
                f"duplicate column name(s) {dupes}"
            )

        frame_for_schema = df[schema_columns]

        # The authoritative source of already-decided ColumnSpecs -- see the
        # docstring. An explicit `schema` wins; else the stored schema only if
        # the caller allows it; else nothing.
        if schema is not None:
            authoritative = schema
        elif consult_stored_schema:
            authoritative = self._schemas.get(table_name)
        else:
            authoritative = None
        stored_names = (
            set(authoritative.column_names()) if authoritative is not None else set()
        )
        new_columns = [c for c in schema_columns if c not in stored_names]

        inferred_by_name: dict[str, object] = {}
        if new_columns:
            # _media_column_hints only ever hints 'full_path' now (P1.8d-2b-1).
            # An explicit `hints` from the caller -- load() passes the incoming
            # project's restored media columns -- wins on any key collision.
            effective_hints = dict(self._media_column_hints(df))
            if hints:
                effective_hints.update(hints)
            sub_hints = {
                name: h
                for name, h in effective_hints.items()
                if name in new_columns
            }
            # infer_schema raises ValueError on a dtype it does not support
            # (datetime64, a nullable extension dtype). Surface that as a
            # SchemaRejection, so every "cannot accept this frame" outcome has
            # the one type and names the table and source.
            try:
                inferred = infer_schema(df[new_columns], hints=sub_hints)
            except ValueError as exc:
                raise SchemaRejection(
                    f"Dataset refused table {table_name!r} from {source!r}: "
                    f"{exc}"
                ) from exc
            inferred_by_name = {s.name: s for s in inferred.columns}

        specs = []
        for name in schema_columns:
            if name in stored_names:
                specs.append(authoritative.spec_for(name))
            else:
                specs.append(inferred_by_name[name])
        # Named to keep it distinct from the `schema` PARAMETER above: this is
        # the schema built for THIS frame, which becomes the table's stored
        # schema on commit.
        built_schema = TableSchema(columns=tuple(specs))

        check = check_frame(frame_for_schema, built_schema)

        if check.rejections:
            raise SchemaRejection(
                f"Dataset refused table {table_name!r} from {source!r}: "
                + "; ".join(check.rejections)
            )

        # Step 4: strict mode also refuses an "unexpected" adjustment -- a
        # width the frame declared that the schema did not expect. A
        # "storage_policy" adjustment never triggers this.
        if self.strict_schema:
            unexpected = [
                a for a in check.adjustments if a.kind == "unexpected"
            ]
            if unexpected:
                raise SchemaRejection(
                    f"Dataset refused table {table_name!r} from {source!r} "
                    f"(strict_schema): unexpected dtype adjustment(s): "
                    + "; ".join(
                        f"{a.column} {a.arrived_as}->{a.stored_as}"
                        for a in unexpected
                    )
                )

        if check.adjustments:
            # An adjustment rewrites a column's dtype, so build a fresh frame
            # and leave the caller's untouched.
            out = df.copy()
            normalised = normalise_frame(
                frame_for_schema, built_schema, check=check
            )
            for adjustment in check.adjustments:
                out[adjustment.column] = normalised[adjustment.column]
        else:
            # Nothing to rewrite. Every call site already hands _prepare_table a
            # frame it just built or copied (a fresh DataFrame, a .copy(), or --
            # for apply_row_updates -- the live stored frame it is re-storing
            # unchanged), so store it as-is. Re-storing the same object on commit
            # is what keeps apply_row_updates' row-id index stamp valid.
            out = df

        messages = [
            f"Table {table_name!r}: column {adjustment.column!r} arrived as "
            f"{adjustment.arrived_as} and is stored as {adjustment.stored_as}."
            for adjustment in check.adjustments
        ]
        provenance_params = None
        if check.adjustments:
            provenance_params = {
                "table": table_name,
                "source": source,
                "adjustments": [
                    {
                        "column": a.column,
                        "arrived_as": a.arrived_as,
                        "stored_as": a.stored_as,
                        "reason": a.reason,
                        "kind": a.kind,
                    }
                    for a in check.adjustments
                ],
            }

        return _PreparedTable(
            table_name=table_name,
            frame=out,
            schema=built_schema,
            messages=messages,
            provenance_params=provenance_params,
        )

    def _commit_prepared(self, prepared: _PreparedTable) -> None:
        """Store what _prepare_table computed. The ONLY caller of _set_table.
        Mutates: the stored table, its schema, _schema_messages, and -- when the
        frame needed adjustment -- the provenance log. Only ever handed a
        _PreparedTable, which only _prepare_table builds, so every stored table
        is still schema-validated."""
        self._set_table(prepared.table_name, prepared.frame)
        self._schemas[prepared.table_name] = prepared.schema
        self._schema_messages.extend(prepared.messages)
        if prepared.provenance_params is not None:
            self.provenance.record(
                "schema_adjustments", prepared.provenance_params
            )

    def _accept_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        *,
        hints: dict[str, ColumnHint] | None = None,
        schema: TableSchema | None = None,
        source: str = "",
    ) -> None:
        """Prepare one table and immediately store it. Every non-load write
        path calls this, never _set_table directly. load() instead calls
        _prepare_table for every table first and _commit_prepared for each only
        once it knows the whole project parses -- an atomic load (P1.8c-2b).

        See _prepare_table for the schema-resolution rules (points a-f). With no
        `schema` argument the stored schema for this table is consulted, exactly
        as before.
        """
        prepared = self._prepare_table(
            table_name, df, hints=hints, schema=schema, source=source
        )
        self._commit_prepared(prepared)

    def _row_index_for(self, table_name: str) -> dict[str, int]:
        """Returns the row_id -> positional-index mapping for one table,
        rebuilding it first if the validity stamp no longer matches the
        live table. See the section comment above for the design.

        The stamp is checked by hand (length, then `ref() is df`) rather
        than by comparing the (length, weakref) tuples with `==` -- a
        tuple `==` would fall through to the weakrefs' own `__eq__`,
        which compares referents with `==` too, and `DataFrame == DataFrame`
        returns a DataFrame of booleans, not the single True/False a
        cache check needs."""
        df = self._get_stored_table(table_name)
        cached = self._row_index_stamp.get(table_name)
        is_valid = (
            cached is not None
            and cached[0] == len(df)
            and cached[1]() is df
        )
        if not is_valid:
            self._row_index[table_name] = {
                row_id: pos for pos, row_id in enumerate(df["row_id"])
            }
            self._row_index_stamp[table_name] = (len(df), weakref.ref(df))
        return self._row_index[table_name]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_folder(self, folder_path: Path) -> None:
        """
        Scans a folder for supported media files (images and videos)
        and creates one row per file in the frames table with row_id,
        full_path, and file_name. The accepted table's TableSchema tags
        full_path as 'media_path'.

        Supported formats are the frozenset in media/extensions.py,
        imported as MEDIA_EXTENSIONS. To add a new format, add its
        extension there — no other changes are needed.

        Args:
            folder_path: Absolute path to the folder containing files.

        """
        # Reset all tables for a fresh load (not just 'frames') so old
        # derived tables don't linger.
        self._id_counter = 0
        self._reset_tables({
            "frames": pd.DataFrame(columns=self.FRAMES_REQUIRED_COLUMNS)
        })

        # Scan the folder for supported media files and create rows.
        found_files = []
        for f in folder_path.iterdir():
            if f.suffix.lower() in MEDIA_EXTENSIONS:
                found_files.append(f)

        rows = []
        if found_files:
            for f in sorted(found_files):
                rows.append({
                    "row_id":    self._next_id(),
                    "full_path": str(f),
                    "file_name": f.name,
                })

        # If no media files found, create placeholder empty table with one row so the UI has something to show.
        if rows:
            self._accept_table("frames", pd.DataFrame(rows), source="load_folder")
        else:
            self._accept_table(
                "frames",
                pd.DataFrame(columns=self.FRAMES_REQUIRED_COLUMNS),
                source="load_folder",
            )

        # P1.8d-2b-1: Dataset no longer writes column tags into
        # ColumnTypeRegistry. The 'frames' schema built by the accept above is
        # the single authority for every column's display tag; AppController
        # reads it straight off schema_for().
        self.provenance.record(
            "load_folder", {"folder_path": str(folder_path)}
        )

    def load_csv_as_primary(
        self,
        csv_path: Path,
        image_column: str | None = None,
    ) -> None:
        """
        Loads a CSV file as the primary data source, without requiring
        a folder of images. Each CSV row becomes one row in the frames table.

        Args:
            csv_path:     Absolute path to the CSV file.
            image_column: Optional name of a column in the CSV that contains
                          file paths to media files. If provided and the files
                          exist, full_path is set from this column so
                          thumbnails can be generated.

        TODO (Student B): Implement this method.
        """
        # Reset all tables for a fresh load (not just 'frames') so old
        # derived tables don't linger.
        self._id_counter = 0
        self._reset_tables({
            "frames": pd.DataFrame(columns=self.FRAMES_REQUIRED_COLUMNS)
        })

        # PLACEHOLDER: reads the CSV and creates rows.
        try:
            csv_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[Dataset] load_csv_as_primary() failed to read CSV: {e}")
            return

        rows = []
        for _, csv_row in csv_df.iterrows():
            row = {"row_id": self._next_id()}

            if image_column and image_column in csv_df.columns:
                path_val = csv_row.get(image_column, "")
                row["full_path"] = str(path_val)
                row["file_name"] = Path(str(path_val)).name
            else:
                row["full_path"] = ""
                row["file_name"] = ""

            for col in csv_df.columns:
                row[col] = csv_row[col]

            rows.append(row)

        self._accept_table(
            "frames", pd.DataFrame(rows), source="load_csv_as_primary"
        )

        # P1.8d-2b-1: no ColumnTypeRegistry write. The schema the accept built
        # is the single authority for every column's display tag.

        self.provenance.record("load_csv_as_primary", {
            "csv_path":     str(csv_path),
            "image_column": image_column,
            "n_rows":       len(rows),
        })

    # ------------------------------------------------------------------
    # CSV merging
    # ------------------------------------------------------------------

    def merge_csv(
        self,
        csv_path: Path,
        join_on: str,
        preprocess: dict | None = None,
    ) -> MergeReport:
        """
        Performs a left join of the CSV onto the frames table.
        Returns a MergeReport without committing any changes.
        The researcher must call confirm_merge() after reviewing the report.

        Args:
            csv_path:   Absolute path to the CSV file.
            join_on:    Column name in the CSV to join on. Matched
                        against file_name in the frames table.
            preprocess: Optional preprocessing rules for key matching.

        Returns:
            A MergeReport describing the quality of the join.

        """
        # Step 1: Read the CSV file into a DataFrame.
        csv_df = pd.read_csv(csv_path)

        # Step 1a: The join column must exist in the CSV (it can be any
        # column the caller chose, not necessarily file_name).
        if join_on not in csv_df.columns:
            raise ValueError(
                f"The CSV has no column '{join_on}' to merge on. "
                f"Available columns: {', '.join(csv_df.columns)}."
            )

        # Step 1b: Reject CSVs that reuse Gelem's reserved column names
        # (the join_on column is allowed).
        reserved = [
            c for c in csv_df.columns
            if c in self.FRAMES_REQUIRED_COLUMNS and c != join_on
        ]
        if reserved:
            raise ValueError(
                f"The CSV contains column name(s) reserved by Gelem: "
                f"{', '.join(reserved)}. The names "
                f"{', '.join(self.FRAMES_REQUIRED_COLUMNS)} are used internally "
                f"by Gelem. Please rename these columns in the CSV before merging."
            )

        # Step 2: apply preproccesing rules to the keys if needed.
        if preprocess is not None:
            pass # TODO: apply preprocessing rules TBD on.

        # Step 2b: Reject one-to-many merges (a CSV key matching >1 image row
        # would duplicate that image, e.g. 20 -> 40). Refuse, don't expand.
        csv_counts    = csv_df[join_on].value_counts()
        duplicate_csv = list(csv_counts[csv_counts > 1].index.astype(str))
        frames_keys   = set(self._tables["frames"]["file_name"])
        one_to_many   = [k for k in duplicate_csv if k in frames_keys]
        if one_to_many:
            report = MergeReport(
                total_csv_rows=len(csv_df),
                total_image_files=len(self._tables["frames"]),
                matched_rows=0,
                duplicate_keys_csv=duplicate_csv,
                one_to_many=one_to_many,
            )
            # _pending_df stays None, so confirm_merge() will not commit.
            return report

        # Step 3: Left join the CSV onto the frames table. A column present in
        # BOTH (other than the keys) would collide, so we suffix them: existing
        # -> <name>_a, incoming -> <name>_b, keeping both instead of crashing.
        frames_df = self._tables["frames"]
        collisions = [
            c for c in csv_df.columns
            if c in frames_df.columns and c not in ("file_name", join_on)
        ]
        joined = frames_df.merge(
            csv_df,
            left_on="file_name",
            right_on=join_on,
            how="left",
            suffixes=("_a", "_b"),
        )
        renamed_columns = {c: (f"{c}_a", f"{c}_b") for c in collisions}

        # Step 4: Calculate statistics and build the report. The CSV-side
        # duplicate_csv and one_to_many were already computed in Step 2b and
        # are reused in the report below. A collided column 'path' arrives in
        # the joined table as 'path_b', so new_columns uses the post-merge name.
        new_columns = [
            (f"{c}_b" if c in collisions else c)
            for c in csv_df.columns if c != join_on
        ]

        if new_columns:
            matched_mask = joined[new_columns[0]].notna()
        else:
            matched_mask = pd.Series([False] * len(joined))

        unmatched_files = list(joined.loc[~matched_mask, "file_name"])
        matched_keys    = set(frames_df.loc[matched_mask.values, "file_name"])
        unmatched_csv   = list(
            csv_df.loc[~csv_df[join_on].isin(matched_keys), join_on].astype(str)
        )

        # Duplicate file names among the loaded images themselves (usually none).
        file_counts     = frames_df["file_name"].value_counts()
        duplicate_files = list(file_counts[file_counts > 1].index)

        report = MergeReport(
            total_csv_rows=len(csv_df),
            total_image_files=len(frames_df),
            matched_rows=int(matched_mask.sum()),
            unmatched_files=unmatched_files,
            unmatched_csv_rows=unmatched_csv,
            duplicate_keys_files=duplicate_files,
            duplicate_keys_csv=duplicate_csv,
            one_to_many=one_to_many,
            renamed_columns=renamed_columns,
        )
        report._pending_df  = joined
        report._new_columns = new_columns
        return report

    def confirm_merge(self, report: MergeReport) -> None:
        """
        Commits the merge described in the MergeReport.

        Args:
            report: The MergeReport returned by merge_csv().
        """
        if report._pending_df is not None:
            self._accept_table(
                "frames", report._pending_df.copy(), source="confirm_merge"
            )

        # P1.8d-2b-1: no ColumnTypeRegistry write. The merged-in columns are
        # tagged by the schema the accept above rebuilt.
        self.provenance.record(
            "confirm_merge", {"matched_rows": report.matched_rows}
        )

    # ------------------------------------------------------------------
    # Column operations
    # ------------------------------------------------------------------

    def add_computed_column(
        self,
        name: str,
        expression: str,
        col_type: str = "numeric",
        table_name: str = "frames",
    ) -> None:
        """
        Evaluates a pandas expression against the named table and adds
        the result as a new column.

        Args:
            name:       Name of the new column.
            expression: A pandas eval-compatible expression.
            col_type:   Column type tag. Defaults to 'numeric'.
            table_name: Which table to add the column to.
        """
        # Work on a copy: _accept_table may reject the frame, and a rejection
        # must leave the stored table untouched (SchemaRejection's contract).
        df = self._get_stored_table(table_name).copy()
        df[name] = df.eval(expression)
        self._accept_table(table_name, df, source="add_computed_column")

        # P1.8d-2b-1: no ColumnTypeRegistry write. `col_type` is kept in the
        # signature for callers but the accepted table's schema is now the
        # authority for the new column's display tag.
        self.provenance.record("add_computed_column", {
            "name":       name,
            "expression": expression,
            "table_name": table_name,
        })

    def add_column(
        self,
        name: str,
        values: pd.Series,
        col_type: str,
        table_name: str = "frames",
    ) -> None:
        """
        Inserts a pre-computed Series as a new column in bulk.
        The Series must be indexed by row_id.

        Args:
            name:       Name of the new column.
            values:     Series indexed by row_id.
            col_type:   Column type tag.
            table_name: Table to add the column to.
        """
        # Work on a copy so a schema rejection leaves the stored table intact.
        df = self._get_stored_table(table_name).copy()
        df[name] = df["row_id"].map(values)
        self._accept_table(table_name, df, source="add_column")
        # P1.8d-2b-1: no ColumnTypeRegistry write -- the schema the accept
        # built is the authority for the new column's display tag. `col_type`
        # stays in the signature for callers.

    def update_row(
        self,
        row_id: str,
        updates: dict,
        table_name: str = "frames",
    ) -> None:
        """
        Updates a single row with new column values. A convenience
        wrapper over apply_row_updates() for the one-row case.

        As of P0.2b, AppController no longer calls this -- it batches a
        whole timer tick's operator results into one apply_row_updates()
        call per table. This method is kept as a legitimate one-row API
        (tests and any future single-row caller use it); it is not dead
        code, it simply has no caller in the app right now.

        Args:
            row_id:     The row to update.
            updates:    Dict of column name to new value.
            table_name: Table containing the row.
        """
        self.apply_row_updates(table_name, {row_id: updates})

    def apply_row_updates(
        self,
        table_name: str,
        updates: dict[str, dict],
    ) -> list[str]:
        """
        Applies a batch of per-row updates in one call. This is the
        primary write path; update_row() is a one-item convenience over
        it.

        Uses the row-id index (see _row_index_for) to place each row's
        values by position, so this costs one dict lookup per row rather
        than a full-column scan per update.

        Args:
            table_name: Table containing the rows.
            updates:    Dict mapping row_id to a dict of column name to
                        new value. A column that does not exist yet is
                        created first and every row not covered by this
                        batch gets None/NaN in it.

        Returns:
            The list of row_ids that could not be placed because they
            are not in the table -- empty when every update landed.
            Previously such a row_id was skipped silently; the caller
            (AppController) now accumulates these per operation_id and
            tells the user how many results a run could not place.
        """
        # Mutate the stored frame in place -- no whole-table copy. This is the
        # primary write path (once per table per timer tick), and a copy here
        # would replace the DataFrame object every tick and force the row-id
        # index to rebuild. For rollback safety on a schema rejection we
        # snapshot only the columns this call touches: the ones it overwrites
        # (a Series copy each) and the ones it creates (dropped on rollback).
        df    = self._get_stored_table(table_name)
        index = self._row_index_for(table_name)

        touched_columns: set[str] = set()
        for col_updates in updates.values():
            touched_columns.update(col_updates.keys())
        column_snapshot: dict[str, pd.Series] = {
            col: df[col].copy() for col in touched_columns if col in df.columns
        }

        unplaceable: list[str] = []
        created_columns: list[str] = []
        try:
            # Column creation, the cell-write loop, infer_objects and the accept
            # are all inside one try: pandas' own setters raise (a Categorical
            # rejects a new label with TypeError, a numeric column rejects a
            # string), and that must degrade exactly like a schema rejection
            # rather than abort the QTimer drain.
            for col in touched_columns:
                if col not in df.columns:
                    df[col] = None
                    created_columns.append(col)
            col_locs = {
                col: df.columns.get_loc(col) for col in touched_columns
            }

            for row_id, col_updates in updates.items():
                pos = index.get(row_id)
                if pos is None:
                    unplaceable.append(row_id)
                    continue
                for col, val in col_updates.items():
                    df.iat[pos, col_locs[col]] = val

            # A column this call created was seeded with None and written cell
            # by cell, so infer_objects is what gives it its real kind (and so
            # a numeric column carries the "numeric" type tag, not "text").
            # infer_schema then keeps that dtype -- text stays open, bool stays
            # bool -- so the only pin still needed is: a whole-number first
            # batch must not lock a still-being-filled column to an integer
            # width, because a later drain may carry a decimal. Widen it to
            # float64.
            for col in created_columns:
                series = df[col].infer_objects()
                if getattr(series.dtype, "kind", None) in ("i", "u"):
                    df[col] = series.astype("float64")
                else:
                    df[col] = series

            self._accept_table(
                table_name, df, source="apply_row_updates"
            )
        except (SchemaRejection, TypeError, ValueError) as exc:
            # Roll back every in-place edit, including a partial one from a
            # write loop that failed midway. This stays correct at any point:
            # a created column is dropped whole no matter how many of its
            # cells were written, and a snapshot is the column exactly as it
            # was before this call touched a single cell of it.
            for col, original in column_snapshot.items():
                df[col] = original
            still_created = [c for c in created_columns if c in df.columns]
            if still_created:
                df.drop(columns=still_created, inplace=True)
            # A bad value in an operator's per-row results must not abort the
            # app: controller.py drains these on a QTimer with no try/except.
            # Downgrade to "none of this batch could be placed" -- the channel
            # AppController already uses to report dropped results -- and
            # record why. Under strict_schema we re-raise, so an operator's own
            # test still fails loudly: that is the whole point of the flag.
            if self.strict_schema:
                raise
            self._schema_messages.append(str(exc))
            return list(updates.keys())
        return unplaceable

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        name: str,
        source_table: str,
        group_by: str | list[str],
        aggregations: dict,
    ) -> None:
        """
        Creates a new table by grouping any existing table and applying
        aggregation functions.

        Args:
            name:         Name for the new table.
            source_table: Name of the table to aggregate from.
            group_by:     Column or list of columns to group by.
            aggregations: Dict mapping column names to aggregation functions.
        """
        # Step 1: Group the source table and apply the aggregation functions.
        source_df = self.get_table(source_table)
        agg_df    = source_df.groupby(group_by).agg(aggregations)
        agg_df    = agg_df.reset_index()

        # Step 2: Assign a new row_id to each aggregated row.
        agg_df["row_id"] = [self._next_id() for _ in range(len(agg_df))]

        # Step 3: Store the new table. P1.8d-2b-1: no ColumnTypeRegistry
        # write -- the schema the accept built carries every column's tag.
        self._accept_table(name, agg_df, source="aggregate")

        # Step 4: Record the operation in the provenance log.
        self.provenance.record("aggregate", {
            "name":         name,
            "source_table": source_table,
            "group_by":     group_by,
            "aggregations": aggregations,
        })

    def create_table_from_rows(
        self,
        name: str,
        row_ids: list[str],
        source_table: str = "frames",
    ) -> None:
        """
        Creates a new table by copying a subset of rows from an existing
        table. Copied rows keep their original row_ids (they are the same
        media items).

        Args:
            name:         Name for the new table.
            row_ids:      List of row_ids to include.
            source_table: Name of the source table.
        """
        # Read-only access to the stored table, then one copy of just
        # the subset -- not a full-table copy (P0.2a / P0.4). The rows
        # are returned in the caller's order, because the controller now
        # owns row order (P0.4) and "save filtered set" must store what
        # is on screen, including a randomised order.
        source_df = self.read_only_view(source_table)
        by_id = source_df.set_index("row_id", drop=False)
        present = [rid for rid in row_ids if rid in by_id.index]
        subset = by_id.loc[present].reset_index(drop=True)
        self._accept_table(name, subset, source="create_table_from_rows")
        self.provenance.record("create_table_from_rows", {
            "name":         name,
            "source_table": source_table,
            "n_rows":       len(subset),
        })

    def create_table_from_df(
        self,
        name: str,
        df: pd.DataFrame,
    ) -> None:
        """
        Creates a new table from a pre-built DataFrame returned by an
        operator's create_table() method. Generates new row_ids for
        each row and stores the result as a named table.

        Args:
            name: Name for the new table.
            df:   The DataFrame returned by the operator. Must not
                  already contain a row_id column.
        """
        result = df.copy().reset_index(drop=True)
        result.insert(
            0,
            "row_id",
            [self._next_id() for _ in range(len(result))],
        )
        # P1.8d-2b-1: no ColumnTypeRegistry write -- the schema the accept
        # built carries every column's display tag.
        self._accept_table(name, result, source="create_table_from_df")

        self.provenance.record("create_table_from_df", {
            "name":    name,
            "n_rows":  len(result),
            "columns": list(result.columns),
        })

    # ------------------------------------------------------------------
    # Table access
    # ------------------------------------------------------------------

    def schema_for(self, table_name: str) -> TableSchema | None:
        """Returns the TableSchema for a stored table, or None if the table
        was never accepted through _accept_table (e.g. a test assigned it
        straight into _tables)."""
        return self._schemas.get(table_name)

    def take_schema_messages(self) -> list[str]:
        """Returns the accumulated plain-English schema-adjustment notes and
        clears the list. The third reporting destination for accept-time
        dtype adjustments, alongside _schema_messages' provenance entry."""
        messages = self._schema_messages
        self._schema_messages = []
        return messages

    def get_table(self, name: str = "frames") -> pd.DataFrame:
        """
        Returns a copy of the named table as a DataFrame.

        Args:
            name: Table name. Defaults to 'frames'.

        Returns:
            A copy of the DataFrame for the named table.

        Raises:
            KeyError: If the table name does not exist.
        """
        return self._get_stored_table(name).copy()

    def read_only_view(self, table_name: str = "frames") -> pd.DataFrame:
        """
        Returns the stored table itself, with no copy, for callers that
        only read.

        Do not mutate the returned DataFrame under any circumstances --
        this may be Dataset's own live, stored table, not a copy. This is
        safe for QueryEngine specifically because "QueryEngine never
        mutates data" is a [NOW] rule (tests/test_dataset.py::
        test_apply_does_not_modify_dataframe), and it is the right choice
        for any other caller that only counts rows or lists columns.
        get_table() remains the correct choice for a caller that needs to
        mutate the result or hold its own copy.

        Args:
            table_name: Table name. Defaults to 'frames'.

        Returns:
            The live DataFrame for the named table -- not a copy.

        Raises:
            KeyError: If the table name does not exist.
        """
        return self._get_stored_table(table_name)

    def snapshot_rows(
        self,
        table_name: str = "frames",
        row_ids: list[str] | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Returns one controlled copy of a defined set of rows, for handing
        to a background worker (or any other caller that needs its own
        copy of less than the whole project).

        The returned DataFrame is the caller's own copy. Dataset never
        sees it again, so the caller may read, mutate, or hold onto it
        however it likes.

        Args:
            table_name: Table to read from. Defaults to 'frames'.
            row_ids:    Which rows to include, and in what order. None
                        means every row, in table order.
            columns:    Which columns to include. None means every
                        column.

        Returns:
            A new DataFrame containing exactly the requested rows, in
            the requested order -- one row per entry in row_ids, same
            length, same order. A caller that pairs this frame with
            row_ids by position (as OperatorRegistry's worker does) can
            rely on that.

        Raises:
            KeyError: If the table name does not exist, or if row_ids
                names a row_id that is not in the table. Silently
                dropping it here would leave the returned frame shorter
                than row_ids, and a caller pairing the two by position
                would then attribute every following row's data to the
                wrong row_id -- wrong output with no error, not merely a
                missing row. See docs/media_architecture.md section 3.6
                decision 11 for the same refuse-rather-than-quietly-fix
                principle applied to a malformed address.
        """
        df = self._get_stored_table(table_name)
        if columns is not None:
            df = df[columns]

        if row_ids is None:
            return df.copy()

        index = self._row_index_for(table_name)
        try:
            positions = [index[rid] for rid in row_ids]
        except KeyError as e:
            raise KeyError(
                f"snapshot_rows: row_id {e} is not in table '{table_name}'."
            ) from None
        return df.iloc[positions].copy()

    def list_tables(self) -> list[str]:
        """
        Returns the names of all tables, always starting with 'frames'.

        Returns:
            List of table names.
        """
        names = list(self._tables.keys())
        if "frames" in names:
            names.remove("frames")
            names = ["frames"] + names
        return names

    def get_row(self, row_id: str, table_name: str = "frames") -> dict:
        """
        Returns all column values for one row as a plain dictionary.

        Args:
            row_id:     The row to retrieve.
            table_name: The table containing the row.

        Returns:
            Dict of column name to value. Empty dict if not found.
        """
        df  = self._get_stored_table(table_name)
        pos = self._row_index_for(table_name).get(row_id)
        if pos is None:
            return {}
        return df.iloc[pos].to_dict()

    # ------------------------------------------------------------------
    # Save and load
    # ------------------------------------------------------------------

    def save(self, project_path: Path) -> None:
        """
        Saves tables as Parquet, provenance as JSON. Media-path columns
        are stored relative to project_path when possible; relative
        inputs are anchored to the current working directory first.

        Args:
            project_path: Path to the project folder.
        """
        project_path.mkdir(parents=True, exist_ok=True)

        # Count media cells that will not parse as an address, across every
        # table, so the "save" provenance entry can report them.
        unparseable_media_cells = 0

        for name, df in self._tables.items():
            # Which columns of THIS table hold media paths? 'full_path' always;
            # every other column the table's own TableSchema tags 'media_path'.
            # P1.8d-2b-1: this used to come from ColumnTypeRegistry; the schema
            # is now the single authority. A table assigned straight into
            # _tables by a test has no schema, so only 'full_path' is rewritten
            # for it -- the same columns the old registry-less path handled.
            media_cols = {"full_path"}
            schema = self.schema_for(name)
            if schema is not None:
                for spec in schema.columns_with_tag("media_path"):
                    media_cols.add(spec.name)

            df_out = df
            cols_to_rewrite = [c for c in df.columns if c in media_cols]
            if cols_to_rewrite:
                # Copy so we only rewrite paths on disk, not in memory.
                df_out = df_out.copy()
                for col in cols_to_rewrite:
                    # Each cell: parse -> absolutise against the working
                    # directory (anchors a relative input, as save() always
                    # has) -> relativise against the project -> format. See
                    # _rewrite_media_cell. The address fragment is preserved;
                    # only the path portion moves.
                    new_values, bad = _rewrite_media_column(
                        list(df_out[col]), project_path, to_stored=True
                    )
                    df_out[col] = new_values
                    unparseable_media_cells += bad
            df_out.to_parquet(project_path / f"{name}.parquet")

        # Store each stored table's declared TableSchema (roles,
        # carry_to_children, dtypes, column order) so a future load() can
        # restore it exactly instead of re-inferring. A table assigned
        # straight into _tables by a test has no schema (schema_for() returns
        # None) and is simply left out.
        table_schemas = {}
        for name in self._tables:
            schema = self.schema_for(name)
            if schema is not None:
                table_schemas[name] = schema.to_dict()
        if table_schemas:
            # The file carries an integer format version of its own, next to
            # the per-table schemas. It belongs to the FILE, not to
            # TableSchema.to_dict() -- that is a value object and must not know
            # it is being written to disk. The version exists so that if a
            # later Gelem changes the schema encoding, load() can detect an
            # older file by its version and re-infer that project rather than
            # refuse to open it.
            schemas_file = {
                "format_version": 1,
                "schemas": table_schemas,
            }
            (project_path / "schemas.json").write_text(
                json.dumps(schemas_file, indent=2)
            )

        # P1.8d-2b-1: column_types.json is retired. schemas.json (written
        # above) now carries every column's display tag, and load() restores
        # each table's tags from it. save() no longer writes the sidecar at
        # all; a project saved by an earlier Gelem still has one and load()
        # reads it as a fallback when there is no schemas.json.

        (project_path / "provenance.json").write_text(
            json.dumps(self.provenance.to_list(), indent=2)
        )
        # NOTE: provenance.json is written just above, before this record()
        # call, so this particular "save" entry is not in the file it just
        # wrote -- a pre-existing known defect (CLAUDE.md, "Data ownership").
        # The unparseable count is still correct in the in-memory log and in
        # the next save.
        self.provenance.record("save", {
            "project_path": str(project_path),
            "unparseable_media_cells": unparseable_media_cells,
        })

    def _read_saved_schemas(
        self, project_path: Path
    ) -> tuple[dict[str, TableSchema], str | None]:
        """Read schemas.json (written by save() since P1.8c-2a) if it is
        present. Mutates nothing.

        Returns (schemas, ignored_message):
          * ({table_name: TableSchema, ...}, None) -- format_version 1, parsed
            cleanly. The dict may be empty if the file listed no schemas.
          * ({}, "<one sentence>") -- the file is there but this build will not
            use it: an unrecognised format_version, JSON that will not parse, or
            a schema entry schema_from_dict rejects. The caller loads every
            table by inference and surfaces the sentence -- a corrupt or
            unknown-version sidecar must never make a project unopenable,
            because the parquet files still hold the data.
          * ({}, None) -- no schemas.json at all. Every table is inferred,
            silently: this is every project saved before P1.8c-2a.
        """
        path = project_path / "schemas.json"
        if not path.exists():
            return {}, None

        # json.loads raises JSONDecodeError (a ValueError subclass) on bad JSON;
        # read_text can raise OSError. Either way the sidecar is unusable.
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            return {}, (
                f"schemas.json could not be read ({exc}); every table was "
                f"loaded by inference instead."
            )

        version = raw.get("format_version") if isinstance(raw, dict) else None
        if version != 1:
            return {}, (
                f"schemas.json has format_version {version!r}, which this build "
                f"of Gelem does not know; every table was loaded by inference "
                f"instead."
            )

        serialised = raw.get("schemas", {})
        if not isinstance(serialised, dict):
            return {}, (
                "schemas.json is malformed (its 'schemas' is not an object); "
                "every table was loaded by inference instead."
            )

        schemas: dict[str, TableSchema] = {}
        try:
            for table_name, payload in serialised.items():
                schemas[table_name] = schema_from_dict(payload)
        except SchemaSerialisationError as exc:
            # schema_from_dict's contract is that every malformed payload raises
            # SchemaSerialisationError, so this one catch is enough.
            return {}, (
                f"schemas.json is malformed ({exc}); every table was loaded by "
                f"inference instead."
            )
        return schemas, None

    def load(self, project_path: Path) -> None:
        """
        Loads a previously saved project from disk, atomically. Every parquet is
        read and validated against its saved schema BEFORE any in-memory state
        is touched, so a project with a bad table leaves the currently open
        project exactly as it was -- same tables, schemas, provenance log and id
        counter. On success this replaces all tables, the provenance log and the
        id counter with the saved project's. Relative media paths are resolved
        back to absolute against project_path.

        schemas.json (written by save() since P1.8c-2a) is honoured when its
        format_version is one this build knows; an unknown version or a corrupt
        file is ignored with a message and every table is loaded by inference.
        A table with no entry in schemas.json is inferred. No schema from the
        previously open project is ever consulted -- see _prepare_table's
        consult_stored_schema argument.

        Each restored schema also carries every column's display tag, so a
        loaded project's media columns come from schemas.json. P1.8d-2b-1
        retired column_types.json: save() no longer writes it, and load() reads
        it only as a fallback for a project saved before P1.8c-2a (no
        schemas.json). A missing column_types.json is never an error.

        Args:
            project_path: Path to an existing project folder.

        Raises:
            FileNotFoundError: If project_path doesn't exist or has no parquet
                files.
            SchemaRejection: If a parquet file does not conform to its schema.
                The previously open project is left completely untouched.
        """
        # -----------------------------------------------------------------
        # Before the point of no return: read and validate everything.
        # Nothing in this half mutates a single field of self.
        # -----------------------------------------------------------------
        if not project_path.exists():
            raise FileNotFoundError(
                f"Project folder '{project_path}' does not exist."
            )
        parquet_files = list(project_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No tables (.parquet) found in '{project_path}'."
            )

        # Saved schemas, if any and if this build understands them.
        restored_schemas, ignored_schemas_message = self._read_saved_schemas(
            project_path
        )

        # Which columns hold media paths? 'full_path' always. Every other
        # media column comes from the restored schemas.
        # P1.8d-2b-1: column_types.json is retired. It is read here ONLY as a
        # fallback for a project with no usable schemas.json -- any project
        # saved before P1.8c-2a. A missing column_types.json is normal and is
        # never an error or a message.
        media_cols = {"full_path"}
        if restored_schemas:
            for schema in restored_schemas.values():
                for spec in schema.columns_with_tag("media_path"):
                    media_cols.add(spec.name)
        else:
            ct_path = project_path / "column_types.json"
            if ct_path.exists():
                legacy_types = json.loads(ct_path.read_text())
                for col, tag in legacy_types.items():
                    if tag == "media_path":
                        media_cols.add(col)

        # Saved provenance log -- read now, install only after the point of no
        # return, so a load that fails below does not replace the log.
        saved_provenance = None
        prov = project_path / "provenance.json"
        if prov.exists():
            saved_provenance = json.loads(prov.read_text())

        load_hints = {
            col: ColumnHint(type_tag="media_path") for col in media_cols
        }
        unparseable_media_cells = 0
        # Per-table notes for an empty table whose saved schema could not be
        # cast onto it -- appended to _schema_messages after the point of no
        # return, like ignored_schemas_message.
        cast_fallback_messages: list[str] = []

        # Parse and validate every parquet into a staged _PreparedTable. A
        # SchemaRejection raised here propagates out with self untouched.
        prepared_tables: list[_PreparedTable] = []
        for path in parquet_files:
            df = pd.read_parquet(path)
            for col in df.columns:
                if col in media_cols:
                    # Each cell: parse -> absolutise against the project
                    # root -> format. See _rewrite_media_cell. The address
                    # fragment is preserved; only the path portion moves.
                    new_values, bad = _rewrite_media_column(
                        list(df[col]), project_path, to_stored=False
                    )
                    df[col] = new_values
                    unparseable_media_cells += bad
            # The saved schema for this table when schemas.json named it, else
            # None. Either way consult_stored_schema is False: the previous
            # project's schema for a same-named table must not leak in.
            declared = restored_schemas.get(path.stem)
            if declared is not None and len(df) == 0:
                # A zero-row parquet does not carry trustworthy column dtypes
                # (pyarrow returns an empty text column as float64). Cast the
                # empty frame to the saved schema's dtypes BEFORE validating --
                # an empty frame casts losslessly, and check_frame rejects on
                # kind before it inspects values, so the cast must precede the
                # check. If a cast is impossible, fall back to inference for
                # this one table.
                try:
                    df = _cast_empty_frame_to_schema(df, declared)
                except (TypeError, ValueError) as exc:
                    cast_fallback_messages.append(
                        f"Table {path.stem!r}: its saved schema could not be "
                        f"applied to the empty table ({exc}); it was loaded by "
                        f"inference instead."
                    )
                    declared = None
            prepared = self._prepare_table(
                path.stem,
                df,
                hints=load_hints,
                schema=declared,
                consult_stored_schema=False,
                source="load",
            )
            # Stage, do not store. Deleting this line and committing here
            # instead would let a load that fails on a later parquet leave a
            # half-replaced project.
            prepared_tables.append(prepared)

        # -----------------------------------------------------------------
        # POINT OF NO RETURN. Every parquet above parsed and validated;
        # nothing below reads project data or can fail on it. Only now is
        # the previously open project discarded.
        # -----------------------------------------------------------------
        self._reset_tables({})
        # P1.8d-2b-1: Dataset no longer touches ColumnTypeRegistry. A project
        # opened second gets its column tags from the schemas the commit loop
        # below installs; there is no registry column map for Dataset to clear.
        # Restore the saved provenance log before the commit loop, so any
        # "schema_adjustments" entry _commit_prepared records is appended to the
        # restored log rather than wiped by a later replace().
        if saved_provenance is not None:
            self.provenance.replace(saved_provenance)
        for prepared in prepared_tables:
            self._commit_prepared(prepared)

        # Restore _id_counter (assumes int-parseable row_id from _next_id()).
        max_id = 0
        for df in self._tables.values():
            if "row_id" in df.columns and not df.empty:
                max_id = max(max_id, int(df["row_id"].astype(int).max()))
        self._id_counter = max_id

        # An empty table whose saved schema could not be cast onto it was
        # loaded by inference -- tell the researcher, one message per table.
        self._schema_messages.extend(cast_fallback_messages)

        # An ignored schemas.json is surfaced as a schema message and in the
        # load provenance entry, so an older or newer project still opens and
        # the researcher can see the schemas were re-inferred.
        if ignored_schemas_message is not None:
            self._schema_messages.append(ignored_schemas_message)
            schemas_note = ignored_schemas_message
        elif restored_schemas:
            schemas_note = f"restored saved schemas for {sorted(restored_schemas)}"
        else:
            schemas_note = "no schemas.json; every table loaded by inference"

        self.provenance.record("load", {
            "project_path": str(project_path),
            "unparseable_media_cells": unparseable_media_cells,
            "schemas": schemas_note,
        })
