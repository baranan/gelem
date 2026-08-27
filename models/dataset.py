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

from media.media_address import (
    MediaAddressError,
    absolutise,
    relativise,
)
from media.media_address import parse as parse_address
from media.media_address import format as format_address


# ---------------------------------------------------------------------------
# Media extensions supported by Gelem
# ---------------------------------------------------------------------------

# Maps file extension (lowercase) to the column type tag it produces.
# Add new extensions here to support additional media formats.
# The column type tag must be registered in ColumnTypeRegistry.
MEDIA_EXTENSIONS: dict[str, str] = {
    # Images
    ".jpg":  "media_path",
    ".jpeg": "media_path",
    ".png":  "media_path",
    ".bmp":  "media_path",
    ".tiff": "media_path",
    ".tif":  "media_path",
    # Videos
    ".mp4":  "media_path",
    ".mov":  "media_path",
    ".avi":  "media_path",
    ".mkv":  "media_path",
    ".webm": "media_path",
    # Future: audio
    # ".wav":  "audio_path",
    # ".mp3":  "audio_path",
}


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

    def __init__(self):
        self._tables: dict[str, pd.DataFrame] = {
            "frames": pd.DataFrame(columns=self.FRAMES_REQUIRED_COLUMNS)
        }
        self.provenance = ProvenanceLog()
        self._id_counter: int = 0
        self._registry = None

        # row_id -> positional index, one dict per table. Lazily built and
        # self-healing -- see _row_index_for().
        self._row_index: dict[str, dict[str, int]] = {}
        # table_name -> (row count, weakref to the DataFrame) the index
        # above was built from. See _row_index_for() for how this is used.
        self._row_index_stamp: dict[str, tuple[int, weakref.ReferenceType]] = {}

    def set_registry(self, registry) -> None:
        """
        Stores a reference to the ColumnTypeRegistry so Dataset can
        register column types when new columns are added.

        Args:
            registry: The ColumnTypeRegistry instance.
        """
        self._registry = registry

    def _next_id(self) -> str:
        """Generates a new unique row_id string."""
        self._id_counter += 1
        return f"{self._id_counter:06d}"

    def _register_column(self, column_name: str, col_type: str) -> None:
        """
        Registers a column with ColumnTypeRegistry if available.

        Args:
            column_name: The column name to register.
            col_type:    The column type tag, e.g. 'media_path', 'numeric'.
        """
        if self._registry is not None:
            self._registry.register_by_tag(column_name, col_type)

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
    # An in-place mutation that changes column values without touching
    # row_id or row count (apply_row_updates, add_computed_column,
    # add_column) leaves the stamp unchanged, which is correct -- row
    # positions did not move, so the cached index is still exactly
    # right. A write that replaces the table with a different DataFrame
    # object (aggregate, create_table_from_df, confirm_merge, a test's
    # direct ds._tables[...] = ... assignment) changes the stamp, so the
    # very next lookup rebuilds rather than trusting stale positions.
    # _set_table() and _reset_tables() below exist so every write goes
    # through one obvious place, not because they need to do anything
    # extra to keep the index correct.

    def _get_stored_table(self, table_name: str) -> pd.DataFrame:
        """Returns the live, stored DataFrame for table_name (no copy).
        Internal only -- callers that need their own copy use get_table()
        or snapshot_rows(); callers that only read use read_only_view()."""
        if table_name not in self._tables:
            raise KeyError(f"Table '{table_name}' does not exist in this project.")
        return self._tables[table_name]

    def _set_table(self, table_name: str, df: pd.DataFrame) -> None:
        """The single place one stored table is written or replaced."""
        self._tables[table_name] = df

    def _reset_tables(self, tables: dict[str, pd.DataFrame]) -> None:
        """The single place the whole _tables dict is replaced, e.g. by a
        fresh load_folder(), load_csv_as_primary(), or load(). Clears the
        index caches too -- not required for correctness (the stamp check
        in _row_index_for would catch every one of these tables being a
        new object anyway), but a table dropped by the reset (e.g. one
        that existed only in the previous project) would otherwise leave
        a dead entry sitting in these dicts forever."""
        self._tables = tables
        self._row_index.clear()
        self._row_index_stamp.clear()

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
        full_path, and file_name. Registers full_path as 'media_path'
        with ColumnTypeRegistry.

        Supported formats are defined in the MEDIA_EXTENSIONS dict at
        the top of this file. To add a new format, add its extension
        there — no other changes are needed.

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
            self._set_table("frames", pd.DataFrame(rows))
        else:
            self._set_table("frames", pd.DataFrame(columns=self.FRAMES_REQUIRED_COLUMNS))

        
        # Register full_path as media_path — works for images and videos.
        self._register_column("full_path", "media_path")

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

        self._set_table("frames", pd.DataFrame(rows))

        if image_column and image_column in csv_df.columns:
            self._register_column("full_path", "media_path")

        for col in csv_df.columns:
            if self._registry is not None:
                inferred = self._registry.infer_type(csv_df[col])
                self._register_column(col, inferred)

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
            self._set_table("frames", report._pending_df.copy())

        for col in report._new_columns:
            if self._registry is not None:
                inferred = self._registry.infer_type(self._tables["frames"][col])
            else:
                inferred = "text"
            self._register_column(col, inferred)

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
        df = self._get_stored_table(table_name)
        df[name] = df.eval(expression)
        self._set_table(table_name, df)

        self._register_column(name, col_type)
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
        df = self._get_stored_table(table_name)
        df[name] = df["row_id"].map(values)
        self._set_table(table_name, df)
        self._register_column(name, col_type)

    def update_row(
        self,
        row_id: str,
        updates: dict,
        table_name: str = "frames",
    ) -> None:
        """
        Updates a single row with new column values. A convenience
        wrapper over apply_row_updates() for the one-row case.
        Called by AppController on the main thread to apply progressive
        operator results one item at a time.
        Never called from a background thread directly.

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
    ) -> None:
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
                        new value. A row_id not present in the table is
                        silently skipped, matching update_row()'s prior
                        behaviour (a no-op mask match). A column that
                        does not exist yet is created first and every
                        row not covered by this batch gets None/NaN in
                        it, matching update_row()'s prior behaviour too.
        """
        df    = self._get_stored_table(table_name)
        index = self._row_index_for(table_name)

        # Create every new column up front so the per-row loop below is
        # pure positional writes, not repeated column creation.
        touched_columns: set[str] = set()
        for col_updates in updates.values():
            touched_columns.update(col_updates.keys())
        for col in touched_columns:
            if col not in df.columns:
                df[col] = None
        col_locs = {col: df.columns.get_loc(col) for col in touched_columns}

        for row_id, col_updates in updates.items():
            pos = index.get(row_id)
            if pos is None:
                continue
            for col, val in col_updates.items():
                df.iat[pos, col_locs[col]] = val

        self._set_table(table_name, df)

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

        # Step 3: Store the new table.
        self._set_table(name, agg_df)

        # Step 4: Register the column types for the new table.
        for col in agg_df.columns:
            if col == "row_id":
                continue
            if self._registry is not None:
                inferred = self._registry.infer_type(agg_df[col])
            else:
                inferred = "text"
            self._register_column(col, inferred)

        # Step 5: Record the operation in the provenance log.
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
        self._set_table(name, subset)
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
        self._set_table(name, result)

        if self._registry is not None:
            for col in result.columns:
                if col == "row_id":
                    continue
                inferred = self._registry.infer_type(result[col])
                self._register_column(col, inferred)

        self.provenance.record("create_table_from_df", {
            "name":    name,
            "n_rows":  len(result),
            "columns": list(result.columns),
        })

    # ------------------------------------------------------------------
    # Table access
    # ------------------------------------------------------------------

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

        # Which columns hold media paths? full_path always; registry adds the rest.
        media_cols = {"full_path"}
        if self._registry is not None:
            for col in self._registry.list_all_columns():
                ct = self._registry.get(col)
                if ct is not None and ct.tag == "media_path":
                    media_cols.add(col)

        # Count media cells that will not parse as an address, across every
        # table, so the "save" provenance entry can report them.
        unparseable_media_cells = 0

        for name, df in self._tables.items():
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

        # Store the column -> tag map so load() restores types exactly.
        if self._registry is not None:
            column_types = {}
            for col in self._registry.list_all_columns():
                ct = self._registry.get(col)
                if ct is not None:
                    column_types[col] = ct.tag
            if column_types:
                (project_path / "column_types.json").write_text(
                    json.dumps(column_types, indent=2)
                )

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

    def load(self, project_path: Path) -> None:
        """
        Loads a previously saved project from disk. Replaces all in-memory
        tables and the provenance log with the saved ones (same "open
        project" pattern as load_folder / load_csv_as_primary). Relative
        `full_path` values are resolved back to absolute against
        project_path.

        Args:
            project_path: Path to an existing project folder.

        Raises:
            FileNotFoundError: If project_path doesn't exist or has no
                parquet files.
        """
        if not project_path.exists():
            raise FileNotFoundError(
                f"Project folder '{project_path}' does not exist."
            )
        parquet_files = list(project_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No tables (.parquet) found in '{project_path}'."
            )

        # Read column types first so we know which columns are media paths.
        column_types = {}
        ct_path = project_path / "column_types.json"
        if ct_path.exists():
            column_types = json.loads(ct_path.read_text())
        media_cols = {col for col, tag in column_types.items() if tag == "media_path"}
        media_cols.add("full_path")  # fallback if no sidecar (saved without registry)

        new_tables: dict[str, pd.DataFrame] = {}
        unparseable_media_cells = 0
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
            new_tables[path.stem] = df
        self._reset_tables(new_tables)

        # Restore _id_counter (assumes int-parseable row_id from _next_id()).
        max_id = 0
        for df in self._tables.values():
            if "row_id" in df.columns and not df.empty:
                max_id = max(max_id, int(df["row_id"].astype(int).max()))
        self._id_counter = max_id

        prov = project_path / "provenance.json"
        if prov.exists():
            self.provenance.replace(json.loads(prov.read_text()))

        if self._registry is not None:
            if column_types:
                for col, tag in column_types.items():
                    try:
                        self._register_column(col, tag)
                    except KeyError:
                        pass  # tag unknown in this build; skip rather than sink the whole load
            elif "full_path" in self._tables.get("frames", pd.DataFrame()).columns:
                # No sidecar — fall back to load_folder's default tagging.
                self._register_column("full_path", "media_path")

        self.provenance.record("load", {
            "project_path": str(project_path),
            "unparseable_media_cells": unparseable_media_cells,
        })
