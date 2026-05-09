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
import pandas as pd


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
        # Reset the table and counter for a fresh load.
        self._id_counter = 0
        self._tables["frames"] = pd.DataFrame(
            columns=self.FRAMES_REQUIRED_COLUMNS
        )

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
            self._tables["frames"] = pd.DataFrame(rows)
        else:
            self._tables["frames"] = pd.DataFrame(columns=self.FRAMES_REQUIRED_COLUMNS)

        
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
        # Reset the table and counter for a fresh load.
        self._id_counter = 0
        self._tables["frames"] = pd.DataFrame(
            columns=self.FRAMES_REQUIRED_COLUMNS
        )

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

        self._tables["frames"] = pd.DataFrame(rows)

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

        # Step 2: apply preproccesing rules to the keys if needed.
        if preprocess is not None:
            pass # TODO: apply preprocessing rules TBD on.

        # Step 3: Perform a left join of csv_df onto self._tables["frames"] using the specified join_on column and file_name.
        frames_df = self._tables["frames"]
        joined = frames_df.merge(
            csv_df,
            left_on="file_name",
            right_on=join_on,
            how="left",
        )

        # Step 4: Calculate statistics and build the report.
        new_columns = [c for c in csv_df.columns if c != join_on]

        if new_columns:
            matched_mask = joined[new_columns[0]].notna()
        else:
            matched_mask = pd.Series([False] * len(joined))

        unmatched_files = list(joined.loc[~matched_mask, "file_name"])
        matched_keys    = set(frames_df.loc[matched_mask.values, "file_name"])
        unmatched_csv   = list(
            csv_df.loc[~csv_df[join_on].isin(matched_keys), join_on].astype(str)
        )

        # Duplicate detection
        file_counts     = frames_df["file_name"].value_counts()
        duplicate_files = list(file_counts[file_counts > 1].index)

        csv_counts    = csv_df[join_on].value_counts()
        duplicate_csv = list(csv_counts[csv_counts > 1].index.astype(str))

        frames_keys = set(frames_df["file_name"])
        one_to_many = [k for k in duplicate_csv if k in frames_keys]

        report = MergeReport(
            total_csv_rows=len(csv_df),
            total_image_files=len(frames_df),
            matched_rows=int(matched_mask.sum()),
            unmatched_files=unmatched_files,
            unmatched_csv_rows=unmatched_csv,
            duplicate_keys_files=duplicate_files,
            duplicate_keys_csv=duplicate_csv,
            one_to_many=one_to_many,
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
            self._tables["frames"] = report._pending_df.copy()

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
        df = self.get_table(table_name)
        df[name] = df.eval(expression)
        self._tables[table_name] = df

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
        df = self.get_table(table_name)
        df[name] = df["row_id"].map(values)
        self._tables[table_name] = df
        self._register_column(name, col_type)

    def update_row(
        self,
        row_id: str,
        updates: dict,
        table_name: str = "frames",
    ) -> None:
        """
        Updates a single row with new column values.
        Called by AppController on the main thread to apply progressive
        operator results one item at a time.
        Never called from a background thread directly.

        Args:
            row_id:     The row to update.
            updates:    Dict of column name to new value.
            table_name: Table containing the row.

        TODO (Student B): Implement this method.
        """
        df = self._tables[table_name]
        mask = df["row_id"] == row_id
        for col, val in updates.items():
            if col not in df.columns:
                df[col] = None
            df.loc[mask, col] = val

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

        TODO (Student B): Implement this method.
        """
        # PLACEHOLDER
        source_df = self.get_table(source_table)
        agg_df = pd.DataFrame({
            "row_id": [self._next_id()],
            "note":   [f"aggregated from {source_table} — not yet implemented"],
        })
        self._tables[name] = agg_df
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
        table. The new table gets its own fresh row_ids.

        Args:
            name:         Name for the new table.
            row_ids:      List of row_ids to include.
            source_table: Name of the source table.

        TODO (Student B): Implement this method.
        """
        # PLACEHOLDER
        source_df = self.get_table(source_table)
        subset = source_df[source_df["row_id"].isin(row_ids)].copy()
        subset = subset.reset_index(drop=True)
        subset["row_id"] = [self._next_id() for _ in range(len(subset))]
        self._tables[name] = subset
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
        self._tables[name] = result

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
        if name not in self._tables:
            raise KeyError(f"Table '{name}' does not exist in this project.")
        return self._tables[name].copy()

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
        df = self.get_table(table_name)
        mask = df["row_id"] == row_id
        rows = df[mask]
        if rows.empty:
            return {}
        return rows.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # Save and load
    # ------------------------------------------------------------------

    def save(self, project_path: Path) -> None:
        """
        Saves all tables as Parquet files and the provenance log as
        JSON to the specified project folder.

        Args:
            project_path: Path to the project folder.

        TODO (Student B): Implement this method.
        """
        # PLACEHOLDER
        project_path.mkdir(parents=True, exist_ok=True)
        print(f"[Dataset] save() — not yet implemented. "
              f"Would save to {project_path}")

    def load(self, project_path: Path) -> None:
        """
        Loads a previously saved project from disk.

        Args:
            project_path: Path to an existing project folder.

        TODO (Student B): Implement this method.
        """
        # PLACEHOLDER
        print(f"[Dataset] load() — not yet implemented. "
              f"Would load from {project_path}")
