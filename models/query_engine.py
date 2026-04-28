"""
models/query_engine.py

QueryEngine determines which rows should be shown and how they should
be organised. It takes a DataFrame as input and returns an ordered list
of row_ids. It never modifies any table, writes to disk, or changes
any state.

Student B is responsible for implementing the real logic in this file.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

@dataclass
class Filter:
    """
    Represents one filtering condition applied to a table column.

    When multiple filters are active, QueryEngine applies all of them
    in sequence — only rows satisfying every filter are returned (AND logic).

    Examples:
        Filter("condition", "eq", "B")
            — keep only rows where condition equals "B"

        Filter("timestamp", "gte", 1.0)
            — keep only rows where timestamp >= 1.0

        Filter("session_id", "isin", ["S01", "S02"])
            — keep only rows where session_id is S01 or S02

        Filter("notes", "contains", "smile")
            — keep only rows where notes contains the substring "smile"
              (case-insensitive)
    """

    column: str
    """The name of the column to filter on."""

    comparison: str
    """
    The type of comparison to make. One of:
        'eq'       — equal to value
        'neq'      — not equal to value
        'lt'       — less than value
        'gt'       — greater than value
        'lte'      — less than or equal to value
        'gte'      — greater than or equal to value
        'isin'     — value is in the list (value must be a list)
        'between'  — value is between value[0] and value[1] inclusive
                     (value must be a list of two items)
        'contains' — column value contains the substring (value must be
                     a string; comparison is case-insensitive)
    """

    value: Any
    """The value or values to compare against."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies this filter to a DataFrame and returns the matching rows.

        Args:
            df: The DataFrame to filter.

        Returns:
            A filtered DataFrame containing only rows that satisfy
            this filter condition.

        TODO (Student B): Implement each comparison type. Use pandas
        boolean indexing: df[mask] where mask is a boolean Series.
        Example for 'eq': mask = df[self.column] == self.value

        Raise a clear ValueError if self.comparison is not one of the
        supported types listed above.
        """
        # PLACEHOLDER: returns all rows unfiltered.
        if self.column not in df.columns:
            return df.iloc[0:0]

        if self.comparison == "eq":
            return df[df[self.column] == self.value]
        elif self.comparison == "neq":
            return df[df[self.column] != self.value]
        elif self.comparison == "lt":
            return df[df[self.column] < self.value]
        elif self.comparison == "gt":
            return df[df[self.column] > self.value]
        elif self.comparison == "lte":
            return df[df[self.column] <= self.value]
        elif self.comparison == "gte":
            return df[df[self.column] >= self.value]
        elif self.comparison == "isin":
            return df[df[self.column].isin(self.value)]
        elif self.comparison == "between":
            return df[df[self.column].between(self.value[0], self.value[1])]
        elif self.comparison == "contains":
            mask = df[self.column].astype(str).str.contains(
                str(self.value), case=False, na=False
            )
            return df[mask]
        else:
            raise ValueError(
                f"Unknown comparison type '{self.comparison}'. "
                f"Must be one of: eq, neq, lt, gt, lte, gte, "
                f"isin, between, contains."
            )


# ---------------------------------------------------------------------------
# QueryEngine
# ---------------------------------------------------------------------------

class QueryEngine:
    """
    Determines which rows should be shown and how they should be organised.

    QueryEngine is stateless — it holds no data of its own. Every method
    takes a DataFrame as input and returns a result without modifying
    anything. The same QueryEngine instance can be used for any table.
    """

    def apply(
        self,
        df: pd.DataFrame,
        filters: list[Filter] | None = None,
        sort_by: str | None = None,
        ascending: bool = True,
        randomise: bool = False,
        seed: int | None = None,
    ) -> list[str]:
        """
        Applies filters and ordering to a DataFrame. Returns an ordered
        list of row_ids for the standard (single panel) gallery view.

        Processing order:
            1. Apply all filters in sequence (AND logic)
            2. Sort or shuffle the result
            3. Extract and return the row_id column as a list

        Args:
            df:        Any table from Dataset. Must have a 'row_id' column.
            filters:   Zero or more Filter objects applied in sequence.
            sort_by:   Column name to sort by. Ignored if randomise=True.
            ascending: Sort direction.
            randomise: If True, shuffle the result randomly.
            seed:      Random seed for reproducibility when randomise=True.

        Returns:
            An ordered list of row_id strings.

        TODO (Student B): Apply each filter by calling filter.apply(df)
        in sequence. Then sort or shuffle. Then return list(df['row_id']).
        """
        # PLACEHOLDER: returns all row_ids in original order.
        result = df.copy()

        if filters:
            for f in filters:
                result = f.apply(result)

        if randomise:
            result = result.sample(frac=1, random_state=seed)
        elif sort_by and sort_by in result.columns:
            result = result.sort_values(sort_by, ascending=ascending)

        return list(result["row_id"])

    def apply_grouped(
        self,
        df: pd.DataFrame,
        group_by: str,
        filters: list[Filter] | None = None,
        sort_by: str | None = None,
        ascending: bool = True,
        randomise: bool = False,
        seed: int | None = None,
    ) -> dict[str, list[str]]:
        """
        Applies filters, then splits rows into groups by the group_by
        column, then sorts or shuffles within each group.

        Returns a dict mapping each group value to an ordered list of
        row_ids.

        Args:
            df:        Any table from Dataset.
            group_by:  Column name to split into groups.
            filters:   Filters applied before grouping.
            sort_by:   Column to sort within each group.
            ascending: Sort direction within each group.
            randomise: If True, shuffle within each group.
            seed:      Random seed.

        Returns:
            A dict mapping group value (as string) to list of row_ids.

        TODO (Student B): Implement this method.
        """
        # PLACEHOLDER
        result = df.copy()

        if filters:
            for f in filters:
                result = f.apply(result)

        if group_by not in result.columns:
            return {"all": list(result["row_id"])}

        group_values = sorted(result[group_by].dropna().unique())

        grouped: dict[str, list[str]] = {}
        for i, gval in enumerate(group_values):
            group_df = result[result[group_by] == gval].copy()

            if randomise:
                group_seed = None if seed is None else seed + i
                group_df   = group_df.sample(frac=1, random_state=group_seed)
            elif sort_by and sort_by in group_df.columns:
                group_df = group_df.sort_values(sort_by, ascending=ascending)

            grouped[str(gval)] = list(group_df["row_id"])

        return grouped

    def get_group_values(self, df: pd.DataFrame, column: str) -> list:
        """
        Returns the sorted list of unique values in a column.
        Used by the UI to populate filter controls and the group-by
        column selector.

        Args:
            df:     The table to inspect.
            column: The column whose unique values to return.

        Returns:
            Sorted list of unique values. Empty list if column does
            not exist.

        TODO (Student B): Use df[column].dropna().unique() and sort.
        """
        # PLACEHOLDER
        if column not in df.columns:
            return []
        return sorted(df[column].dropna().unique().tolist())

    def select_first_after(
        self,
        df: pd.DataFrame,
        time_col: str,
        threshold: float,
        group_col: str,
    ) -> list[str]:
        """
        Within each group defined by group_col, selects the single row
        with the smallest value in time_col that is >= threshold.

        Args:
            df:        The table, already filtered to relevant rows.
            time_col:  Name of the time column.
            threshold: Minimum value to accept.
            group_col: Column that defines groups.

        Returns:
            An ordered list of row_ids, one per group.

        TODO (Student B): Implement this method.
        """
        # PLACEHOLDER
        if time_col not in df.columns or group_col not in df.columns:
            return list(df["row_id"])

        above = df[df[time_col] >= threshold].copy()
        if above.empty:
            return []

        idx      = above.groupby(group_col)[time_col].idxmin()
        selected = above.loc[idx]
        return list(selected["row_id"])
