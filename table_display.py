"""
table_display.py

Which columns the gallery shows for a table, and the default to fall
back on when the researcher has not chosen one yet.

Split out of controller.py (CC-23) because AppController._visible_cols
was a single global list, reset to None by set_active_table, load_folder
and load_csv_as_primary alike -- so leaving a table and coming back lost
the researcher's choice, and a table whose media column was not literally
named full_path opened with an empty gallery even when its schema tagged
that column media_path. TableDisplayState fixes the first: it remembers a
choice per table name, not one choice for the whole session.
resolve_default_visible_columns fixes the second: it falls back to
whatever media_path column the table's schema actually declares, not a
hardcoded name match.

Standard library only -- no Qt, no pandas, no Dataset -- for the same
reason as table_names.py: a plain top-level module with no package
__init__.py in its import path, so nothing can later make ui/ pull in
pandas transitively through here.
"""

from __future__ import annotations


class TableDisplayState:
    """Remembers the researcher's chosen visible columns, per table name.

    Nothing here is persisted -- it lives only for the running session,
    the same way the field it replaces (AppController._visible_cols) did.
    """

    def __init__(self) -> None:
        self._chosen: dict[str, list[str]] = {}

    def remember(self, table_name: str, columns: list[str] | None) -> None:
        """
        Records *columns* as the chosen visible columns for *table_name*.
        ``None`` forgets that table's choice, returning it to the unset
        state recall() reports as ``None``. Stores a copy, so a caller
        mutating the list it passed in afterwards does not change what
        is remembered.
        """
        if columns is None:
            self._chosen.pop(table_name, None)
        else:
            self._chosen[table_name] = list(columns)

    def recall(self, table_name: str) -> list[str] | None:
        """
        The remembered choice for *table_name*, or ``None`` if none has
        been made. Returns a copy, so a caller mutating the result does
        not change what is remembered.
        """
        remembered = self._chosen.get(table_name)
        if remembered is None:
            return None
        return list(remembered)

    def forget_all(self) -> None:
        """
        Clears every remembered choice -- for when the whole dataset is
        being replaced, so a new project does not inherit another
        project's choices.
        """
        self._chosen.clear()


def resolve_default_visible_columns(
    visual_columns: list[str], default_name: str
) -> list[str]:
    """
    The columns a gallery shows for a table with no remembered
    preference: ``[default_name]`` if that name is among
    *visual_columns*, otherwise a one-element list holding the first
    name in *visual_columns*, otherwise -- a table with no visual column
    at all -- an empty list.

    *visual_columns* must already be in the table's own column order;
    this function does not sort or otherwise reorder it.
    """
    if default_name in visual_columns:
        return [default_name]
    if visual_columns:
        return [visual_columns[0]]
    return []
