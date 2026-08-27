"""
models/notifications.py

Frozen payload objects for the controller's batched UI notifications.

Like `models/query_result.py`, this module deliberately imports no
pandas, numpy, PIL or cv2, so UI files may import from it directly
without dragging a data library in behind them.

Why batch at all: a per-row Qt signal forces the gallery into a
per-row scan of its mounted tiles. Collecting a whole timer tick's
worth of changed rows into one payload is the coalescing that
`docs/media_architecture.md` section 6.1 item 7 asks for -- MainWindow
then makes one pass over the mounted tiles for the whole batch instead
of one pass per row.

The controller always sends the notification tagged with the table the
result belonged to, whatever table is currently on screen. Deciding
whether that table is visible is MainWindow's job, done once, not the
controller's and not each widget's.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RowsUpdated:
    """
    One or more rows in `table_name` have new column values.

    Attributes:
        table_name: The table the rows belong to.
        row_ids:    The rows whose data changed, as a tuple so the
                    payload stays immutable and hashable.
    """

    table_name: str
    row_ids: tuple[str, ...]


@dataclass(frozen=True)
class ThumbnailsReady:
    """
    Thumbnails for one or more rows in `table_name` are now available.

    Attributes:
        table_name: The table the rows belong to.
        row_ids:    The rows whose thumbnail is now ready.

    P0.5 note: `docs/media_architecture.md` section 4.6 item 8 requires
    the ready notification to identify table, row **and** column. The
    column is deliberately not added here yet -- P0.5 adds it together
    with the address-keyed artifact store, so that this payload and the
    artifact key change shape in the same change.
    """

    table_name: str
    row_ids: tuple[str, ...]
