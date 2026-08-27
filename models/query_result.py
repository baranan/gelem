"""
models/query_result.py

Three small frozen dataclasses that carry the *ordered query result* --
which rows match the current filters, in what order -- between the
controller and the UI.

Written for P0.4 (docs/media_architecture.md section 6.1). Two things
used to be tangled in GalleryWidget._row_ids:

  * the ordered query result -- data, and the controller must own it;
  * viewport geometry -- the gallery's, and the gallery reports the
    index range it currently shows into the controller's order.

This module is the data half. It deliberately does **not** import pandas,
so a UI file may import ResultLayout without dragging a data library in
behind it.

Grouping note: grouping changes how rows are stacked on screen, not
which rows are visible, so one flat ordered sequence serves both the
flat and the grouped view. In grouped mode the flat order is the
concatenation of QueryEngine.apply_grouped()'s dict in its existing
order, and each group's span is recorded as a (start, stop) pair into
that same flat order.

`groups is None` means flat mode. `groups == ()` means grouped mode with
zero groups. These are different states and must stay different, the same
way None and [] differ for visible columns.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroupSection:
    """
    One group's span within the flat order.

    Attributes:
        label: The group value, already stringified.
        start: Inclusive index into the flat order.
        stop:  Exclusive index into the flat order.
    """

    label: str
    start: int
    stop: int


@dataclass(frozen=True)
class ResultLayout:
    """
    What the UI is told when the result changes. Deliberately carries no
    row ids: the UI is told how many rows there are and where the group
    boundaries fall, and it fetches the row ids it needs to paint from
    the controller.

    Attributes:
        result_id:  Opaque id of the result this layout describes. Every
                    recompute mints a new one. A viewport report naming
                    an old result id is dropped.
        table_name: The table the result was computed against.
        total:      Number of rows in the flat order.
        groups:     One GroupSection per group, in flat-order sequence,
                    or None in flat mode. An empty tuple means grouped
                    mode with zero groups.
    """

    result_id: str
    table_name: str
    total: int
    groups: tuple[GroupSection, ...] | None


@dataclass(frozen=True)
class QueryResult:
    """
    What the controller keeps after each recompute. Never emitted -- the
    controller hands the UI a ResultLayout (via layout()) and answers
    row-id questions through its own accessor methods.

    Attributes:
        result_id:  Opaque id, freshly minted on every recompute.
        table_name: The table the result was computed against.
        row_ids:    The flat order: every matching row id, in display
                    order. In grouped mode this is the concatenation of
                    the groups in their flat-order sequence.
        groups:     One GroupSection per group, or None in flat mode.
                    An empty tuple means grouped mode with zero groups.
    """

    result_id: str
    table_name: str
    row_ids: tuple[str, ...]
    groups: tuple[GroupSection, ...] | None

    def layout(self) -> ResultLayout:
        """Returns the row-id-free view of this result for the UI."""
        return ResultLayout(
            result_id=self.result_id,
            table_name=self.table_name,
            total=len(self.row_ids),
            groups=self.groups,
        )
