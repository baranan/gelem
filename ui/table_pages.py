"""
ui/table_pages.py

Layer A for the table view: the page cache and the group-label lookup,
with no Qt and no pandas, so both can be tested without a QApplication.
(Layer A modules live in ui/ beside their Layer B widget, as
ui/close_prompt.py does. Layer B is ui/result_table_view.py.)

The table view shows a result of any size, so it never holds the whole
result's text. It holds pages of PAGE_SIZE consecutive flat-order rows,
at most MAX_PAGES of them, least recently used dropped first. A page is
identified by (result_id, columns, page index): a new result id or a new
column list makes every older page unreachable, and reset() frees them.

The cache does not fetch anything. lookup() answers either with the text
or with None, which means "this page is missing -- fetch it and put_page()
it". Whoever owns the fetch (Layer B) stays the only thing that talks to
the controller.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict

# Rows per page, and the bound on how many pages stay cached. At 60 columns
# a page is about 12,000 short strings, so 50 pages is a few tens of MB
# at worst.
PAGE_SIZE = 200
MAX_PAGES = 50


class _Page:
    """One cached page: the row ids and the text of its rows."""

    __slots__ = ("row_ids", "row_id_set", "texts")

    def __init__(self, row_ids, texts) -> None:
        self.row_ids = list(row_ids)
        # A set so "does this page hold any of those ids" is one
        # intersection, not a scan per id.
        self.row_id_set = frozenset(self.row_ids)
        self.texts = [list(row) for row in texts]


class TablePageCache:
    """Bounded LRU cache of pages of display text."""

    def __init__(
        self, page_size: int = PAGE_SIZE, max_pages: int = MAX_PAGES
    ) -> None:
        if page_size < 1 or max_pages < 1:
            raise ValueError("page_size and max_pages must be at least 1")
        self._page_size = page_size
        self._max_pages = max_pages
        # Key (result_id, columns, page_index); the OrderedDict's order is
        # recency, oldest first.
        self._pages: OrderedDict[tuple, _Page] = OrderedDict()

    # -- geometry ------------------------------------------------------

    @property
    def page_size(self) -> int:
        return self._page_size

    def page_index_for(self, row: int) -> int:
        """The page a flat-order row lives in."""
        return row // self._page_size

    def page_range(self, page_index: int) -> tuple[int, int]:
        """The flat-order [start, stop) a page covers when full. The
        caller clamps stop to the result total."""
        start = page_index * self._page_size
        return start, start + self._page_size

    def __len__(self) -> int:
        return len(self._pages)

    # -- lookup and fill -----------------------------------------------

    def lookup(
        self, result_id: str, columns: tuple[str, ...], row: int, column: int
    ) -> str | None:
        """Text for (row, column), or None when the page holding the row
        is not cached (the caller should fetch page_index_for(row)).

        A hit makes the page the most recently used. A row beyond the end
        of a cached short page, or a column outside the page, also reads
        as None: there is no text to show for it.
        """
        key = (result_id, columns, self.page_index_for(row))
        page = self._pages.get(key)
        if page is None:
            return None
        self._pages.move_to_end(key)
        offset = row - key[2] * self._page_size
        if offset >= len(page.texts):
            return None
        cells = page.texts[offset]
        if column < 0 or column >= len(cells):
            return None
        return cells[column]

    def has_page(
        self, result_id: str, columns: tuple[str, ...], page_index: int
    ) -> bool:
        """Whether the page is cached (does not touch recency)."""
        return (result_id, columns, page_index) in self._pages

    def put_page(
        self,
        result_id: str,
        columns: tuple[str, ...],
        page_index: int,
        row_ids,
        texts,
    ) -> None:
        """Stores a fetched page as the most recently used, evicting the
        least recently used pages if that takes the cache past its bound."""
        key = (result_id, columns, page_index)
        self._pages[key] = _Page(row_ids, texts)
        self._pages.move_to_end(key)
        while len(self._pages) > self._max_pages:
            self._pages.popitem(last=False)

    # -- invalidation --------------------------------------------------

    def reset(self) -> None:
        """Drops every page."""
        self._pages.clear()

    def invalidate_rows(self, row_ids) -> list[tuple[int, int]]:
        """Drops only the pages that hold any of row_ids and returns the
        flat-order [start, stop) of each dropped page, sorted and without
        duplicates, so the caller can tell the view those rows changed.

        The range's stop is where the page's own rows end, so a short
        last page reports its real length.
        """
        changed = frozenset(row_ids)
        if not changed:
            return []
        doomed = [
            key for key, page in self._pages.items()
            if not page.row_id_set.isdisjoint(changed)
        ]
        ranges: set[tuple[int, int]] = set()
        for key in doomed:
            page = self._pages.pop(key)
            start = key[2] * self._page_size
            ranges.add((start, start + len(page.row_ids)))
        return sorted(ranges)


class GroupLabels:
    """Maps a flat-order row to its group's label.

    Built from the ResultLayout's group spans (objects with label, start
    and stop, in flat-order sequence, as GroupSection is). A lookup is a
    binary search over the span starts.
    """

    def __init__(self, groups) -> None:
        spans = list(groups or ())
        self._starts = [span.start for span in spans]
        self._stops = [span.stop for span in spans]
        self._labels = [span.label for span in spans]

    def label_for(self, row: int) -> str:
        """The label of the group holding row, or "" when no span does
        (no groups at all, or a row outside every span)."""
        i = bisect_right(self._starts, row) - 1
        if i < 0 or row >= self._stops[i]:
            return ""
        return self._labels[i]
