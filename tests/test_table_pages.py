"""
tests/test_table_pages.py

Layer A of the table view (ui/table_pages.py): the bounded page cache and
the group-label lookup. Pure Python -- no Qt, no pandas -- so nothing here
needs a QApplication.
"""

from __future__ import annotations

# run-tests: combined -- names the Qt binding only inside a source-scan assertion; it imports no Qt and shows no widget

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from models.query_result import GroupSection
from ui.table_pages import MAX_PAGES, PAGE_SIZE, GroupLabels, TablePageCache

COLS = ("a", "b")


def _page(cache, page_index, result_id="r1", columns=COLS, size=PAGE_SIZE):
    """Puts a full page whose row ids are 'id<row>' and whose cells are
    '<row>:<column number>'."""
    start = page_index * size
    row_ids = [f"id{row}" for row in range(start, start + size)]
    texts = [[f"{row}:{c}" for c in range(len(columns))] for row in range(start, start + size)]
    cache.put_page(result_id, columns, page_index, row_ids, texts)


def test_constants_match_the_design():
    assert PAGE_SIZE == 200
    assert MAX_PAGES == 50


def test_layer_a_imports_no_qt_and_no_pandas():
    tree = ast.parse((ROOT / "ui" / "table_pages.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"PySide6", "pandas", "numpy"}


def test_a_missing_page_answers_none_and_a_stored_page_answers_text():
    cache = TablePageCache()
    assert cache.lookup("r1", COLS, 5, 0) is None
    _page(cache, 0)
    assert cache.lookup("r1", COLS, 5, 1) == "5:1"
    # Row 200 is on the next page, which is still missing.
    assert cache.lookup("r1", COLS, 200, 0) is None
    assert cache.page_index_for(199) == 0
    assert cache.page_index_for(200) == 1


def test_key_is_result_columns_and_page():
    cache = TablePageCache()
    _page(cache, 0)
    assert cache.lookup("another-result", COLS, 0, 0) is None
    assert cache.lookup("r1", ("a",), 0, 0) is None
    assert cache.lookup("r1", COLS, 0, 0) == "0:0"


def test_lru_never_exceeds_the_bound_and_drops_the_oldest():
    cache = TablePageCache()
    for page in range(MAX_PAGES + 30):
        _page(cache, page)
        assert len(cache) <= MAX_PAGES
    assert len(cache) == MAX_PAGES
    # The first 30 pages were evicted, the newest are present.
    assert not cache.has_page("r1", COLS, 0)
    assert not cache.has_page("r1", COLS, 29)
    assert cache.has_page("r1", COLS, 30)
    assert cache.has_page("r1", COLS, MAX_PAGES + 29)


def test_a_lookup_refreshes_recency():
    cache = TablePageCache(page_size=2, max_pages=3)
    for page in range(3):
        _page(cache, page, size=2)
    # Touch page 0 so page 1 is now the oldest.
    assert cache.lookup("r1", COLS, 0, 0) == "0:0"
    _page(cache, 3, size=2)
    assert cache.has_page("r1", COLS, 0)
    assert not cache.has_page("r1", COLS, 1)


def test_reset_drops_everything():
    cache = TablePageCache()
    for page in range(5):
        _page(cache, page)
    cache.reset()
    assert len(cache) == 0
    assert cache.lookup("r1", COLS, 0, 0) is None


def test_rows_updated_drops_only_pages_holding_those_ids_and_returns_ranges():
    cache = TablePageCache()
    for page in range(4):
        _page(cache, page)

    # id5 is on page 0, id450 on page 2; pages 1 and 3 hold neither.
    ranges = cache.invalidate_rows(["id5", "id450", "id-not-cached"])

    assert ranges == [(0, 200), (400, 600)]
    assert not cache.has_page("r1", COLS, 0)
    assert cache.has_page("r1", COLS, 1)
    assert not cache.has_page("r1", COLS, 2)
    assert cache.has_page("r1", COLS, 3)


def test_rows_updated_with_nothing_cached_or_nothing_changed_returns_empty():
    cache = TablePageCache()
    assert cache.invalidate_rows(["id1"]) == []
    _page(cache, 0)
    assert cache.invalidate_rows([]) == []
    assert cache.invalidate_rows(["id99999"]) == []
    assert cache.has_page("r1", COLS, 0)


def test_rows_updated_range_of_a_short_last_page_is_its_real_length():
    cache = TablePageCache()
    cache.put_page("r1", COLS, 1, ["x200", "x201", "x202"], [["a", "b"]] * 3)
    assert cache.invalidate_rows(["x201"]) == [(200, 203)]


def test_one_row_in_two_cached_column_sets_reports_one_range():
    cache = TablePageCache()
    _page(cache, 0)
    _page(cache, 0, columns=("a",))
    assert cache.invalidate_rows(["id3"]) == [(0, 200)]
    assert len(cache) == 0


# -- group labels -----------------------------------------------------------

def test_group_label_at_span_boundaries():
    groups = GroupLabels(
        (GroupSection("A", 0, 3), GroupSection("B", 3, 4), GroupSection("C", 4, 10))
    )
    assert groups.label_for(0) == "A"      # first row of the first group
    assert groups.label_for(2) == "A"      # last row of A
    assert groups.label_for(3) == "B"      # first row of B (a one-row group)
    assert groups.label_for(4) == "C"      # first row of C
    assert groups.label_for(9) == "C"      # last row overall
    assert groups.label_for(10) == ""      # just past the end
    assert groups.label_for(-1) == ""


def test_group_label_with_no_groups_is_empty():
    assert GroupLabels(()).label_for(0) == ""
    assert GroupLabels(None).label_for(5) == ""


def test_group_label_skips_an_empty_span():
    groups = GroupLabels(
        (GroupSection("A", 0, 2), GroupSection("empty", 2, 2), GroupSection("C", 2, 5))
    )
    assert groups.label_for(1) == "A"
    assert groups.label_for(2) == "C"
    assert groups.label_for(4) == "C"
