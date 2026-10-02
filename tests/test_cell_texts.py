"""
tests/test_cell_texts.py

Two seams the table view reads cells through:

  * Dataset.get_cell_values(table, row_ids, columns) -- raw values, by
    position, in the order asked, an unknown id giving a row of None.
  * AppController.get_cell_texts(table_name, result_id, start, stop,
    columns) -- None when the caller's table or result id is stale,
    otherwise the row ids at flat positions [start, stop) (clamped like
    get_row_ids_in_range) and their display text.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from column_types.text_format import format_cell_text


# -- Dataset.get_cell_values ------------------------------------------------

def test_dataset_values_keep_the_order_of_ids_and_columns(make_controller, tmp_path):
    _, dataset, _ = make_controller(tmp_path, merge_csv=True)
    ids = dataset.get_table("frames")["row_id"].tolist()
    asked = [ids[5], ids[0], ids[3]]
    columns = ["file_name", "timestamp", "condition"]

    rows = dataset.get_cell_values("frames", asked, columns)

    assert len(rows) == 3
    for row_id, row in zip(asked, rows):
        expected = dataset.get_row(row_id, "frames")
        assert row == [expected[c] for c in columns]


def test_dataset_unknown_id_gives_a_row_of_none_in_place(make_controller, tmp_path):
    _, dataset, _ = make_controller(tmp_path)
    ids = dataset.get_table("frames")["row_id"].tolist()

    rows = dataset.get_cell_values(
        "frames", [ids[1], "no-such-id", ids[2]], ["file_name", "full_path"]
    )

    assert len(rows) == 3
    assert rows[1] == [None, None]
    # The rows around it did not shift.
    assert rows[0] == [
        dataset.get_row(ids[1], "frames")["file_name"],
        dataset.get_row(ids[1], "frames")["full_path"],
    ]
    assert rows[2][0] == dataset.get_row(ids[2], "frames")["file_name"]


def test_dataset_returns_only_the_asked_columns(make_controller, tmp_path):
    _, dataset, _ = make_controller(tmp_path, merge_csv=True)
    ids = dataset.get_table("frames")["row_id"].tolist()

    rows = dataset.get_cell_values("frames", ids[:2], ["condition"])

    assert all(len(row) == 1 for row in rows)
    assert dataset.get_cell_values("frames", ids[:2], []) == [[], []]


def test_dataset_unknown_column_or_table_raises_keyerror(make_controller, tmp_path):
    _, dataset, _ = make_controller(tmp_path)
    ids = dataset.get_table("frames")["row_id"].tolist()
    with pytest.raises(KeyError):
        dataset.get_cell_values("frames", ids[:1], ["not_a_column"])
    with pytest.raises(KeyError):
        dataset.get_cell_values("no_such_table", ids[:1], ["file_name"])


def test_dataset_reads_do_not_copy_the_table(make_controller, tmp_path, monkeypatch):
    _, dataset, _ = make_controller(tmp_path)
    ids = dataset.get_table("frames")["row_id"].tolist()

    def refuse(*args, **kwargs):
        raise AssertionError("get_cell_values copied the whole table")

    monkeypatch.setattr(dataset, "get_table", refuse)
    monkeypatch.setattr(dataset, "snapshot_rows", refuse)
    dataset.get_cell_values("frames", ids[:3], ["file_name"])


# -- AppController.get_cell_texts -------------------------------------------

def _ready(make_controller, tmp_path):
    controller, dataset, _ = make_controller(tmp_path, merge_csv=True)
    controller.set_filters([])
    return controller, dataset, controller.get_result_layout()


def test_stale_result_id_returns_none(make_controller, tmp_path):
    controller, _, layout = _ready(make_controller, tmp_path)
    columns = controller.get_column_names()

    assert controller.get_cell_texts("frames", layout.result_id, 0, 5, columns) is not None
    assert controller.get_cell_texts("frames", "an-old-id", 0, 5, columns) is None

    # A new query mints a new id: the previous one is now stale.
    controller.set_filters([])
    new_layout = controller.get_result_layout()
    assert new_layout.result_id != layout.result_id
    assert controller.get_cell_texts("frames", layout.result_id, 0, 5, columns) is None
    assert controller.get_cell_texts("frames", new_layout.result_id, 0, 5, columns) is not None


def test_a_table_that_is_not_active_returns_none(make_controller, tmp_path):
    controller, dataset, layout = _ready(make_controller, tmp_path)
    columns = controller.get_column_names()
    dataset.create_table_from_rows(
        "other", dataset.get_table("frames")["row_id"].tolist()[:3], "frames"
    )

    assert controller.get_active_table() == "frames"
    assert controller.get_cell_texts("other", layout.result_id, 0, 2, columns) is None
    assert controller.get_cell_texts("missing", layout.result_id, 0, 2, columns) is None


def test_unknown_column_returns_none(make_controller, tmp_path):
    controller, _, layout = _ready(make_controller, tmp_path)
    assert controller.get_cell_texts("frames", layout.result_id, 0, 2, ["nope"]) is None


def test_texts_match_the_formatter_and_the_flat_order(make_controller, tmp_path):
    controller, dataset, layout = _ready(make_controller, tmp_path)
    # Sort so the flat order differs from the stored order.
    controller.set_filters([], sort_by="timestamp", ascending=False)
    columns = controller.get_column_names()
    layout = controller.get_result_layout()
    schema = dataset.schema_for("frames")

    page = controller.get_cell_texts("frames", layout.result_id, 2, 7, columns)

    assert page.row_ids == controller.get_row_ids_in_range(2, 7)
    assert len(page.texts) == 5
    for row_id, texts in zip(page.row_ids, page.texts):
        stored = dataset.get_row(row_id, "frames")
        assert len(texts) == len(columns)
        for column, text in zip(columns, texts):
            tag = schema.spec_for(column).type_tag
            assert text == format_cell_text(stored[column], tag)
    # The merged float column came out as text, not a float.
    assert all(isinstance(t, str) for texts in page.texts for t in texts)


def test_range_is_clamped_like_get_row_ids_in_range(make_controller, tmp_path):
    controller, _, layout = _ready(make_controller, tmp_path)
    columns = ["file_name"]
    total = layout.total

    low = controller.get_cell_texts("frames", layout.result_id, -5, 3, columns)
    assert low.row_ids == controller.get_row_ids_in_range(-5, 3)
    assert len(low.row_ids) == 3

    high = controller.get_cell_texts("frames", layout.result_id, total - 2, total + 50, columns)
    assert high.row_ids == controller.get_row_ids_in_range(total - 2, total + 50)
    assert len(high.texts) == 2

    beyond = controller.get_cell_texts("frames", layout.result_id, total + 5, total + 9, columns)
    assert beyond.row_ids == [] and beyond.texts == []


def test_respects_filters_and_the_asked_column_order(make_controller, tmp_path):
    controller, dataset, _ = _ready(make_controller, tmp_path)
    controller.set_filters([], randomise=True, seed=3)
    layout = controller.get_result_layout()

    page = controller.get_cell_texts(
        "frames", layout.result_id, 0, 4, ["timestamp", "file_name"]
    )

    for row_id, texts in zip(page.row_ids, page.texts):
        stored = dataset.get_row(row_id, "frames")
        assert texts[1] == stored["file_name"]
        assert texts[0] == format_cell_text(stored["timestamp"], "numeric")
