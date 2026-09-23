"""
tests/test_query_state.py

CC-37: a table remembers its own filters, sort and grouping.

Before this, AppController.set_active_table() cleared _active_filters and
_group_by outright on every switch, so leaving a filtered table and coming
back lost the filter. Separately, none of load_folder() / load_csv_as_
primary() / load_project() reliably forgot that state -- load_project()
did not touch it at all, so a filter set in one project could survive
into an unrelated one and silently show zero rows.

table_display.TableQueryState is now the single per-table home for this
state (the same pattern as TableDisplayState for visible columns).

Written from the work-item spec, not from the implementation -- each test
is designed to fail against the pre-CC-37 behaviour described above.

Real Dataset + pandas, no Qt widgets shown, so this runs in the combined
(non-isolated) group -- see run_tests.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.query_engine import Filter
from table_display import QueryState

project_root = Path(__file__).parent.parent
TEST_IMAGES  = project_root / "test_images"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _set_some_query_state(controller) -> None:
    """Filters and groups the active table, and checks it actually took."""
    controller.set_filters([Filter("condition", "eq", "positive")])
    controller.set_group_by("condition")
    assert controller.get_query_state().filters, "setup did not take"
    assert controller.get_query_state().group_by == "condition"


# ---------------------------------------------------------------------------
# 1. A table's filters, sort and grouping survive a round trip through
#    another table.
# ---------------------------------------------------------------------------

def test_switching_tables_and_back_restores_filters_sort_and_group_by(
    tmp_path, make_controller
):
    controller, dataset, _ = make_controller(tmp_path, merge_csv=True)

    dataset.create_table_from_df(
        "table_b", pd.DataFrame({"label": ["x", "y"], "value": [1, 2]})
    )

    filters = [Filter("condition", "eq", "positive")]
    controller.set_filters(filters, sort_by="timestamp", ascending=False)
    controller.set_group_by("condition")

    state_a = controller.get_query_state()
    assert state_a.filters == filters
    assert state_a.sort_by == "timestamp"
    assert state_a.ascending is False
    assert state_a.group_by == "condition"

    # Table B starts with none of A's state -- it was never set there.
    controller.set_active_table("table_b")
    state_b = controller.get_query_state()
    assert state_b.filters == []
    assert state_b.sort_by is None
    assert state_b.group_by is None

    # Back to A: everything set on it is restored.
    controller.set_active_table("frames")
    state_a_again = controller.get_query_state()
    assert state_a_again.filters == filters
    assert state_a_again.sort_by == "timestamp"
    assert state_a_again.ascending is False
    assert state_a_again.group_by == "condition"


# ---------------------------------------------------------------------------
# 2. Each load path forgets every table's remembered query state.
# ---------------------------------------------------------------------------

def test_load_folder_forgets_query_state(tmp_path, make_controller):
    controller, _dataset, _ = make_controller(tmp_path, merge_csv=True)
    _set_some_query_state(controller)

    controller.load_folder(TEST_IMAGES)

    state = controller.get_query_state()
    assert state.filters == []
    assert state.group_by is None


def test_load_csv_as_primary_forgets_query_state(tmp_path, make_controller):
    controller, _dataset, _ = make_controller(tmp_path, merge_csv=True)
    _set_some_query_state(controller)

    csv_path = tmp_path / "other.csv"
    csv_path.write_text("temperature,val\n20,1\n21,2\n")
    controller.load_csv_as_primary(csv_path)

    state = controller.get_query_state()
    assert state.filters == []
    assert state.group_by is None


def test_load_project_forgets_query_state(tmp_path, make_controller):
    controller, _dataset, _ = make_controller(tmp_path, merge_csv=True)
    project_dir = tmp_path / "saved_project"
    assert controller.save_project(project_dir) is True

    # Filtered AFTER the save, so the saved project on disk carries none of
    # this -- loading it back must forget it, not merely fail to persist it.
    _set_some_query_state(controller)

    controller.load_project(project_dir)

    state = controller.get_query_state()
    assert state.filters == []
    assert state.group_by is None


# ---------------------------------------------------------------------------
# 3. A remembered filter, sort or group-by naming a column the target table
#    does not have is dropped on the switch, with exactly one error_occurred
#    -- and the switch itself is never refused.
# ---------------------------------------------------------------------------

def test_switching_to_a_table_missing_the_filtered_columns_drops_them_and_reports_once(
    tmp_path, make_controller
):
    # The public Dataset API has no way to make an EXISTING table's schema
    # lose a column -- create_table_from_df()/aggregate() refuse a taken
    # name (P1.7-1) rather than overwriting one, and nothing removes a
    # column from a stored schema. So the only way this situation can
    # occur is that "table_b"'s schema is different from whatever it was
    # when the filter naming "condition"/"session_id" was remembered for
    # it -- which is exactly the case _recall_and_validate_query_state()'s
    # own docstring names ("its schema may have changed ... since the
    # state was remembered"). Reaching into controller._query_state
    # directly is how this module plants that state for the test, the
    # same way CLAUDE.md documents "assigns straight into Dataset._tables"
    # as the accepted way to construct an edge state no public call can.
    controller, dataset, _ = make_controller(tmp_path, merge_csv=True)

    dataset.create_table_from_df(
        "table_b", pd.DataFrame({"label": ["x", "y"], "value": [1, 2]})
    )
    controller._query_state.remember(
        "table_b",
        QueryState(
            filters=[
                Filter("condition", "eq", "positive"),
                Filter("session_id", "eq", "S01"),
            ],
            sort_by="condition",
            group_by="condition",
        ),
    )

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    controller.set_active_table("table_b")

    assert len(errors) == 1, f"expected exactly one error, got {errors}"
    assert "condition" in errors[0]
    assert "session_id" in errors[0]

    state = controller.get_query_state()
    assert state.filters == []
    assert state.sort_by is None
    assert state.group_by is None

    # The switch itself was not refused.
    assert controller.get_active_table() == "table_b"

    # The drop is healed into storage, not just returned once: recalling
    # table_b again does not re-report the same stale entry.
    errors.clear()
    controller.set_active_table("frames")
    controller.set_active_table("table_b")
    assert errors == []


def test_a_valid_remembered_filter_survives_the_switch_with_no_error(
    tmp_path, make_controller
):
    controller, dataset, _ = make_controller(tmp_path, merge_csv=True)

    # table_c has "condition" too, so nothing set on "frames" should be
    # dropped when set_active_table() validates it against table_c's schema.
    dataset.create_table_from_df(
        "table_c", pd.DataFrame({"condition": ["positive", "negative"]})
    )

    filters = [Filter("condition", "eq", "positive")]
    controller.set_filters(filters)

    errors: list[str] = []
    controller.error_occurred.connect(errors.append)

    controller.set_active_table("table_c")
    controller.set_filters(filters)  # table_c's own state, same column name
    controller.set_active_table("frames")

    assert errors == []
    assert controller.get_query_state().filters == filters
