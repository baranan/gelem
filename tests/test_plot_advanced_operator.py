"""
tests/test_plot_advanced_operator.py

PlotAdvancedOperator.create_display -- P1.9a coverage. Before this item
the operator captured a construction-time self._output_dir (a repo-
relative gelem_project/plots folder); it now writes only under
run.paths.outputs_dir, supplied fresh on every run (operators/CLAUDE.md,
"Write only to run.paths").

There was no test of create_display's file-writing behaviour before this
item (tests/test_plot_advanced_guidance.py covers refine_form only, via
PlotAdvancedOperator.__new__ -- it never constructs a run). Written from
the work-item rule: a run given a ProjectPaths under tmp_path writes its
files under that outputs_dir and nowhere else.

Run with:
    python -m pytest tests/test_plot_advanced_operator.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.project_paths import build_project_paths
from operators.descriptor import ExecutionMode
from operators.plot_advanced import PlotAdvancedOperator
from operators.run_context import (
    CancellationToken,
    OperatorRun,
    OperatorRunSpec,
    RunData,
)


def _run(op, *, parameters, paths):
    mode_descriptor = op.descriptor.mode_for(ExecutionMode.DISPLAY)
    spec = OperatorRunSpec(
        operation_id="test-run",
        operator_name=op.name,
        mode=ExecutionMode.DISPLAY,
        mode_descriptor=mode_descriptor,
        parameters=parameters,
        target_table="",
    )
    return OperatorRun(
        spec=spec,
        data=RunData(tables={}, projects={}),
        paths=paths,
        _token=CancellationToken(),
    )


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition": ["a", "a", "b", "b"],
            "score": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_create_display_writes_only_under_run_paths_outputs_dir(tmp_path):
    op = PlotAdvancedOperator()
    paths = build_project_paths(tmp_path / "project", is_workspace=False)
    run = _run(
        op,
        parameters={
            "title": "",
            "chart_type": "scatter",
            "x": "condition",
            "y": "score",
            "color": None,
            "facet": None,
            "aggregate": "none",
        },
        paths=paths,
    )

    result = op.create_display(_df(), run)

    outputs_dir = paths.outputs_dir.resolve()
    png_path = Path(result["artifact_path"]).resolve()
    html_path = Path(result["html_path"]).resolve()

    assert png_path.exists()
    assert html_path.exists()
    assert outputs_dir in png_path.parents, (
        f"{png_path} was not written under {outputs_dir}"
    )
    assert outputs_dir in html_path.parents, (
        f"{html_path} was not written under {outputs_dir}"
    )

    files = [p for p in (tmp_path / "project").rglob("*") if p.is_file()]
    stray = [p for p in files if outputs_dir not in p.resolve().parents]
    assert not stray, f"files written outside outputs_dir: {stray}"


def test_two_runs_do_not_overwrite_each_other(tmp_path):
    op = PlotAdvancedOperator()
    paths = build_project_paths(tmp_path / "project", is_workspace=False)
    parameters = {
        "title": "",
        "chart_type": "scatter",
        "x": "condition",
        "y": "score",
        "color": None,
        "facet": None,
        "aggregate": "none",
    }

    first = op.create_display(_df(), _run(op, parameters=parameters, paths=paths))
    second = op.create_display(_df(), _run(op, parameters=parameters, paths=paths))

    assert first["artifact_path"] != second["artifact_path"]
    assert Path(first["artifact_path"]).exists()
    assert Path(second["artifact_path"]).exists()
