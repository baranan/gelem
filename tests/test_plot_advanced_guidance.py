"""
tests/test_plot_advanced_guidance.py

PlotAdvancedOperator.refine_form (P1.12e-4b) -- the pure function that
restores the live chart_type / aggregate coupling the hand-drawn dialog
had before P1.12e-2 deleted it.

This is the operator's function ALONE. It touches no Qt and no table, so
these tests import only the operator and the form-advice vocabulary and
need no QApplication.

Written from the work-item specification (STEP 6) and from the deleted
_apply_chart_rules / _refresh_warnings logic quoted in the gate, not from
the new implementation.

Run with:
    python -m pytest tests/test_plot_advanced_guidance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from operators.plot_advanced import PlotAdvancedOperator
from operators.form_advice import FormAdvice


def _refine(**values) -> FormAdvice:
    """refine_form is pure and reads nothing off the instance, so a bare
    instance is enough. Give it just the values the case needs."""
    operator = PlotAdvancedOperator.__new__(PlotAdvancedOperator)
    return operator.refine_form(dict(values))


# ===========================================================================
# box and violin -- Aggregate does not apply
# ===========================================================================

def test_box_makes_aggregate_inapplicable():
    advice = _refine(chart_type="box", aggregate="none")
    assert advice.inapplicable == ("aggregate",)
    # It is not ALSO narrowed -- inapplicable is the whole story.
    assert advice.allowed_choices == {}


def test_violin_makes_aggregate_inapplicable():
    advice = _refine(chart_type="violin", aggregate="mean")
    assert advice.inapplicable == ("aggregate",)


# ===========================================================================
# histogram -- Aggregate narrowed to count / sum / mean
# ===========================================================================

def test_histogram_narrows_aggregate():
    advice = _refine(chart_type="histogram", aggregate="sum")
    assert advice.allowed_choices == {"aggregate": ("count", "sum", "mean")}
    # median and none are simply not in the allowed set.
    assert "median" not in advice.allowed_choices["aggregate"]
    assert "none" not in advice.allowed_choices["aggregate"]
    # Not disabled -- the researcher still chooses among the three.
    assert advice.inapplicable == ()


def test_histogram_with_count_warns():
    advice = _refine(chart_type="histogram", aggregate="count")
    assert len(advice.messages) == 1
    message = advice.messages[0]
    assert message.severity == "warning"
    assert message.field == "aggregate"
    assert "Y" in message.text


def test_histogram_with_sum_does_not_warn():
    advice = _refine(chart_type="histogram", aggregate="sum")
    assert advice.messages == ()


# ===========================================================================
# bar with aggregate none -- a warning, no narrowing
# ===========================================================================

def test_bar_with_aggregate_none_warns():
    advice = _refine(chart_type="bar", aggregate="none")
    assert len(advice.messages) == 1
    message = advice.messages[0]
    assert message.severity == "warning"
    assert message.field == "aggregate"
    # bar is not restricted or disabled -- only warned about.
    assert advice.inapplicable == ()
    assert advice.allowed_choices == {}


def test_bar_with_a_real_aggregate_does_not_warn():
    advice = _refine(chart_type="bar", aggregate="mean")
    assert advice.messages == ()


# ===========================================================================
# an incomplete / uninteresting form -- no advice, no exception
# ===========================================================================

def test_scatter_with_everything_blank_returns_no_advice():
    # A scatter with no columns chosen and no aggregate touched: the
    # function must tolerate the empty form and simply say nothing.
    advice = _refine()
    assert advice == FormAdvice()


def test_scatter_returns_no_advice():
    advice = _refine(chart_type="scatter", aggregate="none")
    assert advice == FormAdvice()


def test_an_unknown_chart_type_is_tolerated():
    # Raw values can be anything mid-edit; refine_form must not raise.
    advice = _refine(chart_type="", aggregate=None)
    assert advice == FormAdvice()
