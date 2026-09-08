"""
tests/test_operator_descriptors_match.py

P1.12d-1 consistency check: every operator now carries a `descriptor`
(an operators/descriptor.py OperatorDescriptor), and this module pins
each one against the operator's LEGACY attributes -- `name`, the three
`*_label` strings and `output_columns` -- which stay authoritative at
run time until P1.12d-2/d-3 wire the menu and runner to the descriptor.

If these ever disagree, one of the two descriptions is wrong. The test
does not judge which; it just refuses the drift.

There is deliberately ONE named test per operator (not a loop over a
list of classes) so a failure names the operator directly. A final
coverage test checks that the set of operators pinned here equals the
set of factory keys in operators/operator_config.py, so a ninth
operator cannot be added later without a descriptor and a test noticing.

Grouping: this module imports the operator modules but creates no Qt
widget, so it belongs in run_tests.py's combined group. The substring
heuristic already puts it there (the file names none of the widget
markers), so it carries no `# run-tests:` declaration.
"""

from __future__ import annotations

from operators.descriptor import ExecutionMode
from operators.operator_config import OPERATOR_FACTORIES

from operators.blendshapes import BlendshapeOperator
from operators.blendshape_avatar import BlendshapeAvatarOperator
from operators.mean_face import MeanFaceOperator
from operators.plot_operator import PlotOperator
from operators.plot_advanced import PlotAdvancedOperator
from operators.stats_operator import StatsOperator
from operators.summary_stats import SummaryStatsOperator
from operators.video_frames import VideoFramesOperator


# The operators this module pins, keyed by their `name` attribute. Each
# key has its own named test below. test_every_factory_operator_is_pinned
# compares this mapping's keys against OPERATOR_FACTORIES -- the factory
# keys are read there, never re-typed, so the eight names live in exactly
# one place for the drift check.
OPERATORS_UNDER_TEST = {
    "blendshapes": BlendshapeOperator,
    "blendshape_avatar": BlendshapeAvatarOperator,
    "mean_face": MeanFaceOperator,
    "plot": PlotOperator,
    "plot_advanced": PlotAdvancedOperator,
    "stats": StatsOperator,
    "summary_stats": SummaryStatsOperator,
    "video_frames": VideoFramesOperator,
}


# ---------------------------------------------------------------------------
# Shared assertions. Called by each named test with one operator class --
# this is not a loop over a list, so a failure still points at one operator.
# ---------------------------------------------------------------------------

def _assert_mode_matches_label(descriptor, mode: ExecutionMode, label):
    """A mode of `mode` must exist in the descriptor exactly when the
    matching legacy `*_label` is set, and carry that same label string.
    """
    mode_descriptor = descriptor.mode_for(mode)
    if label is None:
        # No legacy label -> the operator does not implement this mode ->
        # the descriptor must not claim it.
        assert mode_descriptor is None, (
            f"descriptor declares a {mode.name} mode but the operator's "
            f"legacy label for it is None"
        )
    else:
        assert mode_descriptor is not None, (
            f"operator has a {mode.name} label ({label!r}) but the "
            f"descriptor declares no {mode.name} mode"
        )
        assert mode_descriptor.label == label, (
            f"{mode.name} mode label {mode_descriptor.label!r} does not "
            f"match the legacy label {label!r}"
        )


def _assert_descriptor_matches(operator_class):
    """Pin one operator's descriptor against its legacy attributes."""
    descriptor = operator_class.descriptor

    # It exists, and its name is the operator's name.
    assert descriptor is not None, (
        f"{operator_class.__name__} has no descriptor"
    )
    assert descriptor.name == operator_class.name, (
        f"descriptor.name {descriptor.name!r} != operator name "
        f"{operator_class.name!r}"
    )

    # Each of the three modes: present iff the legacy label is set, and
    # the label string matches.
    _assert_mode_matches_label(
        descriptor, ExecutionMode.COLUMNS,
        operator_class.create_columns_label,
    )
    _assert_mode_matches_label(
        descriptor, ExecutionMode.TABLE,
        operator_class.create_table_label,
    )
    _assert_mode_matches_label(
        descriptor, ExecutionMode.DISPLAY,
        operator_class.create_display_label,
    )

    # For a COLUMNS mode, the descriptor's output columns must equal
    # output_columns as (name, tag) pairs, in the same order.
    columns_mode = descriptor.mode_for(ExecutionMode.COLUMNS)
    if columns_mode is not None:
        descriptor_pairs = [
            (column.name, column.type_tag)
            for column in columns_mode.output.columns
        ]
        legacy_pairs = [
            (name, tag) for name, tag in operator_class.output_columns
        ]
        assert descriptor_pairs == legacy_pairs, (
            f"descriptor output columns {descriptor_pairs} != "
            f"output_columns {legacy_pairs}"
        )


# ---------------------------------------------------------------------------
# One named test per operator.
# ---------------------------------------------------------------------------

def test_blendshapes_descriptor_matches_legacy_attributes():
    _assert_descriptor_matches(OPERATORS_UNDER_TEST["blendshapes"])


def test_blendshape_avatar_descriptor_matches_legacy_attributes():
    _assert_descriptor_matches(OPERATORS_UNDER_TEST["blendshape_avatar"])


def test_mean_face_descriptor_matches_legacy_attributes():
    _assert_descriptor_matches(OPERATORS_UNDER_TEST["mean_face"])


def test_plot_descriptor_matches_legacy_attributes():
    _assert_descriptor_matches(OPERATORS_UNDER_TEST["plot"])


def test_plot_advanced_descriptor_matches_legacy_attributes():
    _assert_descriptor_matches(OPERATORS_UNDER_TEST["plot_advanced"])


def test_stats_descriptor_matches_legacy_attributes():
    _assert_descriptor_matches(OPERATORS_UNDER_TEST["stats"])


def test_summary_stats_descriptor_matches_legacy_attributes():
    _assert_descriptor_matches(OPERATORS_UNDER_TEST["summary_stats"])


def test_video_frames_descriptor_matches_legacy_attributes():
    _assert_descriptor_matches(OPERATORS_UNDER_TEST["video_frames"])


# ---------------------------------------------------------------------------
# Coverage: every configured operator must be pinned above.
# ---------------------------------------------------------------------------

def test_every_factory_operator_is_pinned():
    """The operators pinned by the named tests above must be exactly the
    operators the application offers. Adding a ninth operator to
    OPERATOR_FACTORIES without a descriptor and a named test here then
    fails this test.
    """
    pinned = set(OPERATORS_UNDER_TEST)
    configured = set(OPERATOR_FACTORIES)
    assert pinned == configured, (
        f"operators pinned here {sorted(pinned)} != configured operators "
        f"{sorted(configured)}"
    )
