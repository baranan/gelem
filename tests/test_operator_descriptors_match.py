"""
tests/test_operator_descriptors_match.py

Every configured operator's ``descriptor`` (an operators/descriptor.py
OperatorDescriptor) is pinned here against an EXPLICIT table of the
facts it must carry: which execution modes it offers, and -- for a
COLUMNS mode -- the (name, tag) output columns.

HISTORY. Until P1.12d-3 this module pinned each descriptor against the
operator's LEGACY attributes -- the three ``*_label`` strings and
``output_columns`` -- which held the same facts in parallel with the
descriptor. P1.12d-3 deleted those attributes: the descriptor is now the
SINGLE source (the Operators menu reads ``list_operators_for_mode`` off
the descriptors, and AppController builds every run and its column tags
from the mode descriptor). With no second copy left there is nothing to
cross-check, so this module now checks the descriptor against a
hand-written expected table instead -- a real regression guard: a
regression of ``avatar_path``'s tag away from ``media_path`` fails a
named test here.

WHY MENU LABELS ARE NOT PINNED HERE (P1.12d-3 follow-up). This module
used to also pin each mode's exact menu label string. It no longer does,
and must not be made to again. A menu label is Y B's wording and he may
change it whenever he likes; a test in another file that goes red
because he renamed a menu entry is a test pinning a decision that is not
the test's to make. An output column's name and type tag are different
in kind: an unregistered tag renders a placeholder tile in the gallery,
which is a defect, not a preference, so those stay pinned. That a mode
label is present and non-empty at all is enforced where it belongs --
``ModeDescriptor.__post_init__`` rejects an empty or missing ``label``
at construction (operators/descriptor.py), and every operator builds its
descriptor at import time.

There is deliberately ONE named test per operator (not a loop over a
list of classes) so a failure names the operator directly. The final
coverage test checks that the set of operators pinned here equals the
set of factory keys in operators/operator_config.py, so a ninth operator
cannot be added later without a descriptor and a test noticing.

Grouping: this module imports the operator modules but creates no Qt
widget, so it belongs in run_tests.py's combined group.
"""

from __future__ import annotations

from operators.descriptor import ExecutionMode, OperatorDescriptor
from operators.operator_config import OPERATOR_FACTORIES

from operators.blendshapes import BLENDSHAPE_NAMES, BlendshapeOperator
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


# The expected facts, written out by hand. `modes` is the exact set of
# ExecutionModes the operator must declare (the label each carries is
# deliberately NOT pinned -- see the module docstring). `columns` (only
# meaningful for a COLUMNS mode) is the expected list of (name, type_tag)
# output columns, in order.
EXPECTED = {
    "blendshapes": {
        "modes": {ExecutionMode.COLUMNS},
        "columns": [(bs_name, "numeric") for bs_name in BLENDSHAPE_NAMES],
    },
    "blendshape_avatar": {
        "modes": {ExecutionMode.COLUMNS},
        "columns": [("avatar_path", "media_path")],
    },
    "mean_face": {
        "modes": {ExecutionMode.TABLE, ExecutionMode.DISPLAY},
        "columns": None,
    },
    "plot": {
        "modes": {ExecutionMode.COLUMNS},
        "columns": [("plot_path", "media_path")],
    },
    "plot_advanced": {
        "modes": {ExecutionMode.DISPLAY},
        "columns": None,
    },
    "stats": {
        "modes": {ExecutionMode.DISPLAY},
        "columns": None,
    },
    "summary_stats": {
        "modes": {ExecutionMode.DISPLAY},
        "columns": None,
    },
    "video_frames": {
        "modes": {ExecutionMode.TABLE},
        "columns": None,
    },
}


# ---------------------------------------------------------------------------
# Shared assertion. Called by each named test with one operator name --
# this is not a loop over a list, so a failure still points at one operator.
# ---------------------------------------------------------------------------

def _assert_descriptor_matches_expected(name):
    operator_class = OPERATORS_UNDER_TEST[name]
    expected = EXPECTED[name]
    descriptor = operator_class.descriptor

    assert isinstance(descriptor, OperatorDescriptor), (
        f"{operator_class.__name__} has no OperatorDescriptor"
    )
    assert descriptor.name == operator_class.name == name, (
        f"descriptor.name {descriptor.name!r} / operator name "
        f"{operator_class.name!r} / expected {name!r} disagree"
    )

    # The set of declared modes is exactly the expected set. The label
    # each mode carries is not checked here -- see the module docstring.
    declared = {md.mode for md in descriptor.modes}
    assert declared == expected["modes"], (
        f"{name}: descriptor modes {declared} != expected "
        f"{expected['modes']}"
    )

    # For a COLUMNS mode, the OutputSpec columns must be exactly the
    # expected (name, tag) pairs, in order.
    columns_mode = descriptor.mode_for(ExecutionMode.COLUMNS)
    if expected["columns"] is not None:
        assert columns_mode is not None, f"{name}: expected a COLUMNS mode"
        actual_pairs = [
            (column.name, column.type_tag)
            for column in columns_mode.output.columns
        ]
        assert actual_pairs == expected["columns"], (
            f"{name}: descriptor output columns {actual_pairs} != expected "
            f"{expected['columns']}"
        )
    else:
        assert columns_mode is None, (
            f"{name}: descriptor declares a COLUMNS mode but none is expected"
        )


# ---------------------------------------------------------------------------
# One named test per operator.
# ---------------------------------------------------------------------------

def test_blendshapes_descriptor_matches_expected():
    _assert_descriptor_matches_expected("blendshapes")


def test_blendshape_avatar_descriptor_matches_expected():
    _assert_descriptor_matches_expected("blendshape_avatar")


def test_mean_face_descriptor_matches_expected():
    _assert_descriptor_matches_expected("mean_face")


def test_plot_descriptor_matches_expected():
    _assert_descriptor_matches_expected("plot")


def test_plot_advanced_descriptor_matches_expected():
    _assert_descriptor_matches_expected("plot_advanced")


def test_stats_descriptor_matches_expected():
    _assert_descriptor_matches_expected("stats")


def test_summary_stats_descriptor_matches_expected():
    _assert_descriptor_matches_expected("summary_stats")


def test_video_frames_descriptor_matches_expected():
    _assert_descriptor_matches_expected("video_frames")


# ---------------------------------------------------------------------------
# Coverage: every configured operator must be pinned above.
# ---------------------------------------------------------------------------

def test_every_factory_operator_is_pinned():
    """The operators pinned by the named tests above must be exactly the
    operators the application offers. Adding a ninth operator to
    OPERATOR_FACTORIES without a descriptor and a named test here then
    fails this test (the set mismatch), and the named test it forces the
    author to add then fails until the descriptor exists and matches.
    """
    pinned = set(OPERATORS_UNDER_TEST)
    configured = set(OPERATOR_FACTORIES)
    assert pinned == configured, (
        f"operators pinned here {sorted(pinned)} != configured operators "
        f"{sorted(configured)}"
    )
    # And EXPECTED covers the same set -- so a new operator cannot be
    # added to OPERATORS_UNDER_TEST without also stating its expected
    # facts.
    assert set(EXPECTED) == pinned, (
        f"EXPECTED keys {sorted(EXPECTED)} != pinned operators "
        f"{sorted(pinned)}"
    )
