"""
operators/operator_config.py

Single authority for WHICH operators the application offers.

operators_config.yaml lists every operator and marks each enabled or
disabled. This module parses that file and hands main.py a list of
constructed operator instances, in the file's order. main.py keeps only
the knowledge of HOW to build each one -- that knowledge is the per-
operator factory callables in OPERATOR_FACTORIES below.

Qt-free by contract. This module imports no PySide6, no pandas and no
numpy, and every factory imports its operator module INSIDE the callable
rather than at module scope. So `import operators.operator_config` is
cheap and drags in no Qt, mediapipe or plotly. tests/test_operator_config.py
guards the drift between this file and the YAML, and stays in the fast
non-widget test group because it never imports Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml


# ---------------------------------------------------------------------------
# The runtime directories an operator constructor may need.
#
# Only two operators are constructed with a directory today:
# PlotAdvancedOperator wants plots_dir and VideoFramesOperator wants
# frames_dir. main.py already computes both values; it packs them into
# this one frozen object and hands the whole thing to every factory. A
# factory that needs neither simply ignores it. This keeps every factory's
# signature identical -- callable(OperatorRuntimeDirs) -> operator -- so
# build_enabled_operators can call them uniformly.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperatorRuntimeDirs:
    """Filesystem locations passed to operator constructors at build time."""

    plots_dir: Path
    frames_dir: Path


# ---------------------------------------------------------------------------
# The one exception this module raises.
#
# Every failure mode raises this single type: a missing config file, YAML
# that does not parse, a top level without an `operators:` mapping, an
# entry that is not a mapping, an entry with no `enabled` key, an `enabled`
# value that is not a bool, and either direction of drift between the YAML
# and OPERATOR_FACTORIES. The message always names the config file, and
# names the offending entry whenever there is one.
#
# There is deliberately no "fall back to registering everything" and no
# "skip the bad entry and carry on". An operator that disappears without a
# word -- or one that silently reappears -- is exactly the failure this
# item exists to remove, so every disagreement stops the program instead.
# ---------------------------------------------------------------------------
class OperatorConfigError(Exception):
    """Raised when operators_config.yaml is missing, malformed, or has
    drifted from OPERATOR_FACTORIES."""


# ---------------------------------------------------------------------------
# The per-operator factories.
#
# Each factory takes an OperatorRuntimeDirs and returns one constructed
# operator instance. The import of the operator module lives INSIDE the
# function body on purpose: importing THIS module then costs nothing and
# drags in no Qt, mediapipe or plotly, and a heavy or broken operator
# module is only touched when its factory is actually invoked -- that is,
# when its YAML entry is enabled. A disabled operator's module is never
# imported.
#
# The dict key is the operator's `name` class attribute, which must equal
# the key used for that operator in operators_config.yaml. That equality
# is the drift guard the whole item is about:
# tests/test_operator_config.py::test_yaml_keys_equal_factory_keys asserts
# the YAML key set equals this dict's key set. (It does not construct the
# operators to re-check each `.name`, because the test must stay Qt-free
# and construct nothing -- keeping each factory key spelled the same as
# the operator's `name` attribute is a manual invariant of this table.)
# ---------------------------------------------------------------------------
def _build_blendshapes(dirs: OperatorRuntimeDirs):
    # BlendshapeOperator takes no constructor arguments.
    from operators.blendshapes import BlendshapeOperator
    return BlendshapeOperator()


def _build_blendshape_avatar(dirs: OperatorRuntimeDirs):
    # BlendshapeAvatarOperator accepts an optional output_dir; main.py
    # passed nothing before this item, so we keep passing nothing.
    from operators.blendshape_avatar import BlendshapeAvatarOperator
    return BlendshapeAvatarOperator()


def _build_mean_face(dirs: OperatorRuntimeDirs):
    # MeanFaceOperator accepts an optional output_dir; main.py passed
    # nothing before this item.
    from operators.mean_face import MeanFaceOperator
    return MeanFaceOperator()


def _build_plot(dirs: OperatorRuntimeDirs):
    # PlotOperator accepts optional columns / output_dir; main.py passed
    # nothing before this item.
    from operators.plot_operator import PlotOperator
    return PlotOperator()


def _build_summary_stats(dirs: OperatorRuntimeDirs):
    # SummaryStatsOperator accepts an optional columns list; main.py
    # passed nothing before this item.
    from operators.summary_stats import SummaryStatsOperator
    return SummaryStatsOperator()


def _build_plot_advanced(dirs: OperatorRuntimeDirs):
    # PlotAdvancedOperator writes its interactive plots to a directory.
    # main.py built this as PlotAdvancedOperator(output_dir=plots_dir).
    from operators.plot_advanced import PlotAdvancedOperator
    return PlotAdvancedOperator(output_dir=dirs.plots_dir)


def _build_stats(dirs: OperatorRuntimeDirs):
    # StatsOperator takes no constructor arguments. It was registered in
    # code before this item but was missing from operators_config.yaml --
    # the drift this item closes.
    from operators.stats_operator import StatsOperator
    return StatsOperator()


def _build_video_frames(dirs: OperatorRuntimeDirs):
    # VideoFramesOperator writes extracted frames to a directory. main.py
    # built this as VideoFramesOperator(output_dir=frames_dir).
    from operators.video_frames import VideoFramesOperator
    return VideoFramesOperator(output_dir=dirs.frames_dir)


# OPERATOR_FACTORIES: operator name -> callable(OperatorRuntimeDirs) -> instance.
# The set of keys here must equal the set of entry keys in
# operators_config.yaml. Order does not matter -- build order comes from
# the YAML file -- but the entries are kept in menu order for readability.
OPERATOR_FACTORIES: dict[str, Callable[[OperatorRuntimeDirs], object]] = {
    "blendshapes": _build_blendshapes,
    "blendshape_avatar": _build_blendshape_avatar,
    "mean_face": _build_mean_face,
    "plot": _build_plot,
    "summary_stats": _build_summary_stats,
    "plot_advanced": _build_plot_advanced,
    "stats": _build_stats,
    "video_frames": _build_video_frames,
}


# ---------------------------------------------------------------------------
# _read_operator_entries: parse and structurally validate the YAML.
#
# Returns the entries as an ordered list of (name, body) pairs, in file
# order. Every structural problem raises OperatorConfigError with the file
# named. This is the one place that touches the filesystem or the YAML
# parser; load_enabled_operator_names and build_enabled_operators both
# build on it.
# ---------------------------------------------------------------------------
def _read_operator_entries(config_path) -> list[tuple[str, dict]]:
    path = Path(config_path)

    # A missing file is a hard error, not "register everything".
    if not path.is_file():
        raise OperatorConfigError(
            f"operator config file not found: {path}"
        )

    # Parse the YAML. yaml.safe_load raises yaml.YAMLError on malformed
    # input; we wrap it so callers only ever see OperatorConfigError.
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise OperatorConfigError(
            f"could not parse YAML in {path}: {exc}"
        ) from exc

    # The top level must be a mapping carrying an `operators:` mapping.
    if not isinstance(parsed, dict) or not isinstance(parsed.get("operators"), dict):
        raise OperatorConfigError(
            f"{path}: expected a top-level 'operators:' mapping"
        )

    operators_block = parsed["operators"]

    # Validate every entry -- enabled ones and disabled ones alike, so a
    # malformed disabled entry is still caught. A YAML mapping preserves
    # file order under PyYAML's loader, so the returned list is in menu
    # order.
    entries: list[tuple[str, dict]] = []
    for entry_name, entry_body in operators_block.items():
        # Each entry's body must itself be a mapping.
        if not isinstance(entry_body, dict):
            raise OperatorConfigError(
                f"{path}: operator entry '{entry_name}' is not a mapping"
            )
        # Every entry must state enabled explicitly -- true or false.
        if "enabled" not in entry_body:
            raise OperatorConfigError(
                f"{path}: operator entry '{entry_name}' has no 'enabled' key"
            )
        # And the value must be a real boolean, not a string or number.
        if not isinstance(entry_body["enabled"], bool):
            raise OperatorConfigError(
                f"{path}: operator entry '{entry_name}' has a non-boolean "
                f"'enabled' value ({entry_body['enabled']!r})"
            )
        entries.append((entry_name, entry_body))

    return entries


# ---------------------------------------------------------------------------
# load_enabled_operator_names: the enabled names, in file order.
# ---------------------------------------------------------------------------
def load_enabled_operator_names(config_path) -> list[str]:
    """Parse operators_config.yaml and return the names of the entries
    whose `enabled` value is true, in the order they appear in the file.

    Raises OperatorConfigError, naming the file, if the file is missing or
    malformed (see _read_operator_entries for the exact conditions).
    """
    entries = _read_operator_entries(config_path)
    return [name for name, body in entries if body["enabled"] is True]


# ---------------------------------------------------------------------------
# build_enabled_operators: the constructed operators, in file order.
#
# This is what main.py calls. It checks drift in BOTH directions before
# building anything:
#   - a YAML entry that is enabled but has no factory here, and
#   - a factory here whose name appears nowhere in the YAML at all.
# A YAML entry that exists but is disabled is fine and is NOT drift.
#
# Both files ship together in the repo, so a disagreement between them is
# a developer's mistake and should stop the program rather than quietly
# change what the researcher can do.
# ---------------------------------------------------------------------------
def build_enabled_operators(config_path, dirs: OperatorRuntimeDirs) -> list:
    """Read operators_config.yaml and return the constructed operator
    instances for every enabled entry, in file order.

    Raises OperatorConfigError, naming the offending name, on drift in
    either direction between the YAML and OPERATOR_FACTORIES.
    """
    entries = _read_operator_entries(config_path)
    path = Path(config_path)

    # All entry names (enabled or not), and the enabled subset, both in
    # file order.
    all_yaml_names = [name for name, _ in entries]
    enabled_names = [name for name, body in entries if body["enabled"] is True]

    # Drift direction 1: an enabled YAML name we do not know how to build.
    for name in enabled_names:
        if name not in OPERATOR_FACTORIES:
            raise OperatorConfigError(
                f"{path}: operator '{name}' is enabled but has no factory "
                f"in OPERATOR_FACTORIES"
            )

    # Drift direction 2: a factory whose name the YAML never mentions.
    # We check against every YAML name, not just the enabled ones, so a
    # deliberately disabled entry still satisfies this side.
    for name in OPERATOR_FACTORIES:
        if name not in all_yaml_names:
            raise OperatorConfigError(
                f"{path}: OPERATOR_FACTORIES defines '{name}' but the file "
                f"has no entry for it"
            )

    # Build in file order, so the caller registers in file order and the
    # Operators menu comes out in file order.
    return [OPERATOR_FACTORIES[name](dirs) for name in enabled_names]
