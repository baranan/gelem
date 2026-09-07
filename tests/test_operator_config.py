"""
tests/test_operator_config.py

Guards operators/operator_config.py, whose job is to make
operators_config.yaml the single authority for WHICH operators the
application offers.

Written from the rules in the P1.11a work item, not from the module's
implementation:

  - the set of YAML entry keys equals the set of OPERATOR_FACTORIES keys
    (the drift guard -- the point of the item);
  - load_enabled_operator_names returns entries in file order and omits a
    disabled one;
  - a missing file, unparseable YAML, and an entry with no `enabled` key
    each raise the module's own exception type, with the file named;
  - build_enabled_operators raises, naming the offender, for drift in
    either direction between the YAML and OPERATOR_FACTORIES.

This module constructs no operator and imports no Qt binding, so it stays
in run_tests.py's fast non-widget group.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from operators.operator_config import (
    OPERATOR_FACTORIES,
    OperatorConfigError,
    OperatorRuntimeDirs,
    build_enabled_operators,
    load_enabled_operator_names,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CONFIG = REPO_ROOT / "operators_config.yaml"


# The registration order main.py must reproduce, per the work item. This
# is both today's menu order and the order the YAML entries must appear in.
EXPECTED_MENU_ORDER = [
    "blendshapes",
    "blendshape_avatar",
    "mean_face",
    "plot",
    "summary_stats",
    "plot_advanced",
    "stats",
    "video_frames",
]


# A throwaway OperatorRuntimeDirs for the build_enabled_operators calls.
# Every such call in this file raises on drift before any operator is
# constructed, so these paths are never touched.
DUMMY_DIRS = OperatorRuntimeDirs(
    plots_dir=Path("unused_plots"),
    frames_dir=Path("unused_frames"),
)


def _write_config(tmp_path: Path, body: str) -> Path:
    """Write a config file into tmp_path and return its path."""
    path = tmp_path / "operators_config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _enabled_entries_yaml(names) -> str:
    """Build a minimal valid YAML listing each name as an enabled entry,
    in the given order."""
    lines = ["operators:"]
    for name in names:
        lines.append(f"  {name}:")
        lines.append("    enabled: true")
        lines.append(f'    description: "entry for {name}"')
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The drift guard: YAML keys == OPERATOR_FACTORIES keys.
# ---------------------------------------------------------------------------
def test_yaml_keys_equal_factory_keys():
    # Read every entry key from the real config, enabled or not, via a
    # YAML parse. load_enabled_operator_names only returns enabled ones,
    # so parse the file directly for the full set.
    import yaml

    parsed = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    yaml_keys = set(parsed["operators"].keys())
    factory_keys = set(OPERATOR_FACTORIES.keys())

    assert yaml_keys == factory_keys, (
        "operators_config.yaml and OPERATOR_FACTORIES have drifted: "
        f"only in YAML = {yaml_keys - factory_keys}, "
        f"only in factories = {factory_keys - yaml_keys}"
    )


def test_real_config_lists_operators_in_menu_order():
    # Every real entry is enabled today, so the enabled list is the whole
    # file in order, and that order is the Operators menu order.
    assert load_enabled_operator_names(REAL_CONFIG) == EXPECTED_MENU_ORDER


# ---------------------------------------------------------------------------
# load_enabled_operator_names: file order, disabled entries omitted.
# ---------------------------------------------------------------------------
def test_load_enabled_names_is_file_order_and_omits_disabled(tmp_path):
    body = (
        "operators:\n"
        "  gamma:\n"
        "    enabled: true\n"
        "  alpha:\n"
        "    enabled: false\n"
        "  beta:\n"
        "    enabled: true\n"
    )
    path = _write_config(tmp_path, body)

    # gamma before beta (file order, not alphabetical); alpha omitted.
    assert load_enabled_operator_names(path) == ["gamma", "beta"]


# ---------------------------------------------------------------------------
# Failure behaviour: one exception type, file named in the message.
# ---------------------------------------------------------------------------
def test_missing_file_raises_named_error(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"

    with pytest.raises(OperatorConfigError) as excinfo:
        load_enabled_operator_names(missing)

    assert str(missing) in str(excinfo.value)


def test_unparseable_yaml_raises_named_error(tmp_path):
    # An unterminated flow sequence -- yaml.safe_load raises YAMLError.
    path = _write_config(tmp_path, "operators: [1, 2, 3\n")

    with pytest.raises(OperatorConfigError) as excinfo:
        load_enabled_operator_names(path)

    assert str(path) in str(excinfo.value)


def test_entry_without_enabled_key_raises_named_error(tmp_path):
    body = (
        "operators:\n"
        "  blendshapes:\n"
        '    description: "no enabled key here"\n'
    )
    path = _write_config(tmp_path, body)

    with pytest.raises(OperatorConfigError) as excinfo:
        load_enabled_operator_names(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "blendshapes" in message


def test_non_boolean_enabled_value_raises_named_error(tmp_path):
    # `enabled` must be a real YAML boolean. A quoted string such as
    # "true" is not accepted -- otherwise "false" (a truthy string) would
    # silently enable a disabled operator.
    body = (
        "operators:\n"
        "  blendshapes:\n"
        '    enabled: "true"\n'
    )
    path = _write_config(tmp_path, body)

    with pytest.raises(OperatorConfigError) as excinfo:
        load_enabled_operator_names(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "blendshapes" in message


# ---------------------------------------------------------------------------
# build_enabled_operators: drift in either direction raises, naming the
# offender. Neither case reaches operator construction.
# ---------------------------------------------------------------------------
def test_build_raises_for_yaml_name_with_no_factory(tmp_path):
    # Every expected operator name, all enabled, plus one extra name that
    # no factory knows about. The YAML names are hard-coded (not taken
    # from OPERATOR_FACTORIES) on purpose: the extra name must be the one
    # reported, which proves the check walks past the real names and does
    # not merely raise on the first entry -- and it also means this test
    # fails, rather than passing vacuously, if OPERATOR_FACTORIES were
    # emptied.
    names = EXPECTED_MENU_ORDER + ["phantom_operator"]
    path = _write_config(tmp_path, _enabled_entries_yaml(names))

    with pytest.raises(OperatorConfigError) as excinfo:
        build_enabled_operators(path, DUMMY_DIRS)

    assert "phantom_operator" in str(excinfo.value)


def test_build_raises_for_factory_name_absent_from_yaml(tmp_path):
    # Every expected operator name except the last one. That omitted name
    # is a real factory the YAML no longer mentions, so it must be the one
    # reported. Hard-coding the kept names (rather than slicing
    # OPERATOR_FACTORIES.keys()) makes this test fail if the factory table
    # were emptied instead of passing on the wrong offender.
    omitted = EXPECTED_MENU_ORDER[-1]
    kept = EXPECTED_MENU_ORDER[:-1]
    path = _write_config(tmp_path, _enabled_entries_yaml(kept))

    with pytest.raises(OperatorConfigError) as excinfo:
        build_enabled_operators(path, DUMMY_DIRS)

    assert omitted in str(excinfo.value)


def test_present_but_disabled_is_not_drift(tmp_path):
    # A YAML entry that exists but is disabled satisfies the "factory name
    # present in YAML" side, so build_enabled_operators must get past both
    # drift checks. We assert only that no OperatorConfigError is raised
    # for the drift reason -- this test must not construct an operator, so
    # we keep every real entry DISABLED. The result is an empty build,
    # which is a legitimate (if useless) configuration and exactly what
    # exercises "disabled is not drift" without importing mediapipe/Qt.
    lines = ["operators:"]
    for name in EXPECTED_MENU_ORDER:
        lines.append(f"  {name}:")
        lines.append("    enabled: false")
    path = _write_config(tmp_path, "\n".join(lines) + "\n")

    # Every factory name is present in the file, every entry is disabled:
    # no drift, nothing to build.
    assert build_enabled_operators(path, DUMMY_DIRS) == []
