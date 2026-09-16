"""
tests/test_output_copy_warning.py

Layer A of ui/output_copy_warning.py (P1.9b-2): the plain-data wording and
arithmetic behind the copy-on-save warning. No Layer B (QMessageBox) tests
here -- Layer A needs no QApplication.

This module names the Qt binding and QMessageBox only inside a source-scan
assertion (test_layer_a_functions_are_free_of_qt below) and never imports
either, so it carries a "# run-tests: combined" token -- see run_tests.py's
WIDGET_MARKERS comment for why the plain substring heuristic would
otherwise force it into isolation it does not need.

Written from the work-item specification, not from the implementation.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ui.output_copy_warning import (
    format_size_mb_gb,
    should_warn,
    warning_message,
)

_MB = 1024 * 1024
_GB = 1024 * _MB


# ===========================================================================
# format_size_mb_gb -- MB below 1 GB, GB at or above it, one decimal.
# ===========================================================================

def test_small_size_shown_in_mb_with_one_decimal():
    assert format_size_mb_gb(5 * _MB) == "5.0 MB"
    assert format_size_mb_gb(int(2.5 * _MB)) == "2.5 MB"


def test_exactly_one_gb_shown_in_gb():
    assert format_size_mb_gb(_GB) == "1.0 GB"


def test_just_below_one_gb_still_rounds_up_to_gb_not_1024_mb():
    # One byte short of a GB rounds to "1024.0 MB" at one decimal place,
    # which would misleadingly look like it should already be a GB figure.
    # The switch is decided on the ROUNDED MB number, not the raw byte
    # count, so this must read "1.0 GB" instead.
    assert format_size_mb_gb(_GB - 1) == "1.0 GB"


def test_large_size_shown_in_gb_with_one_decimal():
    assert format_size_mb_gb(int(2.3 * _GB)) == "2.3 GB"


# ===========================================================================
# warning_message -- the file count appears, and the mandatory
# unresponsive-window sentence is present (build spec item 2).
# ===========================================================================

def test_warning_message_names_the_file_count_and_size():
    text = warning_message(file_count=7, total_bytes=10 * _MB)
    assert "7" in text
    assert "10.0 MB" in text


def test_warning_message_uses_singular_noun_for_one_file():
    text = warning_message(file_count=1, total_bytes=_MB)
    assert "1 output file " in text
    assert "1 output files" not in text


def test_warning_message_warns_the_window_will_be_unresponsive():
    text = warning_message(file_count=3, total_bytes=_MB)
    assert "unresponsive" in text.lower()


# ===========================================================================
# should_warn -- warn only when total_bytes is strictly greater than the
# threshold. One test, Qt-free and controller-free.
# ===========================================================================

def test_should_warn_only_above_the_threshold():
    assert should_warn(total_bytes=101, threshold_bytes=100) is True
    assert should_warn(total_bytes=100, threshold_bytes=100) is False
    assert should_warn(total_bytes=99, threshold_bytes=100) is False


# ===========================================================================
# Layer A stays Qt-free: format_size_mb_gb, should_warn and warning_message
# import nothing and reference no Qt name in their own bodies. Layer B
# (confirm_output_copy) is exempt -- it is the thin QMessageBox glue and
# the module-level PySide6 import above it exists only for Layer B's sake.
# ===========================================================================

_LAYER_A_FUNCTIONS = ("format_size_mb_gb", "should_warn", "warning_message")


def test_layer_a_functions_are_free_of_qt():
    source = (PROJECT_ROOT / "ui" / "output_copy_warning.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    checked = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in _LAYER_A_FUNCTIONS:
            checked += 1
            for sub in ast.walk(node):
                assert not isinstance(sub, (ast.Import, ast.ImportFrom)), (
                    f"{node.name} imports something inline -- "
                    f"Layer A must stay Qt-free"
                )
                if isinstance(sub, ast.Name):
                    assert "PySide6" not in sub.id
                    assert "QMessageBox" not in sub.id
                if isinstance(sub, ast.Attribute):
                    assert "QMessageBox" not in sub.attr

    assert checked == len(_LAYER_A_FUNCTIONS), (
        "did not find all three Layer A functions in "
        "ui/output_copy_warning.py"
    )


# run-tests: combined -- names the Qt binding in a source-scan assertion but never imports it
