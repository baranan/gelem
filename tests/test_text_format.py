"""
tests/test_text_format.py

column_types/text_format.format_cell_text turns a cell value into display
text by the column's schema type tag. Written from the rules in that
module's docstring: every missing scalar is "", integers are exact, floats
are trimmed to 4 decimals with scientific notation only below 0.0001
(non-zero) or from 1e12 up, and a media address is shown unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from column_types.text_format import format_cell_text


@pytest.mark.parametrize(
    "missing", [None, float("nan"), np.nan, pd.NA, pd.NaT, np.datetime64("NaT")]
)
def test_every_missing_scalar_is_the_empty_string(missing):
    assert format_cell_text(missing, "numeric") == ""
    assert format_cell_text(missing, "text") == ""
    assert format_cell_text(missing, "media_path") == ""
    assert format_cell_text(missing, None) == ""


def test_an_empty_string_stays_empty_and_zero_is_not_missing():
    assert format_cell_text("", "text") == ""
    assert format_cell_text(0, "numeric") == "0"
    assert format_cell_text(0.0, "numeric") == "0"


def test_integers_are_exact():
    assert format_cell_text(42, "numeric") == "42"
    assert format_cell_text(-7, "numeric") == "-7"
    assert format_cell_text(np.int64(5), "numeric") == "5"
    # Beyond float precision: a float round trip would corrupt it.
    big = 9007199254740993
    assert format_cell_text(np.int64(big), "numeric") == "9007199254740993"
    assert format_cell_text(big, "numeric") == "9007199254740993"
    assert format_cell_text(np.int64(np.iinfo(np.int64).max), "numeric") == str(
        np.iinfo(np.int64).max
    )


def test_float_trailing_zeros_are_trimmed_and_decimals_capped_at_four():
    assert format_cell_text(1.5, "numeric") == "1.5"
    assert format_cell_text(3.0, "numeric") == "3"
    assert format_cell_text(np.float64(0.25), "numeric") == "0.25"
    assert format_cell_text(0.123456, "numeric") == "0.1235"
    assert format_cell_text(-2.50, "numeric") == "-2.5"
    assert format_cell_text(0.99999, "numeric") == "1"


def test_small_floats_go_scientific_only_below_one_ten_thousandth():
    # Exactly 0.0001 is still fixed notation.
    assert format_cell_text(0.0001, "numeric") == "0.0001"
    # Just under it is scientific, four significant digits.
    assert format_cell_text(0.00001234, "numeric") == "1.234e-05"
    assert format_cell_text(-0.00001234, "numeric") == "-1.234e-05"
    # Zero is never scientific, however it is signed.
    assert format_cell_text(0.0, "numeric") == "0"
    assert format_cell_text(-0.0, "numeric") == "0"


def test_large_floats_go_scientific_from_1e12():
    assert format_cell_text(999999999999.0, "numeric") == "999999999999"
    assert format_cell_text(1e12, "numeric") == "1.000e+12"
    assert format_cell_text(-2.5e15, "numeric") == "-2.500e+15"


def test_a_media_address_is_shown_unchanged():
    address = "clips/a b.mp4#t=1.5,3.0&r=10,20,30,40"
    assert format_cell_text(address, "media_path") == address
    assert format_cell_text("frames/x.png#f=12", "media_path") == "frames/x.png#f=12"


def test_everything_else_is_str():
    assert format_cell_text("hello", "text") == "hello"
    assert format_cell_text(True, "text") == "True"
    assert format_cell_text(np.bool_(False), "text") == "False"
    assert format_cell_text([1, 2], "text") == "[1, 2]"
    assert format_cell_text(pd.Timestamp("2024-01-02"), None) == str(
        pd.Timestamp("2024-01-02")
    )
