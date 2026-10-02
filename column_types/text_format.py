"""
column_types/text_format.py

Turns one table cell into the plain text a table view shows. Qt-free and
decodes no media: it only looks at the value and at the column's schema
type tag.

The rules, in the order they are applied:

  1. Every missing scalar (None, NaN, pd.NA, NaT) is the empty string.
  2. A column tagged "media_path" shows its address text exactly as stored.
     The fragment (#f=, #t=, ...) is part of the address and is never cut.
  3. Booleans are "True" / "False". (bool is a subclass of int in Python,
     so this has to be decided before the integer rule.)
  4. Integers, Python or numpy, are written exactly, with no rounding.
  5. Floats get up to 4 decimals with trailing zeros trimmed. Scientific
     notation with 4 significant digits is used only when the value is
     non-zero and its absolute value is below 0.0001, or is at least 1e12,
     because fixed notation would show 0 for the first and a wall of
     digits for the second.
  6. Everything else is str(value).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The schema type tag of a column holding media addresses. Written here as
# a plain string, not imported from the registry, so this module stays a
# leaf with no dependency on the rest of column_types.
MEDIA_PATH_TAG = "media_path"

# Floats below this absolute value (and non-zero) switch to scientific
# notation; so do floats at or above SCIENTIFIC_UPPER.
SCIENTIFIC_LOWER = 1e-4
SCIENTIFIC_UPPER = 1e12

# Decimal places kept in fixed notation.
FIXED_DECIMALS = 4

# Significant digits in scientific notation. Python's "e" format counts
# digits after the point, so 4 significant digits is ".3e".
SCIENTIFIC_FORMAT = ".3e"


def _is_missing_scalar(value) -> bool:
    """True for None, NaN, pd.NA and NaT. A list, tuple, dict, set or array
    is never "missing" -- pd.isna would answer elementwise for those, and
    the answer would not be a single bool."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set, frozenset, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        # Something pandas cannot classify is not missing; str() shows it.
        return False


def _format_float(value: float) -> str:
    """Fixed notation (trimmed) or scientific, per the module rules."""
    magnitude = abs(value)
    if magnitude != 0 and (
        magnitude < SCIENTIFIC_LOWER or magnitude >= SCIENTIFIC_UPPER
    ):
        return format(value, SCIENTIFIC_FORMAT)
    text = format(value, f".{FIXED_DECIMALS}f")
    # Trim trailing zeros, then a bare trailing point: "1.5000" -> "1.5",
    # "3.0000" -> "3".
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    # A tiny negative that rounds to zero in fixed notation cannot reach
    # here (it would be below SCIENTIFIC_LOWER), but -0.0 can.
    if text == "-0":
        text = "0"
    return text


def format_cell_text(value, type_tag: str | None) -> str:
    """Display text for one cell.

    Args:
        value:    The raw cell value, as stored.
        type_tag: The column's schema type tag, or None when unknown.
    """
    if _is_missing_scalar(value):
        return ""
    if type_tag == MEDIA_PATH_TAG:
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _format_float(float(value))
    return str(value)
