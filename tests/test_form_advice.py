"""
tests/test_form_advice.py

The form-guidance vocabulary (P1.12e-4a): ``operators/form_advice.py``.

This module is standard-library only, so these tests import nothing else
from the repo and need no QApplication and no data library.

Written from the work-item specification, not the implementation.

Run with:
    python -m pytest tests/test_form_advice.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from operators.form_advice import (
    FormAdvice,
    FormAdviceError,
    FormMessage,
)


# ===========================================================================
# FormMessage -- validation rules
# ===========================================================================

def test_severity_must_be_warning_or_error():
    with pytest.raises(FormAdviceError):
        FormMessage(text="something is off", severity="info")


def test_warning_and_error_are_both_accepted():
    warning = FormMessage(text="you may proceed", severity="warning")
    error = FormMessage(text="this blocks", severity="error")
    assert warning.severity == "warning"
    assert error.severity == "error"


def test_empty_message_text_is_rejected():
    with pytest.raises(FormAdviceError):
        FormMessage(text="   ", severity="warning")


def test_empty_field_name_is_rejected():
    # field is optional, but a blank string is not the same as "no field".
    with pytest.raises(FormAdviceError):
        FormMessage(text="pick one", severity="error", field="")


def test_a_message_about_the_whole_form_has_no_field():
    message = FormMessage(text="these two settings conflict", severity="error")
    assert message.field is None


# ===========================================================================
# FormAdvice -- the empty answer, and validation rules
# ===========================================================================

def test_default_form_advice_is_empty_and_valid():
    advice = FormAdvice()
    assert advice.inapplicable == ()
    assert advice.allowed_choices == {}
    assert advice.messages == ()


def test_empty_name_in_inapplicable_is_rejected():
    with pytest.raises(FormAdviceError):
        FormAdvice(inapplicable=("  ",))


def test_empty_key_in_allowed_choices_is_rejected():
    with pytest.raises(FormAdviceError):
        FormAdvice(allowed_choices={"": ("count",)})


def test_a_list_where_a_tuple_belongs_is_rejected():
    # inapplicable must be a tuple, exactly as descriptor.py refuses a list.
    with pytest.raises(FormAdviceError):
        FormAdvice(inapplicable=["frame_step"])


def test_a_list_of_allowed_values_is_rejected():
    with pytest.raises(FormAdviceError):
        FormAdvice(allowed_choices={"agg": ["count", "sum"]})


def test_a_list_for_the_allowed_choices_mapping_is_rejected():
    with pytest.raises(FormAdviceError):
        FormAdvice(allowed_choices=[("agg", ("count",))])


def test_messages_must_be_a_tuple_not_a_list():
    with pytest.raises(FormAdviceError):
        FormAdvice(messages=[FormMessage(text="hi", severity="warning")])


def test_messages_must_contain_form_messages():
    with pytest.raises(FormAdviceError):
        FormAdvice(messages=("not a message",))


def test_a_well_formed_advice_keeps_what_it_was_given():
    message = FormMessage(text="bar only", severity="error", field="agg")
    advice = FormAdvice(
        inapplicable=("bins",),
        allowed_choices={"agg": ("count", "sum", "mean")},
        messages=(message,),
    )
    assert advice.inapplicable == ("bins",)
    assert advice.allowed_choices == {"agg": ("count", "sum", "mean")}
    assert advice.messages == (message,)
