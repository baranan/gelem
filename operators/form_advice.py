"""
operators/form_advice.py -- the form-guidance vocabulary (P1.12e-4a).

An operator may look at what the researcher has typed into the generated
parameter form so far and hand the form three kinds of guidance:

  * which fields do not apply to the current combination of values,
  * which choices are still allowed for a field, and
  * what to say about the current combination -- a warning the researcher
    may proceed past, or an error that blocks the run.

This module is ONLY that vocabulary: two frozen dataclasses, one
exception type, and the rules a well-formed piece of guidance must
satisfy. It has NO consumers yet. ``operators/base.py`` gains a
``refine_form`` hook that returns ``FormAdvice()`` by default, and
``ui/parameter_dialog.py`` Layer A gains a pure function that applies one
-- but nothing WIRES an operator's advice into the dialog until
P1.12e-4b.

OWNERSHIP (decided): this lives under ``operators/`` because an OPERATOR
returns a ``FormAdvice`` and an operator may never import from the UI. It
is a separate module from ``operators/descriptor.py`` on purpose -- the
descriptor is the static declaration of what a parameter IS, and this is
a per-keystroke answer about one concrete set of values. A different
subject with a different lifetime.

DESIGN CONSTRAINTS -- the same ones ``operators/descriptor.py`` states:
standard library only (no Qt, no pandas, nothing from elsewhere in this
repo), every dataclass frozen, every tuple field a tuple with a list
refused rather than quietly copied.

TWO RULES A FUTURE AUTHOR WILL GET WRONG
=======================================

1. An answer describes the COMPLETE state every time. ``refine_form`` is
   called afresh on every change, and it must recompute the whole answer
   from the parameter declarations and the current values -- which fields
   are inapplicable NOW, which choices are allowed NOW. It must never
   accumulate onto a previous answer: a field that has stopped being
   inapplicable only drops off the list if the list is rebuilt from
   scratch each time.

2. Advice never writes a value back. It may say "this field does not
   apply" or "only these choices are valid", but it never changes what
   the researcher just typed. A form that rewrites the person's input
   while they are still typing fights them -- the cursor jumps, a
   half-typed value vanishes. The form disables an inapplicable field and
   PRESERVES its value; it never substitutes one.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Optional


class FormAdviceError(Exception):
    """
    Raised when a ``FormMessage`` or ``FormAdvice`` is constructed with
    values that cannot form well-formed guidance -- a severity that is
    neither "warning" nor "error", an empty parameter name, an empty
    message text, or a list where a tuple is required.

    Like ``OperatorDescriptorError`` it is a plain ``Exception`` subclass:
    a caller building advice should let it propagate as a programming
    error, not catch and recover from it.
    """


# The only two severities. A "warning" explains and lets the researcher
# proceed; an "error" blocks acceptance of the form.
_VALID_SEVERITIES = ("warning", "error")


def _require_non_empty_name(value: object, what: str) -> None:
    """Raise ``FormAdviceError`` unless ``value`` is a non-empty string
    after stripping blanks.

    Used for every parameter name this module stores: a name that is
    blank could not identify a field.
    """
    if not isinstance(value, str) or value.strip() == "":
        raise FormAdviceError(f"{what} must be a non-empty string.")


def _require_tuple(value: object, what: str) -> None:
    """Raise ``FormAdviceError`` unless ``value`` is a tuple.

    Mirrors ``operators/descriptor.py``'s helper of the same name: a list
    is refused outright rather than quietly copied to a tuple, so a
    caller cannot keep a mutable handle to the guidance's insides.
    """
    if not isinstance(value, tuple):
        raise FormAdviceError(
            f"{what} must be a tuple, got {type(value).__name__}."
        )


@dataclass(frozen=True)
class FormMessage:
    """One thing to tell the researcher about the current values.

    ``text``     -- the sentence shown in the form.
    ``severity`` -- exactly "warning" or "error". A warning explains and
                    lets the researcher proceed; an error blocks
                    acceptance.
    ``field``    -- the parameter name this message belongs beside, or
                    ``None`` when it is about the form as a whole.
    """

    text: str
    severity: str
    field: Optional[str] = None

    def __post_init__(self) -> None:
        # Severity is a closed set of two words.
        if self.severity not in _VALID_SEVERITIES:
            raise FormAdviceError(
                f"FormMessage.severity must be one of {_VALID_SEVERITIES}, "
                f"got {self.severity!r}."
            )
        # An empty message says nothing to the researcher.
        if not isinstance(self.text, str) or self.text.strip() == "":
            raise FormAdviceError(
                "FormMessage.text must be a non-empty string."
            )
        # field is optional, but when given it must name a real parameter.
        if self.field is not None:
            _require_non_empty_name(self.field, "FormMessage.field")


@dataclass(frozen=True)
class FormAdvice:
    """An operator's complete answer for one set of parameter values.

    Every field is optional, so ``FormAdvice()`` is the "no guidance"
    answer -- nothing inapplicable, nothing restricted, nothing to say.

    ``inapplicable``    -- the parameter names that do not apply to the
                           current combination. The form disables each
                           and PRESERVES its value.
    ``allowed_choices`` -- parameter name -> the tuple of values still
                           allowed for it. A field absent from this
                           mapping is unrestricted.
    ``messages``        -- the ``FormMessage`` objects to display.

    ``allowed_choices`` is a plain dict rather than a tuple of pairs: this
    object lives for one keystroke and is never hashed or cached, unlike a
    descriptor. A list is still refused -- as the mapping itself and as
    any field's values.

    See the module docstring for the two rules that keep this safe: the
    answer is recomputed whole every time, and it never writes a value
    back.
    """

    inapplicable: tuple[str, ...] = ()
    allowed_choices: dict = dataclass_field(default_factory=dict)
    messages: tuple[FormMessage, ...] = ()

    def __post_init__(self) -> None:
        # inapplicable: a tuple of non-empty parameter names.
        _require_tuple(self.inapplicable, "FormAdvice.inapplicable")
        for name in self.inapplicable:
            _require_non_empty_name(name, "FormAdvice.inapplicable entry")

        # allowed_choices: a dict of parameter name -> tuple of values. A
        # list anywhere -- as the mapping or as one field's values -- is
        # refused, the same discipline descriptor.py applies to its own
        # collection fields.
        if not isinstance(self.allowed_choices, dict):
            raise FormAdviceError(
                "FormAdvice.allowed_choices must be a dict, got "
                f"{type(self.allowed_choices).__name__}."
            )
        for name, values in self.allowed_choices.items():
            _require_non_empty_name(name, "FormAdvice.allowed_choices key")
            _require_tuple(values, f"FormAdvice.allowed_choices[{name!r}]")

        # messages: a tuple of FormMessage. Each one validated itself at
        # construction; here we only insist the container is a tuple and
        # holds the right type.
        _require_tuple(self.messages, "FormAdvice.messages")
        for message in self.messages:
            if not isinstance(message, FormMessage):
                raise FormAdviceError(
                    "FormAdvice.messages must contain FormMessage objects, "
                    f"got {type(message).__name__}."
                )
