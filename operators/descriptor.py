"""
operators/descriptor.py -- the operator descriptor vocabulary.

WHAT THIS MODULE IS (for a reader who has never seen this design)

Every Gelem operator will eventually carry one ``OperatorDescriptor``: a
pure-data, immutable description of what the operator is, which execution
modes it offers, what data each mode needs, what parameters each mode
takes, and what each mode produces. Three consumers are planned:

  * the Operators menu and the generated parameter dialog are built from
    it (P1.12c -- P1.12e);
  * the result cache keys on the version and determinism information;
  * a planned natural-language interface uses it to drive operators
    reliably.

This file is ONLY the vocabulary -- the set of frozen dataclasses and the
rules that a well-formed descriptor must satisfy. It has no consumers yet;
they arrive in later P1.12 sub-items. Nothing else in the repo imports it
yet, and that is deliberate: this is the contract half of a
contract/consumer seam.

DESIGN CONSTRAINTS

  * Standard library only. This module must be importable from a test, a
    script, or a future headless terminal, so it pulls in no Qt, no
    pandas, no numpy, no imaging library, and nothing from elsewhere in
    this repo.
  * Every data class here is a frozen dataclass, and every collection
    field must be a tuple, never a list, so an instance is genuinely
    immutable once built. ``__post_init__`` refuses a list rather than
    copying it, because silently accepting one would let a caller keep a
    mutable handle to a descriptor's insides.
  * All dataclasses are declared ``kw_only=True``. Several of them have a
    field with a default (for example ``ModeDescriptor.parameters``)
    sitting before a field without one (``ModeDescriptor.output``);
    keyword-only fields sidestep the "non-default argument follows default
    argument" problem without reordering the fields away from the order
    the design lists them in. Callers therefore always pass these by
    keyword.
  * Validation happens in ``__post_init__`` and raises
    ``OperatorDescriptorError`` -- a plain ``Exception`` subclass defined
    here -- so a malformed descriptor fails loudly at construction rather
    than misbehaving later in the UI.
"""

from __future__ import annotations

# Standard library only -- see the module docstring.
import keyword
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OperatorDescriptorError(Exception):
    """
    Raised when a descriptor object is constructed with values that
    cannot form a well-formed operator description -- an empty label, a
    parameter name that is not a Python identifier, an output shape that
    is neither columns nor a table nor a display, and so on.

    It is a plain ``Exception`` subclass on purpose: callers that build
    descriptors should let it propagate as a programming error, not catch
    and recover from it.
    """


# ---------------------------------------------------------------------------
# Small validation helpers.
#
# These are module-private. Each one raises OperatorDescriptorError with a
# message naming the offending field, so a failure points straight at the
# descriptor that is wrong.
# ---------------------------------------------------------------------------

def _require_non_empty(value: str, what: str) -> None:
    """Raise unless ``value`` is a non-empty string after stripping blanks."""
    if not isinstance(value, str) or value.strip() == "":
        raise OperatorDescriptorError(f"{what} must be a non-empty string.")


def _require_identifier(value: str, what: str) -> None:
    """Raise unless ``value`` is a valid Python identifier.

    Internal keys (input names, parameter names, operator names) become
    dictionary keys the operator reads at run time and, for the operator
    name, must line up with a Python class attribute, so anything that is
    not an identifier is refused here.
    """
    _require_non_empty(value, what)
    # str.isidentifier() is true for reserved words too ("class", "None",
    # "return"), and none of those can actually be used as an attribute or
    # a sane dictionary key, so a keyword is refused as well.
    if not value.isidentifier() or keyword.iskeyword(value):
        raise OperatorDescriptorError(
            f"{what} must be a valid Python identifier and not a reserved "
            f"word, got {value!r}."
        )


def _require_tuple(value: object, what: str) -> None:
    """Raise unless ``value`` is a tuple.

    Every collection field in this module is declared as a tuple so that a
    constructed descriptor is genuinely immutable -- a frozen dataclass
    stops the field being rebound, but a list stored in it could still be
    mutated in place, and it would make the descriptor unhashable for the
    planned result-cache key. A list is refused outright rather than
    quietly copied to a tuple.
    """
    if not isinstance(value, tuple):
        raise OperatorDescriptorError(
            f"{what} must be a tuple, got {type(value).__name__}."
        )


def _require_unique(names: tuple[str, ...], what: str) -> None:
    """Raise if ``names`` contains a duplicate."""
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise OperatorDescriptorError(
                f"{what} must be unique, {name!r} appears more than once."
            )
        seen.add(name)


# ---------------------------------------------------------------------------
# Enumerations.
#
# Plain enums, not dataclasses -- they carry no data of their own, just a
# closed set of named choices that the dataclasses below refer to.
# ---------------------------------------------------------------------------

class ExecutionMode(Enum):
    """The three kinds of work an operator mode can do.

    COLUMNS -- add new columns to the rows it is given (per-row analysis).
    TABLE   -- produce a brand-new named table (aggregation / derivation).
    DISPLAY -- produce a result shown in the Results panel and stored
               nowhere.
    """

    COLUMNS = "columns"
    TABLE = "table"
    DISPLAY = "display"


class ModelLifecycle(Enum):
    """How often the runner must build this mode's model, if any.

    NONE         -- the mode uses no model.
    SHARED       -- one immutable, demonstrably thread-safe instance for
                    the whole application.
    PER_WORKER   -- one instance per worker thread or process.
    PER_SEQUENCE -- one isolated instance per clip-run, reset at the
                    sequence boundary (the only correct choice for a model
                    that tracks state across frames).

    See ``operators/CLAUDE.md`` -> "Where a model lives" for why these are
    the four options.
    """

    NONE = "none"
    SHARED = "shared"
    PER_WORKER = "per_worker"
    PER_SEQUENCE = "per_sequence"


class InputKind(Enum):
    """Where one declared data input comes from.

    ACTIVE_TABLE  -- whatever table the researcher is currently looking
                     at; no dialog choice needed.
    NAMED_TABLE   -- the researcher picks a table by name in the generated
                     dialog.
    WHOLE_PROJECT -- the explicit "this operator reads everything" option.
                     Declaring it makes a project-wide read visible in the
                     descriptor instead of leaving it implicit.
    """

    ACTIVE_TABLE = "active_table"
    NAMED_TABLE = "named_table"
    WHOLE_PROJECT = "whole_project"


class MediaRequirement(Enum):
    """What the runner hands the operator for each item it processes.

    A mode declares exactly one of these, and the runner uses it to decide
    what -- if anything -- to decode before calling the operator. The
    wording matches ``operators/CLAUDE.md`` -> "Declared inputs":

    METADATA   -- no media at all. The operator works from the row's
                  ordinary columns; ``media`` is ``None``.
    FRAME      -- a single decoded frame.
    VIDEO_SPAN -- an ordered span of video, for sequential work.
    AUDIO_SPAN -- an audio span.
    ADDRESS    -- the raw media address; the operator resolves it itself
                  (the video frame-extraction operator in this repo is the
                  real example -- a TABLE-mode operator that walks the
                  address on its own).

    This enum is what replaces the current boolean ``requires_image`` when
    P1.12d migrates the runner: ``requires_image`` is only the narrow
    "FRAME or not" version of the same decision.
    """

    METADATA = "metadata"
    FRAME = "frame"
    VIDEO_SPAN = "video_span"
    AUDIO_SPAN = "audio_span"
    ADDRESS = "address"


# ---------------------------------------------------------------------------
# InputSpec -- one declared data input for one mode.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class InputSpec:
    """One piece of data a mode needs handed to it.

    ``name`` is the internal key the operator uses at run time; ``label``
    is what the researcher sees in the generated dialog.
    """

    # Internal key -- must be a Python identifier so the operator can use
    # it as a dictionary key.
    name: str
    # Researcher-facing label in the generated dialog.
    label: str
    # Which source this input is drawn from.
    kind: InputKind
    # Optional one-line explanation shown beside the field.
    help_text: str = ""
    # Whether the mode can run without this input supplied.
    required: bool = True

    def __post_init__(self) -> None:
        # name must be a usable internal key; label must be shown to a
        # human, so it only has to be non-empty.
        _require_identifier(self.name, "InputSpec.name")
        _require_non_empty(self.label, "InputSpec.label")


# ---------------------------------------------------------------------------
# Parameter specs.
#
# ParameterSpec is the frozen base class. Each concrete subclass adds a
# `kind` class attribute -- a plain string, NOT a dataclass field -- that
# the dialog generator in P1.12e reads to pick a widget. Subclasses call
# super().__post_init__() so the shared name/label checks always run.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class ParameterSpec:
    """Common shape of every declared parameter.

    A concrete parameter is one of the subclasses below; this base only
    carries the fields and validation they all share.
    """

    # Internal key the operator reads from ``run.parameters``.
    name: str
    # Researcher-facing label in the generated dialog.
    label: str
    # Optional one-line explanation shown beside the field.
    help_text: str = ""
    # Whether the researcher must supply a value.
    required: bool = True

    def __post_init__(self) -> None:
        _require_identifier(self.name, "ParameterSpec.name")
        _require_non_empty(self.label, "ParameterSpec.label")


@dataclass(frozen=True, kw_only=True)
class NumberParameter(ParameterSpec):
    """A numeric parameter, rendered as a spin box or similar.

    ``minimum``, ``maximum`` and ``default`` are all optional. When both
    bounds are given they must be ordered; when a default is given it must
    sit inside whatever bounds are given.
    """

    kind = "number"

    minimum: Optional[float] = None
    maximum: Optional[float] = None
    # How many decimal places the widget should show / accept.
    decimals: int = 0
    default: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        # Decimal places is a display precision -- it cannot be negative.
        # Checked here so the failure lands at descriptor construction, the
        # same place minimum / maximum / default are checked, rather than
        # in the P1.12e dialog generator.
        if self.decimals < 0:
            raise OperatorDescriptorError(
                f"NumberParameter {self.name!r}: decimals ({self.decimals}) "
                "cannot be negative."
            )
        # Ordered bounds, when both are present.
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise OperatorDescriptorError(
                    f"NumberParameter {self.name!r}: minimum "
                    f"({self.minimum}) must be <= maximum ({self.maximum})."
                )
        # Default inside whatever bounds are given.
        if self.default is not None:
            if self.minimum is not None and self.default < self.minimum:
                raise OperatorDescriptorError(
                    f"NumberParameter {self.name!r}: default "
                    f"({self.default}) is below minimum ({self.minimum})."
                )
            if self.maximum is not None and self.default > self.maximum:
                raise OperatorDescriptorError(
                    f"NumberParameter {self.name!r}: default "
                    f"({self.default}) is above maximum ({self.maximum})."
                )


@dataclass(frozen=True, kw_only=True)
class TextParameter(ParameterSpec):
    """A free-text parameter, rendered as a line edit."""

    kind = "text"

    default: str = ""


@dataclass(frozen=True, kw_only=True)
class BooleanParameter(ParameterSpec):
    """A yes/no parameter, rendered as a checkbox."""

    kind = "boolean"

    default: bool = False


@dataclass(frozen=True, kw_only=True)
class ChoiceParameter(ParameterSpec):
    """A pick-one parameter, rendered as a dropdown.

    ``choices`` is a tuple of ``(value, label)`` pairs: the value is what
    the operator receives, the label is what the researcher sees.
    """

    kind = "choice"

    choices: tuple[tuple[str, str], ...] = ()
    default: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_tuple(self.choices, f"ChoiceParameter {self.name!r} choices")
        # At least one choice, or the dropdown is empty.
        if len(self.choices) == 0:
            raise OperatorDescriptorError(
                f"ChoiceParameter {self.name!r}: needs at least one choice."
            )
        # Each choice must be a (value, label) pair. Checked before the
        # unpacking below so a malformed entry raises our own error type
        # rather than a bare ValueError from tuple unpacking.
        for entry in self.choices:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise OperatorDescriptorError(
                    f"ChoiceParameter {self.name!r}: each choice must be a "
                    f"(value, label) tuple, got {entry!r}."
                )
        # Choice values identify the selection, so they must be distinct.
        values = tuple(value for value, _label in self.choices)
        _require_unique(values, f"ChoiceParameter {self.name!r} choice values")
        # A stated default has to be one of the offered values.
        if self.default is not None and self.default not in values:
            raise OperatorDescriptorError(
                f"ChoiceParameter {self.name!r}: default {self.default!r} "
                f"is not one of the choice values {values!r}."
            )


@dataclass(frozen=True, kw_only=True)
class ColumnParameter(ParameterSpec):
    """A parameter whose value is the name of a column in one input table.

    ``from_input`` names an ``InputSpec`` declared by the SAME mode -- the
    table whose columns the dropdown is populated from. That cross-check
    is enforced by ``ModeDescriptor``, which is the object that can see
    both the parameter and the mode's inputs.
    """

    kind = "column"

    # Name of the InputSpec this column is chosen from.
    from_input: str = ""
    # Whether the researcher may pick more than one column.
    allow_multiple: bool = False
    # Column type tags the chosen column must carry; empty means "any".
    required_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_tuple(
            self.required_tags, f"ColumnParameter {self.name!r} required_tags"
        )


@dataclass(frozen=True, kw_only=True)
class NewTableNameParameter(ParameterSpec):
    """A parameter that names a table the operator is about to create."""

    kind = "new_table_name"

    default: str = ""


# ---------------------------------------------------------------------------
# Outputs.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class OutputColumn:
    """One column a mode adds to its rows.

    ``type_tag`` is the column type tag (``numeric``, ``media_path``, ...)
    the display layer keys on; it need not be registered, but an
    unregistered tag costs the researcher a placeholder tile.
    """

    name: str
    type_tag: str

    def __post_init__(self) -> None:
        _require_non_empty(self.name, "OutputColumn.name")
        _require_non_empty(self.type_tag, "OutputColumn.type_tag")


@dataclass(frozen=True, kw_only=True)
class OutputSpec:
    """What one mode produces. Exactly one of three shapes:

      * ``columns`` non-empty, the other two false -- the mode adds
        columns to existing rows;
      * ``creates_table`` true, ``columns`` empty, ``is_display_only``
        false -- the mode makes a new table;
      * ``is_display_only`` true, ``columns`` empty, ``creates_table``
        false -- the mode makes a Results-panel display and stores
        nothing.

    Any other combination is refused.
    """

    columns: tuple[OutputColumn, ...] = ()
    creates_table: bool = False
    is_display_only: bool = False

    def __post_init__(self) -> None:
        _require_tuple(self.columns, "OutputSpec.columns")
        has_columns = len(self.columns) > 0

        # Exactly one of the three shapes must be declared. Each shape is a
        # boolean; count how many are on and insist the count is one. This
        # rejects every wrong combination at once -- zero declared, or two,
        # or all three.
        shapes_declared = (has_columns, self.creates_table, self.is_display_only)
        if sum(shapes_declared) != 1:
            raise OperatorDescriptorError(
                "OutputSpec must declare exactly one shape: non-empty "
                "columns, or creates_table, or is_display_only -- with the "
                "other two fields false / empty. Got "
                f"columns={has_columns}, creates_table={self.creates_table}, "
                f"is_display_only={self.is_display_only}."
            )

        # Output column names form a namespace the schema layer keys on, so
        # they must be distinct -- the same discipline applied to input and
        # parameter names on ModeDescriptor.
        _require_unique(
            tuple(column.name for column in self.columns),
            "OutputSpec column names",
        )


# ---------------------------------------------------------------------------
# ModeDescriptor -- one execution mode of one operator.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class ModeDescriptor:
    """A single execution mode: its menu label, its inputs, its
    parameters, its output shape, and the run properties the cache needs.
    """

    mode: ExecutionMode
    # The Operators-menu label for this specific mode.
    label: str
    inputs: tuple[InputSpec, ...]
    # What the runner hands the operator per item it processes. Defaults to
    # METADATA so that a mode which forgets to declare gets no media and
    # fails visibly, rather than silently paying for a decode it never
    # asked for. There is deliberately NO cross-check between this and
    # ``mode``: a TABLE-mode operator that resolves addresses itself (video
    # frame extraction is the real case in this repo) legitimately declares
    # ADDRESS, so tying non-METADATA to COLUMNS mode would be wrong.
    media_requirement: MediaRequirement = MediaRequirement.METADATA
    parameters: tuple[ParameterSpec, ...] = ()
    output: OutputSpec
    model_lifecycle: ModelLifecycle = ModelLifecycle.NONE
    # Whether the same inputs and parameters always give the same output.
    deterministic: bool = True
    # Whether the runner may serve a cached result instead of re-running.
    cacheable: bool = True

    def __post_init__(self) -> None:
        _require_non_empty(self.label, "ModeDescriptor.label")
        _require_tuple(self.inputs, "ModeDescriptor.inputs")
        _require_tuple(self.parameters, "ModeDescriptor.parameters")

        # Input names and parameter names each form a namespace; a
        # collision inside either would make a run-time lookup ambiguous.
        input_names = tuple(spec.name for spec in self.inputs)
        _require_unique(input_names, "ModeDescriptor input names")
        parameter_names = tuple(spec.name for spec in self.parameters)
        _require_unique(parameter_names, "ModeDescriptor parameter names")

        # Every ColumnParameter must draw its columns from one of THIS
        # mode's inputs -- a parameter pointing at an input that does not
        # exist here could never be populated.
        input_name_set = set(input_names)
        for spec in self.parameters:
            if isinstance(spec, ColumnParameter):
                if spec.from_input not in input_name_set:
                    raise OperatorDescriptorError(
                        f"ColumnParameter {spec.name!r}: from_input "
                        f"{spec.from_input!r} does not name one of this "
                        f"mode's inputs {sorted(input_name_set)}."
                    )

        # The output shape must match the kind of mode this is.
        if self.mode is ExecutionMode.COLUMNS and len(self.output.columns) == 0:
            raise OperatorDescriptorError(
                "A COLUMNS mode must declare a non-empty output.columns."
            )
        if self.mode is ExecutionMode.TABLE and not self.output.creates_table:
            raise OperatorDescriptorError(
                "A TABLE mode must declare output.creates_table = True."
            )
        if self.mode is ExecutionMode.DISPLAY and not self.output.is_display_only:
            raise OperatorDescriptorError(
                "A DISPLAY mode must declare output.is_display_only = True."
            )


# ---------------------------------------------------------------------------
# OperatorDescriptor -- the operator as a whole.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class OperatorDescriptor:
    """The complete self-description of one operator.

    ``name`` must equal the operator class's ``name`` attribute (the
    consumer in P1.12c enforces that pairing; here we only insist the
    name is a valid identifier). ``version`` is part of the result-cache
    key. ``modes`` lists every execution mode the operator offers, each
    ``ExecutionMode`` at most once.
    """

    name: str
    version: str
    description: str
    modes: tuple[ModeDescriptor, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.name, "OperatorDescriptor.name")
        _require_non_empty(self.version, "OperatorDescriptor.version")
        _require_non_empty(self.description, "OperatorDescriptor.description")
        _require_tuple(self.modes, "OperatorDescriptor.modes")

        # An operator with no modes offers nothing.
        if len(self.modes) == 0:
            raise OperatorDescriptorError(
                "OperatorDescriptor must declare at least one mode."
            )

        # Each ExecutionMode may appear at most once -- two descriptors
        # for the same mode would make the menu entry ambiguous.
        seen_modes: set[ExecutionMode] = set()
        for mode_descriptor in self.modes:
            if mode_descriptor.mode in seen_modes:
                raise OperatorDescriptorError(
                    f"ExecutionMode {mode_descriptor.mode.name} is declared "
                    "more than once."
                )
            seen_modes.add(mode_descriptor.mode)

    def mode_for(self, mode: ExecutionMode) -> Optional[ModeDescriptor]:
        """Return the descriptor for ``mode``, or ``None`` if this
        operator does not offer that mode.
        """
        for mode_descriptor in self.modes:
            if mode_descriptor.mode is mode:
                return mode_descriptor
        return None
