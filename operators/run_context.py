"""
operators/run_context.py -- the operator run object and the immutable spec
behind it.

WHAT THIS MODULE IS (for a reader who has never seen this design)

Everything an operator needs for one run will eventually arrive through a
single ``run`` argument -- its parameters, the frozen table snapshots it may
read, the project directories it may write to, a way to check for
cancellation, and the result sink for per-row output. This module builds
that ``run`` object (``OperatorRun``) and the immutable description of the
run that sits inside it (``OperatorRunSpec``).

This is the CONTRACT half of a contract/consumer seam. It has NO consumers
yet: no operator signature changes, no runner change, no controller change.
P1.12d wires it in. Nothing else in the repo imports this module yet, and
that is deliberate.

DESIGN CONSTRAINTS

  * No Qt, and no import from ``models/``, ``controller.py`` or ``ui/``.
  * pandas is allowed but is not needed here: this module stores whatever
    frames it is handed and never builds or copies one.
  * Frozen dataclasses throughout; every collection field is a tuple or a
    read-only mapping, never a list.
  * Importing this module must not drag PyYAML in. The only name this
    module needs from ``operators.operator_config`` is ``OperatorRuntimeDirs``,
    used purely as a type annotation, so it is imported under
    ``TYPE_CHECKING`` (see the note at that import).
"""

from __future__ import annotations

# Standard library only at run time.
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Optional, TYPE_CHECKING

# The descriptor vocabulary is Qt-free and pandas-free, so importing it here
# is cheap. We import it, we never edit it.
from operators.descriptor import (
    BooleanParameter,
    ChoiceParameter,
    ExecutionMode,
    ModeDescriptor,
    NumberParameter,
)

if TYPE_CHECKING:
    # TYPE_CHECKING is a constant that is False at run time and True only
    # while a static type checker (mypy, pyright) is analysing this file.
    # Importing OperatorRuntimeDirs here therefore gives us the name for
    # annotations WITHOUT importing operators.operator_config at run time --
    # and that module imports PyYAML at module scope, which we do not want
    # to pull into every process that merely touches a run object.
    from operators.operator_config import OperatorRuntimeDirs


# ---------------------------------------------------------------------------
# The one failure type this module raises.
# ---------------------------------------------------------------------------
class OperatorRunError(Exception):
    """Raised when a run object or run spec is constructed with values that
    cannot form a well-formed run -- an empty operation id, a parameter the
    descriptor does not declare, a value outside its declared bounds, two
    snapshots of the same table taken at different versions, and so on.

    It is a plain ``Exception`` subclass on purpose: a malformed run is a
    programming error in the runner, not something an operator recovers
    from.
    """


# ---------------------------------------------------------------------------
# CancellationToken -- the flag that stops a long run.
#
# The token is held by whoever STARTS the run (the runner / controller).
# The operator never sees the token itself: it only ever calls
# run.cancelled(). This keeps the "set" capability on the starting side and
# the "check" capability on the operator side.
# ---------------------------------------------------------------------------
class CancellationToken:
    """A one-way flag: once cancelled, it stays cancelled.

    It wraps ``threading.Event`` rather than a plain ``bool`` attribute
    because the flag is SET on the main thread (when the researcher clicks
    Cancel) and READ on a worker thread (inside the operator loop). A plain
    bool would be a data race across those two threads; ``Event.set()`` and
    ``Event.is_set()`` are documented as safe to call from any thread.
    """

    def __init__(self) -> None:
        # threading.Event is a simple thread-safe flag with set() / clear()
        # / is_set(). We never clear it -- cancellation is one-way.
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation. Idempotent."""
        self._event.set()

    def is_cancelled(self) -> bool:
        """True once cancel() has been called."""
        return self._event.is_set()


# ---------------------------------------------------------------------------
# TableSnapshot -- one frozen table the operator may read.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TableSnapshot:
    """A single table, frozen at one moment, plus its write-ticket version.

    ``frame`` is whatever the caller hands in -- in practice the frozen copy
    ``Dataset`` produced. This class does not copy it and does not inspect
    it; a downstream ``[NOW]`` rule already forbids an operator from
    mutating a frame it was given.

    ``version`` is the per-table write-ticket version (P1.12b): it changes
    on every accepted commit to that table, so two snapshots of the same
    table with the same version are the same data.
    """

    # The stored table's name.
    table_name: str
    # The frame object handed in. Stored as-is, never copied.
    frame: object
    # The per-table write-ticket version at the moment of the snapshot.
    version: int

    def __post_init__(self) -> None:
        # A snapshot with no table name could not be keyed or reported.
        if not isinstance(self.table_name, str) or self.table_name.strip() == "":
            raise OperatorRunError(
                "TableSnapshot.table_name must be a non-empty string."
            )
        # Write-ticket versions start at 1 and only increase; 0 or negative
        # means the caller never actually read a version.
        if not isinstance(self.version, int) or isinstance(self.version, bool):
            raise OperatorRunError(
                f"TableSnapshot.version must be an int, got "
                f"{type(self.version).__name__}."
            )
        if self.version < 1:
            raise OperatorRunError(
                f"TableSnapshot.version must be >= 1, got {self.version}."
            )


# ---------------------------------------------------------------------------
# RunData -- the frozen snapshots an operator may read.
#
# Keyed by the INPUT NAME the operator declared in its descriptor, NOT by
# table name. One declared input is either a single table or a whole-project
# read; the two live in separate mappings so the accessor for each can
# refuse the wrong call with a message that points at the right one.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, init=False)
class RunData:
    """The tables an operator may read for one run, addressed by declared
    input name.

    Construct with two keyword arguments:

      * ``tables``   -- Mapping[input_name, TableSnapshot] for single-table
                        inputs;
      * ``projects`` -- Mapping[input_name, Mapping[table_name, TableSnapshot]]
                        for whole-project inputs.

    Both are stored read-only (``types.MappingProxyType``), so a worker
    cannot add, drop or re-point an entry.

    DECIDED, do not change: ``table(input_name)`` returns the SAME frame
    object on every call -- it does NOT copy per call. A per-call copy of a
    530,000-row frame is a hidden cost paid on every access, and the repo
    already carries the ``[NOW]`` rule that an operator must never modify a
    DataFrame it was given, so the copy would buy nothing. An identity test
    pins this.
    """

    # Declared as fields so the generated __repr__ / __eq__ see them; set
    # by hand in __init__ below because this is a frozen dataclass with a
    # custom constructor.
    _tables: "Mapping[str, TableSnapshot]" = field(default_factory=dict)
    _projects: "Mapping[str, Mapping[str, TableSnapshot]]" = field(default_factory=dict)

    def __init__(
        self,
        *,
        tables: "Mapping[str, TableSnapshot]",
        projects: "Mapping[str, Mapping[str, TableSnapshot]]",
    ) -> None:
        # --- Validation: every value must actually be a TableSnapshot (and
        # every project value a mapping of them). Without this, a wrong
        # value type would surface later as a bare AttributeError from
        # _iter_all_snapshots or from an accessor, escaping this module's
        # "OperatorRunError is the one failure type" contract. ---
        for input_name, snapshot in tables.items():
            if not isinstance(snapshot, TableSnapshot):
                raise OperatorRunError(
                    f"tables[{input_name!r}] must be a TableSnapshot, got "
                    f"{type(snapshot).__name__}."
                )
        for input_name, inner in projects.items():
            if not isinstance(inner, Mapping):
                raise OperatorRunError(
                    f"projects[{input_name!r}] must be a mapping of "
                    f"table_name -> TableSnapshot, got {type(inner).__name__}."
                )
            for table_name, snapshot in inner.items():
                if not isinstance(snapshot, TableSnapshot):
                    raise OperatorRunError(
                        f"projects[{input_name!r}][{table_name!r}] must be a "
                        f"TableSnapshot, got {type(snapshot).__name__}."
                    )

        # --- Validation: an input name may not name both kinds of input. --
        overlap = set(tables) & set(projects)
        if overlap:
            raise OperatorRunError(
                "input name(s) "
                f"{sorted(overlap)!r} appear in both tables and projects; "
                "an input is either a single table or a whole-project read, "
                "not both."
            )

        # --- Validation: one table reached through two inputs must be the
        # same snapshot version. Disagreeing versions mean the snapshots
        # were taken at different moments -- exactly the bug this object
        # exists to prevent. ---
        version_by_table: dict[str, int] = {}
        for snapshot in _iter_all_snapshots(tables, projects):
            known = version_by_table.get(snapshot.table_name)
            if known is not None and known != snapshot.version:
                raise OperatorRunError(
                    f"table {snapshot.table_name!r} is present at two "
                    f"different versions ({known} and {snapshot.version}); "
                    "the snapshots were taken at different moments."
                )
            version_by_table[snapshot.table_name] = snapshot.version

        # --- Store read-only. A frozen dataclass forbids ordinary
        # assignment, so we go through object.__setattr__, the documented
        # escape hatch for exactly this case. The inner project mappings are
        # wrapped too, so no level is mutable. ---
        object.__setattr__(self, "_tables", MappingProxyType(dict(tables)))
        object.__setattr__(
            self,
            "_projects",
            MappingProxyType(
                {
                    name: MappingProxyType(dict(inner))
                    for name, inner in projects.items()
                }
            ),
        )

    # -- Accessors -------------------------------------------------------

    def table(self, input_name: str) -> object:
        """The frame for a single-table input.

        Returns the SAME object every call (see the class docstring).
        Raises ``OperatorRunError`` if the name is unknown, or if it names a
        whole-project input -- in which case the message tells the caller to
        use ``tables()`` instead.
        """
        if input_name in self._projects:
            raise OperatorRunError(
                f"input {input_name!r} is a whole-project input; call "
                "tables() to get its {table_name: frame} mapping."
            )
        if input_name not in self._tables:
            raise OperatorRunError(
                f"unknown input name {input_name!r}; declared inputs are "
                f"{list(self.input_names())}."
            )
        return self._tables[input_name].frame

    def tables(self, input_name: str) -> dict:
        """The ``{table_name: frame}`` mapping for a whole-project input.

        Mirror image of ``table()``: raises ``OperatorRunError`` if the name
        is unknown, or if it names a single-table input -- in which case the
        message names ``table()``.
        """
        if input_name in self._tables:
            raise OperatorRunError(
                f"input {input_name!r} is a single-table input; call "
                "table() to get its frame."
            )
        if input_name not in self._projects:
            raise OperatorRunError(
                f"unknown input name {input_name!r}; declared inputs are "
                f"{list(self.input_names())}."
            )
        # Each frame is the same object it was handed in with; only this
        # outer dict is freshly built.
        return {
            table_name: snapshot.frame
            for table_name, snapshot in self._projects[input_name].items()
        }

    def snapshot(self, input_name: str) -> TableSnapshot:
        """The full ``TableSnapshot`` for a single-table input, for a caller
        that also wants the table name and version.

        Raises ``OperatorRunError`` for an unknown name, and for a
        whole-project input (``snapshot()`` returns one snapshot and is for
        single-table inputs only).
        """
        if input_name in self._projects:
            raise OperatorRunError(
                f"input {input_name!r} is a whole-project input; snapshot() "
                "returns a single TableSnapshot and is for single-table "
                "inputs only."
            )
        if input_name not in self._tables:
            raise OperatorRunError(
                f"unknown input name {input_name!r}; declared inputs are "
                f"{list(self.input_names())}."
            )
        return self._tables[input_name]

    def versions(self) -> dict:
        """``{table_name: version}`` flattened across every input.

        A table reached through two inputs appears once; the constructor has
        already guaranteed the two versions agree.
        """
        flattened: dict[str, int] = {}
        for snapshot in _iter_all_snapshots(self._tables, self._projects):
            flattened[snapshot.table_name] = snapshot.version
        return flattened

    def input_names(self) -> tuple:
        """Every declared input name, single-table inputs first, then
        whole-project inputs, each group in insertion order.
        """
        return tuple(self._tables.keys()) + tuple(self._projects.keys())


def _iter_all_snapshots(
    tables: "Mapping[str, TableSnapshot]",
    projects: "Mapping[str, Mapping[str, TableSnapshot]]",
):
    """Yield every ``TableSnapshot`` reachable through either mapping.

    Module-private helper shared by ``RunData.__init__`` (for the
    version-agreement check) and ``RunData.versions()``.
    """
    for snapshot in tables.values():
        yield snapshot
    for inner in projects.values():
        for snapshot in inner.values():
            yield snapshot


# ---------------------------------------------------------------------------
# OperatorRunSpec -- everything about one run that cannot change while it
# runs.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperatorRunSpec:
    """The immutable description of one run.

    ``parameters`` is stored as ``types.MappingProxyType`` so a worker
    cannot mutate the values mid-run. Row ids are deliberately NOT here: a
    run can cover hundreds of thousands of rows and they travel separately,
    as they do today.
    """

    # Unique id for this run, echoed back through every result payload.
    operation_id: str
    # The operator's ``name`` class attribute.
    operator_name: str
    # Which execution mode this run is.
    mode: ExecutionMode
    # The descriptor for that mode -- the authority on inputs, parameters
    # and output shape.
    mode_descriptor: ModeDescriptor
    # This run's parameter values, keyed by declared parameter name.
    parameters: "Mapping[str, object]"
    # For COLUMNS mode: the table new columns are written into. Empty for
    # TABLE and DISPLAY modes, which store nothing into an existing table.
    target_table: str

    def __post_init__(self) -> None:
        # -- Identity fields must be real, non-empty strings. --
        if not isinstance(self.operation_id, str) or self.operation_id.strip() == "":
            raise OperatorRunError(
                "OperatorRunSpec.operation_id must be a non-empty string."
            )
        if not isinstance(self.operator_name, str) or self.operator_name.strip() == "":
            raise OperatorRunError(
                "OperatorRunSpec.operator_name must be a non-empty string."
            )

        # -- The mode must be the one the descriptor describes. --
        if self.mode is not self.mode_descriptor.mode:
            raise OperatorRunError(
                f"OperatorRunSpec.mode ({self.mode.name}) does not match "
                f"mode_descriptor.mode ({self.mode_descriptor.mode.name})."
            )

        # -- Parameter checks against what the descriptor declares. --
        declared = {spec.name: spec for spec in self.mode_descriptor.parameters}

        # Every required parameter must be supplied.
        for name, spec in declared.items():
            if spec.required and name not in self.parameters:
                raise OperatorRunError(
                    f"required parameter {name!r} is missing from this run."
                )

        # No parameter the descriptor does not declare.
        for name in self.parameters:
            if name not in declared:
                raise OperatorRunError(
                    f"parameter {name!r} is not declared by mode "
                    f"{self.mode.name} of {self.operator_name!r}."
                )

        # Per-type value checks, for whichever declared parameters are
        # actually present in this run.
        for name, value in self.parameters.items():
            spec = declared[name]
            if isinstance(spec, ChoiceParameter):
                allowed = tuple(choice_value for choice_value, _ in spec.choices)
                if value not in allowed:
                    raise OperatorRunError(
                        f"parameter {name!r}: value {value!r} is not one of "
                        f"the declared choice values {allowed!r}."
                    )
            elif isinstance(spec, NumberParameter):
                # A non-numeric value would make the bound comparisons below
                # raise a bare TypeError, which would escape this module's
                # "OperatorRunError is the one failure type" contract. Catch
                # it here as our own error. bool is a subclass of int and is
                # left to pass -- NumberParameter does not forbid it.
                if not isinstance(value, (int, float)):
                    raise OperatorRunError(
                        f"parameter {name!r}: value {value!r} is not a "
                        f"number ({type(value).__name__})."
                    )
                if spec.minimum is not None and value < spec.minimum:
                    raise OperatorRunError(
                        f"parameter {name!r}: value {value!r} is below the "
                        f"declared minimum ({spec.minimum})."
                    )
                if spec.maximum is not None and value > spec.maximum:
                    raise OperatorRunError(
                        f"parameter {name!r}: value {value!r} is above the "
                        f"declared maximum ({spec.maximum})."
                    )
            elif isinstance(spec, BooleanParameter):
                # bool is a subclass of int, so an explicit type check is
                # the only way to reject 0 / 1 / "yes" here.
                if not isinstance(value, bool):
                    raise OperatorRunError(
                        f"parameter {name!r}: value {value!r} is not a bool."
                    )

        # -- target_table must match the mode. COLUMNS writes into an
        # existing table and needs its name; TABLE and DISPLAY store nothing
        # into one, so a non-empty name there is a mistake. --
        if self.mode is ExecutionMode.COLUMNS:
            if not isinstance(self.target_table, str) or self.target_table.strip() == "":
                raise OperatorRunError(
                    "COLUMNS mode requires a non-empty target_table."
                )
        else:
            if self.target_table:
                raise OperatorRunError(
                    f"{self.mode.name} mode requires target_table to be "
                    f"empty, got {self.target_table!r}."
                )

        # -- Freeze the parameter mapping. A frozen dataclass forbids
        # ordinary assignment in __post_init__, so object.__setattr__ is the
        # documented way to replace the field with a read-only view. --
        object.__setattr__(
            self, "parameters", MappingProxyType(dict(self.parameters))
        )


# ---------------------------------------------------------------------------
# OperatorRun -- what an operator actually receives as its ``run`` argument.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperatorRun:
    """The single object through which parameters and runtime services reach
    an operator for one run.

    The operator reads ``run.parameters``, checks ``run.cancelled()`` between
    units of work, writes files under ``run.paths``, and (in COLUMNS mode)
    pushes per-row results through ``run.emit()``.
    """

    # The immutable description of this run.
    spec: OperatorRunSpec
    # The frozen table snapshots this run may read.
    data: RunData
    # This project's directories. Never store these on the operator.
    paths: "OperatorRuntimeDirs"
    # The cancellation flag, held by the runner. Underscore-prefixed
    # because an operator must go through cancelled(), never touch this.
    _token: CancellationToken
    # The per-row result sink, injected by the runner. None until wired.
    _emit_fn: Optional[Callable[..., None]] = None

    @property
    def parameters(self) -> "Mapping[str, object]":
        """This run's parameter values (read-only)."""
        return self.spec.parameters

    def cancelled(self) -> bool:
        """True once the run has been cancelled. Check between units of
        work and return promptly when it is true.
        """
        return self._token.is_cancelled()

    def emit(self, row_id: str, values: dict) -> None:
        """Push one row's new column values to the result sink.

        Forwards to the runner-supplied sink with exactly the signature the
        existing per-row channel already uses::

            _emit_fn(operation_id, target_table, row_id, values)

        ``operation_id`` and ``target_table`` are taken from ``spec``, and a
        COPY of ``values`` is passed so a later in-place edit by the
        operator cannot reach back into a result already handed off.

        Raises ``OperatorRunError`` if there is no sink wired -- a dropped
        result must never be silent -- and raises if the mode is not
        COLUMNS: there is no incremental channel for a TABLE or DISPLAY run
        today, and a cancelled run of those simply has its return value
        discarded.
        """
        # Mode is checked first: a TABLE or DISPLAY run legitimately has no
        # sink, so the accurate diagnostic there is "wrong mode", not
        # "missing sink".
        if self.spec.mode is not ExecutionMode.COLUMNS:
            raise OperatorRunError(
                f"OperatorRun.emit() is only for COLUMNS mode; this run is "
                f"{self.spec.mode.name} mode, which has no incremental "
                "channel."
            )
        if self._emit_fn is None:
            raise OperatorRunError(
                "OperatorRun.emit() called with no result sink wired; a "
                "dropped result must not be silent."
            )
        self._emit_fn(
            self.spec.operation_id,
            self.spec.target_table,
            row_id,
            dict(values),
        )
