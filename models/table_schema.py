"""
models/table_schema.py

The table-schema value object and the pure functions that check and normalise a
DataFrame against one. Nothing in the application calls this yet -- P1.8a builds
the value object in isolation; P1.8b and P1.8c wire it into Dataset.

Authorities -- this module restates none of them, it points:
  * docs/architecture.md §4.2  -- lineage, the three roles, carry_to_children
  * docs/architecture.md §4.3  -- schemas, the dtype policy and its exceptions
  * docs/media_architecture.md §6.2, the P1.8 paragraph -- the item description

Two callers, two rules, deliberately kept in separate functions so the boundary
stays visible: an operator that creates a table declares every dtype and §4.3
then has Dataset validate it, never infer it -- check_frame / normalise_frame
serve that path; the import paths (CSV import, folder load, merge) have no
declaration at all, so §4.2's import defaults apply and infer_schema serves
those, and only those.

The dtype check is value-based, not pair-based. pandas builds columns as int64,
float64 and object by default while §4.3 declares narrow storage (int32, float32,
category), so the ordinary conversion on accept is a NARROWING, and whether a
narrowing loses data cannot be decided from the dtype pair alone -- int64 ->
int32 is exact until a value leaves the int32 range, float64 -> float32 is exact
for some values and not others. So check_frame converts the actual values,
converts them back, and compares.

The dtypes this module handles are bounded on purpose: plain numpy integer,
float, bool and object columns, plus pandas categorical. A datetime64 column, a
timezone-aware datetime, or a pandas nullable extension dtype (Int64, string,
...) is refused with a named rejection -- an honest "not supported yet" beats a
silent fall-through into a path nobody has checked. See _is_supported_dtype.

Qt-free. pandas and numpy are permitted here (this is the data layer). This
module imports nothing from models/dataset.py, controller.py, column_types/ or
ui/.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


class ColumnRole(enum.Enum):
    # The three roles are defined in docs/architecture.md §4.2.
    identifier = "identifier"
    index = "index"
    measurement = "measurement"


@dataclass(frozen=True)
class ColumnSpec:
    # type_tag is the tag ColumnTypeRegistry maps to a renderer ("media_address",
    # "numeric", "text", "boolean_flag", ...). This module does not know the
    # registry and does not check that a tag is registered anywhere.
    name: str
    type_tag: str
    dtype: str
    role: ColumnRole
    carry_to_children: bool


@dataclass(frozen=True)
class TableSchema:
    # An ordered tuple of ColumnSpec. Frozen; the queries below are read-only.
    columns: tuple[ColumnSpec, ...]

    def __post_init__(self) -> None:
        # Accept any iterable from the caller, store a tuple.
        object.__setattr__(self, "columns", tuple(self.columns))
        # Construction rejects a duplicate column name.
        seen: set[str] = set()
        for spec in self.columns:
            if spec.name in seen:
                raise ValueError(
                    f"duplicate column name {spec.name!r} in TableSchema"
                )
            seen.add(spec.name)

    def column_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.columns)

    def spec_for(self, name: str) -> ColumnSpec:
        for spec in self.columns:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def identifiers(self) -> tuple[ColumnSpec, ...]:
        return tuple(s for s in self.columns if s.role is ColumnRole.identifier)

    def indices(self) -> tuple[ColumnSpec, ...]:
        return tuple(s for s in self.columns if s.role is ColumnRole.index)

    def carried_columns(self) -> tuple[ColumnSpec, ...]:
        # docs/architecture.md §4.2: identifiers and indices are carried
        # unconditionally -- they are what reconnects the split pieces -- and
        # every other column is carried when its carry_to_children flag is true.
        carried: list[ColumnSpec] = []
        for spec in self.columns:
            if spec.role in (ColumnRole.identifier, ColumnRole.index):
                carried.append(spec)
            elif spec.carry_to_children:
                carried.append(spec)
        return tuple(carried)

    def columns_with_tag(self, tag: str) -> tuple[ColumnSpec, ...]:
        return tuple(s for s in self.columns if s.type_tag == tag)


@dataclass(frozen=True)
class Adjustment:
    # A conversion that was applied on accept but is recorded so P1.8b can put it
    # in the provenance log. `kind` is one of exactly two strings:
    #   "storage_policy" -- the frame arrived in pandas' default dtype for its
    #                       kind and the schema declares a narrower storage dtype;
    #                       §4.3's policy working as designed, on nearly every
    #                       table.
    #   "unexpected"     -- any other exact conversion. The one that matters: the
    #                       schema declares int64 and the frame arrived int32,
    #                       meaning whoever built the frame declared the wrong
    #                       width.
    # P1.8b's strict mode will raise on "unexpected" only, so this classification
    # is what makes an operator's own test catch a wrong dtype. Strict mode is
    # not built here -- see docs/media_architecture.md §6.2, the P1.8 paragraph.
    column: str
    arrived_as: str
    stored_as: str
    reason: str
    kind: str


@dataclass(frozen=True)
class SchemaCheck:
    # Two separate tuples on purpose: P1.8b treats a blocking rejection and a
    # recorded-but-applied adjustment differently, so they are never merged into
    # one "problems" list.
    rejections: tuple[str, ...]
    adjustments: tuple[Adjustment, ...]

    @property
    def ok(self) -> bool:
        return not self.rejections


# pandas' default dtype per kind, and the narrow storage dtypes §4.3 declares.
# A conversion from the first set to the second is "storage_policy"; every other
# exact conversion is "unexpected". "str"/"string" are here because pandas >= 3
# gives a bare text column a string dtype, not numpy object, by default.
_PANDAS_DEFAULT_DTYPES = frozenset(
    {"int64", "float64", "object", "bool", "str", "string"}
)
_NARROW_STORAGE_DTYPES = frozenset({"int32", "int16", "int8", "float32", "category"})


def _is_supported_dtype(dtype) -> bool:
    """
    True for the dtypes this module has actually reasoned about: numpy integer
    (signed and unsigned), float and bool; any object-kind dtype, which is numpy
    object and also the pandas string dtype that is the pandas >= 3 default for a
    text column; and pandas categorical. Everything else -- datetime64,
    timezone-aware datetime, timedelta, and the numeric pandas nullable
    extension dtypes (Int64, Float64, boolean) -- returns False so the caller
    refuses it outright rather than let it fall through a code path nobody has
    checked.
    """
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    kind = getattr(dtype, "kind", None)
    # Object kind covers numpy object and the pandas string extension dtype;
    # both behave the same for our round trip.
    if kind == "O":
        return True
    # Anything else must be a plain numpy dtype of a numeric or bool kind -- a
    # nullable extension dtype (Int64, ...) is not a numpy dtype and is refused.
    return isinstance(dtype, np.dtype) and kind in ("i", "u", "f", "b")


def _kind_of(dtype_name: str) -> str:
    """Coarse kind of a dtype: "bool", "number" or "text". Crossing kinds is
    always a rejection, whatever the values."""
    dt = pd.api.types.pandas_dtype(dtype_name)
    # bool is numeric to pandas, so it must be tested first.
    if pd.api.types.is_bool_dtype(dt):
        return "bool"
    if pd.api.types.is_numeric_dtype(dt):
        return "number"
    return "text"


def _classify(arrived: str, stored: str) -> str:
    """"storage_policy" if the frame arrived in a pandas default and the schema
    stores a narrower dtype; "unexpected" for every other exact conversion."""
    if arrived in _PANDAS_DEFAULT_DTYPES and stored in _NARROW_STORAGE_DTYPES:
        return "storage_policy"
    return "unexpected"


def _reason_for(kind: str, arrived: str, stored: str) -> str:
    if kind == "storage_policy":
        return (
            f"{arrived} is pandas' default dtype; the schema stores {stored} per "
            f"docs/architecture.md §4.3, and every value survived the round trip"
        )
    return (
        f"the frame declared {arrived} where the schema expects {stored}; the "
        f"values convert exactly but the declared width is not what was expected"
    )


def _round_trip_survivors(original: pd.Series, target_dtype: str):
    """
    Convert `original` to target_dtype, convert the result back to the arriving
    dtype, and compare element-wise. A NaN that stays NaN counts as equal.

    Returns (all_survived, first_bad_position, original_value, round_tripped).
    Vectorised with numpy -- no Python loop over rows; this runs on accept on
    frames that may hold hundreds of thousands of rows.
    """
    converted = original.astype(target_dtype)
    back = converted.astype(original.dtype)
    o = original.to_numpy()
    b = back.to_numpy()
    both_nan = pd.isna(o) & pd.isna(b)
    equal = (o == b) | both_nan
    bad = np.flatnonzero(~equal)
    if bad.size == 0:
        return True, None, None, None
    pos = int(bad[0])
    return False, pos, original.iloc[pos], back.iloc[pos]


def check_frame(df: pd.DataFrame, schema: TableSchema) -> SchemaCheck:
    """
    Pure. Mutates nothing. Returns a frozen SchemaCheck.

    A frame is rejected (must not be stored) when: a column the schema names is
    missing; a column the frame carries is not named by the schema; a column's
    frame dtype or declared dtype is outside the set this module handles (see
    _is_supported_dtype); a column's dtype crosses a kind (text/number/bool); or
    the actual values do not survive the conversion to the stored dtype and back.
    A frame is adjusted (stored, but recorded) when the dtypes differ yet every
    value survives the round trip.
    """
    rejections: list[str] = []
    adjustments: list[Adjustment] = []

    frame_columns = list(df.columns)
    schema_names = schema.column_names()

    # A column the schema names but the frame does not have.
    for name in schema_names:
        if name not in frame_columns:
            rejections.append(
                f"column {name!r} is named by the schema but missing from the frame"
            )

    # A column the frame carries but the schema does not name.
    for name in frame_columns:
        if name not in schema_names:
            rejections.append(
                f"column {name!r} is in the frame but not named by the schema"
            )

    # Dtype differences, only for the columns present on both sides.
    for name in schema_names:
        if name not in frame_columns:
            continue
        stored = schema.spec_for(name).dtype
        frame_dtype = df[name].dtype

        # Step 0: bound what this module claims to handle. An unsupported dtype
        # -- datetime64, timezone-aware datetime, a pandas nullable extension
        # dtype -- is an honest rejection, never a silent fall-through into the
        # round-trip path.
        if not _is_supported_dtype(frame_dtype):
            rejections.append(
                f"column {name!r}: dtype {str(frame_dtype)!r} is not supported "
                f"by TableSchema yet"
            )
            continue
        try:
            stored_dtype = pd.api.types.pandas_dtype(stored)
        except TypeError:
            rejections.append(
                f"column {name!r}: the schema's dtype {stored!r} is not a "
                f"recognised dtype"
            )
            continue
        if not _is_supported_dtype(stored_dtype):
            rejections.append(
                f"column {name!r}: the schema declares dtype {stored!r}, which "
                f"is not supported by TableSchema yet"
            )
            continue

        arrived = str(frame_dtype)
        if arrived == stored:
            continue

        # Step 1: crossing kinds is always a rejection; values are never
        # inspected and no parse is attempted.
        if _kind_of(arrived) != _kind_of(stored):
            rejections.append(
                f"column {name!r}: the frame's {arrived!r} is a "
                f"{_kind_of(arrived)} and the schema's {stored!r} is a "
                f"{_kind_of(stored)}; crossing text/number/bool is always "
                f"rejected and no conversion is attempted"
            )
            continue

        # Step 2: convert the actual values, convert them back, compare.
        try:
            survived, pos, original_value, round_tripped = _round_trip_survivors(
                df[name], stored
            )
        except (TypeError, ValueError) as exc:
            rejections.append(
                f"column {name!r}: converting {arrived!r} -> {stored!r} failed: "
                f"{exc}"
            )
            continue

        if not survived:
            rejections.append(
                f"column {name!r}: converting {arrived!r} -> {stored!r} changes "
                f"value {original_value!r} at row position {pos} (it returns as "
                f"{round_tripped!r}); the conversion is not exact for this frame"
            )
            continue

        # An exact conversion -- record it, classified.
        kind = _classify(arrived, stored)
        adjustments.append(
            Adjustment(
                column=name,
                arrived_as=arrived,
                stored_as=stored,
                reason=_reason_for(kind, arrived, stored),
                kind=kind,
            )
        )

    return SchemaCheck(
        rejections=tuple(rejections),
        adjustments=tuple(adjustments),
    )


def normalise_frame(
    df: pd.DataFrame,
    schema: TableSchema,
    *,
    check: SchemaCheck | None = None,
) -> pd.DataFrame:
    """
    Applies the recorded adjustments and returns a NEW frame. Does not mutate the
    frame it was given.

    The caller checks first: check_frame is the gate, and normalise_frame must
    never be handed a frame that check_frame rejects. If it is, it raises rather
    than half-applying.

    `check` lets a caller that has already run check_frame(df, schema) hand the
    result back so it is not recomputed -- the round trip is not cheap on a
    large frame. When omitted the behaviour is exactly as before: check_frame
    runs here.
    """
    if check is None:
        check = check_frame(df, schema)
    if check.rejections:
        raise ValueError(
            "normalise_frame was called on a frame with rejections: "
            + "; ".join(check.rejections)
        )
    # Build a new frame and convert on the copy; the caller's frame is untouched.
    out = df.copy()
    for adjustment in check.adjustments:
        out[adjustment.column] = out[adjustment.column].astype(adjustment.stored_as)
    return out


@dataclass(frozen=True)
class ColumnHint:
    # Every field optional. infer_schema computes its own default spec for a
    # column, then applies whichever of these fields the caller set on top. This
    # is how an import path says "participant_id is an identifier",
    # "frame_index must stay int64", or "clip_path is a media_address".
    role: ColumnRole | None = None
    type_tag: str | None = None
    dtype: str | None = None
    carry_to_children: bool | None = None


def infer_schema(
    df: pd.DataFrame,
    *,
    hints: Mapping[str, ColumnHint] | None = None,
) -> TableSchema:
    """
    Import-path only. This exists for CSV import, folder load and merge, where
    nobody declared anything. docs/architecture.md §4.3 says dtypes are never
    inferred for a table an operator creates; §4.2 gives the import defaults --
    both are true, for different callers, which is why inference lives in its own
    function.

    The rule, one sentence for every kind: inference keeps the dtype the column
    arrived in (int64, float64, bool, the object/string text dtype, or an
    already-categorical dtype), makes every column a `measurement` with
    `carry_to_children` true, and never narrows -- §4.3's narrow defaults
    (float32, int32, category) govern the dtypes an operator DECLARES when it
    creates a table, not what Gelem infers, and a caller that wants one of them
    passes ColumnHint(dtype=...).

    A dtype outside the supported set (see _is_supported_dtype) -- datetime64, a
    timezone-aware datetime, a pandas nullable extension dtype -- raises rather
    than being guessed at.

    `hints` maps a column name to a ColumnHint applied over that column's
    inferred spec. A hint naming a column not in the frame is an error. The
    import paths in P1.8b supply this -- for identifiers, for media-address type
    tags, and for the §4.3 exception where presentation timestamps, sample
    positions and frame ordinals must stay 64-bit.
    """
    # None means "no hints"; a mutable default in the signature is a footgun.
    if hints is None:
        hints = {}

    # A hint that names no real column is a mistake the caller should hear about.
    for hinted_name in hints:
        if hinted_name not in df.columns:
            raise ValueError(
                f"hint names column {hinted_name!r}, which is not in the frame"
            )

    specs: list[ColumnSpec] = []

    for name in df.columns:
        series = df[name]
        dt = series.dtype
        actual = str(dt)

        # Refuse a dtype this module has not reasoned about, rather than guess.
        if not _is_supported_dtype(dt):
            raise ValueError(
                f"column {name!r} has dtype {actual!r}, which infer_schema does "
                f"not support yet"
            )

        # Inference keeps the dtype the column arrived in and never narrows.
        # After the guard above, dt is a plain numpy integer/float/bool dtype,
        # an object-kind text dtype, or a pandas categorical.
        if isinstance(dt, pd.CategoricalDtype):
            # Already categorical on arrival: that IS its dtype, so keep it.
            role, dtype, tag = ColumnRole.measurement, "category", "text"
        elif dt.kind == "b":
            role, dtype, tag = ColumnRole.measurement, "bool", "boolean_flag"
        elif dt.kind in ("i", "u"):
            role, dtype, tag = ColumnRole.measurement, "int64", "numeric"
        elif dt.kind == "f":
            role, dtype, tag = ColumnRole.measurement, "float64", "numeric"
        else:
            # Every text column -- whether its values repeat or not -- keeps the
            # dtype it arrived in (numpy "object" or the pandas string dtype).
            # A repeated-value column is left open, not turned into a category,
            # because Gelem cannot tell a closed vocabulary from a column a
            # researcher will keep adding labels to, and a closed dtype refuses
            # a new label at write time.
            role, dtype, tag = ColumnRole.measurement, actual, "text"
        carry = True

        # Apply the caller's hint over the inferred spec.
        hint = hints.get(name)
        if hint is not None:
            if hint.role is not None:
                role = hint.role
            if hint.type_tag is not None:
                tag = hint.type_tag
            if hint.dtype is not None:
                dtype = hint.dtype
            if hint.carry_to_children is not None:
                carry = hint.carry_to_children

        specs.append(
            ColumnSpec(
                name=str(name),
                type_tag=tag,
                dtype=dtype,
                role=role,
                carry_to_children=carry,
            )
        )

    return TableSchema(columns=tuple(specs))
