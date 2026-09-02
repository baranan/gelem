"""
tests/test_table_schema.py

Pure tests for models/table_schema.py -- no QApplication, no Dataset, no
controller. Written from docs/architecture.md §4.2 / §4.3 and the P1.8a spec
(as corrected), not from the implementation.
"""

import numpy as np
import pandas as pd
import pytest

from models.table_schema import (
    Adjustment,
    ColumnHint,
    ColumnRole,
    ColumnSpec,
    TableSchema,
    check_frame,
    infer_schema,
    normalise_frame,
)


def _spec(name, role, carry, dtype="int64", tag="numeric"):
    return ColumnSpec(
        name=name,
        type_tag=tag,
        dtype=dtype,
        role=role,
        carry_to_children=carry,
    )


def _schema_for_col(dtype):
    return TableSchema(
        columns=(_spec("col", ColumnRole.measurement, carry=True, dtype=dtype),)
    )


# T1 -- identifiers and indices are carried even with carry_to_children False,
# and a measurement whose flag is True is carried. reaction_time is the named
# case: §4.2's correction note exists because an earlier design dropped it.
def test_t1_carried_columns_covers_identifiers_indices_and_flagged_measurements():
    schema = TableSchema(
        columns=(
            _spec("participant_id", ColumnRole.identifier, carry=False),
            _spec("frame_index", ColumnRole.index, carry=False),
            _spec("reaction_time", ColumnRole.measurement, carry=True),
        )
    )
    carried = {s.name for s in schema.carried_columns()}
    assert carried == {"participant_id", "frame_index", "reaction_time"}


# T2 -- a measurement with carry_to_children False is not carried.
def test_t2_unflagged_measurement_is_not_carried():
    schema = TableSchema(
        columns=(
            _spec("participant_id", ColumnRole.identifier, carry=True),
            _spec("blendshape_score", ColumnRole.measurement, carry=False),
        )
    )
    carried = {s.name for s in schema.carried_columns()}
    assert "blendshape_score" not in carried
    assert carried == {"participant_id"}


# --- dtype check, value-based (replaces the old pair-table tests) ------------


# int64 of small values against an int32 schema: every value survives the round
# trip, so this is a conversion, classified storage_policy (§4.3's narrow
# storage working as designed), with no rejection.
def test_int64_small_values_to_int32_is_storage_policy():
    df = pd.DataFrame({"col": pd.Series([1, 2, 3], dtype="int64")})
    check = check_frame(df, _schema_for_col("int32"))

    assert check.rejections == ()
    assert len(check.adjustments) == 1
    adj = check.adjustments[0]
    assert isinstance(adj, Adjustment)
    assert adj.arrived_as == "int64"
    assert adj.stored_as == "int32"
    assert adj.kind == "storage_policy"


# int64 carrying a value above the int32 range against an int32 schema: the
# large value does not survive, so this REJECTS. The offending value sits at row
# position 2, not row 0 -- the negative check depends on that.
def test_int64_above_int32_range_rejects_and_names_the_value():
    df = pd.DataFrame(
        {"trial_count": pd.Series([1, 2, 3_000_000_000], dtype="int64")}
    )
    schema = TableSchema(
        columns=(
            _spec("trial_count", ColumnRole.measurement, carry=True, dtype="int32"),
        )
    )
    check = check_frame(df, schema)

    assert check.adjustments == ()
    assert len(check.rejections) == 1
    message = check.rejections[0]
    assert "trial_count" in message
    assert "3000000000" in message
    assert "int64" in message and "int32" in message


# float64 whose values are all exactly representable in float32: conversion,
# storage_policy, no rejection.
def test_float64_surviving_float32_is_storage_policy():
    df = pd.DataFrame(
        {"col": pd.Series([0.5, 0.25, 0.125], dtype="float64")}
    )
    check = check_frame(df, _schema_for_col("float32"))

    assert check.rejections == ()
    assert len(check.adjustments) == 1
    assert check.adjustments[0].kind == "storage_policy"


# float64 carrying a value that changes under float32: REJECTS, and the message
# names the offending value. The bad value is at row 0 on purpose -- the
# negative check needs this test to keep passing when only row 0 is compared.
def test_float64_not_surviving_float32_rejects_and_names_the_value():
    df = pd.DataFrame(
        {"score": pd.Series([0.1, 0.5, 0.25], dtype="float64")}
    )
    schema = TableSchema(
        columns=(
            _spec("score", ColumnRole.measurement, carry=True, dtype="float32"),
        )
    )
    check = check_frame(df, schema)

    assert check.adjustments == ()
    assert len(check.rejections) == 1
    message = check.rejections[0]
    assert "score" in message
    assert "0.1" in message
    assert "row position 0" in message


# int64 with 2**53 + 1 against a float64 schema: exact for small values, lossy
# here -- this is the real version of the old "int64 -> float64" rule, which a
# dtype pair alone could never decide. Bad value at row 0.
def test_int64_2pow53_plus_1_to_float64_rejects():
    df = pd.DataFrame(
        {"sample_pos": pd.Series([2 ** 53 + 1, 1, 2], dtype="int64")}
    )
    schema = TableSchema(
        columns=(
            _spec("sample_pos", ColumnRole.measurement, carry=True, dtype="float64"),
        )
    )
    check = check_frame(df, schema)

    assert check.adjustments == ()
    assert len(check.rejections) == 1
    message = check.rejections[0]
    assert "sample_pos" in message
    assert str(2 ** 53 + 1) in message


# int32 frame against an int64 schema: values survive, but the frame declared a
# narrower width than the schema expects -- this is the "unexpected" case that
# P1.8b strict mode will raise on. Assert on the kind, not just on the absence
# of a rejection.
def test_int32_frame_against_int64_schema_is_unexpected():
    df = pd.DataFrame({"col": pd.Series([1, 2, 3], dtype="int32")})
    check = check_frame(df, _schema_for_col("int64"))

    assert check.rejections == ()
    assert len(check.adjustments) == 1
    assert check.adjustments[0].kind == "unexpected"


# A text column against a numeric schema: rejected at the kind gate, with no
# parse attempted.
def test_text_column_against_numeric_schema_rejects_without_parsing():
    df = pd.DataFrame({"col": pd.Series(["a", "b", "c"], dtype="object")})
    check = check_frame(df, _schema_for_col("float64"))

    assert check.adjustments == ()
    assert len(check.rejections) == 1
    message = check.rejections[0]
    assert "col" in message
    assert "text" in message and "number" in message


# object -> category is always exact and comes out as a storage_policy
# conversion.
def test_object_to_category_is_storage_policy():
    df = pd.DataFrame({"col": pd.Series(["x", "x", "y"], dtype="object")})
    check = check_frame(df, _schema_for_col("category"))

    assert check.rejections == ()
    assert len(check.adjustments) == 1
    assert check.adjustments[0].kind == "storage_policy"


# --- structural checks -----------------------------------------------------


# T5 -- a missing column and an extra column each reject, with the column name
# in the message.
def test_t5_missing_and_extra_columns_reject_with_name():
    df = pd.DataFrame({"present": [1], "surprise": [2]})
    schema = TableSchema(
        columns=(
            _spec("present", ColumnRole.measurement, carry=True),
            _spec("absent", ColumnRole.measurement, carry=True),
        )
    )

    check = check_frame(df, schema)

    joined = " || ".join(check.rejections)
    assert "absent" in joined
    assert "surprise" in joined
    assert len(check.rejections) == 2


# T6 -- normalise_frame returns a new frame and leaves the input's dtypes alone.
def test_t6_normalise_frame_does_not_mutate_input():
    df = pd.DataFrame({"col": pd.Series([1, 2, 3], dtype="int32")})
    schema = _schema_for_col("int64")

    out = normalise_frame(df, schema)

    assert out is not df
    assert str(df["col"].dtype) == "int32"
    assert str(out["col"].dtype) == "int64"


# T7 -- normalise_frame on a frame with rejections raises.
def test_t7_normalise_frame_raises_on_rejections():
    df = pd.DataFrame(
        {"col": pd.Series([1, 2, 3], dtype="int64"), "extra": [9, 9, 9]}
    )
    schema = _schema_for_col("int64")

    with pytest.raises(ValueError):
        normalise_frame(df, schema)


# T8 -- a duplicate column name is rejected at TableSchema construction.
def test_t8_duplicate_column_name_rejected():
    with pytest.raises(ValueError):
        TableSchema(
            columns=(
                _spec("dup", ColumnRole.measurement, carry=True),
                _spec("dup", ColumnRole.index, carry=True),
            )
        )


# --- infer_schema --------------------------------------------------------


# T9 -- infer_schema puts a hinted identifier in role identifier and a numeric
# column in role measurement, both with carry_to_children true.
def test_t9_infer_schema_roles_and_carry_default():
    df = pd.DataFrame(
        {
            "participant_id": ["p07", "p07", "p08"],
            "reaction_time": [0.51, 0.62, 0.48],
        }
    )

    schema = infer_schema(
        df, hints={"participant_id": ColumnHint(role=ColumnRole.identifier)}
    )

    pid = schema.spec_for("participant_id")
    rt = schema.spec_for("reaction_time")
    assert pid.role is ColumnRole.identifier
    assert pid.carry_to_children is True
    assert rt.role is ColumnRole.measurement
    assert rt.carry_to_children is True
    # An unhinted float column keeps float64 (see change 1): inference does not
    # narrow numeric width behind the caller's back.
    assert rt.dtype == "float64"


# Second correction: an integer column is a measurement by default, never an
# inferred index, and is not narrowed behind the caller's back -- it keeps int64
# so infer_schema's own output satisfies check_frame on the same frame.
def test_infer_schema_integer_is_measurement_not_index():
    df = pd.DataFrame(
        {
            "frame_index": np.array([0, 1, 2], dtype="int64"),
            "reaction_time_ms": np.array([510, 620, 480], dtype="int64"),
        }
    )

    plain = infer_schema(df)
    assert plain.spec_for("frame_index").role is ColumnRole.measurement
    assert plain.spec_for("reaction_time_ms").role is ColumnRole.measurement
    assert plain.spec_for("reaction_time_ms").dtype == "int64"
    # The inferred schema must not reject the frame it was inferred from.
    assert check_frame(df, plain).ok

    hinted = infer_schema(
        df,
        hints={"frame_index": ColumnHint(role=ColumnRole.index, dtype="int32")},
    )
    fi = hinted.spec_for("frame_index")
    assert fi.role is ColumnRole.index
    assert fi.dtype == "int32"


# Change 1: an unhinted float64 column infers as float64, not float32.
def test_infer_schema_float_column_keeps_float64():
    df = pd.DataFrame(
        {"latency_s": pd.Series([13.351833333333333, 7.9021446], dtype="float64")}
    )

    schema = infer_schema(df)

    assert schema.spec_for("latency_s").dtype == "float64"


# A ColumnHint(dtype="float32") narrows that column in the inferred schema.
def test_infer_schema_float_hint_narrows_to_float32():
    df = pd.DataFrame(
        {"latency_s": pd.Series([13.351833333333333, 7.9021446], dtype="float64")}
    )

    schema = infer_schema(
        df, hints={"latency_s": ColumnHint(dtype="float32")}
    )

    assert schema.spec_for("latency_s").dtype == "float32"


# The hint is a request, not an override: asking for float32 storage on a
# column whose values do not survive float32 is deliberately REFUSED by
# check_frame rather than silently losing precision.
def test_float32_hint_still_rejects_a_value_that_will_not_survive():
    df = pd.DataFrame(
        {"latency_s": pd.Series([13.351833333333333, 7.9021446], dtype="float64")}
    )
    schema = infer_schema(
        df, hints={"latency_s": ColumnHint(dtype="float32")}
    )

    check = check_frame(df, schema)

    assert check.adjustments == ()
    assert len(check.rejections) == 1
    assert "latency_s" in check.rejections[0]
    assert "13.351833333333333" in check.rejections[0]


# The self-consistency property: a schema inferred from a frame must never
# reject that same frame. This is the guardrail that catches a bad inference
# default. Every supported kind here carries at least one value that would NOT
# survive the narrow storage dtype §4.3 names for that kind -- a float not exact
# in float32, an integer outside the int32 range -- so the test fails the moment
# inference narrows any kind behind the caller's back.
def test_inferred_schema_never_rejects_the_frame_it_was_inferred_from():
    df = pd.DataFrame(
        {
            # 13.351833333333333 is not exact in float32.
            "latency_s": pd.Series(
                [13.351833333333333, 7.9021446, 22.447181], dtype="float64"
            ),
            # 3_000_000_000 is outside the int32 range.
            "sample_offset": pd.Series(
                [3_000_000_000, 812_004_117, 41], dtype="int64"
            ),
            # unique-per-row text
            "clip_id": pd.Series(["kx83aq", "qm07zt", "vt59rb"]),
            # repeated-value text
            "condition": pd.Series(["approach", "approach", "withdraw"]),
            # bool
            "responded": pd.Series([True, False, True], dtype="bool"),
            # pandas categorical
            "block": pd.Series(["b2", "b1", "b2"], dtype="category"),
            # text with missing values
            "annotation": pd.Series(["squint onset", None, "brow raise"]),
        }
    )

    schema = infer_schema(df)
    check = check_frame(df, schema)

    assert check.rejections == ()
    # A column that ARRIVES as pandas categorical still infers as category --
    # that is the dtype it arrived in, and inference keeps arrival dtypes.
    assert schema.spec_for("block").dtype == "category"
    # A repeated-value text column that arrived as plain text stays open.
    assert schema.spec_for("condition").dtype != "category"


# A dtype this module has not reasoned about is refused, in both check_frame and
# infer_schema, rather than falling through into an unchecked path.
def test_datetime64_column_is_unsupported():
    df = pd.DataFrame(
        {"t": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])}
    )
    schema = TableSchema(
        columns=(
            _spec("t", ColumnRole.measurement, carry=True, dtype="datetime64[ns]"),
        )
    )

    check = check_frame(df, schema)
    assert len(check.rejections) == 1
    assert "t" in check.rejections[0]
    assert "datetime64" in check.rejections[0]
    assert "not supported" in check.rejections[0]

    with pytest.raises(ValueError):
        infer_schema(df)


def test_nullable_int64_column_is_unsupported():
    df = pd.DataFrame({"n": pd.array([1, 2, pd.NA], dtype="Int64")})
    schema = TableSchema(
        columns=(_spec("n", ColumnRole.measurement, carry=True, dtype="int64"),)
    )

    check = check_frame(df, schema)
    assert len(check.rejections) == 1
    assert "n" in check.rejections[0]
    assert "not supported" in check.rejections[0]

    with pytest.raises(ValueError):
        infer_schema(df)


# A hint naming a column not in the frame is an error, not silence.
def test_infer_schema_hint_for_absent_column_is_error():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        infer_schema(df, hints={"b": ColumnHint(role=ColumnRole.index)})


# A text column keeps whatever text dtype it arrived in, whether its values
# repeat or not -- inference never narrows.
def test_infer_schema_unique_text_keeps_its_arrival_dtype():
    df = pd.DataFrame(
        {"notes": pd.Series(["free text one", "free text two", None, "free text four"])}
    )

    schema = infer_schema(df)

    assert schema.spec_for("notes").dtype == str(df["notes"].dtype)


# A repeated-value text column is NOT closed to a category by inference (the
# opposite of what it used to do): Gelem cannot tell a fixed vocabulary from a
# column a researcher keeps adding labels to, and a category refuses a new
# label at write time. It keeps the text dtype it arrived in.
def test_infer_schema_repeated_text_is_left_open_not_a_category():
    df = pd.DataFrame({"condition": pd.Series(["hit", "hit", "miss", "hit"])})

    schema = infer_schema(df)

    assert schema.spec_for("condition").dtype != "category"
    assert schema.spec_for("condition").dtype == str(df["condition"].dtype)


# The capability is un-defaulted, not lost: a ColumnHint still narrows a
# repeated text column to category on request.
def test_infer_schema_category_hint_still_narrows_repeated_text():
    df = pd.DataFrame({"condition": pd.Series(["hit", "hit", "miss", "hit"])})

    schema = infer_schema(
        df, hints={"condition": ColumnHint(type_tag="text", dtype="category")}
    )

    assert schema.spec_for("condition").dtype == "category"


# A hint carries the media_address type tag that inference cannot guess.
def test_infer_schema_hint_sets_media_address_tag():
    df = pd.DataFrame({"clip_path": ["a.mp4", "b.mp4"]})
    schema = infer_schema(
        df, hints={"clip_path": ColumnHint(type_tag="media_address")}
    )
    assert schema.spec_for("clip_path").type_tag == "media_address"


# --- read-only queries ---------------------------------------------------


def test_schema_queries():
    schema = TableSchema(
        columns=(
            _spec("participant_id", ColumnRole.identifier, carry=True, tag="text"),
            _spec("frame_index", ColumnRole.index, carry=True),
            _spec(
                "frame_address",
                ColumnRole.measurement,
                carry=True,
                tag="media_address",
                dtype="object",
            ),
        )
    )
    assert schema.column_names() == ("participant_id", "frame_index", "frame_address")
    assert {s.name for s in schema.identifiers()} == {"participant_id"}
    assert {s.name for s in schema.indices()} == {"frame_index"}
    assert {s.name for s in schema.columns_with_tag("media_address")} == {"frame_address"}
    with pytest.raises(KeyError):
        schema.spec_for("nope")
