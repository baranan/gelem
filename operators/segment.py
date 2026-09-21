"""
operators/segment.py

SegmentOperator cuts each row of a table into a time-range segment of its
source media -- a video row plus two start/end columns (in seconds)
becomes a row holding a #t=<start>-<end> address (docs/media_architecture.md
§3.2). A segment is an address, not an extracted file (§3.5): this
operator decodes nothing, opens no media file, and writes no files.

Carrying the source row's columns down follows docs/architecture.md §4.2
and operators/CLAUDE.md "Carry the lineage columns" -- neither is restated
here. The carry set itself is computed by the caller (Dataset.columns_to_carry,
via AppController) and arrives on the input snapshot as carry_columns; this
operator only reads it, never computes it and never touches Dataset.
"""

from __future__ import annotations

import math

import pandas as pd

from media.media_address import MediaAddress
from media.media_address import format as format_address
from media.media_address import parse as parse_address
from operators.base import BaseOperator
from operators.descriptor import (
    ColumnParameter,
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModeDescriptor,
    ModelLifecycle,
    NewTableNameParameter,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
)

# The new columns this operator adds, after the carried ones, in order.
_NEW_COLUMNS = (
    "segment_address",
    "segment_index",
    "segment_start",
    "segment_end",
    "segment_duration",
)


def _as_finite_seconds(value: object) -> float | None:
    """Read one start/end cell as a finite float, or None if it is
    missing, non-numeric, or not finite. A bool is not a time value even
    though Python lets float(True) succeed, so it is refused explicitly.
    """
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds):
        return None
    return seconds


def _is_blank_media_value(value: object) -> bool:
    """True if a media cell is missing or blank.

    pd.isna() alone is not enough: str() on a missing (NaN/None) cell
    yields the literal text "nan", which the address parser happily
    accepts as a one-character-short bare path, so a row with no media
    value would otherwise silently produce an address like
    "nan#t=1.0-2.0" instead of being dropped.
    """
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


class SegmentOperator(BaseOperator):
    """Cuts each row of a table into a time-range segment of its source
    media, addressed rather than extracted."""

    name = "segment"

    descriptor = OperatorDescriptor(
        name="segment",
        version="1.0",
        description=(
            "Cuts each row of the active table into a time-range segment "
            "of its source media, using two researcher-chosen columns "
            "holding the start and end time in seconds. Adds a canonical "
            "#t=<start>-<end> address, a per-source segment index, start, "
            "end and duration, and carries the source table's identifier, "
            "index, and carry-flagged columns down. Decodes no media and "
            "writes no files -- a segment is an address, not an extracted "
            "clip."
        ),
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.TABLE,
                label="Cut into segments",
                inputs=(
                    InputSpec(
                        name="active_table",
                        label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                media_requirement=MediaRequirement.METADATA,
                parameters=(
                    ColumnParameter(
                        name="media_column",
                        label="Media column",
                        from_input="active_table",
                        required_tags=("media_path",),
                    ),
                    ColumnParameter(
                        name="start_column",
                        label="Start column (seconds)",
                        from_input="active_table",
                    ),
                    ColumnParameter(
                        name="end_column",
                        label="End column (seconds)",
                        from_input="active_table",
                    ),
                    NewTableNameParameter(
                        name="output_table",
                        label="New table name",
                        default="segments",
                    ),
                ),
                output=OutputSpec(
                    creates_table=True,
                    # P1.7-1: segment_index numbers each source video's
                    # segments in order -- it is a lineage index, not a
                    # measurement, and this is what makes
                    # Dataset.columns_to_carry() always carry it down to
                    # any operator that further splits a segment (a frame
                    # operator, P1.7). Without this hint infer_schema's
                    # plain int-column default (role=measurement) would
                    # apply instead.
                    table_columns=(
                        OutputColumn(
                            name="segment_index", type_tag="numeric",
                            role="index",
                        ),
                    ),
                ),
                model_lifecycle=ModelLifecycle.NONE,
                deterministic=True,
                cacheable=True,
            ),
        ),
    )

    def create_table(self, df: pd.DataFrame, run) -> pd.DataFrame:
        media_column = run.parameters["media_column"]
        start_column = run.parameters["start_column"]
        end_column = run.parameters["end_column"]

        # The carry set is computed by the caller (Dataset.columns_to_carry,
        # docs/architecture.md §4.2) and handed to us on the input snapshot,
        # never recomputed here. Narrow it to columns df actually has, in
        # the carry set's own order -- and drop media_column itself: a
        # segment row already carries the source path inside
        # segment_address, so carrying a second copy in media_column would
        # just repeat the same path string for no use (operators/CLAUDE.md,
        # "Carry the lineage columns", the full_path example). This is a
        # carry decision this operator makes for itself, separate from the
        # general carry_columns narrowing parameter (out of scope here).
        carry_columns = [
            name
            for name in run.data.snapshot("active_table").carry_columns
            if name in df.columns and name != media_column
        ]
        ordered_columns = carry_columns + list(_NEW_COLUMNS)

        # Build one candidate row per surviving input row, in input row
        # order. "_source_path" and "_source_order" are working columns
        # used only to compute segment_index below; they never reach the
        # returned frame.
        candidates: list[dict] = []
        dropped = 0
        for position, (_, source_row) in enumerate(df.iterrows()):
            row_id = source_row["row_id"]
            media_value = source_row[media_column]
            if _is_blank_media_value(media_value):
                dropped += 1
                run.report_row_error(
                    row_id, "MissingMedia",
                    f"column {media_column!r} is missing or blank",
                )
                continue

            start_seconds = _as_finite_seconds(source_row[start_column])
            end_seconds = _as_finite_seconds(source_row[end_column])
            if (
                start_seconds is None
                or end_seconds is None
                or start_seconds < 0
                or end_seconds < 0
                or end_seconds <= start_seconds
            ):
                dropped += 1
                run.report_row_error(
                    row_id, "InvalidTimeRange",
                    f"start ({source_row[start_column]!r}) and end "
                    f"({source_row[end_column]!r}) must both be finite, "
                    "non-negative numbers with end after start",
                )
                continue

            # decision 9 (docs/media_architecture.md §3.6): the source path
            # is the PATH PORTION of the media value, read through the
            # shared parser -- never split on "#" and never built by string
            # concatenation. An already-fragmented value (e.g. a source
            # cell holding #f=...) contributes only its path.
            source_path = parse_address(str(media_value)).path
            start_us = round(start_seconds * 1_000_000)
            end_us = round(end_seconds * 1_000_000)
            segment_address = MediaAddress(
                path=source_path,
                time_range_us=(start_us, end_us),
            )

            row = {name: source_row[name] for name in carry_columns}
            row["segment_address"] = format_address(segment_address)
            row["segment_start"] = start_seconds
            row["segment_end"] = end_seconds
            row["segment_duration"] = end_seconds - start_seconds
            row["_source_path"] = source_path
            row["_source_order"] = position
            candidates.append(row)

        if dropped:
            # A quick-glance count for the status-bar run indicator; the
            # per-row reason for each dropped row is now on the row-error
            # channel above (P1.7-1), not folded into this one aggregate
            # string the way it was before.
            run.log(f"segment: dropped {dropped} row(s); see the row-error report for detail")

        # segment_index: 0-based position among the output rows that share
        # the same source path, ordered by segment_start and then by
        # original row order. Computed by ranking a SORTED COPY of the
        # candidate indices, so the returned frame keeps the candidates in
        # their original (surviving) row order rather than being reordered
        # by this ranking pass.
        rank_order = sorted(
            range(len(candidates)),
            key=lambda i: (
                candidates[i]["_source_path"],
                candidates[i]["segment_start"],
                candidates[i]["_source_order"],
            ),
        )
        next_index_for_path: dict[str, int] = {}
        for i in rank_order:
            path = candidates[i]["_source_path"]
            candidates[i]["segment_index"] = next_index_for_path.get(path, 0)
            next_index_for_path[path] = candidates[i]["segment_index"] + 1

        for row in candidates:
            del row["_source_path"]
            del row["_source_order"]

        return pd.DataFrame(candidates, columns=ordered_columns)
