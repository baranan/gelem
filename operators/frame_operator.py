"""
operators/frame_operator.py

FrameOperator turns each row of a table into one row per frame of its
source media -- a segment row (a #t=<start>-<end> address) or a
whole-video row (a bare path) becomes many rows, each holding a #f=<n>
address (docs/media_architecture.md §3.2, §6.2's "P1.7 Frame operator"
paragraph). A frame is an address, not an extracted file: this operator
decodes no pixels and writes no files. Frame enumeration goes entirely
through MediaResolver.get_frame_times() (the per-file frame-time index)
and the pure function media_address.frames_in_range() -- never
decode_video_span(), which decodes pixels this operator does not need.

Carrying the source row's columns down follows docs/architecture.md §4.2
and operators/CLAUDE.md "Carry the lineage columns", exactly as
operators/segment.py does -- neither is restated here.
"""

from __future__ import annotations

import dataclasses
import pathlib

import pandas as pd

from media.extensions import is_image_path
from media.media_address import (
    MediaAddressError,
    format as format_address,
    frames_in_range,
    parse as parse_address,
)
from media.resolver import MediaResolverError
from operators.base import BaseOperator, OperatorSetupError
from operators.descriptor import (
    ColumnParameter,
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModeDescriptor,
    ModelLifecycle,
    NewTableNameParameter,
    NumberParameter,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
)

# The new columns this operator adds, after the carried ones, in order.
#
#   frame_address       -- the #f=<n> address of this frame (media_path).
#   frame_index         -- this frame's ordinal position in presentation
#                           order within the WHOLE source stream (decision
#                           8), exactly as MediaResolver reports it -- a
#                           lineage index, matching docs/architecture.md
#                           §4.2's own example column of this name.
#   frame_time          -- this frame's absolute presentation time on the
#                           source's own clock, in seconds (None for a
#                           still image, which has no timeline).
#   time_within_segment  -- frame_time minus the row's own range start (0
#                           for a bare whole-video row, whose range starts
#                           at the stream's own zero -- decision 10; None
#                           for a still image, for the same reason
#                           frame_time is).
#   segment_index        -- carried from the source row's own segment_index
#                           when the input already has one (always carried
#                           automatically as an index-role column, per
#                           operators/CLAUDE.md -- so this operator narrows
#                           it OUT of the ordinary carry set and re-adds it
#                           here itself, the same "carry decision the
#                           operator makes" segment.py already makes for
#                           media_column). When the input has none, the two
#                           address shapes differ (P1.7-2 round 6, fix 3):
#                           a bare whole-video row synthesises 0, the only
#                           value that could ever be right for the trivial
#                           single-segment case (decision 4 treats a bare
#                           path as the range covering the whole file); a
#                           #t= row synthesises a MISSING value instead --
#                           it names some particular range of the source,
#                           and 0 would assert a segment number that was
#                           never actually assigned to it. No numbering
#                           rule is invented for that case.
_NEW_COLUMNS = (
    "frame_address",
    "frame_index",
    "frame_time",
    "time_within_segment",
    "segment_index",
)


def _is_blank_media_value(value: object) -> bool:
    """True if a media cell is missing or blank. See
    operators/segment.py's identical helper for why pd.isna() alone is not
    enough (str(NaN) parses as a bare one-character path).
    """
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


class FrameOperator(BaseOperator):
    """Splits each row of a table into one row per frame of its source
    media, addressed rather than extracted."""

    name = "frame"

    descriptor = OperatorDescriptor(
        name="frame",
        version="1.0",
        description=(
            "Splits each row of the active table into one row per frame "
            "of its source media -- a segment row's #t=<start>-<end> "
            "range, or a whole-video row's bare path. Adds a canonical "
            "#f=<n> address, the frame's ordinal in the source stream, "
            "its absolute time on the source clock, its time within the "
            "segment, and the segment index, and carries the source "
            "table's identifier, index, and carry-flagged columns down. "
            "Decodes no pixels and writes no files -- a frame is an "
            "address, not an extracted image."
        ),
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.TABLE,
                label="Split into frames",
                inputs=(
                    InputSpec(
                        name="active_table",
                        label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                # The operator resolves each row's address itself, through
                # run.resolver -- this is the ADDRESS case
                # operators/descriptor.py's own docstring describes.
                media_requirement=MediaRequirement.ADDRESS,
                parameters=(
                    ColumnParameter(
                        name="media_column",
                        label="Media column",
                        from_input="active_table",
                        required_tags=("media_path",),
                    ),
                    NumberParameter(
                        name="frame_step",
                        label="Frame step (keep every Nth)",
                        minimum=1,
                        maximum=10_000,
                        decimals=0,
                        default=1,
                    ),
                    NewTableNameParameter(
                        name="output_table",
                        label="New table name",
                        # Not "frames" -- Dataset pre-populates a table by
                        # that name, and NewTableNameParameter's collision
                        # rule (CLAUDE.md, "Data ownership") refuses a
                        # researcher-typed name that already exists rather
                        # than silently renaming it.
                        default="frame_rows",
                    ),
                ),
                output=OutputSpec(
                    creates_table=True,
                    # P1.7-1's mechanism, its first real consumer for a
                    # media-address output column: frame_address needs
                    # type_tag media_path so the gallery renders it rather
                    # than showing a placeholder, and frame_index /
                    # segment_index need role="index" so they are lineage
                    # columns, not infer_schema's plain int-column default
                    # of role=measurement (which is right for frame_time
                    # and time_within_segment -- true measurements -- so
                    # neither is hinted here).
                    table_columns=(
                        OutputColumn(
                            name="frame_address", type_tag="media_path",
                        ),
                        OutputColumn(
                            name="frame_index", type_tag="numeric",
                            role="index",
                        ),
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
        frame_step = int(run.parameters["frame_step"])

        # The carry set is computed by the caller (Dataset.columns_to_carry,
        # docs/architecture.md §4.2) and handed to us on the input snapshot,
        # never recomputed here. Narrow it to columns df actually has, drop
        # media_column itself (a frame row already carries the source path
        # inside frame_address -- operators/CLAUDE.md's full_path example),
        # and drop segment_index -- this operator adds its OWN segment_index
        # below (see _NEW_COLUMNS), so carrying a same-named column down
        # unchanged would collide with it.
        carry_columns = [
            name
            for name in run.data.snapshot("active_table").carry_columns
            if name in df.columns and name not in (media_column, "segment_index")
        ]

        # P1.7-2 round 6, fix 1: a carried column sharing a name with one
        # of this operator's OWN new columns (segment_index is already
        # excluded above, deliberately -- see its own note) would make the
        # returned frame carry that name twice, the inherited value
        # silently overwritten by the operator's own. Running this
        # operator on a table it produced earlier -- which already has
        # frame_address, frame_index, frame_time and time_within_segment
        # -- does exactly that. Refused before any row is processed,
        # naming the colliding column(s); never silently dropped and never
        # renamed, since either would hide the collision rather than
        # surface it.
        colliding = sorted(name for name in carry_columns if name in _NEW_COLUMNS)
        if colliding:
            raise OperatorSetupError(
                f"the active table already has column(s) {colliding} that "
                f"this operator's own output columns "
                f"({', '.join(_NEW_COLUMNS)}) would collide with -- rename "
                "or drop the colliding column(s) in the source table "
                "before running Split into frames"
            )

        ordered_columns = carry_columns + list(_NEW_COLUMNS)

        has_segment_index = "segment_index" in df.columns

        # get_frame_times() demuxes the whole file the first time it is
        # asked for a given file -- cache its result per distinct (path,
        # stream selector) so a table of many rows against the same source
        # video pays that cost once, not once per row.
        frame_times_cache: dict[tuple, list] = {}

        out_rows: list[dict] = []
        dropped = 0

        for _, source_row in df.iterrows():
            row_id = source_row["row_id"]
            media_value = source_row[media_column]
            if _is_blank_media_value(media_value):
                dropped += 1
                run.report_row_error(
                    row_id, "MissingMedia",
                    f"column {media_column!r} is missing or blank",
                )
                continue

            try:
                addr = parse_address(str(media_value))
            except MediaAddressError as e:
                dropped += 1
                run.report_row_error(row_id, "MalformedAddress", str(e))
                continue

            # P1.7-2 round 6, fix 2: checked once, for every row, before
            # the image-versus-video branch below -- a still image never
            # reaches the resolver at all (its branch just builds a #f=0
            # address and appends a row), so a missing .png used to be
            # reported as a successful output row instead of a skipped
            # one. A video row already fails through get_frame_times()'s
            # own OSError/MediaResolverError catch further down, but
            # checking existence here means it fails the same way an
            # image row does, before either branch runs, rather than only
            # by accident of the video path happening to decode. This is
            # an existence check only -- os.path.exists()/pathlib, never
            # av.open() or Image.open() -- so it decodes nothing and does
            # not touch CLAUDE.md's "only the resolver decodes source
            # media" rule. The OSError catch around get_frame_times stays
            # as the backstop for a file that disappears, or otherwise
            # becomes unreadable, between this check and the real read.
            if not pathlib.Path(addr.path).exists():
                dropped += 1
                run.report_row_error(
                    row_id, "MissingMedia",
                    f"file does not exist: {addr.path}",
                )
                continue

            # P1.7-2 round 6, fix 3: see the module's own note on
            # segment_index above -- a bare row synthesises 0 (the only
            # correct value for the trivial single-segment case), a #t=
            # row with no source segment_index synthesises a missing
            # value instead, never an invented number.
            if has_segment_index:
                segment_index_value = source_row["segment_index"]
            elif addr.time_range_us is not None:
                segment_index_value = None
            else:
                segment_index_value = 0
            carried = {name: source_row[name] for name in carry_columns}

            # A still image has exactly one frame, at ordinal 0, and no
            # timeline (media/resolver.py's FramePayload docstring) -- so
            # it is emitted as a single frame row rather than reported as
            # a skipped one. This is decided, not merely the easy default:
            # the resolver already treats "#f=0" on an image as valid
            # everywhere else (resolve_frame's still-image path), a bare
            # video row is already the trivial single-segment case
            # (decision 4), and Gelem's own rule is to fail toward keeping
            # data (docs/architecture.md §4.2) -- dropping every image row
            # from a table that mixes images and videos would silently
            # lose rows a researcher has every reason to expect back.
            if is_image_path(addr.path):
                frame_addr = dataclasses.replace(
                    addr, frame=0, time_us=None, time_range_us=None,
                )
                row = dict(carried)
                row["frame_address"] = format_address(frame_addr)
                row["frame_index"] = 0
                row["frame_time"] = None
                row["time_within_segment"] = None
                row["segment_index"] = segment_index_value
                out_rows.append(row)
                continue

            cache_key = (addr.path, addr.stream)
            frame_times = frame_times_cache.get(cache_key)
            if frame_times is None:
                try:
                    frame_times = run.resolver.get_frame_times(addr)
                # OSError alongside MediaResolverError/MediaAddressError:
                # a missing, permission-denied or otherwise unreadable
                # file can reach av.open() inside the resolver's decoder
                # pool and raise a raw OSError-family exception instead of
                # a MediaResolverError -- this call is guarded against
                # exactly that. Reported under the same "UnreadableMedia"
                # kind as a MediaResolverError, not a separate one: from
                # the researcher's side both mean the same thing -- this
                # row's media could not be read -- and CLAUDE.md's media
                # rule keeps the internal exception taxonomy PyAV raises
                # an implementation detail of the resolver, not something
                # every caller should have to sort into its own category.
                except (MediaResolverError, MediaAddressError, OSError) as e:
                    dropped += 1
                    run.report_row_error(row_id, "UnreadableMedia", str(e))
                    continue
                frame_times_cache[cache_key] = frame_times

            try:
                indices = frames_in_range(addr, frame_times)
            except MediaAddressError as e:
                dropped += 1
                run.report_row_error(row_id, "InvalidTimeRange", str(e))
                continue

            if not indices:
                dropped += 1
                run.report_row_error(
                    row_id, "InvalidTimeRange",
                    "the row's time range contains no frames",
                )
                continue

            # The segment's own declared start (decision 3's range start),
            # never the first CAPTURED frame's time -- a range's start and
            # its earliest frame need not coincide. A bare whole-video row
            # has no declared range; decision 4 treats it as the range
            # covering the whole file, whose start is the stream's own
            # zero (decision 10).
            if addr.time_range_us is not None:
                segment_start_us = addr.time_range_us[0]
            else:
                segment_start_us = 0

            for local_position, frame_index in enumerate(indices):
                if local_position % frame_step != 0:
                    continue
                pts_us = frame_times[frame_index]
                frame_addr = dataclasses.replace(
                    addr, frame=frame_index, time_us=None, time_range_us=None,
                )
                row = dict(carried)
                row["frame_address"] = format_address(frame_addr)
                row["frame_index"] = frame_index
                row["frame_time"] = pts_us / 1_000_000
                row["time_within_segment"] = (
                    (pts_us - segment_start_us) / 1_000_000
                )
                row["segment_index"] = segment_index_value
                out_rows.append(row)

        if dropped:
            # A quick-glance count for the status-bar run indicator; the
            # per-row reason for each dropped row is on the row-error
            # report above (operators/CLAUDE.md, report_row_error).
            run.log(
                f"frame: dropped {dropped} row(s); see the row-error "
                f"report for detail"
            )

        return pd.DataFrame(out_rows, columns=ordered_columns)
