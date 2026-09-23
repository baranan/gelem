"""
operators/video_frames.py

VideoFramesOperator extracts frames from videos into a frame-level table.

Input table: one row per video, with at least one column holding the
video file path (default: full_path) and any number of metadata columns
(participant_id, condition, session, etc.). The column's cell may be a
bare video path (the whole file), a #t= time point or range, or a #f=
frame ordinal -- media/media_address.py's address grammar.

Output table: one row per kept frame. Every column from the source
video row is copied onto each of its frame rows, full_path is
overwritten to point at the saved frame JPEG, and two new columns are
added: frame_number (the source frame's ordinal, from the resolver) and
video_file (the source video's filename).

The researcher chooses two parameters via the dialog; both travel in
run.parameters, never on the operator instance:
    video_column  -- which column holds the video path or address
    frame_step    -- keep every Nth frame (1 keeps everything)

Frame JPEGs are saved under run.paths.outputs_dir / "frames", one fresh
subfolder per create_table() call (named run_YYYY.MM.DD_HH.MM.SS.cs) so
re-runs never overwrite previous extractions. Within a run, filenames are
prefixed with the source video's stem so frames from different videos do
not collide on disk.

Decoding goes through run.resolver (media/resolver.py) -- CLAUDE.md's
media rule ("only the media resolver decodes source media") and
docs/media_architecture.md section 3.3. This operator never opens a
video file itself.
"""

from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd

from operators.base import BaseOperator
from operators.descriptor import (
    ColumnParameter,
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModelLifecycle,
    ModeDescriptor,
    NumberParameter,
    OperatorDescriptor,
    OutputSpec,
)
from media.extensions import looks_like_video_extension
from media.media_address import (
    MediaAddressError,
    format as format_address,
    parse as parse_address,
)
from media.resolver import MediaResolverError


class VideoFramesOperator(BaseOperator):
    """
    Extracts frames from each video row into a new frame-level table.
    """

    name = "video_frames"

    # ------------------------------------------------------------------
    # Descriptor (P1.12d-1). What create_table() ACTUALLY does today:
    #  - one TABLE mode: it builds a brand-new frame-level table
    #    (creates_table), one row per kept frame;
    #  - TWO parameters, both read by create_table() from run.parameters
    #    (P1.12d-2a -- they no longer touch the operator instance):
    #      video_column -- a column of the active table holding the video
    #                      path or address
    #      frame_step   -- keep every Nth frame; declared min 1,
    #                      max 10_000, default 1
    #  - media_requirement ADDRESS: create_table() resolves each row's
    #    address itself, through run.resolver -- this is the ADDRESS case
    #    the descriptor enum calls out by name.
    #  - model_lifecycle NONE;
    #  - deterministic FALSE / cacheable FALSE: every run writes into a
    #    fresh timestamped subfolder --
    #        now = datetime.now()
    #        stamp = now.strftime("%Y.%m.%d_%H.%M.%S") + ...
    #        run_dir = run.paths.outputs_dir / "frames" / f"run_{stamp}"
    #    -- and the output table's overwritten full_path values carry
    #    that timestamp, so identical inputs give different output.
    # ------------------------------------------------------------------
    descriptor = OperatorDescriptor(
        name="video_frames",
        version="1.0",
        description=(
            "Reads a video-path column of the active table, decodes each "
            "video through the shared media resolver, and builds a new "
            "frame-level table with one row per kept frame (every Nth "
            "frame of a bare path or #t= range; the single frame of a "
            "#t= point or #f= address). Each frame row copies the source "
            "video row's columns, repoints full_path at the saved frame "
            "JPEG, and adds frame_number and video_file."
        ),
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.TABLE,
                label="Export frames as files",
                inputs=(
                    InputSpec(
                        name="active_table",
                        label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                media_requirement=MediaRequirement.ADDRESS,
                parameters=(
                    ColumnParameter(
                        name="video_column",
                        label="Video path column",
                        from_input="active_table",
                        # full_path is Gelem's conventional media-path
                        # column, so preselect it when the table offers it.
                        # If the default names a column the input does not
                        # offer, the field falls back to blank rather than
                        # showing a column that is not there.
                        default="full_path",
                    ),
                    NumberParameter(
                        name="frame_step",
                        label="Frame step (keep every Nth)",
                        minimum=1,
                        maximum=10_000,
                        decimals=0,
                        default=1,
                    ),
                ),
                output=OutputSpec(creates_table=True),
                model_lifecycle=ModelLifecycle.NONE,
                deterministic=False,
                cacheable=False,
            ),
        ),
    )

    # No __init__: this operator holds nothing on self. It writes only
    # under run.paths.outputs_dir, supplied fresh on every run -- an
    # output directory captured at construction time would be shared by
    # every concurrent run of this singleton (operators/CLAUDE.md, "Write
    # only to run.paths").

    # No get_parameters_dialog(). video_column and frame_step are declared
    # on the descriptor above; MainWindow builds the form from them
    # (ui/parameter_dialog.py) and passes the collected values to the
    # controller in run.parameters. This module contains no Qt.
    #
    # video_column carries default="full_path", so the generated form
    # arrives with that column preselected -- matching the hand-drawn
    # dialog it replaced. When the active table has no full_path column the
    # field falls back to blank and the researcher picks one explicitly.

    def _write_frame_row(
        self, src_row: pd.Series, run_dir: Path, video_stem: str,
        video_filename: str, addr_hash: str, payload,
    ) -> dict:
        """Save one decoded frame as a JPEG under run_dir and build the
        output row for it. Shared by both the span (many frames) and the
        single-frame (point/ordinal address) paths below, so the row shape
        cannot drift between them.

        payload.frame_ordinal is the resolver's own answer, and it decides
        whether one exists -- never something this operator computes or
        assumes. The span path always requests with_ordinals=True, so it
        is never None there. A #t= point resolve (never a #f=, which
        always gets a real ordinal) is the one address shape where it can
        still be None -- only when nothing has yet built that file's
        frame-time index.

        The FILENAME cannot fall back to a constant (0, say) when the
        ordinal is missing: two #t= point rows on the SAME video, at two
        different times, would both resolve to "<stem>_frame_000000.jpg"
        and the second write would silently overwrite the first's
        picture on disk, while the table still lists both rows as if
        each had its own frame. presentation_time_us is the fallback
        instead -- FramePayload always reports it for a real video frame
        regardless of whether an ordinal exists, and two different times
        can never collide on it. The stored frame_number column is left
        exactly as the resolver reported it -- None, not a fabricated 0
        -- so a reader is never told a real position that was not
        actually known.

        Review round 5: frame ordinal (or presentation time) alone is
        still not enough -- neither one names a region or a stream
        selector, so two rows addressing different #r= crops (or
        different &v=/&a= streams) of the SAME frame would still write
        the same filename, and the second row's JPEG would silently
        overwrite the first's. addr_hash -- an 8-hex-character digest of
        the row's own canonical address string, computed once per row by
        the caller -- disambiguates that: it differs whenever the region,
        stream selector, or anything else in the address differs, and is
        identical across every frame drawn from the SAME row's span, so
        it adds no needless churn there.
        """
        if payload.frame_ordinal is not None:
            name_suffix = f"frame_{payload.frame_ordinal:06d}"
        else:
            name_suffix = f"t_{payload.presentation_time_us}us"
        out_path = run_dir / f"{video_stem}_{name_suffix}_{addr_hash}.jpg"
        self.save_image(payload.pixels, out_path)
        row = src_row.to_dict()
        row["full_path"] = str(out_path)
        row["frame_number"] = payload.frame_ordinal
        row["video_file"] = video_filename
        row.pop("row_id", None)
        return row

    def create_table(
        self,
        df: pd.DataFrame,
        run,
    ) -> pd.DataFrame:
        from datetime import datetime

        # Parameters arrive in run.parameters, validated against this
        # operator's descriptor before the run started -- never off self.
        video_column: str = run.parameters["video_column"]
        frame_step: int = int(run.parameters["frame_step"])

        work = df.copy()
        if video_column not in work.columns:
            print(
                f"[VideoFramesOperator] Column "
                f"'{video_column}' not in input table. "
                f"Available: {list(work.columns)}"
            )
            return pd.DataFrame()

        # Each create_table() call writes into its own subfolder so
        # re-runs never overwrite a previous run's frames. The 2-digit
        # centiseconds suffix keeps the name unique even for back-to-back
        # runs within the same second.
        now = datetime.now()
        stamp = (
            now.strftime("%Y.%m.%d_%H.%M.%S")
            + f".{now.microsecond // 10000:02d}"
        )
        run_dir = run.paths.outputs_dir / "frames" / f"run_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        out_rows: list[dict] = []
        videos_seen = 0
        videos_skipped = 0
        non_videos_skipped = 0
        total_rows = len(work)

        for row_num, (_, src_row) in enumerate(work.iterrows(), start=1):
            cell_raw = src_row[video_column]
            if not cell_raw:
                videos_skipped += 1
                continue
            cell = str(cell_raw)

            # A cheap, cell-level gate before ever asking the resolver to
            # do anything: media/extensions.py's VIDEO_EXTENSIONS is the
            # one shared authority for what counts as a VIDEO extension --
            # not column_types/renderers.py's own VIDEO_EXTENSIONS, which
            # is a display-layer concern this operator must not depend
            # on, and not the broader MEDIA_EXTENSIONS (images included),
            # which would let a still-image cell through this gate only
            # to fail much later, deep inside decode_video_span(), and be
            # counted as a decoded-but-failed video rather than what the
            # gate's own name says: not a video. looks_like_video_extension
            # checks both spellings a cell can end in its extension with
            # -- the whole string, or the portion before a '#' fragment (a
            # bare cell has no fragment, so only the whole-string check
            # ever applies to it; an addressed cell like "clip.mp4#t=1.0"
            # ends in ".mp4" only in its pre-'#' portion).
            if looks_like_video_extension(cell) is None:
                print(
                    f"[VideoFramesOperator] Not a video, skipping: {cell}"
                )
                non_videos_skipped += 1
                continue

            try:
                addr = parse_address(cell)
            except MediaAddressError as e:
                run.log(f"row {row_num} of {total_rows}: skipped -- {e}")
                videos_skipped += 1
                continue

            # The address's own parsed path, not Path(cell): cell may
            # carry a fragment (#t=, #f=, ...) that must never leak into a
            # filename.
            video_path = Path(addr.path)
            video_stem = video_path.stem
            video_filename = video_path.name
            is_point_address = addr.selects_single_frame

            # An 8-hex-character digest of this row's own canonical
            # address string -- computed once per row, since every frame
            # this row yields (one, for a point address; possibly many,
            # for a span) shares the same region and stream selector.
            # Without it, two rows selecting different #r= crops (or
            # different &v=/&a= streams) of the SAME frame would name the
            # same file and the second write would silently overwrite the
            # first's picture -- frame_ordinal / presentation_time_us name
            # a moment in the file, never a region or a stream.
            addr_hash = hashlib.sha256(
                format_address(addr).encode("utf-8")
            ).hexdigest()[:8]

            run.log(f"row {row_num} of {total_rows}: {video_filename}")

            kept = 0
            span_iter = None
            try:
                if is_point_address:
                    # #t= point or #f=: exactly one frame. A #f= address's
                    # ordinal is always real (resolve_frame builds its
                    # index unconditionally for that case); a #t= point's
                    # is real only if some earlier with_ordinals=True span
                    # or #f= resolve already built the index for this file
                    # -- frame ordinals are entirely the resolver's call
                    # (see decode_video_span's with_ordinals below), never
                    # something this operator primes into existence.
                    payload = run.resolver.resolve_frame(addr, "analysis")
                    out_rows.append(self._write_frame_row(
                        src_row, run_dir, video_stem, video_filename,
                        addr_hash, payload,
                    ))
                    kept = 1
                else:
                    # Bare path or #t= range: walk it in order, keeping
                    # every Nth frame (the first is always kept).
                    # with_ordinals=True: this operator numbers every kept
                    # frame by its real, file-wide position, so it asks
                    # the resolver to build (or reuse) the per-file
                    # frame-time index rather than trusting its own
                    # enumeration -- span_index would only equal the real
                    # ordinal by coincidence for a bare path starting at
                    # time 0, and would be silently wrong for a #t= range
                    # starting elsewhere in the file.
                    span_iter = run.resolver.decode_video_span(
                        addr, "analysis", with_ordinals=True
                    )
                    for span_index, payload in enumerate(span_iter):
                        if span_index % frame_step != 0:
                            continue
                        out_rows.append(self._write_frame_row(
                            src_row, run_dir, video_stem, video_filename,
                            addr_hash, payload,
                        ))
                        kept += 1
            except (MediaResolverError, MediaAddressError, OSError) as e:
                run.log(
                    f"row {row_num} of {total_rows}: skipped {video_filename} "
                    f"-- {e}"
                )
                videos_skipped += 1
                continue
            finally:
                # decode_video_span acquires its decoder pool handle
                # lazily and releases it when exhausted, closed, or
                # garbage-collected -- but a raise mid-iteration (above)
                # abandons it without exhausting it, so close it
                # explicitly on every exit path.
                if span_iter is not None:
                    span_iter.close()

            videos_seen += 1
            print(
                f"[VideoFramesOperator] {video_filename}: "
                f"{kept} frames kept (step={frame_step})"
            )

        if videos_seen == 0:
            raise ValueError(
                f"VideoFramesOperator found 0 usable video rows in column "
                f"'{video_column}'."
            )

        print(
            f"[VideoFramesOperator] Done — "
            f"{len(out_rows)} frames from {videos_seen} videos "
            f"({non_videos_skipped} non-video, "
            f"{videos_skipped} skipped)"
        )
        return pd.DataFrame(out_rows)
