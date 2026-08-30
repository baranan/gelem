"""
media/media_address.py

MediaAddress -- the parsed, structured form of a media value, and the pure
logic that gives its fragment grammar exact meaning.

See docs/media_architecture.md section 3.6 ("Address semantics -- settled")
for the decisions this module implements, and section 3.2 for the grammar.
Each function below names the decision number it carries out.

This module is pure logic. It never opens a file, never imports pandas,
numpy, PyAV, cv2, PIL or Qt, and never assumes a frame rate. Where a
decision requires real file data (the resolver's job, P1.2), the function
below takes that data as a plain argument -- a list of frame times, a
frame's pixel width -- rather than reading it itself.

Times are held internally as exact integer microseconds throughout, never
as floating-point seconds, so that two addresses which should be identical
always compare and hash identically (decision 9's whole reason for
existing). Region fractions are held the same way, as integer millionths.
"""

import bisect
import dataclasses
import functools
import pathlib
import posixpath
import re
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

# No wildcard import ever picks up the bare builtin `format` by accident --
# this module deliberately shadows it (matching the resolver interface
# named in docs/media_architecture.md section 3.3), so `__all__` makes
# that shadowing explicit rather than a `from ... import *` surprise.
__all__ = [
    "MediaAddressError",
    "StreamSelector",
    "Region",
    "PixelRegion",
    "MediaAddress",
    "from_path",
    "parse",
    "format",
    "absolutise",
    "relativise",
    "canonical_key",
    "resolve_source",
    "select_frame",
    "frames_in_range",
    "region_to_pixels",
]


class MediaAddressError(ValueError):
    """Raised for any malformed or degenerate address (decision 11)."""


# ---------------------------------------------------------------------------
# The structured value types.
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class StreamSelector:
    """An explicit &v=<n> or &a=<n> component (decision 7)."""

    kind: str   # 'v' or 'a'
    index: int  # zero-based, within streams of that kind

    def __post_init__(self):
        if self.kind not in ("v", "a"):
            raise MediaAddressError(
                f"stream selector kind must be 'v' or 'a', got {self.kind!r}"
            )
        if self.index < 0:
            raise MediaAddressError("stream selector index cannot be negative")


@dataclasses.dataclass(frozen=True)
class Region:
    """A #r=x,y,w,h component, held as integer millionths (decision 5).

    x, y, w, h are fractions of the upright frame in the closed interval
    0 to 1, each represented here as an integer from 0 to 1_000_000.
    """

    x: int
    y: int
    w: int
    h: int

    def __post_init__(self):
        for name, value in (("x", self.x), ("y", self.y), ("w", self.w), ("h", self.h)):
            if not (0 <= value <= 1_000_000):
                raise MediaAddressError(
                    f"region {name} is out of range 0 to 1: {value / 1_000_000}"
                )
        if self.w <= 0 or self.h <= 0:
            raise MediaAddressError("region width and height must be positive")
        if self.x + self.w > 1_000_000:
            raise MediaAddressError("region extends past the right edge (x + w > 1)")
        if self.y + self.h > 1_000_000:
            raise MediaAddressError("region extends past the bottom edge (y + h > 1)")


@dataclasses.dataclass(frozen=True)
class PixelRegion:
    """The result of region_to_pixels: a rectangle in whole pixels."""

    left: int
    top: int
    width: int
    height: int


@dataclasses.dataclass(frozen=True, eq=False)
class MediaAddress:
    """An immutable address into a media file.

    Holds at most one of a frame ordinal, a time point, or a time range
    (decision 8: the parser never converts between these, so they are kept
    as separate fields rather than one ambiguous "point" field). A bare
    path with none of the three set means "the whole file", which decision
    4 treats as the range covering it.

    Equality and hashing are defined on the canonical stored string
    (decision 9), not on the fields directly -- see __eq__ below. This is
    what makes two spellings of the same address collapse to one cache
    entry, and what keeps a frame address and a time address that happen
    to resolve to the same frame permanently distinct (decision 8).

    The only supported way to build one from a filesystem path is
    from_path(). Do not construct MediaAddress by joining strings.
    """

    path: str
    stream: Optional[StreamSelector] = None
    frame: Optional[int] = None
    time_us: Optional[int] = None
    time_range_us: Optional[Tuple[int, int]] = None
    region: Optional[Region] = None

    def __post_init__(self):
        # At most one of frame / time_us / time_range_us may be set --
        # a point and a range, or two kinds of point, cannot both apply.
        point_fields = [self.frame, self.time_us, self.time_range_us]
        if sum(f is not None for f in point_fields) > 1:
            raise MediaAddressError(
                "an address holds at most one of a frame ordinal, a time "
                "point, or a time range"
            )

        # This is the single source of truth for these invariants, guarding
        # direct construction (dataclasses.replace(), tests) exactly as it
        # guards parse(). parse() rejects a negative token earlier, with a
        # string-specific message, but never on a different invariant than
        # the one enforced here -- so there is one rule, checked here, with
        # an earlier, friendlier error message where the string is available.
        if self.frame is not None and self.frame < 0:
            raise MediaAddressError("frame ordinal cannot be negative")
        if self.time_us is not None and self.time_us < 0:
            raise MediaAddressError("time cannot be negative")
        if self.time_range_us is not None:
            start_us, end_us = self.time_range_us
            if start_us < 0 or end_us < 0:
                raise MediaAddressError("a time range cannot hold a negative time")
            # Decision 11: a reversed or zero-length range is refused, never
            # reordered and never clamped.
            if end_us <= start_us:
                raise MediaAddressError(
                    f"a time range must not be reversed or empty: "
                    f"{start_us} to {end_us} microseconds"
                )

    def __eq__(self, other):
        if not isinstance(other, MediaAddress):
            return NotImplemented
        return format(self) == format(other)

    def __hash__(self):
        return hash(format(self))


# ---------------------------------------------------------------------------
# Path escaping (decision 1).
# ---------------------------------------------------------------------------
#
# Inside the path portion, '%' is written '%25' and '#' is written '%23'.
# Nothing else is escaped -- '&', ',' and '-' only carry meaning after the
# fragment starts, so a path containing them needs no treatment.
#
# Encode order matters: '%' must be escaped first, so the '%' introduced
# by escaping '#' is not itself re-escaped. Decode reverses that order.
# str.replace() does a single left-to-right non-overlapping pass and never
# rescans its own output, which is what makes this pair of passes exactly
# reversible.

def _escape_path(path: str) -> str:
    escaped = path.replace("%", "%25")
    escaped = escaped.replace("#", "%23")
    return escaped


def _unescape_path(escaped: str) -> str:
    unescaped = escaped.replace("%23", "#")
    unescaped = unescaped.replace("%25", "%")
    return unescaped


def _to_posix(path) -> str:
    """Normalise a path string to forward slashes (decision 9), using
    Windows path syntax rules -- backslash separators, drive letters, UNC
    -- regardless of which OS this process happens to be running on.

    Gelem is a Windows desktop application, so its paths are Windows paths
    even when a test runs on a non-Windows machine. Using
    `pathlib.PureWindowsPath` explicitly, rather than `pathlib.PurePath`
    (which resolves to whatever the *host* OS is), is what makes this
    deterministic on any machine instead of silently depending on where
    the process happens to run.
    """
    return pathlib.PureWindowsPath(path).as_posix()


def _normalise_path_portion(path: str) -> str:
    """Apply decision 9's forward-slash normalisation to a path string,
    while leaving an empty string empty.

    `pathlib.PureWindowsPath("")` is `.`, which is not what an empty path
    portion means -- an address like `#f=1` with no path, or a genuinely
    blank media cell, must stay blank rather than turning into the current
    directory.
    """
    if path == "":
        return ""
    return _to_posix(path)


def from_path(path) -> MediaAddress:
    """Turn a filesystem path into a bare-path MediaAddress.

    This is the only supported way to do so. It normalises the separator
    to forward slashes (decision 9's canonical form), so drive letters and
    UNC paths survive intact.
    """
    return MediaAddress(path=_normalise_path_portion(str(path)))


# ---------------------------------------------------------------------------
# Numeric token parsing. No floating point anywhere below.
# ---------------------------------------------------------------------------

_INT_TOKEN = re.compile(r"^(0|[1-9][0-9]*)$")
_DECIMAL_TOKEN = re.compile(r"^(\d+)(?:\.(\d+))?$")


def _parse_int_token(value: str, what: str) -> int:
    """Parse a plain non-negative integer with no sign, no leading zero."""
    if value.startswith("-"):
        raise MediaAddressError(f"{what} cannot be negative: {value!r}")
    if not _INT_TOKEN.match(value):
        raise MediaAddressError(f"{what} is not a plain integer: {value!r}")
    return int(value)


def _parse_decimal_micros(value: str, what: str) -> int:
    """Parse a non-negative decimal string into exact integer millionths.

    More than six digits after the decimal point is refused rather than
    rounded (decision 9) -- rounding a time silently is exactly the
    failure this section exists to prevent.
    """
    if value.startswith("-"):
        raise MediaAddressError(f"{what} cannot be negative: {value!r}")
    match = _DECIMAL_TOKEN.match(value)
    if not match:
        raise MediaAddressError(f"{what} is not a valid decimal value: {value!r}")
    whole_part, frac_part = match.group(1), match.group(2) or ""
    if len(frac_part) > 6:
        raise MediaAddressError(
            f"{what} has more than six decimal places, which would need "
            f"rounding rather than an exact value: {value!r}"
        )
    frac_part = frac_part.ljust(6, "0")
    return int(whole_part) * 1_000_000 + int(frac_part)


def _format_micros(value: int) -> str:
    """Format an integer-millionths value with exactly six decimal places."""
    whole, frac = divmod(value, 1_000_000)
    return f"{whole}.{frac:06d}"


# ---------------------------------------------------------------------------
# parse() / format() -- the grammar, and decision 9's canonical form.
# ---------------------------------------------------------------------------

def parse(address_string: str) -> MediaAddress:
    """Parse an address string into a MediaAddress.

    Accepts any order of the fragment's components -- format() always
    writes them back in canonical order (decision 9) -- so two spellings
    of one address parse to equal MediaAddress values.
    """
    if "#" not in address_string:
        # Unescape first (decision 1), then apply decision 9's forward-slash
        # normalisation to the path portion -- exactly what from_path() does,
        # so parse(format(a)) == a holds for any address however it was built,
        # including one whose path was spelled with backslashes.
        return MediaAddress(path=_normalise_path_portion(_unescape_path(address_string)))

    # The first literal '#' is always the fragment delimiter: any '#' that
    # was part of the original path has already been escaped to '%23' by
    # from_path() before this string was ever built (decision 1).
    raw_path, fragment = address_string.split("#", 1)
    # Same unescape-then-normalise the no-fragment branch does.
    path = _normalise_path_portion(_unescape_path(raw_path))

    if fragment == "":
        raise MediaAddressError("an empty fragment ('#' with nothing after it) is an error")

    stream: Optional[StreamSelector] = None
    frame: Optional[int] = None
    time_us: Optional[int] = None
    time_range_us: Optional[Tuple[int, int]] = None
    region: Optional[Region] = None
    seen = set()

    for component in fragment.split("&"):
        if "=" not in component:
            raise MediaAddressError(f"malformed address component: {component!r}")
        key, value = component.split("=", 1)

        if key in ("v", "a"):
            if "stream" in seen:
                raise MediaAddressError("an address may name only one stream selector")
            seen.add("stream")
            stream = StreamSelector(kind=key, index=_parse_int_token(value, "stream index"))

        elif key == "f":
            if "point" in seen:
                raise MediaAddressError("an address may hold only one of f= or t=")
            seen.add("point")
            frame = _parse_int_token(value, "frame ordinal")

        elif key == "t":
            if "point" in seen:
                raise MediaAddressError("an address may hold only one of f= or t=")
            seen.add("point")
            # A '-' is a range separator only past the first character --
            # a leading '-' is a negative sign, refused as negative below
            # rather than misread as an empty range start.
            separator_index = value.find("-", 1)
            if separator_index != -1:
                start_str, end_str = value[:separator_index], value[separator_index + 1:]
                start_us = _parse_decimal_micros(start_str, "range start")
                end_us = _parse_decimal_micros(end_str, "range end")
                # Reversed or zero-length is refused below, by
                # MediaAddress.__post_init__ -- the single source of truth
                # for that rule, not re-checked here.
                time_range_us = (start_us, end_us)
            else:
                time_us = _parse_decimal_micros(value, "time")

        elif key == "r":
            if "region" in seen:
                raise MediaAddressError("an address may name only one region")
            seen.add("region")
            region = _parse_region(value)

        else:
            raise MediaAddressError(f"unknown address component {key!r}={value!r}")

    return MediaAddress(
        path=path,
        stream=stream,
        frame=frame,
        time_us=time_us,
        time_range_us=time_range_us,
        region=region,
    )


def _parse_region(value: str) -> Region:
    parts = value.split(",")
    if len(parts) != 4:
        raise MediaAddressError(
            f"a region needs exactly four comma-separated values: {value!r}"
        )
    micros = [_parse_decimal_micros(part, "region value") for part in parts]
    x, y, w, h = micros
    # Region's own __post_init__ carries out the rest of decision 5 and 11's
    # region validation (bounds, positive area, x+w and y+h within 1).
    return Region(x=x, y=y, w=w, h=h)


# The default stream selector, per decision 7: naming the lowest-index
# video stream explicitly is the same address as naming no stream at all,
# so it is the one selector value the canonical form elides.
_DEFAULT_STREAM = StreamSelector(kind="v", index=0)


def format(addr: MediaAddress) -> str:
    """Format a MediaAddress back to its canonical stored string.

    Component order is fixed (decision 9): stream selector, then f= or
    t=, then r=. The path keeps whatever form addr.path holds -- this is
    the *stored* form and may be project-relative; canonical_key() below
    produces the resolved *key* form used for the artifact cache.
    """
    parts: List[str] = []

    if addr.stream is not None and addr.stream != _DEFAULT_STREAM:
        parts.append(f"{addr.stream.kind}={addr.stream.index}")

    if addr.frame is not None:
        parts.append(f"f={addr.frame}")
    elif addr.time_us is not None:
        parts.append(f"t={_format_micros(addr.time_us)}")
    elif addr.time_range_us is not None:
        start_us, end_us = addr.time_range_us
        parts.append(f"t={_format_micros(start_us)}-{_format_micros(end_us)}")

    if addr.region is not None:
        r = addr.region
        parts.append(
            "r=" + ",".join(_format_micros(v) for v in (r.x, r.y, r.w, r.h))
        )

    escaped_path = _escape_path(addr.path)
    if not parts:
        return escaped_path
    return escaped_path + "#" + "&".join(parts)


# ---------------------------------------------------------------------------
# absolutise() / relativise() -- swapping the path portion against a base.
#
# Both are pure and take their base as an argument: nothing here reads
# Path.cwd() or the environment. models/dataset.py is where an ambient
# directory (the working directory, the project root) is read and passed
# in -- see decision 1's "no code builds an address by joining strings" and
# the P0.2c work item.
#
# Both swap the path portion with dataclasses.replace and never touch the
# fragment (frame ordinal, time, range, region, stream selector).
# ---------------------------------------------------------------------------

_DRIVE_LETTER_PATH = re.compile(r"^[A-Za-z]:/")


def _is_absolute_posix_style(path: str) -> bool:
    """True for a posix-style absolute path, a UNC path (spelled '//server/
    share/...' after normalisation), or a drive-letter path such as
    'C:/Users/x'. All three are forms an address path may hold, since
    normalisation fixes the separator but not the path's OS flavour.
    """
    if path.startswith("/"):
        return True
    if _DRIVE_LETTER_PATH.match(path):
        return True
    return False


def absolutise(addr: MediaAddress, base: str) -> MediaAddress:
    """Return `addr` with its path portion made absolute against `base`.

    If the path is already absolute (a POSIX root, a UNC path, or a
    drive-letter path), `addr` is returned unchanged. Otherwise the path is
    resolved against `base` and a copy is returned. The path portion is
    normalised to forward slashes first, defensively, in case the caller
    built the address by hand rather than through parse() or from_path().
    """
    # Defensive normalisation -- parse()/from_path() already do this, but a
    # hand-built MediaAddress might carry backslashes.
    posix_path = _normalise_path_portion(addr.path)

    if _is_absolute_posix_style(posix_path):
        # Already absolute. Only spend a dataclasses.replace if the
        # normalisation actually changed the spelling.
        if posix_path == addr.path:
            return addr
        return dataclasses.replace(addr, path=posix_path)

    # Relative: join onto the normalised base and collapse any '..'.
    root = _normalise_path_portion(base)
    joined = posixpath.join(root, posix_path)
    resolved = posixpath.normpath(joined).replace("\\", "/")
    return dataclasses.replace(addr, path=resolved)


def relativise(addr: MediaAddress, project_root: str) -> MediaAddress:
    """Return `addr` with its path portion made relative to `project_root`
    when it lies inside it; otherwise return `addr` unchanged.

    The containment test is `PureWindowsPath.relative_to`, which raises for
    anything that is not a subpath -- a sibling directory, a parent, or a
    different drive letter. That raise IS CLAUDE.md's [NOW] rule "only paths
    inside the project folder become relative": relative_to decides, this
    function does not re-implement the check.

    Note: PureWindowsPath comparison is case-insensitive, so a cell spelled
    'c:/proj/x.mp4' under project root 'C:/proj' relativises, and on the way
    back through absolutise() comes back carrying the project's case
    ('C:/proj/x.mp4'). This is intentional -- it makes two spellings of one
    in-project file converge on a single artifact-cache key. It does not
    conflict with decision 9's "case is not folded", which is about the key
    form of paths that stay absolute (paths outside the project, which this
    function returns unchanged).
    """
    posix_path = _normalise_path_portion(addr.path)

    # A path with no root cannot be measured against project_root -- leave
    # it. (relative_to would treat it as relative to the anchor and could
    # succeed misleadingly.)
    if not _is_absolute_posix_style(posix_path):
        return addr

    try:
        # PureWindowsPath so drive letters and UNC roots are understood the
        # same way on every OS this test might run on, and so a different
        # drive raises here rather than silently joining.
        relative = pathlib.PureWindowsPath(posix_path).relative_to(
            pathlib.PureWindowsPath(project_root)
        )
    except ValueError:
        # Not a subpath of project_root -- a sibling, a parent, or a
        # different drive. The address stays absolute.
        return addr

    return dataclasses.replace(addr, path=relative.as_posix())


# ---------------------------------------------------------------------------
# canonical_key() -- the artifact-cache key form (decision 9).
# ---------------------------------------------------------------------------

def canonical_key(addr: MediaAddress, project_root: str) -> str:
    """The artifact-cache key form: like format(), but with the path
    resolved to absolute against project_root. Case is not folded
    (decision 9) -- two spellings differing only in case hash differently,
    deliberately, in exchange for never risking a wrong picture.

    Expressed as format(absolutise(...)) so there is exactly one definition
    of "make absolute", shared with models/dataset.py's save/load path.
    """
    return format(absolutise(addr, project_root))


@functools.lru_cache(maxsize=8192)
def resolve_source(cell: str, project_root: str) -> Tuple[str, str]:
    """For a stored media cell, return
    ``(canonical_key_string, absolute_source_path)``.

    Memoised: this is pure (no filesystem access -- absolutise() is
    string arithmetic) and the display path calls it once per media tile
    render. The cache is bounded; a project switch just leaves stale
    entries to be evicted.

    The display path (``column_types/renderers.py``) needs both: the
    canonical key form to look an artifact up in the cache, and the
    absolute filesystem path a decoder opens. Providing them together
    here keeps renderers free of a project root and of any address
    parsing of their own -- the controller calls this once per media cell
    and drops both values into the render context.

    The canonical key is exactly ``canonical_key(parse(cell),
    project_root)``; the path is that same absolutised address's path
    portion, so the two never disagree about which file is meant.
    """
    addr = absolutise(parse(cell), project_root)
    return format(addr), addr.path


# ---------------------------------------------------------------------------
# select_frame() / frames_in_range() -- the semantic core (decisions 2, 3,
# 4, 8, 11). Given a stream's frame presentation times, pure arithmetic --
# no file is read here, which is what lets these rules be tested without a
# decoder and is exactly what the resolver (P1.2) will call once it has
# real timings.
# ---------------------------------------------------------------------------

_POLICIES = ("first", "midpoint")

# Used only when a single-frame stream gives no real gap to measure from
# (see _estimated_stream_end_us below). Deliberately not a small, plausible
# guess -- CLAUDE.md's generality rule says a number with no principled
# basis must not become a hardcoded constant pretending to be one, so this
# is instead an unmistakably-arbitrary "treat as open-ended" sentinel: about
# thirty-one thousand years in microseconds.
_UNKNOWN_STREAM_LENGTH_US = 10**15


def _estimated_stream_end_us(frame_times: Sequence[int]) -> int:
    """An estimated end-of-stream time, one frame past the last known one.

    Decision 2 says the final frame's interval runs to the true end of
    the stream, which this pure-logic function is never told -- only the
    resolver knows the real file duration. This estimate (the last frame's
    own gap, projected forward once) is used only to give the whole-file
    range (decision 4) a concrete end, and to catch a query time that is
    unambiguously beyond every known frame. It is not a claim about the
    real file duration.
    """
    if len(frame_times) == 0:
        raise MediaAddressError("frame_times must not be empty")
    if len(frame_times) == 1:
        # No second frame to measure a real gap from. Guessing a duration
        # here would be exactly the kind of made-up per-file number
        # CLAUDE.md's generality rule forbids, so this is treated as
        # open-ended instead of bounded by an arbitrary guess.
        return frame_times[0] + _UNKNOWN_STREAM_LENGTH_US
    last_gap = frame_times[-1] - frame_times[-2]
    return frame_times[-1] + last_gap


def _frame_at_time(frame_times: Sequence[int], target_us: int) -> int:
    """The frame whose presentation interval contains target_us (decision
    2): interval i is [frame_times[i], frame_times[i+1]), except the last
    frame's interval, which this function treats as open-ended.
    """
    index = bisect.bisect_right(frame_times, target_us) - 1
    if index < 0:
        raise MediaAddressError(
            f"time {target_us} microseconds is before the first frame"
        )
    return index


def _range_bounds_us(
    addr: MediaAddress,
    frame_times: Sequence[int],
    stream_end_us: int,
) -> Tuple[int, int]:
    if addr.time_range_us is not None:
        # An explicit range already carries its own end.
        return addr.time_range_us
    if addr.frame is not None or addr.time_us is not None:
        raise MediaAddressError("this address is a point, not a range")
    # A bare path is the range covering the whole file (decision 4), whose
    # end is the same estimate select_frame()/frames_in_range() already
    # computed once, rather than a second, independent guess.
    return frame_times[0], stream_end_us


def _member_bounds(
    addr: MediaAddress,
    frame_times: Sequence[int],
    stream_end_us: int,
) -> Tuple[int, int, int, int]:
    """The range's own (start_us, end_us) together with the [lo, hi) index
    range into frame_times of the frames it actually contains (decision 3:
    a <= p < b). Shared by select_frame() and frames_in_range() so there
    is one definition of range membership, not two that could drift apart.
    """
    start_us, end_us = _range_bounds_us(addr, frame_times, stream_end_us)
    if start_us > stream_end_us:
        raise MediaAddressError(
            f"range start {start_us} microseconds is beyond the end of the stream"
        )
    lo = bisect.bisect_left(frame_times, start_us)
    hi = bisect.bisect_left(frame_times, end_us)
    return start_us, end_us, lo, hi


def _nearest_member(
    frame_times: Sequence[int],
    lo: int,
    hi: int,
    target_us: int,
) -> int:
    """The index in [lo, hi) whose presentation time is nearest target_us,
    ties going to the earlier frame. Caller guarantees hi > lo.
    """
    pos = bisect.bisect_left(frame_times, target_us, lo, hi)
    if pos <= lo:
        return lo
    if pos >= hi:
        return hi - 1
    earlier, later = pos - 1, pos
    distance_to_earlier = target_us - frame_times[earlier]
    distance_to_later = frame_times[later] - target_us
    return earlier if distance_to_earlier <= distance_to_later else later


def select_frame(
    addr: MediaAddress,
    frame_times: Sequence[int],
    policy: str = "first",
) -> int:
    """Return the single frame index this address selects.

    frame_times holds a stream's frame presentation times, in ascending
    presentation order, as exact integer microseconds.

    - A frame-ordinal address returns that ordinal directly, unconverted
      (decision 8) -- only checked against the frame count.
    - A time-point address returns the frame whose interval contains it
      (decision 2).
    - A range address (or a bare path, treated as the whole-file range
      per decision 4) returns the representative frame chosen by
      `policy`, chosen only from the frames the range actually contains
      under decision 3 (never a frame outside it, even one whose display
      interval overlaps the range's start): 'first' is the earliest frame
      in the range; 'midpoint' is the frame in the range nearest to
      `start + (end - start) / 2`, ties going to the earlier frame. A
      range containing no frame at all raises (decision 11) rather than
      falling back to a frame outside it or returning nothing silently.
    """
    if policy not in _POLICIES:
        raise MediaAddressError(f"unknown representative-frame policy: {policy!r}")

    if addr.frame is not None:
        # Decision 8: never converted through frame_times. Only its bound
        # is checked (decision 11: a frame ordinal beyond the last frame).
        if addr.frame >= len(frame_times):
            raise MediaAddressError(
                f"frame ordinal {addr.frame} is beyond the last frame "
                f"({len(frame_times)} frames in this stream)"
            )
        return addr.frame

    stream_end_us = _estimated_stream_end_us(frame_times)

    if addr.time_us is not None:
        if addr.time_us > stream_end_us:
            raise MediaAddressError(
                f"time {addr.time_us} microseconds is beyond the end of the stream"
            )
        return _frame_at_time(frame_times, addr.time_us)

    start_us, end_us, lo, hi = _member_bounds(addr, frame_times, stream_end_us)
    if lo >= hi:
        raise MediaAddressError(
            f"the range {start_us} to {end_us} microseconds contains no frames"
        )
    if policy == "first":
        return lo
    target_us = start_us + (end_us - start_us) // 2
    return _nearest_member(frame_times, lo, hi, target_us)


def frames_in_range(addr: MediaAddress, frame_times: Sequence[int]) -> List[int]:
    """Return every frame index whose presentation time p satisfies the
    address's half-open range a <= p < b (decision 3).

    A narrow range that captured no frame of its own returns an empty
    list here -- the correct answer to "which frames were captured in
    this window". select_frame() above answers a related but different
    question (which single frame represents this range for display) and,
    since decision 4's fix, only ever returns a member of this same set;
    it raises rather than returning anything for a range this function
    reports as empty.

    Not one of the four functions named in the P0.3 brief, added
    alongside select_frame because decision 3's tiling property --
    adjacent ranges share no frames and cover their union exactly -- is a
    membership question, not a single-representative-frame question, and
    needs its own query to test directly.

    Valid for a range address, or a bare path (the whole-file range,
    decision 4). Raises for a point address, and for a range whose start
    is beyond the end of the stream (decision 11), exactly as
    select_frame() does for the same address.
    """
    stream_end_us = _estimated_stream_end_us(frame_times)
    _, _, lo, hi = _member_bounds(addr, frame_times, stream_end_us)
    return list(range(lo, hi))


# ---------------------------------------------------------------------------
# region_to_pixels() -- decision 5's edge-rounding rule.
# ---------------------------------------------------------------------------

def region_to_pixels(region: Region, width: int, height: int) -> PixelRegion:
    """Convert a normalised region to whole pixels against a WxH frame.

    Rounds the four edges, not the origin and size separately: the left
    edge is round(x*W), the right edge is round((x+w)*W), and the pixel
    width is their difference. This is what makes two adjacent regions
    rejoin with no seam and no overlap, which computing width and height
    as independently-rounded fractions would not guarantee.

    Uses exact rational arithmetic (Fraction), not floating point, so the
    rounding matches the millionths stored in `region` exactly.
    """
    if width <= 0 or height <= 0:
        raise MediaAddressError("frame width and height must be positive")

    left = _round_edge(region.x, width)
    right = _round_edge(region.x + region.w, width)
    top = _round_edge(region.y, height)
    bottom = _round_edge(region.y + region.h, height)

    # A Region always has positive area in normalised space
    # (Region.__post_init__), but at a low enough target resolution its
    # edges can round to the same pixel column or row, producing a
    # rectangle with zero pixels in it -- exactly the "plausible but
    # wrong picture" failure this whole module exists to prevent, so it
    # is refused here rather than handed back as a usable-looking result.
    if right <= left or bottom <= top:
        raise MediaAddressError(
            f"region rounds to a zero-pixel rectangle at {width}x{height}: "
            f"left={left}, right={right}, top={top}, bottom={bottom}"
        )

    return PixelRegion(left=left, top=top, width=right - left, height=bottom - top)


def _round_edge(micros: int, dimension: int) -> int:
    return round(Fraction(micros, 1_000_000) * dimension)
