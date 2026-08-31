"""
settings/settings.py

GelemSettings -- the frozen set of machine-tunable values, with their
module-level defaults and bounds, and a tolerant constructor that never
raises on a bad saved value.

A corrupt or out-of-range persisted value must never stop the app
starting. `from_values()` therefore clamps an out-of-range number, falls
back to the default for an unparseable one, and returns a list of
plain-English problem messages describing every correction it made. The
caller (main.py) prints those to the console and carries on.

docs/architecture.md section 9 is the authority for what each value means
and when it takes effect. This module only owns the numbers and the
validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


# ---------------------------------------------------------------------------
# Defaults and bounds. These numbers live here and are referenced, never
# copied, everywhere else.
# ---------------------------------------------------------------------------

# In-memory decoded-image cache ceiling, in bytes. 500 MiB default; 16 MiB
# to 64 GiB.
DEFAULT_PICTURE_MEMORY_MAX_BYTES = 500 * 1024 * 1024
PICTURE_MEMORY_MAX_BYTES_RANGE = (16 * 1024 * 1024, 64 * 1024 * 1024 * 1024)

# On-disk derived-JPEG cache ceiling, in bytes. 1 GiB default; 64 MiB to
# 1 TiB.
DEFAULT_PICTURE_DISK_MAX_BYTES = 1024 * 1024 * 1024
PICTURE_DISK_MAX_BYTES_RANGE = (64 * 1024 * 1024, 1024 * 1024 * 1024 * 1024)

# Number of background threads the ArtifactStore uses to decode and resize
# source media. 2 default; 1 to 32.
DEFAULT_WORKER_COUNT = 2
WORKER_COUNT_RANGE = (1, 32)

# Thumbnail target size, (width, height) in pixels. Each side 32 to 1024.
DEFAULT_THUMBNAIL_SIZE = (150, 150)
THUMBNAIL_SIDE_RANGE = (32, 1024)

# Preview target size, (width, height) in pixels. Each side 64 to 4096.
DEFAULT_PREVIEW_SIZE = (600, 600)
PREVIEW_SIDE_RANGE = (64, 4096)


# ---------------------------------------------------------------------------
# Tolerant parsing helpers. Each appends at most one problem message and
# never raises.
# ---------------------------------------------------------------------------

def _parse_int_field(
    raw,
    default: int,
    bounds: tuple[int, int],
    label: str,
    problems: list[str],
) -> int:
    """Parse `raw` as an integer and clamp it into `bounds`.

    An absent value (None) is not a problem -- it just means "use the
    default". An unparseable value falls back to the default and records
    one problem. An out-of-range value is clamped and records one problem.
    """
    low, high = bounds

    # Absent -- the common case for a first run. No message.
    if raw is None:
        return default

    # Unparseable -- fall back to the default and say so.
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        problems.append(
            f"Could not read the {label} setting (saved value {raw!r}); "
            f"using the default of {default}."
        )
        return default

    # Out of range -- clamp to the nearer bound and say so.
    if value < low:
        problems.append(
            f"The {label} setting ({value}) is below the minimum of {low}; "
            f"using {low}."
        )
        return low
    if value > high:
        problems.append(
            f"The {label} setting ({value}) is above the maximum of {high}; "
            f"using {high}."
        )
        return high

    return value


def _parse_size_field(
    raw,
    default: tuple[int, int],
    side_bounds: tuple[int, int],
    label: str,
    problems: list[str],
) -> tuple[int, int]:
    """Parse `raw` as a (width, height) pixel size and clamp each side
    into `side_bounds`.

    `raw` may be the persisted "WxH" string, or an already-parsed
    (w, h) sequence. Absent (None) means "use the default", with no
    message. Anything unparseable falls back to the default with one
    problem. If either side is clamped, one problem is recorded.
    """
    low, high = side_bounds

    if raw is None:
        return default

    # Accept either "150x150" or a 2-element sequence.
    width_raw = None
    height_raw = None
    if isinstance(raw, str):
        parts = raw.lower().split("x")
        if len(parts) == 2:
            width_raw, height_raw = parts[0].strip(), parts[1].strip()
    else:
        try:
            width_raw, height_raw = raw
        except (TypeError, ValueError):
            width_raw = height_raw = None

    try:
        if width_raw is None or height_raw is None:
            raise ValueError
        width = int(width_raw)
        height = int(height_raw)
    except (TypeError, ValueError):
        problems.append(
            f"Could not read the {label} setting (saved value {raw!r}); "
            f"using the default of {default[0]}x{default[1]}."
        )
        return default

    # Clamp each side independently.
    clamped_width = min(max(width, low), high)
    clamped_height = min(max(height, low), high)
    if (clamped_width, clamped_height) != (width, height):
        problems.append(
            f"The {label} setting ({width}x{height}) has a side outside "
            f"{low}..{high}; using {clamped_width}x{clamped_height}."
        )
    return (clamped_width, clamped_height)


# ---------------------------------------------------------------------------
# The settings object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GelemSettings:
    """The machine-tunable values, all validated. Frozen: build a new one
    rather than mutating."""

    picture_memory_max_bytes: int = DEFAULT_PICTURE_MEMORY_MAX_BYTES
    picture_disk_max_bytes: int = DEFAULT_PICTURE_DISK_MAX_BYTES
    worker_count: int = DEFAULT_WORKER_COUNT
    thumbnail_size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE
    preview_size: tuple[int, int] = DEFAULT_PREVIEW_SIZE

    @classmethod
    def from_values(
        cls, mapping: Mapping
    ) -> tuple["GelemSettings", list[str]]:
        """Build a GelemSettings from a mapping of field name -> raw value.

        Never raises. `mapping` keys are the field names above; a missing
        key, a None value, an unparseable value and an out-of-range value
        are all handled -- each correction appends one plain-English
        message to the returned list. A corrupt saved value must never
        stop the app starting.

        One cross-field rule: if the preview's larger side is smaller than
        the thumbnail's larger side, the preview size is set to the
        thumbnail size and that is reported.
        """
        problems: list[str] = []

        picture_memory_max_bytes = _parse_int_field(
            mapping.get("picture_memory_max_bytes"),
            DEFAULT_PICTURE_MEMORY_MAX_BYTES,
            PICTURE_MEMORY_MAX_BYTES_RANGE,
            "picture memory limit",
            problems,
        )
        picture_disk_max_bytes = _parse_int_field(
            mapping.get("picture_disk_max_bytes"),
            DEFAULT_PICTURE_DISK_MAX_BYTES,
            PICTURE_DISK_MAX_BYTES_RANGE,
            "picture disk limit",
            problems,
        )
        worker_count = _parse_int_field(
            mapping.get("worker_count"),
            DEFAULT_WORKER_COUNT,
            WORKER_COUNT_RANGE,
            "worker count",
            problems,
        )
        thumbnail_size = _parse_size_field(
            mapping.get("thumbnail_size"),
            DEFAULT_THUMBNAIL_SIZE,
            THUMBNAIL_SIDE_RANGE,
            "thumbnail size",
            problems,
        )
        preview_size = _parse_size_field(
            mapping.get("preview_size"),
            DEFAULT_PREVIEW_SIZE,
            PREVIEW_SIDE_RANGE,
            "preview size",
            problems,
        )

        # Cross-field rule: a preview must not be smaller than a thumbnail.
        # Compare on the larger side of each, since that is what enters the
        # artifact key as the resolution.
        if max(preview_size) < max(thumbnail_size):
            problems.append(
                f"The preview size ({preview_size[0]}x{preview_size[1]}) is "
                f"smaller than the thumbnail size "
                f"({thumbnail_size[0]}x{thumbnail_size[1]}); using the "
                f"thumbnail size for the preview as well."
            )
            preview_size = thumbnail_size

        return (
            cls(
                picture_memory_max_bytes=picture_memory_max_bytes,
                picture_disk_max_bytes=picture_disk_max_bytes,
                worker_count=worker_count,
                thumbnail_size=thumbnail_size,
                preview_size=preview_size,
            ),
            problems,
        )
