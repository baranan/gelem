"""
media/artifact_key.py

ArtifactKey -- what a derived image *is*, independent of who asked for it.

`docs/media_architecture.md` section 4.5 is the authority. The current
ArtifactStore keys its cache on ``(row_id, artifact_type)``, which names
the UI subscriber, not the picture: two media columns on one row collide,
the same row_id in two tables collides, and a second project reuses the
ids. The correct key is what the picture is made of:

    canonical media address   (media_address.canonical_key)
  + source fingerprint        (size and mtime)
  + purpose                   ('thumbnail' | 'preview')
  + requested resolution      (max side, in pixels)
  + representative-frame policy   (media_address.POLICIES)
  + renderer cache version    (bumped by hand when our output changes)

The table, row and column identify the UI subscriber waiting for the
result -- who to repaint when the picture arrives -- and are deliberately
absent here.

This module is a pure value object. Like `models/query_result.py` and
`models/notifications.py` it imports no Qt, pandas, numpy, PIL or cv2, so
UI files and worker threads may both import it.
"""

from __future__ import annotations

import dataclasses
import hashlib

from media.media_address import POLICIES

# Bumped BY HAND whenever a change to how Gelem renders a thumbnail or
# preview could alter the pixels it produces (a different resize filter, a
# colour-space fix, a new border). An old cached picture then simply stops
# being reused -- it is not overwritten and not served stale.
RENDERER_CACHE_VERSION = 1

# The two purposes ArtifactStore produces today. 'preview' is the larger
# variant a big tile asks for; 'thumbnail' is the small one.
PURPOSES = ("thumbnail", "preview")


@dataclasses.dataclass(frozen=True)
class SourceFingerprint:
    """Just enough of the source file to notice it changed under a stable
    path: its size in bytes and its modification time in nanoseconds.

    `docs/media_architecture.md` section 4.5 allows "a stronger hash where
    needed"; size-and-mtime is the cheap default and is all P0.5b-1 uses.
    """

    size: int
    mtime_ns: int


@dataclasses.dataclass(frozen=True)
class ArtifactKey:
    """The immutable identity of one derived image.

    Frozen and hashable, so it works directly as a dict key in
    ArtifactStore's index and memory cache. `stable_hash()` gives the
    on-disk filename -- a hash rather than the fields themselves because
    the canonical address contains a filesystem path.
    """

    canonical_address: str
    fingerprint: SourceFingerprint
    purpose: str
    resolution: int
    policy: str = "first"
    renderer_version: int = RENDERER_CACHE_VERSION

    def __post_init__(self) -> None:
        if self.purpose not in PURPOSES:
            raise ValueError(
                f"artifact purpose must be one of {PURPOSES}, got {self.purpose!r}"
            )
        if self.policy not in POLICIES:
            raise ValueError(
                f"representative-frame policy must be one of {POLICIES}, "
                f"got {self.policy!r}"
            )
        if self.resolution <= 0:
            raise ValueError(
                f"requested resolution must be a positive pixel count, "
                f"got {self.resolution!r}"
            )

    def stable_hash(self) -> str:
        """A stable, process-independent hex digest of every field.

        Used as the artifact's filename on disk. Uses hashlib rather than
        the builtin hash(), which is salted per process and would give a
        different filename every run.
        """
        parts = (
            self.canonical_address,
            str(self.fingerprint.size),
            str(self.fingerprint.mtime_ns),
            self.purpose,
            str(self.resolution),
            self.policy,
            str(self.renderer_version),
        )
        # \x1f (unit separator) cannot appear in any of these fields, so
        # the join is unambiguous.
        joined = "\x1f".join(parts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]
