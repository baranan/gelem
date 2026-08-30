"""
artifacts/artifact_codec.py

ArtifactCodec -- the one place that reads and writes Gelem's own derived
JPEG cache files.

`CLAUDE.md`'s media rules draw a line between two operations that both
"open an image":

  * decoding a *user's* media file -- the resolver's job (P1.2)
  * encoding and reading back Gelem's *own* cached thumbnails -- this

A blanket ban on `Image.open` outside the resolver would either forbid
legitimate cache I/O or push the cache internals into the resolver.
ArtifactCodec is the narrow boundary that lets a guardrail name the two
owners precisely.

The boundary is enforced by BEHAVIOUR, not by source inspection: every
path handed to the codec must resolve to a location under the cache root,
or the codec raises `ArtifactCodecError`. A test hands it a source-media
path and asserts the raise. A source-inspection test that named a
function would die the moment that function was split;
`tests/test_controller_async_contracts.py` was nearly bitten by exactly
that.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

# Quality of the JPEGs we write. Matches the value ArtifactStore used
# inline before the codec existed, so cached pictures do not change.
_JPEG_QUALITY = 85


class ArtifactCodecError(RuntimeError):
    """Raised when the codec is asked to touch a path outside the cache
    root -- the signal that source-media decoding has leaked into the
    artifact path, or vice versa."""


class ArtifactCodec:
    """Reads and writes derived JPEGs, and only under one directory."""

    def __init__(self, cache_root: Path):
        # resolve() once at construction so every later containment check
        # compares fully-resolved paths and a symlinked or '..'-laden
        # argument cannot sidestep the boundary.
        self._root = Path(cache_root).resolve()

    def _require_inside(self, path: Path) -> Path:
        """Return `path` resolved, or raise if it is not under the root."""
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError:
            raise ArtifactCodecError(
                f"ArtifactCodec refuses a path outside the artifact cache "
                f"root.\n  root: {self._root}\n  path: {resolved}\n"
                f"Source media is decoded by the resolver, not here."
            )
        return resolved

    def write_jpeg(self, dest: Path, image: np.ndarray) -> Path:
        """Encode an RGB uint8 array as a JPEG under the cache root.

        Written to a unique temp file and then atomically moved into
        place with os.replace. Two worker threads generating a thumbnail
        for the same media file compute the same ArtifactKey and so the
        same `dest` (P0.5b-1 does not coalesce duplicate requests -- that
        is P0.5b-2); without the temp-then-replace they would write the
        same path concurrently and could leave a truncated JPEG.
        """
        dest = self._require_inside(dest)
        tmp = dest.with_name(
            f"{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        moved = False
        try:
            Image.fromarray(image).save(tmp, "JPEG", quality=_JPEG_QUALITY)
            os.replace(tmp, dest)
            moved = True
        finally:
            # os.replace consumed tmp on success; only clean up when it
            # did not run (encode failed, disk full, permissions).
            if not moved:
                try:
                    tmp.unlink()
                except OSError:
                    pass
        return dest

    def read_image(self, path: Path) -> Image.Image:
        """Load a cached JPEG. Fully read before returning, so the caller
        never holds a lazy handle to a file the cache may evict."""
        path = self._require_inside(path)
        image = Image.open(path)
        image.load()
        return image
