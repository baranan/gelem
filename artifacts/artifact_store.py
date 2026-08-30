"""
artifacts/artifact_store.py

ArtifactStore manages thumbnail and preview images used by the gallery
for fast display. It handles both image and video source files.

For images: thumbnails are generated using PIL.
For videos:  the first frame is extracted using OpenCV, then the same
             PIL-based resizing pipeline is applied.

Identity (P0.5b-1, docs/media_architecture.md section 4.5). A derived
image is identified by an `ArtifactKey` -- canonical media address, source
fingerprint, purpose, resolution, representative-frame policy and renderer
cache version -- NOT by the row that asked for it. The row, table and
column name the UI subscriber waiting for the picture; they are not part
of the key. This is what lets two media columns on one row, and two
tables pointing at one file, behave correctly.

Fingerprint memo (a design decision for P0.5b-1, not spelled out in 4.5).
The fingerprint is part of the key, but a cache lookup on the paint path
must not call `stat()`. So ArtifactStore keeps a memo of
`canonical address -> (size, mtime)`. A `stat()` only ever happens on the
request path (`refresh_fingerprint`, from `request_thumbnail`'s worker),
never in a lookup. An address in the memo is one of three states:

  * **absent** -- a lookup misses. Nothing is served.
  * **seeded-unverified** -- put there by `load_index` from the persisted
    (size, mtime). A lookup IS served from it, so a reopened project shows
    its cached pictures at once instead of decoding every visible source
    image on the main thread. The trade is a briefly stale preview if the
    source changed since the save; the next `request_thumbnail` for that
    address re-stats, gets a different fingerprint, and regenerates. For
    display (not analysis) that trade is deliberate.
  * **verified** -- put there by a fresh `refresh_fingerprint` this
    session. A lookup is served, and a duplicate `request_thumbnail`
    short-circuits without spawning a worker.

`load_index`'s docstring is the authority on the seeded-unverified case.

Reading and writing the JPEGs themselves goes through `ArtifactCodec`,
which is the boundary `CLAUDE.md`'s media rules name: derived artifacts
are encoded and read back only by the codec. The matching half -- source
media decoded only by the resolver, so nothing else opens an image at all
-- waits on P1.2; until then `_generate_thumbnails` still decodes source
media here.

This file is written centrally (not by a student).
"""

from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
import json
import threading

import numpy as np
from PIL import Image

from artifacts.artifact_codec import ArtifactCodec, ArtifactCodecError
from media.artifact_key import ArtifactKey, SourceFingerprint

# Import the extension sets from dataset so they stay in sync.
# We only need to know which extensions are videos here.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

THUMBNAIL_SIZE = (150, 150)
PREVIEW_SIZE   = (600, 600)

# The resolution (max side, in pixels) that enters the ArtifactKey for
# each purpose. Derived from the size tuples above rather than restated,
# so the key and the resize stay in step. THUMBNAIL_SIZE / PREVIEW_SIZE
# remain plain module constants in this diff -- turning them into
# machine-independent settings is P0.5b-2, together with worker count.
THUMBNAIL_RESOLUTION = max(THUMBNAIL_SIZE)
PREVIEW_RESOLUTION   = max(PREVIEW_SIZE)

# For now the store always records the first frame of a video (or the
# whole image) and ignores any frame or time specifier in the address.
# The key still carries the policy explicitly, so a later 'midpoint'
# policy (which needs real per-frame timings -- P1.2) produces a
# different key and does not collide with these pictures.
REPRESENTATIVE_FRAME_POLICY = "first"

DEFAULT_CACHE_MAX_BYTES = 500 * 1024 * 1024

# Bumped when the on-disk index layout changes. load_index() discards an
# index written under a different version rather than half-reading it.
INDEX_FORMAT_VERSION = 2


class ArtifactStore:
    """
    Stores and retrieves derived visual files for gallery display.

    Supports both image and video source files. For video files,
    thumbnails are generated from the first frame.
    """

    def __init__(self, artifacts_dir: Path):
        self._dir = artifacts_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._codec = ArtifactCodec(self._dir)

        # index and fingerprint memo are touched by worker threads and by
        # the main thread, so both live under _lock.
        self._index: dict[ArtifactKey, Path] = {}
        self._fingerprints: dict[str, SourceFingerprint] = {}
        # Addresses whose memo fingerprint came from a fresh stat this
        # session (refresh_fingerprint), as opposed to being seeded from a
        # persisted index by load_index (which may be stale). A request
        # for an unverified address always spawns a worker so the source
        # is re-stat-ed, rather than short-circuiting on a maybe-stale
        # fingerprint.
        self._verified: set[str] = set()
        self._lock = threading.Lock()

        # The in-memory LRU is populated only by get_pixmap(), on the main
        # thread, so it needs no lock. put() (worker thread) writes the
        # index and disk only; the first paint after generation reads the
        # JPEG back through the codec and fills this cache.
        self._cache: OrderedDict[ArtifactKey, Image.Image] = OrderedDict()
        self._cache_bytes: int = 0
        self._cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES

        self.on_thumbnail_ready = None

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    @staticmethod
    def _resolution_for(purpose: str) -> int:
        return THUMBNAIL_RESOLUTION if purpose == "thumbnail" else PREVIEW_RESOLUTION

    def _key(
        self,
        canonical_address: str,
        fingerprint: SourceFingerprint,
        purpose: str,
        resolution: int,
        policy: str = REPRESENTATIVE_FRAME_POLICY,
    ) -> ArtifactKey:
        return ArtifactKey(
            canonical_address=canonical_address,
            fingerprint=fingerprint,
            purpose=purpose,
            resolution=resolution,
            policy=policy,
        )

    def _complete_key(
        self,
        canonical_address: str,
        purpose: str,
        resolution: int,
        policy: str,
    ) -> ArtifactKey | None:
        """Build the full key for a lookup from the identity fields a
        caller can know without touching the filesystem, completing the
        fingerprint from the memo. Returns None when the address has no
        memo entry -- that is a cache miss, deliberately."""
        with self._lock:
            fingerprint = self._fingerprints.get(canonical_address)
        if fingerprint is None:
            return None
        try:
            return self._key(
                canonical_address, fingerprint, purpose, resolution, policy
            )
        except ValueError:
            # A bad purpose / resolution / policy is a caller error, but a
            # lookup returns None for every other "not available" case, so
            # it returns None here too rather than raising out of get().
            return None

    # ------------------------------------------------------------------
    # Fingerprint memo
    # ------------------------------------------------------------------

    def refresh_fingerprint(
        self,
        canonical_address: str,
        source_path: Path,
    ) -> SourceFingerprint | None:
        """Stat the source file and record its fingerprint in the memo.

        Called from the request path (request_thumbnail and its worker),
        never from a lookup. Returns None if the file cannot be stat-ed.
        """
        try:
            stat = Path(source_path).stat()
        except OSError:
            return None
        fingerprint = SourceFingerprint(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        with self._lock:
            self._fingerprints[canonical_address] = fingerprint
            self._verified.add(canonical_address)
        return fingerprint

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        canonical_address: str,
        purpose: str,
        resolution: int | None = None,
        policy: str = REPRESENTATIVE_FRAME_POLICY,
    ) -> Path | None:
        """
        Returns the file path of a stored artifact, or None.

        Args:
            canonical_address: The canonical media address the picture is
                               of (media_address.canonical_key form).
            purpose:           'thumbnail' or 'preview'.
            resolution:        Requested max side in pixels. Defaults to
                               the standard resolution for the purpose.
            policy:            Representative-frame policy.
        """
        if resolution is None:
            resolution = self._resolution_for(purpose)
        key = self._complete_key(canonical_address, purpose, resolution, policy)
        if key is None:
            return None
        with self._lock:
            return self._index.get(key, None)

    def put(self, key: ArtifactKey, image: np.ndarray) -> Path:
        """
        Stores an artifact and updates the index.

        Args:
            key:   The full ArtifactKey identifying the picture.
            image: An RGB uint8 numpy array.

        Returns:
            The Path where the artifact was saved.
        """
        dest = self._dir / f"{key.stable_hash()}.jpg"
        self._codec.write_jpeg(dest, image)
        with self._lock:
            self._index[key] = dest
        return dest

    def get_pixmap(
        self,
        canonical_address: str,
        purpose: str,
        resolution: int | None = None,
        policy: str = REPRESENTATIVE_FRAME_POLICY,
    ):
        """
        Returns a PIL Image ready for conversion to QPixmap in the UI,
        using the in-memory LRU cache to avoid redundant disk reads.
        Returns None if the artifact does not exist (including when the
        address has no fingerprint memo entry yet).

        On a memory-cache hit this touches no filesystem at all -- no
        stat, no exists, no open. A memory miss that hits the disk index
        reads the JPEG back through ArtifactCodec.
        """
        if resolution is None:
            resolution = self._resolution_for(purpose)
        key = self._complete_key(canonical_address, purpose, resolution, policy)
        if key is None:
            return None

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        with self._lock:
            path = self._index.get(key, None)
        if path is None:
            return None

        try:
            image = self._codec.read_image(path)
        except (OSError, ValueError, ArtifactCodecError):
            # ArtifactCodecError: a persisted index carried a path that
            # resolves outside this cache root (project moved between
            # machines). Treat as a miss, not a crash.
            return None
        self._add_to_cache(key, image)
        return image

    def request_thumbnail(
        self,
        row_id: str,
        address: str,
        source_path: Path,
        table_name: str,
    ) -> None:
        """
        Queues thumbnail and preview generation for a picture in a
        background thread. Handles both image and video source files.

        Args:
            row_id:      The row whose tile is waiting -- echoed back
                         through on_thumbnail_ready(table_name, row_id).
                         Identity of the PICTURE is `address`, not this.
            address:     Canonical media address the picture is of.
            source_path: Absolute filesystem path to decode from. Identity
                         and fetch-route are different things: `address`
                         says what the picture is, `source_path` says
                         where to read it.
            table_name:  Table the row belongs to. Echoed back so the
                         controller can tag the ready notification.

        Short-circuit: if the address already has both a thumbnail and a
        preview in the index (checked against the memo), nothing is
        queued. It does NOT detect that a worker for the same key is
        already running, so two rapid requests still spawn two threads --
        request coalescing is P0.5b-2.

        The one-thread-per-request model is unchanged -- a bounded worker
        pool is P0.5b-2.
        """
        with self._lock:
            fingerprint = self._fingerprints.get(address)
            verified = address in self._verified
        if verified and fingerprint is not None and self._both_present(
            address, fingerprint
        ):
            # The picture is already cached under a freshly-stat-ed
            # fingerprint (an earlier row this session referenced the same
            # file). Still tell the subscriber -- the async path does the
            # same via _notify_ready, and once the renderer's direct
            # fallback goes away (P0.5b-3) this is the only signal the tile
            # gets to repaint. An UNVERIFIED (load-seeded) fingerprint
            # falls through to the worker, which re-stats and regenerates
            # if the source changed since the project was saved.
            self._notify_ready(table_name, row_id)
            return

        thread = threading.Thread(
            target=self._generate_thumbnails,
            args=(row_id, address, Path(source_path), table_name),
            daemon=True,
        )
        thread.start()

    def _both_present(self, address: str, fingerprint: SourceFingerprint) -> bool:
        thumb_key = self._key(
            address, fingerprint, "thumbnail", THUMBNAIL_RESOLUTION
        )
        preview_key = self._key(
            address, fingerprint, "preview", PREVIEW_RESOLUTION
        )
        with self._lock:
            return thumb_key in self._index and preview_key in self._index

    def set_cache_max_bytes(self, max_bytes: int) -> None:
        """
        Sets the maximum memory the cache may use, in bytes.

        Args:
            max_bytes: Maximum cache size in bytes.
        """
        self._cache_max_bytes = max_bytes
        self._evict_if_needed()

    def save_index(self, project_path: Path) -> None:
        """Saves the artifact index to disk as versioned JSON."""
        with self._lock:
            records = [
                {
                    "address":          key.canonical_address,
                    "size":             key.fingerprint.size,
                    "mtime_ns":         key.fingerprint.mtime_ns,
                    "purpose":          key.purpose,
                    "resolution":       key.resolution,
                    "policy":           key.policy,
                    "renderer_version": key.renderer_version,
                    "path":             str(path),
                }
                for key, path in self._index.items()
            ]
        payload = {"format_version": INDEX_FORMAT_VERSION, "artifacts": records}
        index_path = project_path / "artifact_index.json"
        index_path.write_text(json.dumps(payload, indent=2))

    def load_index(self, project_path: Path) -> None:
        """Loads the artifact index from disk.

        An index whose format version does not match is discarded whole,
        not half-read. Callers must reset() first -- load_index() replaces
        the index but is not a substitute for clearing the memory image
        cache.

        Seeds the fingerprint memo from the persisted (size, mtime), so a
        project that was fully thumbnailed reopens with its cache usable
        with no paint-path decode. Those fingerprints are the freshness as
        of the last save, not a fresh stat, so none is marked verified:
        the next request_thumbnail() for such an address always spawns a
        worker (rather than short-circuiting), and if the source changed
        since the save the worker's fresh fingerprint no longer matches
        and it regenerates. Until that request happens a get_pixmap()
        lookup serves the persisted picture. For display (not analysis) a
        briefly-stale preview is an accepted trade against decoding every
        visible source image on the main thread after every reload.
        P0.5b-1 issues no such request after load_project (the eager sites
        are load_folder / load_csv_as_primary only); P0.5b-3's
        demand-driven requests close that gap.
        """
        index_path = project_path / "artifact_index.json"
        if not index_path.exists():
            return
        try:
            payload = json.loads(index_path.read_text())
        except (ValueError, OSError):
            return
        if (
            not isinstance(payload, dict)
            or payload.get("format_version") != INDEX_FORMAT_VERSION
        ):
            # Written under the old (row_id, artifact_type) scheme or an
            # unknown one. Discard rather than mixing schemes.
            return

        new_index: dict[ArtifactKey, Path] = {}
        new_fingerprints: dict[str, SourceFingerprint] = {}
        for record in payload.get("artifacts", []):
            try:
                fingerprint = SourceFingerprint(
                    size=record["size"], mtime_ns=record["mtime_ns"]
                )
                key = ArtifactKey(
                    canonical_address=record["address"],
                    fingerprint=fingerprint,
                    purpose=record["purpose"],
                    resolution=record["resolution"],
                    policy=record.get("policy", REPRESENTATIVE_FRAME_POLICY),
                    renderer_version=record["renderer_version"],
                )
            except (KeyError, TypeError, ValueError):
                continue
            new_index[key] = Path(record["path"])
            # Last writer wins; every record for one address carries the
            # same fingerprint (they were saved together).
            new_fingerprints[record["address"]] = fingerprint

        with self._lock:
            self._index = new_index
            self._fingerprints = new_fingerprints
            # Every seeded fingerprint is the freshness as of the last
            # save, not a fresh stat -- so none is verified.
            self._verified = set()

    def reset(self) -> None:
        """Clears the index, the memory cache and the fingerprint memo.

        load_project() must call this BEFORE load_index(): otherwise a new
        project's index lands on top of the previous project's live image
        cache and fingerprint memo, and an old picture can appear under a
        new row (docs/media_architecture.md section 4.5).

        It does NOT cancel in-flight _generate_thumbnails workers from the
        previous project -- there is no cancellation in the thread-per-
        request model (that is P0.5b-2). A straggler can still write its
        key into the fresh index. Address+fingerprint keying makes that
        harmless for display: the new project looks up its own addresses
        and never matches the straggler's key, so no wrong picture is
        shown. The stale entry is only dead weight (it can get written to
        the new project's saved index and reloaded, but its address is
        never requested there).
        """
        with self._lock:
            self._index.clear()
            self._fingerprints.clear()
            self._verified.clear()
        self._cache.clear()
        self._cache_bytes = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_thumbnails(
        self,
        row_id: str,
        address: str,
        source_path: Path,
        table_name: str,
    ) -> None:
        """
        Runs in a background thread. Generates thumbnail and preview
        images from the source file (image or video) and stores them
        under their ArtifactKeys. Calls on_thumbnail_ready(table_name,
        row_id) when done.

        For image files: loads via PIL.
        For video files: extracts the first frame via OpenCV.

        (Both are source-media decodes that P1.2 will route through the
        resolver. This diff leaves them as they were.)
        """
        try:
            if not source_path.exists():
                return

            fingerprint = self.refresh_fingerprint(address, source_path)
            if fingerprint is None:
                return

            # A concurrent request may already have produced both.
            if self._both_present(address, fingerprint):
                self._notify_ready(table_name, row_id)
                return

            suffix = source_path.suffix.lower()
            if suffix in VIDEO_EXTENSIONS:
                img = self._first_frame_as_pil(source_path)
                if img is None:
                    return
            else:
                img = Image.open(source_path).convert("RGB")

            thumb = img.copy()
            thumb.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
            self.put(
                self._key(address, fingerprint, "thumbnail", THUMBNAIL_RESOLUTION),
                np.array(thumb, dtype=np.uint8),
            )

            preview = img.copy()
            preview.thumbnail(PREVIEW_SIZE, Image.LANCZOS)
            self.put(
                self._key(address, fingerprint, "preview", PREVIEW_RESOLUTION),
                np.array(preview, dtype=np.uint8),
            )

        except Exception as e:
            print(f"[ArtifactStore] Failed to generate thumbnails "
                  f"for {address}: {e}")
            return

        print(f"[ArtifactStore] Thumbnail ready for {address}")
        self._notify_ready(table_name, row_id)

    def _notify_ready(self, table_name: str, row_id: str) -> None:
        if self.on_thumbnail_ready is not None:
            self.on_thumbnail_ready(table_name, row_id)

    def _first_frame_as_pil(self, video_path: Path) -> Image.Image | None:
        """
        Extracts the first frame of a video file and returns it as a
        PIL Image in RGB mode.

        Uses OpenCV (cv2). Returns None if OpenCV is not installed or
        if the video cannot be read.

        Args:
            video_path: Path to the video file.

        Returns:
            A PIL Image, or None.
        """
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            ok, frame = cap.read()
            cap.release()

            if not ok or frame is None:
                print(f"[ArtifactStore] Could not read first frame "
                      f"from {video_path}")
                return None

            # OpenCV returns BGR — convert to RGB.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)

        except ImportError:
            print("[ArtifactStore] OpenCV (cv2) not installed — "
                  "cannot generate video thumbnail.")
            return None
        except Exception as e:
            print(f"[ArtifactStore] Video frame error for {video_path}: {e}")
            return None

    def _add_to_cache(
        self,
        key: ArtifactKey,
        image: Image.Image,
    ) -> None:
        """Adds an image to the LRU cache and evicts if over limit."""
        estimated_bytes = image.width * image.height * 3
        self._cache[key] = image
        self._cache.move_to_end(key)
        self._cache_bytes += estimated_bytes
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Removes least recently used entries until within memory limit."""
        while self._cache_bytes > self._cache_max_bytes and self._cache:
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= evicted.width * evicted.height * 3
