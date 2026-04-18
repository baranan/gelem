"""
artifacts/artifact_store.py

ArtifactStore manages thumbnail and preview images used by the gallery
for fast display. It handles both image and video source files.

For images: thumbnails are generated using PIL.
For videos:  the first frame is extracted using OpenCV, then the same
             PIL-based resizing pipeline is applied.

This file is written centrally (not by a student).
"""

from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
import threading

import numpy as np
from PIL import Image

# Import the extension sets from dataset so they stay in sync.
# We only need to know which extensions are videos here.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

THUMBNAIL_SIZE = (150, 150)
PREVIEW_SIZE   = (600, 600)

DEFAULT_CACHE_MAX_BYTES = 500 * 1024 * 1024


class ArtifactStore:
    """
    Stores and retrieves derived visual files for gallery display.

    Supports both image and video source files. For video files,
    thumbnails are generated from the first frame.
    """

    def __init__(self, artifacts_dir: Path):
        self._dir = artifacts_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[tuple[str, str], Path] = {}
        self._cache: OrderedDict[tuple[str, str], Image.Image] = OrderedDict()
        self._cache_bytes: int = 0
        self._cache_max_bytes: int = DEFAULT_CACHE_MAX_BYTES
        self._lock = threading.Lock()
        self.on_thumbnail_ready = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, row_id: str, artifact_type: str) -> Path | None:
        """
        Returns the file path of a stored artifact, or None.

        Args:
            row_id:        The row the artifact belongs to.
            artifact_type: 'thumbnail' or 'preview'.
        """
        key = (row_id, artifact_type)
        with self._lock:
            return self._index.get(key, None)

    def put(
        self,
        row_id: str,
        artifact_type: str,
        image: np.ndarray | Path,
    ) -> Path:
        """
        Stores an artifact and updates the index.

        Args:
            row_id:        The row the artifact belongs to.
            artifact_type: 'thumbnail' or 'preview'.
            image:         The image to store (numpy array or Path).

        Returns:
            The Path where the artifact was saved.
        """
        filename = f"{row_id}_{artifact_type}.jpg"
        dest     = self._dir / filename

        if isinstance(image, Path):
            import shutil
            shutil.copy2(image, dest)
        else:
            pil_image = Image.fromarray(image)
            pil_image.save(dest, "JPEG", quality=85)

        key = (row_id, artifact_type)
        with self._lock:
            self._index[key] = dest

        return dest

    def get_pixmap(self, row_id: str, artifact_type: str):
        """
        Returns a PIL Image ready for conversion to QPixmap in the UI,
        using the in-memory LRU cache to avoid redundant disk reads.
        Returns None if the artifact does not exist.

        Args:
            row_id:        The row the artifact belongs to.
            artifact_type: 'thumbnail' or 'preview'.
        """
        key = (row_id, artifact_type)

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        path = self.get(row_id, artifact_type)
        if path is None or not path.exists():
            return None

        image = Image.open(path)
        image.load()
        self._add_to_cache(key, image)
        return image

    def request_thumbnail(self, row_id: str, full_path: Path) -> None:
        """
        Queues thumbnail and preview generation for an item in a
        background thread. Handles both image and video source files.

        Args:
            row_id:    The item to generate thumbnails for.
            full_path: Path to the full-resolution source file.
        """
        if (self.get(row_id, "thumbnail") is not None and
                self.get(row_id, "preview") is not None):
            return

        thread = threading.Thread(
            target=self._generate_thumbnails,
            args=(row_id, full_path),
            daemon=True,
        )
        thread.start()

    def set_cache_max_bytes(self, max_bytes: int) -> None:
        """
        Sets the maximum memory the cache may use, in bytes.

        Args:
            max_bytes: Maximum cache size in bytes.
        """
        self._cache_max_bytes = max_bytes
        self._evict_if_needed()

    def save_index(self, project_path: Path) -> None:
        """Saves the artifact index to disk as JSON."""
        import json
        index_data = {
            f"{k[0]}|{k[1]}": str(v)
            for k, v in self._index.items()
        }
        index_path = project_path / "artifact_index.json"
        index_path.write_text(json.dumps(index_data, indent=2))

    def load_index(self, project_path: Path) -> None:
        """Loads the artifact index from disk."""
        import json
        index_path = project_path / "artifact_index.json"
        if not index_path.exists():
            return
        index_data = json.loads(index_path.read_text())
        with self._lock:
            self._index = {
                tuple(k.split("|")): Path(v)
                for k, v in index_data.items()
            }

    def reset(self) -> None:
        """Clears the index and cache completely."""
        with self._lock:
            self._index.clear()
        self._cache.clear()
        self._cache_bytes = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_thumbnails(self, row_id: str, full_path: Path) -> None:
        """
        Runs in a background thread. Generates thumbnail and preview
        images from the source file (image or video) and saves them
        to disk. Calls on_thumbnail_ready when done.

        For image files: loads via PIL.
        For video files: extracts the first frame via OpenCV.

        Args:
            row_id:    The item being processed.
            full_path: Path to the source media file.
        """
        try:
            if not full_path.exists():
                return

            # Load the source as a PIL Image, dispatching on extension.
            suffix = full_path.suffix.lower()

            if suffix in VIDEO_EXTENSIONS:
                img = self._first_frame_as_pil(full_path)
                if img is None:
                    return
            else:
                # Treat as image.
                img = Image.open(full_path).convert("RGB")

            # Generate thumbnail.
            thumb = img.copy()
            thumb.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
            self.put(row_id, "thumbnail", np.array(thumb, dtype=np.uint8))

            # Generate preview.
            preview = img.copy()
            preview.thumbnail(PREVIEW_SIZE, Image.LANCZOS)
            self.put(row_id, "preview", np.array(preview, dtype=np.uint8))

        except Exception as e:
            print(f"[ArtifactStore] Failed to generate thumbnails "
                  f"for {row_id}: {e}")
            return

        print(f"[ArtifactStore] Thumbnail ready for {row_id}")
        if self.on_thumbnail_ready is not None:
            self.on_thumbnail_ready(row_id)

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
        key: tuple[str, str],
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
