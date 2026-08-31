"""
settings/settings_store.py

SettingsStore -- load and save a GelemSettings through a trivial string
key/value backend.

The backend protocol is two methods:

    get(key: str) -> str | None
    set(key: str, value: str) -> None

Everything is persisted as a string (sizes as "WxH", byte counts as
decimal digits), so a backend never has to know a value's type. The store
owns the persisted key names; they appear nowhere else in the codebase.
"""

from __future__ import annotations

from typing import Protocol

from settings.settings import GelemSettings


class SettingsBackend(Protocol):
    """The two-method contract a SettingsStore backend must satisfy."""

    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str) -> None: ...


def _size_to_str(size: tuple[int, int]) -> str:
    """(150, 150) -> '150x150'."""
    width, height = size
    return f"{int(width)}x{int(height)}"


class SettingsStore:
    """Reads and writes a GelemSettings via a string key/value backend."""

    # The persisted key names. These live here and nowhere else -- callers
    # deal in GelemSettings field names, never in these strings.
    _KEY_PICTURE_MEMORY = "artifacts/picture_memory_max_bytes"
    _KEY_PICTURE_DISK = "artifacts/picture_disk_max_bytes"
    _KEY_WORKER_COUNT = "artifacts/worker_count"
    _KEY_THUMBNAIL_SIZE = "artifacts/thumbnail_size"
    _KEY_PREVIEW_SIZE = "artifacts/preview_size"

    def __init__(self, backend: SettingsBackend):
        self._backend = backend

    def load(self) -> tuple[GelemSettings, list[str]]:
        """Read the raw strings from the backend and hand them to
        GelemSettings.from_values, which tolerates anything missing or
        malformed. Returns the settings plus a list of plain-English
        problem messages (empty when everything loaded cleanly)."""
        raw = {
            "picture_memory_max_bytes": self._backend.get(
                self._KEY_PICTURE_MEMORY
            ),
            "picture_disk_max_bytes": self._backend.get(
                self._KEY_PICTURE_DISK
            ),
            "worker_count": self._backend.get(self._KEY_WORKER_COUNT),
            "thumbnail_size": self._backend.get(self._KEY_THUMBNAIL_SIZE),
            "preview_size": self._backend.get(self._KEY_PREVIEW_SIZE),
        }
        return GelemSettings.from_values(raw)

    def save(self, settings: GelemSettings) -> None:
        """Write every field to the backend as a string."""
        self._backend.set(
            self._KEY_PICTURE_MEMORY, str(settings.picture_memory_max_bytes)
        )
        self._backend.set(
            self._KEY_PICTURE_DISK, str(settings.picture_disk_max_bytes)
        )
        self._backend.set(
            self._KEY_WORKER_COUNT, str(settings.worker_count)
        )
        self._backend.set(
            self._KEY_THUMBNAIL_SIZE, _size_to_str(settings.thumbnail_size)
        )
        self._backend.set(
            self._KEY_PREVIEW_SIZE, _size_to_str(settings.preview_size)
        )
