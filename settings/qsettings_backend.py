"""
settings/qsettings_backend.py

QSettingsBackend -- the real SettingsStore backend, backed by Qt's
QSettings (the platform-native store: registry on Windows, plist on
macOS, an ini file on Linux).

This is the ONLY file in the settings package -- and the only file
outside main.py's Qt setup -- that imports PySide6. Everything else in
settings/ is Qt-free so it can be tested without a QApplication.

QSettings() with no arguments uses the application and organization names
that main.py sets on the QApplication before anything reads settings, so
this backend takes no configuration.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings


class QSettingsBackend:
    """A string key/value store backed by QSettings. Satisfies the
    SettingsStore backend protocol: get(key) and set(key, value)."""

    def __init__(self):
        # Relies on QApplication.setApplicationName / setOrganizationName
        # having already run in main.main().
        self._qsettings = QSettings()

    def get(self, key: str) -> str | None:
        """Return the stored string for `key`, or None if unset. QSettings
        may hand back a non-string on some platforms; coerce so the store
        always sees a string or None."""
        value = self._qsettings.value(key)
        if value is None:
            return None
        return str(value)

    def set(self, key: str, value: str) -> None:
        """Persist `value` (always a string) under `key`."""
        self._qsettings.setValue(key, str(value))
