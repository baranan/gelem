"""
settings/settings_gateway.py

SettingsGateway -- the plain-data editing face of the machine-tunable
settings.

The dialog (P0.5b-2ii-c2b2) never touches a SettingsStore or a
GelemSettings. It asks the gateway to describe the editable fields, shows
them, and hands back a name -> raw-value mapping of the fields it wants to
change for the gateway to validate and persist. The mapping is a PARTIAL
update: any field it does not name keeps the value currently in the store.
AppController sits between the two and only passes the calls through.

This module is Qt-free and holds no knowledge that a dialog exists. It
owns nothing: the numbers, their bounds and the validation all live in
settings/settings.py; the persisted key layout lives in
settings/settings_store.py.

docs/architecture.md section 9 is the authority for what each value means
and when it takes effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from settings.settings import (
    GelemSettings,
    PICTURE_MEMORY_MAX_BYTES_RANGE,
    PICTURE_DISK_MAX_BYTES_RANGE,
    WORKER_COUNT_RANGE,
    THUMBNAIL_MAX_SIDE_RANGE,
    PREVIEW_MAX_SIDE_RANGE,
)
from settings.settings_store import SettingsStore


@dataclass(frozen=True)
class SettingField:
    """One editable setting, described as plain data for a UI to render.

    `unit` is one of the plain strings "bytes", "count", "pixels". It
    says what the number IS, not how to display it: choosing to show a
    byte value in MiB is the dialog's job, not this file's.
    """

    name: str
    label: str
    help_text: str
    minimum: int
    maximum: int
    unit: str
    restart_required: bool
    current_value: int


# The five fields, in the order docs/architecture.md section 9's table
# lists them. minimum and maximum are ALWAYS read from the *_RANGE tuples
# in settings/settings.py -- never retyped as literals here -- so a bound
# change in one place cannot silently disagree with another.
#
# Each entry: (name, label, help_text, range_tuple, unit, restart_required)
_FIELD_SPECS = (
    (
        "picture_memory_max_bytes",
        "Picture memory limit",
        "Ceiling on the RAM the in-memory decoded-image cache may hold. "
        "Over it, the least recently used images are dropped and "
        "regenerate from disk on next view. Takes effect immediately.",
        PICTURE_MEMORY_MAX_BYTES_RANGE,
        "bytes",
        False,
    ),
    (
        "picture_disk_max_bytes",
        "Picture disk limit",
        "Ceiling on the total size of the derived-JPEG files in a "
        "project's artifacts folder. Over it, the oldest are deleted and "
        "regenerate on demand. Takes effect immediately.",
        PICTURE_DISK_MAX_BYTES_RANGE,
        "bytes",
        False,
    ),
    (
        "worker_count",
        "Background worker threads",
        "How many background threads decode and resize source media for "
        "thumbnails and previews. Higher uses more CPU and RAM for faster "
        "gallery fill. Takes effect on restart.",
        WORKER_COUNT_RANGE,
        "count",
        True,
    ),
    (
        "thumbnail_max_side",
        "Thumbnail size",
        "Largest side, in pixels, of a gallery thumbnail. This number "
        "also decides, per tile, whether a tile asks for a thumbnail or a "
        "preview. Takes effect on restart.",
        THUMBNAIL_MAX_SIDE_RANGE,
        "pixels",
        True,
    ),
    (
        "preview_max_side",
        "Preview size",
        "Largest side, in pixels, of the larger preview image used for "
        "bigger tiles and quick previews. Must not be smaller than the "
        "thumbnail size. Takes effect on restart.",
        PREVIEW_MAX_SIDE_RANGE,
        "pixels",
        True,
    ),
)


class SettingsGateway:
    """The plain-data editing face over a SettingsStore.

    Exactly two public methods: describe_fields() and save_values().
    Nothing else is public -- the dialog needs no more than this.
    """

    def __init__(self, store: SettingsStore):
        self._store = store

    def describe_fields(self) -> list[SettingField]:
        """Return one SettingField per machine-tunable value, in section
        9's order, each carrying its current persisted value.

        The current value is read through the store, so it is the same
        clamped/defaulted value the app would boot with.
        """
        # Load the current settings once; from_values corrections are not
        # our concern here -- we only report what is in effect.
        current_settings, _problems = self._store.load()

        fields: list[SettingField] = []
        for name, label, help_text, bounds, unit, restart_required in _FIELD_SPECS:
            minimum, maximum = bounds
            fields.append(
                SettingField(
                    name=name,
                    label=label,
                    help_text=help_text,
                    minimum=minimum,
                    maximum=maximum,
                    unit=unit,
                    restart_required=restart_required,
                    current_value=getattr(current_settings, name),
                )
            )
        return fields

    def save_values(self, mapping: Mapping) -> list[str]:
        """Apply a PARTIAL update `mapping` and persist the result. Never
        raises.

        `mapping` is a field name -> raw value mapping naming only the
        fields the caller wants to change. It is OVERLAID onto the values
        currently in the store -- a field the mapping does not name keeps
        its current persisted value, it is not reset to a default.

        The overlaid full mapping is then run through
        GelemSettings.from_values, so an unparseable or out-of-range value
        is corrected exactly as it is on load, and the cross-field rule
        (preview not smaller than thumbnail) is applied BEFORE anything is
        written. The corrected settings are then persisted through the
        store.

        Returns from_values' list of plain-English problem messages
        (empty when every value was accepted as given).
        """
        # Start from what is currently persisted, so an omitted field
        # survives untouched. Keys come from _FIELD_SPECS, not retyped.
        current_settings, _problems = self._store.load()
        merged: dict = {
            name: getattr(current_settings, name)
            for name, *_rest in _FIELD_SPECS
        }
        # Overlay the caller's partial update.
        merged.update(mapping)

        corrected_settings, problems = GelemSettings.from_values(merged)
        self._store.save(corrected_settings)
        return list(problems)
