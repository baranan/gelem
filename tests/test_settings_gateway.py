"""
tests/test_settings_gateway.py

The settings plain-data face (P0.5b-2ii-c2b1): SettingsGateway.describe_fields
and save_values, plus AppController.get_settings_fields / apply_settings
passing calls through to it. No dialog in this item.

Written from the work-item specification, not from the implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from settings.settings import (
    PICTURE_MEMORY_MAX_BYTES_RANGE,
    PICTURE_DISK_MAX_BYTES_RANGE,
    WORKER_COUNT_RANGE,
    THUMBNAIL_MAX_SIDE_RANGE,
    PREVIEW_MAX_SIDE_RANGE,
)
from settings.settings_store import SettingsStore
from settings.settings_gateway import SettingsGateway, SettingField

from controller import AppController


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class DictBackend:
    """A SettingsStore backend backed by a plain dict; every value a string."""

    def __init__(self, data: dict | None = None):
        self.data: dict[str, str] = dict(data or {})

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        assert isinstance(value, str), f"backend given a non-string: {value!r}"
        self.data[key] = value


class _StubDataset:
    def set_registry(self, registry) -> None:
        pass


class _StubStore:
    """Records exactly what the controller pushes into the artifact store.

    `set_disk_cache_max_bytes` returns a SweepResult-shaped object whose
    `files_deleted` count the test controls via `disk_files_deleted`
    (default 0, i.e. the sweep removed nothing).
    """

    def __init__(self):
        self.on_thumbnail_ready = None
        self.memory_calls: list[int] = []
        self.disk_calls: list[int] = []
        self.disk_files_deleted: int = 0

    def set_memory_cache_max_bytes(self, nbytes: int) -> None:
        self.memory_calls.append(nbytes)

    def set_disk_cache_max_bytes(self, nbytes: int):
        self.disk_calls.append(nbytes)
        return SimpleNamespace(files_deleted=self.disk_files_deleted)


class _StubGateway:
    """A SettingsGateway stand-in that remembers what it was told and
    reports it straight back through describe_fields()."""

    def __init__(self, start: dict[str, int]):
        self._values = dict(start)
        self.saved_mapping: dict | None = None

    def save_values(self, mapping) -> list[str]:
        self.saved_mapping = dict(mapping)
        self._values.update({k: int(v) for k, v in mapping.items()})
        return []

    def describe_fields(self) -> list:
        return [
            SimpleNamespace(name=name, current_value=value)
            for name, value in self._values.items()
        ]


class _CorrectingStubGateway:
    """A SettingsGateway stand-in that CORRECTS the memory ceiling it is
    given -- it always clamps it to a fixed smaller number -- and reports
    the corrected value through describe_fields(). This is what lets a
    test tell "pushed the raw value" from "pushed the saved value" apart.
    """

    #  Whatever memory ceiling it is handed, this is what it persists.
    CORRECTED_MEMORY = 100 * 1024 * 1024

    def __init__(self, start: dict[str, int]):
        self._values = dict(start)
        self.saved_mapping: dict | None = None

    def save_values(self, mapping) -> list[str]:
        self.saved_mapping = dict(mapping)
        self._values.update({k: int(v) for k, v in mapping.items()})
        # The correction: the memory ceiling never persists as given.
        self._values["picture_memory_max_bytes"] = self.CORRECTED_MEMORY
        return ["the memory ceiling was corrected"]

    def describe_fields(self) -> list:
        return [
            SimpleNamespace(name=name, current_value=value)
            for name, value in self._values.items()
        ]


def _make_controller(gateway, store):
    return AppController(
        _StubDataset(),
        object(),   # query_engine -- untouched by __init__
        store,
        object(),   # registry -- untouched by __init__
        object(),   # operator_registry -- untouched by __init__
        settings_gateway=gateway,
    )


# ===========================================================================
# CHECK 1 -- the wiring, proved by deletion.
# ===========================================================================

def test_apply_settings_pushes_both_ceilings_into_the_store():
    start = {
        "picture_memory_max_bytes": 500 * 1024 * 1024,
        "picture_disk_max_bytes": 1024 * 1024 * 1024,
        "worker_count": 2,
        "thumbnail_max_side": 150,
        "preview_max_side": 600,
    }
    gateway = _StubGateway(start)
    store = _StubStore()
    controller = _make_controller(gateway, store)

    new_memory = 321 * 1024 * 1024
    new_disk = 654 * 1024 * 1024
    problems = controller.apply_settings({
        "picture_memory_max_bytes": new_memory,
        "picture_disk_max_bytes": new_disk,
    })

    assert problems == []
    # Both setters called exactly once, with exactly the saved numbers.
    assert store.memory_calls == [new_memory]
    assert store.disk_calls == [new_disk]


def test_settings_methods_raise_without_a_gateway():
    store = _StubStore()
    controller = AppController(
        _StubDataset(), object(), store, object(), object()
    )
    with pytest.raises(RuntimeError):
        controller.get_settings_fields()
    with pytest.raises(RuntimeError):
        controller.apply_settings({})


def test_get_settings_fields_forwards_to_gateway():
    gateway = _StubGateway({"worker_count": 2})
    controller = _make_controller(gateway, _StubStore())
    fields = controller.get_settings_fields()
    assert [f.name for f in fields] == ["worker_count"]


# ===========================================================================
# CHECK B -- FIX 2: apply_settings pushes the value SAVED, not the value it
# was handed. Proved with a gateway that always corrects the memory ceiling.
# ===========================================================================

def test_apply_settings_pushes_the_corrected_value_not_the_raw_one():
    start = {
        "picture_memory_max_bytes": 500 * 1024 * 1024,
        "picture_disk_max_bytes": 1024 * 1024 * 1024,
        "worker_count": 2,
        "thumbnail_max_side": 150,
        "preview_max_side": 600,
    }
    gateway = _CorrectingStubGateway(start)
    store = _StubStore()
    controller = _make_controller(gateway, store)

    raw_memory = 999 * 1024 * 1024   # what the caller asks for
    problems = controller.apply_settings({
        "picture_memory_max_bytes": raw_memory,
    })

    assert problems == ["the memory ceiling was corrected"]
    # The store was told the CORRECTED number, never the raw request.
    assert store.memory_calls == [_CorrectingStubGateway.CORRECTED_MEMORY]
    assert raw_memory not in store.memory_calls


# ===========================================================================
# CHECK 4 -- the persisted layout, pinned. This is the first code in the app
# that ever writes these five keys, so the layout is pinned on purpose.
# ===========================================================================

def test_save_values_writes_exactly_the_five_artifacts_keys():
    backend = DictBackend()
    gateway = SettingsGateway(SettingsStore(backend))

    problems = gateway.save_values({
        "picture_memory_max_bytes": 268435456,
        "picture_disk_max_bytes": 536870912,
        "worker_count": 4,
        "thumbnail_max_side": 128,
        "preview_max_side": 512,
    })

    assert problems == []
    assert backend.data == {
        "artifacts/picture_memory_max_bytes": "268435456",
        "artifacts/picture_disk_max_bytes": "536870912",
        "artifacts/worker_count": "4",
        "artifacts/thumbnail_max_side": "128",
        "artifacts/preview_max_side": "512",
    }


# ===========================================================================
# CHECK 5 -- the metadata is not retyped and not vacuous.
# ===========================================================================

def test_describe_fields_metadata_comes_from_the_range_constants():
    gateway = SettingsGateway(SettingsStore(DictBackend()))
    fields = gateway.describe_fields()

    assert all(isinstance(f, SettingField) for f in fields)

    assert [f.name for f in fields] == [
        "picture_memory_max_bytes",
        "picture_disk_max_bytes",
        "worker_count",
        "thumbnail_max_side",
        "preview_max_side",
    ]

    expected_bounds = [
        PICTURE_MEMORY_MAX_BYTES_RANGE,
        PICTURE_DISK_MAX_BYTES_RANGE,
        WORKER_COUNT_RANGE,
        THUMBNAIL_MAX_SIDE_RANGE,
        PREVIEW_MAX_SIDE_RANGE,
    ]
    for field, (low, high) in zip(fields, expected_bounds):
        assert field.minimum == low
        assert field.maximum == high

    assert [f.restart_required for f in fields] == [
        False, False, True, True, True,
    ]

    assert [f.unit for f in fields] == [
        "bytes", "bytes", "count", "pixels", "pixels",
    ]


# ===========================================================================
# CHECK 6 -- correction happens before persistence.
# ===========================================================================

def test_cross_field_correction_is_applied_before_the_value_is_written():
    backend = DictBackend()
    gateway = SettingsGateway(SettingsStore(backend))

    problems = gateway.save_values({
        "thumbnail_max_side": 800,
        "preview_max_side": 300,   # smaller than the thumbnail
    })

    # One message about the preview being lifted.
    assert any("preview" in message.lower() for message in problems)

    # The value that actually landed on disk is the corrected one, not 300.
    assert backend.data["artifacts/preview_max_side"] == "800"
    assert backend.data["artifacts/thumbnail_max_side"] == "800"

    # And it reads back that way through describe_fields().
    preview_field = next(
        f for f in gateway.describe_fields() if f.name == "preview_max_side"
    )
    assert preview_field.current_value == 800


# ===========================================================================
# CHECK A -- FIX 1: save_values overlays a partial mapping onto what is
# already stored; the fields the caller omits keep their persisted values
# and are NOT reset to defaults.
# ===========================================================================

# Five NON-default in-range values, so a reset-to-defaults is visibly
# different from a correct overlay. (Defaults are 500 MiB / 1 GiB / 2 /
# 150 / 600.)
_SEED_PERSISTED = {
    "artifacts/picture_memory_max_bytes": "268435456",   # 256 MiB
    "artifacts/picture_disk_max_bytes": "536870912",      # 512 MiB
    "artifacts/worker_count": "5",
    "artifacts/thumbnail_max_side": "200",
    "artifacts/preview_max_side": "700",
}


def test_save_values_overlays_a_partial_mapping_and_keeps_the_rest():
    backend = DictBackend(_SEED_PERSISTED)
    gateway = SettingsGateway(SettingsStore(backend))

    # Change ONE field only.
    problems = gateway.save_values({"worker_count": 8})
    assert problems == []

    # The named field changed; the other four are exactly as seeded.
    assert backend.data == {
        "artifacts/picture_memory_max_bytes": "268435456",
        "artifacts/picture_disk_max_bytes": "536870912",
        "artifacts/worker_count": "8",
        "artifacts/thumbnail_max_side": "200",
        "artifacts/preview_max_side": "700",
    }


# ===========================================================================
# CHECK C -- the overlay does not weaken the cross-field correction; in fact
# it makes a case possible the old replace-everything behaviour would have
# hidden: raising the thumbnail alone must still lift a now-too-small
# preview that the caller never mentioned.
# ===========================================================================

def test_raising_thumbnail_alone_lifts_the_stored_preview():
    backend = DictBackend({
        "artifacts/thumbnail_max_side": "150",
        "artifacts/preview_max_side": "600",
    })
    gateway = SettingsGateway(SettingsStore(backend))

    problems = gateway.save_values({"thumbnail_max_side": 800})

    # The preview the caller never named was lifted to match, and said so.
    assert any("preview" in message.lower() for message in problems)
    assert backend.data["artifacts/thumbnail_max_side"] == "800"
    assert backend.data["artifacts/preview_max_side"] == "800"


# ===========================================================================
# P0.5b-2ii-c2b2a -- apply_settings pushes a ceiling ONLY when that ceiling's
# stored value actually changed (gateway value before the save vs after),
# and reports a disk-ceiling sweep that deleted cached picture files.
#
# Every test below seeds all five fields with distinct, non-default,
# in-range values (defaults are 500 MiB / 1 GiB / 2 / 150 / 600), so a
# wrong push is visible.
# ===========================================================================

# 256 MiB / 512 MiB / 5 / 200 / 700 -- none of them a default, all in range,
# and preview (700) >= thumbnail (200).
_FIVE_DISTINCT_SEED = {
    "picture_memory_max_bytes": 256 * 1024 * 1024,
    "picture_disk_max_bytes": 512 * 1024 * 1024,
    "worker_count": 5,
    "thumbnail_max_side": 200,
    "preview_max_side": 700,
}


class _RevertingMemoryGateway:
    """A SettingsGateway stand-in that accepts a memory-ceiling change in
    its mapping but persists the value that was ALREADY stored -- it
    corrects the submission straight back. describe_fields() therefore
    reports the same memory number before and after the save, even though
    the caller's mapping named that field.
    """

    def __init__(self, start: dict[str, int]):
        self._values = dict(start)
        self.saved_mapping: dict | None = None

    def save_values(self, mapping) -> list[str]:
        self.saved_mapping = dict(mapping)
        stored_memory = self._values["picture_memory_max_bytes"]
        self._values.update({k: int(v) for k, v in mapping.items()})
        # The correction: the memory ceiling is put straight back to what
        # was already stored, so nothing actually changed.
        self._values["picture_memory_max_bytes"] = stored_memory
        return ["the memory ceiling could not be changed and was kept"]

    def describe_fields(self) -> list:
        return [
            SimpleNamespace(name=name, current_value=value)
            for name, value in self._values.items()
        ]


def test_t1_changing_only_thumbnail_pushes_neither_ceiling():
    gateway = _StubGateway(_FIVE_DISTINCT_SEED)
    store = _StubStore()
    controller = _make_controller(gateway, store)

    # A field that is neither ceiling.
    problems = controller.apply_settings({"thumbnail_max_side": 300})

    assert problems == []
    assert store.memory_calls == []
    assert store.disk_calls == []


def test_t2_changing_only_the_memory_ceiling_pushes_only_it():
    gateway = _StubGateway(_FIVE_DISTINCT_SEED)
    store = _StubStore()
    controller = _make_controller(gateway, store)

    new_memory = 333 * 1024 * 1024   # in range, distinct from the 256 MiB seed
    problems = controller.apply_settings(
        {"picture_memory_max_bytes": new_memory}
    )

    assert problems == []
    # The memory ceiling was pushed with exactly the saved number.
    assert store.memory_calls == [new_memory]
    # The disk ceiling did not move, so its sweep never ran.
    assert store.disk_calls == []


def test_t3_a_value_corrected_back_to_the_stored_one_pushes_nothing():
    # The caller submits a DIFFERENT memory ceiling; the gateway corrects
    # it straight back to the value already stored. Nothing changed, so
    # nothing must be pushed.
    #
    # An implementation that keyed off the caller's mapping KEYS -- "the
    # mapping named picture_memory_max_bytes, therefore push it" -- would
    # call set_memory_cache_max_bytes here and fail this test. The correct
    # implementation compares the gateway's before/after values, sees no
    # change, and pushes nothing.
    gateway = _RevertingMemoryGateway(_FIVE_DISTINCT_SEED)
    store = _StubStore()
    controller = _make_controller(gateway, store)

    problems = controller.apply_settings(
        {"picture_memory_max_bytes": 999 * 1024 * 1024}
    )

    assert problems == ["the memory ceiling could not be changed and was kept"]
    assert store.memory_calls == []
    assert store.disk_calls == []


def test_t4_a_disk_ceiling_cut_that_deletes_files_reports_the_count():
    gateway = _StubGateway(_FIVE_DISTINCT_SEED)
    store = _StubStore()
    store.disk_files_deleted = 12
    controller = _make_controller(gateway, store)

    # Genuinely lower the disk ceiling (512 MiB seed -> 100 MiB, in range).
    new_disk = 100 * 1024 * 1024
    problems = controller.apply_settings({"picture_disk_max_bytes": new_disk})

    assert store.disk_calls == [new_disk]
    # One message names the 12 deleted files.
    assert any("12" in message for message in problems)


def test_t4b_the_sweep_note_is_singular_when_exactly_one_file_went():
    # Researcher-facing text: "deleted 1 cached picture file", not "files".
    gateway = _StubGateway(_FIVE_DISTINCT_SEED)
    store = _StubStore()
    store.disk_files_deleted = 1
    controller = _make_controller(gateway, store)

    new_disk = 100 * 1024 * 1024   # 512 MiB seed -> 100 MiB, in range
    problems = controller.apply_settings({"picture_disk_max_bytes": new_disk})

    assert store.disk_calls == [new_disk]
    # The singular noun phrase, with no trailing "s" on "file".
    assert any("deleted 1 cached picture file." in message for message in problems)
    assert not any("1 cached picture files" in message for message in problems)


def test_t5_a_disk_ceiling_change_that_deletes_nothing_adds_no_message():
    # This gateway returns one correction message and leaves files_deleted
    # at 0 on the store. The returned list must be exactly that one
    # correction message -- no sweep note tacked on.
    #
    # T5 is also the only test in this file that fails an implementation
    # which skips BOTH ceiling pushes whenever the gateway's correction
    # list is non-empty: here the correction list is non-empty AND the
    # disk ceiling genuinely changed, so the disk push (and its sweep)
    # must still happen -- assert store.disk_calls below pins that.
    gateway = _CorrectingStubGateway(_FIVE_DISTINCT_SEED)
    store = _StubStore()   # disk_files_deleted stays 0
    controller = _make_controller(gateway, store)

    # A real disk-ceiling change (512 MiB seed -> 100 MiB).
    problems = controller.apply_settings(
        {"picture_disk_max_bytes": 100 * 1024 * 1024}
    )

    assert store.disk_calls == [100 * 1024 * 1024]
    assert problems == ["the memory ceiling was corrected"]
