"""
tests/test_settings.py

The settings mechanism (P0.5b-2ii-c1): GelemSettings validation,
SettingsStore round-tripping, and ArtifactStore honouring the injected
values. No UI here -- the dialog is c2.

Written from the work-item specification, not from the implementation.
"""

from __future__ import annotations

import ast
import os
import sys
import warnings
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from settings.settings import (
    GelemSettings,
    DEFAULT_PICTURE_MEMORY_MAX_BYTES,
    DEFAULT_PICTURE_DISK_MAX_BYTES,
    DEFAULT_WORKER_COUNT,
    DEFAULT_THUMBNAIL_SIZE,
    DEFAULT_PREVIEW_SIZE,
    PICTURE_MEMORY_MAX_BYTES_RANGE,
    WORKER_COUNT_RANGE,
    THUMBNAIL_SIDE_RANGE,
)
from settings.settings_store import SettingsStore

from artifacts.artifact_store import ArtifactStore, SweepResult
from media.artifact_key import ArtifactKey, SourceFingerprint


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------

class DictBackend:
    """A SettingsStore backend backed by a plain dict. Enforces the
    contract: every persisted value is a string."""

    def __init__(self, data: dict | None = None):
        self.data: dict[str, str] = dict(data or {})

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        assert isinstance(value, str), f"backend given a non-string: {value!r}"
        self.data[key] = value


def _seed_disk_thumb(store: ArtifactStore, address: str, nbytes: int, mtime_ns: int):
    """Add one on-disk 'thumbnail' index entry for `address`."""
    fingerprint = SourceFingerprint(size=1, mtime_ns=1)
    key = ArtifactKey(
        address, fingerprint, "thumbnail", store.resolution_for("thumbnail")
    )
    path = store._dir / f"{key.stable_hash()}.jpg"
    path.write_bytes(b"\x00" * nbytes)
    os.utime(path, ns=(mtime_ns, mtime_ns))
    store._index[key] = path
    store._fingerprints[address] = fingerprint
    return key, path


# ===========================================================================
# 1. Defaults: an empty mapping yields exactly the documented defaults, with
#    no problem messages.
# ===========================================================================

def test_empty_mapping_gives_documented_defaults_and_no_problems():
    settings, problems = GelemSettings.from_values({})

    assert problems == []
    assert settings.picture_memory_max_bytes == DEFAULT_PICTURE_MEMORY_MAX_BYTES
    assert settings.picture_disk_max_bytes == DEFAULT_PICTURE_DISK_MAX_BYTES
    assert settings.worker_count == DEFAULT_WORKER_COUNT
    assert settings.thumbnail_size == DEFAULT_THUMBNAIL_SIZE
    assert settings.preview_size == DEFAULT_PREVIEW_SIZE


def test_default_construction_matches_from_values_defaults():
    # The dataclass field defaults and the from_values fallback must agree.
    assert GelemSettings() == GelemSettings.from_values({})[0]


# ===========================================================================
# 2. Clamping: a value past a bound is pulled to the bound, and one problem
#    message is recorded per corrected value.
# ===========================================================================

def test_out_of_range_values_are_clamped_with_one_message_each():
    low_mem, _high_mem = PICTURE_MEMORY_MAX_BYTES_RANGE
    low_workers, high_workers = WORKER_COUNT_RANGE
    low_side, _high_side = THUMBNAIL_SIDE_RANGE

    # thumbnail kept small (below its min) so the clamped value stays under
    # the preview's larger side and the cross-field rule does NOT also fire.
    settings, problems = GelemSettings.from_values({
        "picture_memory_max_bytes": str(low_mem - 1),   # below min
        "worker_count": str(high_workers + 50),          # above max
        "thumbnail_size": "5x5",                          # both sides below min
    })

    assert settings.picture_memory_max_bytes == low_mem
    assert settings.worker_count == high_workers
    assert settings.thumbnail_size == (low_side, low_side)

    # One message for each of the three corrections, nothing else.
    assert len(problems) == 3
    assert all(isinstance(message, str) and message for message in problems)


def test_high_side_clamp_is_reported():
    _low_side, high_side = THUMBNAIL_SIDE_RANGE
    settings, problems = GelemSettings.from_values({
        "thumbnail_size": f"{high_side + 500}x{high_side + 500}",
        # preview large enough that only the thumbnail clamp is reported.
        "preview_size": f"{high_side + 1000}x{high_side + 1000}",
    })
    assert settings.thumbnail_size == (high_side, high_side)
    assert len(problems) >= 1


def test_in_range_values_pass_through_untouched():
    settings, problems = GelemSettings.from_values({
        "picture_memory_max_bytes": "268435456",   # 256 MiB, in range
        "picture_disk_max_bytes": "536870912",      # 512 MiB, in range
        "worker_count": "4",
        "thumbnail_size": "128x128",
        "preview_size": "512x512",
    })

    assert problems == []
    assert settings.picture_memory_max_bytes == 268435456
    assert settings.picture_disk_max_bytes == 536870912
    assert settings.worker_count == 4
    assert settings.thumbnail_size == (128, 128)
    assert settings.preview_size == (512, 512)


# ===========================================================================
# 3. An unparseable stored value falls back to its default, records a
#    problem, and never raises.
# ===========================================================================

def test_unparseable_values_fall_back_to_defaults_without_raising():
    settings, problems = GelemSettings.from_values({
        "worker_count": "not-a-number",
        "picture_disk_max_bytes": "",
        "thumbnail_size": "totally bogus",
        "preview_size": "600xNaN",
    })

    assert settings.worker_count == DEFAULT_WORKER_COUNT
    assert settings.picture_disk_max_bytes == DEFAULT_PICTURE_DISK_MAX_BYTES
    assert settings.thumbnail_size == DEFAULT_THUMBNAIL_SIZE
    assert settings.preview_size == DEFAULT_PREVIEW_SIZE

    # One message per unreadable value.
    assert len(problems) == 4


def test_from_values_never_raises_on_junk():
    # A grab-bag of the worst inputs a corrupt store could hand back.
    junk = {
        "picture_memory_max_bytes": object(),
        "picture_disk_max_bytes": None,
        "worker_count": "3.5",
        "thumbnail_size": ["a", "b", "c"],
        "preview_size": 12345,
    }
    settings, problems = GelemSettings.from_values(junk)   # must not raise
    assert isinstance(settings, GelemSettings)
    # None (picture_disk) is "absent", not a problem; the other four are.
    assert len(problems) == 4


# ===========================================================================
# 4. Cross-field rule: a preview smaller than the thumbnail is lifted to the
#    thumbnail size and that is reported.
# ===========================================================================

def test_preview_smaller_than_thumbnail_is_corrected():
    settings, problems = GelemSettings.from_values({
        "thumbnail_size": "800x800",
        "preview_size": "300x300",
    })

    assert settings.thumbnail_size == (800, 800)
    assert settings.preview_size == (800, 800)
    assert any("preview" in message.lower() for message in problems)


def test_preview_equal_to_thumbnail_is_not_corrected():
    settings, problems = GelemSettings.from_values({
        "thumbnail_size": "400x400",
        "preview_size": "400x400",
    })
    assert settings.preview_size == (400, 400)
    assert problems == []


# ===========================================================================
# 5. SettingsStore save/load round trip through a dict-backed fake backend.
# ===========================================================================

def test_settings_store_round_trip_through_dict_backend():
    original = GelemSettings(
        picture_memory_max_bytes=256 * 1024 * 1024,
        picture_disk_max_bytes=700 * 1024 * 1024,
        worker_count=6,
        thumbnail_size=(120, 90),
        preview_size=(800, 640),
    )
    backend = DictBackend()
    store = SettingsStore(backend)

    store.save(original)
    # Everything persisted as a string, sizes as "WxH".
    assert all(isinstance(value, str) for value in backend.data.values())
    assert "120x90" in backend.data.values()

    reloaded, problems = store.load()
    assert problems == []
    assert reloaded == original


def test_settings_store_load_on_empty_backend_gives_defaults():
    reloaded, problems = SettingsStore(DictBackend()).load()
    assert problems == []
    assert reloaded == GelemSettings()


def test_settings_store_key_names_do_not_leak_field_names():
    # The store owns opaque key names; the raw GelemSettings field names
    # should not be what lands in the backend.
    backend = DictBackend()
    SettingsStore(backend).save(GelemSettings())
    assert "worker_count" not in backend.data
    assert any("worker" in key for key in backend.data)   # but namespaced


# ===========================================================================
# 6. ArtifactStore honours the injected sizes in resolution_for and
#    purpose_for_tile_size.
# ===========================================================================

def test_artifact_store_uses_injected_sizes(tmp_path):
    store = ArtifactStore(
        tmp_path / "artifacts",
        thumbnail_size=(64, 64),
        preview_size=(256, 256),
    )

    assert store.resolution_for("thumbnail") == 64
    assert store.resolution_for("preview") == 256

    # The tile-size -> purpose boundary sits at the thumbnail resolution.
    assert store.purpose_for_tile_size(64) == "thumbnail"
    assert store.purpose_for_tile_size(65) == "preview"
    assert store.purpose_for_tile_size(1000) == "preview"


def test_artifact_store_defaults_match_settings_defaults(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    assert store.resolution_for("thumbnail") == max(DEFAULT_THUMBNAIL_SIZE)
    assert store.resolution_for("preview") == max(DEFAULT_PREVIEW_SIZE)


# ===========================================================================
# 7. set_memory_cache_max_bytes actually evicts down to the new ceiling.
# ===========================================================================

def test_set_memory_cache_max_bytes_evicts_now(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")

    # Five 100x100 RGB images -> 30_000 bytes each by the store's estimate.
    for index in range(5):
        key = ArtifactKey(
            f"C:/x/{index}.png",
            SourceFingerprint(size=index + 1, mtime_ns=1),
            "thumbnail",
            store.resolution_for("thumbnail"),
        )
        store._add_to_cache(key, Image.new("RGB", (100, 100)))

    assert len(store._cache) == 5
    assert store._cache_bytes == 5 * 100 * 100 * 3

    # Room for two images only.
    store.set_memory_cache_max_bytes(60_000)

    assert store._cache_bytes <= 60_000
    assert len(store._cache) == 2


def test_set_memory_cache_max_bytes_rejects_below_one(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ValueError):
        store.set_memory_cache_max_bytes(0)


# ===========================================================================
# 8. set_disk_cache_max_bytes actually runs the sweep and returns its result.
# ===========================================================================

def test_set_disk_cache_max_bytes_sweeps_now(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")

    # Three indexed 100-byte JPEGs with distinct mtimes; not in mtime order.
    made = []
    for index, mtime_ns in enumerate((300_000_000, 100_000_000, 200_000_000)):
        key, path = _seed_disk_thumb(
            store, f"C:/m/{index}.png", nbytes=100, mtime_ns=mtime_ns
        )
        made.append((mtime_ns, key, path))

    # 300 bytes on disk, ceiling 250 -> exactly one (the oldest) must go.
    result = store.set_disk_cache_max_bytes(250)

    assert isinstance(result, SweepResult)
    assert result.files_deleted == 1
    assert store._disk_cache_max_bytes == 250

    oldest_mtime, oldest_key, oldest_path = min(made, key=lambda item: item[0])
    assert not oldest_path.exists()
    assert oldest_key not in store._index


def test_set_disk_cache_max_bytes_rejects_below_one(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ValueError):
        store.set_disk_cache_max_bytes(0)


# ===========================================================================
# 9. QSettingsBackend satisfies the get/set/None contract (against a temp
#    ini file, never the real user store).
# ===========================================================================

def test_qsettings_backend_contract(tmp_path):
    from PySide6.QtCore import QSettings
    from settings.qsettings_backend import QSettingsBackend

    backend = QSettingsBackend()
    # Redirect away from the real per-user store.
    backend._qsettings = QSettings(
        str(tmp_path / "gelem.ini"), QSettings.Format.IniFormat
    )

    assert backend.get("artifacts/missing") is None
    backend.set("artifacts/thumbnail_size", "150x150")
    assert backend.get("artifacts/thumbnail_size") == "150x150"

    # A full store round trip works on it too.
    settings_store = SettingsStore(backend)
    settings_store.save(GelemSettings(worker_count=7))
    reloaded, problems = settings_store.load()
    assert problems == []
    assert reloaded.worker_count == 7


# ===========================================================================
# 10. Guardrail: nothing outside main.py and settings/ imports settings/ or
#     QSettings. Inside settings/, only qsettings_backend.py touches PySide6.
# ===========================================================================

def _imported_modules(path: Path) -> list[str]:
    # Some unrelated source files in the tree carry latent SyntaxWarnings
    # (a stray backslash escape in a string); this scan is not the place
    # to surface them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
            names.extend(
                f"{node.module}.{alias.name}" for alias in node.names
            )
    return names


def _production_py_files() -> list[Path]:
    """Every .py file that ships in the app: excludes tests/ and the
    scratch/venv trees."""
    skip_top = {"tests", "docs", ".git", "gelem_project", "test_images"}
    result = []
    for path in PROJECT_ROOT.rglob("*.py"):
        parts = path.relative_to(PROJECT_ROOT).parts
        if parts[0] in skip_top:
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in parts):
            continue
        result.append(path)
    return result


def test_only_main_and_settings_package_import_settings():
    offenders = []
    for path in _production_py_files():
        rel = path.relative_to(PROJECT_ROOT)
        if rel.parts[0] == "settings":
            continue
        if rel.as_posix() == "main.py":
            continue
        modules = _imported_modules(path)
        if any(m == "settings" or m.startswith("settings.") for m in modules):
            offenders.append(str(rel))
        if "QSettings" in path.read_text(encoding="utf-8"):
            offenders.append(f"{rel} (mentions QSettings)")
    assert not offenders, f"forbidden settings/QSettings reach: {offenders}"


def test_only_qsettings_backend_imports_pyside_in_settings_package():
    settings_dir = PROJECT_ROOT / "settings"
    offenders = []
    for path in settings_dir.rglob("*.py"):
        modules = _imported_modules(path)
        touches_qt = any(m == "PySide6" or m.startswith("PySide6.") for m in modules)
        if touches_qt and path.name != "qsettings_backend.py":
            offenders.append(path.name)
    assert not offenders, f"settings/ files importing PySide6: {offenders}"


# ===========================================================================
# 11. Guardrail: main.py's ArtifactStore(...) call in create_app actually
#     forwards all five settings values, each as an attribute read of the
#     loaded settings object -- not a literal and not a module constant.
#
#     This is the one guardrail for the whole item's purpose. Without it,
#     deleting the five keyword arguments from main.py leaves every other
#     test green (they build ArtifactStore directly and never exercise
#     main.py's wiring).
# ===========================================================================

_REQUIRED_ARTIFACT_STORE_KWARGS = {
    "worker_count",
    "disk_cache_max_bytes",
    "memory_cache_max_bytes",
    "thumbnail_size",
    "preview_size",
}


def _create_app_function(main_tree: ast.Module) -> ast.FunctionDef:
    """The `create_app` function node in main.py."""
    for node in ast.walk(main_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "create_app":
            return node
    raise AssertionError("main.py has no create_app function")


def _settings_object_name(create_app: ast.FunctionDef) -> str:
    """The local name bound to the loaded GelemSettings inside create_app.

    Found from the tuple-unpacking assignment whose right-hand side is a
    `<something>.load()` call -- i.e. `gelem_settings, problems =
    settings_store.load()`. The first target name is the settings object.
    """
    for node in ast.walk(create_app):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "load"
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
            and node.targets[0].elts
            and isinstance(node.targets[0].elts[0], ast.Name)
        ):
            return node.targets[0].elts[0].id
    raise AssertionError(
        "could not find `<settings>, <problems> = <store>.load()` in create_app"
    )


def test_main_forwards_all_five_settings_values_into_artifact_store():
    main_path = PROJECT_ROOT / "main.py"
    main_tree = ast.parse(main_path.read_text(encoding="utf-8"))

    create_app = _create_app_function(main_tree)
    settings_name = _settings_object_name(create_app)

    # There is exactly one ArtifactStore(...) call in create_app -- the
    # real-mode branch. The fake-data branch returns early and builds a
    # FakeController instead. If a second one is ever added, this assert
    # fires and the test must be told which call to inspect.
    artifact_store_calls = [
        node
        for node in ast.walk(create_app)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ArtifactStore"
    ]
    assert len(artifact_store_calls) == 1, (
        f"expected exactly one ArtifactStore(...) call in create_app, "
        f"found {len(artifact_store_calls)}"
    )
    call = artifact_store_calls[0]

    keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}

    missing = sorted(_REQUIRED_ARTIFACT_STORE_KWARGS - set(keywords))

    # A value is acceptable only if it reads an attribute off the loaded
    # settings object (e.g. `gelem_settings.worker_count`). A literal or a
    # bare module constant (an ast.Name / ast.Constant) is a hardcoded
    # number sneaking back in.
    hardcoded = []
    for name in sorted(_REQUIRED_ARTIFACT_STORE_KWARGS & set(keywords)):
        value = keywords[name]
        reads_settings_attr = (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == settings_name
        )
        if not reads_settings_attr:
            hardcoded.append(name)

    assert not missing and not hardcoded, (
        "main.py's ArtifactStore(...) does not forward settings correctly:\n"
        f"  missing keyword arguments: {missing or 'none'}\n"
        f"  present but not read from {settings_name}.*: {hardcoded or 'none'}"
    )
