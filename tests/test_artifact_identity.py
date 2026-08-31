"""
tests/test_artifact_identity.py

P0.5b-1 -- artifact identity (docs/media_architecture.md section 4.5).

A derived image is identified by an ArtifactKey -- canonical media
address, source fingerprint, purpose, resolution, representative-frame
policy, renderer cache version -- not by the row that asked for it.

Written from the work-item spec, not from the implementation. New file
on purpose: tests/test_dataset.py runs everything in it twice, so nothing
new goes there.

Run with:
    python -m pytest tests/test_artifact_identity.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest
from PIL import Image

from artifacts.artifact_store import ArtifactStore, DEFAULT_THUMBNAIL_SIZE

# The thumbnail resolution (max side) is now instance state on an
# ArtifactStore, derived from its configured thumbnail size. A store built
# with defaults uses this; these key-identity tests build ArtifactKeys by
# hand, so they need the same number as a value.
THUMBNAIL_RESOLUTION = max(DEFAULT_THUMBNAIL_SIZE)
from artifacts.artifact_codec import ArtifactCodec, ArtifactCodecError
from media.artifact_key import ArtifactKey, SourceFingerprint
from media.media_address import POLICIES, parse, canonical_key, resolve_source

from models.dataset import Dataset
from models.query_engine import QueryEngine
from column_types.registry import ColumnTypeRegistry
from operators.operator_registry import OperatorRegistry
from controller import AppController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_png(path: Path, colour: tuple[int, int, int]) -> None:
    Image.new("RGB", (240, 240), colour).save(path)


def _address_of(path: Path, root: Path) -> tuple[str, str]:
    """(canonical address, absolute source path) for a media file, the
    same way AppController resolves a cell."""
    return resolve_source(str(path), str(root))


def _wait_for_thumbnail(store: ArtifactStore) -> threading.Event:
    """Attach a chained ready-callback that sets a fresh Event. Returns
    the Event -- call .wait(timeout) then .clear() before the next
    request."""
    event = threading.Event()
    previous = store.on_thumbnail_ready

    def _on_ready(table_name, row_id):
        if previous is not None:
            previous(table_name, row_id)
        event.set()

    store.on_thumbnail_ready = _on_ready
    return event


def _centre_colour(pixmap):
    image = pixmap.toImage()
    return image.pixelColor(image.width() // 2, image.height() // 2)


def _build_controller(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)
    dataset = Dataset()
    dataset.set_registry(registry)
    op_registry = OperatorRegistry()
    controller = AppController(dataset, QueryEngine(), store, registry, op_registry)
    return controller, dataset, store


# ===========================================================================
# 2. A request for a second media column produces its OWN cached artifact
#    under its own key. This is the test that actually guards the fix --
#    the gallery-seam xfail passes even with a broken cache, via the
#    Image.open fallback (see that test's note).
# ===========================================================================

def test_second_media_column_gets_its_own_cached_artifact(tmp_path):
    red_path = tmp_path / "red.png"
    blue_path = tmp_path / "blue.png"
    _solid_png(red_path, (255, 0, 0))
    _solid_png(blue_path, (0, 0, 255))

    store = ArtifactStore(tmp_path / "artifacts")
    event = _wait_for_thumbnail(store)

    red_addr, red_src = _address_of(red_path, tmp_path)
    blue_addr, blue_src = _address_of(blue_path, tmp_path)

    store.request_thumbnail("row1", red_addr, Path(red_src), "frames")
    assert event.wait(timeout=30), "red thumbnail was never generated"
    event.clear()

    store.request_thumbnail("row1", blue_addr, Path(blue_src), "frames")
    assert event.wait(timeout=30), "blue thumbnail was never generated"

    red_thumb = store.get(red_addr, "thumbnail")
    blue_thumb = store.get(blue_addr, "thumbnail")
    assert red_thumb is not None, "no cached artifact for the first column's address"
    assert blue_thumb is not None, "no cached artifact for the SECOND column's address"
    assert red_thumb != blue_thumb, "both addresses share one cache file"

    red_px = Image.open(red_thumb).convert("RGB").getpixel((5, 5))
    blue_px = Image.open(blue_thumb).convert("RGB").getpixel((5, 5))
    assert red_px[0] > red_px[2], f"first column's artifact is not red: {red_px}"
    assert blue_px[2] > blue_px[0], f"second column's artifact is not blue: {blue_px}"


# ===========================================================================
# 3. Two rows in different tables, same file, same purpose, same
#    resolution -> ONE cache entry. (4.5: "this answers media sharing for
#    free".)
# ===========================================================================

def test_same_file_two_tables_share_one_cache_entry(tmp_path):
    source = tmp_path / "shared.png"
    _solid_png(source, (10, 200, 30))

    store = ArtifactStore(tmp_path / "artifacts")
    event = _wait_for_thumbnail(store)
    addr, src = _address_of(source, tmp_path)

    store.request_thumbnail("rowA", addr, Path(src), "tableA")
    assert event.wait(timeout=30)

    first_thumb = store.get(addr, "thumbnail")
    first_preview = store.get(addr, "preview")
    assert first_thumb is not None and first_preview is not None
    jpg_count = len(list((tmp_path / "artifacts").glob("*.jpg")))
    assert jpg_count == 2, f"expected exactly thumbnail + preview, got {jpg_count}"

    # A row in a different table, same file: the store already holds both
    # artifacts for this address, so this short-circuits -- no new file.
    store.request_thumbnail("rowB", addr, Path(src), "tableB")

    assert store.get(addr, "thumbnail") == first_thumb
    assert store.get(addr, "preview") == first_preview
    assert len(list((tmp_path / "artifacts").glob("*.jpg"))) == 2


# ===========================================================================
# 4. Same file, different time range in the address -> different keys.
# ===========================================================================

def test_time_range_in_address_changes_the_key(tmp_path):
    root = str(tmp_path)
    whole = canonical_key(parse("C:/videos/p01.mp4"), root)
    clip = canonical_key(parse("C:/videos/p01.mp4#t=1.000000-2.000000"), root)
    other_clip = canonical_key(parse("C:/videos/p01.mp4#t=2.000000-3.000000"), root)
    assert whole != clip != other_clip
    assert whole != other_clip

    fp = SourceFingerprint(size=1234, mtime_ns=5678)
    k_whole = ArtifactKey(whole, fp, "thumbnail", THUMBNAIL_RESOLUTION)
    k_clip = ArtifactKey(clip, fp, "thumbnail", THUMBNAIL_RESOLUTION)
    k_other = ArtifactKey(other_clip, fp, "thumbnail", THUMBNAIL_RESOLUTION)

    assert k_whole != k_clip
    assert k_clip != k_other
    assert len({k_whole.stable_hash(), k_clip.stable_hash(), k_other.stable_hash()}) == 3


# ===========================================================================
# 5. Same path, changed size or mtime -> different key. Construct keys
#    directly; do not rewrite a real file.
# ===========================================================================

def test_changed_source_fingerprint_changes_the_key():
    address = "C:/data/photo.png"
    base_fp = SourceFingerprint(size=100_000, mtime_ns=1_000_000_000)
    bigger = SourceFingerprint(size=100_001, mtime_ns=1_000_000_000)
    newer = SourceFingerprint(size=100_000, mtime_ns=2_000_000_000)

    base = ArtifactKey(address, base_fp, "thumbnail", 150)
    grown = ArtifactKey(address, bigger, "thumbnail", 150)
    touched = ArtifactKey(address, newer, "thumbnail", 150)

    assert base != grown
    assert base != touched
    assert base.stable_hash() != grown.stable_hash()
    assert base.stable_hash() != touched.stable_hash()


# ===========================================================================
# 6. Load folder A, then folder B: no row of B shows a picture from A.
#
#    P0.5b-3i: load_folder no longer generates thumbnails eagerly, so this
#    drives the demand path -- render (miss -> placeholder, request queued),
#    wait for the worker, render again (hit). The property under test is
#    unchanged: after load_folder(B) reset() has wiped folder A's cache, so
#    the only picture the demand request can produce is folder B's blue one.
# ===========================================================================

def test_load_folder_a_then_b_shows_no_a_picture(qapp, tmp_path):
    folder_a = tmp_path / "a"
    folder_b = tmp_path / "b"
    folder_a.mkdir()
    folder_b.mkdir()
    _solid_png(folder_a / "shot_a.png", (255, 0, 0))
    _solid_png(folder_b / "shot_b.png", (0, 0, 255))

    controller, _, store = _build_controller(tmp_path)
    event = _wait_for_thumbnail(store)

    controller.load_folder(folder_a)
    controller.load_folder(folder_b)  # resets the store

    row_id = controller.get_all_row_ids()[0]
    row = controller.get_row(row_id)

    def _render():
        return controller.render_column_value(
            "full_path", row["full_path"], 150, "thumbnail",
            {"row_id": row_id, "column_name": "full_path"},
        )

    placeholder = _render()  # cache miss -> placeholder, demand request queued
    assert placeholder is not None and not placeholder.isNull()
    assert event.wait(timeout=30), "demand-driven thumbnail was never generated"

    pixmap = _render()  # now a cache hit
    assert pixmap is not None and not pixmap.isNull()
    colour = _centre_colour(pixmap)
    assert colour.blue() > colour.red(), (
        f"a folder-B row rendered a folder-A (red) picture: {colour.getRgb()}"
    )


# ===========================================================================
# 7. Load project A, then project B: same assertion, exercising the
#    reset()-before-load_index() fix.
# ===========================================================================

def test_load_project_a_then_b_shows_no_a_picture(qapp, tmp_path):
    folder_a = tmp_path / "fa"
    folder_b = tmp_path / "fb"
    folder_a.mkdir()
    folder_b.mkdir()
    red = folder_a / "ra.png"
    blue = folder_b / "rb.png"
    _solid_png(red, (255, 0, 0))
    _solid_png(blue, (0, 0, 255))

    project_a = tmp_path / "proj_a"
    project_b = tmp_path / "proj_b"

    controller, _, store = _build_controller(tmp_path)
    event = _wait_for_thumbnail(store)

    def _render_first_row():
        row_id = controller.get_all_row_ids()[0]
        row = controller.get_row(row_id)
        return controller.render_column_value(
            "full_path", row["full_path"], 150, "thumbnail",
            {"row_id": row_id, "column_name": "full_path"},
        )

    # Fully thumbnail and save each project, so its artifact_index.json
    # holds a real entry. P0.5b-3i: the thumbnail is generated on demand
    # when the first row renders, not eagerly by load_folder.
    controller.load_folder(folder_a)
    _render_first_row()  # cache miss -> queues the demand request
    assert event.wait(timeout=30), "folder A thumbnail never generated"
    controller.save_project(project_a)

    event.clear()
    controller.load_folder(folder_b)
    _render_first_row()
    assert event.wait(timeout=30), "folder B thumbnail never generated"
    controller.save_project(project_b)

    # Open project A. load_index() seeds the fingerprint memo from the
    # saved index, so the cache is usable at once -- no wait needed.
    controller.load_project(project_a)
    a_addr, _ = resolve_source(str(red), str(project_a))
    assert store.get(a_addr, "thumbnail") is not None, (
        "project A's persisted artifact was not usable after reopening it"
    )
    colour_a = _centre_colour(_render_first_row())
    assert colour_a.red() > colour_a.blue(), "project A did not render its red picture"

    # Switch to project B. reset() runs before load_index(), so project
    # A's populated cache -- index, memory and fingerprint memo -- is gone.
    controller.load_project(project_b)
    assert store.get(a_addr, "thumbnail") is None, (
        "project A's artifact survived a project switch -- reset() did not "
        "run before load_index()"
    )
    colour_b = _centre_colour(_render_first_row())
    assert colour_b.blue() > colour_b.red(), (
        f"project B rendered project A's picture: {colour_b.getRgb()}"
    )


# ===========================================================================
# 8. The codec refuses a path outside the cache root.
# ===========================================================================

def test_codec_refuses_path_outside_cache_root(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    codec = ArtifactCodec(cache_root)

    # A source media file, deliberately outside the cache.
    outside = tmp_path / "participant_01.mp4"
    outside.write_bytes(b"not really a video")

    with pytest.raises(ArtifactCodecError):
        codec.read_image(outside)

    with pytest.raises(ArtifactCodecError):
        codec.write_jpeg(tmp_path / "escape.jpg", np.zeros((8, 8, 3), dtype=np.uint8))

    # A path inside the root is accepted and round-trips.
    inside = cache_root / "ok.jpg"
    codec.write_jpeg(inside, np.full((8, 8, 3), 128, dtype=np.uint8))
    assert inside.exists()
    assert codec.read_image(inside).size == (8, 8)
    # No temp files left behind on a successful write.
    assert list(cache_root.glob("*.tmp")) == []


# ===========================================================================
# 9. A fingerprint seeded from a persisted index is NOT trusted to
#    short-circuit: the next request re-stats the source, so a file that
#    changed since the project was saved is regenerated, not served stale.
# ===========================================================================

def test_persisted_fingerprint_is_re_stated_on_next_request(tmp_path):
    source = tmp_path / "s.png"
    _solid_png(source, (220, 0, 0))  # red

    store = ArtifactStore(tmp_path / "artifacts")
    event = _wait_for_thumbnail(store)
    addr, src = _address_of(source, tmp_path)

    store.request_thumbnail("r", addr, Path(src), "frames")
    assert event.wait(timeout=30)
    store.save_index(tmp_path)  # persists the red picture's fingerprint

    # The source file is replaced and its mtime bumped.
    _solid_png(source, (0, 0, 220))  # blue
    import os
    import time
    future = time.time() + 30
    os.utime(source, (future, future))

    store2 = ArtifactStore(tmp_path / "artifacts")
    event2 = _wait_for_thumbnail(store2)
    store2.load_index(tmp_path)  # seeds the STALE (red) fingerprint

    # A bare lookup still serves the stale picture -- that window is
    # documented and accepted for display.
    assert store2.get(addr, "thumbnail") is not None

    # But a request does not short-circuit on the unverified fingerprint:
    # the worker re-stats, the fingerprint no longer matches, and the
    # blue picture is regenerated under a new key.
    store2.request_thumbnail("r", addr, Path(src), "frames")
    assert event2.wait(timeout=30)

    fresh = store2.get(addr, "thumbnail")
    pixel = Image.open(fresh).convert("RGB").getpixel((5, 5))
    assert pixel[2] > pixel[0], (
        f"request did not regenerate after the source changed: {pixel}"
    )


# ===========================================================================
# 10. Representative-frame policy is part of stable_hash(), not merely part
#     of dataclass equality.
#
#     No other test in the repo builds an ArtifactKey with a policy other
#     than the "first" default. policy sits in BOTH the dataclass fields
#     (so __eq__ / __hash__ separate two policies) AND the stable_hash()
#     parts tuple (so the on-disk filename separates them too). If it were
#     dropped from stable_hash() alone, two policies would be two distinct
#     index entries mapping to ONE filename on disk, and whichever write
#     ran second would silently overwrite the other policy's picture.
# ===========================================================================
def test_policy_is_part_of_the_stable_hash(tmp_path):
    # A range address, because policy only means anything for a range --
    # "first" frame in the range versus the frame nearest its midpoint.
    address = canonical_key(
        parse("C:/videos/p01.mp4#t=1.000000-4.000000"), str(tmp_path)
    )
    fingerprint = SourceFingerprint(size=4321, mtime_ns=8765)

    assert "first" in POLICIES and "midpoint" in POLICIES

    first = ArtifactKey(
        address, fingerprint, "thumbnail", THUMBNAIL_RESOLUTION, policy="first"
    )
    midpoint = ArtifactKey(
        address, fingerprint, "thumbnail", THUMBNAIL_RESOLUTION, policy="midpoint"
    )

    # Every field except policy is identical between the two keys.
    assert first.canonical_address == midpoint.canonical_address
    assert first.fingerprint == midpoint.fingerprint
    assert first.purpose == midpoint.purpose
    assert first.resolution == midpoint.resolution
    assert first.renderer_version == midpoint.renderer_version
    assert first.policy != midpoint.policy

    # Dataclass equality already tells them apart...
    assert first != midpoint
    # ...but the DISK FILENAME must tell them apart too, or the second
    # policy's picture overwrites the first's under one name.
    assert first.stable_hash() != midpoint.stable_hash(), (
        "policy is in dataclass equality but missing from stable_hash() -- "
        "two policies would share one JPEG on disk"
    )


# ===========================================================================
# 11. The SAME row_id in two different tables, pointing at DIFFERENT media,
#     yields two distinct cached artifacts -- each with its own pixels.
#
#     create_table_from_rows() deliberately keeps row ids when copying
#     rows, so row_id "7" can exist in both "frames" and "segments" and
#     mean two different files. docs/media_architecture.md section 4.5
#     lists this as one of the five ways the old (row_id, artifact_type)
#     key was wrong. test_same_file_two_tables_share_one_cache_entry above
#     covers the opposite requirement (same file, two tables -> one entry);
#     this covers the collision case it never touched.
# ===========================================================================
def test_same_row_id_two_tables_different_media_do_not_collide(tmp_path):
    green_path = tmp_path / "green.png"
    yellow_path = tmp_path / "yellow.png"
    _solid_png(green_path, (0, 200, 0))
    _solid_png(yellow_path, (220, 220, 0))

    store = ArtifactStore(tmp_path / "artifacts")
    event = _wait_for_thumbnail(store)

    green_addr, green_src = _address_of(green_path, tmp_path)
    yellow_addr, yellow_src = _address_of(yellow_path, tmp_path)

    # Same row_id "7"; two different tables; two different source files.
    store.request_thumbnail("7", green_addr, Path(green_src), "frames")
    assert event.wait(timeout=30), "green thumbnail was never generated"
    event.clear()

    store.request_thumbnail("7", yellow_addr, Path(yellow_src), "segments")
    assert event.wait(timeout=30), "yellow thumbnail was never generated"

    green_thumb = store.get(green_addr, "thumbnail")
    yellow_thumb = store.get(yellow_addr, "thumbnail")
    assert green_thumb is not None, "no cached artifact for row 7 in 'frames'"
    assert yellow_thumb is not None, "no cached artifact for row 7 in 'segments'"
    assert green_thumb != yellow_thumb, (
        "row_id 7 in two tables collapsed to one cache file -- the key is "
        "keyed by the row, not by the media address"
    )

    green_px = Image.open(green_thumb).convert("RGB").getpixel((5, 5))
    yellow_px = Image.open(yellow_thumb).convert("RGB").getpixel((5, 5))
    assert green_px[1] > green_px[0] and green_px[1] > green_px[2], (
        f"row 7 / 'frames' artifact is not green: {green_px}"
    )
    assert yellow_px[0] > 150 and yellow_px[1] > 150 and yellow_px[2] < 100, (
        f"row 7 / 'segments' artifact is not yellow: {yellow_px}"
    )
