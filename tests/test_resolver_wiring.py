"""
tests/test_resolver_wiring.py

P1.2c-1 -- wiring the shared MediaResolver into ArtifactStore and the
per-row COLUMNS runner (operators/operator_registry.py).

Written from the work-item specification, not from the implementation.
New file on purpose, alongside tests/test_artifact_identity.py and
tests/test_media_requirement.py rather than inside either -- this item
touches both ArtifactStore and OperatorRegistry and neither existing file
is the natural home for a test that spans both.

Covers:
  (b) ArtifactStore thumbnails a generated video from its first frame
      through the INJECTED resolver, and an EXIF-rotated JPEG thumbnail
      comes out upright.
  (c) A FRAME operator run receives resolver pixels for a plain image row
      and for a #t= video address row.
  (d) ArtifactStore and one FRAME run share the SAME resolver instance,
      built in one place -- pinned by object identity, not by inspecting
      main.py's source text.

Video fixture generation follows tests/test_media_resolver.py's own
pattern (skip cleanly if ffmpeg is missing); that file is not imported,
per its own note that the pattern is copied, not shared.

Run with:
    python -m pytest tests/test_resolver_wiring.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageOps

from models.dataset import Dataset
from models.query_engine import QueryEngine
from artifacts.artifact_store import ArtifactStore
from column_types.registry import ColumnTypeRegistry
from operators.base import BaseOperator
from operators.descriptor import (
    ExecutionMode,
    InputKind,
    InputSpec,
    MediaRequirement,
    ModeDescriptor,
    OperatorDescriptor,
    OutputColumn,
    OutputSpec,
)
from operators.operator_registry import OperatorRegistry
from controller import AppController
from media.artifact_key import ArtifactKey, SourceFingerprint
from media.media_address import resolve_source
from media.resolver import MediaResolver

TEST_IMAGES = project_root / "test_images"
FFMPEG_MISSING = shutil.which("ffmpeg") is None


# ---------------------------------------------------------------------------
# Fixture generation -- copied from tests/test_media_resolver.py's own
# helper of the same purpose, not imported (see that file's own note).
# ---------------------------------------------------------------------------

def _run_ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-y", *args], check=True, capture_output=True)


def _generate_known_frame_video(tmp_path, width=32, height=32, fps=10, duration_s=1):
    """A lossless video whose every pixel of frame N equals N."""
    out_path = tmp_path / "known_frames.mkv"
    _run_ffmpeg([
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration_s}",
        "-vf", "format=gray,geq=lum='N'",
        "-pix_fmt", "gray",
        "-c:v", "ffv1",
        str(out_path),
    ])
    return out_path


# ---------------------------------------------------------------------------
# (b) ArtifactStore decodes through the injected resolver.
# ---------------------------------------------------------------------------

def _wait_for_thumbnail(store: ArtifactStore) -> threading.Event:
    event = threading.Event()
    store.on_thumbnail_ready = lambda table_name, row_id: event.set()
    return event


@pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg is not on PATH")
def test_artifact_store_thumbnails_video_first_frame_through_injected_resolver(tmp_path):
    video_path = _generate_known_frame_video(tmp_path)
    resolver = MediaResolver(max_open_decoders=2)
    try:
        store = ArtifactStore(tmp_path / "artifacts", resolver=resolver)
        address, source = resolve_source(str(video_path), str(tmp_path))

        event = _wait_for_thumbnail(store)
        store.request_thumbnail("r", address, Path(source), "frames")
        assert event.wait(timeout=30), "thumbnail was never generated"

        thumb_path = store.get(address, "thumbnail")
        assert thumb_path is not None, "no cached thumbnail for the video address"

        # Frame 0's pixels are all 0 (the known-frame video's "pixel of
        # frame N equals N" rule) -- a uniform field, so PIL's LANCZOS
        # resize and JPEG re-encode introduce no visible drift. This would
        # fail if the store decoded any frame other than the first, or
        # decoded nothing at all (a grey placeholder is not solid black).
        arr = np.array(Image.open(thumb_path).convert("RGB"), dtype=np.uint8)
        assert arr.max() <= 2, (
            f"expected frame 0 (uniform, near-zero) through the injected "
            f"resolver, got a max pixel value of {arr.max()}"
        )
    finally:
        resolver.close()


def test_artifact_store_thumbnail_of_exif_rotated_jpeg_is_upright(tmp_path):
    # Quadrant colours, non-square, so a 90-degree correction is visible as
    # a dimension swap -- the same fixture shape as
    # tests/test_media_resolver.py::test_still_jpeg_exif_orientation.
    width, height = 40, 24
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (255, 0, 0) if x < width // 2 else (0, 0, 255)

    jpeg_path = tmp_path / "photo.jpg"
    exif = image.getexif()
    exif[274] = 6  # Orientation tag: a common real-camera value.
    image.save(jpeg_path, format="JPEG", quality=100, exif=exif)

    # Independent ground truth: PIL's own exif_transpose on a fresh open.
    expected_size = ImageOps.exif_transpose(Image.open(jpeg_path)).size

    resolver = MediaResolver(max_open_decoders=2)
    try:
        store = ArtifactStore(tmp_path / "artifacts", resolver=resolver)
        address, source = resolve_source(str(jpeg_path), str(tmp_path))

        event = _wait_for_thumbnail(store)
        store.request_thumbnail("r", address, Path(source), "frames")
        assert event.wait(timeout=30), "thumbnail was never generated"

        thumb_path = store.get(address, "thumbnail")
        assert thumb_path is not None
        thumb = Image.open(thumb_path).convert("RGB")

        # Both dimensions are well under the default thumbnail_max_side
        # (150), so PIL.Image.thumbnail() never enlarges or shrinks it --
        # an exact size match proves the EXIF rotation was applied, not
        # merely "some resize happened to land on the right numbers".
        assert thumb.size == expected_size, (
            f"thumbnail is {thumb.size}, expected the EXIF-upright "
            f"{expected_size} -- orientation 6 swaps width and height"
        )
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# Cache invalidation: a picture already on disk under the OLD (pre-
# resolver) renderer_version must never be served as a hit for the same
# address any more -- media/artifact_key.py's RENDERER_CACHE_VERSION is
# the field that must have moved to make that true (docs/media_
# architecture.md sections 4.5 and 4.7: every OTHER ArtifactKey field --
# address, fingerprint, purpose, resolution, policy -- is unchanged by
# P1.2c-1, so only a renderer_version bump can stop an old picture from
# still matching).
# ---------------------------------------------------------------------------

def test_stale_pre_resolver_cache_is_invalidated_by_renderer_version(tmp_path):
    # Same EXIF-rotated fixture as the upright test above: orientation 6
    # swaps a 40x24 source to 24x40, a real dimension change, not just
    # relabelling.
    width, height = 40, 24
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (255, 0, 0) if x < width // 2 else (0, 0, 255)
    jpeg_path = tmp_path / "photo.jpg"
    exif = image.getexif()
    exif[274] = 6
    image.save(jpeg_path, format="JPEG", quality=100, exif=exif)

    resolver = MediaResolver(max_open_decoders=2)
    try:
        store = ArtifactStore(tmp_path / "artifacts", resolver=resolver)
        address, source = resolve_source(str(jpeg_path), str(tmp_path))

        stat = jpeg_path.stat()
        fingerprint = SourceFingerprint(size=stat.st_size, mtime_ns=stat.st_mtime_ns)

        # Seed BOTH the thumbnail and preview entries under renderer_
        # version=1 -- the version every picture made by the pre-P1.2c-1
        # PIL/cv2 decode (no EXIF correction) was keyed under -- with a
        # deliberately WRONG picture (still 40x24, i.e. still sideways)
        # standing in for what that old decode actually produced. Both
        # purposes are seeded and marked "verified" so
        # request_thumbnail()'s synchronous short-circuit -- the exact
        # path that must stop firing for a stale entry -- is genuinely
        # exercised, not merely a slow-path regeneration that happens to
        # also cover the bug.
        wrong_picture = Image.new("RGB", (width, height), (0, 255, 0))
        old_paths: dict[str, Path] = {}
        for purpose in ("thumbnail", "preview"):
            old_key = ArtifactKey(
                address, fingerprint, purpose,
                store.resolution_for(purpose), renderer_version=1,
            )
            old_path = store._dir / f"{old_key.stable_hash()}.jpg"
            wrong_picture.save(old_path, format="JPEG", quality=100)
            store._index[old_key] = old_path
            old_paths[purpose] = old_path
        store._fingerprints[address] = fingerprint
        store._verified.add(address)

        event = _wait_for_thumbnail(store)
        store.request_thumbnail("r", address, Path(source), "frames")
        assert event.wait(timeout=30), "no ready callback fired at all"

        new_thumb_path = store.get(address, "thumbnail")
        assert new_thumb_path is not None, (
            "no thumbnail is cached under the CURRENT renderer version -- "
            "request_thumbnail() short-circuited on the stale version-1 "
            "entry instead of queuing a fresh decode"
        )
        assert new_thumb_path != old_paths["thumbnail"], (
            "the current-version lookup returned the OLD (stale, "
            "pre-resolver) picture -- RENDERER_CACHE_VERSION was not "
            "advanced, so the old entry still matches"
        )

        regenerated = Image.open(new_thumb_path).convert("RGB")
        assert regenerated.size == (height, width), (
            f"regenerated thumbnail is {regenerated.size}, expected the "
            f"EXIF-upright {(height, width)} -- served from the stale "
            f"cache instead of being regenerated through the resolver"
        )
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# (c) The per-row COLUMNS runner hands a FRAME operator resolver pixels,
#     for a plain image row and for a #t= video address row.
# ---------------------------------------------------------------------------

def _columns_descriptor(name, media_requirement):
    return OperatorDescriptor(
        name=name,
        version="1.0",
        description="Test double recording the media it is handed.",
        modes=(
            ModeDescriptor(
                mode=ExecutionMode.COLUMNS,
                label="Recording",
                inputs=(
                    InputSpec(
                        name="active_table",
                        label="Active table",
                        kind=InputKind.ACTIVE_TABLE,
                    ),
                ),
                media_requirement=media_requirement,
                parameters=(),
                output=OutputSpec(
                    columns=(OutputColumn(name="out", type_tag="numeric"),)
                ),
            ),
        ),
    )


class _RecordingOperator(BaseOperator):
    """Records the media (and, for (d), the run.resolver) it is handed."""

    def __init__(self, name="recording_op"):
        super().__init__()
        self.name = name
        self.descriptor = _columns_descriptor(name, MediaRequirement.FRAME)
        self.media_by_row: dict = {}
        self.resolvers_seen: list = []

    def create_columns(self, row_id, media, metadata, run):
        self.media_by_row[row_id] = media
        self.resolvers_seen.append(run.resolver)
        return {"out": 1.0}


def _run_columns_and_wait(controller, operator_name, row_ids, monkeypatch):
    """Start a create_columns run, join every worker thread it spawned,
    then pump the controller's drain by hand (no Qt event loop here) --
    the same pattern tests/test_media_requirement.py uses."""
    created: list[threading.Thread] = []
    real_thread = threading.Thread

    class _Tracked(real_thread):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    monkeypatch.setattr(threading, "Thread", _Tracked)
    try:
        controller.run_create_columns(operator_name, row_ids)
    finally:
        monkeypatch.setattr(threading, "Thread", real_thread)

    for thread in created:
        thread.join(timeout=10)
        assert not thread.is_alive(), "worker thread did not finish in time"

    for _ in range(4):
        controller._drain_queues()


def _make_controller(tmp_path, *, resolver=None):
    resolver = resolver or MediaResolver(max_open_decoders=4)
    store = ArtifactStore(tmp_path / "artifacts", resolver=resolver)
    registry = ColumnTypeRegistry()
    registry.setup_defaults(store)

    dataset = Dataset()
    dataset.load_folder(TEST_IMAGES)

    op_registry = OperatorRegistry()
    controller = AppController(
        dataset, QueryEngine(), store, registry, op_registry, resolver=resolver
    )
    controller.set_filters([])
    return controller, dataset, op_registry, resolver


@pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg is not on PATH")
def test_frame_operator_receives_resolver_pixels_for_image_and_time_video(
    tmp_path, monkeypatch
):
    controller, dataset, op_registry, resolver = _make_controller(tmp_path)

    # Row A: an ordinary image row from load_folder, untouched.
    image_row_id = controller.get_visible_row_ids()[0]
    image_row = dataset.get_row(image_row_id, "frames")
    image_address, _ = resolve_source(image_row["full_path"], str(tmp_path))

    # Row B: a second row whose full_path is overwritten to a #t= address
    # into a generated video -- apply_row_updates canonicalises it exactly
    # as a fresh import would (CLAUDE.md's media canonicalisation rule).
    video_row_id = controller.get_visible_row_ids()[1]
    video_path = _generate_known_frame_video(tmp_path, fps=10, duration_s=1)
    # fps=10, t=0.5s -> frame index 5 -> every pixel of that frame is 5.
    dataset.apply_row_updates(
        "frames", {video_row_id: {"full_path": f"{video_path.as_posix()}#t=0.500000"}}
    )
    video_row = dataset.get_row(video_row_id, "frames")
    video_address, _ = resolve_source(video_row["full_path"], str(tmp_path))

    op = _RecordingOperator()
    op_registry.register(op)

    _run_columns_and_wait(
        controller, op.name, [image_row_id, video_row_id], monkeypatch
    )

    assert set(op.media_by_row) == {image_row_id, video_row_id}

    # Image row: the registry's pixels equal an independent direct decode
    # of the same address through the same resolver.
    expected_image = resolver.resolve_frame(image_address, "analysis").pixels
    np.testing.assert_array_equal(op.media_by_row[image_row_id], expected_image)

    # Video row: the registry honoured the #t= selector -- frame 5's
    # pixels are uniformly 5, not the first frame (which would be 0).
    video_pixels = op.media_by_row[video_row_id]
    assert isinstance(video_pixels, np.ndarray)
    assert video_pixels.dtype == np.uint8
    assert video_pixels.max() <= 6 and video_pixels.min() >= 4, (
        f"expected frame 5 (uniform, value~5) from the #t=0.5 address, "
        f"got min={video_pixels.min()} max={video_pixels.max()}"
    )

    resolver.close()


# ---------------------------------------------------------------------------
# (d) ArtifactStore and one FRAME run share the SAME resolver instance,
#     built in one place. Pinned by identity, not by main.py's text.
# ---------------------------------------------------------------------------

def test_artifact_store_and_a_frame_run_share_one_resolver_instance(
    tmp_path, monkeypatch
):
    resolver = MediaResolver(max_open_decoders=2)
    try:
        controller, dataset, op_registry, _ = _make_controller(
            tmp_path, resolver=resolver
        )

        op = _RecordingOperator("identity_op")
        op_registry.register(op)
        row_ids = controller.get_visible_row_ids()[:1]

        _run_columns_and_wait(controller, op.name, row_ids, monkeypatch)

        assert op.resolvers_seen, "create_columns() was never called"
        # Every FRAME run's run.resolver IS the resolver ArtifactStore was
        # built with -- one instance, built in one place, shared by both.
        assert all(seen is resolver for seen in op.resolvers_seen)
        assert controller._store._resolver is resolver
        assert controller._resolver is resolver
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# Follow-up: AppController.run_create_columns absolutises a FRAME run's
# full_path cells on `snapshot` before the worker starts (see the block
# above the model_lifecycle check). `snapshot` comes from
# Dataset.snapshot_rows(), which returns `df.iloc[positions].copy()` --
# models/dataset.py's own explicit copy, not the table Dataset stores. This
# test proves that in behaviour: Dataset's own row accessor must report the
# ORIGINAL (still relative-shaped, uncanonicalised-to-absolute) cell after
# a FRAME run, not the absolute address the run resolved for its own use.
# ---------------------------------------------------------------------------

def test_frame_run_does_not_rewrite_the_stored_table_full_path_cell(
    tmp_path, monkeypatch
):
    controller, dataset, op_registry, resolver = _make_controller(tmp_path)
    try:
        row_id = controller.get_visible_row_ids()[0]
        stored_before = dataset.get_row(row_id, "frames")["full_path"]

        op = _RecordingOperator("no_mutate_op")
        op_registry.register(op)
        _run_columns_and_wait(controller, op.name, [row_id], monkeypatch)

        # The run did decode something (otherwise this test would prove
        # nothing about the path it claims to guard).
        assert row_id in op.media_by_row

        stored_after = dataset.get_row(row_id, "frames")["full_path"]
        assert stored_after == stored_before, (
            "Dataset's own stored table changed after a FRAME run -- the "
            "controller's full_path absolutisation leaked into Dataset's "
            "table instead of staying on the run's private snapshot copy"
        )
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# Follow-up 2: a missing media cell (NaN -- the common case for a
# partially populated merge) must stay blank through FRAME pre-resolution,
# never turn into a plausible-looking but wrong absolute path such as
# "<project_root>/nan" (str(float('nan')) == "nan", which the address
# parser happily accepts as a one-segment relative path).
# ---------------------------------------------------------------------------

def test_frame_run_leaves_a_blank_media_cell_blank_and_skips_the_row(
    tmp_path, monkeypatch, capsys
):
    controller, dataset, op_registry, resolver = _make_controller(tmp_path)
    try:
        row_ids = controller.get_visible_row_ids()
        blank_row_id, ok_row_id = row_ids[0], row_ids[1]

        dataset.apply_row_updates(
            "frames", {blank_row_id: {"full_path": float("nan")}}
        )
        assert pd.isna(dataset.get_row(blank_row_id, "frames")["full_path"])

        op = _RecordingOperator("blank_media_op")
        op_registry.register(op)
        capsys.readouterr()  # discard registration/setup output
        _run_columns_and_wait(
            controller, op.name, [blank_row_id, ok_row_id], monkeypatch
        )
        printed = capsys.readouterr().out

        # The blank row was skipped -- no media handed to the operator --
        # while the ordinary row still decoded normally, proving the skip
        # is specific to the blank cell, not a broken run.
        assert blank_row_id not in op.media_by_row
        assert ok_row_id in op.media_by_row

        # Nothing resolved the blank cell into a bogus path: no printed
        # message names a file path ending in "nan" (what
        # "<project_root>/nan" or a bare "nan" would look like).
        assert "nan" not in printed.lower(), (
            f"a blank media cell was resolved into something naming "
            f"'nan' instead of being left blank: {printed!r}"
        )

        # Dataset's own stored table still holds the original NaN --
        # untouched, same guarantee as the non-blank case.
        assert pd.isna(dataset.get_row(blank_row_id, "frames")["full_path"])
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# Follow-up: a FRAME run must resolve a stored media cell against the SAME
# base the display path uses (AppController._project_root), never against
# ProjectPaths (run.paths -- where an operator WRITES its outputs). The two
# re-root on different events: save_project() (Save As) moves ProjectPaths
# to the new folder but deliberately leaves _project_root where it was,
# since Dataset.save() never rewrites the in-memory cells (only the ON-DISK
# copy is relativised -- see models/dataset.py's _rewrite_media_cell and
# AppController.save_project()'s own note). A cell that is still stored
# relative (the common case for a CSV-imported path, which is never
# absolutised at import time -- Dataset.load_csv_as_primary just
# canonicalise_cell()s it) must therefore keep resolving against the
# ORIGINAL folder after a Save As, not the new one.
# ---------------------------------------------------------------------------

def test_frame_run_resolves_a_relative_cell_against_project_root_not_project_paths(
    tmp_path, monkeypatch
):
    original_folder = tmp_path / "original"
    original_folder.mkdir()
    image_path = original_folder / "photo.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(image_path)

    csv_path = original_folder / "data.csv"
    pd.DataFrame({"image": ["photo.png"], "label": ["x"]}).to_csv(
        csv_path, index=False
    )

    resolver = MediaResolver(max_open_decoders=2)
    try:
        store = ArtifactStore(tmp_path / "artifacts", resolver=resolver)
        registry = ColumnTypeRegistry()
        registry.setup_defaults(store)
        dataset = Dataset()
        op_registry = OperatorRegistry()
        controller = AppController(
            dataset, QueryEngine(), store, registry, op_registry, resolver=resolver
        )

        # Loads through the CONTROLLER (not dataset.load_folder directly),
        # so _project_root is set the way a real CSV import sets it --
        # the CSV's own folder -- and the image column is stored via
        # canonicalise_cell(), which does NOT absolutise: "photo.png"
        # stays relative, resolved against _project_root at read time.
        controller.load_csv_as_primary(csv_path, image_column="image")
        row_id = controller.get_visible_row_ids()[0]
        stored_cell = dataset.get_row(row_id, "frames")["full_path"]
        assert "/" not in stored_cell and "\\" not in stored_cell, (
            f"test setup did not produce a relative cell: {stored_cell!r}"
        )
        assert controller._project_root == original_folder

        # Save As to a DIFFERENT, otherwise-empty folder. ProjectPaths
        # (run.paths) now names it; _project_root must not move (pinned
        # separately by tests/test_controller_project_paths.py).
        new_folder = tmp_path / "saved_elsewhere"
        assert controller.save_project(new_folder) is True
        assert controller._project_paths.root == new_folder
        assert controller._project_root == original_folder
        # The image was never copied into new_folder -- only a Save-As
        # target for OPERATOR OUTPUTS, which this row's cell is not.
        assert not (new_folder / "photo.png").exists()

        op = _RecordingOperator("relative_cell_op")
        op_registry.register(op)
        _run_columns_and_wait(controller, op.name, [row_id], monkeypatch)

        assert row_id in op.media_by_row, (
            "the FRAME run never decoded the row -- it resolved the "
            "relative cell against the wrong base (run.paths.root, which "
            "Save As moved) instead of _project_root (which did not move)"
        )
        expected = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        np.testing.assert_array_equal(op.media_by_row[row_id], expected)
    finally:
        resolver.close()


# ---------------------------------------------------------------------------
# Follow-up 3: the identity test above builds its OWN resolver and hands it
# to a test helper -- it never touches main.py, so it cannot catch main.py
# itself building two resolvers and giving one to ArtifactStore and a
# different one to AppController. This test instead calls main.create_app()
# in real (non-fake) mode -- the actual production construction path -- and
# checks the components it returns.
#
# Three real side effects of create_app() are neutralised so the test
# touches no shared state: the real per-user QSettings store (registry /
# plist / ini file), the real per-user workspace folder under AppData, and
# a real, visible MainWindow. All three names -- QSettingsBackend,
# default_workspaces_root, MainWindow -- are imported LOCALLY inside
# create_app's function body, so each is re-resolved from ITS OWN
# module's namespace on every call; patching the attribute there (not on
# `main`) is what actually takes effect. MediaResolver is patched the same
# way, not to change its behaviour, but to count real construction calls.
# ---------------------------------------------------------------------------

class _DictSettingsBackend:
    """The same minimal get/set backend test_settings.py's DictBackend is,
    kept local here so this file does not import that module for one
    class. Stands in for QSettingsBackend so the test never touches the
    real per-user settings store."""

    def __init__(self):
        self._data: dict[str, str] = {}

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = str(value)


class _FakeMainWindow:
    """Stands in for ui.main_window.MainWindow. This test is about
    component construction and wiring, not the UI -- it never shows a
    real widget, so it needs no isolated run_tests.py group."""

    def __init__(self, controller):
        self.controller = controller

    def show(self) -> None:
        pass


def test_main_create_app_wires_one_resolver_into_store_and_controller(
    tmp_path, monkeypatch, qapp
):
    import main as main_module
    from media.resolver import MediaResolver as RealMediaResolver

    monkeypatch.setattr(
        "settings.qsettings_backend.QSettingsBackend", _DictSettingsBackend
    )
    monkeypatch.setattr(
        "models.project_paths.default_workspaces_root", lambda: tmp_path
    )
    monkeypatch.setattr("ui.main_window.MainWindow", _FakeMainWindow)

    constructed: list = []

    class _CountingMediaResolver(RealMediaResolver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            constructed.append(self)

    monkeypatch.setattr("media.resolver.MediaResolver", _CountingMediaResolver)

    window, resolver = main_module.create_app()
    try:
        assert len(constructed) == 1, (
            f"expected exactly one MediaResolver constructed by "
            f"create_app(), got {len(constructed)}"
        )
        assert resolver is constructed[0]

        controller = window.controller
        assert controller._resolver is resolver, (
            "AppController was built with a different MediaResolver than "
            "the one create_app() returned"
        )
        assert controller._store._resolver is resolver, (
            "ArtifactStore and AppController do not share the same "
            "MediaResolver instance -- create_app() built or wired a "
            "second one"
        )
    finally:
        resolver.close()
