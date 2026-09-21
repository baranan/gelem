"""
ui/fake_controller.py

FakeController is a stand-in for AppController that allows Student A
to develop and test all UI widgets without needing any real data layer.

It uses real images (and videos) from the test_images/ folder so the
gallery looks realistic, but it never touches Dataset, QueryEngine,
ArtifactStore, or any operator. All data is hardcoded or generated on
the fly. It still decodes real media for its thumbnails, so it takes a
MediaResolver the same way the real components do -- CLAUDE.md's media
rule ("only the media resolver decodes source media") applies here too,
since this file is reachable from real main.py via --fake-data, not
test-only scaffolding.

Usage (already wired into main.py --fake-data):
    from ui.fake_controller import FakeController
    from media.resolver import MediaResolver
    resolver = MediaResolver(max_open_decoders=2)
    controller = FakeController(test_images_folder, resolver=resolver)
    window = MainWindow(controller)
"""

from __future__ import annotations
from pathlib import Path
import threading
import tempfile
import uuid

from PySide6.QtCore import QObject, Signal, QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

from models.query_result import GroupSection, QueryResult
from models.notifications import ThumbnailsReady
from models.output_copy import OutputCopyPlan


def _pixels_to_pixmap(pixels, max_side: int) -> QPixmap:
    """A decoded RGB uint8 frame (media.resolver.FramePayload.pixels) to a
    QPixmap scaled to fit within max_side x max_side, keeping aspect
    ratio.

    Pure Qt, on purpose: this file must import no numpy, pandas or PIL
    (it decodes real media now, through the injected MediaResolver, so
    the source-decode guard applies to it -- tests/
    test_source_decode_guard.py). `pixels` is a numpy array, but nothing
    here imports numpy -- it only reads attributes (.shape) and calls a
    method (.tobytes()) on the array object the resolver already handed
    in, exactly as this module already reads attributes off a Path or a
    QPixmap without importing pathlib or PySide6.QtGui's own module a
    second time for that purpose.
    """
    height, width = pixels.shape[0], pixels.shape[1]
    qimage = QImage(
        pixels.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888,
    )
    pixmap = QPixmap.fromImage(qimage)
    return pixmap.scaled(
        max_side, max_side,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class FakeController(QObject):
    """
    A drop-in replacement for AppController that returns hardcoded
    or generated data. Implements the same signals and public methods
    as AppController so MainWindow and all widgets work without changes.

    Student A: you can call any method on self._controller in your
    widgets and it will work with both FakeController and the real
    AppController. Never check which type the controller is.
    """

    # ── Signals — identical to AppController ─────────────────────────
    result_changed           = Signal(object)
    row_selected             = Signal(dict)
    columns_updated          = Signal(list)
    tables_updated           = Signal(list)
    active_table_changed     = Signal(str)
    thumbnails_ready         = Signal(object)
    rows_updated             = Signal(object)
    operator_progress        = Signal(int)
    operator_complete        = Signal(str)
    live_runs_changed        = Signal()
    operator_log_changed     = Signal()
    merge_report_ready       = Signal(object)
    error_occurred           = Signal(str)
    display_result_ready     = Signal(dict)
    table_created            = Signal(str)

    def __init__(self, test_images_folder: Path, *, resolver):
        super().__init__()

        # The one shared MediaResolver (media/resolver.py) -- REQUIRED,
        # no default, matching how the real components take it (CLAUDE.md's
        # media rule; docs/architecture.md section 9). Only _generate_thumb
        # and render_column_value's thumbnail-not-ready fallback use it.
        self._resolver = resolver

        # Resolved to absolute up front: MediaResolver.resolve_frame()
        # requires an absolute address (a real project always absolutises
        # a stored cell before handing it to the resolver -- see
        # AppController._resolve_media_cell), and FakeController has no
        # project-root concept of its own to do that lazily, so a caller
        # passing a relative folder (main.py's Path("test_images")) must
        # not leak a relative path into self._metadata / self._path_map.
        test_images_folder = Path(test_images_folder).resolve()
        self._folder = test_images_folder

        # Scan for real media files (images and videos).
        from media.extensions import MEDIA_EXTENSIONS
        self._image_files: list[Path] = []
        if test_images_folder.exists():
            for f in sorted(test_images_folder.iterdir()):
                if f.suffix.lower() in MEDIA_EXTENSIONS:
                    self._image_files.append(f)

        self._row_ids: list[str] = [
            f"{i+1:06d}" for i in range(len(self._image_files))
        ]

        self._path_map: dict[str, Path] = {
            row_id: path
            for row_id, path in zip(self._row_ids, self._image_files)
        }

        import random
        random.seed(42)
        self._metadata: dict[str, dict] = {}
        conditions = ["positive", "negative", "neutral"]
        sessions   = ["S01", "S02", "S03"]
        for row_id, path in self._path_map.items():
            self._metadata[row_id] = {
                "row_id":             row_id,
                "full_path":          str(path),
                "file_name":          path.name,
                "condition":          random.choice(conditions),
                "session_id":         random.choice(sessions),
                "trial_id":           f"T{random.randint(1, 20):02d}",
                "timestamp":          round(random.uniform(0.0, 30.0), 3),
                "bs_jawOpen":         round(random.uniform(0.0, 0.8), 3),
                "bs_mouthSmileLeft":  round(random.uniform(0.0, 0.6), 3),
                "bs_mouthSmileRight": round(random.uniform(0.0, 0.6), 3),
                "frame_index":        random.randint(0, 240),
                "is_keyframe":        random.choice([True, False]),
            }

        # Second visual column, purely so the visible-columns selector
        # has more than one entry to toggle in fake mode. Each row's
        # "alt_view" points at the *next* row's image, so when both
        # columns are shown the two tiles in each cell are visibly
        # different pictures.
        n = len(self._row_ids)
        for i, row_id in enumerate(self._row_ids):
            if n > 1:
                next_path = self._path_map[self._row_ids[(i + 1) % n]]
                self._metadata[row_id]["alt_view"] = str(next_path)

        # Column type tags — updated to use current type names.
        # 'media_path' replaces 'image_path'.
        # 'text' replaces 'categorical'.
        self._column_types: dict[str, str] = {
            "full_path":          "media_path",
            "alt_view":           "media_path",
            "condition":          "text",
            "session_id":         "text",
            "trial_id":           "text",
            "timestamp":          "numeric",
            "bs_jawOpen":         "numeric",
            "bs_mouthSmileLeft":  "numeric",
            "bs_mouthSmileRight": "numeric",
            "frame_index":        "numeric",
            "is_keyframe":        "boolean_flag",
        }

        self._visual_columns: list[str] = ["full_path"]
        self._visible_ids:    list[str] = list(self._row_ids)

        # P0.4: the fake owns the ordered result the same way the real
        # controller does. _result is a real QueryResult; the accessors
        # below are small real implementations, not stubs.
        self._result:           QueryResult | None       = None
        self._result_index:     dict[str, int]           = {}
        self._displayed_ranges: dict[str, tuple[int, int]] = {}

        self._thumb_dir = Path(tempfile.gettempdir()) / "gelem_fake_thumbs"
        self._thumb_dir.mkdir(exist_ok=True)
        self._thumb_cache: dict[str, object] = {}

        self.__active_table: str = "frames"

        self._thumb_queue: list[str] = []
        self._thumb_timer = QTimer(self)
        self._thumb_timer.setInterval(50)
        self._thumb_timer.timeout.connect(self._drain_thumb_queue)
        self._thumb_timer.start()

    # ── Startup ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Emits initial signals to populate the UI."""
        self.columns_updated.emit(list(self._column_types.keys()))
        self.tables_updated.emit(["frames"])
        self._emit_result(self._visible_ids)

        for row_id in self._row_ids:
            self._request_thumbnail(row_id)

    # ── Ordered result (P0.4) ────────────────────────────────────────

    def _emit_result(self, flat_order, groups=None) -> None:
        """
        Stores a fresh QueryResult and emits result_changed with its
        layout -- the fake's equivalent of AppController._refresh_result.
        """
        row_ids = list(flat_order)
        self._result = QueryResult(
            result_id=str(uuid.uuid4()),
            table_name=self.__active_table,
            row_ids=tuple(row_ids),
            groups=groups,
        )
        self._result_index = {rid: i for i, rid in enumerate(row_ids)}
        self._displayed_ranges = {}
        self.result_changed.emit(self._result.layout())

    def _emit_grouped_result(self, grouped: dict) -> None:
        """Builds one flat order plus group spans from a group->ids dict."""
        flat_order: list[str] = []
        sections: list[GroupSection] = []
        for label, ids in grouped.items():
            start = len(flat_order)
            flat_order.extend(ids)
            sections.append(GroupSection(str(label), start, len(flat_order)))
        self._emit_result(flat_order, tuple(sections))

    def get_result_layout(self):
        if self._result is None:
            from models.query_result import ResultLayout
            return ResultLayout("", self.__active_table, 0, None)
        return self._result.layout()

    def get_visible_row_ids(self) -> list[str]:
        return list(self._result.row_ids) if self._result else []

    def get_row_ids_in_range(self, start: int, stop: int) -> list[str]:
        if self._result is None:
            return []
        n = len(self._result.row_ids)
        lo = max(0, min(start, n))
        hi = max(lo, min(stop, n))
        return list(self._result.row_ids[lo:hi])

    def get_result_index(self, row_id: str):
        return self._result_index.get(row_id)

    def order_by_result(self, row_ids: list[str]) -> list[str]:
        if self._result is None:
            return []
        placed = [
            (self._result_index[rid], rid)
            for rid in row_ids
            if rid in self._result_index
        ]
        placed.sort(key=lambda pair: pair[0])
        return [rid for _, rid in placed]

    def report_displayed_range(self, viewport_key, start, stop, result_id) -> None:
        if self._result is None or result_id != self._result.result_id:
            return
        self._displayed_ranges[viewport_key] = (start, stop)

    def clear_displayed_range(self, viewport_key) -> None:
        self._displayed_ranges.pop(viewport_key, None)

    def get_displayed_ranges(self) -> list[tuple[int, int]]:
        return sorted(self._displayed_ranges.values(), key=lambda r: r[0])

    # ── Thumbnail generation ──────────────────────────────────────────

    def _request_thumbnail(self, row_id: str) -> None:
        thread = threading.Thread(
            target=self._generate_thumb,
            args=(row_id,),
            daemon=True,
        )
        thread.start()

    def _generate_thumb(self, row_id: str) -> None:
        """Background thread: generates a thumbnail through the injected
        MediaResolver. One call handles both an image and a video --
        resolve_frame() dispatches on the path's extension internally and
        returns a representative frame either way (docs/media_architecture.md
        section 3.3), so there is no VIDEO_EXT branch to maintain here.
        """
        try:
            path = self._path_map.get(row_id)
            if path is None or not path.exists():
                return
            payload = self._resolver.resolve_frame(str(path), "display")
            self._thumb_cache[row_id] = _pixels_to_pixmap(payload.pixels, 150)
            self._thumb_queue.append(row_id)
        except Exception as e:
            print(f"[FakeController] Thumbnail error for {row_id}: {e}")

    def _drain_thumb_queue(self) -> None:
        # Drain the whole queue into one ThumbnailsReady batch, mirroring
        # the real controller's batched notification.
        ready: list[str] = []
        while self._thumb_queue:
            ready.append(self._thumb_queue.pop(0))
        if ready:
            self.thumbnails_ready.emit(
                ThumbnailsReady(
                    table_name=self.__active_table,
                    row_ids=tuple(ready),
                )
            )

    # ── Public API — same signatures as AppController ─────────────────

    def load_folder(self, folder_path: Path) -> None:
        print(f"[FakeController] load_folder({folder_path}) — ignored in fake mode")

    def load_csv(
        self,
        csv_path: Path,
        target_table: str,
        csv_key: str,
        target_key: str,
        preprocess=None,
    ) -> None:
        print(f"[FakeController] load_csv({csv_path.name}) — fake merge")
        from models.dataset import MergeReport
        report = MergeReport(
            target_table=target_table,
            total_csv_rows=len(self._row_ids),
            total_target_rows=len(self._row_ids),
            matched_rows=len(self._row_ids),
        )
        self.merge_report_ready.emit(report)

    def confirm_merge(self, report, expand_table_name: str | None = None) -> None:
        # table-name-validation round 5 (narrowed in round 6):
        # ui/main_window.py always calls
        # confirm_merge(report, dialog.chosen_expand_table_name) now, so
        # this must at least accept the same arguments AppController
        # does. Round 5 raised unconditionally, which broke the ORDINARY
        # merge case: load_csv() above already fakes a real result for
        # that path (no would_expand, matched_rows set, nothing pending
        # to commit beyond the fake UI refresh below), so that path is
        # restored exactly as it was before round 5. Only an expansion
        # offer has nothing honest to fake -- report.would_expand is
        # never set by load_csv() today, so this branch is not reachable
        # through the fake controller's own flow yet, but stays as a
        # deliberate refusal rather than silently building a fake new
        # table if that ever changes.
        if report.would_expand:
            raise NotImplementedError(
                "--fake-data mode does not support merging. Run the real "
                "application to merge a CSV."
            )
        self.columns_updated.emit(list(self._column_types.keys()))
        self._emit_result(self._visible_ids)

    def load_csv_as_primary(self, csv_path: Path, image_column=None) -> None:
        print("[FakeController] load_csv_as_primary — not supported in fake mode")

    def set_filters(self, filters, sort_by=None, ascending=True,
                    randomise=False, seed=None) -> None:
        import random
        if randomise:
            rng = random.Random(seed)
            ids = list(self._row_ids)
            rng.shuffle(ids)
            self._visible_ids = ids
        elif filters:
            self._visible_ids = self._row_ids[::2]
        else:
            self._visible_ids = list(self._row_ids)
        self._emit_result(self._visible_ids)

    def set_group_by(self, column_name) -> None:
        if column_name is None:
            self._emit_result(self._visible_ids)
        else:
            n     = len(self._visible_ids)
            third = max(1, n // 3)
            grouped = {
                "positive": self._visible_ids[:third],
                "negative": self._visible_ids[third:2*third],
                "neutral":  self._visible_ids[2*third:],
            }
            self._emit_grouped_result(grouped)

    def set_visible_columns(self, column_names: list[str]) -> None:
        self._visual_columns = column_names
        self._emit_result(self._visible_ids)

    def get_effective_visible_columns(self) -> list[str]:
        """
        Mirrors AppController.get_effective_visible_columns() so UI code
        that calls it works in --fake-data mode too. The fake has no
        None/[] ambiguity to resolve -- has_visible_columns_preference()
        is always True here, so the current visible columns already are
        the effective state.
        """
        return list(self._visual_columns)

    def has_visible_columns_preference(self) -> bool:
        return True

    def clear_visible_columns_preference(self) -> None:
        self._visual_columns = ["full_path"]
        self._emit_result(self._visible_ids)

    def select_row(self, row_id: str) -> None:
        metadata = self._metadata.get(row_id, {"row_id": row_id})
        self.row_selected.emit(metadata)

    def run_create_columns(self, operator_name: str, row_ids: list[str],
                           parameters: dict | None = None) -> None:
        print(f"[FakeController] run_create_columns({operator_name}, {len(row_ids)} rows)")
        total = len(row_ids)
        for i, row_id in enumerate(row_ids):
            percent = int((i + 1) / total * 100)
            QTimer.singleShot(i * 20, lambda p=percent: self.operator_progress.emit(p))
        QTimer.singleShot(
            len(row_ids) * 20 + 100,
            lambda: self.operator_complete.emit(operator_name)
        )

    def run_create_table(self, operator_name: str, row_ids: list[str],
                         parameters: dict | None = None) -> None:
        print(f"[FakeController] run_create_table({operator_name}, parameters={parameters})")
        table_name = f"{operator_name}_result"
        QTimer.singleShot(300, lambda: self.tables_updated.emit(["frames", table_name]))
        QTimer.singleShot(350, lambda: self.table_created.emit(table_name))
        QTimer.singleShot(400, lambda: self.operator_complete.emit(operator_name))

    def run_create_display(self, operator_name: str, row_ids: list[str],
                           parameters: dict | None = None) -> None:
        print(f"[FakeController] run_create_display({operator_name})")
        import numpy as np
        from PIL import Image

        result_path = Path(tempfile.gettempdir()) / "fake_result.png"
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            arr[i, :, 0] = i
            arr[i, :, 1] = 128
            arr[i, :, 2] = 255 - i
        Image.fromarray(arr).save(str(result_path))

        result = {
            "operator_name": operator_name,
            "artifact_path": str(result_path),
            "n_frames":      len(row_ids),
            "summary": {
                "bs_jawOpen": {"mean": 0.42, "sd": 0.12, "min": 0.0, "max": 0.8, "median": 0.40},
                "bs_mouthSmileLeft": {"mean": 0.31, "sd": 0.08, "min": 0.0, "max": 0.6, "median": 0.30},
            }
        }

        # Only the interactive plot operator returns an html_path in reality.
        # Gate the fake html_path on the operator name so the button does not
        # wrongly appear for summary_stats or mean_face results.
        if operator_name == "plot_advanced":
            # Write a tiny standalone HTML file to the temp folder. as_uri()
            # needs an ABSOLUTE path (tempfile.gettempdir() gives one), and the
            # file must exist for the browser to open it rather than 404.
            html_result_path = Path(tempfile.gettempdir()) / "fake_plot.html"
            html_result_path.write_text(
                "<html><body><h2>Fake interactive plot (--fake-data mode)</h2>"
                "<p>Stands in for a Plotly HTML file.</p></body></html>",
                encoding="utf-8",
            )
            result["html_path"] = str(html_result_path) 

        QTimer.singleShot(500, lambda: self.display_result_ready.emit(result))

    def add_computed_column(self, name: str, expression: str,
                             col_type: str = "numeric") -> None:
        import random
        for row_id in self._row_ids:
            self._metadata[row_id][name] = round(random.uniform(0, 1), 3)
        self._column_types[name] = col_type
        self.columns_updated.emit(list(self._column_types.keys()))

    def aggregate(self, name: str, group_by, aggregations: dict) -> None:
        print(f"[FakeController] aggregate({name}) — fake")
        self.tables_updated.emit(["frames", name])

    def set_active_table(self, name: str) -> None:
        self.__active_table = name
        self._emit_result(self._visible_ids[:5])

    def save_filtered_as_table(self, name: str) -> None:
        print(f"[FakeController] save_filtered_as_table({name}) — fake")
        self.tables_updated.emit(["frames", name])

    def export_csv(self, path: Path, row_ids=None) -> None:
        print(f"[FakeController] export_csv({path}) — not implemented in fake mode")

    def plan_output_copy(self, dest_folder: Path) -> OutputCopyPlan:
        """
        Mirrors AppController.plan_output_copy() (P1.9b-1;
        tests/test_fake_controller_contract.py fails without this).
        Fake mode has no Dataset and no ProjectPaths, so there is
        nothing to copy -- an empty plan, the same answer the real
        controller gives when self._project_paths is None.
        """
        return OutputCopyPlan(entries=(), total_bytes=0, conflicts=())

    def save_project(self, project_path: Path) -> bool:
        """
        Mirrors AppController.save_project()'s return type (unsaved-work
        item; tests/test_fake_controller_contract.py fails without this
        method existing at all). Fake mode has no Dataset to write, so
        this never actually saves anything -- always False, the same
        answer a real refusal or failure would give ui/close_prompt.py's
        close flow.
        """
        print(f"[FakeController] save_project({project_path}) — not implemented in fake mode")
        return False

    def load_project(self, project_path: Path) -> None:
        print(f"[FakeController] load_project({project_path}) — not implemented in fake mode")

    def get_table_names(self) -> list[str]:
        return ["frames"]

    def get_active_table(self) -> str:
        """Mirrors AppController.get_active_table()."""
        return self.__active_table

    def get_column_names(self, table_name: str | None = None) -> list[str]:
        # Mirrors AppController.get_column_names(table_name=None). The fake
        # has one table of generated rows, so table_name is accepted and
        # ignored.
        return list(self._column_types.keys())

    def get_visual_column_names(self, table_name: str | None = None) -> list[str]:
        # Mirrors AppController.get_visual_column_names(table_name=None).
        return [
            col for col, tag in self._column_types.items()
            if tag == "media_path"
        ]

    # Mirror the new AppController public methods so UI code that calls
    # them (per issue #32) works in --fake-data mode too.
    #
    # P1.8d-2b-3: each of these was defined twice in this file. The
    # second copy won at class-build time, and its bodies were all
    # inert stubs -- get_column_type in particular returned None for
    # every column, which is why FilterPanel (ui/filter_panel.py:199)
    # built no per-column filter control in --fake-data mode. The live
    # bodies below are kept; the stub copies are gone.

    def get_column_type(self, column_name: str, table_name: str | None = None):
        """
        Returns a column-type object carrying at least `.tag`, mirroring
        AppController.get_column_type(column_name, table_name=None).

        The fake keeps its own tag map (self._column_types) rather than a
        TableSchema, so table_name is accepted and ignored -- there is
        one table. _registry is the fake's _FakeRegistry stand-in, whose
        get() wraps the tag in an object FilterPanel can read .tag off,
        exactly as the real ColumnType does.
        """
        return self._registry.get(column_name)

    def get_all_row_ids(self, table_name: str | None = None) -> list[str]:
        """
        Returns every row_id in the fake's single table.

        The real controller looks up *table_name* in the dataset; the
        fake has only one table of generated rows, so it returns those
        row_ids regardless of which table name is asked for.
        """
        return list(self._row_ids)

    def get_operator(self, operator_name: str):
        """
        Returns the operator registered under *operator_name*, mirroring
        AppController.get_operator(). _op_registry is the fake's
        FakeOpRegistry stand-in, whose get() returns None for every name
        -- matching the real controller when a name is not found.
        """
        return self._op_registry.get(operator_name)

    def get_write_read_conflict_warnings(
        self, operator_name: str, mode_name: str
    ) -> list[str]:
        """
        Mirrors AppController.get_write_read_conflict_warnings(), added
        by P1.12f-2 (tests/test_fake_controller_contract.py fails
        without this). Fake mode never has a second run in flight --
        there is no live-run registry here at all -- so there is never
        a conflict to report.
        """
        return []

    def get_live_runs(self) -> list[dict]:
        """
        Mirrors AppController.get_live_runs() (run-indicator-1,
        run-indicator-1-fix, run-indicator-2;
        tests/test_fake_controller_contract.py fails without this). The
        real controller returns one
        {"operation_id", "label", "table_name", "message"} dict per live
        run -- there is no live-run registry in fake mode --
        live_runs_changed and operator_log_changed are declared above for
        signal parity but neither is ever emitted -- so there is never a
        live run to report.
        """
        return []

    def is_save_blocked(self) -> bool:
        """
        Mirrors AppController.is_save_blocked() (P1.9b-1;
        tests/test_fake_controller_contract.py fails without this). There
        is no live-run registry in fake mode (see get_live_runs() above),
        so saving is never blocked.
        """
        return False

    def has_unsaved_changes(self) -> bool:
        """
        Mirrors AppController.has_unsaved_changes() (unsaved-work item;
        tests/test_fake_controller_contract.py fails without this).
        Fake mode has no Dataset and no table_versions() to compare, so
        there is nothing to report as unsaved -- always False, so
        --fake-data's window closes without ever being asked to save.
        """
        return False

    def cancel_run(self, operation_id: str) -> None:
        """
        Mirrors AppController.cancel_run() (run-indicator-3;
        tests/test_fake_controller_contract.py fails without this). There
        is no live-run registry in fake mode (see get_live_runs() above),
        so there is never a live run to cancel -- the same no-op the real
        controller documents for an operation_id that is not live.
        """
        print(f"[FakeController] cancel_run({operation_id}) — fake, no-op")

    # Settings pass-throughs -- mirror AppController.get_settings_fields
    # and apply_settings so --fake-data does not crash when settings UI
    # calls them. There is no settings store in fake mode, so both are
    # inert: no fields to edit, no corrections to report.
    def get_settings_fields(self) -> list:
        return []

    def apply_settings(self, values: dict) -> list[str]:
        return []

    def get_output_copy_warning_threshold_bytes(self) -> int:
        """
        Mirrors AppController.get_output_copy_warning_threshold_bytes()
        (P1.9b-1; tests/test_fake_controller_contract.py fails without
        this). There is no settings store in fake mode (see the settings
        pass-throughs above), so this is inert, same as get_settings_fields()
        / apply_settings() -- a plain literal, not imported from settings/
        (tests/test_settings.py::test_only_main_and_settings_package_import_settings
        forbids ui/ reaching into settings/ at all; only main.py and the
        settings/ package itself may). 1 GiB matches
        settings.settings.DEFAULT_OUTPUT_COPY_WARNING_THRESHOLD_BYTES --
        keep the two in sync by hand if that default ever changes.
        """
        return 1024 * 1024 * 1024

    def get_group_values(self, column: str) -> list:
        fake_values = {
            "condition":  ["positive", "negative", "neutral"],
            "session_id": ["S01", "S02", "S03"],
            "trial_id":   [f"T{i:02d}" for i in range(1, 6)],
        }
        if column in fake_values:
            return fake_values[column]
        # For every other column (numeric, boolean, ...) return the real
        # sorted unique values from the fake metadata, mirroring the real
        # QueryEngine.get_group_values so range/boolean controls bind to
        # actual data.
        values = {
            md[column]
            for md in self._metadata.values()
            if column in md and md[column] is not None
        }
        try:
            return sorted(values)
        except TypeError:
            return sorted(values, key=str)

    def get_row(self, row_id: str, _table_name: str = "frames") -> dict:
        return self._metadata.get(row_id, {"row_id": row_id})

    def get_artifact_pixmap(
        self,
        address: str,
        purpose: str = "thumbnail",
        resolution: int | None = None,
    ):
        # Mirrors AppController's P0.5b-1 signature: artifacts are
        # identified by media address, not by row. The fake keys its
        # thumbnail cache by row id and has no address map, so it returns
        # None -- exactly what the real store returns for an address with
        # no cache entry. UI code already handles None.
        return None

    def render_column_value(
        self,
        column_name: str,
        value,
        size: int,
        mode: str = "thumbnail",
        context: dict | None = None,
    ):
        """
        Renders a column value as a QPixmap (thumbnail mode) or QWidget
        (detail mode).

        For media_path columns in thumbnail mode: uses the cached
        thumbnail if available, otherwise resolves one directly through
        the injected MediaResolver as a fallback.
        For media_path columns in detail mode: delegates to the real
        renderer so the video player or ZoomableImageView is returned.
        For other columns: returns a colored placeholder.
        """
        if self._column_types.get(column_name) == "media_path" and value:
            if mode == "thumbnail":
                cached = self._thumb_cache.get(self._find_row_id_for_path(value))
                if cached is not None:
                    return cached.scaled(
                        size, size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                # Thumbnail not ready yet -- resolve directly as a fallback.
                try:
                    payload = self._resolver.resolve_frame(value, "display")
                    return _pixels_to_pixmap(payload.pixels, size)
                except Exception:
                    return None
            else:
                # detail mode — use the real renderer so images and
                # videos display correctly even in fake mode.
                from column_types.renderers import make_media_path_renderer
                renderer = make_media_path_renderer(None)
                return renderer(value, size, mode)

        from column_types.registry import _make_placeholder_pixmap, _make_placeholder_widget
        if mode == "detail":
            return _make_placeholder_widget(column_name)
        return _make_placeholder_pixmap(size, column_name)

    def render_result_image(
        self,
        artifact_path: str,
        size: int,
        mode: str = "detail",
    ):
        """
        Mirrors AppController.render_result_image(): a result image is a
        produced file, not a table cell, so it renders straight through the
        media renderer with no schema lookup. DetailWidget.show_result
        calls this in --fake-data mode too (FakeController.run_create_display
        emits a result dict with an 'artifact_path').
        """
        from column_types.renderers import make_media_path_renderer
        return make_media_path_renderer(None)(artifact_path, size, mode)

    def _find_row_id_for_path(self, full_path: str) -> str | None:
        for row_id, path in self._path_map.items():
            if str(path) == full_path:
                return row_id
        return None

    # ── Internal registry access (used by FilterPanel) ────────────────

    class _FakeRegistry:
        """Minimal registry stand-in for FilterPanel's column type checks."""
        def __init__(self, column_types: dict):
            self._types = column_types

        def get(self, column_name: str):
            tag = self._types.get(column_name)
            if tag is None:
                return None
            class FakeType:
                pass
            ft       = FakeType()
            ft.tag   = tag
            ft.label = column_name
            return ft

    @property
    def _registry(self):
        return self._FakeRegistry(self._column_types)

    @property
    def _op_registry(self):
        from operators.descriptor import ExecutionMode

        class FakeOpRegistry:
            # Mirrors OperatorRegistry.list_operators_for_mode(mode):
            # (operator_name, mode label) for every operator declaring
            # that mode, in registration order. Canned here so the
            # Operators menu renders in --fake-data mode.
            _BY_MODE = {
                ExecutionMode.COLUMNS: [
                    ("blendshapes",       "Extract blendshapes"),
                    ("blendshape_avatar", "Render blendshape avatar"),
                    ("plot",              "Plot columns"),
                ],
                ExecutionMode.TABLE: [
                    ("mean_face", "Mean face table"),
                ],
                ExecutionMode.DISPLAY: [
                    ("mean_face",     "Mean face (quick view)"),
                    ("summary_stats", "Summary statistics"),
                    ("plot_advanced", "Plot (interactive)"),
                ],
            }

            def list_operators_for_mode(self, mode):
                return list(self._BY_MODE.get(mode, []))

            def get(self, name):
                return None
        return FakeOpRegistry()

    @property
    def _dataset(self):
        controller = self
        class FakeDataset:
            def get_row(self, row_id, table_name="frames"):
                return controller._metadata.get(row_id, {"row_id": row_id})
            def get_table(self, name="frames"):
                import pandas as pd
                rows = list(controller._metadata.values())
                return pd.DataFrame(rows) if rows else pd.DataFrame()
        return FakeDataset()

    @property
    def _active_table(self):
        return self.__active_table

    @_active_table.setter
    def _active_table(self, value):
        self.__active_table = value
