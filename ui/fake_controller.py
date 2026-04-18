"""
ui/fake_controller.py

FakeController is a stand-in for AppController that allows Student A
to develop and test all UI widgets without needing any real data layer.

It uses real images from the test_images/ folder so the gallery looks
realistic, but it never touches Dataset, QueryEngine, ArtifactStore,
or any operator. All data is hardcoded or generated on the fly.

Usage (already wired into main.py --fake-data):
    from ui.fake_controller import FakeController
    controller = FakeController(test_images_folder)
    window = MainWindow(controller)
"""

from __future__ import annotations
from pathlib import Path
import threading
import tempfile

from PySide6.QtCore import QObject, Signal, QTimer


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
    gallery_updated          = Signal(list)
    grouped_gallery_updated  = Signal(dict)
    row_selected             = Signal(dict)
    columns_updated          = Signal(list)
    tables_updated           = Signal(list)
    thumbnail_ready          = Signal(str)
    row_updated              = Signal(str)
    operator_progress        = Signal(int)
    operator_complete        = Signal(str)
    merge_report_ready       = Signal(object)
    error_occurred           = Signal(str)
    display_result_ready     = Signal(dict)
    table_created            = Signal(str)

    def __init__(self, test_images_folder: Path):
        super().__init__()

        self._folder = test_images_folder

        # Scan for real media files (images and videos).
        from models.dataset import MEDIA_EXTENSIONS
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
            }

        # Column type tags — updated to use current type names.
        # 'media_path' replaces 'image_path'.
        # 'text' replaces 'categorical'.
        self._column_types: dict[str, str] = {
            "full_path":          "media_path",
            "condition":          "text",
            "session_id":         "text",
            "trial_id":           "text",
            "timestamp":          "numeric",
            "bs_jawOpen":         "numeric",
            "bs_mouthSmileLeft":  "numeric",
            "bs_mouthSmileRight": "numeric",
        }

        self._visual_columns: list[str] = ["full_path"]
        self._visible_ids:    list[str] = list(self._row_ids)

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
        self.gallery_updated.emit(self._visible_ids)

        for row_id in self._row_ids:
            self._request_thumbnail(row_id)

    # ── Thumbnail generation ──────────────────────────────────────────

    def _request_thumbnail(self, row_id: str) -> None:
        thread = threading.Thread(
            target=self._generate_thumb,
            args=(row_id,),
            daemon=True,
        )
        thread.start()

    def _generate_thumb(self, row_id: str) -> None:
        """Background thread: generates thumbnail using Pillow or OpenCV."""
        try:
            from pathlib import Path as P
            path = self._path_map.get(row_id)
            if path is None or not path.exists():
                return

            VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

            if path.suffix.lower() in VIDEO_EXT:
                # Extract first frame for video thumbnail.
                import cv2
                cap = cv2.VideoCapture(str(path))
                ok, frame = cap.read()
                cap.release()
                if not ok:
                    return
                import cv2 as _cv2
                frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                from PIL import Image
                img = Image.fromarray(frame_rgb)
            else:
                from PIL import Image
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    img.thumbnail((150, 150), Image.LANCZOS)
                    self._thumb_cache[row_id] = img.copy()
                    self._thumb_queue.append(row_id)
                    return

            img.thumbnail((150, 150), Image.LANCZOS)
            self._thumb_cache[row_id] = img.copy()
            self._thumb_queue.append(row_id)

        except Exception as e:
            print(f"[FakeController] Thumbnail error for {row_id}: {e}")

    def _drain_thumb_queue(self) -> None:
        while self._thumb_queue:
            row_id = self._thumb_queue.pop(0)
            self.thumbnail_ready.emit(row_id)

    # ── Public API — same signatures as AppController ─────────────────

    def load_folder(self, folder_path: Path) -> None:
        print(f"[FakeController] load_folder({folder_path}) — ignored in fake mode")

    def load_csv(self, csv_path: Path, join_on: str, preprocess=None) -> None:
        print(f"[FakeController] load_csv({csv_path.name}) — fake merge")
        from models.dataset import MergeReport
        report = MergeReport(
            total_csv_rows=len(self._row_ids),
            total_image_files=len(self._row_ids),
            matched_rows=len(self._row_ids),
        )
        self.merge_report_ready.emit(report)

    def confirm_merge(self, report) -> None:
        self.columns_updated.emit(list(self._column_types.keys()))
        self.gallery_updated.emit(self._visible_ids)

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
        self.gallery_updated.emit(self._visible_ids)

    def set_group_by(self, column_name) -> None:
        if column_name is None:
            self.gallery_updated.emit(self._visible_ids)
        else:
            n     = len(self._visible_ids)
            third = max(1, n // 3)
            grouped = {
                "positive": self._visible_ids[:third],
                "negative": self._visible_ids[third:2*third],
                "neutral":  self._visible_ids[2*third:],
            }
            self.grouped_gallery_updated.emit(grouped)

    def set_visible_columns(self, column_names: list[str]) -> None:
        self._visual_columns = column_names
        self.gallery_updated.emit(self._visible_ids)

    def get_visible_columns(self) -> list[str]:
        return list(self._visual_columns)

    def select_row(self, row_id: str) -> None:
        metadata = self._metadata.get(row_id, {"row_id": row_id})
        self.row_selected.emit(metadata)

    def run_create_columns(self, operator_name: str, row_ids: list[str]) -> None:
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
                         group_by=None) -> None:
        print(f"[FakeController] run_create_table({operator_name}, group_by={group_by})")
        table_name = f"{operator_name}_result"
        QTimer.singleShot(300, lambda: self.tables_updated.emit(["frames", table_name]))
        QTimer.singleShot(350, lambda: self.table_created.emit(table_name))
        QTimer.singleShot(400, lambda: self.operator_complete.emit(operator_name))

    def run_create_display(self, operator_name: str, row_ids: list[str]) -> None:
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
        self.gallery_updated.emit(self._visible_ids[:5])

    def save_filtered_as_table(self, name: str) -> None:
        print(f"[FakeController] save_filtered_as_table({name}) — fake")
        self.tables_updated.emit(["frames", name])

    def export_csv(self, path: Path, row_ids=None) -> None:
        print(f"[FakeController] export_csv({path}) — not implemented in fake mode")

    def save_project(self, project_path: Path) -> None:
        print(f"[FakeController] save_project({project_path}) — not implemented in fake mode")

    def load_project(self, project_path: Path) -> None:
        print(f"[FakeController] load_project({project_path}) — not implemented in fake mode")

    def get_table_names(self) -> list[str]:
        return ["frames"]

    def get_column_names(self) -> list[str]:
        return list(self._column_types.keys())

    def get_visual_column_names(self) -> list[str]:
        return [
            col for col, tag in self._column_types.items()
            if tag == "media_path"
        ]

    def get_group_values(self, column: str) -> list:
        fake_values = {
            "condition":  ["positive", "negative", "neutral"],
            "session_id": ["S01", "S02", "S03"],
            "trial_id":   [f"T{i:02d}" for i in range(1, 6)],
        }
        return fake_values.get(column, ["A", "B", "C"])

    def get_row(self, row_id: str, _table_name: str = "frames") -> dict:
        return self._metadata.get(row_id, {"row_id": row_id})

    def get_artifact_pixmap(self, row_id: str, artifact_type: str):
        if artifact_type == "thumbnail":
            return self._thumb_cache.get(row_id, None)
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

        For media_path columns in thumbnail mode: converts the cached
        PIL thumbnail to a QPixmap.
        For media_path columns in detail mode: delegates to the real
        renderer so the video player or ZoomableImageView is returned.
        For other columns: returns a colored placeholder.
        """
        if column_name == "full_path" and value:
            if mode == "thumbnail":
                # Use cached PIL thumbnail if available, otherwise load from disk.
                from column_types.renderers import _pil_to_pixmap
                from PIL import Image
                pil_image = self._thumb_cache.get(
                    self._find_row_id_for_path(value)
                )
                if pil_image is None:
                    # Thumbnail not ready yet — load directly from disk as fallback.
                    try:
                        with Image.open(value) as img:
                            img = img.convert("RGB")
                            img.thumbnail((size, size), Image.LANCZOS)
                            return _pil_to_pixmap(img)
                    except Exception:
                        return None
                img = pil_image.copy()
                img.thumbnail((size, size), Image.LANCZOS)
                return _pil_to_pixmap(img)
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
        class FakeOpRegistry:
            def list_create_columns_operators(self):
                return [
                    ("blendshapes",       "Extract blendshapes"),
                    ("blendshape_avatar", "Render blendshape avatar"),
                    ("plot",              "Plot columns"),
                ]
            def list_create_table_operators(self):
                return [("mean_face", "Mean face table")]
            def list_create_display_operators(self):
                return [
                    ("mean_face",     "Mean face (quick view)"),
                    ("summary_stats", "Summary statistics"),
                    ("plot_advanced", "Plot (interactive)"),
                ]
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
