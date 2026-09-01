"""
main.py

Entry point for the Gelem application.

Creates all components, wires them together, and starts the Qt
event loop.

Run with:
    python main.py

Run with fake data (no real images needed):
    python main.py --fake-data
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer


def create_app(fake_data: bool = False):
    """
    Creates and wires all application components.

    Args:
        fake_data: If True, uses FakeController with test images.
                   No real data components are created. This mode
                   is intended for Student A to develop and test
                   UI widgets independently.

    Returns:
        The MainWindow instance (already visible).
    """
    from ui.main_window import MainWindow

    if fake_data:
        # Use FakeController — no real data layer needed.
        from ui.fake_controller import FakeController
        test_folder = Path("test_images")
        if not test_folder.exists():
            test_folder = Path(".")
        controller = FakeController(test_folder)
        window = MainWindow(controller)
        window.show()
        # Start emitting signals after the window has connected them.
        QTimer.singleShot(100, controller.start)
        return window

    # Real mode — create all components.
    import tempfile
    from models.dataset import Dataset
    from models.query_engine import QueryEngine
    from artifacts.artifact_store import ArtifactStore
    from column_types.registry import ColumnTypeRegistry
    from operators.operator_registry import OperatorRegistry
    from operators.blendshapes import BlendshapeOperator
    from operators.blendshape_avatar import BlendshapeAvatarOperator
    from operators.mean_face import MeanFaceOperator
    from operators.plot_operator import PlotOperator
    from operators.summary_stats import SummaryStatsOperator
    from operators.plot_advanced import PlotAdvancedOperator
    from operators.stats_operator import StatsOperator
    from operators.video_frames import VideoFramesOperator
    from controller import AppController
    from settings.qsettings_backend import QSettingsBackend
    from settings.settings_store import SettingsStore

    # Load the machine-tunable settings. The store never raises: a corrupt
    # or out-of-range saved value is clamped or defaulted, and each such
    # correction comes back as a plain-English message we print here.
    # Components receive the resulting plain values as constructor
    # arguments -- never the store or the GelemSettings object itself
    # (docs/architecture.md section 9).
    settings_store = SettingsStore(QSettingsBackend())
    gelem_settings, settings_problems = settings_store.load()
    for problem in settings_problems:
        print(f"[settings] {problem}")

    artifacts_dir = Path(tempfile.gettempdir()) / "gelem_artifacts"
    # This is the pre-project scratch cache -- where thumbnails land
    # before any project has been saved or opened. The first
    # save_project() or load_project() call binds the store to
    # project_path / "artifacts" via ArtifactStore.set_artifacts_dir()
    # (P0.5b-2ii-a), so a saved project keeps its thumbnails and reopens
    # without regenerating them.

    project_root = Path(__file__).resolve().parent
    frames_dir = project_root / "gelem_project" / "frames"
    # TODO: Same migration as artifacts_dir above — once Dataset.save()
    # / load() define a real project folder, point frames_dir at it:
    #     frames_dir = project_path / "frames"
    # TODO (OPTIONAL): let the researcher pick a custom destination per
    # extraction via a folder picker in the parameter dialog.

    plots_dir = project_root / "gelem_project" / "plots"
    # TODO: Same migration as artifacts_dir / frames_dir above — once
    # Dataset.save() / load() define a real project folder, point
    # plots_dir at it:
    #     plots_dir = project_path / "plots"

    dataset           = Dataset()
    query_engine      = QueryEngine()
    artifact_store    = ArtifactStore(
        artifacts_dir,
        worker_count=gelem_settings.worker_count,
        disk_cache_max_bytes=gelem_settings.picture_disk_max_bytes,
        memory_cache_max_bytes=gelem_settings.picture_memory_max_bytes,
        thumbnail_max_side=gelem_settings.thumbnail_max_side,
        preview_max_side=gelem_settings.preview_max_side,
    )
    registry          = ColumnTypeRegistry()
    operator_registry = OperatorRegistry()

    registry.setup_defaults(artifact_store)

    operator_registry.register(BlendshapeOperator())
    operator_registry.register(BlendshapeAvatarOperator())
    operator_registry.register(MeanFaceOperator())
    operator_registry.register(PlotOperator())
    operator_registry.register(SummaryStatsOperator())
    operator_registry.register(PlotAdvancedOperator(output_dir=plots_dir))
    operator_registry.register(StatsOperator())
    operator_registry.register(VideoFramesOperator(output_dir=frames_dir))

    controller = AppController(
        dataset=dataset,
        query_engine=query_engine,
        artifact_store=artifact_store,
        registry=registry,
        operator_registry=operator_registry,
    )

    window = MainWindow(controller)
    window.show()
    return window


def main():
    """Application entry point."""
    fake_data = "--fake-data" in sys.argv

    app = QApplication(sys.argv)
    app.setApplicationName("Gelem")
    app.setOrganizationName("ResearchLab")

    window = create_app(fake_data=fake_data)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()