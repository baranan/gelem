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
    from operators.thumbnail import ThumbnailOperator
    from operators.blendshapes import BlendshapeOperator
    from operators.blendshape_avatar import BlendshapeAvatarOperator
    from operators.mean_face import MeanFaceOperator
    from operators.plot_operator import PlotOperator
    from operators.summary_stats import SummaryStatsOperator
    from operators.plot_advanced import PlotAdvancedOperator
    from operators.stats_operator import StatsOperator
    from controller import AppController

    artifacts_dir = Path(tempfile.gettempdir()) / "gelem_artifacts"
    # TODO: When Student B implements save() and load(), replace this
    # temp folder with a real project folder chosen by the researcher.
    # The artifacts folder should live inside the project folder:
    #     artifacts_dir = project_path / "artifacts"
    # This ensures thumbnails are preserved when a project is saved and
    # reopened. The temp folder is only appropriate during development.

    dataset           = Dataset()
    query_engine      = QueryEngine()
    artifact_store    = ArtifactStore(artifacts_dir)
    registry          = ColumnTypeRegistry()
    operator_registry = OperatorRegistry()

    registry.setup_defaults(artifact_store)

    operator_registry.register(ThumbnailOperator())
    operator_registry.register(BlendshapeOperator())
    operator_registry.register(BlendshapeAvatarOperator())
    operator_registry.register(MeanFaceOperator())
    operator_registry.register(PlotOperator())
    operator_registry.register(SummaryStatsOperator())
    operator_registry.register(PlotAdvancedOperator())
    operator_registry.register(StatsOperator())

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