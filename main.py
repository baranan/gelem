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
        (MainWindow instance (already visible), MediaResolver). Every
        mode builds a real MediaResolver now -- --fake-data has no
        Dataset/ArtifactStore/OperatorRegistry, but FakeController still
        decodes real thumbnails from test_images/ through one, the same
        "only the media resolver decodes source media" rule the real
        components follow. The caller (main()) closes the resolver after
        the Qt event loop returns.
    """
    from ui.main_window import MainWindow

    if fake_data:
        # Use FakeController -- no real Dataset/ArtifactStore/OperatorRegistry
        # needed, but real thumbnail decoding still goes through a real
        # MediaResolver (CLAUDE.md's media rule; docs/architecture.md
        # section 9 is the authority for max_open_decoders in real mode --
        # fake mode has no settings store at all, so this is a small,
        # fake-mode-only constant, not a setting).
        from ui.fake_controller import FakeController
        from media.resolver import MediaResolver
        test_folder = Path("test_images")
        if not test_folder.exists():
            test_folder = Path(".")
        resolver = MediaResolver(max_open_decoders=2)
        controller = FakeController(test_folder, resolver=resolver)
        window = MainWindow(controller)
        window.show()
        # Start emitting signals after the window has connected them.
        QTimer.singleShot(100, controller.start)
        return window, resolver

    # Real mode — create all components.
    from models.dataset import Dataset
    from models.query_engine import QueryEngine
    from models.project_paths import create_workspace, default_workspaces_root
    from artifacts.artifact_store import ArtifactStore
    from column_types.registry import ColumnTypeRegistry
    from operators.operator_registry import OperatorRegistry
    from operators.operator_config import build_enabled_operators
    from controller import AppController
    from media.resolver import MediaResolver
    from settings.qsettings_backend import QSettingsBackend
    from settings.settings_store import SettingsStore
    from settings.settings_gateway import SettingsGateway

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

    # The plain-data editing face over the same store. AppController gets
    # this -- never the store or the GelemSettings object -- and only
    # passes calls through to it (docs/architecture.md section 9). The
    # dialog that drives it is P0.5b-2ii-c2b2.
    settings_gateway = SettingsGateway(settings_store)

    # P1.9a: before any project is saved or opened, the app works inside a
    # WORKSPACE folder -- laid out exactly like a saved project (an
    # artifacts/ and an outputs/ subfolder under one root) under the
    # per-user app-data folder, never the OS temp directory. One new
    # workspace folder is created per launch and is never deleted by this
    # item. AppController.save_project() / load_project() replace this
    # ProjectPaths with one rooted at the researcher's chosen folder, at
    # the same point they re-root ArtifactStore (docs/media_architecture.md
    # section 4.7).
    workspace_paths = create_workspace(default_workspaces_root())

    project_root = Path(__file__).resolve().parent

    # ONE shared MediaResolver for the whole app -- the only place a source
    # image or video is decoded (CLAUDE.md's media rules). Injected into
    # both ArtifactStore and AppController as a REQUIRED constructor
    # argument (no default, no None fallback): docs/architecture.md
    # section 9 is the authority for max_open_decoders. Closed after the
    # Qt event loop returns, at the bottom of this function's caller.
    resolver = MediaResolver(max_open_decoders=gelem_settings.max_open_decoders)

    dataset           = Dataset()
    query_engine      = QueryEngine()
    artifact_store    = ArtifactStore(
        workspace_paths.artifacts_dir,
        resolver=resolver,
        worker_count=gelem_settings.worker_count,
        disk_cache_max_bytes=gelem_settings.picture_disk_max_bytes,
        memory_cache_max_bytes=gelem_settings.picture_memory_max_bytes,
        thumbnail_max_side=gelem_settings.thumbnail_max_side,
        preview_max_side=gelem_settings.preview_max_side,
    )
    registry          = ColumnTypeRegistry()
    operator_registry = OperatorRegistry()

    registry.setup_defaults(artifact_store)

    # operators_config.yaml is the single authority for WHICH operators
    # the application offers, and its entry order is the Operators menu
    # order. main.py keeps only the knowledge of HOW to construct each one
    # (see operators/operator_config.py). A missing config file, malformed
    # YAML, or drift between the file and the factory table raises
    # OperatorConfigError here and stops startup rather than quietly
    # changing what the researcher can do.
    #
    # No directories are passed to build_enabled_operators() -- every
    # operator that writes files writes under run.paths.outputs_dir,
    # supplied fresh on every run rather than at construction time.
    operators_config_path = project_root / "operators_config.yaml"
    for operator in build_enabled_operators(operators_config_path):
        operator_registry.register(operator)

    controller = AppController(
        dataset=dataset,
        query_engine=query_engine,
        artifact_store=artifact_store,
        registry=registry,
        operator_registry=operator_registry,
        resolver=resolver,
        settings_gateway=settings_gateway,
        # Handed to every OperatorRun as run.paths, and replaced wholesale
        # by save_project()/load_project().
        project_paths=workspace_paths,
    )

    window = MainWindow(controller)
    window.show()
    return window, resolver


def main():
    """Application entry point."""
    fake_data = "--fake-data" in sys.argv

    app = QApplication(sys.argv)
    app.setApplicationName("Gelem")
    app.setOrganizationName("ResearchLab")

    window, resolver = create_app(fake_data=fake_data)

    try:
        exit_code = app.exec()
    finally:
        # Closes every idle pooled decoder now; one a live decode is still
        # using closes itself once that use ends (media/resolver.py's
        # MediaResolver.close() docstring). None in --fake-data mode.
        if resolver is not None:
            resolver.close()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()