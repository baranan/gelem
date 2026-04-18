"""
ui/main_window.py

MainWindow is the top-level application window. It holds all the
major widgets and connects them to AppController signals.

Layout:
    Left:   FilterPanel (sidebar)
    Centre: Gallery area (one or more GalleryWidgets)
    Right:  DetailWidget + ResultsPanel (tabbed)

Student A is responsible for implementing this class.
"""

from __future__ import annotations
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QLabel, QComboBox, QToolBar,
    QFileDialog, QMessageBox, QTabWidget, QInputDialog
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

from ui.gallery_widget import GalleryWidget
from ui.filter_panel import FilterPanel
from ui.detail_widget import DetailWidget
from ui.results_panel import ResultsPanel
from ui.run_operator_dialog import RunOperatorDialog
from ui.save_table_dialog import SaveTableDialog


class MainWindow(QMainWindow):
    """
    Top-level application window.

    Connects AppController signals to UI widgets and routes UI events
    back to AppController methods.

    TODO (Student A): Implement grouped gallery view (multiple
    GalleryWidgets in a shared scroll area).
    TODO (Student A): Implement the column selector for visible columns.
    """

    def __init__(self, controller):
        """
        Creates the MainWindow and all child widgets.

        Args:
            controller: The AppController instance.
        """
        super().__init__()
        self._controller = controller
        self._galleries: list[GalleryWidget] = []

        self.setWindowTitle("Gelem — Visual Data Explorer")
        self.resize(1400, 900)

        self._build_menu()
        self._build_toolbar()
        self._build_central_widget()
        self._connect_signals()

    # ── Building the UI ───────────────────────────────────────────────

    def _build_menu(self) -> None:
        """Creates the menu bar."""
        menubar = self.menuBar()

        # File menu.
        file_menu = menubar.addMenu("File")

        new_from_folder_action = QAction("New project from folder...", self)
        new_from_folder_action.triggered.connect(self._on_new_from_folder)
        file_menu.addAction(new_from_folder_action)

        new_from_csv_action = QAction("New project from CSV...", self)
        new_from_csv_action.triggered.connect(self._on_new_from_csv)
        file_menu.addAction(new_from_csv_action)

        file_menu.addSeparator()

        open_action = QAction("Open existing project...", self)
        open_action.triggered.connect(self._on_load_project)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        csv_action = QAction("Merge CSV...", self)
        csv_action.triggered.connect(self._on_merge_csv)
        file_menu.addAction(csv_action)

        file_menu.addSeparator()

        save_action = QAction("Save project...", self)
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        export_action = QAction("Export CSV...", self)
        export_action.triggered.connect(self._on_export_csv)
        file_menu.addAction(export_action)

        save_filtered_action = QAction(
            "Save filtered set as new table...", self
        )
        save_filtered_action.triggered.connect(self._on_save_filtered_set)
        file_menu.addAction(save_filtered_action)

        # Operators menu — rebuilt every time it is opened so it always
        # reflects the currently registered operators.
        self._operators_menu = menubar.addMenu("Operators")
        self._operators_menu.aboutToShow.connect(self._refresh_operators_menu)

    def _build_toolbar(self) -> None:
        """Creates the toolbar with table selector."""
        toolbar = QToolBar("Main toolbar")
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel("Table: "))
        self._table_combo = QComboBox()
        self._table_combo.addItem("frames")
        self._table_combo.currentTextChanged.connect(
            self._controller.set_active_table
        )
        toolbar.addWidget(self._table_combo)

    def _build_central_widget(self) -> None:
        """
        Builds the main three-panel layout:
            Left:   FilterPanel
            Centre: Gallery area
            Right:  Detail/Stats/Results tabs
        """
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left: FilterPanel.
        self._filter_panel = FilterPanel(self._controller)
        splitter.addWidget(self._filter_panel)

        # Centre: Gallery area.
        gallery_container = QWidget()
        gallery_layout    = QVBoxLayout(gallery_container)
        gallery_layout.setContentsMargins(0, 0, 0, 0)

        self._main_gallery = GalleryWidget(self._controller)
        self._galleries    = [self._main_gallery]
        gallery_layout.addWidget(self._main_gallery)
        splitter.addWidget(gallery_container)

        # Right: tabbed panels — Detail and Results only.
        right_tabs = QTabWidget()

        self._detail_widget = DetailWidget(self._controller)
        right_tabs.addTab(self._detail_widget, "Detail")

        self._results_panel = ResultsPanel(self._controller)
        right_tabs.addTab(self._results_panel, "Results")

        splitter.addWidget(right_tabs)
        splitter.setSizes([220, 800, 380])

    # ── Operators menu ────────────────────────────────────────────────

    def _refresh_operators_menu(self) -> None:
        """
        Rebuilds the Operators menu just before it is shown.
        Builds three sections from the operator labels registered
        in OperatorRegistry — no hardcoded operator names here.

        Section 1: "Add columns" — operators with create_columns_label
        Section 2: "Create table" — operators with create_table_label
        Section 3: "Show result" — operators with create_display_label
        """
        self._operators_menu.clear()

        columns_ops = self._controller._op_registry.list_create_columns_operators()
        table_ops   = self._controller._op_registry.list_create_table_operators()
        display_ops = self._controller._op_registry.list_create_display_operators()

        # ── Section 1: Add columns (per-row operators) ────────────────
        if columns_ops:
            label = QAction("── Add columns to table ──", self)
            label.setEnabled(False)
            self._operators_menu.addAction(label)

            for op_name, op_label in columns_ops:
                action = QAction(op_label, self)
                action.triggered.connect(
                    lambda checked, n=op_name:
                    self._on_run_create_columns(n)
                )
                self._operators_menu.addAction(action)

        # ── Section 2: Create table (group operators) ─────────────────
        if table_ops:
            self._operators_menu.addSeparator()
            label = QAction("── Create new table ──", self)
            label.setEnabled(False)
            self._operators_menu.addAction(label)

            for op_name, op_label in table_ops:
                action = QAction(op_label, self)
                action.triggered.connect(
                    lambda checked, n=op_name:
                    self._on_run_create_table(n)
                )
                self._operators_menu.addAction(action)

        # ── Section 3: Show result (display operators) ────────────────
        if display_ops:
            self._operators_menu.addSeparator()
            label = QAction("── Show result ──", self)
            label.setEnabled(False)
            self._operators_menu.addAction(label)

            for op_name, op_label in display_ops:
                action = QAction(op_label, self)
                action.triggered.connect(
                    lambda checked, n=op_name:
                    self._on_run_create_display(n)
                )
                self._operators_menu.addAction(action)

        if not columns_ops and not table_ops and not display_ops:
            empty = QAction("No operators registered", self)
            empty.setEnabled(False)
            self._operators_menu.addAction(empty)

    # ── Operator run handlers ─────────────────────────────────────────

    def _show_scope_and_params_dialog(
        self,
        operator_name: str,
    ) -> list[str] | None:
        """
        Shows the scope dialog and then the operator's parameter dialog.
        Returns the chosen row_ids, or None if the researcher cancelled.

        Args:
            operator_name: Name of the operator being run.

        Returns:
            List of row_ids to run on, or None if cancelled.
        """
        selected_ids = self._main_gallery.get_selected_row_ids()
        visible_ids  = self._main_gallery._row_ids
        all_ids      = list(
            self._controller._dataset.get_table(
                self._controller._active_table
            )["row_id"]
        )

        # Step 1: scope dialog.
        scope_dialog = RunOperatorDialog(
            operator_name=operator_name,
            selected_ids=selected_ids,
            visible_ids=visible_ids,
            all_ids=all_ids,
            parent=self,
        )
        if scope_dialog.exec() == 0:
            return None

        row_ids = scope_dialog.chosen_row_ids
        if not row_ids:
            return None

        # Step 2: operator parameter dialog (if the operator has one).
        operator = self._controller._op_registry.get(operator_name)
        if operator is not None:
            param_dialog = operator.get_parameters_dialog(parent=self)
            if param_dialog is not None:
                if param_dialog.exec() == 0:
                    return None

        return row_ids

    def _on_run_create_columns(self, operator_name: str) -> None:
        """
        Shows the scope and parameter dialogs, then runs
        create_columns() on the chosen rows.
        """
        row_ids = self._show_scope_and_params_dialog(operator_name)
        if row_ids is None:
            return
        self._controller.run_create_columns(operator_name, row_ids)

    def _on_run_create_table(self, operator_name: str) -> None:
        """
        Shows the scope and parameter dialogs, then runs
        create_table() on the chosen rows.

        The group_by parameter is read from the operator instance
        after the parameter dialog runs — the dialog stores it as
        operator._group_by.
        """
        row_ids = self._show_scope_and_params_dialog(operator_name)
        if row_ids is None:
            return

        # Read the group_by parameter set by the parameter dialog.
        operator = self._controller._op_registry.get(operator_name)
        group_by = getattr(operator, "_group_by", None)

        self._controller.run_create_table(operator_name, row_ids, group_by)

    def _on_run_create_display(self, operator_name: str) -> None:
        """
        Shows the scope and parameter dialogs, then runs
        create_display() on the chosen rows.
        """
        row_ids = self._show_scope_and_params_dialog(operator_name)
        if row_ids is None:
            return
        self._controller.run_create_display(operator_name, row_ids)

    # ── Signal connections ────────────────────────────────────────────

    def _connect_signals(self) -> None:
        """
        Connects AppController signals to widget slots, and widget
        signals to AppController methods.
        """
        ctrl = self._controller

        # Controller -> UI
        ctrl.gallery_updated.connect(self._on_gallery_updated)
        ctrl.grouped_gallery_updated.connect(self._on_grouped_gallery_updated)
        ctrl.columns_updated.connect(self._on_columns_updated)
        ctrl.tables_updated.connect(self._on_tables_updated)
        ctrl.thumbnail_ready.connect(self._on_thumbnail_ready)
        ctrl.row_updated.connect(self._on_row_updated)
        ctrl.row_selected.connect(self._on_row_selected)
        ctrl.display_result_ready.connect(self._on_display_result)
        ctrl.error_occurred.connect(self._on_error)
        ctrl.merge_report_ready.connect(self._on_merge_report)
        ctrl.operator_complete.connect(self._on_operator_complete)
        ctrl.table_created.connect(self._on_table_created)

        # FilterPanel -> Controller
        self._filter_panel.filters_changed.connect(
            lambda filters: ctrl.set_filters(filters)
        )
        self._filter_panel.group_by_changed.connect(ctrl.set_group_by)
        self._filter_panel.tile_size_changed.connect(
            self._on_tile_size_changed
        )
        self._filter_panel.randomise_clicked.connect(
            lambda: ctrl.set_filters(
                self._filter_panel.get_active_filters(),
                randomise=True,
                seed=None,
            )
        )

        # Gallery -> Controller
        self._main_gallery.selection_changed.connect(
            self._stats_panel_removed_placeholder
        )
        self._main_gallery.tile_double_clicked.connect(
            lambda ids: ctrl.select_row(ids[0]) if ids else None
        )

    def _stats_panel_removed_placeholder(self, row_ids: list[str]) -> None:
        """
        Placeholder slot for the gallery selection_changed signal.
        Previously connected to StatisticsPanel.update_selection().
        StatisticsPanel has been removed — summary statistics are now
        run explicitly via the Operators menu (SummaryStatsOperator)
        and appear as a tab in ResultsPanel.

        TODO (Student A): If you want selection count to show somewhere
        in the UI (e.g. a status bar label), connect it here.
        """
        pass  # Nothing to do — results panel is operator-driven.

    # ── Signal handlers ───────────────────────────────────────────────

    def _on_row_selected(self, metadata: dict) -> None:
        """Shows the selected row in the detail panel."""
        row_id = metadata.get("row_id")
        if row_id:
            self._detail_widget.show_rows([row_id])

    def _on_gallery_updated(self, row_ids: list[str]) -> None:
        """Updates the main gallery with a new flat list of row_ids."""
        self._main_gallery.set_row_ids(row_ids)

    def _on_grouped_gallery_updated(self, grouped: dict) -> None:
        """
        Updates the gallery area for grouped view.
        Currently flattens all groups into one gallery.
        TODO (Student A): Implement multiple synchronized galleries,
        one per group, in a shared scroll area.
        """
        all_row_ids = []
        for group_row_ids in grouped.values():
            all_row_ids.extend(group_row_ids)
        self._main_gallery.set_row_ids(all_row_ids)

    def _on_columns_updated(self, column_names: list[str]) -> None:
        """Refreshes filter panel when columns change."""
        self._filter_panel.refresh_columns(column_names)

    def _on_tables_updated(self, table_names: list[str]) -> None:
        """Updates the table selector combo when tables change."""
        self._table_combo.blockSignals(True)
        current = self._table_combo.currentText()
        self._table_combo.clear()
        for name in table_names:
            self._table_combo.addItem(name)
        idx = self._table_combo.findText(current)
        if idx >= 0:
            self._table_combo.setCurrentIndex(idx)
        self._table_combo.blockSignals(False)

    def _on_thumbnail_ready(self, row_id: str) -> None:
        """Repaints the tile for row_id when its thumbnail arrives."""
        for gallery in self._galleries:
            gallery.on_thumbnail_ready(row_id)

    def _on_row_updated(self, row_id: str) -> None:
        """Repaints the tile for row_id when its data changes."""
        for gallery in self._galleries:
            gallery.on_row_updated(row_id)

    def _on_tile_size_changed(self, size: int) -> None:
        """Updates tile size across all galleries."""
        for gallery in self._galleries:
            gallery.set_tile_size(size)

    def _on_display_result(self, result: dict) -> None:
        """
        Shows a create_display operator result in the results panel
        and switches to the Results tab.
        """
        self._results_panel.show_result(result)
        # Switch to the Results tab so the researcher sees it immediately.
        right_tabs = self._results_panel.parent()
        if hasattr(right_tabs, 'indexOf'):
            right_tabs.setCurrentWidget(self._results_panel)

    def _on_operator_complete(self, operator_name: str) -> None:
        """Called when any operator finishes."""
        # Could show a status bar message or hide a progress bar here.
        print(f"[MainWindow] Operator complete: {operator_name}")

    def _on_table_created(self, table_name: str) -> None:
        """
        Called when a create_table operator has stored its result.
        Asks the researcher if they want to switch to the new table.
        """
        reply = QMessageBox.question(
            self,
            "New table created",
            f"Table '{table_name}' was created.\n\n"
            f"Switch to it now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._controller.set_active_table(table_name)

    def _on_error(self, message: str) -> None:
        """Shows an error dialog."""
        QMessageBox.warning(self, "Gelem — Error", message)

    def _on_merge_report(self, report) -> None:
        """
        Shows the merge diagnostics report and asks whether to proceed.
        TODO (Student A): Replace with a proper merge diagnostics dialog.
        """
        reply = QMessageBox.question(
            self,
            "Merge report",
            f"{report.summary()}\n\nProceed with merge?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._controller.confirm_merge(report)

    # ── Menu actions ──────────────────────────────────────────────────

    def _on_new_from_folder(self) -> None:
        """Opens a folder chooser to start a new project from images."""
        folder = QFileDialog.getExistingDirectory(
            self, "New project — select image folder"
        )
        if folder:
            from pathlib import Path
            self._controller.load_folder(Path(folder))

    def _on_new_from_csv(self) -> None:
        """
        Opens a CSV file to start a new project without images.
        TODO (Student A): Replace the simple input dialog with a proper
        dialog that shows CSV column names and lets the researcher
        select which column contains image paths.
        """
        from pathlib import Path

        path, _ = QFileDialog.getOpenFileName(
            self, "New project — open CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return

        image_col, ok = QInputDialog.getText(
            self,
            "Image column",
            "If your CSV has a column containing image file paths,\n"
            "enter its name here. Leave blank if there are no images.",
            text=""
        )

        image_column = (
            image_col.strip() if ok and image_col.strip() else None
        )
        self._controller.load_csv_as_primary(Path(path), image_column)

    def _on_merge_csv(self) -> None:
        """Opens a CSV file chooser for merging metadata."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Merge CSV", "", "CSV files (*.csv)"
        )
        if path:
            from pathlib import Path
            self._controller.load_csv(Path(path), join_on="file_name")

    def _on_save_project(self) -> None:
        """Opens a folder chooser for saving the project."""
        folder = QFileDialog.getExistingDirectory(
            self, "Save project to folder"
        )
        if folder:
            from pathlib import Path
            self._controller.save_project(Path(folder))

    def _on_load_project(self) -> None:
        """Opens a folder chooser for loading a project."""
        folder = QFileDialog.getExistingDirectory(
            self, "Open project folder"
        )
        if folder:
            from pathlib import Path
            self._controller.load_project(Path(folder))

    def _on_export_csv(self) -> None:
        """Opens a file save dialog for CSV export."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV files (*.csv)"
        )
        if path:
            from pathlib import Path
            self._controller.export_csv(Path(path))

    def _on_save_filtered_set(self) -> None:
        """Saves the currently visible rows as a new permanent table."""
        visible_ids = self._main_gallery._row_ids
        if not visible_ids:
            QMessageBox.information(
                self,
                "No rows visible",
                "There are no visible rows to save. "
                "Load a folder first.",
            )
            return

        dialog = SaveTableDialog(n_rows=len(visible_ids), parent=self)
        if dialog.exec() == 0:
            return

        self._controller.save_filtered_as_table(dialog.table_name)
