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
    QFileDialog, QMessageBox, QTabWidget,
    QStackedWidget, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

from shared_widgets.checkable_combo_box import CheckableComboBox
from ui.gallery_widget import GalleryWidget
from ui.filter_panel import FilterPanel
from ui.detail_widget import DetailWidget
from ui.results_panel import ResultsPanel
from ui.run_operator_dialog import RunOperatorDialog
from ui.save_table_dialog import SaveTableDialog
from ui.csv_image_column_dialog import CsvImageColumnDialog
from ui.merge_report_dialog import MergeReportDialog


class MainWindow(QMainWindow):
    """
    Top-level application window.

    Connects AppController signals to UI widgets and routes UI events
    back to AppController methods.

    The gallery area is a QStackedWidget with two pages: a flat view
    (a single GalleryWidget) and a grouped view (one GalleryWidget per
    group value, stacked vertically in a shared scroll area). The
    controller decides which to show by emitting gallery_updated or
    grouped_gallery_updated.
    """

    # Height of each per-group gallery in the grouped view. Bounded so
    # the outer scroll area pages between groups while each group scrolls
    # internally for overflow.
    _GROUP_GALLERY_HEIGHT = 340

    def __init__(self, controller):
        """
        Creates the MainWindow and all child widgets.

        Args:
            controller: The AppController instance.
        """
        super().__init__()
        self._controller = controller
        self._galleries: list[GalleryWidget] = []
        # Current tile size, mirrored from the filter panel so new
        # per-group galleries can be created at the right size.
        self._tile_size = 150
        # Current per-group gallery height, adjustable from the filter
        # panel's "Group height" slider.
        self._group_gallery_height = self._GROUP_GALLERY_HEIGHT

        self.setWindowTitle("Gelem — Visual Data Explorer")
        self.resize(1400, 900)

        self._build_menu()
        self._build_toolbar()
        self._build_central_widget()
        self._build_status_bar()
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

        # QToolBar top-aligns the widgets it hosts, so we put the controls
        # in our own container whose QHBoxLayout vertically centres them
        # and adds equal padding above and below the row.
        controls = QWidget()
        row = QHBoxLayout(controls)
        row.setContentsMargins(8, 7, 8, 7)
        row.setSpacing(6)

        row.addWidget(QLabel("Table: "), 0, Qt.AlignmentFlag.AlignVCenter)
        self._table_combo = QComboBox()
        self._table_combo.addItem("frames")
        self._table_combo.currentTextChanged.connect(
            self._controller.set_active_table
        )
        row.addWidget(self._table_combo, 0, Qt.AlignmentFlag.AlignVCenter)

        # A little breathing room between the table and columns controls.
        row.addSpacing(16)

        # Visible-columns selector. A combo box of checkable visual
        # columns, letting the researcher choose which columns each tile
        # shows. It is repopulated on open so it always reflects the
        # currently registered visual columns.
        row.addWidget(QLabel("Columns: "), 0, Qt.AlignmentFlag.AlignVCenter)
        self._columns_combo = CheckableComboBox()
        self._columns_combo.set_placeholder("Visible columns")
        self._columns_combo.about_to_show.connect(self._refresh_columns_combo)
        self._columns_combo.selection_changed.connect(
            self._apply_visible_columns
        )
        row.addWidget(self._columns_combo, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)

        toolbar.addWidget(controls)

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

        # Centre: Gallery area — a stack of a flat view and a grouped view.
        self._gallery_stack = QStackedWidget()

        # Page 0 — flat view: a single gallery of all visible rows.
        self._main_gallery = GalleryWidget(self._controller)
        self._galleries    = [self._main_gallery]
        self._gallery_stack.addWidget(self._main_gallery)

        # Page 1 — grouped view: one gallery per group, stacked vertically
        # inside a shared scroll area. Sections are added/removed as the
        # grouping changes; the trailing stretch keeps them pinned to top.
        self._grouped_scroll = QScrollArea()
        self._grouped_scroll.setWidgetResizable(True)
        self._grouped_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._grouped_container = QWidget()
        self._grouped_layout    = QVBoxLayout(self._grouped_container)
        self._grouped_layout.setContentsMargins(4, 4, 4, 4)
        self._grouped_layout.setSpacing(8)
        self._grouped_layout.addStretch(1)
        self._grouped_scroll.setWidget(self._grouped_container)
        self._gallery_stack.addWidget(self._grouped_scroll)

        splitter.addWidget(self._gallery_stack)

        # Right: tabbed panels — Detail and Results only.
        # Kept as an instance attribute so handlers can switch focus
        # between Detail and Results in response to user actions.
        self._right_tabs = QTabWidget()

        self._detail_widget = DetailWidget(self._controller)
        self._right_tabs.addTab(self._detail_widget, "Detail")

        self._results_panel = ResultsPanel(self._controller)
        self._right_tabs.addTab(self._results_panel, "Results")

        splitter.addWidget(self._right_tabs)
        splitter.setSizes([180, 840, 380])

    def _build_status_bar(self) -> None:
        """
        Adds a status bar with a permanent label that reports gallery
        selection: "{N} items" when nothing is selected, "{K} of {N}
        selected" otherwise. The label is updated by
        _on_selection_changed (selection edits) and the gallery-update
        handlers (filter changes).
        """
        self._selection_label = QLabel()
        self._selection_label.setStyleSheet("padding: 0 8px;")
        # addPermanentWidget pins it to the right side so transient
        # messages from QStatusBar.showMessage() don't displace it.
        self.statusBar().addPermanentWidget(self._selection_label)
        self._refresh_status_bar()

    def _refresh_status_bar(self) -> None:
        """
        Updates the selection-count label from the live galleries'
        current selected and visible counts. In flat mode this is just
        the main gallery; in grouped mode the counts are aggregated
        across every group gallery.
        """
        selected = len(self._collect_selected_row_ids())
        visible  = len(self._collect_visible_row_ids())
        if selected:
            text = f"{selected} of {visible} selected"
        else:
            text = f"{visible} items"
        self._selection_label.setText(text)

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

    # ── Visible-columns selector ──────────────────────────────────────

    def _refresh_columns_combo(self) -> None:
        """
        Repopulates the visible-columns combo from the controller.

        Lists every visual column (those that render as tiles), checking
        the ones in the controller's current visible-column selection.
        Repopulating does not fire selection_changed, so it never loops
        back into _apply_visible_columns.
        """
        self._columns_combo.set_items(
            self._controller.get_visual_column_names(),
            checked=self._controller.get_visible_columns(),
        )

    def _apply_visible_columns(self, column_names: list[str]) -> None:
        """
        Pushes the chosen visible columns to every gallery and records
        them on the controller.

        Galleries are updated first so that the gallery refresh the
        controller triggers re-lays out with the new columns already in
        place.
        """
        for gallery in self._galleries:
            gallery.set_visible_columns(column_names)
        self._controller.set_visible_columns(column_names)

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
        selected_ids = self._collect_selected_row_ids()
        visible_ids  = self._collect_visible_row_ids()
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
        self._filter_panel.group_height_changed.connect(
            self._on_group_height_changed
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
            self._on_selection_changed
        )
        self._main_gallery.tile_double_clicked.connect(
            self._on_tile_double_clicked
        )

    def _on_tile_double_clicked(
        self, clicked_ids: list[str], gallery: GalleryWidget | None = None
    ) -> None:
        """
        Handles tile double-click. When several tiles are currently
        selected and the double-clicked tile is one of them, opens the
        whole selection in the detail panel side-by-side. Otherwise
        opens just the double-clicked tile.

        Args:
            clicked_ids: row_ids of the double-clicked tile.
            gallery:     The gallery that emitted the event. Defaults to
                         the flat main gallery; per-group galleries pass
                         themselves so selection is read from the right one.
        """
        if not clicked_ids:
            return

        gallery  = gallery or self._main_gallery
        selected = gallery.get_selected_row_ids()
        if len(selected) > 1 and clicked_ids[0] in selected:
            # Preserve gallery order so panels read left-to-right the
            # same way the tiles do.
            ordered = [
                rid for rid in gallery._row_ids
                if rid in selected
            ]
            self._detail_widget.show_rows(ordered)
            self._right_tabs.setCurrentWidget(self._detail_widget)
        else:
            # Single-item path goes through the controller so other
            # listeners (e.g. row_selected signal) still fire.
            self._controller.select_row(clicked_ids[0])

    def _on_selection_changed(self, row_ids: list[str]) -> None:
        """
        Updates the status bar when the gallery selection changes.
        The row_ids argument is the new selection set; the actual count
        is read back from the gallery so all sources of truth agree.
        """
        self._refresh_status_bar()

    # ── Signal handlers ───────────────────────────────────────────────

    def _on_row_selected(self, metadata: dict) -> None:
        """
        Shows the selected row in the detail panel and brings the Detail
        tab forward. Picking a row is an explicit ask to see it; if the
        Results tab is showing (e.g. after a create_display operator),
        the user expects the Detail tab to take focus on their click.
        """
        row_id = metadata.get("row_id")
        if row_id:
            self._detail_widget.show_rows([row_id])
            self._right_tabs.setCurrentWidget(self._detail_widget)

    def _on_gallery_updated(self, row_ids: list[str]) -> None:
        """
        Shows the flat view: a single gallery of all visible rows.

        Tears down any per-group galleries left over from grouped mode
        and points broadcasts back at the main gallery.
        """
        self._clear_grouped_galleries()
        self._galleries = [self._main_gallery]
        self._main_gallery.set_row_ids(row_ids)
        self._gallery_stack.setCurrentWidget(self._main_gallery)
        self._refresh_status_bar()

    def _on_grouped_gallery_updated(self, grouped: dict) -> None:
        """
        Shows the grouped view: one gallery per group value, stacked
        vertically in the shared scroll area.

        Args:
            grouped: Maps each group value to its ordered list of row_ids.
        """
        self._rebuild_grouped_galleries(grouped)
        self._gallery_stack.setCurrentWidget(self._grouped_scroll)
        self._refresh_status_bar()

    # ── Grouped-view construction ─────────────────────────────────────

    def _rebuild_grouped_galleries(self, grouped: dict) -> None:
        """
        Replaces the grouped view's sections with one per group.

        Updates self._galleries so thumbnail, row-update and tile-size
        broadcasts reach every visible group gallery. Falls back to the
        main gallery when there are no groups.
        """
        self._clear_grouped_galleries()

        galleries: list[GalleryWidget] = []
        for group_value, row_ids in grouped.items():
            section, gallery = self._build_group_section(group_value, row_ids)
            # Insert before the trailing stretch so sections stay top-aligned.
            self._grouped_layout.insertWidget(
                self._grouped_layout.count() - 1, section
            )
            gallery.set_row_ids(row_ids)
            galleries.append(gallery)

        self._galleries = galleries or [self._main_gallery]

    def _build_group_section(
        self, group_value, row_ids: list[str]
    ) -> tuple[QWidget, GalleryWidget]:
        """
        Builds one grouped-view section: a header label plus a
        bounded-height gallery for the group's rows.

        Args:
            group_value: The group's value (used in the header label).
            row_ids:     The row_ids belonging to this group.

        Returns:
            (section_widget, gallery) — the gallery is returned so the
            caller can register it for broadcasts and load its rows.
        """
        section = QWidget()
        layout  = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QLabel(f"{group_value}  ({len(row_ids)})")
        header.setStyleSheet("font-weight: bold; padding: 2px 4px;")
        layout.addWidget(header)

        gallery = GalleryWidget(self._controller)
        gallery.setFixedHeight(self._group_gallery_height)
        # Match the flat gallery's current display settings.
        gallery.set_visible_columns(self._controller.get_visible_columns())
        gallery.set_tile_size(self._tile_size)
        gallery.tile_double_clicked.connect(
            lambda ids, g=gallery: self._on_tile_double_clicked(ids, g)
        )
        gallery.selection_changed.connect(self._on_selection_changed)
        layout.addWidget(gallery)

        return section, gallery

    def _clear_grouped_galleries(self) -> None:
        """Removes all group sections, leaving the trailing stretch."""
        while self._grouped_layout.count() > 1:
            item   = self._grouped_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _collect_selected_row_ids(self) -> list[str]:
        """
        Returns the selected row_ids across every live gallery, in
        order and de-duplicated. In flat mode this is just the main
        gallery; in grouped mode it spans all group galleries.
        """
        return self._collect_row_ids(
            lambda g: g.get_selected_row_ids()
        )

    def _collect_visible_row_ids(self) -> list[str]:
        """
        Returns the visible row_ids across every live gallery, in order
        and de-duplicated (used as the 'visible' operator scope).
        """
        return self._collect_row_ids(lambda g: g._row_ids)

    def _collect_row_ids(self, getter) -> list[str]:
        """Flattens getter(gallery) over self._galleries, keeping order
        and dropping duplicates."""
        result: list[str] = []
        seen: set[str] = set()
        for gallery in self._galleries:
            for row_id in getter(gallery):
                if row_id not in seen:
                    seen.add(row_id)
                    result.append(row_id)
        return result

    def _on_columns_updated(self, column_names: list[str]) -> None:
        """Refreshes filter panel and columns combo when columns change."""
        self._filter_panel.refresh_columns(column_names)
        self._refresh_columns_combo()

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
        self._tile_size = size
        for gallery in self._galleries:
            gallery.set_tile_size(size)

    def _on_group_height_changed(self, height: int) -> None:
        """
        Updates the per-group gallery height. Applies live to the
        currently displayed grouped sections and is remembered so that
        sections built later use the same height.
        """
        self._group_gallery_height = height
        if self._gallery_stack.currentWidget() is self._grouped_scroll:
            for gallery in self._galleries:
                gallery.setFixedHeight(height)

    def _on_display_result(self, result: dict) -> None:
        """
        Shows a create_display operator result in the results panel
        and switches to the Results tab.
        """
        self._results_panel.show_result(result)
        # Switch to the Results tab so the researcher sees it immediately.
        self._right_tabs.setCurrentWidget(self._results_panel)

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
        Shows the merge diagnostics dialog. The researcher sees the
        match/unmatch counts and per-issue lists (unmatched files,
        unmatched CSV rows, duplicate keys) before deciding whether
        to commit the merge.
        """
        dialog = MergeReportDialog(report, parent=self)
        if dialog.exec() != 0 and dialog.accepted_merge:
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

        After the file is chosen, CsvImageColumnDialog reads the CSV
        header and lets the researcher pick which column (if any)
        holds image file paths from a combo box, with a one-line
        preview of that column's first value so they can confirm
        their pick.
        """
        from pathlib import Path

        path, _ = QFileDialog.getOpenFileName(
            self, "New project — open CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return

        csv_path = Path(path)
        dialog = CsvImageColumnDialog(csv_path, parent=self)
        if dialog.exec() == 0:
            return

        self._controller.load_csv_as_primary(csv_path, dialog.image_column)

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
