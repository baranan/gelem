"""
ui/results_panel.py

ResultsPanel displays outputs from create_display operators — mean face
images, statistics tables, interactive plots, or any other result dict
an operator returns.

Each time a create_display operator completes, its output is added as a
new tab inside this panel. Tabs can be closed individually (losing that
result forever). The panel never auto-runs any operator — results only
appear when the researcher explicitly triggers an operator from the
Operators menu.

ResultsPanel replaces the old separate StatisticsPanel. Summary
statistics are just another create_display operator (SummaryStatsOperator)
whose result appears here like any other.

Student A is responsible for implementing this class.

Key Qt concept used here:
    QTabWidget with setTabsClosable(True) provides built-in close buttons
    (the × on each tab). Connect tabCloseRequested(int) signal to a slot
    that removes the tab at that index.
"""

from __future__ import annotations
import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QFileDialog, QScrollArea,
    QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap


class ResultsPanel(QWidget):
    """
    Shows create_display operator outputs as closeable tabs.

    Each call to show_result() adds a new tab. Tabs have an × button
    that removes them permanently. The panel keeps all results from the
    current session until explicitly closed.

    Each result tab contains:
        - A media display area (image, if artifact_path is present)
        - A statistics table (if summary dict is present)
        - A label identifying the operator and run time

    TODO (Student A): Support richer result types as operators are added:
        - Interactive HTML plots (open in browser or embed QWebEngineView)
        - Multi-image results (e.g. one mean face per condition)
    """

    def __init__(self, controller, parent=None):
        """
        Creates the ResultsPanel.

        Args:
            controller: The AppController instance.
            parent:     Optional parent widget.
        """
        super().__init__(parent)
        self._controller = controller

        # Counter to make tab labels unique when the same operator
        # runs multiple times.
        self._run_counter: int = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Placeholder shown when no results exist yet.
        self._placeholder = QLabel("No results yet.\nRun an operator from the Operators menu.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888888; font-size: 13px;")
        layout.addWidget(self._placeholder)

        # Tab widget — hidden until the first result arrives.
        # tabsClosable(True) adds an × button to every tab automatically.
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.setVisible(False)

        # When the researcher clicks × on a tab, remove it.
        # tabCloseRequested emits the index of the tab to close.
        self._tabs.tabCloseRequested.connect(self._on_tab_close_requested)

        layout.addWidget(self._tabs)

    # ── Public API ────────────────────────────────────────────────────

    def show_result(self, result: dict) -> None:
        """
        Adds a new tab displaying the operator result.

        Called by MainWindow when display_result_ready signal arrives
        from AppController.

        The result dict may contain any combination of:
            'operator_name': str  — used as the tab label.
            'artifact_path': str  — path to an image file to display.
            'summary':       dict — nested {column: {stat: value}} dict
                                    shown as a statistics table.
            'table':         list — list of dicts shown as a table
                                    (used by StatsOperator).
            'plot_html':     str  — path to an interactive HTML file.
            'n_rows':        int  — number of rows the operator ran on.

        Args:
            result: Dict from a create_display operator.
        """
        self._run_counter += 1
        operator_name = result.get("operator_name", "Result")
        timestamp     = datetime.datetime.now().strftime("%H:%M:%S")

        # Build the tab label: operator name + run counter if repeated.
        tab_label = f"{operator_name} #{self._run_counter}"

        # Build the tab content widget.
        tab_widget = self._build_result_widget(result, operator_name, timestamp)

        # Add the tab and switch to it.
        idx = self._tabs.addTab(tab_widget, tab_label)
        self._tabs.setCurrentIndex(idx)

        # Show the tab widget, hide the placeholder.
        self._placeholder.setVisible(False)
        self._tabs.setVisible(True)

    # ── Internal helpers ──────────────────────────────────────────────

    def _on_tab_close_requested(self, index: int) -> None:
        """
        Called when the researcher clicks × on a tab.
        Removes the tab and its content permanently.
        If no tabs remain, shows the placeholder again.

        Args:
            index: The index of the tab to close.
        """
        self._tabs.removeTab(index)

        if self._tabs.count() == 0:
            self._tabs.setVisible(False)
            self._placeholder.setVisible(True)

    def _build_result_widget(
        self,
        result: dict,
        operator_name: str,
        timestamp: str,
    ) -> QWidget:
        """
        Builds the content widget for one result tab.

        Displays an image if 'artifact_path' is present, a statistics
        table if 'summary' or 'table' is present, and a header label
        identifying the operator and run time.

        Args:
            result:        The result dict from the operator.
            operator_name: Human-readable operator name.
            timestamp:     Time the result was received (HH:MM:SS).

        Returns:
            A QWidget containing the result display.

        TODO (Student A): Add support for 'plot_html' key — open the
        HTML file in the system browser or embed a QWebEngineView.
        """
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Header ────────────────────────────────────────────────────
        header_row = QHBoxLayout()

        header = QLabel(f"{operator_name}  ·  {timestamp}")
        header.setStyleSheet("font-weight: bold; font-size: 12px; color: #2E5F8A;")
        header_row.addWidget(header)

        n_rows = result.get("n_rows") or result.get("n_frames")
        if n_rows is not None:
            n_label = QLabel(f"{n_rows} rows")
            n_label.setStyleSheet("color: #888888; font-size: 11px;")
            header_row.addWidget(n_label)

        header_row.addStretch()

        # Save PNG button — only shown if there is an image to save.
        artifact_path = result.get("artifact_path", "")
        if artifact_path:
            save_btn = QPushButton("Save as PNG")
            save_btn.clicked.connect(
                lambda checked, p=artifact_path: self._save_as_png(p)
            )
            header_row.addWidget(save_btn)

        layout.addLayout(header_row)

        # ── Image ──────────────────────────────────────────────────────
        if artifact_path:
            pixmap = QPixmap(str(artifact_path))
            if not pixmap.isNull():
                image_label = QLabel()
                image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                image_label.setMinimumHeight(200)
                image_label.setStyleSheet(
                    "background-color: #F5F5F5; border: 1px solid #CCCCCC;"
                )
                scaled = pixmap.scaled(
                    400, 300,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                image_label.setPixmap(scaled)
                layout.addWidget(image_label)

        # ── Statistics table (from 'summary' key) ────────────────────
        # The 'summary' key contains a nested dict:
        #   {column_name: {stat_name: value, ...}, ...}
        # We display this as a table with one row per column.
        summary = result.get("summary", {})
        if summary:
            layout.addWidget(self._build_summary_table(summary))

        # ── Generic table (from 'table' key) ─────────────────────────
        # The 'table' key contains a list of dicts (used by StatsOperator).
        # Each dict is one row; keys become column headers.
        table_data = result.get("table", [])
        if table_data:
            layout.addWidget(self._build_generic_table(table_data))

        # ── Interpretation text (from StatsOperator etc.) ────────────
        interpretation = result.get("interpretation", "")
        if interpretation:
            interp_label = QLabel(interpretation)
            interp_label.setWordWrap(True)
            interp_label.setStyleSheet(
                "font-size: 11px; color: #444444; padding: 4px;"
                "background-color: #F9F9F9; border: 1px solid #EEEEEE;"
            )
            layout.addWidget(interp_label)

        layout.addStretch()
        return container

    def _build_summary_table(self, summary: dict) -> QWidget:
        """
        Builds a QTableWidget displaying the summary statistics dict.

        The summary dict has the structure:
            {column_name: {'mean': float, 'sd': float, 'min': float,
                           'max': float, 'median': float, 'n': int}}

        Args:
            summary: The nested statistics dict.

        Returns:
            A QTableWidget widget.
        """
        # Collect all stat keys from the first entry to build headers.
        all_stats = []
        for stats in summary.values():
            if isinstance(stats, dict):
                all_stats = list(stats.keys())
                break

        headers  = ["Column"] + all_stats
        table    = QTableWidget(len(summary), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)

        for row_idx, (col_name, stats) in enumerate(summary.items()):
            table.setItem(row_idx, 0, QTableWidgetItem(col_name))
            if isinstance(stats, dict):
                for col_idx, stat_name in enumerate(all_stats, start=1):
                    val = stats.get(stat_name, "")
                    if isinstance(val, float):
                        text = f"{val:.4g}"
                    else:
                        text = str(val)
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                          Qt.AlignmentFlag.AlignVCenter)
                    table.setItem(row_idx, col_idx, item)

        table.resizeColumnsToContents()
        return table

    def _build_generic_table(self, table_data: list[dict]) -> QWidget:
        """
        Builds a QTableWidget from a list of dicts (one dict per row).
        Used by StatsOperator which returns a 'table' key.

        Args:
            table_data: List of dicts, each representing one table row.

        Returns:
            A QTableWidget widget.
        """
        if not table_data:
            return QWidget()

        headers = list(table_data[0].keys())
        table   = QTableWidget(len(table_data), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)

        for row_idx, row_dict in enumerate(table_data):
            for col_idx, key in enumerate(headers):
                val  = row_dict.get(key, "")
                text = f"{val:.4g}" if isinstance(val, float) else str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                      Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row_idx, col_idx, item)

        table.resizeColumnsToContents()
        return table

    def _save_as_png(self, artifact_path: str) -> None:
        """
        Saves the result image as a PNG file chosen by the researcher.

        Args:
            artifact_path: Path to the source image file.
        """
        pixmap = QPixmap(artifact_path)
        if pixmap.isNull():
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save result as PNG", "", "PNG files (*.png)"
        )
        if path:
            pixmap.save(path, "PNG")
