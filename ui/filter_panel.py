"""
ui/filter_panel.py

FilterPanel is a sidebar that automatically generates filter controls
from the active table's columns.

    text columns (low cardinality)  -> toggle buttons, one per unique value
    text columns (high cardinality) -> text search input (contains filter)
    media_path columns              -> no filter control (not filterable)
    numeric columns                 -> placeholder (recode tool, later stage)
    Group-by selector               -> dropdown to choose a grouping column
    Tile-size slider                -> controls gallery tile size
    Randomise button                -> shuffles the current gallery order

The threshold for switching between toggle buttons and text search is
defined by CATEGORICAL_THRESHOLD. If a text column has fewer unique
values than this threshold, toggle buttons are shown. Otherwise a
text search input is shown.

FilterPanel never accesses the data table directly. It reads column
metadata from the controller and calls controller methods when controls
change.

Student A is responsible for implementing this class.
"""

from __future__ import annotations
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QComboBox, QScrollArea,
    QGroupBox, QLineEdit, QSizePolicy
)
from PySide6.QtCore import Signal, Qt

from models.query_engine import Filter


# Number of unique values below which toggle buttons are shown.
# Above this threshold, a text search input is shown instead.
CATEGORICAL_THRESHOLD = 20


class FilterPanel(QWidget):
    """
    Auto-generated filter controls for the active table.

    Signals:
        filters_changed:   Emitted when any filter control changes.
                           Carries the current list of active Filter objects.
        group_by_changed:  Emitted when the group-by column changes.
                           Carries the column name or None.
        tile_size_changed: Emitted when the tile-size slider moves.
                           Carries the new size in pixels.
        randomise_clicked: Emitted when the Randomise button is clicked.
    """

    filters_changed   = Signal(list)
    group_by_changed  = Signal(object)
    tile_size_changed = Signal(int)
    randomise_clicked = Signal()

    def __init__(self, controller, parent=None):
        """
        Creates the FilterPanel.

        Args:
            controller: The AppController instance.
            parent:     Optional parent widget.
        """
        super().__init__(parent)
        self._controller     = controller
        self._active_filters: dict[str, Filter] = {}
        # Key: column name. Value: active Filter or None.
        self._toggle_values: dict[str, set] = {}
        # Key: column name. Value: set of currently-checked toggle values.

        self.setMinimumWidth(180)
        self.setMaximumWidth(180)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        outer.addWidget(scroll)

        self._inner  = QWidget()
        self._layout = QVBoxLayout(self._inner)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(4)
        self._layout.addStretch()
        scroll.setWidget(self._inner)

        self._build_fixed_controls()

    def _build_fixed_controls(self) -> None:
        """
        Builds the controls that are always present: tile-size slider,
        group-by selector, and randomise button.
        """
        # Tile size slider.
        size_box    = QGroupBox("Tile size")
        size_layout = QVBoxLayout(size_box)
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setMinimum(80)
        self._size_slider.setMaximum(600)
        self._size_slider.setValue(150)
        self._size_slider.valueChanged.connect(
            lambda v: self.tile_size_changed.emit(v)
        )
        size_layout.addWidget(self._size_slider)
        self._layout.insertWidget(0, size_box)

        # Group-by selector.
        group_box    = QGroupBox("Group gallery by")
        group_layout = QVBoxLayout(group_box)
        self._group_combo = QComboBox()
        self._group_combo.addItem("None", userData=None)
        self._group_combo.currentIndexChanged.connect(
            self._on_group_by_changed
        )
        group_layout.addWidget(self._group_combo)
        self._layout.insertWidget(1, group_box)

        # Randomise button.
        self._randomise_btn = QPushButton("Randomise order")
        self._randomise_btn.clicked.connect(self.randomise_clicked.emit)
        self._layout.insertWidget(2, self._randomise_btn)

    # ── Public API ────────────────────────────────────────────────────

    def refresh_columns(self, column_names: list[str]) -> None:
        """
        Rebuilds the filter controls based on the current column list.
        Called when columns_updated signal arrives from AppController.

        For each text column:
            - If the column has fewer than CATEGORICAL_THRESHOLD unique
              values, shows toggle buttons (one per value).
            - Otherwise shows a text search input.

        For numeric columns: shows a placeholder label for now.
        For media_path columns: no filter control (not filterable).

        Args:
            column_names: List of all registered column names.

        TODO (Student A): Implement recode tool for numeric columns.
        """
        # Remove old column controls (keep fixed controls at indices 0-2,
        # plus the stretch at the end).
        while self._layout.count() > 4:
            item = self._layout.takeAt(3)
            if item.widget():
                item.widget().deleteLater()

        # Update group-by combo.
        self._group_combo.blockSignals(True)
        self._group_combo.clear()
        self._group_combo.addItem("None", userData=None)
        for col in column_names:
            self._group_combo.addItem(col, userData=col)
        self._group_combo.blockSignals(False)

        # Build a filter control for each column.
        for col in column_names:
            col_type = self._controller._registry.get(col)
            if col_type is None:
                continue

            if col_type.tag == "text":
                self._add_text_filter(col)
            elif col_type.tag in ("media_path",):
                # Media columns — no filter control needed.
                pass
            else:
                # Placeholder for numeric and boolean types.
                label = QLabel(f"{col} ({col_type.tag})")
                label.setStyleSheet("color: #888888; font-size: 11px;")
                self._layout.insertWidget(
                    self._layout.count() - 1, label
                )

    def _add_text_filter(self, column: str) -> None:
        """
        Adds a filter control for a text column.

        If the column has fewer than CATEGORICAL_THRESHOLD unique values,
        shows a row of toggle buttons — one per unique value. Clicking
        a button activates an 'eq' filter for that value.

        If the column has CATEGORICAL_THRESHOLD or more unique values,
        shows a text input. As the researcher types, a 'contains' filter
        is applied (case-insensitive substring match).

        Args:
            column: The column name to create a filter control for.

        TODO (Student A): Implement multi-select for toggle buttons
        (allow more than one value active at once, using 'isin' filter).
        """
        values = self._controller.get_group_values(column)
        if not values:
            return

        group = QGroupBox(column)

        if len(values) < CATEGORICAL_THRESHOLD:
            # Low cardinality — show toggle buttons stacked vertically
            # so each button can stay narrow and labels are never truncated.
            layout = QVBoxLayout(group)
            layout.setSpacing(2)
            layout.setContentsMargins(4, 4, 4, 4)

            # Buttons are recreated unchecked, so reset state to match.
            self._toggle_values[column] = set()
            self._active_filters.pop(column, None)

            for val in values:
                btn = QPushButton(str(val))
                btn.setCheckable(True)
                btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                btn.clicked.connect(
                    lambda checked, c=column, v=val:
                    self._on_text_toggle(c, v, checked)
                )
                layout.addWidget(btn)

        else:
            # High cardinality — show text search input.
            layout = QVBoxLayout(group)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(2)

            hint = QLabel(f"{len(values)} unique values")
            hint.setStyleSheet("color: #888888; font-size: 10px;")
            layout.addWidget(hint)

            search = QLineEdit()
            search.setPlaceholderText("Type to filter...")
            search.setClearButtonEnabled(True)
            # Emit a 'contains' filter as the researcher types.
            search.textChanged.connect(
                lambda text, c=column:
                self._on_text_search(c, text)
            )
            layout.addWidget(search)

        self._layout.insertWidget(self._layout.count() - 1, group)

    def _on_text_toggle(
        self,
        column: str,
        value: str,
        checked: bool,
    ) -> None:
        """
        Called when a text filter toggle button is clicked.

        Multiple values can be active at once per column. Their union is
        represented as a single Filter(column, "isin", [...]). When no
        values are checked, the column's filter is removed entirely.

        Args:
            column:  The column being filtered.
            value:   The value being toggled.
            checked: True if the button is now active.
        """
        values = self._toggle_values.setdefault(column, set())
        if checked:
            values.add(value)
        else:
            values.discard(value)

        if values:
            self._active_filters[column] = Filter(
                column, "isin", sorted(values, key=str)
            )
        else:
            self._active_filters.pop(column, None)

        self.filters_changed.emit(list(self._active_filters.values()))

    def _on_text_search(self, column: str, text: str) -> None:
        """
        Called when the researcher types in a text search input.
        Activates a 'contains' filter for non-empty text, or removes
        the filter if the text is cleared.

        Args:
            column: The column being filtered.
            text:   The current search string.
        """
        if text.strip():
            self._active_filters[column] = Filter(
                column, "contains", text.strip()
            )
        else:
            self._active_filters.pop(column, None)

        self.filters_changed.emit(list(self._active_filters.values()))

    def _on_group_by_changed(self, index: int) -> None:
        """
        Called when the group-by combo selection changes.

        Args:
            index: The selected index in the combo box.
        """
        column = self._group_combo.itemData(index)
        self.group_by_changed.emit(column)

    def get_active_filters(self) -> list[Filter]:
        """
        Returns the currently active list of Filter objects.

        Returns:
            List of active Filter objects.
        """
        return list(self._active_filters.values())
