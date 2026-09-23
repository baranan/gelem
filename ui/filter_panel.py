"""
ui/filter_panel.py

FilterPanel is a sidebar that automatically generates filter controls
from the active table's columns.

    text columns (low cardinality)  -> toggle buttons, one per unique value
    text columns (high cardinality) -> text search input (contains filter)
    media_path columns              -> no filter control (not filterable)
    numeric columns                 -> min/max range spin boxes ('between')
    boolean columns                 -> All / True / False selector ('eq')
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
    QGroupBox, QLineEdit, QSizePolicy, QDoubleSpinBox
)
from PySide6.QtCore import Signal, Qt

from models.query_engine import Filter
from table_display import QueryState


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
        group_height_changed: Emitted when the group-height slider moves.
                           Carries the new per-group gallery height in pixels.
        randomise_clicked: Emitted when the Randomise button is clicked.
    """

    filters_changed      = Signal(list)
    group_by_changed     = Signal(object)
    tile_size_changed    = Signal(int)
    group_height_changed = Signal(int)
    randomise_clicked    = Signal()

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
        #
        # This is not an independent choice the researcher made here --
        # refresh_columns() rebuilds it from scratch every time (clearing
        # it, then either resetting a column's entry as its control is
        # rebuilt, or restoring it from the controller in
        # _restore_query_state()). Nothing in this file sets it from a
        # control without also calling filters_changed.emit(), and
        # nothing reads it as a value the controller does not already
        # have -- see _restore_query_state()'s docstring for the drift
        # this used to allow.
        self._toggle_values: dict[str, set] = {}
        # Key: column name. Value: set of currently-checked toggle values.
        self._numeric_spins: dict[str, tuple] = {}
        # Key: column name. Value: (min_spin, max_spin, col_min, col_max)
        # for the numeric range controls, so both ends can be read together.
        self._toggle_buttons: dict[str, dict] = {}
        # Key: column name. Value: {toggle value: its QPushButton} -- so
        # _restore_query_state() can check the right buttons back on
        # without rebuilding them.
        self._text_searches: dict[str, QLineEdit] = {}
        # Key: column name. Value: its QLineEdit, for the same reason.
        self._boolean_combos: dict[str, QComboBox] = {}
        # Key: column name. Value: its QComboBox, for the same reason.

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

        # Group-by selector, plus the per-group height slider that only
        # affects the grouped view (kept together in one box so the
        # column-control index arithmetic in refresh_columns is unchanged).
        group_box    = QGroupBox("Group gallery by")
        group_layout = QVBoxLayout(group_box)
        self._group_combo = QComboBox()
        self._group_combo.addItem("None", userData=None)
        self._group_combo.currentIndexChanged.connect(
            self._on_group_by_changed
        )
        group_layout.addWidget(self._group_combo)

        self._group_height_label = QLabel("Group height")
        group_layout.addWidget(self._group_height_label)
        self._group_height_slider = QSlider(Qt.Orientation.Horizontal)
        self._group_height_slider.setMinimum(180)
        self._group_height_slider.setMaximum(700)
        self._group_height_slider.setValue(340)
        self._group_height_slider.valueChanged.connect(
            lambda v: self.group_height_changed.emit(v)
        )
        group_layout.addWidget(self._group_height_slider)

        # The group-height control only does something while a grouping
        # column is active, so it starts disabled (combo defaults to None)
        # and is re-enabled in _on_group_by_changed.
        self._set_group_height_enabled(False)

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

        For numeric columns: a min/max range control ('between' filter).
        For boolean columns: an All / True / False selector ('eq' filter).
        For media_path columns: no filter control (not filterable).

        Args:
            column_names: List of all registered column names.
        """
        # Remove old column controls (keep fixed controls at indices 0-2,
        # plus the stretch at the end).
        while self._layout.count() > 4:
            item = self._layout.takeAt(3)
            if item.widget():
                item.widget().deleteLater()

        # Every control referenced below is about to be destroyed and
        # rebuilt (or dropped, if its column is gone), so drop every
        # reference to the old ones -- including self._active_filters
        # itself: a column no longer in column_names would otherwise
        # keep a stale entry here forever, since no add_*_filter call
        # ever runs for it again to clear it.
        self._numeric_spins.clear()
        self._toggle_buttons.clear()
        self._text_searches.clear()
        self._boolean_combos.clear()
        self._toggle_values.clear()
        self._active_filters.clear()

        # Update group-by combo.
        self._group_combo.blockSignals(True)
        self._group_combo.clear()
        self._group_combo.addItem("None", userData=None)
        for col in column_names:
            self._group_combo.addItem(col, userData=col)
        self._group_combo.blockSignals(False)

        # Build a filter control for each column.
        for col in column_names:
            col_type = self._controller.get_column_type(col)
            if col_type is None:
                continue

            if col_type.tag == "text":
                self._add_text_filter(col)
            elif col_type.tag == "numeric":
                self._add_numeric_filter(col)
            elif col_type.tag == "boolean_flag":
                self._add_boolean_filter(col)
            elif col_type.tag in ("media_path",):
                # Media columns — no filter control needed.
                pass
            else:
                # Unknown type — show a non-interactive label so the
                # column is at least visible in the panel.
                label = QLabel(f"{col} ({col_type.tag})")
                label.setStyleSheet("color: #888888; font-size: 11px;")
                self._layout.insertWidget(
                    self._layout.count() - 1, label
                )

        # Controls now exist for every filterable column -- show whatever
        # the controller is actually holding for the (possibly just
        # switched-to) active table, rather than leaving every control at
        # its just-built default of "no filter".
        self._restore_query_state()

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
            self._toggle_buttons[column] = {}

            for val in values:
                btn = QPushButton(str(val))
                btn.setCheckable(True)
                btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                btn.clicked.connect(
                    lambda checked, c=column, v=val:
                    self._on_text_toggle(c, v, checked)
                )
                layout.addWidget(btn)
                self._toggle_buttons[column][val] = btn

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
            self._text_searches[column] = search

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

    def _add_numeric_filter(self, column: str) -> None:
        """
        Adds a min/max range control for a numeric column.

        The column's observed minimum and maximum bound two spin boxes
        (a lower "≥" bound and an upper "≤" bound). Moving either emits a
        Filter(column, "between", [lo, hi]). When both ends are back at
        the full observed range, the filter is removed entirely so the
        column stops constraining the view.

        Columns with fewer than two distinct numeric values can't be
        meaningfully ranged, so they fall back to a static label.

        Args:
            column: The numeric column to create a range control for.
        """
        values = [
            v for v in self._controller.get_group_values(column)
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if len(values) < 2:
            label = QLabel(f"{column} (numeric)")
            label.setStyleSheet("color: #888888; font-size: 11px;")
            self._layout.insertWidget(self._layout.count() - 1, label)
            return

        col_min, col_max = float(min(values)), float(max(values))

        # Whole-number columns read better without decimal places.
        is_integer = all(float(v).is_integer() for v in values)
        decimals   = 0 if is_integer else 3
        step       = 1.0 if is_integer else max((col_max - col_min) / 100, 0.001)

        # Rebuilding the control resets it to the full range, i.e. no
        # active filter — match that here.
        self._active_filters.pop(column, None)

        group  = QGroupBox(column)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        min_spin = self._make_range_spin(col_min, col_max, decimals, step, "≥ ")
        max_spin = self._make_range_spin(col_min, col_max, decimals, step, "≤ ")
        min_spin.setValue(col_min)
        max_spin.setValue(col_max)
        layout.addWidget(min_spin)
        layout.addWidget(max_spin)

        self._numeric_spins[column] = (min_spin, max_spin, col_min, col_max)
        min_spin.valueChanged.connect(
            lambda _v, c=column: self._on_numeric_changed(c)
        )
        max_spin.valueChanged.connect(
            lambda _v, c=column: self._on_numeric_changed(c)
        )

        self._layout.insertWidget(self._layout.count() - 1, group)

    def _make_range_spin(
        self,
        col_min: float,
        col_max: float,
        decimals: int,
        step: float,
        prefix: str,
    ) -> QDoubleSpinBox:
        """Builds one bound spin box for a numeric range control."""
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(col_min, col_max)
        spin.setSingleStep(step)
        spin.setPrefix(prefix)
        return spin

    def _cap_numeric_range_spins(
        self,
        min_spin: QDoubleSpinBox,
        max_spin: QDoubleSpinBox,
        col_min: float,
        col_max: float,
        lo: float,
        hi: float,
    ) -> None:
        """
        Stops the two handles of a numeric range control crossing each
        other: resets each spin's own Qt range to the full column range,
        then tightens min_spin's maximum to hi and max_spin's minimum to
        lo, so neither can be dragged or typed past the other. Signals
        are blocked around this so tightening one bound doesn't recurse
        back into the control's own valueChanged handler.

        Shared by _on_numeric_changed() (after a live edit) and
        _restore_one_filter() (after setValue() restores a remembered
        filter) -- both need the same invariant re-established, or the
        control that just had its value set programmatically accepts an
        edit past its intended bound as though no filter were active.
        """
        min_spin.blockSignals(True)
        max_spin.blockSignals(True)

        min_spin.setMinimum(col_min)
        min_spin.setMaximum(col_max)

        max_spin.setMinimum(col_min)
        max_spin.setMaximum(col_max)

        min_spin.setMaximum(hi)
        max_spin.setMinimum(lo)

        min_spin.blockSignals(False)
        max_spin.blockSignals(False)

    def _on_numeric_changed(self, column: str) -> None:
        """
        Called when either bound of a numeric range control changes.

        Keeps the two bounds from crossing, then emits a 'between' filter
        for the current [lo, hi] — or removes the filter when the range
        covers the whole column again.

        Args:
            column: The numeric column being filtered.
        """
        entry = self._numeric_spins.get(column)
        if entry is None:
            return
        min_spin, max_spin, col_min, col_max = entry

        lo, hi = min_spin.value(), max_spin.value()
        self._cap_numeric_range_spins(min_spin, max_spin, col_min, col_max, lo, hi)

        if lo <= col_min and hi >= col_max:
            # Full range selected — no constraint.
            self._active_filters.pop(column, None)
        else:
            self._active_filters[column] = Filter(column, "between", [lo, hi])

        self.filters_changed.emit(list(self._active_filters.values()))

    def _add_boolean_filter(self, column: str) -> None:
        """
        Adds an All / True / False selector for a boolean column.

        "All" clears the filter; "True"/"False" emit Filter(column, "eq",
        value).

        Args:
            column: The boolean column to create a selector for.
        """
        # Rebuilding the control resets it to "All", i.e. no active filter.
        self._active_filters.pop(column, None)

        group  = QGroupBox(column)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        combo = QComboBox()
        combo.addItem("All", userData=None)
        combo.addItem("True", userData=True)
        combo.addItem("False", userData=False)
        combo.currentIndexChanged.connect(
            lambda idx, c=column, cb=combo:
            self._on_boolean_changed(c, cb.itemData(idx))
        )
        layout.addWidget(combo)
        self._boolean_combos[column] = combo

        self._layout.insertWidget(self._layout.count() - 1, group)

    def _on_boolean_changed(self, column: str, value) -> None:
        """
        Called when a boolean selector changes.

        Args:
            column: The boolean column being filtered.
            value:  True, False, or None ("All" — clears the filter).
        """
        if value is None:
            self._active_filters.pop(column, None)
        else:
            self._active_filters[column] = Filter(column, "eq", value)

        self.filters_changed.emit(list(self._active_filters.values()))

    def _restore_query_state(self) -> None:
        """
        Re-syncs every control -- and self._active_filters -- to what
        the controller is actually holding for the active table.

        Called once, at the end of refresh_columns(), after every
        control has been rebuilt fresh (and so is showing "no filter").
        Before this existed, a table switch or a load left every control
        showing that just-rebuilt blank state even though the controller
        had already restored or forgotten a real filter set --
        self._active_filters was a second record of "what is filtered"
        that only ever moved forward from a click here, never back from
        the controller, so the two silently disagreed until the
        researcher next touched a control by hand.

        Every control is set with its signals blocked: restoring a
        filter must not re-emit filters_changed/group_by_changed and
        trigger a second, redundant query on top of the one that already
        produced the gallery on screen.
        """
        state: QueryState = self._controller.get_query_state()
        self._active_filters = {f.column: f for f in state.filters}

        for column, filter_ in self._active_filters.items():
            self._restore_one_filter(column, filter_)

        self._group_combo.blockSignals(True)
        index = self._index_for_data(self._group_combo, state.group_by)
        self._group_combo.setCurrentIndex(index if index != -1 else 0)
        self._group_combo.blockSignals(False)
        # Read back what the combo actually landed on, not the raw
        # state.group_by value it was asked to match -- a remembered
        # group_by absent from this table's columns falls back to the
        # combo's "None" entry (index -1 above), and the slider must
        # agree with what the combo is showing, not with a value the
        # combo could not select.
        self._set_group_height_enabled(
            self._group_combo.currentData() is not None
        )

    def _index_for_data(self, combo: QComboBox, data) -> int:
        """
        The index of the item in *combo* whose userData equals *data*, or
        -1 if none does. Not QComboBox.findData(): its Qt-level equality
        check is not guaranteed to treat two ``None`` payloads (the
        "None"/"All" entry every combo here carries at index 0) as
        equal, and this only ever needs a plain Python comparison.
        """
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                return i
        return -1

    def _restore_one_filter(self, column: str, filter_: Filter) -> None:
        """
        Sets the one control built for *column* to match *filter_*,
        signals blocked. A column whose remembered filter's operator
        does not match the kind of control built for it (e.g. the
        column's cardinality changed since the filter was set, so it now
        has a text search box instead of toggle buttons) is left at its
        default -- self._active_filters still carries the filter, so it
        is not lost, only not shown as checked/typed/dragged here.
        """
        if column in self._toggle_buttons:
            checked = set(filter_.value) if filter_.comparison == "isin" else set()
            self._toggle_values[column] = checked
            for value, btn in self._toggle_buttons[column].items():
                btn.blockSignals(True)
                btn.setChecked(value in checked)
                btn.blockSignals(False)

        elif column in self._text_searches:
            text = filter_.value if filter_.comparison == "contains" else ""
            search = self._text_searches[column]
            search.blockSignals(True)
            search.setText(text)
            search.blockSignals(False)

        elif column in self._numeric_spins:
            min_spin, max_spin, col_min, col_max = self._numeric_spins[column]
            lo, hi = filter_.value if filter_.comparison == "between" else (col_min, col_max)
            min_spin.blockSignals(True)
            max_spin.blockSignals(True)
            min_spin.setValue(lo)
            max_spin.setValue(hi)
            min_spin.blockSignals(False)
            max_spin.blockSignals(False)
            # Without this, the rebuilt spins keep the full column range
            # as their own Qt bounds instead of being mutually capped at
            # [lo, hi] the way a live edit leaves them -- so the very
            # next edit on either spin is free to cross the other and
            # produce an inverted (lo > hi) 'between' filter, which the
            # QueryEngine then applies as an always-empty mask.
            self._cap_numeric_range_spins(min_spin, max_spin, col_min, col_max, lo, hi)

        elif column in self._boolean_combos:
            combo = self._boolean_combos[column]
            value = filter_.value if filter_.comparison == "eq" else None
            index = self._index_for_data(combo, value)
            if index != -1:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)

    def _on_group_by_changed(self, index: int) -> None:
        """
        Called when the group-by combo selection changes.

        Args:
            index: The selected index in the combo box.
        """
        column = self._group_combo.itemData(index)
        # The group-height slider only matters when a real grouping
        # column is selected (anything other than "None").
        self._set_group_height_enabled(column is not None)
        self.group_by_changed.emit(column)

    def _set_group_height_enabled(self, enabled: bool) -> None:
        """
        Enables or disables the group-height label and slider together.
        Disabled (greyed-out) when no grouping column is active, since the
        slider has no effect on the flat gallery view.

        Args:
            enabled: True to enable the control, False to grey it out.
        """
        self._group_height_label.setEnabled(enabled)
        self._group_height_slider.setEnabled(enabled)

    def get_active_filters(self) -> list[Filter]:
        """
        Returns the currently active list of Filter objects.

        Returns:
            List of active Filter objects.
        """
        return list(self._active_filters.values())
