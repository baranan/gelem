"""
ui/parameter_dialog.py

The GENERATED operator-parameter dialog (P1.12e-1).

Every operator mode declares its parameters as a tuple of
``ParameterSpec`` subclasses on its ``ModeDescriptor`` (see
``operators/descriptor.py``). Today each operator hand-builds a QDialog to
collect those values. This module builds that dialog straight from the
declarations instead, so a new operator gets a working parameter form with
no UI code of its own.

Nothing calls this yet. ``ui/main_window.py`` still asks each operator for
``get_parameters_dialog(...)``. P1.12e-2 does the substitution; this file
only has to satisfy the same contract MainWindow already relies on:

  * the object MainWindow shows is a ``QDialog`` -- it calls ``exec()`` and
    treats a zero return as "cancelled";
  * after the dialog is accepted MainWindow calls
    ``parameter_values() -> dict`` and hands the dict to the controller,
    which validates it against the descriptor. A dialog without that
    method is a bug MainWindow raises on, so ``ParameterDialog`` exposes
    it.

TWO LAYERS, ONE FILE -- the shape ``ui/settings_dialog.py`` established:

  * Layer A -- module-level, Qt-free functions. Every piece of
    researcher-facing wording and every coercion / filter / validation
    rule lives here, so the tests exercise them with no QApplication.

  * Layer B -- ``class ParameterDialog(QDialog)``, thin glue. It builds one
    widget per ``FieldSpec``, reads the widgets back into a plain dict, and
    calls Layer A. It holds no rules of its own.

Importing ``operators.descriptor`` into a UI file is fine: that module is
standard-library only and pulls in no data library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
)

from operators.descriptor import (
    BooleanParameter,
    ChoiceParameter,
    ColumnParameter,
    NewTableNameParameter,
    NumberParameter,
    TextParameter,
)


# ---------------------------------------------------------------------------
# Layer A -- plain data, no Qt.
# ---------------------------------------------------------------------------


class ParameterFormError(Exception):
    """Raised by ``collect_parameters`` when a REQUIRED parameter was left
    empty.

    This is a courtesy check, not a second rule set: ``OperatorRunSpec``
    (``operators/run_context.py``) remains the authority on whether a set
    of parameter values is valid and refuses a bad run regardless. This
    check exists only so the researcher hears "you left X empty" while the
    dialog is still open, rather than seeing the run rejected afterwards.
    The message names the offending parameter by its researcher-facing
    label.
    """


@dataclass(frozen=True)
class FieldSpec:
    """One row of the generated form, in researcher-facing terms.

    A ``FieldSpec`` is what Layer B turns into a widget and what
    ``collect_parameters`` reads back. It is deliberately flat: every kind
    of parameter is described by the same dataclass, with only the fields
    that kind needs filled in.

    Fields common to every kind:
      * ``name``      -- the internal key the operator reads from
                         ``run.parameters``; the key this field's value
                         lands under in the collected dict.
      * ``kind``      -- the ``ParameterSpec`` subclass's ``kind`` string
                         ("number", "text", "boolean", "choice", "column",
                         "new_table_name"). Layer B picks the widget from
                         this.
      * ``label``     -- what the researcher sees beside the field.
      * ``help_text`` -- optional one-line explanation.
      * ``required``  -- whether an empty value is refused before the
                         dialog closes.
      * ``default``   -- the value the widget starts on, or ``None``.

    Kind-specific fields (unused ones stay at their defaults):
      * ``choices``        -- ``(value, label)`` pairs for a "choice".
      * ``minimum`` / ``maximum`` / ``decimals`` -- a "number"'s bounds and
                         display precision.
      * ``column_names``   -- the column names offered by a "column",
                         already filtered to the right input and tags.
      * ``allow_multiple`` -- whether a "column" lets the researcher pick
                         more than one.
    """

    name: str
    kind: str
    label: str
    help_text: str = ""
    required: bool = True
    default: object = None

    # choice
    choices: tuple[tuple[str, str], ...] = ()

    # number
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    decimals: int = 0

    # column
    column_names: tuple[str, ...] = ()
    allow_multiple: bool = False


def build_field_specs(mode_descriptor, columns_by_input) -> tuple[FieldSpec, ...]:
    """Turn a ``ModeDescriptor``'s declared parameters into an ordered tuple
    of ``FieldSpec``.

    The order of the result matches the order the parameters are declared
    in, so the form reads top-to-bottom the way the operator author wrote
    it.

    ``columns_by_input`` maps a declared input's NAME to the columns that
    input offers, each as a ``(column_name, type_tag)`` pair. A
    ``ColumnParameter`` names one of the mode's inputs in ``from_input``;
    this function offers only that input's columns, and -- when the
    parameter's ``required_tags`` is non-empty -- only the columns whose
    tag is one of those. An empty ``required_tags`` offers every column of
    the input.

    In P1.12e-2 the caller builds ``columns_by_input`` from the controller:
    resolve each ``InputSpec`` to a table name, then pair
    ``controller.get_column_names(table)`` with
    ``controller.get_column_type(column, table).tag`` for each column. Both
    accessors are public, so no boundary rule is crossed.
    """
    specs: list[FieldSpec] = []

    for parameter in mode_descriptor.parameters:
        # -- a plain number: spin box, with whatever bounds were declared --
        if isinstance(parameter, NumberParameter):
            specs.append(
                FieldSpec(
                    name=parameter.name,
                    kind=parameter.kind,
                    label=parameter.label,
                    help_text=parameter.help_text,
                    required=parameter.required,
                    default=parameter.default,
                    minimum=parameter.minimum,
                    maximum=parameter.maximum,
                    decimals=parameter.decimals,
                )
            )
            continue

        # -- yes / no: a checkbox --
        if isinstance(parameter, BooleanParameter):
            specs.append(
                FieldSpec(
                    name=parameter.name,
                    kind=parameter.kind,
                    label=parameter.label,
                    help_text=parameter.help_text,
                    required=parameter.required,
                    default=parameter.default,
                )
            )
            continue

        # -- pick one of a fixed list: a dropdown --
        if isinstance(parameter, ChoiceParameter):
            specs.append(
                FieldSpec(
                    name=parameter.name,
                    kind=parameter.kind,
                    label=parameter.label,
                    help_text=parameter.help_text,
                    required=parameter.required,
                    default=parameter.default,
                    choices=tuple(parameter.choices),
                )
            )
            continue

        # -- free text, and the "name a new table" case: a line edit.
        #    NewTableNameParameter is checked before TextParameter only for
        #    readability; neither subclasses the other. --
        if isinstance(parameter, (TextParameter, NewTableNameParameter)):
            specs.append(
                FieldSpec(
                    name=parameter.name,
                    kind=parameter.kind,
                    label=parameter.label,
                    help_text=parameter.help_text,
                    required=parameter.required,
                    default=parameter.default,
                )
            )
            continue

        # -- pick a column (or several) from one named input --
        if isinstance(parameter, ColumnParameter):
            # The columns the named input offers, each (name, tag). A caller
            # that did not supply this input contributes no columns rather
            # than raising -- the dropdown is simply empty.
            available = tuple(columns_by_input.get(parameter.from_input, ()))

            if parameter.required_tags:
                allowed_tags = set(parameter.required_tags)
                offered = tuple(
                    column_name
                    for column_name, type_tag in available
                    if type_tag in allowed_tags
                )
            else:
                # Empty required_tags means "any column of this input".
                offered = tuple(column_name for column_name, _tag in available)

            # A single-selection ColumnParameter may name a column to
            # preselect. A default that names a column this input does not
            # offer falls back to no preselection, rather than the form
            # showing a column that is not there. (allow_multiple together
            # with a default is refused at descriptor construction, so this
            # only ever matters for the single-selection case.)
            default_column = parameter.default
            if default_column is not None and default_column not in offered:
                default_column = None

            specs.append(
                FieldSpec(
                    name=parameter.name,
                    kind=parameter.kind,
                    label=parameter.label,
                    help_text=parameter.help_text,
                    required=parameter.required,
                    default=default_column,
                    column_names=offered,
                    allow_multiple=parameter.allow_multiple,
                )
            )
            continue

        # -- an unknown ParameterSpec subclass. This item's job is to find
        #    exactly this case: a declared kind the form cannot render. It
        #    should not happen for the six kinds that exist today. --
        raise ParameterFormError(
            f"parameter {parameter.name!r}: kind "
            f"{type(parameter).__name__} cannot be rendered by the "
            f"generated parameter form."
        )

    return tuple(specs)


def _coerce_number(raw_value, decimals: int):
    """Coerce a spin box's value to the number the run spec expects.

    ``decimals == 0`` means the parameter is an integer, so a whole number
    is returned even though the widget may hand back a float. A non-zero
    ``decimals`` keeps it a float.
    """
    if decimals == 0:
        return int(round(float(raw_value)))
    return float(raw_value)


def _is_blank(raw_value) -> bool:
    """Whether a text-like raw value counts as "the researcher left it empty"."""
    if raw_value is None:
        return True
    if isinstance(raw_value, str) and raw_value.strip() == "":
        return True
    return False


def collect_parameters(field_specs, raw_values) -> dict:
    """Turn what the widgets hold into the parameters dict a run expects.

    ``raw_values`` maps a field name to that widget's current value, in
    whatever type the widget hands back: a number for a spin box, a string
    for a line edit or single-column dropdown, a bool for a checkbox, the
    chosen choice VALUE for a dropdown, and a list of column names for a
    multi-column picker.

    The returned dict is keyed by parameter name and is ready to hand to
    the controller. Rules applied here:

      * numbers are coerced (int when ``decimals == 0``, else float);
      * a multi-column selection becomes a tuple;
      * an OPTIONAL text, choice or column parameter the researcher left
        unset is OMITTED from the dict entirely -- an empty string or empty
        selection is never passed through;
      * a REQUIRED parameter left unset raises ``ParameterFormError``,
        naming the parameter by its label.

    A number and a boolean have no "unset" state -- a spin box and a
    checkbox always hold a value -- so they are always emitted, whatever
    ``required`` says. An operator that wants "use my own default when the
    researcher does not care" declares the default on the ``NumberParameter``
    / ``BooleanParameter`` itself.

    ``OperatorRunSpec`` still validates the result; see
    ``ParameterFormError``.
    """
    collected: dict = {}

    for spec in field_specs:
        raw_value = raw_values.get(spec.name)

        # -- number: a spin box always holds a value, so it is never
        #    "unset". Coerce and keep. --
        if spec.kind == "number":
            collected[spec.name] = _coerce_number(raw_value, spec.decimals)
            continue

        # -- boolean: a checkbox always holds a state; keep it as a bool. --
        if spec.kind == "boolean":
            collected[spec.name] = bool(raw_value)
            continue

        # -- choice: the raw value is the chosen choice value, or None when
        #    nothing is selected. --
        if spec.kind == "choice":
            if raw_value is None or raw_value == "":
                if spec.required:
                    raise ParameterFormError(
                        f"Please choose a value for {spec.label!r}."
                    )
                continue
            collected[spec.name] = raw_value
            continue

        # -- text and new-table-name: a line edit. Blank means unset. --
        if spec.kind in ("text", "new_table_name"):
            if _is_blank(raw_value):
                if spec.required:
                    raise ParameterFormError(
                        f"Please enter a value for {spec.label!r}."
                    )
                continue
            collected[spec.name] = raw_value.strip()
            continue

        # -- column: one dropdown, or a multi-select list. --
        if spec.kind == "column":
            if spec.allow_multiple:
                picked = tuple(raw_value or ())
                if not picked:
                    if spec.required:
                        raise ParameterFormError(
                            f"Please choose at least one column for "
                            f"{spec.label!r}."
                        )
                    continue
                collected[spec.name] = picked
                continue

            if _is_blank(raw_value):
                if spec.required:
                    raise ParameterFormError(
                        f"Please choose a column for {spec.label!r}."
                    )
                continue
            collected[spec.name] = raw_value.strip()
            continue

        # Every kind build_field_specs produces is handled above; a miss
        # here means the two functions drifted.
        raise ParameterFormError(
            f"parameter {spec.name!r}: field kind {spec.kind!r} has no "
            f"collection rule."
        )

    return collected


# ---------------------------------------------------------------------------
# Layer B -- the Qt dialog. Thin glue over Layer A. No rules live here.
# ---------------------------------------------------------------------------

# A spin box needs finite bounds even when the parameter declared none.
# These are wide enough for any real research parameter and keep the widget
# from rejecting a typed value.
_INT_SPIN_FLOOR = -1_000_000_000
_INT_SPIN_CEIL = 1_000_000_000
_FLOAT_SPIN_FLOOR = -1.0e12
_FLOAT_SPIN_CEIL = 1.0e12


class ParameterDialog(QDialog):
    """A parameter form generated from a mode descriptor.

    Construct with the mode descriptor, the ``columns_by_input`` mapping
    (see ``build_field_specs``), and an optional parent. Call ``exec()``;
    on accept, read the values back with ``parameter_values()``.

    The widget chosen per kind:
      * number         -- QSpinBox (decimals 0) or QDoubleSpinBox;
      * text           -- QLineEdit;
      * new_table_name -- QLineEdit;
      * boolean        -- QCheckBox;
      * choice         -- QComboBox, item data = the choice value (a blank
                          first entry unless a default pre-selects one);
      * column, single -- QComboBox of column names (a blank first entry is
                          always present; the field starts on its default
                          column when the default names one this input
                          offers, otherwise on the blank entry);
      * column, multi  -- QListWidget in multi-selection mode.
    """

    def __init__(self, mode_descriptor, columns_by_input, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{mode_descriptor.label} -- parameters")

        # Layer A decides every field; Layer B only renders them.
        self._field_specs = build_field_specs(mode_descriptor, columns_by_input)

        # name -> the widget built for that field.
        self._widgets: dict = {}

        self._build()

    # -- building -------------------------------------------------------

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        for spec in self._field_specs:
            layout.addWidget(QLabel(spec.label))

            widget = self._widget_for(spec)
            self._widgets[spec.name] = widget
            layout.addWidget(widget)

            if spec.help_text:
                help_label = QLabel(spec.help_text)
                help_label.setWordWrap(True)
                layout.addWidget(help_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _widget_for(self, spec: FieldSpec):
        """The one widget this field's kind calls for."""
        if spec.kind == "number":
            if spec.decimals == 0:
                spin = QSpinBox()
                low = int(spec.minimum) if spec.minimum is not None else _INT_SPIN_FLOOR
                high = int(spec.maximum) if spec.maximum is not None else _INT_SPIN_CEIL
                spin.setRange(low, high)
                # QSpinBox.setValue needs an int; a declared default may be a
                # float (NumberParameter.default is Optional[float]).
                spin.setValue(int(round(self._clamped_start(spec, low, high))))
                return spin
            spin = QDoubleSpinBox()
            spin.setDecimals(spec.decimals)
            low = spec.minimum if spec.minimum is not None else _FLOAT_SPIN_FLOOR
            high = spec.maximum if spec.maximum is not None else _FLOAT_SPIN_CEIL
            spin.setRange(low, high)
            spin.setValue(self._clamped_start(spec, low, high))
            return spin

        if spec.kind == "boolean":
            box = QCheckBox()
            box.setChecked(bool(spec.default))
            return box

        if spec.kind == "choice":
            combo = QComboBox()
            # A combo with items always has a selection, so without a blank
            # entry a "required, no default" field can never read as empty
            # and collect_parameters' required check would be unreachable --
            # the researcher would silently get the first choice. Give it a
            # blank first entry (data None) unless a default pre-selects a
            # real one.
            if spec.default is None:
                combo.addItem("", None)
            for choice_value, choice_label in spec.choices:
                combo.addItem(choice_label, choice_value)
            if spec.default is not None:
                index = combo.findData(spec.default)
                if index >= 0:
                    combo.setCurrentIndex(index)
            return combo

        if spec.kind in ("text", "new_table_name"):
            line = QLineEdit()
            if spec.default:
                line.setText(str(spec.default))
            return line

        if spec.kind == "column":
            if spec.allow_multiple:
                listing = QListWidget()
                listing.setSelectionMode(
                    QListWidget.SelectionMode.ExtendedSelection
                )
                for column_name in spec.column_names:
                    listing.addItem(QListWidgetItem(column_name))
                return listing

            combo = QComboBox()
            # A blank first entry (data None) is always present. An optional
            # field needs it to say "none"; a required one needs it so
            # collect_parameters' required check is reachable rather than
            # the researcher silently getting the first column.
            combo.addItem("", None)
            for column_name in spec.column_names:
                combo.addItem(column_name, column_name)
            # A single-selection column field may name a column to start on.
            # build_field_specs has already dropped a default that names a
            # column this input does not offer, so findData either hits a
            # real entry or the field stays on the blank one.
            if spec.default is not None:
                index = combo.findData(spec.default)
                if index >= 0:
                    combo.setCurrentIndex(index)
            return combo

        # build_field_specs never produces another kind.
        raise ParameterFormError(
            f"parameter {spec.name!r}: no widget for field kind {spec.kind!r}."
        )

    @staticmethod
    def _clamped_start(spec: FieldSpec, low, high):
        """The spin box's opening value: the declared default, else the
        nearest end of the range to zero."""
        if spec.default is not None:
            start = spec.default
        elif low <= 0 <= high:
            start = 0
        else:
            start = low
        # Keep it inside the range whatever the default said.
        return max(low, min(high, start))

    # -- reading back --------------------------------------------------

    def _raw_values(self) -> dict:
        """Field name -> the widget's current value, in the widget's own type."""
        raw: dict = {}
        for spec in self._field_specs:
            widget = self._widgets[spec.name]

            if spec.kind == "number":
                raw[spec.name] = widget.value()
            elif spec.kind == "boolean":
                raw[spec.name] = widget.isChecked()
            elif spec.kind == "choice":
                raw[spec.name] = widget.currentData()
            elif spec.kind in ("text", "new_table_name"):
                raw[spec.name] = widget.text()
            elif spec.kind == "column":
                if spec.allow_multiple:
                    # Walk the list in row order, not selection order:
                    # QListWidget.selectedItems() order is not guaranteed to
                    # match what the researcher sees.
                    raw[spec.name] = [
                        widget.item(row).text()
                        for row in range(widget.count())
                        if widget.item(row).isSelected()
                    ]
                else:
                    raw[spec.name] = widget.currentData()
            else:  # pragma: no cover -- kinds are closed
                raw[spec.name] = None

        return raw

    def parameter_values(self) -> dict:
        """The parameters dict, keyed by descriptor parameter name.

        This is the method ``ui/main_window.py`` calls after the dialog is
        accepted. It delegates every rule to ``collect_parameters``.
        """
        return collect_parameters(self._field_specs, self._raw_values())

    # -- OK -----------------------------------------------------------

    def _on_ok(self) -> None:
        """Validate through Layer A before closing.

        A ``ParameterFormError`` here means a required field is empty; show
        it and keep the dialog open so the researcher can fix it, rather
        than closing and letting the controller reject the run.
        """
        try:
            collect_parameters(self._field_specs, self._raw_values())
        except ParameterFormError as error:
            QMessageBox.warning(self, "Missing value", str(error))
            return
        self.accept()
