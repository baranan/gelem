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

from dataclasses import dataclass, field
from html import escape as _html_escape
from typing import Optional

from PySide6.QtCore import Qt
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
from operators.form_advice import FormAdvice, FormAdviceError, FormMessage


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
# Layer A -- applying a piece of operator FormAdvice against the current
# values. Still Qt-free: this decides what the widget layer must show and
# whether acceptance is blocked, but touches no widget.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedForm:
    """Everything the widget layer needs to reflect one ``FormAdvice``
    against the current values.

      * ``disabled_fields`` -- parameter names whose widget is disabled,
        because the operator called them inapplicable. Their values are
        PRESERVED, never cleared.
      * ``allowed_values``  -- parameter name -> the tuple of values still
        allowed, for every field the operator restricted, passed through
        as the operator declared it. A field absent here is unrestricted.
        An entry MAY name a field that is also in ``disabled_fields`` --
        the operator restricted it and called it inapplicable in the same
        answer. The widget layer (P1.12e-4b) must apply ``disabled_fields``
        first and never prune a disabled widget's items, so its preserved
        value survives.
      * ``messages``        -- every ``FormMessage`` to display: the ones
        the operator returned, plus any this layer generated for an
        applicable field whose current value is not among its allowed
        values.
      * ``blocked``         -- whether acceptance is blocked. See
        ``resolve_form`` for the two things that block.
    """

    disabled_fields: tuple[str, ...] = ()
    allowed_values: dict = field(default_factory=dict)
    messages: tuple[FormMessage, ...] = ()
    blocked: bool = False


def _violates_restriction(current_value, allowed_values, spec: FieldSpec) -> bool:
    """Whether this field's current raw value falls outside the values the
    operator still allows.

    A blank / unset field is NOT a violation here -- ``collect_parameters``
    is what refuses a required field left empty, and blocking twice for
    one problem only confuses. For a multi-column picker every picked
    column must be allowed; an empty pick counts as blank.
    """
    # A multi-column selection is a list of column names; each one must be
    # allowed.
    if spec.kind == "column" and spec.allow_multiple:
        picked = tuple(current_value or ())
        if not picked:
            return False
        return any(column not in allowed_values for column in picked)

    # Every other kind holds a single scalar value.
    if _is_blank(current_value):
        return False
    return current_value not in allowed_values


def _restriction_message(spec: FieldSpec, allowed_values) -> FormMessage:
    """The message this layer generates for an applicable field whose
    value is not allowed.

    The operator states the restriction ("a histogram allows only count,
    sum and mean"); naming the field and listing the valid choices is the
    form's own job, so the operator does not also have to say "and your
    current value is wrong".
    """
    choices = ", ".join(str(value) for value in allowed_values)
    return FormMessage(
        text=(
            f"{spec.label}: the current value is not allowed here. "
            f"Choose one of: {choices}."
        ),
        severity="error",
        field=spec.name,
    )


def resolve_form(field_specs, advice, raw_values) -> ResolvedForm:
    """Fold one ``FormAdvice`` together with the current values into the
    concrete state the widget layer renders.

    ``field_specs`` -- the fields, from ``build_field_specs``.
    ``advice``      -- what the operator's ``refine_form`` returned
                       (``FormAdvice()`` when it has no guidance, or the
                       operator declares none).
    ``raw_values``  -- parameter name -> the widget's current raw value,
                       the same shape ``collect_parameters`` reads.

    Blocking is decided HERE, from two sources:

      * any operator ``FormMessage`` of severity "error";
      * any APPLICABLE field whose current value is not among its allowed
        values -- this layer generates that message itself.

    A field the operator listed as inapplicable is disabled and its value
    is left untouched, and a restriction on an inapplicable field never
    blocks: the researcher cannot edit a disabled field, so blocking on
    one would trap them with no way out.

    RAISES ``FormAdviceError`` if ``allowed_choices`` names a multi-select
    column field (a ``ColumnParameter`` with ``allow_multiple=True``).
    Narrowing one would mean un-picking a column the researcher already
    chose to keep the selection consistent with the restriction, and
    advice must never write a value back (see ``operators/form_advice.py``).
    The check lives here rather than on ``FormAdvice`` itself: a
    ``FormAdvice`` is a bare mapping of parameter names to tuples with no
    field-kind information, so it cannot tell a multi-select column field
    from any other by name alone. This function is the one place advice
    and the field declarations meet, so it is the one place that can
    refuse the combination -- and it raises rather than degrading,
    because an operator that does this has a bug to fix, not a case the
    form should silently paper over. An operator that needs to react to a
    multi-select field marks it inapplicable instead.

    Pure: it reads its three arguments, mutates none of them, and holds no
    state between calls. Called again with different values, the result is
    computed entirely from those values and that advice -- never
    accumulated onto the previous answer.
    """
    for spec in field_specs:
        if (
            spec.kind == "column"
            and spec.allow_multiple
            and spec.name in advice.allowed_choices
        ):
            raise FormAdviceError(
                f"FormAdvice.allowed_choices names {spec.name!r}, a "
                f"multi-select column field. A multi-select field cannot "
                f"be narrowed -- narrowing it would have to un-pick a "
                f"column the researcher already chose, and advice must "
                f"never write a value back. Mark {spec.name!r} "
                f"inapplicable instead."
            )

    inapplicable = set(advice.inapplicable)

    # Disabled fields, in form order.
    disabled_fields = tuple(
        spec.name for spec in field_specs if spec.name in inapplicable
    )

    # The operator's restrictions, copied so the caller cannot reach back
    # into the advice through the result.
    allowed_values = dict(advice.allowed_choices)

    # Start from the operator's own messages; generated ones are appended.
    messages = list(advice.messages)

    # An operator error message blocks on its own.
    blocked = any(
        message.severity == "error" for message in advice.messages
    )

    # A restriction bites only on an APPLICABLE field. Walk the fields in
    # form order so any generated messages read top-to-bottom.
    for spec in field_specs:
        if spec.name not in allowed_values:
            continue
        if spec.name in inapplicable:
            # A restriction on a disabled field never blocks.
            continue
        current_value = raw_values.get(spec.name)
        allowed = allowed_values[spec.name]
        if _violates_restriction(current_value, allowed, spec):
            blocked = True
            messages.append(_restriction_message(spec, allowed))

    return ResolvedForm(
        disabled_fields=disabled_fields,
        allowed_values=allowed_values,
        messages=tuple(messages),
        blocked=blocked,
    )


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
    (see ``build_field_specs``), an optional parent, and an optional
    ``advice_provider``. Call ``exec()``; on accept, read the values back
    with ``parameter_values()``.

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

    ``advice_provider`` (P1.12e-4b) is a callable ``raw_values -> FormAdvice``
    -- an operator's ``refine_form`` bound method. The dialog depends on
    the FUNCTION, never on the operator object. It is called once before
    the dialog is first shown and again on every field change; its answer
    is folded through ``resolve_form`` and rendered:

      * a field the operator calls inapplicable is DISABLED, its value
        left untouched;
      * a field the operator restricted has its now-disallowed items
        DISABLED, never removed -- rebuilding the list on every change is
        the accumulating state the design forbids. The current selection is
        always kept: a disabled current item is fine, and OK just stays
        blocked. Only single-value fields (``choice``, single ``column``)
        can be narrowed this way -- ``resolve_form`` REFUSES advice that
        narrows a multi-select column field, because there the only way to
        keep it consistent would be to un-pick a column the researcher
        already chose, and advice must never write a value back;
      * every ``FormMessage`` is shown, warnings and errors in different
        colours;
      * OK is disabled while ``resolve_form`` reports the form blocked.

    When no ``advice_provider`` is given (or the operator's ``refine_form``
    returns the default empty ``FormAdvice()``), the form behaves exactly
    as it did before P1.12e-4b.
    """

    def __init__(
        self,
        mode_descriptor,
        columns_by_input,
        parent=None,
        advice_provider=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"{mode_descriptor.label} -- parameters")

        # Layer A decides every field; Layer B only renders them.
        self._field_specs = build_field_specs(mode_descriptor, columns_by_input)

        # name -> the widget built for that field.
        self._widgets: dict = {}

        # A callable raw_values -> FormAdvice. The default gives no
        # guidance, so a form with no provider behaves as it did before
        # this item.
        self._advice_provider = advice_provider or (
            lambda raw_values: FormAdvice()
        )

        # Re-entrancy guard. Applying advice changes widgets, which fires
        # their change signals, which would ask for advice again. While
        # this flag is set, _on_field_changed returns immediately.
        self._applying_advice = False

        # The dropdown fields the last advice narrowed. Only these need
        # re-widening when a later advice lifts the restriction -- a field
        # that was never narrowed is already in its full state.
        self._narrowed_fields: set[str] = set()

        # Filled in by _build().
        self._message_label: Optional[QLabel] = None
        self._ok_button = None

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

        # The one place operator FormMessages are shown. Rich text so a
        # warning and an error read in different colours; hidden until
        # there is something to say.
        self._message_label = QLabel()
        self._message_label.setWordWrap(True)
        self._message_label.setTextFormat(Qt.TextFormat.RichText)
        self._message_label.setVisible(False)
        layout.addWidget(self._message_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        layout.addWidget(buttons)

        # Wire every widget's change signal to a single handler, then run
        # the advice once so the form opens in its resolved state -- before
        # it is ever shown.
        self._connect_change_signals()
        self._apply_advice()

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

    # -- form guidance ----------------------------------------------

    def _connect_change_signals(self) -> None:
        """Route every widget's "the researcher changed me" signal to
        ``_on_field_changed``.

        One signal per kind -- the one that fires on a user edit and
        carries no argument we need.
        """
        for spec in self._field_specs:
            widget = self._widgets[spec.name]
            if spec.kind == "number":
                widget.valueChanged.connect(self._on_field_changed)
            elif spec.kind == "boolean":
                widget.toggled.connect(self._on_field_changed)
            elif spec.kind == "choice":
                widget.currentIndexChanged.connect(self._on_field_changed)
            elif spec.kind in ("text", "new_table_name"):
                widget.textChanged.connect(self._on_field_changed)
            elif spec.kind == "column":
                if spec.allow_multiple:
                    widget.itemSelectionChanged.connect(self._on_field_changed)
                else:
                    widget.currentIndexChanged.connect(self._on_field_changed)

    def _on_field_changed(self, *_args) -> None:
        """A field changed. Recompute and re-apply the whole advice --
        unless we are the ones changing it right now."""
        if self._applying_advice:
            return
        self._apply_advice()

    def _apply_advice(self) -> None:
        """Ask the provider for advice on the current values, fold it
        through ``resolve_form``, and render the result.

        The re-entrancy guard is held across the render: applying advice
        changes widgets, which can fire a field's change signal back into
        ``_on_field_changed`` mid-render -- without the guard that would
        call straight back in here. (Narrowing used to be the concrete
        case that triggered this, by un-picking a multi-select column's
        now-disallowed entry; ``resolve_form`` now refuses that advice
        outright, so the guard is defence in depth rather than a path a
        test can currently force.)

        ``resolve_form`` can itself raise -- see its docstring for when --
        and that is left to propagate rather than caught here: the
        provider-call try/except below is only for a broken *operator*;
        a ``resolve_form`` refusal means the FormAdvice it returned is
        malformed, which is the same kind of bug ``build_field_specs``
        raising ``ParameterFormError`` is for an unrenderable parameter --
        loud, at the point the bad advice was produced.

        A broken refine_form must not take down the Operators menu (this
        runs during dialog construction) or freeze all further guidance
        (it runs again on every change, in a Qt slot). Its contract is to
        be pure, to return a FormAdvice, and to tolerate an incomplete
        form; a raise, a None (forgotten return) or a wrong type all
        degrade this cycle to a plain form, and the bug surfaces in the
        operator's own tests rather than here.
        """
        raw_values = self._raw_values()
        try:
            advice = self._advice_provider(raw_values)
            if not isinstance(advice, FormAdvice):
                advice = FormAdvice()
        except Exception:
            advice = FormAdvice()

        resolved = resolve_form(self._field_specs, advice, raw_values)

        self._applying_advice = True
        try:
            self._render_resolved(resolved)
        finally:
            self._applying_advice = False

    def _render_resolved(self, resolved: ResolvedForm) -> None:
        """Reflect one ``ResolvedForm`` onto the widgets.

        Order matters and follows the P1.12e-4a note: disable the
        inapplicable fields FIRST, then narrow allowed values -- an
        inapplicable field is skipped by the narrowing pass so its items
        are never pruned. Narrowing disables the now-disallowed items
        without removing them, so the field's current selection is always
        kept and OK just stays blocked (via ``resolved.blocked``) until the
        researcher picks an allowed value. This only ever reaches a
        single-value field (``choice``, single ``column``): ``resolve_form``
        refuses advice that narrows a multi-select column field before this
        method is ever called.
        """
        disabled = set(resolved.disabled_fields)

        # 1. Disabled fields first. A disabled widget keeps its value; we
        #    never touch its items.
        for spec in self._field_specs:
            self._widgets[spec.name].setEnabled(spec.name not in disabled)

        # 2. Narrow the still-editable dropdowns. A field that is also
        #    disabled is skipped -- its items must not be pruned. Widening
        #    only touches a field a PREVIOUS advice actually narrowed;
        #    every other field is already in its full state, so re-walking
        #    its rows on every keystroke would be pure overhead.
        now_narrowed: set[str] = set()
        for spec in self._field_specs:
            if spec.name in disabled:
                # Left untouched this pass; its items keep whatever state
                # they had, so remember a still-standing narrowing.
                if spec.name in self._narrowed_fields:
                    now_narrowed.add(spec.name)
                continue
            allowed = resolved.allowed_values.get(spec.name)
            if allowed is None:
                if spec.name in self._narrowed_fields:
                    self._widen_widget(spec)
            else:
                self._narrow_widget(spec, allowed)
                now_narrowed.add(spec.name)
        self._narrowed_fields = now_narrowed

        # 3. The messages, warnings and errors visually distinct.
        self._render_messages(resolved.messages)

        # 4. OK follows `blocked` exactly.
        if self._ok_button is not None:
            self._ok_button.setEnabled(not resolved.blocked)

    def _widen_widget(self, spec: FieldSpec) -> None:
        """Re-enable every item of a dropdown field -- the state a field
        with no restriction must be in.

        Only ever called for ``choice`` or a single-selection ``column``:
        ``resolve_form`` refuses advice that narrows a multi-select column
        field, so ``_narrowed_fields`` (see ``_render_resolved``) never
        contains one, and this is never asked to widen one either.
        """
        widget = self._widgets[spec.name]
        model = widget.model()
        for row in range(widget.count()):
            item = model.item(row)
            if item is not None:
                item.setEnabled(True)

    def _narrow_widget(self, spec: FieldSpec, allowed) -> None:
        """Disable the items of this field that are no longer allowed.

        Never removes an item: removing loses the current selection and
        forces the original list to be restored on the next change. The
        blank "unset" entry (data ``None``) is always left enabled.

        Only ever called for ``choice`` or a single-selection ``column``:
        ``resolve_form`` refuses advice that narrows a multi-select column
        field before ``allowed`` can reach here (see its docstring for why
        -- narrowing one would mean un-picking a column the researcher
        already chose, and advice must never write a value back).
        """
        allowed_set = set(allowed)
        widget = self._widgets[spec.name]
        model = widget.model()
        for row in range(widget.count()):
            data = widget.itemData(row)
            item = model.item(row)
            if item is None:
                continue
            item.setEnabled(data is None or data in allowed_set)

    def _render_messages(self, messages) -> None:
        """Show the ``FormMessage`` list, each line coloured by severity.

        The wording is the operator's / Layer A's; this only decides the
        colour and the "Warning" / "Error" prefix.
        """
        if not messages:
            self._message_label.clear()
            self._message_label.setVisible(False)
            return

        blocks = []
        for message in messages:
            if message.severity == "error":
                colour, prefix = "#B00020", "Error"
            else:
                colour, prefix = "#B36B00", "Warning"
            blocks.append(
                f'<div style="color: {colour}; margin-bottom: 4px;">'
                f"<b>{prefix}:</b> {_html_escape(message.text)}</div>"
            )
        self._message_label.setText("".join(blocks))
        self._message_label.setVisible(True)

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
