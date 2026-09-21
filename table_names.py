"""
table_names.py

The one place that decides two things about a table name the researcher is
about to create: what free name to suggest (resolve_table_name), and
whether the name currently in a field is fine to submit (validate_new_table_name).

Table-name-validation round 4: moved here, unchanged, from controller.py
(resolve_table_name) and ui/parameter_dialog.py (validate_new_table_name).
Three consumers share these two functions -- ui/parameter_dialog.py's
operator-parameter NewTableNameParameter field, ui/merge_report_dialog.py's
expand-table-name field, and controller.py's own TABLE-mode operator
output-naming logic (it resolves the suggested default shown in the
operator parameter dialog, and resolves the final name again just before
calling Dataset.create_table_from_df to store the result) -- and before
this round ui/merge_report_dialog.py had to reach into controller.py for
one of them. A single Qt-free module both ui/ and controller.py can
import from is the fix; moving either function into the other's old home
would just relocate the same cross-boundary reach.

Standard library only -- no Qt, no pandas, no Dataset. This is deliberate,
not incidental: the ui/ files that import from here (ui/parameter_dialog.py,
ui/merge_report_dialog.py) must never end up transitively importing pandas
through it. This module is a plain top-level file with no package
__init__.py in its import path at all. A models/table_names.py would be
safe today too -- models/__init__.py happens to be empty -- but only for
as long as that stays true; a top-level module has no package __init__.py
in its import path to begin with, so there is nothing for a future edit
to accidentally reintroduce.

Dataset's own refusal at store time (create_table_from_df,
Dataset.confirm_merge) is the actual backstop for every consumer; every
function here is a courtesy shown before that point, never a second
authority -- see each function's own docstring.
"""

from __future__ import annotations

from typing import Optional


def resolve_table_name(suggested: str, existing) -> str:
    """The name a TABLE-mode run's result should actually be stored under:
    `suggested` itself if it is free, otherwise `suggested` suffixed `_1`,
    `_2`, ... up to the first free one -- `existing` is checked with `in`,
    so a set is cheapest, though a list or a dict's keys work too.

    This is the ONE place that picks a fresh name; Dataset.create_table_from_df
    refuses a collision outright rather than working around it, so every
    caller that wants a table to land under a name close to what it asked
    for, rather than fail, resolves it through this function first.
    format_table_name_changed_message (controller.py) is the matching
    wording half, for when the resolved name surprises the researcher.
    """
    if suggested not in existing:
        return suggested
    n = 1
    while f"{suggested}_{n}" in existing:
        n += 1
    return f"{suggested}_{n}"


def validate_new_table_name(name, existing) -> Optional[str]:
    """The researcher-facing message for a NewTableNameParameter field's
    current text, or ``None`` if it is fine to submit as is.

    Mirrors EXACTLY what ``models/dataset.py``'s ``create_table_from_df``
    refuses at store time -- ``if name in self._tables: raise
    ValueError(...)``, an EXACT, case-sensitive string membership test
    against every currently stored table name, with no case-folding and
    no partial match. ``existing`` is whatever the caller hands in (a
    list, tuple or set all work with ``in``) -- ``AppController`` builds
    it from ``Dataset.list_tables()``, the exact same names
    ``create_table_from_df`` checks against, so the red text here can
    never disagree with the backstop refusal a researcher would
    otherwise only discover after the run finished.

    ``name`` is compared STRIPPED, because ``collect_parameters`` strips
    a ``new_table_name`` field's text the same way before it ever reaches
    a run's parameters (and therefore before it ever reaches
    ``create_table_from_df``) -- so ``"segments "`` is flagged exactly
    when ``"segments"`` would be, matching what will actually be stored,
    never what the raw, unstripped widget text happens to spell.

    A blank or whitespace-only name is NOT flagged here -- ``None`` is
    returned instead. That is a DIFFERENT, already-existing rule
    (``collect_parameters``'s required-field check, surfaced through a
    ``QMessageBox`` when OK is clicked, not inline) and this function
    must not duplicate or race it with a second, differently-timed
    complaint about the same empty field.

    This is a courtesy, never a second authority: another run can still
    claim the name while this dialog stays open, so ``Dataset``'s own
    refusal at store time remains the real guarantee (``CLAUDE.md``,
    "Data ownership") -- this only lets the researcher hear about an
    ALREADY-taken name while they are still looking at the field, rather
    than only after the run finishes.
    """
    if name is None:
        return None
    stripped = name.strip()
    if stripped == "":
        return None
    if stripped in existing:
        return f'A table named "{stripped}" already exists.'
    return None
