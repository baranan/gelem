"""
tests/test_accept_canonicalisation.py

Tests for P1.8e-2b-1: Dataset._prepare_table canonicalises a media-address
column's cells at accept time, but only for a column whose ColumnSpec is
being NEWLY inferred (not one the authoritative schema already names) and
only when the result actually looks like media -- never for an ordinary
text column that happens to have a few cells ending in a media extension.

Written from the work item's specification, not from the implementation.
The defect this closes: a column not named 'full_path' (which is always
force-hinted 'media_path' regardless of its values) gets no such
protection today -- one cell with an unescaped '#' fails
media_address.parse(), so infer_type_tag demotes the WHOLE column to
"text" and every row in it loses its thumbnail.

apply_row_updates' value path, save(), the import sites (load_folder,
load_csv_as_primary), and load()'s own code are none of them EDITED by
this item -- see tests/test_import_canonicalisation.py for the import
sites' own tests (run separately; this file does not re-invoke it).
load()'s BEHAVIOUR is not immune, though: it calls the same
_prepare_table this item changed, and a table absent from schemas.json
(or one whose empty-frame cast to its saved schema fails) gets every
column treated as new, so this item's canonicalisation does run during
load for such a table -- see the comment above the
_canonicalise_new_media_columns call site in _prepare_table.

Run with: python -m pytest tests/test_accept_canonicalisation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from models.dataset import Dataset, _cell_could_be_media_path
from models.table_schema import ColumnHint


# ---------------------------------------------------------------------------
# 1. The Step 1(a) scenario: a hash-named value demotes a column to "text"
#    today: canonicalising it at accept keeps it "media_path".
# ---------------------------------------------------------------------------

def test_hash_bearing_media_column_stays_media_path_and_is_canonical():
    ds = Dataset()
    df = pd.DataFrame({"clip": ["videos/a.png", "videos/b#2.png"]})
    ds.create_table_from_df("t", df)

    assert ds.schema_for("t").spec_for("clip").type_tag == "media_path"

    cells = ds.get_table("t")["clip"].tolist()
    # Literal expected canonical cell: forward-slash path (already was),
    # with the literal '#' escaped to '%23' by canonicalise_cell.
    assert cells == ["videos/a.png", "videos/b%232.png"]


# ---------------------------------------------------------------------------
# 2. The discard rule: a bare (no-separator) name ending in ".png" is not
#    a media path (media_architecture's rule -- no directory, no fragment,
#    nothing for the resolver to open), so the trial tag stays "text" and
#    the column must be left byte-for-byte unchanged, hash and all.
# ---------------------------------------------------------------------------

def test_bare_png_name_without_separator_stays_text_and_unchanged():
    ds = Dataset()
    df = pd.DataFrame({"thumb_name": ["a.png", "b#2.png"]})
    ds.create_table_from_df("t", df)

    assert ds.schema_for("t").spec_for("thumb_name").type_tag == "text"
    assert ds.get_table("t")["thumb_name"].tolist() == ["a.png", "b#2.png"]


# ---------------------------------------------------------------------------
# 3. Ordinary prose is unaffected.
# ---------------------------------------------------------------------------

def test_ordinary_prose_column_stays_text_and_unchanged():
    ds = Dataset()
    values = ["high arousal", "the participant reported feeling calm"]
    df = pd.DataFrame({"notes": values})
    ds.create_table_from_df("t", df)

    assert ds.schema_for("t").spec_for("notes").type_tag == "text"
    assert ds.get_table("t")["notes"].tolist() == values


# ---------------------------------------------------------------------------
# 4. IDENTITY -- the pin that matters most. Checked by object identity,
#    not value, directly against _accept_table/_prepare_table (the
#    private seam this item's contract is about), because create_table_
#    from_df always copies its input itself before this seam ever sees it.
# ---------------------------------------------------------------------------

def test_prepare_table_returns_the_same_object_when_nothing_changed():
    ds = Dataset()
    df = pd.DataFrame({
        "row_id": ["1", "2"],
        "clip":   ["videos/a.png", "videos/b.png"],  # already canonical
    })
    ds._accept_table("t", df, source="test")

    assert ds.read_only_view("t") is df, (
        "a frame whose media cells were already canonical must be stored "
        "as the SAME object -- apply_row_updates relies on this to keep "
        "its row-id index valid across ticks"
    )


def test_prepare_table_returns_a_new_object_when_a_cell_changed():
    ds = Dataset()
    df = pd.DataFrame({
        "row_id": ["1", "2"],
        "clip":   ["videos/a.png", "videos/b#2.png"],  # needs escaping
    })
    ds._accept_table("t", df, source="test")

    assert ds.read_only_view("t") is not df, (
        "a frame whose media cells needed canonicalising must be stored "
        "as a NEW object, leaving the caller's frame untouched"
    )


# ---------------------------------------------------------------------------
# 5. A column the authoritative schema already names is NOT a candidate --
#    re-canonicalising an EXISTING media column's cells on every accept is
#    P1.8e-2b-2's job, not this one's. This is not a defect; it is the
#    scope line this item draws.
# ---------------------------------------------------------------------------

def test_existing_media_column_is_not_recanonicalised_on_re_accept():
    ds = Dataset()
    df1 = pd.DataFrame({
        "row_id": ["1", "2"],
        "clip":   ["videos/a.png", "videos/b.png"],
    })
    ds._accept_table("t", df1, source="test")
    assert ds.schema_for("t").spec_for("clip").type_tag == "media_path"

    # Re-accept the SAME table: 'clip' is now named by the stored,
    # authoritative schema, so new_columns excludes it and this item's
    # canonicalisation never runs on it -- P1.8e-2b-2's job, not a defect.
    df2 = pd.DataFrame({
        "row_id": ["1", "2"],
        "clip":   ["videos/c#3.png", "videos/d.png"],
    })
    ds._accept_table("t", df2, source="test")

    assert ds.get_table("t")["clip"].tolist() == [
        "videos/c#3.png", "videos/d.png",
    ]


# ---------------------------------------------------------------------------
# 6. A blank / all-null column accepts unchanged.
# ---------------------------------------------------------------------------

def test_all_null_column_accepts_unchanged():
    ds = Dataset()
    df = pd.DataFrame({"clip": [None, None]})
    ds.create_table_from_df("t", df)

    assert ds.schema_for("t").spec_for("clip").type_tag == "text"
    assert ds.get_table("t")["clip"].isna().all()


# ---------------------------------------------------------------------------
# 7. A stray "media_path" hint on a column that is really ordinary text must
#    not rewrite that column's values -- only load()'s project-wide hint
#    union could produce such a stray hint in practice, but the contract is
#    tested directly here at the seam that would carry the damage: a wrong
#    hint may cost a wrong tag, never a rewritten value.
# ---------------------------------------------------------------------------

def test_media_path_hint_on_non_media_text_leaves_values_unchanged():
    ds = Dataset()
    values = ["high arousal", "the participant reported feeling calm"]
    df = pd.DataFrame({"row_id": ["1", "2"], "notes": values})

    ds._accept_table(
        "t", df,
        hints={"notes": ColumnHint(type_tag="media_path")},
        source="test",
    )

    assert ds.get_table("t")["notes"].tolist() == values


# ---------------------------------------------------------------------------
# 8. The cheap pre-gate: a column of ordinary prose (no cell's extension
#    looks like media) must be screened out by string operations alone --
#    canonicalise_cell must never run on it at all. Counted directly,
#    rather than inferred from the stored values staying unchanged, so a
#    pre-gate that let the column through and merely happened to leave it
#    unchanged would still fail this test.
#
#    This departs from CLAUDE.md's Testing rule "Test at component seams,
#    not internals" -- asserting a call count on canonicalise_cell pins an
#    internal, not a public seam. Deliberate: the pre-gate's entire reason
#    to exist is to avoid doing that work at all, and "work not done" has
#    no outward-visible seam to assert on -- the stored value and tag look
#    identical whether the pre-gate skipped the column or the full trial
#    ran and happened to change nothing. Test 3 above already covers the
#    public behaviour (value and tag unchanged); this test is the only way
#    to pin the cost claim the pre-gate exists for.
# ---------------------------------------------------------------------------

def test_prose_column_is_screened_out_before_canonicalise_cell_runs(monkeypatch):
    import models.dataset as dataset_module

    calls = []
    original = dataset_module.canonicalise_cell

    def _counting_canonicalise_cell(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(
        dataset_module, "canonicalise_cell", _counting_canonicalise_cell
    )

    ds = Dataset()
    values = ["high arousal", "the participant reported feeling calm"]
    df = pd.DataFrame({"notes": values})
    ds.create_table_from_df("t", df)

    assert calls == [], (
        "the pre-gate must screen out a column with no media-extension "
        f"cell before canonicalise_cell ever runs on it; it was called "
        f"with {calls}"
    )
    assert ds.get_table("t")["notes"].tolist() == values


# ---------------------------------------------------------------------------
# 9. Regression: Unicode case-folding can change a string's LENGTH (Turkish
#     dotted capital 'I' lowercases to the two-codepoint 'i(dot above)'), so
#     computing the '#' index on one version of a cell (original or lowered)
#     and then slicing a DIFFERENT version with that index can land one
#     character short of where '#' actually falls. Found by code review,
#     reproduced directly before the fix: this exact cell made the pre-gate
#     wrongly return False. The fix now lives in the shared
#     media.extensions.looks_like_media_extension (P1.8e-2b-1 part 2a), so
#     this pins both that function and the dataset-level pre-gate built on
#     it.
# ---------------------------------------------------------------------------

def test_pregate_handles_a_hash_after_a_length_changing_lowercase_fold():
    from media.extensions import looks_like_media_extension

    # The Turkish dotted capital I is one codepoint; .lower() turns it into
    # two ('i' + combining dot above), so len(cell) != len(cell.lower()).
    cell = "İclip.mp4#f=1234"
    assert len(cell) != len(cell.lower())
    assert looks_like_media_extension(cell) is not None
    assert _cell_could_be_media_path(cell)


# ---------------------------------------------------------------------------
# 10. Regression: table_schema._looks_like_media_path's Gate 1 must check
#     ONLY the whole path portion, never the pre-gate's two-spelling
#     (whole-or-before-'#') fallback. Found by code review, reproduced
#     directly before the fix: sharing looks_like_media_extension (built
#     for a RAW, unparsed cell, where a '#' might genuinely introduce a
#     fragment) with Gate 1 (which runs on an ALREADY-PARSED path_portion,
#     where any surviving '#' is a literal character in the real filename,
#     e.g. an escaped '%23' unescaped back) let a genuinely non-media file
#     match a media extension on the substring before that literal '#'.
# ---------------------------------------------------------------------------

def test_escaped_hash_before_a_non_media_extension_does_not_look_like_media():
    ds = Dataset()
    values = ["photos/clip.mp4%23notes.txt", "videos/take.mp4%23readme.txt"]
    df = pd.DataFrame({"notes": values})
    ds.create_table_from_df("t", df)

    assert ds.schema_for("t").spec_for("notes").type_tag == "text"
    assert ds.get_table("t")["notes"].tolist() == values
