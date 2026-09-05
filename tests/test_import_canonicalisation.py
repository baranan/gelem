"""
tests/test_import_canonicalisation.py

Tests for P1.8e-2a: Dataset.load_folder() and Dataset.load_csv_as_primary()
canonicalise the full_path cells they write, using media_address's
canonicalise_path (load_folder, real filesystem paths) and canonicalise_cell
(load_csv_as_primary, values a human wrote that may already carry a
fragment).

Written from the work item's specification, not from the implementation.
The defect this closes: a filename containing a literal '#' (a real, if
unusual, character in a Windows or Mac filename) was stored raw, so
media_address.parse() raised on that cell -- it could never be keyed for
a demand-driven artifact request, so its tile stayed a permanent grey
placeholder. Canonicalising escapes the '#' so the cell parses. Note the
schema tag on full_path is unaffected either way: it comes from the
forced hint in Dataset._media_column_hints(), not from value inference.

_prepare_table / _accept_table / merge_csv / confirm_merge are untouched
by this item -- accept-time canonicalisation is P1.8e-2b.

Run with: python -m pytest tests/test_import_canonicalisation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from models.dataset import Dataset
from media.media_address import canonicalise_cell, parse
from models.table_schema import infer_type_tag


# ---------------------------------------------------------------------------
# load_folder
# ---------------------------------------------------------------------------

def _make_two_file_folder(tmp_path: Path) -> Path:
    """A folder holding one normally-named image and one whose name
    contains a literal '#' -- the scenario from the work item's Step 1."""
    (tmp_path / "a.png").write_bytes(b"fake")
    (tmp_path / "b#2.png").write_bytes(b"fake")
    return tmp_path


def test_load_folder_hash_named_file_is_tagged_media_path_and_canonical(tmp_path):
    ds = Dataset()
    folder = _make_two_file_folder(tmp_path)
    ds.load_folder(folder)

    df = ds.get_table("frames")
    schema = ds.schema_for("frames")
    assert schema.spec_for("full_path").type_tag == "media_path"

    by_name = {row.file_name: row.full_path for row in df.itertuples()}

    # Literal expected cell for the '#' file: forward slashes, and the
    # literal '#' escaped to '%23' by canonicalise_path.
    expected_hash_cell = folder.as_posix() + "/b%232.png"
    assert by_name["b#2.png"] == expected_hash_cell

    expected_plain_cell = folder.as_posix() + "/a.png"
    assert by_name["a.png"] == expected_plain_cell

    # Both stored cells must now parse as a media address -- this is the
    # actual mechanism of the defect, independent of the schema tag.
    for cell in df["full_path"]:
        parse(cell)  # must not raise


def test_load_folder_stores_forward_slashes_for_windows_style_path(tmp_path):
    ds = Dataset()
    folder = _make_two_file_folder(tmp_path)
    ds.load_folder(folder)

    df = ds.get_table("frames")
    for cell in df["full_path"]:
        assert "\\" not in cell, cell


def test_load_folder_leaves_file_name_as_the_os_native_basename(tmp_path):
    ds = Dataset()
    folder = _make_two_file_folder(tmp_path)
    ds.load_folder(folder)

    df = ds.get_table("frames")
    names = set(df["file_name"])
    assert names == {"a.png", "b#2.png"}


def test_load_folder_infer_type_tag_is_self_describing_after_canonicalisation(tmp_path):
    """The forced hint in _media_column_hints() still sets the schema tag
    (that does not change either way in this item), but the values
    themselves must also become self-describing: infer_type_tag, called
    directly and without any hint, must recognise the canonicalised
    column as media on its own. Called on the raw, pre-canonicalisation
    cells it does not -- the '#' in the raw cell fails to parse, so that
    value does not look like a media path and the whole column reads as
    "text"."""
    ds = Dataset()
    folder = _make_two_file_folder(tmp_path)
    ds.load_folder(folder)

    df = ds.get_table("frames")
    stored_cells = df["full_path"].tolist()
    assert infer_type_tag(pd.Series(stored_cells)) == "media_path"

    raw_cells = [str(folder / "a.png"), str(folder / "b#2.png")]
    assert infer_type_tag(pd.Series(raw_cells)) == "text"


def test_load_folder_empty_folder_still_accepts(tmp_path):
    ds = Dataset()
    ds.load_folder(tmp_path)  # must not raise

    df = ds.get_table("frames")
    assert len(df) == 0
    assert set(Dataset.FRAMES_REQUIRED_COLUMNS).issubset(df.columns)


# ---------------------------------------------------------------------------
# load_csv_as_primary
# ---------------------------------------------------------------------------

def test_load_csv_as_primary_fragment_survives(tmp_path):
    """A CSV cell that already carries a fragment ('#f=1234') came from a
    human who wrote it deliberately. canonicalise_cell (not
    canonicalise_path) must be used here, so the fragment is parsed and
    kept, not escaped away."""
    csv_path = tmp_path / "data.csv"
    csv_df = pd.DataFrame({
        "media_col":  ["clip.mp4#f=1234", "videos/clip2.mp4"],
        "some_score": [1, 2],
    })
    csv_df.to_csv(csv_path, index=False)

    ds = Dataset()
    ds.load_csv_as_primary(csv_path, image_column="media_col")

    df = ds.get_table("frames")
    cells = df["full_path"].tolist()
    assert cells == ["clip.mp4#f=1234", "videos/clip2.mp4"]


# ---------------------------------------------------------------------------
# Composition property: the pin P1.8e-2b must not break.
# ---------------------------------------------------------------------------

def test_stored_full_path_cells_are_already_canonical(tmp_path):
    """For every full_path cell this item stores -- from either import
    site -- canonicalise_cell(cell) == cell. This is the property
    P1.8e-2b's accept-time canonicalisation must preserve: re-running an
    already-canonical cell through canonicalise_cell must be a no-op."""
    ds_folder = Dataset()
    folder_dir = tmp_path / "folder"
    folder_dir.mkdir()
    folder = _make_two_file_folder(folder_dir)
    ds_folder.load_folder(folder)
    for cell in ds_folder.get_table("frames")["full_path"]:
        assert canonicalise_cell(cell) == cell, cell

    csv_path = tmp_path / "data.csv"
    csv_df = pd.DataFrame({"media_col": ["clip.mp4#f=1234", "videos/clip2.mp4"]})
    csv_df.to_csv(csv_path, index=False)
    ds_csv = Dataset()
    ds_csv.load_csv_as_primary(csv_path, image_column="media_col")
    for cell in ds_csv.get_table("frames")["full_path"]:
        assert canonicalise_cell(cell) == cell, cell
