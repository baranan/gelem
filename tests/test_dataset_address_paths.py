"""
tests/test_dataset_address_paths.py

Tests for P0.2c (address-aware project paths): Dataset.save() / load() must
rewrite the *path portion* of a media address -- relative-if-inside on save,
absolute on load -- while leaving the address fragment (#f=, #t=, #r=, the
stream selector) untouched, and must spell the path with forward slashes
(docs/media_architecture.md decision 9).

Written from the P0.2c work-item specification and section 7's "save
project -> load project" invariant, not from the implementation. Each test
is designed to fail against a models/dataset.py that still does string
surgery on the whole cell value (_rel_if_inside / _abs_against).

Kept separate from tests/test_dataset.py, which executes its own tests at
import time as well as under pytest, so everything in that file runs twice.
Follows the precedent of tests/test_dataset_access_paths.py.

Run with:
    python -m pytest tests/test_dataset_address_paths.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from models.dataset import Dataset
from media.media_address import (
    Region,
    StreamSelector,
    from_path,
    parse,
)
from media.media_address import format as format_address


# ---------------------------------------------------------------------------
# Fixture: a frames table holding one media address of every grammar form
# from section 3.2 -- half inside the project folder, half outside it.
# ---------------------------------------------------------------------------

def _addr(path: str, **fragment) -> str:
    """The canonical stored string for `path` plus a fragment, built through
    the media_address module (decision 1: never by joining strings)."""
    return format_address(dataclasses.replace(from_path(path), **fragment))


# form name -> (is_inside_project, fragment kwargs for dataclasses.replace)
_FORMS = [
    ("bare_path",            True,  {}),
    ("frame_ordinal",        False, {"frame": 100}),
    ("time_point",           True,  {"time_us": 1_500_000}),
    ("time_range",           False, {"time_range_us": (1_000_000, 4_000_000)}),
    ("region",               True,  {"region": Region(x=0, y=0, w=500_000, h=500_000)}),
    ("full_with_stream",     False, {
        "stream": StreamSelector(kind="v", index=1),
        "frame": 50,
        "region": Region(x=100_000, y=100_000, w=400_000, h=400_000),
    }),
]


def _make_dataset(tmp_path: Path):
    """Build a Dataset whose `frames` table has one `full_path` address per
    grammar form. Returns (dataset, project_dir, before_strings).

    `full_path` is always treated as a media column by save()/load(), so no
    registry is needed. A second column, `note`, holds a copy of each
    address string but is never registered -- property 7 checks it is left
    completely alone.
    """
    project_dir = tmp_path / "proj"
    inside_dir = project_dir / "media"
    outside_dir = tmp_path / "external"
    inside_dir.mkdir(parents=True)
    outside_dir.mkdir(parents=True)

    before_strings = []
    file_names = []
    for index, (name, is_inside, fragment) in enumerate(_FORMS):
        base_dir = inside_dir if is_inside else outside_dir
        media_file = base_dir / f"{name}.mp4"
        media_file.touch()
        # str(media_file) is OS-native (backslashes on Windows); from_path()
        # inside _addr() normalises it to forward slashes.
        before_strings.append(_addr(str(media_file), **fragment))
        file_names.append(f"{name}.mp4")

    frames = pd.DataFrame({
        "row_id":    [f"{i + 1:06d}" for i in range(len(_FORMS))],
        "full_path": before_strings,
        "file_name": file_names,
        "note":      list(before_strings),  # same strings, unregistered column
    })

    ds = Dataset()
    ds._tables["frames"] = frames
    return ds, project_dir, before_strings


def _save_load(tmp_path: Path):
    """Run one save/load cycle and return (loaded_dataset, project_dir,
    before_strings, after_strings)."""
    ds, project_dir, before = _make_dataset(tmp_path)
    ds.save(project_dir)

    ds2 = Dataset()
    ds2.load(project_dir)
    after = list(ds2.get_table("frames")["full_path"])
    return ds2, project_dir, before, after


# ---------------------------------------------------------------------------
# Section 7 invariant, testable half: four properties over the round trip.
# ---------------------------------------------------------------------------

def test_p02c_every_stored_cell_parses_without_raising(tmp_path):
    _ds, project_dir, _before, _after = _save_load(tmp_path)
    stored = list(pd.read_parquet(project_dir / "frames.parquet")["full_path"])
    for cell in stored:
        parse(cell)  # must not raise


def test_p02c_parse_of_loaded_equals_parse_of_original(tmp_path):
    # MediaAddress equality is canonical-string equality, so this pins the
    # fragment exactly -- a dropped or mangled #f= / #t= / #r= fails here.
    _ds, _project_dir, before, after = _save_load(tmp_path)
    assert len(before) == len(after)
    for original, loaded in zip(before, after):
        assert parse(loaded) == parse(original), (
            f"address changed across save/load: {original!r} -> {loaded!r}"
        )


def test_p02c_path_portion_names_the_same_file_after_the_round_trip(tmp_path):
    _ds, _project_dir, before, after = _save_load(tmp_path)
    for original, loaded in zip(before, after):
        original_file = Path(parse(original).path).resolve()
        loaded_file = Path(parse(loaded).path).resolve()
        assert original_file == loaded_file, (
            f"path portion resolves to a different file: "
            f"{original_file} -> {loaded_file}"
        )


def test_p02c_save_load_save_is_byte_identical_in_the_parquet(tmp_path):
    ds2, project_dir, _before, _after = _save_load(tmp_path)
    first_save = pd.read_parquet(project_dir / "frames.parquet")["full_path"].tolist()

    # Re-save the loaded project to the SAME project folder: the stored
    # cells must come out byte-for-byte identical. (Saving to a different
    # folder legitimately changes which paths are in-project.)
    ds2.save(project_dir)
    second_save = pd.read_parquet(project_dir / "frames.parquet")["full_path"].tolist()

    assert first_save == second_save, (
        f"a second save produced different cell values:\n"
        f"  first:  {first_save}\n"
        f"  second: {second_save}"
    )


# ---------------------------------------------------------------------------
# Storage form: relative-if-inside, absolute-if-outside, forward slashes.
# ---------------------------------------------------------------------------

def test_p02c_in_project_cells_stored_relative_external_cells_stored_absolute(tmp_path):
    ds, project_dir, _before = _make_dataset(tmp_path)
    ds.save(project_dir)
    stored = list(pd.read_parquet(project_dir / "frames.parquet")["full_path"])

    for (name, is_inside, _fragment), cell in zip(_FORMS, stored):
        path_portion = parse(cell).path
        if is_inside:
            assert not _looks_absolute(path_portion), (
                f"{name}: in-project cell should be stored relative; got {cell!r}"
            )
        else:
            assert _looks_absolute(path_portion), (
                f"{name}: external cell should be stored absolute; got {cell!r}"
            )


def test_p02c_stored_and_loaded_forms_use_forward_slashes(tmp_path):
    ds, project_dir, _before = _make_dataset(tmp_path)
    ds.save(project_dir)
    stored = list(pd.read_parquet(project_dir / "frames.parquet")["full_path"])
    assert all("\\" not in cell for cell in stored), (
        f"stored cells must use forward slashes (decision 9); got {stored}"
    )

    ds2 = Dataset()
    ds2.load(project_dir)
    loaded = list(ds2.get_table("frames")["full_path"])
    assert all("\\" not in cell for cell in loaded), (
        f"loaded cells must use forward slashes (decision 9); got {loaded}"
    )


def _looks_absolute(path_portion: str) -> bool:
    """A POSIX root, a UNC path, or a drive-letter path."""
    if path_portion.startswith("/"):
        return True
    if len(path_portion) >= 3 and path_portion[1] == ":" and path_portion[2] == "/":
        return True
    return False


# ---------------------------------------------------------------------------
# An untagged column that happens to hold an address-shaped string.
# ---------------------------------------------------------------------------

def test_p02c_a_non_media_column_holding_an_address_string_is_left_alone(tmp_path):
    ds, project_dir, before = _make_dataset(tmp_path)
    ds.save(project_dir)

    stored_note = list(pd.read_parquet(project_dir / "frames.parquet")["note"])
    assert stored_note == before, (
        f"the unregistered `note` column must be stored verbatim; got {stored_note}"
    )

    ds2 = Dataset()
    ds2.load(project_dir)
    loaded_note = list(ds2.get_table("frames")["note"])
    assert loaded_note == before, (
        f"the unregistered `note` column must load verbatim; got {loaded_note}"
    )


# ---------------------------------------------------------------------------
# A cell that will not parse: stored verbatim, loaded verbatim, counted.
# ---------------------------------------------------------------------------

def _last_entry(ds: Dataset, action: str) -> dict:
    for entry in reversed(ds.provenance.to_list()):
        if entry["action"] == action:
            return entry
    raise AssertionError(f"no {action!r} entry in provenance")


def test_p02c_an_unparseable_cell_round_trips_verbatim_and_is_counted(tmp_path):
    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    # A reversed range is refused at parse time (decision 11), so this is a
    # cell save()/load() cannot rewrite. It must be stored and loaded
    # exactly as written, and reported in the provenance entry -- not
    # silently swallowed.
    bad_cell = "clips/trial.mp4#t=5.000000-2.000000"
    good_cell = _addr("clips/trial.mp4", frame=7)

    ds = Dataset()
    ds._tables["frames"] = pd.DataFrame({
        "row_id":    ["000001", "000002"],
        "full_path": [good_cell, bad_cell],
        "file_name": ["trial.mp4", "trial.mp4"],
    })
    ds.save(project_dir)

    save_entry = _last_entry(ds, "save")
    assert save_entry["params"].get("unparseable_media_cells") == 1, (
        f"save should report exactly one unparseable media cell; got {save_entry}"
    )

    stored = list(pd.read_parquet(project_dir / "frames.parquet")["full_path"])
    assert bad_cell in stored, f"the bad cell must be stored verbatim; got {stored}"

    ds2 = Dataset()
    ds2.load(project_dir)
    loaded = list(ds2.get_table("frames")["full_path"])
    assert bad_cell in loaded, f"the bad cell must load verbatim; got {loaded}"

    load_entry = _last_entry(ds2, "load")
    assert load_entry["params"].get("unparseable_media_cells") == 1, (
        f"load should report exactly one unparseable media cell; got {load_entry}"
    )
