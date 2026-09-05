"""
tests/test_media_canonicalise.py

Tests for canonicalise_path() and canonicalise_cell() (P1.8e-1), written
from the work item's spec and from the ruling that settled the
idempotence question for canonicalise_path (see media_address.py's
docstrings for the reasoning; not restated here).

Run with: python -m pytest tests/test_media_canonicalise.py
"""

import pathlib
import sys

# Add project root to Python path, matching the other test modules.
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from media.media_address import canonicalise_cell, canonicalise_path, parse
from models.table_schema import infer_type_tag


# ---------------------------------------------------------------------------
# The six values from the work item's Step 1, held here literally so every
# test below refers to the same fixed set.
# ---------------------------------------------------------------------------

VALUES = [
    "videos/clip.mp4",
    "C:\\vids\\face.jpg",
    "clip.mp4#f=1234",
    "face.jpg",
    "photos/img#2.png",
    "photos/a%23b.png",
]

# A value carrying a stream selector and a region, for the idempotence
# tests that need one beyond the six.
VALUE_WITH_STREAM_AND_REGION = "clip.mp4#v=1&t=1.500000-2.500000&r=0.100000,0.200000,0.300000,0.400000"


# ---------------------------------------------------------------------------
# Literal expected outputs for each of the six values, through each
# function. Written out explicitly rather than derived, per the work item.
# ---------------------------------------------------------------------------

def test_canonicalise_path_literal_outputs():
    expected = {
        "videos/clip.mp4": "videos/clip.mp4",
        "C:\\vids\\face.jpg": "C:/vids/face.jpg",
        "clip.mp4#f=1234": "clip.mp4%23f=1234",
        "face.jpg": "face.jpg",
        "photos/img#2.png": "photos/img%232.png",
        "photos/a%23b.png": "photos/a%2523b.png",
    }
    for value in VALUES:
        assert canonicalise_path(value) == expected[value], value


def test_canonicalise_cell_literal_outputs():
    expected = {
        "videos/clip.mp4": "videos/clip.mp4",
        "C:\\vids\\face.jpg": "C:/vids/face.jpg",
        "clip.mp4#f=1234": "clip.mp4#f=1234",
        "face.jpg": "face.jpg",
        "photos/img#2.png": "photos/img%232.png",
        "photos/a%23b.png": "photos/a%23b.png",
    }
    for value in VALUES:
        assert canonicalise_cell(value) == expected[value], value


# ---------------------------------------------------------------------------
# canonicalise_cell idempotence -- unchanged from the original spec.
# ---------------------------------------------------------------------------

def test_canonicalise_cell_is_idempotent_on_the_six_values():
    for value in VALUES:
        once = canonicalise_cell(value)
        twice = canonicalise_cell(once)
        assert once == twice, value


def test_canonicalise_cell_is_idempotent_on_a_value_with_stream_and_region():
    once = canonicalise_cell(VALUE_WITH_STREAM_AND_REGION)
    twice = canonicalise_cell(once)
    assert once == twice


# ---------------------------------------------------------------------------
# canonicalise_path -- NOT idempotence. Per the ruling: canonicalise_path
# is a one-way encoder, never idempotent, and must never be made so by
# detecting already-escaped input (nothing can distinguish a literal
# "%23" in a filename from an escaped "#"). These three tests replace the
# dropped "idempotence on all six" requirement.
# ---------------------------------------------------------------------------

def test_canonicalise_path_round_trips_to_the_original_normalised_path():
    # TEST 1 -- the real correctness property: the escaped form parses
    # back to exactly the path we started from, separator-normalised,
    # with nothing lost. Holds even for the three values that already
    # contain '#' or '%'.
    for value in VALUES:
        recovered = parse(canonicalise_path(value)).path
        expected = pathlib.PureWindowsPath(value).as_posix()
        assert recovered == expected, value


def test_canonicalise_path_output_is_a_stable_cell():
    # TEST 2 -- composition: canonicalise_path's output is already a
    # canonical cell, so pushing it through canonicalise_cell changes
    # nothing. P1.8e-2 relies on this: a folder-scanned path and a CSV
    # cell can land in the same column, and both must survive a
    # re-accept through canonicalise_cell unchanged.
    for value in VALUES:
        once = canonicalise_path(value)
        assert canonicalise_cell(once) == once, value


def test_canonicalise_path_is_deliberately_not_idempotent():
    # TEST 3 -- the one-way behaviour, stated as intended, not as a
    # defect. Each of these three values already contains a literal '#'
    # or '%'; calling canonicalise_path on its own output re-escapes the
    # '%' the first call introduced. This is proof canonicalise_path is
    # an encoder, not a normaliser -- calling it on its own output is a
    # caller error, not a case it should detect and undo. Every string
    # below was verified against a real run before being written here.
    doubled = {
        "clip.mp4#f=1234": "clip.mp4%2523f=1234",
        "photos/img#2.png": "photos/img%25232.png",
        "photos/a%23b.png": "photos/a%252523b.png",
    }
    for value, expected_twice in doubled.items():
        once = canonicalise_path(value)
        twice = canonicalise_path(once)
        assert twice == expected_twice, value
        assert twice != once, value


# ---------------------------------------------------------------------------
# Blank string.
# ---------------------------------------------------------------------------

def test_blank_string_returns_blank_from_both_functions():
    assert canonicalise_path("") == ""
    assert canonicalise_cell("") == ""


# ---------------------------------------------------------------------------
# canonicalise_cell's fallback disagreement with canonicalise_path -- the
# reason there are two functions.
# ---------------------------------------------------------------------------

def test_the_two_functions_disagree_on_a_value_with_a_fragment():
    cell_result = canonicalise_cell("clip.mp4#f=1234")
    path_result = canonicalise_path("clip.mp4#f=1234")
    # canonicalise_cell parses the fragment and keeps it.
    assert cell_result == "clip.mp4#f=1234"
    # canonicalise_path treats the same string as a literal path and
    # escapes the '#'.
    assert path_result == "clip.mp4%23f=1234"
    assert cell_result != path_result


# ---------------------------------------------------------------------------
# The '%23' ambiguity inherent to the escape scheme (decision 1). A cell
# already spelled with an escaped '#' parses successfully on the first
# try -- canonicalise_cell takes the PARSE branch, not the fallback, and
# returns it unchanged. The ambiguity surfaces one step later, when that
# canonical cell is parsed: a file genuinely named "a%23b.png" on disk is
# read downstream under the filename "a#b.png" instead. Pinned here with
# the literal expected path so P1.8e-2 cannot silently change it.
# ---------------------------------------------------------------------------

def test_canonicalise_cell_leaves_an_already_escaped_value_unchanged():
    value = "photos/a%23b.png"
    assert canonicalise_cell(value) == value


def test_the_percent23_ambiguity_surfaces_on_parse_not_on_canonicalise():
    canonical = canonicalise_cell("photos/a%23b.png")
    assert parse(canonical).path == "photos/a#b.png"


# ---------------------------------------------------------------------------
# parse(canonicalise_cell(x)) never raises.
# ---------------------------------------------------------------------------

def test_parse_of_canonicalise_cell_output_never_raises():
    for value in VALUES:
        parse(canonicalise_cell(value))  # must not raise


# ---------------------------------------------------------------------------
# The defect this item exists to make fixable: after canonicalise_cell,
# "photos/img#2.png" is tagged "media_path" by table_schema's public
# inference entry point, infer_type_tag, instead of "text". Goes through
# the public seam, not the private _looks_like_media_path gate.
# table_schema is read-only in this item -- nothing about it changes here.
# ---------------------------------------------------------------------------

def test_canonicalised_cell_is_recognised_as_media_by_table_schema():
    before = infer_type_tag(pd.Series(["photos/img#2.png"]))
    after = infer_type_tag(pd.Series([canonicalise_cell("photos/img#2.png")]))
    assert before == "text"
    assert after == "media_path"
