"""
tests/test_media_address.py

One test per decision in docs/media_architecture.md section 3.6 ("Address
semantics -- settled"), written from that section's text rather than from
media/media_address.py's implementation. A test name says which decision
it guards.

Two tests generate fixtures on demand rather than reading committed media:
- the lossless known-frame video required by section 7 (small, no
  participants, safe to generate in the repo's own test run);
- an attempt at a video-stream edit list, for decision 12 (needs a real
  recording from GELEM_FIXTURES and skips cleanly without one).
Both skip with a clear message if ffmpeg/ffprobe are not on PATH.

Run with: python -m pytest tests/test_media_address.py
"""

import dataclasses
import os
import pathlib
import shutil
import subprocess
import sys

# Add project root to Python path, matching the other test modules.
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

from media.media_address import (
    MediaAddress,
    MediaAddressError,
    Region,
    StreamSelector,
    absolutise,
    canonical_key,
    format,
    frames_in_range,
    from_path,
    parse,
    region_to_pixels,
    relativise,
    select_frame,
)


# ---------------------------------------------------------------------------
# Decision 1 -- escaping.
# ---------------------------------------------------------------------------

HOSTILE_PATHS = [
    "plain/path with spaces.mp4",
    "C:/Users/name #1/100% clip.mp4",
    "//server/share/name,with-dashes&stuff#2.mp4",
    "videos/a-b,c&d#e%f.mov",
    "C:/Users/x/only%percent.mp4",
    "C:/Users/x/only#hash.mp4",
    "videos/%25 already looks escaped.mp4",
]


def test_decision1_escaping_round_trips_hostile_paths_without_a_fragment():
    for original in HOSTILE_PATHS:
        addr = from_path(original)
        recovered = parse(format(addr)).path
        assert recovered == original, f"round trip failed for {original!r}"


def test_decision1_escaping_round_trips_hostile_paths_with_a_fragment():
    for original in HOSTILE_PATHS:
        addr = dataclasses.replace(from_path(original), time_us=1_500_000)
        recovered = parse(format(addr)).path
        assert recovered == original, f"round trip with a fragment failed for {original!r}"


def test_decision1_from_path_normalises_windows_separators():
    # Decision 1's rationale is that backslash escaping collides with
    # Windows separators, so separators are handled by normalisation
    # (decision 9: paths use forward slashes), not by escaping. A
    # backslash spelling is therefore not expected to survive literally --
    # what must hold is that from_path()'s own output is stable under a
    # further round trip.
    drive_letter = from_path(r"C:\Users\name\clip.mp4")
    assert drive_letter.path == "C:/Users/name/clip.mp4"
    assert parse(format(drive_letter)).path == drive_letter.path

    unc = from_path(r"\\server\share\clip.mp4")
    assert unc.path == "//server/share/clip.mp4"
    assert parse(format(unc)).path == unc.path


# ---------------------------------------------------------------------------
# Decisions 2, 3, 4 -- time point, range endpoints, representative frame.
# ---------------------------------------------------------------------------

def _constant_25fps(n_frames):
    """n_frames frame times, 40_000 microseconds (25 fps) apart, from 0."""
    return [i * 40_000 for i in range(n_frames)]


IRREGULAR_FRAME_TIMES = [
    0, 30_000, 55_000, 90_000, 200_000,
    210_000, 500_000, 900_000, 1_000_000, 1_050_000,
]


def test_decision2_time_point_selects_the_frame_whose_interval_contains_it():
    frame_times = _constant_25fps(10)
    # 0.039999s is still inside frame 0's interval [0, 0.04); 0.04s exactly
    # is the start of frame 1's interval. They must land on different
    # frames -- the whole reason interval-containment was chosen over
    # nearest-frame or floor/ceiling rules.
    just_under = parse("clip.mp4#t=0.039999")
    exactly_at = parse("clip.mp4#t=0.040000")
    assert select_frame(just_under, frame_times) == 0
    assert select_frame(exactly_at, frame_times) == 1


def test_decision2_time_point_against_an_irregular_sequence():
    # 0.06s (60_000 us) falls inside frame 2's interval [55_000, 90_000).
    addr = parse("clip.mp4#t=0.060000")
    assert select_frame(addr, IRREGULAR_FRAME_TIMES) == 2


def test_decision3_adjacent_ranges_tile_with_no_overlap_and_no_gap():
    frame_times = _constant_25fps(100)  # 4 seconds at 25 fps
    first_half = parse("clip.mp4#t=0.000000-2.000000")
    second_half = parse("clip.mp4#t=2.000000-4.000000")
    whole = parse("clip.mp4#t=0.000000-4.000000")

    first_members = frames_in_range(first_half, frame_times)
    second_members = frames_in_range(second_half, frame_times)
    whole_members = frames_in_range(whole, frame_times)

    assert set(first_members).isdisjoint(second_members)
    assert sorted(first_members + second_members) == whole_members
    # The frame at exactly 2.0s belongs to the range that starts there,
    # not the one that ends there (half-open, decision 3).
    frame_at_two_seconds = 50
    assert frame_at_two_seconds not in first_members
    assert frame_at_two_seconds in second_members


def test_decision3_a_two_second_range_at_25fps_holds_exactly_fifty_frames():
    frame_times = _constant_25fps(100)
    a_range = parse("clip.mp4#t=0.000000-2.000000")
    assert len(frames_in_range(a_range, frame_times)) == 50


def test_decision4_both_policies_choose_a_frame_inside_the_range():
    frame_times = _constant_25fps(100)
    a_range = parse("clip.mp4#t=2.000000-4.000000")
    members = set(frames_in_range(a_range, frame_times))
    assert select_frame(a_range, frame_times, policy="first") in members
    assert select_frame(a_range, frame_times, policy="midpoint") in members
    # 'first' is specifically the frame at the range's own start.
    assert select_frame(a_range, frame_times, policy="first") == 50


def test_decision4_a_bare_path_behaves_as_the_whole_file_range():
    frame_times = _constant_25fps(100)
    bare = from_path("clip.mp4")
    assert frames_in_range(bare, frame_times) == list(range(100))
    assert select_frame(bare, frame_times, policy="first") == 0
    assert select_frame(bare, frame_times, policy="midpoint") in range(100)


def test_decision4_both_policies_always_choose_a_member_of_frames_in_range():
    # The invariant, pinned directly rather than checking the two policies
    # separately: for any range holding at least one frame, whichever
    # frame either policy picks must be one that frames_in_range() itself
    # reports for that same address -- never a frame from outside the
    # range, however plausible ("what was on screen a moment earlier")
    # that frame might otherwise look. This is what CLAUDE.md section 7's
    # "a segment's thumbnail comes from inside that segment's own time
    # range" actually requires at this layer.
    irregular = [0, 40_000, 90_000, 130_000, 500_000, 900_000, 1_000_000]

    cases = [
        # A range starting exactly on a frame boundary (the case the
        # original, since-corrected definition happened to get right).
        (_constant_25fps(100), parse("clip.mp4#t=2.000000-4.000000")),
        # A range starting partway through a frame's own display interval
        # -- the case the original definition got wrong, per this
        # amendment: the frame at 0us is still "on screen" at 10_000us,
        # but is not itself in [10_000, 95_000).
        (irregular, parse("clip.mp4#t=0.010000-0.095000")),
        # A range narrower than the frame interval surrounding it (the
        # 40_000-to-90_000 and 90_000-to-130_000 gaps are both wider than
        # this 10_000us range), which still captures exactly one frame.
        (irregular, parse("clip.mp4#t=0.085000-0.095000")),
        # The whole-file range from a bare path (decision 4).
        (_constant_25fps(100), from_path("clip.mp4")),
    ]

    for frame_times, addr in cases:
        members = frames_in_range(addr, frame_times)
        assert members, f"test case is not actually non-empty: {format(addr)}"
        for policy in ("first", "midpoint"):
            chosen = select_frame(addr, frame_times, policy=policy)
            assert chosen in members, (
                f"{policy!r} chose frame {chosen}, not a member of "
                f"{members} for {format(addr)}"
            )


def test_decision11_a_range_with_no_frames_raises_at_resolve():
    # Distinct from the zero-length range decision 11 already refuses at
    # parse time (start == end): this range is well-formed (start < end)
    # but falls entirely between two frames, so it legitimately captures
    # none. Never fall back to a frame outside the range, and never
    # return nothing silently.
    frame_times = [0, 40_000, 200_000]
    addr = parse("clip.mp4#t=0.041000-0.043000")
    assert frames_in_range(addr, frame_times) == []
    with pytest.raises(MediaAddressError, match="no frames"):
        select_frame(addr, frame_times, policy="first")
    with pytest.raises(MediaAddressError, match="no frames"):
        select_frame(addr, frame_times, policy="midpoint")


# ---------------------------------------------------------------------------
# frames_in_range() shares select_frame()'s resolve-time refusals
# (decision 11), and a single-frame stream is not spuriously unqueryable.
# ---------------------------------------------------------------------------

def test_decision11_frames_in_range_also_refuses_a_start_beyond_the_stream():
    frame_times = _constant_25fps(10)  # runs to 0.36s
    far_beyond = parse("clip.mp4#t=10.000000-11.000000")
    with pytest.raises(MediaAddressError):
        select_frame(far_beyond, frame_times)
    with pytest.raises(MediaAddressError):
        frames_in_range(far_beyond, frame_times)


def test_a_single_frame_stream_is_queryable_shortly_past_its_only_frame():
    # With no second frame to measure a real gap from, the module must
    # not guess an arbitrary one (CLAUDE.md's generality rule) -- so a
    # query a few microseconds past the sole known frame must not be
    # spuriously refused as "beyond the end of the stream".
    frame_times = [1_000_000]
    addr = parse("clip.mp4#t=1.000010")
    assert select_frame(addr, frame_times) == 0


# ---------------------------------------------------------------------------
# Decision 5 -- region pixel arithmetic and validation.
# ---------------------------------------------------------------------------

def test_decision5_adjacent_regions_rejoin_with_no_gap_and_no_overlap():
    left_half = Region(x=0, y=0, w=500_000, h=1_000_000)
    right_half = Region(x=500_000, y=0, w=500_000, h=1_000_000)

    left_pixels = region_to_pixels(left_half, width=3200, height=1200)
    right_pixels = region_to_pixels(right_half, width=3200, height=1200)

    assert left_pixels.left == 0
    assert left_pixels.width == 1600
    assert right_pixels.left == 1600
    assert right_pixels.width == 1600
    # No gap, no overlap: the second half starts exactly where the first ends.
    assert left_pixels.left + left_pixels.width == right_pixels.left


def test_decision5_region_to_pixels_refuses_a_result_that_rounds_to_zero_area():
    # A Region always has positive area in normalised space, but at a low
    # enough target resolution its edges can round to the same pixel row
    # or column -- a valid address producing an unusable, zero-pixel
    # picture, which must be refused rather than handed back silently.
    tiny_but_valid = Region(x=100_000, y=0, w=5_000, h=1_000_000)
    with pytest.raises(MediaAddressError):
        region_to_pixels(tiny_but_valid, width=20, height=20)


def test_decision5_region_rejects_out_of_range_and_zero_area_values():
    with pytest.raises(MediaAddressError):
        parse("clip.mp4#r=1.5,0,0.5,0.5")          # x out of range
    with pytest.raises(MediaAddressError):
        parse("clip.mp4#r=0,0,0,0.5")               # zero width
    with pytest.raises(MediaAddressError):
        parse("clip.mp4#r=0,0,0.5,-0.5")             # negative height
    with pytest.raises(MediaAddressError):
        parse("clip.mp4#r=0.8,0,0.5,0.5")            # x + w > 1
    with pytest.raises(MediaAddressError):
        parse("clip.mp4#r=0,0.8,0.5,0.5")            # y + h > 1


# ---------------------------------------------------------------------------
# Decisions 7, 9 -- stream selection and canonical form.
# ---------------------------------------------------------------------------

def test_decision7_the_default_stream_selector_is_elided_in_canonical_form():
    with_explicit_default = parse("clip.mp4#v=0&f=5")
    without_selector = parse("clip.mp4#f=5")
    assert with_explicit_default == without_selector
    assert format(with_explicit_default) == "clip.mp4#f=5"
    assert hash(with_explicit_default) == hash(without_selector)

    # A non-default selector is not elided, and is not the same address.
    other_stream = parse("clip.mp4#v=1&f=5")
    assert other_stream != without_selector


def test_decision9_component_order_is_normalised_regardless_of_input_order():
    written_backwards = parse("clip.mp4#r=0,0,1,1&t=1.000000")
    written_forwards = parse("clip.mp4#t=1.000000&r=0.000000,0.000000,1.000000,1.000000")
    assert written_backwards == written_forwards
    assert format(written_backwards) == "clip.mp4#t=1.000000&r=0.000000,0.000000,1.000000,1.000000"


def test_decision9_seven_decimal_places_raises_rather_than_rounds():
    with pytest.raises(MediaAddressError):
        parse("clip.mp4#t=1.1234567")


def test_decision9_format_is_idempotent():
    addr = dataclasses.replace(
        from_path("videos/p1.mp4"),
        time_range_us=(1_000_000, 4_000_000),
        region=Region(x=0, y=0, w=500_000, h=500_000),
    )
    once = format(addr)
    twice = format(parse(once))
    assert once == twice


def test_decision9_stored_form_and_key_form_differ_as_specified():
    addr = dataclasses.replace(from_path("videos/p1.mp4"), time_us=1_000_000)
    stored = format(addr)
    key = canonical_key(addr, project_root="C:/projects/study1")
    assert stored == "videos/p1.mp4#t=1.000000"
    assert key == "C:/projects/study1/videos/p1.mp4#t=1.000000"
    assert stored != key

    # Case is deliberately not folded (decision 9): two spellings of one
    # real path differing only in case produce two different keys.
    same_but_different_case = canonical_key(addr, project_root="c:/projects/study1")
    assert same_but_different_case != key


def test_decision9_two_spellings_collapse_to_one_canonical_string_and_hash():
    a = parse("clip.mp4#v=0&r=0,0,1,1&t=1.000000")
    b = parse("clip.mp4#t=1.0&r=0.0,0.0,1.0,1.0")
    assert format(a) == format(b)
    assert hash(a) == hash(b)
    assert a == b


# ---------------------------------------------------------------------------
# Decision 8 -- frame identity is never converted to or from time.
# ---------------------------------------------------------------------------

def test_decision8_a_frame_address_and_a_time_address_are_never_equal():
    # On a nominal 25 fps stream, frame 100 and t=4.0s would name the same
    # picture -- and must still be different addresses (decision 8).
    by_frame = parse("clip.mp4#f=100")
    by_time = parse("clip.mp4#t=4.000000")
    assert by_frame != by_time
    assert hash(by_frame) != hash(by_time)


def test_decision8_select_frame_never_recomputes_a_frame_ordinal_from_timing():
    by_frame = parse("clip.mp4#f=7")
    # Two wildly different timing sequences of the same length. If
    # select_frame ever multiplied a nominal frame rate by the ordinal, or
    # otherwise consulted frame_times to resolve a frame address, these
    # would disagree. They must not.
    dense = _constant_25fps(10)
    sparse = [i * 5_000_000 for i in range(10)]
    assert select_frame(by_frame, dense) == 7
    assert select_frame(by_frame, sparse) == 7

    # There is no conversion method on MediaAddress at all.
    field_names = {f.name for f in dataclasses.fields(MediaAddress)}
    assert field_names == {"path", "stream", "frame", "time_us", "time_range_us", "region"}
    assert not hasattr(by_frame, "to_time")
    assert not hasattr(by_frame, "as_time")
    assert not hasattr(by_frame, "to_frame")


# ---------------------------------------------------------------------------
# Decisions 10, 11 -- time origin (non-negative) and degenerate values.
# ---------------------------------------------------------------------------

def test_decision10_a_negative_time_is_refused():
    with pytest.raises(MediaAddressError, match="negative"):
        parse("clip.mp4#t=-1.000000")


def test_decision11_a_negative_frame_ordinal_is_refused():
    with pytest.raises(MediaAddressError, match="negative"):
        parse("clip.mp4#f=-1")


def test_decision11_a_reversed_range_raises_and_is_never_reordered():
    # The dangerous option named in decision 11 is silently swapping this
    # to #t=2-5. Confirm it raises instead -- there is no code path here
    # that could hand back a MediaAddress with the endpoints swapped.
    with pytest.raises(MediaAddressError, match="reversed"):
        parse("clip.mp4#t=5.000000-2.000000")


def test_decision11_a_zero_length_range_is_refused():
    with pytest.raises(MediaAddressError):
        parse("clip.mp4#t=3.000000-3.000000")


def test_decision11_a_negative_range_endpoint_is_refused():
    with pytest.raises(MediaAddressError, match="negative"):
        parse("clip.mp4#t=-1.000000-2.000000")


def test_decision11_an_empty_fragment_is_an_error():
    with pytest.raises(MediaAddressError):
        parse("clip.mp4#")


def test_decision11_each_malformed_form_names_the_problem():
    # A spot check that error messages are specific enough to act on,
    # not a single generic "invalid address" string.
    cases = {
        "clip.mp4#t=-1.000000": "negative",
        "clip.mp4#f=-1": "negative",
        "clip.mp4#t=5.000000-2.000000": "reversed",
        "clip.mp4#t=1.1234567": "decimal places",
        "clip.mp4#": "fragment",
        "clip.mp4#z=1": "unknown",
    }
    for address_string, expected_fragment in cases.items():
        with pytest.raises(MediaAddressError) as excinfo:
            parse(address_string)
        assert expected_fragment in str(excinfo.value), (
            f"{address_string!r} raised {excinfo.value!r}, which does not "
            f"name the problem ({expected_fragment!r})"
        )


def test_decision11_a_frame_ordinal_beyond_the_last_frame_is_refused_at_resolve_time():
    frame_times = _constant_25fps(10)
    addr = parse("clip.mp4#f=10")  # valid at parse time -- 0..9 exist, not 10
    with pytest.raises(MediaAddressError):
        select_frame(addr, frame_times)


# ---------------------------------------------------------------------------
# Section 7 -- the lossless known-frame fixture. Generated on demand,
# small, contains no participants -- see docs/fixtures.md. This is the
# fixture's own self-check; end-to-end frame-selection tests against it
# belong to P1.2, which has a real resolver to decode with.
# ---------------------------------------------------------------------------

FFMPEG_MISSING = shutil.which("ffmpeg") is None
FFPROBE_MISSING = shutil.which("ffprobe") is None


def _generate_known_frame_video(tmp_path, width=64, height=64, fps=25, duration_s=2):
    """A lossless video whose every pixel of frame N equals N.

    Grayscale (-pix_fmt gray) sidesteps any RGB/YUV colour conversion, and
    FFV1 is lossless, so "is this frame 1450" has an exact machine-checkable
    answer with nothing approximate anywhere in the pipeline.
    """
    out_path = tmp_path / "known_frames.mkv"
    command = [
        "ffmpeg", "-hide_banner", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration_s}",
        "-vf", "format=gray,geq=lum='N'",
        "-pix_fmt", "gray",
        "-c:v", "ffv1",
        str(out_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return out_path


def _decode_gray_frames(path, width, height):
    command = [
        "ffmpeg", "-hide_banner", "-v", "error",
        "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    frame_size = width * height
    data = result.stdout
    frame_count = len(data) // frame_size
    return [data[i * frame_size:(i + 1) * frame_size] for i in range(frame_count)]


@pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg is not on PATH")
def test_lossless_known_frame_fixture_self_check(tmp_path):
    width, height, fps, duration_s = 64, 64, 25, 2
    video_path = _generate_known_frame_video(tmp_path, width, height, fps, duration_s)

    frames = _decode_gray_frames(video_path, width, height)
    assert len(frames) == fps * duration_s

    for index, frame_bytes in enumerate(frames):
        assert set(frame_bytes) == {index}, (
            f"frame {index}'s pixels do not all report its own index"
        )


# ---------------------------------------------------------------------------
# Decision 12 -- frame ordinal after an edit list. Not covered by any
# committed fixture (docs/fixtures.md's Known Gaps table). This attempts
# to produce one from a real recording and checks with ffprobe whether it
# actually worked, per the instruction not to assume the command produced
# an edit list just because it ran without error.
# ---------------------------------------------------------------------------

GELEM_FIXTURES = os.environ.get("GELEM_FIXTURES")


@pytest.mark.skipif(FFMPEG_MISSING or FFPROBE_MISSING, reason="ffmpeg/ffprobe not on PATH")
@pytest.mark.skipif(GELEM_FIXTURES is None, reason="GELEM_FIXTURES is not set")
def test_decision12_edit_list_on_video_stream_attempt(tmp_path):
    source = pathlib.Path(GELEM_FIXTURES) / "sid89_video.mp4"
    if not source.exists():
        pytest.skip(f"expected fixture not found: {source}")

    # A non-keyframe-aligned start (10.3s; keyframes are at even seconds
    # in this recording) forces ffmpeg's stream-copy trim to either drop
    # frames before the requested start or hide them with an edit list.
    out_path = tmp_path / "elst_attempt.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-y",
        "-ss", "10.3", "-i", str(source), "-t", "3",
        "-c", "copy", "-map", "0:v:0",
        str(out_path),
    ]
    subprocess.run(command, check=True, capture_output=True)

    probe = subprocess.run(
        ["ffprobe", "-hide_banner", "-v", "debug", str(out_path)],
        capture_output=True, text=True,
    )
    # The mapped output holds only the video stream, so it is always
    # stream index 0 -- this line names that stream specifically, not
    # some other stream that happens to also carry an edit list.
    has_video_stream_elst = "Processing st: 0, edit list" in probe.stderr

    if not has_video_stream_elst:
        pytest.skip(
            "this attempt did not produce a video-stream edit list; "
            "decision 12 stays unverified -- see docs/media_architecture.md "
            "section 3.6, item 12"
        )
    # Reaching here means has_video_stream_elst is already True -- the
    # skip above is this test's only real check; there is nothing further
    # to assert.


# ---------------------------------------------------------------------------
# P0.2c -- address-aware project paths.
#
# parse() now applies decision 9's forward-slash normalisation to the path
# portion (previously only from_path() did), and two new pure functions,
# absolutise() and relativise(), swap the path portion of an address
# against a base without ever touching the fragment.
#
# Written from the P0.2c work-item specification, not from the
# implementation. Each test is designed to fail against a media_address.py
# that predates P0.2c.
# ---------------------------------------------------------------------------

# Every grammar form from section 3.2, as keyword arguments to
# dataclasses.replace() on a bare-path address. Used by the round-trip
# identity test so no form is left unchecked.
_GRAMMAR_FORMS = {
    "bare path": {},
    "frame ordinal (#f=)": {"frame": 1234},
    "time point (#t=point)": {"time_us": 1_500_000},
    "time range (#t=range)": {"time_range_us": (1_000_000, 4_000_000)},
    "region (#r=)": {"region": Region(x=100_000, y=200_000, w=300_000, h=400_000)},
    "full form with a stream selector": {
        "stream": StreamSelector(kind="v", index=1),
        "frame": 50,
        "region": Region(x=0, y=0, w=500_000, h=1_000_000),
    },
}


def test_p02c_parse_normalises_a_backslash_spelled_path_without_a_fragment():
    # parse() must apply the same _to_posix() normalisation from_path()
    # applies, so parse(format(a)) == a holds however the address was built.
    addr = parse(r"C:\videos\p1.mp4")
    assert addr.path == "C:/videos/p1.mp4"


def test_p02c_parse_normalises_a_backslash_spelled_path_with_a_fragment():
    # The path portion is normalised; the fragment is parsed as before.
    addr = parse(r"C:\videos\p1.mp4#t=1.500000")
    assert addr.path == "C:/videos/p1.mp4"
    assert addr.time_us == 1_500_000


def test_p02c_parse_leaves_an_empty_path_empty_rather_than_making_it_dot():
    # PureWindowsPath("") is ".", which is not what an empty path portion
    # means -- parse() must guard that case.
    assert parse("").path == ""


def test_p02c_parse_of_a_backslash_absolute_path_round_trips_through_format():
    original = parse(r"D:\study\clips\trial 3.mov")
    assert parse(format(original)) == original


def test_p02c_absolutise_is_a_noop_on_an_already_absolute_path_all_three_forms():
    # POSIX root, UNC, and drive-letter are all "already absolute" and must
    # come back unchanged (same canonical string) whatever base is passed.
    base = "C:/some/other/place"
    for already_absolute in ("/mnt/data/p1.mp4", "//server/share/p1.mp4", "E:/media/p1.mp4"):
        addr = from_path(already_absolute)
        result = absolutise(addr, base)
        assert result == addr, f"absolutise changed an already-absolute path: {already_absolute!r}"
        assert result.path == already_absolute


def test_p02c_absolutise_resolves_a_relative_path_against_its_base():
    addr = dataclasses.replace(from_path("videos/p1.mp4"), time_us=2_000_000)
    result = absolutise(addr, "C:/projects/study1")
    assert result.path == "C:/projects/study1/videos/p1.mp4"
    # The fragment is untouched.
    assert result.time_us == 2_000_000


def test_p02c_relativise_leaves_an_external_path_untouched():
    # A sibling directory of the project root is not inside it.
    project_root = "C:/projects/study1"
    external = from_path("C:/projects/study2/p1.mp4")
    assert relativise(external, project_root) == external


def test_p02c_relativise_leaves_a_different_drive_untouched():
    # PureWindowsPath.relative_to raises for a different drive letter; that
    # raise is the [NOW] rule "only paths inside the project folder become
    # relative", so relativise must hand the address straight back.
    project_root = "C:/projects/study1"
    other_drive = from_path("D:/projects/study1/p1.mp4")
    assert relativise(other_drive, project_root) == other_drive


def test_p02c_relativise_of_an_already_relative_path_is_a_noop():
    addr = from_path("videos/p1.mp4")
    assert relativise(addr, "C:/projects/study1") == addr


def test_p02c_relativise_then_absolutise_is_the_identity_for_every_grammar_form():
    # An address whose path lies inside the project root: relativising it
    # and then absolutising against the same root must reproduce it exactly,
    # fragment and all. MediaAddress equality is canonical-string equality,
    # so this pins the fragment.
    project_root = "C:/projects/study1"
    for form_name, fragment in _GRAMMAR_FORMS.items():
        inside = dataclasses.replace(
            from_path("C:/projects/study1/videos/p1.mp4"), **fragment
        )
        round_tripped = absolutise(relativise(inside, project_root), project_root)
        assert round_tripped == inside, f"identity failed for {form_name}"


def test_p02c_a_literal_hash_and_percent_survive_relativise_and_absolutise():
    # The path portion carries a literal '#' and a literal '%'. Neither the
    # rewriting nor the escaping may corrupt them.
    project_root = "C:/projects/study1"
    original = dataclasses.replace(
        from_path("C:/projects/study1/weird #1/100% done.mp4"),
        time_us=3_000_000,
    )
    round_tripped = absolutise(relativise(original, project_root), project_root)
    assert round_tripped == original
    # The stored string escapes '#' and '%'; parsing it recovers the exact
    # path, literal characters intact.
    stored = format(relativise(original, project_root))
    assert "%23" in stored and "%25" in stored
    assert parse(stored).path == "weird #1/100% done.mp4"


def test_p02c_canonical_key_is_format_of_absolutise():
    # canonical_key() must have exactly one definition of "make absolute" --
    # absolutise() -- so the two never drift apart.
    addr = dataclasses.replace(from_path("videos/p1.mp4"), time_us=1_000_000)
    project_root = "C:/projects/study1"
    assert canonical_key(addr, project_root) == format(absolutise(addr, project_root))
