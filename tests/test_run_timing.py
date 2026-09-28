"""
tests/test_run_timing.py

run_timing.py's pure, Qt-free arithmetic: the elapsed-time formatter, the
run-time estimate, the long-run threshold decision, and the coarse phrase
the long-run warning dialog shows. No controller, no operator registry,
no Qt -- every function here takes plain numbers and returns plain numbers
or strings.

Written from the work-item specification, not from the implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_timing import (
    ESTIMATE_MIN_ROWS,
    estimate_remaining_seconds,
    exceeds_long_run_threshold,
    format_approximate_duration,
    format_duration,
)


# ===========================================================================
# 1. format_duration: "{d}d:{h}h:{mm}m:{ss}s", minutes/seconds two digits,
#    days/hours unpadded.
# ===========================================================================

def test_format_duration_zero():
    assert format_duration(0) == "0d:0h:00m:00s"


def test_format_duration_59_seconds():
    assert format_duration(59) == "0d:0h:00m:59s"


def test_format_duration_3599_seconds():
    # One second short of an hour -- 59 minutes, 59 seconds, no hour yet.
    assert format_duration(3599) == "0d:0h:59m:59s"


def test_format_duration_one_hour_exactly():
    assert format_duration(3600) == "0d:1h:00m:00s"


def test_format_duration_over_a_day():
    # 1 day, 2 hours, 3 minutes, 4 seconds.
    total = 86400 + 2 * 3600 + 3 * 60 + 4
    assert format_duration(total) == "1d:2h:03m:04s"


def test_format_duration_many_days_unpadded():
    # 10 days -- the days field is never zero-padded.
    assert format_duration(10 * 86400) == "10d:0h:00m:00s"


def test_format_duration_never_negative():
    # Would still pass if violated? No. A negative reading (a clock
    # anomaly, or a caller's arithmetic mistake) must never render as a
    # negative duration in the status bar.
    assert format_duration(-5) == "0d:0h:00m:00s"


def test_format_duration_truncates_fractional_seconds():
    assert format_duration(59.9) == "0d:0h:00m:59s"


# ===========================================================================
# 2. estimate_remaining_seconds: no estimate before ESTIMATE_MIN_ROWS rows
#    finished AFTER the anchor; correct arithmetic once there are enough.
#
#    The anchor is the first DRAIN TICK to observe any row finished, not
#    literally "the first finished row" -- finished_count_at_first is how
#    many rows had already finished AT that tick (see run_timing.py's own
#    docstring). Most of these tests use finished_count_at_first=1, which
#    reproduces "the first row itself is the anchor" -- the simplest case,
#    and the one every pre-anchor-fix test already assumed.
# ===========================================================================

def test_no_estimate_with_no_first_finished_time():
    assert estimate_remaining_seconds(
        finished_row_count=0,
        rows_requested=100,
        first_finished_monotonic=None,
        finished_count_at_first=None,
        now=1000.0,
    ) is None


def test_no_estimate_before_estimate_min_rows_after_the_anchor():
    # Exactly ESTIMATE_MIN_ROWS rows finished counts the anchor row
    # itself, so only ESTIMATE_MIN_ROWS - 1 have finished AFTER it --
    # one short.
    assert estimate_remaining_seconds(
        finished_row_count=ESTIMATE_MIN_ROWS,
        rows_requested=10_000,
        first_finished_monotonic=0.0,
        finished_count_at_first=1,
        now=100.0,
    ) is None


def test_estimate_appears_the_row_after_the_threshold():
    # ESTIMATE_MIN_ROWS + 1 total finished == ESTIMATE_MIN_ROWS finished
    # AFTER the anchor -- the threshold this function documents.
    result = estimate_remaining_seconds(
        finished_row_count=ESTIMATE_MIN_ROWS + 1,
        rows_requested=10_000,
        first_finished_monotonic=0.0,
        finished_count_at_first=1,
        now=100.0,
    )
    assert result is not None


def test_estimate_arithmetic_is_correct():
    # 31 rows finished, the anchor (count 1) at t=0, now at t=30 -- so 30
    # rows finished in 30 seconds after the anchor: rate = 1 row/second.
    # rows_requested=130, finished=31 -> 99 rows left -> 99 seconds.
    result = estimate_remaining_seconds(
        finished_row_count=31,
        rows_requested=130,
        first_finished_monotonic=0.0,
        finished_count_at_first=1,
        now=30.0,
    )
    assert result == 99.0


def test_estimate_is_none_when_rows_requested_is_zero():
    # A TABLE/DISPLAY run's rows_requested is 0 (this function does not
    # know that; it just refuses to divide by an empty run).
    assert estimate_remaining_seconds(
        finished_row_count=50,
        rows_requested=0,
        first_finished_monotonic=0.0,
        finished_count_at_first=1,
        now=100.0,
    ) is None


def test_estimate_is_zero_when_every_row_is_already_finished():
    result = estimate_remaining_seconds(
        finished_row_count=100,
        rows_requested=100,
        first_finished_monotonic=0.0,
        finished_count_at_first=1,
        now=100.0,
    )
    assert result == 0.0


def test_estimate_recomputes_from_scratch_each_call():
    # Would still pass if violated? No. The function takes no state
    # between calls -- calling it twice with a later `now` and a higher
    # finished_row_count must give a fresh, independently correct answer,
    # not one derived from the previous call.
    first = estimate_remaining_seconds(
        finished_row_count=31, rows_requested=1000,
        first_finished_monotonic=0.0, finished_count_at_first=1, now=30.0,
    )
    second = estimate_remaining_seconds(
        finished_row_count=61, rows_requested=1000,
        first_finished_monotonic=0.0, finished_count_at_first=1, now=60.0,
    )
    assert first == 969.0    # (1000 - 31) / 1.0
    assert second == 939.0   # (1000 - 61) / 1.0


# ===========================================================================
# 2b. Anchor fix (run-time-estimate anchor-fix item): finished_count_at_
#     first must be the REAL count observed at the anchor tick, not
#     assumed to be 1 -- several rows can finish before the drain's first
#     tick ever notices any did, especially on the parallel path where
#     more than one consumer thread finishes rows concurrently between
#     two ticks.
# ===========================================================================

def test_anchor_boundary_and_arithmetic_with_a_nontrivial_anchor_count():
    """The work item's own spec: an anchor observed with count 10 at
    t=0 (10 rows had already finished by the first tick that noticed
    any had). At t=30: count 39 is one row short of the boundary (29
    rows since the anchor) and must give no estimate; count 40 lands
    exactly on the boundary (30 rows since the anchor, matching this
    function's existing ">=" convention -- see test_estimate_appears_
    the_row_after_the_threshold above) and must give an estimate. At
    that count, with rows_requested=100: remaining =
    (100 - 40) rows / (30 rows / 30 s) = 60 s.
    """
    common = dict(
        first_finished_monotonic=0.0, finished_count_at_first=10, now=30.0,
    )

    assert estimate_remaining_seconds(
        finished_row_count=39, rows_requested=100, **common,
    ) is None

    result = estimate_remaining_seconds(
        finished_row_count=40, rows_requested=100, **common,
    )
    assert result == 60.0


def test_anchor_count_matters_not_just_the_anchor_time():
    # Would still pass if violated? No. Two runs with the SAME anchor
    # time and the SAME current finished_row_count, but different
    # finished_count_at_first, must give different rates -- if the
    # function silently ignored finished_count_at_first (e.g. reverted
    # to assuming the anchor count is always 1), both would give the
    # same answer.
    few_already_finished = estimate_remaining_seconds(
        finished_row_count=40, rows_requested=100,
        first_finished_monotonic=0.0, finished_count_at_first=1, now=30.0,
    )
    many_already_finished = estimate_remaining_seconds(
        finished_row_count=40, rows_requested=100,
        first_finished_monotonic=0.0, finished_count_at_first=10, now=30.0,
    )
    assert few_already_finished != many_already_finished
    assert few_already_finished == 60.0 * 30.0 / 39.0   # (100-40)/(39/30)
    assert many_already_finished == 60.0                # (100-40)/(30/30)


# ===========================================================================
# 3. Reversal check: the formatter must read its clock from the `now`
#    argument, never from time.monotonic() itself.
# ===========================================================================

def test_estimate_uses_the_given_now_not_the_real_clock():
    # Would still pass if violated? No. If this function secretly called
    # time.monotonic() instead of using `now`, a `now` chosen far from the
    # real wall clock (here, a small fabricated value) would give a
    # nonsensical or wildly different elapsed time than the one the
    # arithmetic below expects. Real time.monotonic() readings are large
    # (seconds since an arbitrary epoch, typically a big number); 30.0 is
    # deliberately tiny and controlled.
    result = estimate_remaining_seconds(
        finished_row_count=31, rows_requested=130,
        first_finished_monotonic=0.0, finished_count_at_first=1, now=30.0,
    )
    assert result == 99.0


def test_format_duration_does_not_read_the_real_clock():
    # format_duration takes a duration directly, not a `now` -- this pins
    # that a fabricated, non-realistic value produces the exact expected
    # string rather than something derived from the real clock.
    assert format_duration(65) == "0d:0h:01m:05s"


# ===========================================================================
# 4. exceeds_long_run_threshold.
# ===========================================================================

def test_exceeds_long_run_threshold_true_when_over():
    assert exceeds_long_run_threshold(31 * 60, 30) is True


def test_exceeds_long_run_threshold_false_when_under():
    assert exceeds_long_run_threshold(29 * 60, 30) is False


def test_exceeds_long_run_threshold_false_when_exactly_equal():
    # Strictly greater than, not greater-or-equal -- a run that lands
    # exactly on the threshold is not "longer than" it.
    assert exceeds_long_run_threshold(30 * 60, 30) is False


# ===========================================================================
# 5. format_approximate_duration -- the coarse phrase the long-run warning
#    dialog shows ("about 14 hours"), distinct from format_duration's
#    precise d:h:m:s.
# ===========================================================================

def test_approximate_duration_under_a_minute():
    assert format_approximate_duration(30) == "under a minute"


def test_approximate_duration_minutes():
    text = format_approximate_duration(5 * 60)
    assert "5" in text
    assert "minute" in text


def test_approximate_duration_singular_minute():
    assert format_approximate_duration(65) == "about 1 minute"


def test_approximate_duration_hours():
    text = format_approximate_duration(14 * 3600)
    assert "14" in text
    assert "hour" in text


def test_approximate_duration_singular_hour():
    assert format_approximate_duration(3700) == "about 1 hour"


def test_approximate_duration_days():
    text = format_approximate_duration(3 * 86400)
    assert "3" in text
    assert "day" in text
