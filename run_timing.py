"""
run_timing.py

Qt-free arithmetic for the status-bar run indicator's elapsed time and the
run-time estimate for a COLUMNS run. No pandas, no Qt: every function here
takes plain numbers and returns plain numbers or strings, so it is testable
with no controller and no operator registry.

controller.py's format_run_indicator_text() calls into this module for the
duration formatting and the remaining-time arithmetic; it owns the wording,
this module owns the numbers. AppController calls exceeds_long_run_threshold()
once, the first time a run's estimate becomes available, to decide whether to
emit long_run_estimated().
"""

from __future__ import annotations


# A COLUMNS run's estimate is measured on the run itself, not a separate
# pilot. Every row already finished AT the anchor tick (the first drain
# tick to observe a nonzero finished count -- see estimate_remaining_
# seconds's own docstring) is excluded from the rate -- those rows often
# carry one-off costs (model warm-up, first decode of a source file) that
# would skew a rate measured from row zero, and in any case have no
# individual timestamp, only the anchor tick's. Once this many MORE rows
# have finished after the anchor, the rate since the anchor is trusted
# enough to project a remaining time. 30 is small enough to give an
# estimate early in a long run and large enough that a handful of unlucky
# slow rows right at the start do not dominate it.
ESTIMATE_MIN_ROWS = 30


def format_duration(seconds: float) -> str:
    """Format a duration as "{d}d:{h}h:{mm}m:{ss}s" -- minutes and seconds
    always two digits, days and hours unpadded. Negative or non-finite input
    is clamped to zero: a duration is never shown as negative."""
    if not (seconds == seconds) or seconds == float("inf"):  # NaN or +inf
        seconds = 0.0
    total_seconds = int(seconds)
    if total_seconds < 0:
        total_seconds = 0
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{days}d:{hours}h:{minutes:02d}m:{secs:02d}s"


def format_approximate_duration(seconds: float) -> str:
    """A coarse, human phrase for a total run-time estimate shown in the
    long-run warning dialog -- "about 14 hours", not the precise
    d:h:m:s the status bar uses. Rounds to the coarsest unit that still
    says something meaningful: minutes under an hour, hours under a day,
    days beyond that."""
    if seconds < 60:
        return "under a minute"
    if seconds < 3600:
        minutes = round(seconds / 60)
        unit = "minute" if minutes == 1 else "minutes"
        return f"about {minutes} {unit}"
    if seconds < 86400:
        hours = round(seconds / 3600)
        unit = "hour" if hours == 1 else "hours"
        return f"about {hours} {unit}"
    days = round(seconds / 86400)
    unit = "day" if days == 1 else "days"
    return f"about {days} {unit}"


def estimate_remaining_seconds(
    *,
    finished_row_count: int,
    rows_requested: int,
    first_finished_monotonic: float | None,
    finished_count_at_first: int | None,
    now: float,
) -> float | None:
    """The estimated seconds left in a COLUMNS run, or None if there is not
    yet enough data to trust a rate.

    The anchor is the first DRAIN TICK at which any row had finished, not
    "the first finished row" -- a drain runs only periodically (every 50ms
    on the main thread), so by the time it first observes a nonzero count,
    several rows may already have finished, especially on the parallel
    path where multiple consumer threads finish rows concurrently between
    two ticks. `finished_count_at_first` is how many rows had already
    finished at that anchor tick; `first_finished_monotonic` is when the
    anchor tick happened. Both are recorded together, once, the first time
    the count is observed to be nonzero (see AppController._apply_finished_
    row_counts).

    `rate` is the number of rows finished AFTER the anchor -- finished_row_
    count minus finished_count_at_first -- divided by the time elapsed
    since the anchor. The rows already counted AT the anchor are excluded
    from the rate entirely: they may have finished at any point before the
    anchor tick (including one-off costs like model warm-up), so there is
    no reliable timestamp for them individually, only for the tick that
    first noticed them. No estimate is given until at least
    ESTIMATE_MIN_ROWS rows have finished after the anchor.

    A row counts as finished whether it succeeded or ended in a row error
    -- the caller (operators/operator_registry.py, via on_row_finished) is
    responsible for counting both; this function just does the arithmetic
    on whatever count it is given.
    """
    if first_finished_monotonic is None or finished_count_at_first is None:
        return None
    if rows_requested <= 0:
        return None
    rows_since_anchor = finished_row_count - finished_count_at_first
    if rows_since_anchor < ESTIMATE_MIN_ROWS:
        return None
    elapsed_since_anchor = now - first_finished_monotonic
    if elapsed_since_anchor <= 0:
        return None
    rate = rows_since_anchor / elapsed_since_anchor
    if rate <= 0:
        return None
    rows_left = rows_requested - finished_row_count
    if rows_left <= 0:
        return 0.0
    return rows_left / rate


def exceeds_long_run_threshold(
    estimated_total_seconds: float, threshold_minutes: int
) -> bool:
    """True if an estimated total run time exceeds the researcher's
    long_run_warning_minutes setting."""
    return estimated_total_seconds > threshold_minutes * 60
