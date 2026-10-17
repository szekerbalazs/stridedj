"""Day parts → concrete local windows (DST-safe), quarter-hour grid, free-slot search.

Pure functions, no I/O: `now` is always passed in, so every rule here is unit-testable around a DST change.

Two kinds of time arithmetic are kept strictly apart, because mixing them is the classic calendar bug:

* **Wall-clock** ("06:00 on Sunday in Zurich"): built with `datetime.combine(day, time, tzinfo=ZoneInfo(...))`, which
  looks up the UTC offset valid on *that* day. Shifting an existing datetime by `timedelta(days=1)` instead is wrong
  whenever it carries a fixed offset (as parsed from API JSON, e.g. `+02:00`): across the October change, "Saturday
  06:00 + 1 day" lands on Sunday 05:00 local.
* **Elapsed time** ("45 minutes after this instant"): done in UTC. Python compares and subtracts two datetimes that
  share the same `tzinfo` object by their wall-clock fields (PEP 495), so with a `ZoneInfo` "01:00 → 04:00" on the
  night the clocks go back is 3 h, although 4 h elapse. Converting to UTC first makes every comparison and
  difference an honest one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from types import MappingProxyType
from zoneinfo import ZoneInfo

from stridedj.domain.calendar import DayPart, TimeSlot

DAY_PART_HOURS: Mapping[DayPart, tuple[time, time]] = MappingProxyType(
    {
        DayPart.MORNING: (time(6, 0), time(10, 0)),
        DayPart.MIDDAY: (time(11, 30), time(14, 0)),
        DayPart.EVENING: (time(17, 0), time(21, 30)),
        DayPart.ANY: (time(6, 0), time(21, 30)),
    }
)
"""Local wall-clock bounds of each day part. All lie well clear of the 01:00-03:00 band where DST changes happen."""

MIN_LEAD_MIN = 15
"""A slot never starts sooner than this after `now`: time to read the proposal, approve it and put shoes on."""

DEFAULT_GRID_MIN = 15
DEFAULT_MIN_GAP_MIN = 30

_Interval = tuple[datetime, datetime]  # (start, end) in UTC, internal only


def window_for(day: date, day_part: DayPart, tz: ZoneInfo) -> TimeSlot:
    """The local window of `day_part` on `day`, e.g. morning = 06:00-10:00 in `tz` with that day's UTC offset."""
    start_t, end_t = DAY_PART_HOURS[day_part]
    return TimeSlot(start=datetime.combine(day, start_t, tzinfo=tz), end=datetime.combine(day, end_t, tzinfo=tz))


def round_up_to_grid(dt: datetime, grid_min: int = DEFAULT_GRID_MIN) -> datetime:
    """Round `dt` up to the next local wall-clock multiple of `grid_min` (identity if it is already on the grid).

    The grid is defined on the wall clock of `dt`'s own timezone (minutes past the hour), so convert to the user's
    zone first. The shift itself is applied as elapsed time (via UTC), which keeps the result a real instant even
    in the hour around a DST change.
    """
    _check_grid(grid_min)
    tz = _tz_of(dt)
    past = timedelta(minutes=dt.minute % grid_min, seconds=dt.second, microseconds=dt.microsecond)
    if not past:
        return dt
    return (dt.astimezone(UTC) + timedelta(minutes=grid_min) - past).astimezone(tz)


def merge_intervals(intervals: Iterable[TimeSlot]) -> list[TimeSlot]:
    """Sort and merge overlapping or touching intervals (09:00-10:00 + 10:00-11:00 → 09:00-11:00), returned in UTC.

    Used to combine busy blocks from several calendars: the result is the minimal set of disjoint busy intervals.
    """
    merged = _merge((slot.start.astimezone(UTC), slot.end.astimezone(UTC)) for slot in intervals)
    return [TimeSlot(start=start, end=end) for start, end in merged]


def clip_to_window(intervals: Iterable[TimeSlot], window: TimeSlot) -> list[TimeSlot]:
    """The parts of `intervals` that fall inside `window` (UTC), dropping those that do not overlap it at all."""
    lo, hi = window.start.astimezone(UTC), window.end.astimezone(UTC)
    clipped: list[TimeSlot] = []
    for slot in intervals:
        start, end = max(slot.start.astimezone(UTC), lo), min(slot.end.astimezone(UTC), hi)
        if start < end:
            clipped.append(TimeSlot(start=start, end=end))
    return clipped


def free_slots(
    window: TimeSlot,
    busy: Iterable[TimeSlot],
    duration_min: float,
    *,
    now: datetime,
    max_results: int = 3,
    grid_min: int = DEFAULT_GRID_MIN,
    buffer_min: int = 0,
    min_gap_min: int = DEFAULT_MIN_GAP_MIN,
) -> list[TimeSlot]:
    """Up to `max_results` free slots of `duration_min` inside `window`, earliest first, in the window's timezone.

    * starts lie on the local `grid_min` grid and no earlier than `now + MIN_LEAD_MIN` (rounded up to the grid);
    * busy intervals are padded by `buffer_min` on both sides (travel/shower time), then subtracted;
    * consecutive offers are separated by at least `min_gap_min` of free time (end of one → start of the next), so
      the user gets genuinely different options ("06:00, 07:30 or 09:00") instead of three adjacent quarter-hours;
    * a slot that would cross a DST change is skipped: its local start/end would misstate its length (see module
      docstring). Day-part windows never contain a DST change, so this only matters for custom windows.

    Returns an empty list when nothing fits; never raises for "no room".
    """
    if duration_min <= 0:
        raise ValueError("duration_min must be > 0")
    if buffer_min < 0 or min_gap_min < 0:
        raise ValueError("buffer_min and min_gap_min must be >= 0")
    _check_grid(grid_min)
    if max_results <= 0:
        return []

    tz = _tz_of(window.start)
    _tz_of(now)  # reject a naive `now` before converting it
    earliest = _grid_up_utc(now.astimezone(UTC) + timedelta(minutes=MIN_LEAD_MIN), tz, grid_min)
    pad = timedelta(minutes=buffer_min)
    padded = _merge((b.start.astimezone(UTC) - pad, b.end.astimezone(UTC) + pad) for b in busy)
    window_utc = (window.start.astimezone(UTC), window.end.astimezone(UTC))
    gaps = _free_gaps(window_utc, padded, not_before=earliest)
    return _pick_spread(
        gaps,
        duration=timedelta(minutes=duration_min),
        min_gap=timedelta(minutes=min_gap_min),
        tz=tz,
        grid_min=grid_min,
        max_results=max_results,
    )


# ─────────────────────────────── internals (all intervals in UTC) ───────────────────────────────
def _merge(intervals: Iterable[_Interval]) -> list[_Interval]:
    merged: list[_Interval] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _free_gaps(window: _Interval, busy: list[_Interval], *, not_before: datetime) -> list[_Interval]:
    """`window` minus the (sorted, disjoint) `busy` intervals, starting no earlier than `not_before`."""
    cursor, window_end = max(window[0], not_before), window[1]
    gaps: list[_Interval] = []
    for busy_start, busy_end in busy:
        if busy_end <= cursor:
            continue
        if busy_start >= window_end:
            break
        if busy_start > cursor:
            gaps.append((cursor, busy_start))
        cursor = max(cursor, busy_end)
    if cursor < window_end:
        gaps.append((cursor, window_end))
    return gaps


def _pick_spread(
    gaps: list[_Interval],
    *,
    duration: timedelta,
    min_gap: timedelta,
    tz: tzinfo,
    grid_min: int,
    max_results: int,
) -> list[TimeSlot]:
    """Greedy earliest-first placement: take the first grid start that fits, then jump ahead by `min_gap`."""
    picked: list[TimeSlot] = []
    not_before: datetime | None = None
    step = timedelta(minutes=grid_min)
    for gap_start, gap_end in gaps:
        start = _grid_up_utc(gap_start if not_before is None else max(gap_start, not_before), tz, grid_min)
        while start + duration <= gap_end and len(picked) < max_results:
            slot = _local_slot(start, start + duration, tz)
            if slot is None:  # crosses a DST change: try the next grid start
                start += step
                continue
            picked.append(slot)
            not_before = start + duration + min_gap
            start = _grid_up_utc(not_before, tz, grid_min)
    return picked


def _local_slot(start_utc: datetime, end_utc: datetime, tz: tzinfo) -> TimeSlot | None:
    """The slot in local time, or None if its local wall-clock length differs from its real length (DST change)."""
    start, end = start_utc.astimezone(tz), end_utc.astimezone(tz)
    if end.replace(tzinfo=None) - start.replace(tzinfo=None) != end_utc - start_utc:
        return None
    return TimeSlot(start=start, end=end)


def _grid_up_utc(dt_utc: datetime, tz: tzinfo, grid_min: int) -> datetime:
    return round_up_to_grid(dt_utc.astimezone(tz), grid_min).astimezone(UTC)


def _tz_of(dt: datetime) -> tzinfo:
    """The tzinfo of an aware datetime. Naive datetimes are rejected: `astimezone(None)` would silently use the
    server's local zone."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("timezone-aware datetime required")
    return dt.tzinfo


def _check_grid(grid_min: int) -> None:
    if grid_min <= 0 or 60 % grid_min:
        raise ValueError("grid_min must divide 60 (e.g. 5, 10, 15, 30, 60)")
