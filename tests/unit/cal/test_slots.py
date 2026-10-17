from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, timezone
from itertools import pairwise
from zoneinfo import ZoneInfo

import pytest

from stridedj.cal.slots import (
    DAY_PART_HOURS,
    clip_to_window,
    free_slots,
    merge_intervals,
    round_up_to_grid,
    window_for,
)
from stridedj.domain.calendar import DayPart, TimeSlot

ZRH = ZoneInfo("Europe/Zurich")
SAT = date(2026, 10, 24)  # last day of summer time (CEST, UTC+2)
SUN = date(2026, 10, 25)  # clocks go back 03:00 CEST → 02:00 CET (UTC+1)
LONG_AGO = datetime(2026, 10, 1, tzinfo=UTC)  # a `now` that never constrains the window


def local(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), tzinfo=ZRH)


def utc(month: int, day: int, hh: int, mm: int = 0) -> datetime:
    return datetime(2026, month, day, hh, mm, tzinfo=UTC)


def span(start: datetime, end: datetime) -> TimeSlot:
    return TimeSlot(start=start, end=end)


def starts(slots: list[TimeSlot]) -> list[datetime]:
    return [s.start for s in slots]


# ─────────────────────────────── windows and DST ───────────────────────────────
@pytest.mark.parametrize(
    ("day", "start_utc", "end_utc"),
    [
        (SAT, utc(10, 24, 4), utc(10, 24, 8)),  # 06:00-10:00 CEST
        (SUN, utc(10, 25, 5), utc(10, 25, 9)),  # 06:00-10:00 CET, after the clocks went back
    ],
)
def test_morning_window_follows_the_october_dst_change(day: date, start_utc: datetime, end_utc: datetime) -> None:
    window = window_for(day, DayPart.MORNING, ZRH)
    assert window.start.astimezone(UTC) == start_utc
    assert window.end.astimezone(UTC) == end_utc
    assert (window.start.hour, window.start.minute) == (6, 0)


def test_fixed_offset_plus_one_day_is_the_bug_window_for_avoids() -> None:
    saturday_6am_from_api = datetime(2026, 10, 24, 6, tzinfo=timezone(timedelta(hours=2)))  # parsed "+02:00"
    shifted = saturday_6am_from_api + timedelta(days=1)
    assert shifted.astimezone(ZRH).hour == 5  # an hour early on Sunday
    assert window_for(SUN, DayPart.MORNING, ZRH).start.astimezone(UTC) == utc(10, 25, 5)


def test_morning_window_follows_the_march_dst_change() -> None:
    assert window_for(date(2026, 3, 28), DayPart.MORNING, ZRH).start.astimezone(UTC) == utc(3, 28, 5)  # CET
    assert window_for(date(2026, 3, 29), DayPart.MORNING, ZRH).start.astimezone(UTC) == utc(3, 29, 4)  # CEST


@pytest.mark.parametrize(
    ("part", "bounds"),
    [
        (DayPart.MORNING, ((6, 0), (10, 0))),
        (DayPart.MIDDAY, ((11, 30), (14, 0))),
        (DayPart.EVENING, ((17, 0), (21, 30))),
        (DayPart.ANY, ((6, 0), (21, 30))),
    ],
)
def test_day_part_windows_are_local_wall_clock(part: DayPart, bounds: tuple[tuple[int, int], tuple[int, int]]) -> None:
    window = window_for(SUN, part, ZRH)
    assert window.start.tzinfo is ZRH
    assert ((window.start.hour, window.start.minute), (window.end.hour, window.end.minute)) == bounds
    assert DAY_PART_HOURS[part] == (time(*bounds[0]), time(*bounds[1]))


# ─────────────────────────────── grid ───────────────────────────────
@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (local(SAT, 17, 0), local(SAT, 17, 0)),
        (local(SAT, 17, 0).replace(second=1), local(SAT, 17, 15)),
        (local(SAT, 17, 5), local(SAT, 17, 15)),
        (local(SAT, 17, 14).replace(second=59, microsecond=999_999), local(SAT, 17, 15)),
        (local(SAT, 17, 45).replace(microsecond=1), local(SAT, 18, 0)),
        (local(SAT, 23, 50), local(SUN, 0, 0)),
    ],
)
def test_round_up_to_quarter_hour(given: datetime, expected: datetime) -> None:
    assert round_up_to_grid(given, 15) == expected


def test_round_up_to_other_grids() -> None:
    assert round_up_to_grid(local(SAT, 17, 5), 30) == local(SAT, 17, 30)
    assert round_up_to_grid(local(SAT, 17, 5), 60) == local(SAT, 18, 0)


def test_round_up_inside_the_repeated_hour_is_elapsed_time() -> None:
    first_0250 = utc(10, 25, 0, 50).astimezone(ZRH)  # 02:50 CEST, ten minutes before the clocks go back
    rounded = round_up_to_grid(first_0250, 15)
    assert rounded.astimezone(UTC) == utc(10, 25, 1, 0)  # ten minutes later, not seventy
    assert (rounded.hour, rounded.minute, rounded.utcoffset()) == (2, 0, timedelta(hours=1))


def test_round_up_rejects_naive_datetimes_and_bad_grids() -> None:
    with pytest.raises(ValueError, match="aware"):
        round_up_to_grid(datetime(2026, 10, 24, 17, 5), 15)  # noqa: DTZ001 (the point of the test)
    for grid in (0, 7, 45, 90):
        with pytest.raises(ValueError, match="grid_min"):
            round_up_to_grid(local(SAT, 17, 5), grid)


# ─────────────────────────────── free_slots ───────────────────────────────
def test_free_morning_gives_spread_out_options_from_the_window_start() -> None:
    slots = free_slots(window_for(SAT, DayPart.MORNING, ZRH), [], 60, now=LONG_AGO)
    assert starts(slots) == [local(SAT, 6), local(SAT, 7, 30), local(SAT, 9)]
    assert all(s.duration_min == 60 for s in slots)
    assert all(s.start.tzinfo is ZRH for s in slots)  # local time, ready to show the user


def test_starts_are_on_the_quarter_hour_grid_after_odd_busy_blocks() -> None:
    busy = [span(local(SAT, 5, 50), local(SAT, 6, 7)), span(local(SAT, 7, 20), local(SAT, 7, 38))]
    slots = free_slots(window_for(SAT, DayPart.MORNING, ZRH), busy, 45, now=LONG_AGO)
    assert [(s.start, s.end) for s in slots] == [
        (local(SAT, 6, 15), local(SAT, 7, 0)),
        (local(SAT, 7, 45), local(SAT, 8, 30)),
        (local(SAT, 9, 0), local(SAT, 9, 45)),
    ]
    assert all(s.start.minute % 15 == 0 and s.start.second == 0 for s in slots)


@pytest.mark.parametrize(
    ("now", "first_start"),
    [
        (local(SAT, 17, 0), local(SAT, 17, 15)),  # now + 15 min is already on the grid
        (local(SAT, 17, 5), local(SAT, 17, 30)),  # 17:20 rounded up
        (utc(10, 24, 15, 5), local(SAT, 17, 30)),  # same instant given in UTC
        (local(SAT, 16, 0), local(SAT, 17, 0)),  # well before the window: the window start wins
    ],
)
def test_never_in_the_past(now: datetime, first_start: datetime) -> None:
    slots = free_slots(window_for(SAT, DayPart.EVENING, ZRH), [], 30, now=now)
    assert slots[0].start == first_start


def test_window_nearly_over() -> None:
    evening = window_for(SAT, DayPart.EVENING, ZRH)
    now = local(SAT, 21, 0)  # earliest start 21:15, window ends 21:30
    assert free_slots(evening, [], 30, now=now) == []
    assert starts(free_slots(evening, [], 15, now=now)) == [local(SAT, 21, 15)]


def test_overlapping_and_adjacent_busy_blocks_are_subtracted() -> None:
    busy = [
        span(local(SAT, 17, 30), local(SAT, 18, 30)),
        span(local(SAT, 17, 0), local(SAT, 18, 0)),  # overlaps the first, given out of order
        span(utc(10, 24, 16, 30), utc(10, 24, 17)),  # 18:30-19:00 local, adjacent, and in UTC
        span(local(SAT, 20, 0), local(SAT, 21, 0)),
    ]
    evening = window_for(SAT, DayPart.EVENING, ZRH)  # free: 19:00-20:00 and 21:00-21:30
    assert [(s.start, s.end) for s in free_slots(evening, busy, 60, now=LONG_AGO)] == [(local(SAT, 19), local(SAT, 20))]
    assert starts(free_slots(evening, busy, 30, now=LONG_AGO)) == [local(SAT, 19), local(SAT, 21)]
    assert starts(free_slots(evening, busy, 30, now=LONG_AGO, min_gap_min=0)) == [
        local(SAT, 19),
        local(SAT, 19, 30),
        local(SAT, 21),
    ]


def test_slot_may_start_when_a_busy_block_ends() -> None:
    busy = [span(local(SAT, 17), local(SAT, 18))]  # intervals are half-open: [start, end)
    slots = free_slots(window_for(SAT, DayPart.EVENING, ZRH), busy, 60, now=LONG_AGO)
    assert slots[0].start == local(SAT, 18)


def test_buffer_pads_busy_blocks_on_both_sides() -> None:
    busy = [span(local(SAT, 17), local(SAT, 18)), span(local(SAT, 19, 30), local(SAT, 21, 30))]
    slots = free_slots(window_for(SAT, DayPart.EVENING, ZRH), busy, 60, now=LONG_AGO, buffer_min=10)
    assert [(s.start, s.end) for s in slots] == [(local(SAT, 18, 15), local(SAT, 19, 15))]  # 18:10 → grid 18:15
    assert free_slots(window_for(SAT, DayPart.EVENING, ZRH), busy, 75, now=LONG_AGO, buffer_min=10) == []


def test_offers_are_separated_by_the_minimum_gap() -> None:
    slots = free_slots(window_for(SAT, DayPart.ANY, ZRH), [], 40, now=LONG_AGO, max_results=10)
    gaps = [b.start.astimezone(UTC) - a.end.astimezone(UTC) for a, b in pairwise(slots)]
    assert len(slots) == 10
    assert all(gap >= timedelta(minutes=30) for gap in gaps)
    assert all(not a.overlaps(b) for a, b in pairwise(slots))


def test_max_results_and_too_long_runs() -> None:
    morning = window_for(SAT, DayPart.MORNING, ZRH)
    assert len(free_slots(morning, [], 30, now=LONG_AGO, max_results=1)) == 1
    assert free_slots(morning, [], 30, now=LONG_AGO, max_results=0) == []
    assert free_slots(morning, [], 241, now=LONG_AGO) == []  # longer than the 4 h window
    assert len(free_slots(morning, [], 240, now=LONG_AGO)) == 1


def test_fully_busy_window_gives_nothing() -> None:
    busy = [span(local(SAT, 5), local(SAT, 11))]
    assert free_slots(window_for(SAT, DayPart.MORNING, ZRH), busy, 15, now=LONG_AGO) == []


def test_invalid_arguments() -> None:
    morning = window_for(SAT, DayPart.MORNING, ZRH)
    with pytest.raises(ValueError, match="duration_min"):
        free_slots(morning, [], 0, now=LONG_AGO)
    with pytest.raises(ValueError, match="buffer_min"):
        free_slots(morning, [], 30, now=LONG_AGO, buffer_min=-5)
    with pytest.raises(ValueError, match="aware"):
        free_slots(morning, [], 30, now=datetime(2026, 10, 24, 5))  # noqa: DTZ001 (the point of the test)


def test_a_window_spanning_the_dst_change_counts_real_time() -> None:
    night = span(local(SUN, 0), local(SUN, 5))  # 00:00 CEST → 05:00 CET is 6 h of real time
    slots = free_slots(night, [], 60, now=LONG_AGO, max_results=10, min_gap_min=0)
    # one candidate (02:00 CEST → 02:00 CET) crosses the change and is skipped; the rest are true hours
    assert [s.start.astimezone(UTC) for s in slots] == [
        utc(10, 24, 22),
        utc(10, 24, 23),
        utc(10, 25, 1),
        utc(10, 25, 2),
        utc(10, 25, 3),
    ]
    assert all(s.end.astimezone(UTC) - s.start.astimezone(UTC) == timedelta(hours=1) for s in slots)
    assert all(s.duration_min == 60 for s in slots)
    assert all(s.start.minute == 0 for s in slots)


# ─────────────────────────────── interval helpers ───────────────────────────────
def test_merge_intervals_sorts_merges_and_returns_utc() -> None:
    merged = merge_intervals(
        [
            span(local(SAT, 12), local(SAT, 13)),
            span(utc(10, 24, 7), utc(10, 24, 8)),  # 09:00-10:00 local
            span(local(SAT, 9, 30), local(SAT, 11)),  # overlaps the previous one
            span(local(SAT, 11), local(SAT, 11, 30)),  # touches it
        ]
    )
    assert [(m.start, m.end) for m in merged] == [
        (utc(10, 24, 7), utc(10, 24, 9, 30)),
        (utc(10, 24, 10), utc(10, 24, 11)),
    ]
    assert all(m.start.tzinfo is UTC for m in merged)


def test_clip_to_window() -> None:
    window = window_for(SAT, DayPart.MORNING, ZRH)
    clipped = clip_to_window(
        [
            span(local(SAT, 5), local(SAT, 6, 30)),
            span(local(SAT, 9, 30), local(SAT, 12)),
            span(local(SAT, 4), local(SAT, 6)),
        ],
        window,
    )
    assert [(c.start, c.end) for c in clipped] == [
        (utc(10, 24, 4), utc(10, 24, 4, 30)),
        (utc(10, 24, 7, 30), utc(10, 24, 8)),
    ]
