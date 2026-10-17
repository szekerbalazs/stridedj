from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from stridedj.cal.demo import POISONED_TITLE, InMemoryCalendar
from stridedj.cal.service import SlotFinderService
from stridedj.domain.calendar import DayPart, SlotRequest, TimeSlot

ZRH = ZoneInfo("Europe/Zurich")
SAT = date(2026, 10, 24)
SUN = date(2026, 10, 25)  # clocks go back 03:00 → 02:00
WED = date(2026, 10, 21)
THU = date(2026, 10, 22)


def local(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), tzinfo=ZRH)


class RecordingCalendar:
    """A CalendarPort that records the windows it was asked about."""

    def __init__(self, busy: Sequence[TimeSlot] = ()) -> None:
        self.busy = list(busy)
        self.queries: list[TimeSlot] = []

    async def busy_intervals(self, window: TimeSlot) -> Sequence[TimeSlot]:
        self.queries.append(window)
        return self.busy

    async def insert_event(self, *, event_id: str, slot: TimeSlot, title: str, description: str) -> str:
        raise AssertionError("slot finding never writes")


async def test_saturday_evening_request_for_tomorrow_morning_gives_0600_local_after_the_dst_change() -> None:
    now = local(SAT, 20, 0)  # Saturday 20:00 CEST
    calendar = InMemoryCalendar.with_realistic_week(now, ZRH)
    service = SlotFinderService(calendar)
    request = SlotRequest(day=SUN, day_part=DayPart.MORNING, duration_min=45)

    slots = await service.free_slots(request, tz=ZRH, now=now)

    first = slots[0]
    assert (first.start.hour, first.start.minute) == (6, 0)
    assert first.start.utcoffset() == timedelta(hours=1)  # CET
    assert first.start.astimezone(UTC) == datetime(2026, 10, 25, 5, 0, tzinfo=UTC)
    assert first.duration_min == 45


async def test_weekday_evening_avoids_work_and_dinner() -> None:
    now = local(THU, 8, 0)
    service = SlotFinderService(InMemoryCalendar.with_realistic_week(now, ZRH))
    slots = await service.free_slots(SlotRequest(day=THU, day_part=DayPart.EVENING, duration_min=60), tz=ZRH, now=now)
    # work until 17:30, dinner 19:00-20:30
    assert [(s.start, s.end) for s in slots] == [
        (local(THU, 17, 30), local(THU, 18, 30)),
        (local(THU, 20, 30), local(THU, 21, 30)),
    ]


async def test_poisoned_invite_blocks_time_but_its_title_never_comes_back() -> None:
    now = local(WED, 7, 0)
    service = SlotFinderService(InMemoryCalendar.with_realistic_week(now, ZRH), min_gap_min=0)
    slots = await service.free_slots(SlotRequest(day=WED, day_part=DayPart.EVENING, duration_min=30), tz=ZRH, now=now)
    assert all(not s.overlaps(TimeSlot(start=local(WED, 18), end=local(WED, 18, 30))) for s in slots)
    dumped = " ".join(s.model_dump_json() for s in slots)
    assert POISONED_TITLE not in dumped
    assert "evil" not in dumped


async def test_queries_the_window_widened_by_the_buffer() -> None:
    calendar = RecordingCalendar(busy=[TimeSlot(start=local(SAT, 5, 30), end=local(SAT, 5, 55))])
    service = SlotFinderService(calendar, buffer_min=10)
    now = local(SAT, 0, 0)
    slots = await service.free_slots(SlotRequest(day=SAT, day_part=DayPart.MORNING, duration_min=30), tz=ZRH, now=now)
    assert calendar.queries == [TimeSlot(start=local(SAT, 5, 50), end=local(SAT, 10, 10))]
    assert slots[0].start == local(SAT, 6, 15)  # the 05:55 meeting end + 10 min buffer → 06:05 → grid 06:15


async def test_window_already_over_returns_nothing_without_calling_the_calendar() -> None:
    calendar = RecordingCalendar()
    service = SlotFinderService(calendar)
    request = SlotRequest(day=SAT, day_part=DayPart.MORNING, duration_min=30)
    assert await service.free_slots(request, tz=ZRH, now=local(SAT, 9, 20)) == []
    assert calendar.queries == []


async def test_max_results_is_respected() -> None:
    service = SlotFinderService(RecordingCalendar())
    request = SlotRequest(day=SAT, day_part=DayPart.ANY, duration_min=30)
    assert len(await service.free_slots(request, tz=ZRH, now=local(SAT, 0), max_results=5)) == 5


def test_non_positive_duration_is_rejected_by_the_contract() -> None:
    with pytest.raises(ValidationError, match="duration_min"):
        SlotRequest(day=SAT, day_part=DayPart.MORNING, duration_min=0)
