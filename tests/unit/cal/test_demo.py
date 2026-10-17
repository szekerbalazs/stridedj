from __future__ import annotations

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

from stridedj.cal.demo import POISONED_TITLE, InMemoryCalendar, realistic_week
from stridedj.domain.calendar import TimeSlot

ZRH = ZoneInfo("Europe/Zurich")
SAT = date(2026, 10, 24)
SUN = date(2026, 10, 25)


def local(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), tzinfo=ZRH)


def whole_week() -> TimeSlot:
    return TimeSlot(start=datetime(2026, 10, 1, tzinfo=UTC), end=datetime(2026, 11, 30, tzinfo=UTC))


def test_realistic_week_is_relative_to_now_and_dst_safe() -> None:
    events = realistic_week(local(SAT, 20), ZRH)
    days = {e.slot.start.date() for e in events}
    assert min(days) == SAT
    assert max(days) == date(2026, 10, 30)  # 7 days: Sat .. Fri
    work = [e for e in events if e.title == "Work"]
    assert len(work) == 5
    assert all(e.slot.start.weekday() < 5 for e in work)
    assert all((e.slot.start.hour, e.slot.start.minute, e.slot.end.hour) == (8, 30, 17) for e in work)
    family_lunch = next(e for e in events if e.title == "Family lunch")
    assert family_lunch.slot.start.astimezone(UTC) == datetime(2026, 10, 25, 11, tzinfo=UTC)  # 12:00 CET


def test_poisoned_event_is_included_by_default_and_can_be_left_out() -> None:
    assert any(e.title == POISONED_TITLE for e in realistic_week(local(SAT, 20), ZRH))
    assert all(e.title != POISONED_TITLE for e in realistic_week(local(SAT, 20), ZRH, include_poisoned=False))


async def test_busy_intervals_never_expose_titles() -> None:
    calendar = InMemoryCalendar.with_realistic_week(local(SAT, 20), ZRH)
    calendar.inject_poisoned_event(TimeSlot(start=local(SUN, 7), end=local(SUN, 7, 30)))

    busy = await calendar.busy_intervals(whole_week())

    assert busy
    assert all(type(b) is TimeSlot and set(type(b).model_fields) == {"start", "end"} for b in busy)
    exposed = " ".join(b.model_dump_json() for b in busy) + repr(busy) + repr(calendar)
    for fragment in ("Ignore previous", "evil.com", "Dinner", "Work"):
        assert fragment not in exposed


async def test_busy_intervals_are_clipped_merged_and_sorted() -> None:
    calendar = InMemoryCalendar()
    calendar.add_busy(TimeSlot(start=local(SAT, 9), end=local(SAT, 10)))
    calendar.add_busy(TimeSlot(start=local(SAT, 5), end=local(SAT, 6, 30)))
    calendar.add_busy(TimeSlot(start=local(SAT, 9, 30), end=local(SAT, 11)))
    morning = TimeSlot(start=local(SAT, 6), end=local(SAT, 10))

    busy = await calendar.busy_intervals(morning)

    assert [(b.start, b.end) for b in busy] == [(local(SAT, 6), local(SAT, 6, 30)), (local(SAT, 9), local(SAT, 10))]


async def test_insert_event_is_idempotent_by_id_and_recorded() -> None:
    calendar = InMemoryCalendar()
    run = TimeSlot(start=local(SUN, 6), end=local(SUN, 7))

    first = await calendar.insert_event(event_id="abcde12345", slot=run, title="Run: 10k Z2", description="plan")
    again = await calendar.insert_event(event_id="abcde12345", slot=run, title="Run: 10k Z2", description="plan")

    assert first == again == "abcde12345"
    assert calendar.insert_calls == 2
    assert [e.event_id for e in calendar.inserted] == ["abcde12345"]
    assert calendar.inserted[0].slot == run
    assert await calendar.busy_intervals(run) == [run]  # the run now blocks its own time


def test_inserted_is_a_copy() -> None:
    calendar = InMemoryCalendar()
    calendar.inserted.append(None)  # type: ignore[arg-type]
    assert calendar.inserted == []
