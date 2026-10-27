"""Calendar contracts. Only start/end times ever cross this boundary: event titles are private and untrusted."""

from __future__ import annotations

from datetime import UTC, date
from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, Field, model_validator

from stridedj.domain.base import Frozen, SlotId


class DayPart(StrEnum):
    """Part of a day the user wants to run in; resolved to concrete local times in `cal/slots.py`."""

    MORNING = "morning"
    MIDDAY = "midday"
    EVENING = "evening"
    ANY = "any"


class Weekday(StrEnum):
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"

    @classmethod
    def of(cls, day: date) -> Weekday:
        return list(cls)[day.weekday()]


class TimeSlot(Frozen):
    start: AwareDatetime
    end: AwareDatetime

    # Comparisons and differences go through UTC: Python compares two datetimes that share a ZoneInfo by WALL CLOCK
    # (PEP 495), which is wrong across a DST change (e.g. the last Sunday of October in Europe).
    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end.astimezone(UTC) <= self.start.astimezone(UTC):
            raise ValueError("end must be after start")
        return self

    @property
    def duration_min(self) -> float:
        return (self.end.astimezone(UTC) - self.start.astimezone(UTC)).total_seconds() / 60.0

    def overlaps(self, other: TimeSlot) -> bool:
        s1, e1 = self.start.astimezone(UTC), self.end.astimezone(UTC)
        s2, e2 = other.start.astimezone(UTC), other.end.astimezone(UTC)
        return s1 < e2 and s2 < e1


class SlotRequest(Frozen):
    """What the agent asks the calendar layer for: a date, a part of the day and a length."""

    day: date
    day_part: DayPart
    duration_min: int = Field(gt=0)


class StartStatus(StrEnum):
    FREE = "free"
    BUSY = "busy"
    TOO_SOON = "too_soon"  # already passed, or less than the minimum lead time away


class StartCheck(Frozen):
    """The answer to "is exactly this start free?", with the nearest free starts that day when it is not."""

    slot: TimeSlot
    status: StartStatus
    alternatives: tuple[TimeSlot, ...] = ()


class SlotOption(Frozen):
    """A free slot offered to the agent. The id is what `propose_plan` refers to."""

    slot_id: SlotId
    slot: TimeSlot
