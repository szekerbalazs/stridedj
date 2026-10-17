"""Calendar contracts. Only start/end times ever cross this boundary: event titles are private and untrusted."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, model_validator

from stridedj.domain.base import Frozen, SlotId


class DayPart(StrEnum):
    """Part of a day the user wants to run in; resolved to concrete local times in `cal/slots.py`."""

    MORNING = "morning"
    MIDDAY = "midday"
    EVENING = "evening"
    ANY = "any"


class TimeSlot(Frozen):
    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self

    @property
    def duration_min(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0

    def overlaps(self, other: TimeSlot) -> bool:
        return self.start < other.end and other.start < self.end


class SlotRequest(Frozen):
    """What the agent asks the calendar layer for: a date, a part of the day and a length."""

    day: date
    day_part: DayPart
    duration_min: int


class SlotOption(Frozen):
    """A free slot offered to the agent. The id is what `propose_plan` refers to."""

    slot_id: SlotId
    slot: TimeSlot
