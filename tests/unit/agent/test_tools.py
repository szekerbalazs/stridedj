"""Tool contracts: argument validation and strict schemas (what the model can and must send)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stridedj.agent.tools import FindSlotsArgs, IntervalDistanceArgs
from stridedj.domain import DayPart, Weekday
from stridedj.llm.schema import strict_json_schema


def _slots(start_time: str | None) -> FindSlotsArgs:
    return FindSlotsArgs(
        weekday=Weekday.SUNDAY, date="2026-10-18", day_part=DayPart.ANY, duration_min=75, start_time=start_time
    )


@pytest.mark.parametrize(
    ("given", "stored"), [("8:00", "08:00"), ("08:00", "08:00"), (" 18:30 ", "18:30"), (None, None)]
)
def test_start_time_is_normalised(given: str | None, stored: str | None) -> None:
    assert _slots(given).start_time == stored


@pytest.mark.parametrize("bad", ["25:00", "8", "8.00", "08:60", "at 8", "8am"])
def test_start_time_must_be_a_24h_clock_time(bad: str) -> None:
    with pytest.raises(ValidationError, match="HH:MM"):
        _slots(bad)


def test_start_time_is_optional_for_code_but_required_and_nullable_for_the_model() -> None:
    assert (
        FindSlotsArgs(weekday=Weekday.SUNDAY, date="2026-10-18", day_part=DayPart.ANY, duration_min=75).start_time
        is None
    )
    schema = strict_json_schema(FindSlotsArgs)
    assert "start_time" in schema["required"]
    assert {"type": "null"} in schema["properties"]["start_time"]["anyOf"]


def test_interval_args_reject_kilometres_passed_as_metres() -> None:
    with pytest.raises(ValidationError, match="metres"):
        IntervalDistanceArgs(reps=5, rep_m=1, warmup_km=2, cooldown_km=1, recovery_m=None)


def test_weekday_is_a_required_closed_set_for_the_model() -> None:
    schema = strict_json_schema(FindSlotsArgs)
    assert "weekday" in schema["required"]
    assert schema["properties"]["weekday"]["enum"] == [d.value for d in Weekday]
