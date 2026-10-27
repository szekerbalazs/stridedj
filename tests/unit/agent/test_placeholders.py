"""Placeholders: the model names the facts of a proposal, code fills them in (or replaces a rule-breaking message)."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from stridedj.agent.placeholders import (
    DEFAULT_SESSION_OFFER,
    PROPOSAL_PLACEHOLDERS,
    problem,
    render_offer_message,
    render_proposal_message,
)
from stridedj.domain import (
    CadenceTarget,
    DurationEstimate,
    EstimateMethod,
    PlanId,
    RunPlan,
    RunType,
    TimeSlot,
    UserId,
)

TZ = ZoneInfo("Europe/Zurich")


def _plan() -> RunPlan:
    start = datetime(2026, 10, 28, 17, 30, tzinfo=UTC)  # 18:30 in Zurich
    return RunPlan(
        plan_id=PlanId("p1"),
        user_id=UserId("tg:1"),
        run_type=RunType.Z2,
        distance_km=10.0,
        duration=DurationEstimate(
            run_type=RunType.Z2,
            distance_km=10.0,
            method=EstimateMethod.RECENT_SIMILAR,
            n_runs_used=3,
            p50_min=64.7,
            p80_min=68.6,
            slot_min=75,
            note="test",
        ),
        slot=TimeSlot(start=start, end=datetime(2026, 10, 28, 18, 45, tzinfo=UTC)),
        cadence=CadenceTarget(run_type=RunType.Z2, spm=160, n_runs=8, lookback_days=120),
        playlist=None,
        created_at=start,
    )


def test_placeholders_are_filled_from_the_plan() -> None:
    text, replaced = render_proposal_message("Booked {start}, about {expected} ({p80} at most). {music}.", _plan(), TZ)
    assert text == "Booked Wed 28 Oct, 18:30, about 65 min (69 min at most). no playlist."
    assert replaced is None


def test_an_empty_session_leaves_no_gap() -> None:
    text, _ = render_proposal_message("Booked {start}: {session}.", _plan(), TZ)
    assert text == "Booked Wed 28 Oct, 18:30."


@pytest.mark.parametrize(
    "template",
    [
        "Your Z2 run is booked for {start}, about {expected}.",  # "Z2" is a zone name
        "Your 10 km Z2 run starts at 18:30 on {date}.",  # numbers that ARE the plan's facts
        "Dein 10-km-Lauf ist für {start} geplant, etwa {expected}.",
        "Your zone 2 run is booked for {start}.",  # the plan's zone, even if the runner wrote "Z2"
    ],
)
def test_numbers_that_are_facts_of_the_plan_are_accepted(template: str) -> None:
    assert render_proposal_message(template, _plan(), TZ)[1] is None


def test_numbers_the_runner_wrote_are_accepted_but_not_miscopied_ones() -> None:
    said = "5x1000 with 400 m jogs, zone 2, tomorrow evening"
    ok = "Your 5 × 1,000 m with 400 m jogs, zone 2, is booked for {start}."  # a thousands separator
    assert render_proposal_message(ok, _plan(), TZ, runner_said=said)[1] is None
    miscopied = "Your 6 × 1000 m is booked for {start}."
    assert render_proposal_message(miscopied, _plan(), TZ, runner_said=said)[1] == (
        "number(s) that are not facts of the plan: 6"
    )


@pytest.mark.parametrize(
    ("template", "reason"),
    [
        ("Booked for 18:00, about {expected}.", "number(s) that are not facts of the plan: 18:00"),
        ("Your 12 km run is booked for {start}.", "number(s) that are not facts of the plan: 12"),
        ("Booked {start} at {pace}.", "unknown placeholder(s) {pace}"),
    ],
)
def test_a_message_that_breaks_the_rule_is_replaced_by_the_standard_one(template: str, reason: str) -> None:
    text, replaced = render_proposal_message(template, _plan(), TZ)
    assert replaced == reason
    assert text == "Planned: 10 km Z2 easy on Wed 28 Oct, 18:30, about 65 min. no playlist."


def test_session_offers_carry_no_numbers() -> None:
    assert render_offer_message("Which session would you like?") == ("Which session would you like?", None)
    text, replaced = render_offer_message("Option 1 is 8 x 400 m, option 2 is 6 x 800 m.")
    assert (text, replaced) == (DEFAULT_SESSION_OFFER, "number(s) that are not facts of the plan: 1, 2, 400, 6, 8, 800")


def test_every_placeholder_has_a_value() -> None:
    from stridedj.agent.placeholders import plan_values

    assert set(plan_values(_plan(), TZ)) == set(PROPOSAL_PLACEHOLDERS)
    assert problem("{" + "}{".join(PROPOSAL_PLACEHOLDERS) + "}", PROPOSAL_PLACEHOLDERS) is None
