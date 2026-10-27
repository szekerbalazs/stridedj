from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from stridedj.domain import (
    Activity,
    ActivityId,
    AgentReply,
    CadenceTarget,
    DurationEstimate,
    EstimateMethod,
    HeartRateZones,
    PlanId,
    ReplyKind,
    RunPlan,
    RunType,
    TimeSlot,
    UserId,
)

T0 = datetime(2026, 10, 17, 18, 0, tzinfo=UTC)


def _estimate(**overrides: object) -> DurationEstimate:
    base: dict[str, object] = {
        "run_type": RunType.Z2,
        "distance_km": 10.0,
        "method": EstimateMethod.RECENT_SIMILAR,
        "n_runs_used": 6,
        "p50_min": 55.0,
        "p80_min": 58.0,
        "slot_min": 75,
        "note": "from 6 recent runs",
    }
    base.update(overrides)
    return DurationEstimate.model_validate(base)


def test_duration_estimate_invariants() -> None:
    assert _estimate().slot_min == 75
    with pytest.raises(ValidationError, match="multiple of 5"):
        _estimate(slot_min=72)
    with pytest.raises(ValidationError, match="p80 must be >= p50"):
        _estimate(p50_min=60.0, p80_min=58.0)
    with pytest.raises(ValidationError, match="cover p80"):
        _estimate(p80_min=80.0, slot_min=75)


def test_models_are_frozen_and_reject_unknown_fields() -> None:
    est = _estimate()
    with pytest.raises(ValidationError):
        est.p50_min = 1.0  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Activity.model_validate(
            {
                "activity_id": "1",
                "source": "demo",
                "start": T0,
                "distance_m": 10_000,
                "elapsed_s": 3300,
                "startLatitude": 47.37,  # GPS must never be accepted
            }
        )


def test_time_slot_requires_aware_and_ordered() -> None:
    with pytest.raises(ValidationError):
        TimeSlot(start=datetime(2026, 10, 17, 18, 0), end=datetime(2026, 10, 17, 19, 0))  # noqa: DTZ001
    with pytest.raises(ValidationError, match="end must be after start"):
        TimeSlot(start=T0, end=T0)
    slot = TimeSlot(start=T0, end=T0 + timedelta(minutes=75))
    assert slot.duration_min == 75
    assert slot.overlaps(TimeSlot(start=T0 + timedelta(minutes=60), end=T0 + timedelta(minutes=90)))
    assert not slot.overlaps(TimeSlot(start=T0 + timedelta(minutes=75), end=T0 + timedelta(minutes=90)))


@pytest.mark.parametrize(("hr", "zone"), [(100, 1), (124.9, 1), (125, 2), (159, 3), (174, 4), (175, 5), (200, 5)])
def test_zone_of(hr: float, zone: int) -> None:
    assert HeartRateZones(upper_bounds_bpm=(125, 145, 160, 175)).zone_of(hr) == zone


def test_activity_pace_uses_moving_time() -> None:
    a = Activity(activity_id=ActivityId("1"), source="demo", start=T0, distance_m=10_000, elapsed_s=3600, moving_s=3300)
    assert a.distance_km == 10
    assert a.pace_s_per_km == pytest.approx(330)


def test_agent_reply_length_limit() -> None:
    ok = AgentReply(kind=ReplyKind.ANSWER, message="fine", missing_fields=[])
    assert ok.kind is ReplyKind.ANSWER
    with pytest.raises(ValidationError, match="longer than"):
        AgentReply(kind=ReplyKind.ANSWER, message="x" * 801, missing_fields=[])


def test_plan_hash_changes_with_content() -> None:
    plan = RunPlan(
        plan_id=PlanId("p1"),
        user_id=UserId("u1"),
        run_type=RunType.Z2,
        distance_km=10,
        duration=_estimate(),
        slot=TimeSlot(start=T0, end=T0 + timedelta(minutes=75)),
        cadence=CadenceTarget(run_type=RunType.Z2, spm=168, n_runs=8, lookback_days=120),
        playlist=None,
        created_at=T0,
    )
    same = plan.model_copy()
    moved = plan.model_copy(update={"slot": TimeSlot(start=T0 + timedelta(hours=1), end=T0 + timedelta(minutes=135))})
    assert plan.content_hash() == same.content_hash()
    assert plan.content_hash() != moved.content_hash()


def test_time_slot_across_dst_change_uses_real_elapsed_time() -> None:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Europe/Zurich")
    start = datetime(2026, 10, 25, 1, 0, tzinfo=tz)  # CEST (UTC+2)
    end = datetime(2026, 10, 25, 4, 0, tzinfo=tz)  # CET (UTC+1): 4 hours later, not 3
    assert TimeSlot(start=start, end=end).duration_min == 240
