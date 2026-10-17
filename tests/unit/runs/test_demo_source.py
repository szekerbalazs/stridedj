from __future__ import annotations

from collections import Counter
from datetime import UTC, date, datetime, timedelta

import pytest

from stridedj.domain import LapIntensity, RunType
from stridedj.runs.analysis import DEFAULT_HR_ZONES, classify_all, work_lap_cadence_spm
from stridedj.runs.sources.demo import DEMO_NOW, DEMO_TZ, HISTORY_START, DemoRunSource, generate_history


@pytest.fixture(scope="module")
def history() -> list:
    return generate_history()


def test_same_seed_same_history_different_seed_different_history() -> None:
    first = [a.model_dump() for a in generate_history(seed=7)]
    assert first == [a.model_dump() for a in generate_history(seed=7)]
    assert first != [a.model_dump() for a in generate_history(seed=8)]


def test_moving_now_forward_only_adds_runs(history: list) -> None:
    cutoff = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    earlier = generate_history(now=cutoff)
    finished_before = [a for a in history if a.start + timedelta(seconds=a.elapsed_s) <= cutoff]
    assert [a.model_dump() for a in earlier] == [a.model_dump() for a in finished_before]


def test_basic_invariants(history: list) -> None:
    assert 140 <= len(history) <= 190
    starts = [a.start for a in history]
    assert starts == sorted(starts, reverse=True)  # newest first
    assert len({a.activity_id for a in history}) == len(history)
    for activity in history:
        assert activity.start.utcoffset() == timedelta(0)
        assert activity.start.date() >= HISTORY_START
        assert activity.start + timedelta(seconds=activity.elapsed_s) <= DEMO_NOW
        assert activity.moving_s is not None
        assert activity.moving_s <= activity.elapsed_s
        assert activity.hr_zone_s is not None
        assert activity.avg_cadence_spm is not None
        assert activity.elevation_gain_m is not None
        assert 0.0 <= activity.elevation_gain_m / activity.distance_km <= 25.0


def test_three_to_five_runs_per_full_week(history: list) -> None:
    local_days = [a.start.astimezone(DEMO_TZ).date() for a in history]
    per_week = Counter(day - timedelta(days=day.weekday()) for day in local_days)
    last_full_week = DEMO_NOW.date() - timedelta(days=DEMO_NOW.weekday() + 7)
    assert all(3 <= n <= 5 for week, n in per_week.items() if week <= last_full_week)


def test_long_runs_only_in_spring_and_nothing_over_15km_recently(history: list) -> None:
    long_runs = [a for a in history if a.distance_km > 15.0]
    assert len(long_runs) >= 8
    assert all(date(2026, 2, 1) <= a.start.date() <= date(2026, 5, 31) for a in long_runs)
    assert max(a.distance_km for a in long_runs) <= 24.0
    assert any(a.start.month == 3 and a.distance_km == pytest.approx(21.1) for a in long_runs)
    four_months_ago = DEMO_NOW - timedelta(days=120)
    assert all(a.distance_km <= 15.0 for a in history if a.start >= four_months_ago)


def test_run_mix_and_hr_consistency(history: list) -> None:
    runs = classify_all(history, DEFAULT_HR_ZONES)
    mix = Counter(r.run_type for r in runs)
    assert mix[RunType.Z2] > mix[RunType.INTERVALS] > mix[RunType.Z4] > 0
    assert mix[RunType.Z1] > 0
    assert mix[RunType.Z3] > 0
    for run in runs:
        activity = run.activity
        assert activity.avg_hr_bpm is not None
        if run.run_type is RunType.INTERVALS:
            continue
        # steady runs: the average HR sits in the zone that dominates the time in zone
        assert DEFAULT_HR_ZONES.zone_of(activity.avg_hr_bpm) == int(run.run_type.value[1])


def test_interval_sessions_have_structured_laps(history: list) -> None:
    sessions = [r.activity for r in classify_all(history, DEFAULT_HR_ZONES) if r.run_type is RunType.INTERVALS]
    assert sessions
    for activity in sessions:
        assert activity.laps[0].intensity is LapIntensity.WARMUP
        assert activity.laps[-1].intensity is LapIntensity.COOLDOWN
        assert sum(lap.distance_m for lap in activity.laps) == pytest.approx(activity.distance_m, abs=1.0)
        offsets = [lap.start_offset_s for lap in activity.laps]
        assert offsets == sorted(offsets)
        work_cadence = work_lap_cadence_spm(activity)
        assert work_cadence is not None
        assert 180.0 <= work_cadence <= 190.0


def test_cadence_by_effort_matches_typical_ranges(history: list) -> None:
    runs = classify_all(history, DEFAULT_HR_ZONES)
    easy = [r.activity.avg_cadence_spm for r in runs if r.run_type is RunType.Z2 and r.activity.distance_km <= 14]
    threshold = [r.activity.avg_cadence_spm for r in runs if r.run_type is RunType.Z4]
    assert all(c is not None and 159.0 <= c <= 173.0 for c in easy)
    assert all(c is not None and 170.0 <= c <= 183.0 for c in threshold)


async def test_fetch_activities_filters_sorts_and_limits() -> None:
    source = DemoRunSource()
    assert source.name == "demo"
    since = DEMO_NOW - timedelta(days=30)
    recent = await source.fetch_activities(since=since, limit=1000)
    assert recent
    assert all(a.start >= since for a in recent)
    assert [a.start for a in recent] == sorted((a.start for a in recent), reverse=True)
    assert await source.fetch_activities(since=since, limit=3) == recent[:3]


def test_naive_now_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        DemoRunSource(now=datetime(2026, 10, 17, 18, 0))  # noqa: DTZ001
