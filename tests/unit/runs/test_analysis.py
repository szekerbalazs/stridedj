from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from stridedj.domain import Activity, ActivityId, ClassifiedRun, Lap, LapIntensity, RunType
from stridedj.runs.analysis import (
    DEFAULT_HR_ZONES,
    POPULATION_CADENCE_SPM,
    cadence_target,
    cadence_target_ewma,
    classify,
    classify_all,
    count_work_bouts,
    ols_slope,
    running_profile,
    weighted_quantile,
    work_lap_cadence_spm,
)
from stridedj.runs.sources.demo import DEMO_NOW, generate_history

NOW = datetime(2026, 10, 17, 16, 0, tzinfo=UTC)


def make_activity(
    activity_id: str = "a1",
    *,
    days_ago: float = 1.0,
    km: float = 10.0,
    minutes: float = 60.0,
    hr: float | None = None,
    zones: tuple[float, float, float, float, float] | None = None,
    cadence: float | None = None,
    laps: Sequence[Lap] = (),
) -> Activity:
    return Activity(
        activity_id=ActivityId(activity_id),
        source="demo",
        start=NOW - timedelta(days=days_ago),
        distance_m=km * 1000.0,
        elapsed_s=minutes * 60.0,
        moving_s=minutes * 60.0,
        avg_hr_bpm=hr,
        avg_cadence_spm=cadence,
        hr_zone_s=zones,
        laps=tuple(laps),
    )


def lap(kind: LapIntensity, *, offset: float = 0.0, seconds: float = 120.0, cadence: float | None = None) -> Lap:
    return Lap(start_offset_s=offset, duration_s=seconds, distance_m=400.0, avg_cadence_spm=cadence, intensity=kind)


def interval_laps(work_cadence: float = 186.0, rest_cadence: float = 150.0, reps: int = 4) -> list[Lap]:
    laps = [lap(LapIntensity.WARMUP, seconds=600.0, cadence=165.0)]
    for i in range(reps):
        laps.append(lap(LapIntensity.INTERVAL, seconds=180.0, cadence=work_cadence))
        if i < reps - 1:
            laps.append(lap(LapIntensity.RECOVERY, seconds=90.0, cadence=rest_cadence))
    laps.append(lap(LapIntensity.COOLDOWN, seconds=500.0, cadence=163.0))
    return laps


def classified(activity: Activity, run_type: RunType) -> ClassifiedRun:
    return ClassifiedRun(activity=activity, run_type=run_type, reason="test")


# ─────────────────────────────── classification ───────────────────────────────
def test_lap_structure_with_alternating_work_and_rest_is_intervals() -> None:
    # zone time says "mostly Z2" (warm-up, recoveries), the laps say intervals: structure wins
    run = classify(make_activity(zones=(300, 1500, 600, 500, 100), laps=interval_laps(reps=4)), DEFAULT_HR_ZONES)
    assert run.run_type is RunType.INTERVALS
    assert run.reason == "4 work bouts alternating with recoveries"


def test_active_auto_laps_without_rests_are_not_intervals() -> None:
    auto_laps = [lap(LapIntensity.ACTIVE, offset=i * 360.0, seconds=360.0) for i in range(10)]
    activity = make_activity(zones=(100, 2800, 300, 0, 0), laps=auto_laps)
    assert count_work_bouts(activity) == 1
    assert classify(activity, DEFAULT_HR_ZONES).run_type is RunType.Z2


def test_single_work_bout_with_rest_is_not_intervals() -> None:
    laps = [lap(LapIntensity.WARMUP), lap(LapIntensity.ACTIVE), lap(LapIntensity.REST), lap(LapIntensity.COOLDOWN)]
    assert count_work_bouts(make_activity(laps=laps)) == 1


def test_dominant_zone_by_time_in_zone_with_reason() -> None:
    run = classify(make_activity(zones=(300.0, 2808.0, 400.0, 92.0, 0.0), hr=150.0), DEFAULT_HR_ZONES)
    assert run.run_type is RunType.Z2  # zone time beats the average HR (150 bpm would be Z3)
    assert run.reason == "78% of time in Z2"


def test_falls_back_to_average_hr_without_zone_times() -> None:
    run = classify(make_activity(hr=150.0), DEFAULT_HR_ZONES)
    assert run.run_type is RunType.Z3
    assert run.reason == "avg HR 150 bpm is in Z3"
    assert classify(make_activity(hr=176.0), DEFAULT_HR_ZONES).run_type is RunType.Z5


def test_all_zero_zone_times_fall_back_to_average_hr() -> None:
    assert classify(make_activity(zones=(0, 0, 0, 0, 0), hr=120.0), DEFAULT_HR_ZONES).run_type is RunType.Z1


def test_without_any_hr_data_assumes_easy_run() -> None:
    run = classify(make_activity(), DEFAULT_HR_ZONES)
    assert run.run_type is RunType.Z2
    assert "no heart-rate data" in run.reason


# ─────────────────────────────── cadence ───────────────────────────────
def test_cadence_target_is_median_of_last_n_runs_of_the_type() -> None:
    cadences = [160, 161, 162, 163, 164, 165, 166, 167, 168, 169]  # index 0 is the newest run
    runs = [
        classified(make_activity(f"r{i}", days_ago=1 + 3 * i, cadence=c), RunType.Z2) for i, c in enumerate(cadences)
    ]
    runs.append(classified(make_activity("other", days_ago=2, cadence=190.0), RunType.Z4))  # other type: ignored
    target = cadence_target(runs, RunType.Z2, now=NOW, n_last=8)
    assert target.n_runs == 8
    assert target.spm == pytest.approx(163.5)  # median of 160..167
    assert target.trend_spm_per_30d is not None
    assert target.trend_spm_per_30d < 0  # newer runs have lower cadence in this fixture


def test_cadence_target_for_intervals_uses_work_laps_only() -> None:
    activity = make_activity(cadence=168.0, laps=interval_laps(work_cadence=186.0, rest_cadence=150.0))
    assert work_lap_cadence_spm(activity) == pytest.approx(186.0)
    target = cadence_target([classified(activity, RunType.INTERVALS)], RunType.INTERVALS, now=NOW)
    assert target.spm == pytest.approx(186.0)
    assert target.n_runs == 1
    assert target.trend_spm_per_30d is None  # fewer than 4 runs


def test_cadence_target_population_default_without_data() -> None:
    old = classified(make_activity(days_ago=200, cadence=170.0), RunType.Z2)  # outside the 120-day lookback
    future = classified(make_activity(days_ago=-1, cadence=170.0), RunType.Z2)  # after `now`: never used
    for runs in ([], [old, future]):
        target = cadence_target(runs, RunType.Z2, now=NOW)
        assert target.n_runs == 0
        assert target.spm == POPULATION_CADENCE_SPM[RunType.Z2] == 166.0
        assert target.trend_spm_per_30d is None
    assert cadence_target([], RunType.INTERVALS, now=NOW).spm == 184.0


def test_cadence_trend_is_positive_on_demo_history() -> None:
    runs = classify_all(generate_history(), DEFAULT_HR_ZONES)
    for run_type in (RunType.Z2, RunType.INTERVALS):
        target = cadence_target(runs, run_type, now=DEMO_NOW)
        assert target.n_runs == 8
        assert target.trend_spm_per_30d is not None
        assert target.trend_spm_per_30d > 0
    assert 165.0 <= cadence_target(runs, RunType.Z2, now=DEMO_NOW).spm <= 171.0
    assert 183.0 <= cadence_target(runs, RunType.INTERVALS, now=DEMO_NOW).spm <= 190.0


def test_ewma_cadence_weights_recent_runs_more() -> None:
    runs = [
        classified(make_activity("new", days_ago=1, cadence=170.0), RunType.Z2),
        classified(make_activity("old", days_ago=91, cadence=160.0), RunType.Z2),  # two half-lives older
    ]
    target = cadence_target_ewma(runs, RunType.Z2, now=NOW, half_life_days=45.0)
    assert target.n_runs == 2
    assert target.spm == pytest.approx((170.0 * 1.0 + 160.0 * 0.25) / 1.25, rel=1e-3)
    assert cadence_target_ewma([], RunType.Z3, now=NOW).n_runs == 0


# ─────────────────────────────── profile ───────────────────────────────
def test_running_profile_aggregates() -> None:
    runs = [
        classified(make_activity("a", days_ago=3, km=10.0, minutes=60.0, cadence=166.0), RunType.Z2),
        classified(make_activity("b", days_ago=10, km=8.0, minutes=50.0, cadence=164.0), RunType.Z2),
        classified(make_activity("c", days_ago=20, km=12.0, minutes=75.0), RunType.Z2),
    ]
    profile = running_profile(runs, RunType.Z2, now=NOW)
    assert profile.n_runs == 3
    assert profile.median_distance_km == pytest.approx(10.0)
    assert profile.median_pace_s_per_km == pytest.approx(375.0)
    assert profile.median_cadence_spm == pytest.approx(165.0)
    assert profile.last_run_days_ago == 3
    assert profile.pace_trend_s_per_km_per_30d is None  # fewer than 4 runs

    empty = running_profile(runs, RunType.Z4, now=NOW)
    assert empty.n_runs == 0
    assert empty.median_pace_s_per_km is None


def test_profile_on_demo_history_shows_improving_pace() -> None:
    runs = classify_all(generate_history(), DEFAULT_HR_ZONES)
    profile = running_profile(runs, RunType.Z2, now=DEMO_NOW)
    assert profile.n_runs > 20
    assert profile.last_run_days_ago is not None
    assert profile.last_run_days_ago <= 7
    assert profile.pace_trend_s_per_km_per_30d is not None
    assert profile.pace_trend_s_per_km_per_30d < 0  # negative = getting faster


# ─────────────────────────────── helpers ───────────────────────────────
def test_weighted_quantile_matches_hazen_and_respects_weights() -> None:
    assert weighted_quantile([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1], 0.5) == pytest.approx(2.5)
    assert weighted_quantile([10.0, 20.0], [1.0, 0.0001], 0.5) == pytest.approx(10.0, abs=0.01)
    assert weighted_quantile([5.0, 1.0, 3.0], [1, 1, 1], 0.8) >= weighted_quantile([5.0, 1.0, 3.0], [1, 1, 1], 0.5)
    with pytest.raises(ValueError):
        weighted_quantile([], [], 0.5)


def test_ols_slope() -> None:
    assert ols_slope([0, 1, 2, 3], [1, 3, 5, 7]) == pytest.approx(2.0)
    assert ols_slope([1, 1, 1], [1, 2, 3]) is None
