"""Run typing (interval structure / dominant HR zone) and cadence targets from the last N runs of a type.

Everything here is a pure function of (runs, now): no I/O, no clock, no globals. That keeps the data-science core
testable with hand-built activities and reproducible in a notebook.

Why classify from heart rate and laps instead of trusting a label: Garmin's "activity name" is free text (and private),
and its workout labels are only present for structured workouts. Time in the watch's own HR zones is present for
every run with a sensor, so it is the most reliable signal of what kind of run it *was*.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime

import numpy as np

from stridedj.domain import (
    Activity,
    ActivityId,
    CadenceTarget,
    ClassifiedRun,
    HeartRateZones,
    LapIntensity,
    RunningProfile,
    RunType,
)
from stridedj.runs.intervals import IntervalAnalysis

SECONDS_PER_DAY = 86_400.0

DEFAULT_ZONE_UPPER_BOUNDS_BPM: tuple[int, int, int, int] = (125, 145, 160, 175)
DEFAULT_HR_ZONES = HeartRateZones(upper_bounds_bpm=DEFAULT_ZONE_UPPER_BOUNDS_BPM)
"""Fallback zones (max HR ≈ 190) used when the athlete has not configured their watch zones."""

RIEGEL_EXPONENT = 1.06
"""Textbook Riegel exponent T2 = T1·(D2/D1)^1.06 (Riegel 1981). Used to put runs of different length on one scale."""

WORK_INTENSITIES = frozenset({LapIntensity.ACTIVE, LapIntensity.INTERVAL})
REST_INTENSITIES = frozenset({LapIntensity.REST, LapIntensity.RECOVERY})
MIN_WORK_BOUTS = 2
MIN_TREND_RUNS = 4

POPULATION_CADENCE_SPM: dict[RunType, float] = {
    RunType.Z1: 162.0,
    RunType.Z2: 166.0,
    RunType.Z3: 172.0,
    RunType.Z4: 176.0,
    RunType.Z5: 180.0,
    RunType.INTERVALS: 184.0,
}
"""Recreational-runner cadences by effort, used only when the user has no runs of that type (n_runs = 0)."""

_ZONE_RUN_TYPES = (RunType.Z1, RunType.Z2, RunType.Z3, RunType.Z4, RunType.Z5)


# ─────────────────────────────── small numeric helpers (shared with predict.py) ───────────────────────────────
def age_days(start: datetime, now: datetime) -> float:
    """How many days before `now` a run started (negative for runs after `now`, which callers must exclude)."""
    return (now - start).total_seconds() / SECONDS_PER_DAY


def ols_slope(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Ordinary least-squares slope of y on x; None when x has no spread (the slope is not identified)."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    xc = xa - xa.mean()
    denom = float(np.dot(xc, xc))
    if len(xa) < 2 or denom == 0.0:
        return None
    return float(np.dot(xc, ya - ya.mean()) / denom)


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    """Weighted empirical quantile with midpoint (Hazen) plotting positions, linearly interpolated.

    With equal weights this is the classic Hazen quantile; with recency weights a recent run counts more. Monotone in
    q, so p80 >= p50 always holds for the same sample.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if len(v) == 0 or len(v) != len(w) or float(w.sum()) <= 0.0:
        raise ValueError("weighted_quantile needs equally long, non-empty values/weights with positive total weight")
    order = np.argsort(v)
    v, w = v[order], w[order]
    positions = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, positions, v))


def recency_weight(age: float, half_life_days: float) -> float:
    """Exponential decay: a run `half_life_days` old counts half as much as one from today."""
    return float(0.5 ** (max(age, 0.0) / half_life_days))


# ─────────────────────────────── classification ───────────────────────────────
def count_work_bouts(activity: Activity) -> int:
    """Number of work bouts that are separated by rests/recoveries in the lap structure.

    Laps are reduced to a token sequence (W = active/interval, R = rest/recovery; warm-up, cool-down and "other" are
    ignored) and consecutive equal tokens are merged. Merging is what keeps a steady run with 1-km ACTIVE auto-laps at
    one bout: only a rest *between* two work blocks splits them.
    """
    symbols: list[str] = []
    for lap in activity.laps:
        if lap.intensity in WORK_INTENSITIES:
            symbol = "W"
        elif lap.intensity in REST_INTENSITIES:
            symbol = "R"
        else:
            continue
        if not symbols or symbols[-1] != symbol:
            symbols.append(symbol)
    return symbols.count("W")


def classify(activity: Activity, zones: HeartRateZones, intervals: IntervalAnalysis | None = None) -> ClassifiedRun:
    """Assign a run type, in decreasing order of evidence quality.

    0. The per-sample series (`runs/intervals.py`: pace, watch stops, heart rate), when stored, finds interval sessions
       that laps cannot (1-km auto-laps hide the reps and their rests).
    1. Lap structure with >= 2 work bouts alternating with rests → intervals. Checked first because the *average* HR of
       an interval session (warm-up, recoveries) usually lands in an easy zone, so zone time alone would mislabel it.
    2. Dominant zone by time in zone (`hr_zone_s`), as recorded by the watch.
    3. Average HR mapped through the user's zones (sources without zone times).
    4. No HR at all: assume an easy Z2 run (the most common run type), and say so in the reason.
    """
    if intervals is not None and intervals.is_intervals:
        return ClassifiedRun(
            activity=activity,
            run_type=RunType.INTERVALS,
            reason=intervals.reason,
            work_cadence_spm=intervals.work_cadence_spm,
        )
    bouts = count_work_bouts(activity)
    if bouts >= MIN_WORK_BOUTS:
        return ClassifiedRun(
            activity=activity, run_type=RunType.INTERVALS, reason=f"{bouts} work bouts alternating with recoveries"
        )
    if activity.hr_zone_s is not None and sum(activity.hr_zone_s) > 0:
        shares = np.asarray(activity.hr_zone_s, dtype=np.float64) / sum(activity.hr_zone_s)
        zone_index = int(np.argmax(shares))  # ties resolve to the lower (more conservative) zone
        return ClassifiedRun(
            activity=activity,
            run_type=_ZONE_RUN_TYPES[zone_index],
            reason=f"{shares[zone_index]:.0%} of time in Z{zone_index + 1}",
        )
    if activity.avg_hr_bpm is not None:
        zone = zones.zone_of(activity.avg_hr_bpm)
        return ClassifiedRun(
            activity=activity,
            run_type=_ZONE_RUN_TYPES[zone - 1],
            reason=f"avg HR {activity.avg_hr_bpm:.0f} bpm is in Z{zone}",
        )
    return ClassifiedRun(activity=activity, run_type=RunType.Z2, reason="no heart-rate data: assumed an easy Z2 run")


def classify_all(
    activities: Sequence[Activity],
    zones: HeartRateZones,
    intervals: Mapping[ActivityId, IntervalAnalysis] | None = None,
) -> list[ClassifiedRun]:
    found = intervals or {}
    return [classify(activity, zones, found.get(activity.activity_id)) for activity in activities]


# ─────────────────────────────── cadence ───────────────────────────────
def work_lap_cadence_spm(activity: Activity) -> float | None:
    """Duration-weighted cadence of the work laps only (None if no work lap has a cadence).

    For intervals the whole-run average mixes 185-spm work bouts with 155-spm recovery jogs; the music should match
    the work bouts, so only those count.
    """
    laps = [lap for lap in activity.laps if lap.intensity in WORK_INTENSITIES and lap.avg_cadence_spm is not None]
    if not laps:
        return None
    cadences = [lap.avg_cadence_spm for lap in laps if lap.avg_cadence_spm is not None]
    return float(np.average(cadences, weights=[lap.duration_s for lap in laps]))


def run_cadence_spm(run: ClassifiedRun) -> float | None:
    """The cadence that characterises a run of its type (work-lap cadence for intervals, else the run average)."""
    if run.run_type is RunType.INTERVALS:
        return run.work_cadence_spm or work_lap_cadence_spm(run.activity)
    return run.activity.avg_cadence_spm


def _runs_of_type(
    runs: Sequence[ClassifiedRun], run_type: RunType, *, now: datetime, lookback_days: float
) -> list[ClassifiedRun]:
    """Runs of one type that started strictly before `now` and within the lookback, newest first."""
    selected = [
        run for run in runs if run.run_type is run_type and 0.0 < age_days(run.activity.start, now) <= lookback_days
    ]
    return sorted(selected, key=lambda run: run.activity.start, reverse=True)


def _cadence_points(
    runs: Sequence[ClassifiedRun], run_type: RunType, *, now: datetime, lookback_days: int
) -> list[tuple[float, float]]:
    """(age_days, cadence_spm) for runs of the type with a usable cadence, newest first."""
    points: list[tuple[float, float]] = []
    for run in _runs_of_type(runs, run_type, now=now, lookback_days=lookback_days):
        spm = run_cadence_spm(run)
        if spm is not None:
            points.append((age_days(run.activity.start, now), spm))
    return points


def _trend_per_30d(ages: Sequence[float], values: Sequence[float]) -> float | None:
    """OLS slope of value over time, in units per 30 days (positive = rising). None with fewer than 4 runs."""
    if len(values) < MIN_TREND_RUNS:
        return None
    slope_per_day = ols_slope([-a for a in ages], values)  # time axis runs forward: older runs have smaller x
    return None if slope_per_day is None else slope_per_day * 30.0


def cadence_target(
    runs: Sequence[ClassifiedRun], run_type: RunType, *, now: datetime, n_last: int = 8, lookback_days: int = 120
) -> CadenceTarget:
    """Target cadence for a run type: the median of the last `n_last` runs of that type within the lookback.

    Why a median of the last few runs: cadence is a slowly drifting habit; the median ignores the odd hilly run or
    the run where the sensor glitched, and "last 8" adapts within ~2 months to a change in form. The trend is
    reported separately (OLS slope over the whole lookback) so the agent can mention it, not fold it in.
    """
    points = _cadence_points(runs, run_type, now=now, lookback_days=lookback_days)
    if not points:
        return CadenceTarget(
            run_type=run_type, spm=POPULATION_CADENCE_SPM[run_type], n_runs=0, lookback_days=lookback_days
        )
    recent = [spm for _, spm in points[:n_last]]
    return CadenceTarget(
        run_type=run_type,
        spm=float(statistics.median(recent)),
        n_runs=len(recent),
        lookback_days=lookback_days,
        trend_spm_per_30d=_trend_per_30d([a for a, _ in points], [spm for _, spm in points]),
    )


def cadence_target_ewma(
    runs: Sequence[ClassifiedRun],
    run_type: RunType,
    *,
    now: datetime,
    half_life_days: float = 45.0,
    lookback_days: int = 365,
) -> CadenceTarget:
    """Alternative estimator for the notebook: exponentially weighted mean (half-life 45 days) of all runs of a type.

    Trade-off vs. `cadence_target`: uses every run (lower variance) and adapts smoothly, but a mean is pulled by
    outliers (a sensor glitch at 120 spm) where the median of the last 8 is not.
    """
    points = _cadence_points(runs, run_type, now=now, lookback_days=lookback_days)
    if not points:
        return CadenceTarget(
            run_type=run_type, spm=POPULATION_CADENCE_SPM[run_type], n_runs=0, lookback_days=lookback_days
        )
    ages = [a for a, _ in points]
    values = [spm for _, spm in points]
    weights = [recency_weight(a, half_life_days) for a in ages]
    return CadenceTarget(
        run_type=run_type,
        spm=float(np.average(values, weights=weights)),
        n_runs=len(points),
        lookback_days=lookback_days,
        trend_spm_per_30d=_trend_per_30d(ages, values),
    )


# ─────────────────────────────── profile ───────────────────────────────
def normalised_pace_s_per_km(activity: Activity, *, reference_km: float = 10.0) -> float:
    """Pace scaled to a reference distance with the Riegel exponent (pace ∝ D^(b-1)).

    Without this, a month with more long runs (naturally slower per km) would look like lost fitness.
    """
    return activity.pace_s_per_km * float((reference_km / activity.distance_km) ** (RIEGEL_EXPONENT - 1.0))


def running_profile(
    runs: Sequence[ClassifiedRun], run_type: RunType, *, now: datetime, lookback_days: int = 120
) -> RunningProfile:
    """Aggregate view of the recent runs of one type: the only shape in which run history reaches the LLM."""
    recent = _runs_of_type(runs, run_type, now=now, lookback_days=lookback_days)
    all_before_now = _runs_of_type(runs, run_type, now=now, lookback_days=math.inf)
    last_days_ago = math.floor(age_days(all_before_now[0].activity.start, now)) if all_before_now else None
    if not recent:
        return RunningProfile(run_type=run_type, lookback_days=lookback_days, n_runs=0, last_run_days_ago=last_days_ago)

    cadences = [spm for spm in (run_cadence_spm(run) for run in recent) if spm is not None]
    ages = [age_days(run.activity.start, now) for run in recent]
    return RunningProfile(
        run_type=run_type,
        lookback_days=lookback_days,
        n_runs=len(recent),
        median_distance_km=float(statistics.median(run.activity.distance_km for run in recent)),
        median_pace_s_per_km=float(statistics.median(run.activity.pace_s_per_km for run in recent)),
        median_cadence_spm=float(statistics.median(cadences)) if cadences else None,
        last_run_days_ago=last_days_ago,
        pace_trend_s_per_km_per_30d=_trend_per_30d(ages, [normalised_pace_s_per_km(run.activity) for run in recent]),
    )
