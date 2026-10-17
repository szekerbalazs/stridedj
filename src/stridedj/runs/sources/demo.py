"""Synthetic, seeded run history with a fitness trend: for tests, evals and the offline demo.

Why synthetic data: the agent, the evals and the notebook must run without anybody's Garmin account, and the data
must exercise every regime of the duration model. In particular the history has long runs (16–24 km, including a
21.1 km run in March) only in February–May and *nothing longer than 15 km from June on*, so "how long will a 21 km
easy run take?" in October must be answered by extrapolation, not by looking up similar recent runs.

Generative model (all constants below, one place to read it):

* fitness: the easy (Z2) pace at 10 km on flat ground improves 1 %/month from 6:10/km on 2026-01-05;
* run type: pace factor relative to that easy pace (Z1 1.08, Z3 0.90, Z4 0.85, interval work bouts ≈ 0.76);
* distance: pace ∝ (D / 10 km)^0.07, i.e. a Riegel-like time exponent of 1.07;
* elevation: 0–25 m gain per km, +0.35 % time per m/km;
* day-to-day form: log-normal noise (σ = 2.5 %);
* cadence: 166.5 + 18·(speed − easy speed) + 0.4 spm/month + N(0, 1) → easy ≈ 164–170, threshold ≈ 174–180,
  interval work bouts ≈ 182–188 spm;
* heart rate: inside the zone of the run type for the default zone bounds (125, 145, 160, 175); time in zone is
  dominated by that zone. Interval sessions get structured laps (warm-up / work / recovery / cool-down).

Determinism: each week's plan and each day's run draw from their own `numpy` generator seeded with
(seed, week, weekday). A run therefore depends only on the seed and its date, so moving `now` forward only *adds*
runs and never changes earlier ones (prefix stability), which is what a real source does too.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from stridedj.domain import Activity, ActivityId, Lap, LapIntensity, RunType
from stridedj.runs.analysis import DEFAULT_HR_ZONES

DEMO_TZ = ZoneInfo("Europe/Zurich")
HISTORY_START = date(2026, 1, 5)  # a Monday
DEMO_NOW = datetime(2026, 10, 17, 18, 0, tzinfo=DEMO_TZ)
DEFAULT_SEED = 2026

LONG_RUN_SEASON = (date(2026, 2, 1), date(2026, 5, 31))
HALF_MARATHON_DAY = date(2026, 3, 22)  # a Sunday in March: the one 21.1 km run
OFF_SEASON_MAX_KM = 15.0  # from June on, nothing longer than this

BASE_EASY_PACE_S_PER_KM = 370.0  # 6:10/km at 10 km, flat, on HISTORY_START
MONTHLY_PACE_IMPROVEMENT = 0.01
DAYS_PER_MONTH = 30.44
RIEGEL_EXPONENT = 1.07
ELEVATION_PENALTY_PER_M_PER_KM = 0.0035
PACE_NOISE_SIGMA = 0.025

PACE_FACTOR: dict[RunType, float] = {RunType.Z1: 1.08, RunType.Z2: 1.00, RunType.Z3: 0.90, RunType.Z4: 0.85}

CADENCE_BASE_SPM = 166.5
CADENCE_SPM_PER_M_PER_S = 18.0
CADENCE_TREND_SPM_PER_MONTH = 0.4

# typical average HR per run type (bpm); always inside the zone for the default bounds (125, 145, 160, 175)
HR_RANGE_BPM: dict[RunType, tuple[float, float]] = {
    RunType.Z1: (115.0, 123.0),
    RunType.Z2: (130.0, 141.0),
    RunType.Z3: (147.0, 157.0),
    RunType.Z4: (162.0, 171.0),
}
# share of moving time in Z1..Z5 per run type before noise; the run type's own zone always dominates
ZONE_SHARES: dict[RunType, tuple[float, float, float, float, float]] = {
    RunType.Z1: (0.78, 0.20, 0.02, 0.0, 0.0),
    RunType.Z2: (0.12, 0.76, 0.11, 0.01, 0.0),
    RunType.Z3: (0.06, 0.20, 0.64, 0.10, 0.0),
    RunType.Z4: (0.05, 0.12, 0.18, 0.60, 0.05),
}

# interval menus: (rep distance m, reps range, work pace factor, recovery seconds, recovery is a standing rest)
INTERVAL_MENU: tuple[tuple[float, tuple[int, int], float, float, bool], ...] = (
    (400.0, (8, 10), 0.73, 75.0, True),
    (800.0, (5, 6), 0.75, 120.0, False),
    (1000.0, (4, 6), 0.76, 120.0, False),
    (1600.0, (3, 4), 0.79, 180.0, False),
)

WEEKDAYS_BY_RUN_COUNT: dict[int, tuple[int, ...]] = {3: (1, 3, 6), 4: (1, 3, 5, 6), 5: (0, 1, 3, 5, 6)}
MONDAY, TUESDAY, THURSDAY, SUNDAY = 0, 1, 3, 6


@dataclass(frozen=True)
class _WeekPlan:
    weekdays: tuple[int, ...]
    has_intervals: bool
    tempo_type: RunType | None
    deload: bool


class DemoRunSource:
    """`RunSource` over a deterministic synthetic history from 2026-01-05 up to `now`.

    `now` is injected (default: 2026-10-17 18:00 Europe/Zurich, the fixed "today" of tests and evals); a run is only
    included once it has *finished* before `now`.
    """

    def __init__(self, *, now: datetime = DEMO_NOW, seed: int = DEFAULT_SEED) -> None:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        self._now = now
        self._seed = seed
        self._history: tuple[Activity, ...] | None = None

    @property
    def name(self) -> str:
        return "demo"

    def history(self) -> tuple[Activity, ...]:
        """The full synthetic history up to `now`, newest first (generated once, then cached)."""
        if self._history is None:
            self._history = tuple(generate_history(now=self._now, seed=self._seed))
        return self._history

    async def fetch_activities(self, *, since: datetime, limit: int) -> Sequence[Activity]:
        """Running activities started at or after `since`, newest first, at most `limit`."""
        return [activity for activity in self.history() if activity.start >= since][:limit]


# ─────────────────────────────── generation ───────────────────────────────
def generate_history(*, now: datetime = DEMO_NOW, seed: int = DEFAULT_SEED) -> list[Activity]:
    """All synthetic runs that finished before `now`, newest first."""
    activities: list[Activity] = []
    week = 0
    while (week_start := HISTORY_START + timedelta(weeks=week)) <= now.astimezone(DEMO_TZ).date():
        plan = _plan_week(seed, week)
        for weekday in plan.weekdays:
            day = week_start + timedelta(days=weekday)
            activity = _generate_run(seed, week, weekday, day, plan)
            if activity.start + timedelta(seconds=activity.elapsed_s) <= now:
                activities.append(activity)
        week += 1
    return sorted(activities, key=lambda a: a.start, reverse=True)


def _plan_week(seed: int, week: int) -> _WeekPlan:
    rng = np.random.default_rng((seed, week))
    n_runs = int(rng.choice([3, 4, 5], p=[0.3, 0.45, 0.25]))
    has_intervals = bool(rng.random() < 0.8)
    r = rng.random()
    tempo_type = RunType.Z3 if r < 0.3 else RunType.Z4 if r < 0.6 else None
    return _WeekPlan(WEEKDAYS_BY_RUN_COUNT[n_runs], has_intervals, tempo_type, deload=week % 4 == 3)


def _run_type_for(weekday: int, plan: _WeekPlan, rng: np.random.Generator) -> RunType | None:
    """None means "long run" (a Z2 run with its own distance rule)."""
    if weekday == SUNDAY:
        return None
    if weekday == TUESDAY and plan.has_intervals:
        return RunType.INTERVALS
    if weekday == THURSDAY and plan.tempo_type is not None:
        return plan.tempo_type
    recovery_probability = 0.6 if weekday == MONDAY else 0.3  # Monday follows the Sunday long run
    return RunType.Z1 if rng.random() < recovery_probability else RunType.Z2


def _in_long_run_season(day: date) -> bool:
    return LONG_RUN_SEASON[0] <= day <= LONG_RUN_SEASON[1]


def _distance_km(run_type: RunType | None, day: date, plan: _WeekPlan, rng: np.random.Generator) -> float:
    if run_type is None:  # Sunday long run
        if day == HALF_MARATHON_DAY:
            return 21.1
        km = rng.uniform(16.0, 24.0) if _in_long_run_season(day) and not plan.deload else rng.uniform(11.0, 15.0)
    elif run_type is RunType.Z1:
        km = rng.uniform(5.0, 8.0)
    elif run_type in (RunType.Z3, RunType.Z4):
        km = rng.uniform(6.0, 12.0)
    else:
        km = rng.uniform(6.0, 14.0)
    if plan.deload:
        km *= 0.85
    if day > LONG_RUN_SEASON[1]:
        km = min(km, OFF_SEASON_MAX_KM)
    return round(float(km), 2)


def _start_time(day: date, weekday: int, rng: np.random.Generator) -> datetime:
    if weekday >= 5:  # weekend: mid-morning
        minutes = int(rng.integers(8 * 60 + 30, 10 * 60 + 30))
    elif rng.random() < 0.3:  # early morning
        minutes = int(rng.integers(6 * 60 + 15, 7 * 60 + 30))
    else:  # after work
        minutes = int(rng.integers(17 * 60 + 45, 19 * 60 + 30))
    local = datetime.combine(day, time(minutes // 60, minutes % 60), tzinfo=DEMO_TZ)
    return local.astimezone(UTC)


def _months_since_start(day: date) -> float:
    return (day - HISTORY_START).days / DAYS_PER_MONTH


def _easy_pace_s_per_km(day: date) -> float:
    """Fitness trend: the flat 10 km Z2 pace improves 1 % per month."""
    return BASE_EASY_PACE_S_PER_KM * float((1.0 - MONTHLY_PACE_IMPROVEMENT) ** _months_since_start(day))


def _cadence_spm(pace_s_per_km: float, day: date, rng: np.random.Generator, *, sigma: float = 1.0) -> float:
    """Cadence rises with speed relative to today's easy speed, plus a slow technique trend."""
    speed_m_per_s = 1000.0 / pace_s_per_km
    easy_speed_m_per_s = 1000.0 / _easy_pace_s_per_km(day)
    trend = CADENCE_TREND_SPM_PER_MONTH * _months_since_start(day)
    noise = float(rng.normal(0.0, sigma))
    return round(CADENCE_BASE_SPM + CADENCE_SPM_PER_M_PER_S * (speed_m_per_s - easy_speed_m_per_s) + trend + noise, 1)


def _elevation_per_km(rng: np.random.Generator) -> float:
    r = rng.random()
    if r < 0.6:
        return float(rng.uniform(0.0, 8.0))
    if r < 0.9:
        return float(rng.uniform(8.0, 18.0))
    return float(rng.uniform(18.0, 25.0))


def _zone_seconds(run_type: RunType, moving_s: float, rng: np.random.Generator) -> tuple[float, ...]:
    base = np.asarray(ZONE_SHARES[run_type], dtype=np.float64)
    noisy = base * np.exp(rng.normal(0.0, 0.15, size=5))
    shares = noisy / noisy.sum()
    return tuple(round(float(s * moving_s), 1) for s in shares)


def _generate_run(seed: int, week: int, weekday: int, day: date, plan: _WeekPlan) -> Activity:
    rng = np.random.default_rng((seed, week, weekday))
    run_type = _run_type_for(weekday, plan, rng)
    start = _start_time(day, weekday, rng)
    activity_id = ActivityId(f"demo-{day:%Y%m%d}")
    if run_type is RunType.INTERVALS:
        return _interval_session(activity_id, start, day, rng)
    distance_km = _distance_km(run_type, day, plan, rng)
    return _steady_run(activity_id, start, day, run_type or RunType.Z2, distance_km, rng)


def _steady_run(
    activity_id: ActivityId, start: datetime, day: date, run_type: RunType, distance_km: float, rng: np.random.Generator
) -> Activity:
    elevation_per_km = _elevation_per_km(rng)
    pace = (
        _easy_pace_s_per_km(day)
        * PACE_FACTOR[run_type]
        * (distance_km / 10.0) ** (RIEGEL_EXPONENT - 1.0)
        * (1.0 + ELEVATION_PENALTY_PER_M_PER_KM * elevation_per_km)
        * math.exp(rng.normal(0.0, PACE_NOISE_SIGMA))
    )
    moving_s = round(pace * distance_km, 1)
    pauses_s = float(rng.uniform(0.0, 90.0)) + (float(rng.uniform(180.0, 360.0)) if rng.random() < 0.05 else 0.0)
    lo, hi = HR_RANGE_BPM[run_type]
    drift_bpm = 3.0 if distance_km > 15.0 else 0.0  # cardiac drift on long runs, still inside the zone
    avg_hr = min(float(rng.uniform(lo, hi)) + drift_bpm, hi + 2.0)
    zone_s = _zone_seconds(run_type, moving_s, rng)
    return Activity(
        activity_id=activity_id,
        source="demo",
        start=start,
        distance_m=round(distance_km * 1000.0, 1),
        elapsed_s=round(moving_s + pauses_s, 1),
        moving_s=moving_s,
        elevation_gain_m=round(elevation_per_km * distance_km, 1),
        avg_hr_bpm=round(avg_hr, 1),
        max_hr_bpm=round(avg_hr + float(rng.uniform(6.0, 14.0)), 1),
        avg_cadence_spm=_cadence_spm(pace, day, rng),
        hr_zone_s=(zone_s[0], zone_s[1], zone_s[2], zone_s[3], zone_s[4]),
    )


@dataclass
class _LapBuilder:
    """Accumulates laps of a structured session and the totals derived from them."""

    laps: list[Lap]
    offset_s: float = 0.0

    def add(
        self, *, distance_m: float, duration_s: float, hr: float, cadence: float | None, kind: LapIntensity
    ) -> None:
        self.laps.append(
            Lap(
                start_offset_s=round(self.offset_s, 1),
                duration_s=round(duration_s, 1),
                distance_m=round(distance_m, 1),
                avg_hr_bpm=round(hr, 1),
                avg_cadence_spm=cadence,
                intensity=kind,
            )
        )
        self.offset_s += duration_s


def _interval_session(activity_id: ActivityId, start: datetime, day: date, rng: np.random.Generator) -> Activity:
    """Warm-up, N work bouts with recoveries (jog or standing rest), cool-down; totals are derived from the laps."""
    easy = _easy_pace_s_per_km(day)
    rep_m, (min_reps, max_reps), work_factor, recovery_s, standing_rest = INTERVAL_MENU[
        int(rng.integers(0, len(INTERVAL_MENU)))
    ]
    reps = int(rng.integers(min_reps, max_reps + 1))
    work_kind = LapIntensity.INTERVAL if rng.random() < 0.5 else LapIntensity.ACTIVE
    builder = _LapBuilder(laps=[])

    warmup_m = float(rng.uniform(2000.0, 3000.0))
    warmup_pace = easy * 1.03
    builder.add(
        distance_m=warmup_m,
        duration_s=warmup_pace * warmup_m / 1000.0,
        hr=float(rng.uniform(128.0, 136.0)),
        cadence=_cadence_spm(warmup_pace, day, rng),
        kind=LapIntensity.WARMUP,
    )
    for rep in range(reps):
        work_pace = easy * work_factor * math.exp(rng.normal(0.0, 0.015))
        builder.add(
            distance_m=rep_m,
            duration_s=work_pace * rep_m / 1000.0,
            hr=min(164.0 + 2.0 * rep + float(rng.uniform(0.0, 4.0)), 186.0),
            cadence=_cadence_spm(work_pace, day, rng, sigma=0.8),
            kind=work_kind,
        )
        if rep == reps - 1:
            break
        if standing_rest:
            builder.add(
                distance_m=float(rng.uniform(0.0, 30.0)),
                duration_s=recovery_s,
                hr=float(rng.uniform(138.0, 146.0)),
                cadence=None,
                kind=LapIntensity.REST,
            )
        else:
            jog_pace = easy * 1.25
            builder.add(
                distance_m=1000.0 * recovery_s / jog_pace,
                duration_s=recovery_s,
                hr=float(rng.uniform(146.0, 153.0)),
                cadence=_cadence_spm(jog_pace, day, rng),
                kind=LapIntensity.RECOVERY,
            )
    cooldown_m = float(rng.uniform(1500.0, 2500.0))
    cooldown_pace = easy * 1.05
    builder.add(
        distance_m=cooldown_m,
        duration_s=cooldown_pace * cooldown_m / 1000.0,
        hr=float(rng.uniform(134.0, 142.0)),
        cadence=_cadence_spm(cooldown_pace, day, rng),
        kind=LapIntensity.COOLDOWN,
    )
    return _activity_from_laps(activity_id, start, tuple(builder.laps), rng)


def _activity_from_laps(
    activity_id: ActivityId, start: datetime, laps: tuple[Lap, ...], rng: np.random.Generator
) -> Activity:
    durations = np.asarray([lap.duration_s for lap in laps], dtype=np.float64)
    hrs = np.asarray([lap.avg_hr_bpm or 0.0 for lap in laps], dtype=np.float64)
    moving = [lap for lap in laps if lap.intensity is not LapIntensity.REST]
    cadence_laps = [lap for lap in laps if lap.avg_cadence_spm is not None]
    distance_m = sum(lap.distance_m for lap in laps)
    moving_s = sum(lap.duration_s for lap in moving)
    zone_s = [0.0] * 5
    for lap in laps:
        zone_s[DEFAULT_HR_ZONES.zone_of(lap.avg_hr_bpm or 0.0) - 1] += lap.duration_s
    return Activity(
        activity_id=activity_id,
        source="demo",
        start=start,
        distance_m=round(distance_m, 1),
        elapsed_s=round(float(durations.sum()) + float(rng.uniform(0.0, 60.0)), 1),
        moving_s=round(moving_s, 1),
        elevation_gain_m=round(float(rng.uniform(0.0, 8.0)) * distance_m / 1000.0, 1),
        avg_hr_bpm=round(float(np.average(hrs, weights=durations)), 1),
        max_hr_bpm=round(float(hrs.max()) + float(rng.uniform(2.0, 5.0)), 1),
        avg_cadence_spm=round(
            float(
                np.average(
                    [lap.avg_cadence_spm for lap in cadence_laps if lap.avg_cadence_spm is not None],
                    weights=[lap.duration_s for lap in cadence_laps],
                )
            ),
            1,
        ),
        hr_zone_s=(zone_s[0], zone_s[1], zone_s[2], zone_s[3], zone_s[4]),
        laps=laps,
    )
