"""Interval sessions: the total distance from its parts, and session options built from the runner's targets.

Arithmetic stays in code, not in the LLM:

* `interval_session_distance`: "4x800, 2 km warm-up, 1 km cool-down" → total. Recoveries are standing rests (0 m)
  unless the runner names a jog. The assumption is reported back
  (`recovery_assumed`) so the agent can say it.
* `build_session_options`: 2-3 sessions (short reps, medium reps, a pyramid) with target paces per rep distance
  (`runs/targets.py`), standing rests and an expected duration from the session's own structure.

Plans always use distance reps (the runner's preference), never time reps.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

from stridedj.domain import DurationEstimate, EstimateMethod, IntervalSession, IntervalSessionDistance, RunType
from stridedj.domain.planning import format_pace
from stridedj.runs.predict import slot_minutes

DEFAULT_RECOVERY_M = 0.0
"""Recovery between repetitions when the runner gives none: a standing rest."""

DEFAULT_WARMUP_KM = 2.0
DEFAULT_COOLDOWN_KM = 1.5
REST_S_BY_REP_M: tuple[tuple[int, int], ...] = ((400, 75), (800, 90), (100_000, 120))  # planned standing rests
SHORT_REP_M, MEDIUM_REP_M = 400, 800
DEFAULT_REPS: Mapping[int, int] = {SHORT_REP_M: 8, MEDIUM_REP_M: 6}
PYRAMID_M: tuple[int, ...] = (400, 800, 1200, 800, 400)
MIN_REPS = 3
MIN_EASY_KM = 1.5  # warm-up + cool-down at least, when sized to a total
WARMUP_SHARE = 0.6  # of the easy distance, when sized to a total


def interval_session_distance(
    *, reps: int, rep_m: float, warmup_km: float, cooldown_km: float, recovery_m: float | None = None
) -> IntervalSessionDistance:
    """Total = warm-up + reps × rep + (reps − 1) × recovery + cool-down (no recovery after the last repetition)."""
    if reps < 1 or rep_m <= 0:
        raise ValueError("reps must be >= 1 and rep_m > 0")
    if warmup_km < 0 or cooldown_km < 0 or (recovery_m is not None and recovery_m < 0):
        raise ValueError("warm-up, cool-down and recovery must be >= 0")
    assumed = recovery_m is None
    recovery_each_m = DEFAULT_RECOVERY_M if recovery_m is None else recovery_m
    work_km = reps * rep_m / 1000.0
    recovery_km = (reps - 1) * recovery_each_m / 1000.0
    return IntervalSessionDistance(
        total_km=round(warmup_km + work_km + recovery_km + cooldown_km, 2),
        warmup_km=warmup_km,
        work_km=round(work_km, 2),
        recovery_km=round(recovery_km, 2),
        cooldown_km=cooldown_km,
        recovery_assumed=assumed,
    )


def rest_after(rep_m: int) -> int:
    return next(rest for limit, rest in REST_S_BY_REP_M if rep_m <= limit)


def build_session_options(
    *,
    pace_of: Callable[[int], float],
    easy_pace_s_per_km: float,
    overhead_s: float,
    spread: float,
    buffer_min: float,
    n_history: int,
    measured_rest_s: Callable[[int], float] = rest_after,
    typical_reps: Mapping[int, int] | None = None,
    warmup_km: float = DEFAULT_WARMUP_KM,
    cooldown_km: float = DEFAULT_COOLDOWN_KM,
    max_minutes: int | None = None,
    total_km: float | None = None,
) -> list[IntervalSession]:
    """Short reps, medium reps and a pyramid. With `max_minutes`, rep counts shrink (to ≥ 3) until the slot fits;
    options that still don't fit are left out. The label shows the planned rest; the duration uses the rest the runner
    actually takes (`measured_rest_s`: usually longer than planned).

    With `total_km` ("9 km in total"), each option keeps its reps and the rest of the distance becomes warm-up and
    cool-down (60/40); an option whose reps leave less than 1.5 km for them gets fewer reps (≥ 3) or is left out."""
    reps_of = {**DEFAULT_REPS, **(typical_reps or {})}

    def make(kind: str, reps: tuple[int, ...]) -> IntervalSession | None:
        warm, cool = warmup_km, cooldown_km
        if total_km is not None:
            easy_km = total_km - sum(reps) / 1000
            if easy_km < MIN_EASY_KM - 1e-9:
                return None
            warm = round(easy_km * WARMUP_SHARE, 1)
            cool = round(easy_km - warm, 1)
        paces = tuple(round(pace_of(r), 1) for r in reps)
        rests = tuple(rest_after(r) for r in reps[:-1])
        work_s = sum(r / 1000 * p for r, p in zip(reps, paces, strict=True))
        easy_s = (warm + cool) * easy_pace_s_per_km
        rests_s = sum(measured_rest_s(r) for r in reps[:-1])
        p50_min = (easy_s + work_s + rests_s + overhead_s) / 60
        p80_min = p50_min * spread
        total = round(warm + sum(reps) / 1000 + cool, 2)
        note = (
            f"From the session's structure: warm-up and cool-down at your easy pace ({format_pace(easy_pace_s_per_km)}"
            f"/km), reps at target pace, your usual standing rests, + {overhead_s / 60:.0f} min of usual extra time "
            f"(from {n_history} past sessions); 80% adds your usual spread (+{spread - 1:.0%})."
        )
        duration = DurationEstimate(
            run_type=RunType.INTERVALS,
            distance_km=total,
            method=EstimateMethod.SESSION,
            n_runs_used=n_history,
            p50_min=round(p50_min, 1),
            p80_min=round(p80_min, 1),
            slot_min=slot_minutes(p80_min, buffer_min),
            note=note,
        )
        return IntervalSession(
            session_id=kind,
            kind=kind,
            reps_m=reps,
            target_paces_s_per_km=paces,
            rest_s=rests,
            warmup_km=warm,
            cooldown_km=cool,
            total_km=total,
            duration=duration,
        )

    def fitted(kind: str, rep_m: int) -> IntervalSession | None:
        for n in range(reps_of[rep_m], MIN_REPS - 1, -1):
            option = make(kind, (rep_m,) * n)
            if option is not None and (max_minutes is None or option.duration.slot_min <= max_minutes):
                return option
        return None

    options = [fitted("short", SHORT_REP_M), fitted("medium", MEDIUM_REP_M), make("pyramid", PYRAMID_M)]
    return [o for o in options if o is not None and (max_minutes is None or o.duration.slot_min <= max_minutes)]


def typical_spread(log_residual_quantile: float | None) -> float:
    """The personal model's p80/p50 ratio, or a conservative 8 % without a model."""
    return math.exp(log_residual_quantile) if log_residual_quantile is not None else 1.08
