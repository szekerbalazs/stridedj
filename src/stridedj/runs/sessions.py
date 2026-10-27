"""Planned interval sessions: the total distance from its parts, added up in code rather than by the LLM.

"4x800, 2 km warm-up, 1 km cool-down" names the work bouts but not the jogs between them. Asking the runner for
those turned planning into an interrogation, so a missing recovery defaults to half a repetition
between consecutive repetitions (4x800 → 3 × 400 m), a common jog recovery. The assumption is reported back
(`recovery_assumed`) so the agent can say it, and the runner can correct it.
"""

from __future__ import annotations

from stridedj.domain import IntervalSessionDistance

DEFAULT_RECOVERY_SHARE = 0.5
"""Recovery jog between repetitions, as a share of one repetition's distance, when the runner gives none."""


def interval_session_distance(
    *, reps: int, rep_m: float, warmup_km: float, cooldown_km: float, recovery_m: float | None = None
) -> IntervalSessionDistance:
    """Total = warm-up + reps × rep + (reps − 1) × recovery + cool-down (no recovery after the last repetition)."""
    if reps < 1 or rep_m <= 0:
        raise ValueError("reps must be >= 1 and rep_m > 0")
    if warmup_km < 0 or cooldown_km < 0 or (recovery_m is not None and recovery_m < 0):
        raise ValueError("warm-up, cool-down and recovery must be >= 0")
    assumed = recovery_m is None
    recovery_each_m = rep_m * DEFAULT_RECOVERY_SHARE if recovery_m is None else recovery_m
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
