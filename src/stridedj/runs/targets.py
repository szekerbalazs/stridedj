"""Interval pace targets per rep distance: start from the runner's own targets, adapt after every session.

Store facts, derive views: targets are never stored. They are recomputed from (initial targets, start date, the
interval sessions detected in the runs since then), so they are reproducible and change only with new runs.
The rule:

* clean session at a target distance (no rep slower than target + 2 s/km, last reps ≤ 3 % slower than the first)
  → the next target is 2 s/km faster;
* missed or faded → unchanged;
* more than 6 weeks without that distance → 5 s/km easier first (fitness for it fades).

Without configured targets, the anchor distances (400 m and 800 m) start from the runner's history: the median of
the last three sessions' median rep pace at that distance, rounded to 5 s/km, plus the 6-week rule. Every other
distance follows the runner's pace-distance curve through the anchors (a power law, like Riegel's): old sessions of
another kind (e.g. tempo-like time reps that detect as 1200 m) would otherwise make a 1200 m slower than an 800 m.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum

from pydantic import AwareDatetime, Field

from stridedj.domain import Frozen
from stridedj.runs.intervals import IntervalAnalysis

CLEAN_STEP_S = 2.0
TOLERANCE_S = 2.0
FADE_RATIO = 1.03
GAP_DAYS = 42
GAP_STEP_S = 5.0
MIN_REPS = 3
HISTORY_SESSIONS = 3
ANCHOR_DISTANCES_M = (400, 800)
DEFAULT_EXPONENT = 1.06  # Riegel, when only one target distance is known
EXPONENT_RANGE = (1.0, 1.25)


class Verdict(StrEnum):
    CLEAN = "clean"
    MISSED = "missed"
    FADED = "faded"


class SessionVerdict(Frozen):
    start: AwareDatetime
    distance_m: int
    n_reps: int = Field(ge=1)
    median_pace_s_per_km: float
    target_s_per_km: float
    verdict: Verdict


Session = tuple[datetime, IntervalAnalysis]


def round_to_5(seconds: float) -> float:
    return round(seconds / 5) * 5.0


def reps_by_distance(analysis: IntervalAnalysis) -> dict[int, list[float]]:
    """Rep paces grouped by nominal distance (in rep order)."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for rep in analysis.reps:
        grouped[rep.nominal_m].append(rep.pace_s_per_km)
    return dict(grouped)


def judge(paces: Sequence[float], target: float) -> Verdict:
    if max(paces) > target + TOLERANCE_S:
        return Verdict.MISSED
    first, last = statistics.fmean(paces[:2]), statistics.fmean(paces[-2:])
    return Verdict.FADED if last > first * FADE_RATIO else Verdict.CLEAN


def adapt_targets(
    initial: Mapping[int, float], *, since: datetime, sessions: Sequence[Session], now: datetime
) -> tuple[dict[int, float], list[SessionVerdict]]:
    """Walk the sessions after `since` in order, applying the rule; then the 6-week rule up to `now`."""
    targets = dict(initial)
    last_seen = dict.fromkeys(targets, since)
    verdicts: list[SessionVerdict] = []
    for start, analysis in sorted(sessions, key=lambda s: s[0]):
        if start < since:
            continue
        for distance, paces in reps_by_distance(analysis).items():
            if distance not in targets or len(paces) < MIN_REPS:
                continue
            if (start - last_seen[distance]).days > GAP_DAYS:
                targets[distance] += GAP_STEP_S
            verdict = judge(paces, targets[distance])
            verdicts.append(
                SessionVerdict(
                    start=start,
                    distance_m=distance,
                    n_reps=len(paces),
                    median_pace_s_per_km=statistics.median(paces),
                    target_s_per_km=targets[distance],
                    verdict=verdict,
                )
            )
            if verdict is Verdict.CLEAN:
                targets[distance] -= CLEAN_STEP_S
            last_seen[distance] = start
    for distance, seen in last_seen.items():
        if (now - seen).days > GAP_DAYS:
            targets[distance] += GAP_STEP_S
    return targets, verdicts


def targets_from_history(sessions: Sequence[Session], *, now: datetime) -> dict[int, float]:
    """Per anchor distance: median of the last three sessions' median rep pace, rounded to 5 s; +5 s after 6 weeks."""
    by_distance: dict[int, list[tuple[datetime, float]]] = defaultdict(list)
    for start, analysis in sessions:
        for distance, paces in reps_by_distance(analysis).items():
            if distance in ANCHOR_DISTANCES_M and len(paces) >= MIN_REPS:
                by_distance[distance].append((start, statistics.median(paces)))
    targets: dict[int, float] = {}
    for distance, entries in by_distance.items():
        recent = sorted(entries, reverse=True)[:HISTORY_SESSIONS]
        target = round_to_5(statistics.median(pace for _, pace in recent))
        if (now - recent[0][0]).days > GAP_DAYS:
            target += GAP_STEP_S
        targets[distance] = target
    return targets


def pace_for(distance_m: int, targets: Mapping[int, float]) -> float:
    """Target pace for any rep distance: the target itself, or the power law T = a·d^b through the targets."""
    if distance_m in targets:
        return targets[distance_m]
    if not targets:
        raise ValueError("no targets to derive a pace from")
    points = sorted(targets.items())
    if len(points) == 1:
        exponent = DEFAULT_EXPONENT
    else:
        xs = [math.log(d) for d, _ in points]
        ys = [math.log(pace * d / 1000) for d, pace in points]  # log rep time
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        exponent = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum((x - mx) ** 2 for x in xs)
        exponent = min(max(exponent, EXPONENT_RANGE[0]), EXPONENT_RANGE[1])
    anchor_d, anchor_pace = min(points, key=lambda p: abs(math.log(p[0] / distance_m)))  # the closest target
    rep_time = anchor_pace * anchor_d / 1000 * float((distance_m / anchor_d) ** exponent)
    return float(rep_time / distance_m * 1000)
