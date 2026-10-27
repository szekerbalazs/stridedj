"""Interval detection from the per-sample series: reps, recoveries and a verdict. Pure (no I/O).

Why not laps: Garmin laps are 1-km auto-laps unless a structured workout is used, so a 400 m rep and its recovery
hide inside one "active" lap, and a lap-based detector finds no interval session at all.

Three signals, each covering a blind spot of another:
* **Pace** (distance over timer time, so smart recording's 3-8 s gaps don't matter) finds the reps exactly and reacts
  instantly; heart rate lags 20-60 s and cannot time a 400 m rep on its own.
* **Watch stops** (elapsed time grows while the timer does not) are rests, although no sample exists during them:
  the recorded fast stretches alone would look like one continuous fast run.
* **Heart rate** confirms the structure: it falls over every rest, whether the watch was stopped or kept running
  (standing with the timer on shows no pace at all, only the HR drop).

Algorithm: smooth pace over ±5 s of timer time → split the moving pace into an easy and a fast level (Otsu's
threshold, weighted by time) → fast blocks of ≥ 25 s and ≥ 80 m (a block interrupted by < 20 s, e.g. a traffic light,
stays one) → the reps are the blocks at the main set's pace (a pick-up in the warm-up is not a rep) → an interval
session has ≥ 3 reps adding up to ≥ 1.2 km (less is strides or short activations), clearly faster than the
easy level, none holding half the fast distance (that is a tempo run), recoveries ≤ 6 min, and heart rate dropping
≥ 12 bpm over them (median; when HR was recorded).

Tuned on a labelled set of real runs; the constants are named so they can be revisited.
"""

from __future__ import annotations

import statistics
from bisect import bisect_left, bisect_right
from collections.abc import Sequence

from pydantic import Field

from stridedj.domain import Frozen, RunSeries

STOP_S = 20.0  # a watch stop at least this long is a rest; shorter ones (traffic lights) are ignored
SMOOTH_HALF_WINDOW_S = 5.0
MOVING_MPS = 1.0  # slower is walking / standing
MIN_REP_S = 25.0
MIN_REP_M = 80.0
MAX_GAP_IN_REP_S = 20.0
MIN_CONTRAST = 1.15  # fast level ≥ 15 % faster than the easy level
MIN_REPS = 3
MIN_WORK_M = 1200.0  # fast distance of a session; less is strides or short activations (e.g. 4 × 200 m)
MAX_RECOVERY_S = 6 * 60.0
MIN_HR_DROP_BPM = 12.0  # median drop over the recoveries
MAX_SINGLE_BLOCK_SHARE = 0.5  # one block with more than half the fast distance: a tempo run
MAX_REP_PACE_RATIO = 1.05  # a rep is at most 5 % slower than the typical rep
STANDARD_REP_M = (100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500, 1600, 2000, 3000)


class Rep(Frozen):
    distance_m: float = Field(gt=0)
    duration_s: float = Field(gt=0, description="timer time of the rep")
    pace_s_per_km: float = Field(gt=0)
    hr_max_bpm: float | None = None
    cadence_spm: float | None = None
    recovery_s: float | None = Field(default=None, description="wall-clock time until the next rep (None: last)")
    recovery_m: float | None = Field(default=None, description="distance covered until the next rep (≈ 0: standing)")
    recovery_hr_drop_bpm: float | None = Field(default=None, description="HR at the rep's end minus at the next start")

    @property
    def nominal_m(self) -> int:
        return nominal_distance_m(self.distance_m)


class IntervalAnalysis(Frozen):
    is_intervals: bool
    reason: str
    reps: tuple[Rep, ...] = ()
    easy_pace_s_per_km: float | None = None
    fast_pace_s_per_km: float | None = None

    @property
    def work_cadence_spm(self) -> float | None:
        """Duration-weighted cadence of the reps: what interval music should match (not the recovery jogs)."""
        timed = [(r.cadence_spm, r.duration_s) for r in self.reps if r.cadence_spm is not None]
        total = sum(d for _, d in timed)
        return sum(c * d for c, d in timed) / total if total > 0 else None


def nominal_distance_m(distance_m: float) -> int:
    """GPS distance of a rep → the distance the runner meant (±15 % of a standard one), else rounded to 50 m."""
    best = min(STANDARD_REP_M, key=lambda d: abs(d - distance_m))
    return best if abs(best - distance_m) <= 0.15 * best else int(round(distance_m / 50) * 50)


def describe(reps: Sequence[Rep]) -> str:
    """ "5 × 1000 m" or "400 m / 800 m / 400 m" (a pyramid or a mix)."""
    distances = [rep.nominal_m for rep in reps]
    if len(set(distances)) == 1:
        return f"{len(reps)} × {distances[0]} m"
    return " / ".join(f"{d} m" for d in distances)


def analyse_series(series: RunSeries) -> IntervalAnalysis:
    n = len(series)
    stop_before = [False] * n  # sample i follows a watch stop ≥ STOP_S
    for i in range(1, n):
        paused = (series.elapsed_s[i] - series.elapsed_s[i - 1]) - (series.timer_s[i] - series.timer_s[i - 1])
        stop_before[i] = paused >= STOP_S
    speed = _smoothed_speed(series, stop_before)
    weights = [_dt(series, i, stop_before) for i in range(n)]
    moving = [(v, w) for v, w in zip(speed, weights, strict=True) if v >= MOVING_MPS and w > 0]
    if len(moving) < 10:
        return IntervalAnalysis(is_intervals=False, reason="too little moving data")
    threshold = _otsu([v for v, _ in moving], [w for _, w in moving])
    easy = _weighted_median([(v, w) for v, w in moving if v < threshold])
    fast = _weighted_median([(v, w) for v, w in moving if v >= threshold])
    if easy is None or fast is None or fast < MIN_CONTRAST * easy:
        return IntervalAnalysis(is_intervals=False, reason="steady pace: no clearly faster level")

    reps = _build_reps(series, _main_set(series, _fast_blocks(series, speed, stop_before, threshold)))

    def verdict(is_intervals: bool, reason: str) -> IntervalAnalysis:
        return IntervalAnalysis(
            is_intervals=is_intervals,
            reason=reason,
            reps=tuple(reps),
            easy_pace_s_per_km=1000 / easy,
            fast_pace_s_per_km=1000 / fast,
        )

    if len(reps) < MIN_REPS:
        return verdict(False, f"{len(reps)} fast block(s): not a repeated structure")
    if sum(r.distance_m for r in reps) < MIN_WORK_M:
        return verdict(False, f"only {sum(r.distance_m for r in reps):.0f} m fast: strides or activations")
    if max(r.distance_m for r in reps) > MAX_SINGLE_BLOCK_SHARE * sum(r.distance_m for r in reps):
        return verdict(False, "one long fast block: a tempo run")
    recoveries = [r.recovery_s for r in reps if r.recovery_s is not None]
    if statistics.median(recoveries) > MAX_RECOVERY_S:
        return verdict(False, "fast blocks too far apart")
    drops = [r.recovery_hr_drop_bpm for r in reps if r.recovery_hr_drop_bpm is not None]
    if drops and statistics.median(drops) < MIN_HR_DROP_BPM:
        return verdict(False, "heart rate hardly drops between the fast blocks")
    return verdict(True, f"{describe(reps)} with recoveries")


# ─────────────────────────────── steps ───────────────────────────────
def _dt(series: RunSeries, i: int, stop_before: Sequence[bool]) -> float:
    """Timer time a sample stands for (to the next sample, none across a stop)."""
    if i + 1 >= len(series) or stop_before[i + 1]:
        return 0.0
    return max(0.0, series.timer_s[i + 1] - series.timer_s[i])


def _smoothed_speed(series: RunSeries, stop_before: Sequence[bool]) -> list[float]:
    """Distance over ±5 s of timer time, never across a watch stop (m/s)."""
    n = len(series)
    starts = [0] * n
    for i in range(1, n):
        starts[i] = i if stop_before[i] else starts[i - 1]
    ends = [n - 1] * n
    for i in range(n - 2, -1, -1):
        ends[i] = i if stop_before[i + 1] else ends[i + 1]
    timer = series.timer_s
    out: list[float] = []
    for i in range(n):
        lo = max(starts[i], bisect_left(timer, timer[i] - SMOOTH_HALF_WINDOW_S, starts[i], ends[i] + 1))
        hi = min(ends[i], bisect_right(timer, timer[i] + SMOOTH_HALF_WINDOW_S, starts[i], ends[i] + 1) - 1)
        span = timer[hi] - timer[lo]
        out.append((series.distance_m[hi] - series.distance_m[lo]) / span if span > 0 else 0.0)
    return out


def _otsu(values: Sequence[float], weights: Sequence[float], bins: int = 64) -> float:
    """The threshold that best separates two levels (max between-class variance), weighted by time."""
    lo, hi = min(values), max(values)
    if hi <= lo:
        return hi
    width = (hi - lo) / bins
    hist = [0.0] * bins
    for v, w in zip(values, weights, strict=True):
        hist[min(bins - 1, int((v - lo) / width))] += w
    total = sum(hist)
    centres = [lo + (k + 0.5) * width for k in range(bins)]
    mean_all = sum(h * c for h, c in zip(hist, centres, strict=True)) / total
    best, best_k, w0, sum0 = -1.0, 0, 0.0, 0.0
    for k in range(bins - 1):
        w0 += hist[k]
        sum0 += hist[k] * centres[k]
        w1 = total - w0
        if w0 <= 0 or w1 <= 0:
            continue
        m0, m1 = sum0 / w0, (mean_all * total - sum0) / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best:
            best, best_k = between, k
    return lo + (best_k + 1) * width


def _weighted_median(pairs: Sequence[tuple[float, float]]) -> float | None:
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return v
    return pairs[-1][0]


def _fast_blocks(
    series: RunSeries, speed: Sequence[float], stop_before: Sequence[bool], threshold: float
) -> list[tuple[int, int]]:
    """Fast blocks (merged over gaps < 20 s of timer time without a watch stop), kept if ≥ 25 s and ≥ 80 m."""
    blocks: list[tuple[int, int]] = []
    start: int | None = None
    last_fast: int | None = None
    for i in range(len(series)):
        fast = speed[i] >= threshold
        if stop_before[i] and start is not None and last_fast is not None:
            blocks.append((start, last_fast))
            start = last_fast = None
        if not fast:
            gap = series.timer_s[i] - series.timer_s[last_fast] if last_fast is not None else 0.0
            if start is not None and last_fast is not None and gap > MAX_GAP_IN_REP_S:
                blocks.append((start, last_fast))
                start = last_fast = None
            continue
        if start is None:
            start = i
        last_fast = i
    if start is not None and last_fast is not None:
        blocks.append((start, last_fast))
    return [
        (a, b)
        for a, b in blocks
        if series.timer_s[b] - series.timer_s[a] >= MIN_REP_S
        and series.distance_m[b] - series.distance_m[a] >= MIN_REP_M
    ]


def _pace(series: RunSeries, block: tuple[int, int]) -> float:
    a, b = block
    return (series.timer_s[b] - series.timer_s[a]) / (series.distance_m[b] - series.distance_m[a]) * 1000


def _main_set(series: RunSeries, blocks: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop blocks clearly slower than the typical one (a pick-up during the warm-up is not a rep)."""
    if not blocks:
        return []
    typical = statistics.median(_pace(series, block) for block in blocks)
    return [block for block in blocks if _pace(series, block) <= typical * MAX_REP_PACE_RATIO]


def _build_reps(series: RunSeries, blocks: Sequence[tuple[int, int]]) -> list[Rep]:
    reps: list[Rep] = []
    for k, (a, b) in enumerate(blocks):
        distance = series.distance_m[b] - series.distance_m[a]
        duration = series.timer_s[b] - series.timer_s[a]
        hrs = [h for h in series.hr_bpm[a : b + 1] if h]
        cadences = [c for c in series.cadence_spm[a : b + 1] if c]
        recovery_s = recovery_m = drop = None
        if k + 1 < len(blocks):
            nxt = blocks[k + 1][0]
            recovery_s = series.elapsed_s[nxt] - series.elapsed_s[b]
            recovery_m = series.distance_m[nxt] - series.distance_m[b]
            end_hr = _hr_near(series, b, backwards=True)
            low_hr = _min_hr_between(series, b, nxt)
            drop = end_hr - low_hr if end_hr is not None and low_hr is not None else None
        reps.append(
            Rep(
                distance_m=distance,
                duration_s=duration,
                pace_s_per_km=duration / distance * 1000,
                hr_max_bpm=max(hrs) if hrs else None,
                cadence_spm=statistics.fmean(cadences) if cadences else None,
                recovery_s=recovery_s,
                recovery_m=recovery_m,
                recovery_hr_drop_bpm=drop,
            )
        )
    return reps


def _hr_near(series: RunSeries, i: int, *, backwards: bool) -> float | None:
    indices = range(i, max(-1, i - 6), -1) if backwards else range(i, min(len(series), i + 6))
    return next((h for j in indices if (h := series.hr_bpm[j])), None)


def _min_hr_between(series: RunSeries, a: int, b: int) -> float | None:
    """Lowest HR from the end of one rep into the first seconds of the next (HR lags: it is lowest at the restart)."""
    hrs = [h for h in series.hr_bpm[a : min(len(series), b + 6)] if h]
    return min(hrs) if hrs else None
