"""Duration and race-time prediction: how long will a run of type X over D km take (median and 80th percentile)?

Regimes, tried in order; the first with enough data wins and is reported (`EstimateMethod`), so the agent can be
honest about how much the number is worth:

1. RECENT_SIMILAR: >= 3 runs of the same type within ±25 % of the distance in the last 90 days. Each is scaled to
   the target distance with the personal exponent, T·(D/Dᵢ)^b, and the recency-weighted (half-life 45 days)
   empirical p50/p80 is returned. Model-light: the data speak for themselves where we have data.
2. REGRESSION: the personal model, a weighted log-log regression
       log T = a + b·log D + Σ type effects + c·(elevation gain per km)  [+ d·age_days, optional]
   fit on all runs, weighted by recency (half-life 90 days within the last 180 days; older runs × 0.1). The old
   runs are kept on purpose: when the target distance has not been run recently (no long run since May), the old
   long runs are the only evidence about the exponent b, and down-weighting them shrinks toward that evidence
   without letting stale fitness dominate the level.
3. RIEGEL: T2 = T1·(D2/D1)^1.06 from the most recent run of the same type, when the regression cannot be fit.
4. DEFAULT: a conservative pace per type; p80 = 1.15 × p50.

Why log-log: Riegel's law T = a·D^b is linear in logs (b > 1 is fatigue), type and hill effects are multiplicative
(a threshold run is ~15 % faster per km, not 60 s faster), and day-to-day noise is proportional to duration, so
residuals are homoscedastic on the log scale. Every coefficient is readable: b is the personal fatigue exponent,
exp(type effect) the pace ratio to an easy run, 100·c the % time per metre of climb per km.

Why p80: the calendar slot is a newsvendor problem. Overrunning (late for the next thing) costs about four times as
much as an idle quarter-hour, so the cost-optimal slot is the 4/(4+1) = 80th percentile of the running time; on
top come 10 minutes of buffer (getting out, stretching), rounded up to the 15-minute calendar grid.

Everything here is pure: a function of (runs, now, config). `now` is a parameter, and every function only looks
at runs that started strictly before it, which is what makes the rolling-origin backtest leakage-free.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import Field
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor

from stridedj.domain import (
    ClassifiedRun,
    DurationEstimate,
    EstimateMethod,
    RacePrediction,
    RunType,
)
from stridedj.domain.base import Frozen
from stridedj.domain.planning import SLOT_GRANULARITY_MIN
from stridedj.runs.analysis import RIEGEL_EXPONENT, age_days, recency_weight, weighted_quantile

Estimator = Literal["ols", "huber", "quantile"]
ESTIMATORS: tuple[Estimator, ...] = ("ols", "huber", "quantile")

DEFAULT_PACE_S_PER_KM: dict[RunType, float] = {
    RunType.Z1: 390.0,  # 6:30/km
    RunType.Z2: 360.0,  # 6:00/km
    RunType.Z3: 320.0,  # 5:20/km
    RunType.Z4: 300.0,  # 5:00/km
    RunType.Z5: 280.0,  # 4:40/km
    RunType.INTERVALS: 330.0,  # 5:30/km including warm-up, recoveries and cool-down
}
DEFAULT_P80_FACTOR = 1.15
RIEGEL_P80_FACTOR = 1.10  # one anchor run: day-to-day form alone is ±5 %, plus the textbook exponent's error
MIN_LOG_DISTANCE_SPREAD = math.log(1.3)  # runs must span >= 30 % in distance for the exponent to be identified

RACE_EFFORTS: tuple[tuple[float, RunType], ...] = (
    (5.0, RunType.Z4),
    (10.0, RunType.Z4),
    (21.0975, RunType.Z4),
    (42.195, RunType.Z3),
)
"""Race proxy: training runs are not races, so a race time is predicted as a run at a sustainable hard effort.

5 km to half marathon are raced around lactate threshold (the user's Z4); a marathon is run below it (Z3). The
proxy is conservative: race-day tapering, pacing and adrenaline usually make a real race a few % faster.
"""

BACKTEST_COLUMNS = (
    "activity_id",
    "date",
    "distance_km",
    "run_type",
    "actual_min",
    "pred_p50_min",
    "pred_p80_min",
    "method",
    "fold",
    "n_train",
    "train_end",
)


class PredictorConfig(Frozen):
    """All tunables of the duration model in one place (so the notebook can sweep them)."""

    recent_window_days: int = Field(default=90, gt=0)
    distance_tolerance: float = Field(default=0.25, gt=0, lt=1)
    min_similar_runs: int = Field(default=3, ge=1)
    recency_half_life_days: float = Field(default=45.0, gt=0)

    estimator: Estimator = "ols"
    regression_window_days: int = Field(default=180, gt=0)
    regression_half_life_days: float = Field(default=90.0, gt=0)
    old_run_weight: float = Field(default=0.1, ge=0, le=1, description="extra weight factor beyond the window")
    history_days: int = Field(default=730, gt=0, description="runs older than this are ignored entirely")
    min_regression_runs: int = Field(default=8, ge=3, description="runs inside the regression window")
    min_runs_per_type: int = Field(default=2, ge=1, description="a type effect needs at least this many runs")
    exponent_bounds: tuple[float, float] = (0.95, 1.30)
    include_fitness_trend: bool = Field(default=False, description="add age_days as a covariate (fitness trend)")
    quantile: float = Field(default=0.8, gt=0.5, lt=1)

    default_exponent: float = Field(default=RIEGEL_EXPONENT, gt=0)
    buffer_min: float = Field(default=10.0, ge=0)


# ─────────────────────────────── helpers ───────────────────────────────
def slot_minutes(p80_min: float, buffer_min: float) -> int:
    """Calendar slot: p80 + buffer, rounded UP to the 15-minute grid (never below the p80)."""
    granularity = SLOT_GRANULARITY_MIN
    return max(granularity, math.ceil((p80_min + buffer_min) / granularity - 1e-9) * granularity)


def _fmt_km(km: float) -> str:
    return f"{km:.1f}".removesuffix(".0")


def _fmt_pace(s_per_km: float) -> str:
    minutes, seconds = divmod(round(s_per_km), 60)
    return f"{minutes}:{seconds:02d}/km"


@dataclass(frozen=True)
class _Row:
    """One training run, reduced to the model's variables."""

    run_type: RunType
    age_days: float
    distance_km: float
    duration_min: float
    elevation_per_km: float


def _rows(runs: Sequence[ClassifiedRun], *, now: datetime, history_days: int) -> list[_Row]:
    """Runs that started strictly before `now` and within the history horizon (the leakage guard lives here)."""
    rows: list[_Row] = []
    for run in runs:
        activity = run.activity
        age = age_days(activity.start, now)
        if not 0.0 < age <= history_days:
            continue
        rows.append(
            _Row(
                run_type=run.run_type,
                age_days=age,
                distance_km=activity.distance_km,
                duration_min=run.planning_duration_s / 60.0,
                # no elevation recorded (treadmill, some watches): treated as flat
                elevation_per_km=(activity.elevation_gain_m or 0.0) / activity.distance_km,
            )
        )
    return rows


# ─────────────────────────────── the personal regression model ───────────────────────────────
@dataclass(frozen=True, eq=False)
class PersonalModel:
    """A fitted weighted log-log model of running time. Build it with `fit_personal_model`.

    Features are standardised before fitting (helps Huber's optimiser; OLS and quantile regression are unaffected),
    and `coefficients()` maps them back to the readable, unstandardised scale.
    """

    estimator: Estimator
    run_types: tuple[RunType, ...]  # supported types; the first is the baseline (no dummy)
    feature_names: tuple[str, ...]
    include_fitness_trend: bool
    exponent: float
    n_recent: int  # runs inside the regression window (the "n" we report)
    n_total: int
    log_residual_quantile: float  # weighted q-quantile of the p50 model's in-sample log residuals
    typical_elevation_per_km: Mapping[RunType, float]
    mean: NDArray[np.float64] = field(repr=False)
    scale: NDArray[np.float64] = field(repr=False)
    p50_model: Any = field(repr=False)
    p80_model: Any = field(default=None, repr=False)  # only for estimator="quantile"

    def supports(self, run_type: RunType) -> bool:
        return run_type in self.run_types

    def _design(self, run_type: RunType, distance_km: float, elevation_per_km: float) -> NDArray[np.float64]:
        raw = _feature_vector(
            run_type, math.log(distance_km), elevation_per_km, 0.0, self.run_types, self.include_fitness_trend
        )
        return ((raw - self.mean) / self.scale).reshape(1, -1)

    def predict_min(
        self, run_type: RunType, distance_km: float, *, elevation_per_km: float | None = None
    ) -> tuple[float, float]:
        """(p50, p80) running time in minutes, for a run today (age 0) on typical terrain unless given."""
        if not self.supports(run_type):
            raise ValueError(f"the personal model has no effect for run type {run_type.value!r}")
        elevation = self.typical_elevation_per_km[run_type] if elevation_per_km is None else elevation_per_km
        x = self._design(run_type, distance_km, elevation)
        log_p50 = float(self.p50_model.predict(x)[0])
        log_p80 = log_p50 + self.log_residual_quantile
        if self.p80_model is not None:
            direct = float(self.p80_model.predict(x)[0])
            # Two separately fitted quantile lines can cross when extrapolating (different slopes); a p80 below the
            # p50 is meaningless, so the residual-based p80 of the median model takes over in that case.
            if direct > log_p50:
                log_p80 = direct
        p50 = math.exp(log_p50)
        return p50, max(math.exp(log_p80), p50)

    def coefficients(self) -> dict[str, float]:
        """Coefficients of the p50 model on the original scale (log T per unit of each feature) plus the intercept."""
        coef = np.asarray(self.p50_model.coef_, dtype=np.float64) / self.scale
        intercept = float(self.p50_model.intercept_) - float(np.dot(coef, self.mean))
        return {"intercept": intercept, **{name: float(c) for name, c in zip(self.feature_names, coef, strict=True)}}


def _feature_names(run_types: Sequence[RunType], include_trend: bool) -> tuple[str, ...]:
    trend = ("age_days",) if include_trend else ()
    return ("log_distance_km", "elevation_gain_m_per_km", *trend, *(f"type_{t.value}" for t in run_types[1:]))


def _feature_vector(
    run_type: RunType,
    log_km: float,
    elevation_per_km: float,
    age: float,
    run_types: Sequence[RunType],
    include_trend: bool,
) -> NDArray[np.float64]:
    trend = [age] if include_trend else []
    dummies = [1.0 if run_type is t else 0.0 for t in run_types[1:]]
    return np.asarray([log_km, elevation_per_km, *trend, *dummies], dtype=np.float64)


def _regression_weights(rows: Sequence[_Row], config: PredictorConfig) -> NDArray[np.float64]:
    """Recency decay inside the window, a strong extra discount outside it (shrinkage toward old evidence)."""
    return np.asarray(
        [
            recency_weight(r.age_days, config.regression_half_life_days)
            * (1.0 if r.age_days <= config.regression_window_days else config.old_run_weight)
            for r in rows
        ],
        dtype=np.float64,
    )


def _fit_estimator(
    estimator: Estimator, x: NDArray[np.float64], y: NDArray[np.float64], w: NDArray[np.float64], quantile: float
) -> tuple[Any, Any]:
    if estimator == "ols":
        return LinearRegression().fit(x, y, sample_weight=w), None
    if estimator == "huber":
        return HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000).fit(x, y, sample_weight=w), None
    median = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(x, y, sample_weight=w)
    upper = QuantileRegressor(quantile=quantile, alpha=0.0, solver="highs").fit(x, y, sample_weight=w)
    return median, upper


def _supported_types(rows: Sequence[_Row], min_runs: int) -> tuple[RunType, ...]:
    """Types with enough runs for their own effect; Z2 (the most common, easiest to read) is the baseline."""
    counts = Counter(r.run_type for r in rows)
    supported = [t for t in RunType if counts[t] >= min_runs]
    if not supported:
        return ()
    baseline = RunType.Z2 if RunType.Z2 in supported else max(supported, key=lambda t: counts[t])
    return (baseline, *(t for t in supported if t is not baseline))


def fit_personal_model(
    runs: Sequence[ClassifiedRun], *, now: datetime, config: PredictorConfig | None = None
) -> PersonalModel | None:
    """Fit the personal log-log model on runs before `now`; None when the data cannot identify it.

    "Cannot identify" means: fewer than `min_regression_runs` runs inside the regression window, distances that
    span less than 30 % (the exponent is then pure noise), or a fitted exponent outside `exponent_bounds`.
    """
    cfg = config or PredictorConfig()
    all_rows = _rows(runs, now=now, history_days=cfg.history_days)
    run_types = _supported_types(all_rows, cfg.min_runs_per_type)
    rows = [r for r in all_rows if r.run_type in run_types]  # a type without its own effect would bias the baseline
    n_recent = sum(r.age_days <= cfg.regression_window_days for r in rows)
    if n_recent < cfg.min_regression_runs:
        return None
    log_km = np.log([r.distance_km for r in rows])
    if float(np.ptp(log_km)) < MIN_LOG_DISTANCE_SPREAD:
        return None

    raw = np.vstack(
        [
            _feature_vector(r.run_type, lk, r.elevation_per_km, r.age_days, run_types, cfg.include_fitness_trend)
            for r, lk in zip(rows, log_km, strict=True)
        ]
    )
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale == 0.0] = 1.0
    x = (raw - mean) / scale
    y = np.log([r.duration_min for r in rows])
    w = _regression_weights(rows, cfg)

    p50_model, p80_model = _fit_estimator(cfg.estimator, x, y, w, cfg.quantile)
    exponent = float(p50_model.coef_[0]) / float(scale[0])
    if not cfg.exponent_bounds[0] <= exponent <= cfg.exponent_bounds[1]:
        return None
    # OLS/Huber p80 = exp(prediction + weighted q-quantile of the log residuals); kept for "quantile" as well, as
    # the fallback when its two quantile lines cross
    residuals = y - np.asarray(p50_model.predict(x), dtype=np.float64)
    residual_q = max(0.0, weighted_quantile(residuals.tolist(), w.tolist(), cfg.quantile))

    typical_elevation = {
        t: weighted_quantile(
            [r.elevation_per_km for r in rows if r.run_type is t],
            [wi for r, wi in zip(rows, w, strict=True) if r.run_type is t],
            0.5,
        )
        for t in run_types
    }
    return PersonalModel(
        estimator=cfg.estimator,
        run_types=run_types,
        feature_names=_feature_names(run_types, cfg.include_fitness_trend),
        include_fitness_trend=cfg.include_fitness_trend,
        exponent=exponent,
        n_recent=n_recent,
        n_total=len(rows),
        log_residual_quantile=residual_q,
        typical_elevation_per_km=typical_elevation,
        mean=mean,
        scale=scale,
        p50_model=p50_model,
        p80_model=p80_model,
    )


# ─────────────────────────────── the predictor (regime selection) ───────────────────────────────
class DurationPredictor:
    """Answers many (type, distance) questions for one history and one `now`; the personal model is fit once."""

    def __init__(self, runs: Sequence[ClassifiedRun], *, now: datetime, config: PredictorConfig | None = None) -> None:
        self._config = config or PredictorConfig()
        self._now = now
        self._runs = [r for r in runs if 0.0 < age_days(r.activity.start, now) <= self._config.history_days]
        self._model = fit_personal_model(self._runs, now=now, config=self._config)

    @property
    def model(self) -> PersonalModel | None:
        return self._model

    @property
    def exponent(self) -> float:
        """The personal exponent when the model could be fit, else the textbook one."""
        return self._model.exponent if self._model is not None else self._config.default_exponent

    def estimate(
        self, run_type: RunType, distance_km: float, *, elevation_gain_per_km: float | None = None
    ) -> DurationEstimate:
        """Duration estimate from the first regime with enough data (see the module docstring)."""
        if not 0.0 < distance_km <= 100.0:
            raise ValueError("distance_km must be in (0, 100]")
        similar = self._similar_runs(run_type, distance_km)
        if len(similar) >= self._config.min_similar_runs:
            return self._recent_similar(run_type, distance_km, similar)
        regression = self.regression_estimate(
            run_type, distance_km, elevation_gain_per_km=elevation_gain_per_km, n_similar=len(similar)
        )
        if regression is not None:
            return regression
        anchor = self._anchor(run_type)
        if anchor is not None:
            return self._riegel(run_type, distance_km, anchor)
        return self._default(run_type, distance_km)

    def race_predictions(self) -> list[RacePrediction]:
        """Predicted race times for 5 km, 10 km, half and full marathon (see `RACE_EFFORTS` for the proxy)."""
        predictions: list[RacePrediction] = []
        for distance_km, effort in RACE_EFFORTS:
            estimate = self.estimate(effort, distance_km)
            predictions.append(
                RacePrediction(
                    distance_km=distance_km,
                    predicted_min=estimate.p50_min,
                    method=estimate.method,
                    n_runs_used=estimate.n_runs_used,
                )
            )
        return predictions

    # ── regime 1 ──
    def _similar_range(self, distance_km: float) -> tuple[float, float]:
        tolerance = self._config.distance_tolerance
        return distance_km * (1.0 - tolerance), distance_km * (1.0 + tolerance)

    def _similar_runs(self, run_type: RunType, distance_km: float) -> list[ClassifiedRun]:
        tolerance = self._config.distance_tolerance
        return [
            r
            for r in self._runs
            if r.run_type is run_type
            and age_days(r.activity.start, self._now) <= self._config.recent_window_days
            and abs(r.activity.distance_km / distance_km - 1.0) <= tolerance
        ]

    def _recent_similar(
        self, run_type: RunType, distance_km: float, similar: Sequence[ClassifiedRun]
    ) -> DurationEstimate:
        b = self.exponent
        scaled = [r.planning_duration_s / 60.0 * (distance_km / r.activity.distance_km) ** b for r in similar]
        weights = [
            recency_weight(age_days(r.activity.start, self._now), self._config.recency_half_life_days) for r in similar
        ]
        p50 = weighted_quantile(scaled, weights, 0.5)
        p80 = weighted_quantile(scaled, weights, self._config.quantile)
        lo, hi = self._similar_range(distance_km)
        exponent_text = f"personal exponent {b:.2f}" if self._model is not None else f"textbook exponent {b:.2f}"
        note = (
            f"Based on {len(similar)} {run_type.value} runs of {_fmt_km(lo)}–{_fmt_km(hi)} km in the last "
            f"{self._config.recent_window_days} days, scaled to {_fmt_km(distance_km)} km ({exponent_text})."
        )
        return self._build(run_type, distance_km, EstimateMethod.RECENT_SIMILAR, len(similar), p50, p80, note)

    # ── regime 2 ──
    def regression_estimate(
        self,
        run_type: RunType,
        distance_km: float,
        *,
        elevation_gain_per_km: float | None = None,
        n_similar: int | None = None,
    ) -> DurationEstimate | None:
        """The personal-model estimate, or None if the model is unavailable for this run type.

        Public so the extrapolation evaluation can force this regime.
        """
        model = self._model
        if model is None or not model.supports(run_type):
            return None
        p50, p80 = model.predict_min(run_type, distance_km, elevation_per_km=elevation_gain_per_km)
        if n_similar is None:
            n_similar = len(self._similar_runs(run_type, distance_km))
        lo, hi = self._similar_range(distance_km)
        count = "No" if n_similar == 0 else f"Only {n_similar}"
        runs_word = "run" if n_similar <= 1 else "runs"
        note = (
            f"{count} {run_type.value} {runs_word} of {_fmt_km(lo)}–{_fmt_km(hi)} km in the last "
            f"{self._config.recent_window_days} days → extrapolated with the personal model fit on "
            f"{model.n_recent} runs from the last {self._config.regression_window_days} days "
            f"(exponent {model.exponent:.2f}, {model.estimator})."
        )
        return self._build(run_type, distance_km, EstimateMethod.REGRESSION, model.n_recent, p50, p80, note)

    # ── regime 3 ──
    def _anchor(self, run_type: RunType) -> ClassifiedRun | None:
        same_type = [r for r in self._runs if r.run_type is run_type]
        return max(same_type, key=lambda r: r.activity.start) if same_type else None

    def _riegel(self, run_type: RunType, distance_km: float, anchor: ClassifiedRun) -> DurationEstimate:
        exponent = self._config.default_exponent
        anchor_min = anchor.planning_duration_s / 60.0
        p50 = anchor_min * (distance_km / anchor.activity.distance_km) ** exponent
        reason = (
            "Too little history for the personal model"
            if self._model is None
            else (f"Too few {run_type.value} runs for the personal model")
        )
        days = math.floor(age_days(anchor.activity.start, self._now))
        note = (
            f"{reason} → Riegel's formula (exponent {exponent:.2f}) from your "
            f"{_fmt_km(anchor.activity.distance_km)} km {run_type.value} run {days} days ago."
        )
        return self._build(run_type, distance_km, EstimateMethod.RIEGEL, 1, p50, p50 * RIEGEL_P80_FACTOR, note)

    # ── regime 4 ──
    def _default(self, run_type: RunType, distance_km: float) -> DurationEstimate:
        pace = DEFAULT_PACE_S_PER_KM[run_type]
        p50 = pace * distance_km / 60.0
        note = (
            f"No usable {run_type.value} history → default pace {_fmt_pace(pace)}, "
            f"p80 = p50 + {DEFAULT_P80_FACTOR - 1:.0%}."
        )
        return self._build(run_type, distance_km, EstimateMethod.DEFAULT, 0, p50, p50 * DEFAULT_P80_FACTOR, note)

    def _build(
        self,
        run_type: RunType,
        distance_km: float,
        method: EstimateMethod,
        n_runs: int,
        p50: float,
        p80: float,
        note: str,
    ) -> DurationEstimate:
        p50_min = round(p50, 1)
        p80_min = max(round(p80, 1), p50_min)
        return DurationEstimate(
            run_type=run_type,
            distance_km=distance_km,
            method=method,
            n_runs_used=n_runs,
            p50_min=p50_min,
            p80_min=p80_min,
            slot_min=slot_minutes(p80_min, self._config.buffer_min),  # from the p80 the user sees
            note=note,
        )


def estimate_duration(
    runs: Sequence[ClassifiedRun],
    run_type: RunType,
    distance_km: float,
    *,
    now: datetime,
    config: PredictorConfig | None = None,
    elevation_gain_per_km: float | None = None,
) -> DurationEstimate:
    """One-shot convenience wrapper around `DurationPredictor`."""
    predictor = DurationPredictor(runs, now=now, config=config)
    return predictor.estimate(run_type, distance_km, elevation_gain_per_km=elevation_gain_per_km)


def race_predictions(
    runs: Sequence[ClassifiedRun], *, now: datetime, config: PredictorConfig | None = None
) -> list[RacePrediction]:
    return DurationPredictor(runs, now=now, config=config).race_predictions()


# ─────────────────────────────── evaluation harness (for the notebook) ───────────────────────────────
def _config_for(estimator: Estimator, config: PredictorConfig | None) -> PredictorConfig:
    return (config or PredictorConfig()).model_copy(update={"estimator": estimator})


def _elevation_per_km(run: ClassifiedRun) -> float:
    return (run.activity.elevation_gain_m or 0.0) / run.activity.distance_km


def _result_row(
    target: ClassifiedRun, estimate: DurationEstimate, *, fold: int, train: Sequence[ClassifiedRun]
) -> dict[str, Any]:
    return {
        "activity_id": target.activity.activity_id,
        "date": pd.Timestamp(target.activity.start),
        "distance_km": target.activity.distance_km,
        "run_type": target.run_type.value,
        "actual_min": target.planning_duration_s / 60.0,
        "pred_p50_min": estimate.p50_min,
        "pred_p80_min": estimate.p80_min,
        "method": estimate.method.value,
        "fold": fold,
        "n_train": len(train),
        "train_end": pd.Timestamp(max(r.activity.start for r in train)) if train else pd.NaT,
    }


def rolling_origin_backtest(
    runs: Sequence[ClassifiedRun],
    *,
    estimator: Estimator = "ols",
    min_train: int = 30,
    step: int = 1,
    config: PredictorConfig | None = None,
    use_actual_elevation: bool = True,
) -> pd.DataFrame:
    """Rolling-origin evaluation: predict each run (from the `min_train`-th on, every `step`-th) from the past only.

    For fold k the origin is the start of the target run; the predictor receives only runs that started strictly
    before it, and `now` = the origin, so recency windows and weights are what they would have been on that day.
    `use_actual_elevation` feeds the target run's real climb to the regression (a planned route is known in
    advance); set False to evaluate with the typical terrain instead.

    Returns raw predictions, one row per target run (columns: `BACKTEST_COLUMNS`); metrics are computed downstream.
    """
    if min_train < 1 or step < 1:
        raise ValueError("min_train and step must be >= 1")
    cfg = _config_for(estimator, config)
    ordered = sorted(runs, key=lambda r: r.activity.start)
    rows: list[dict[str, Any]] = []
    for fold, index in enumerate(range(min_train, len(ordered), step)):
        target = ordered[index]
        origin = target.activity.start
        train = [r for r in ordered[:index] if r.activity.start < origin]
        predictor = DurationPredictor(train, now=origin, config=cfg)
        estimate = predictor.estimate(
            target.run_type,
            target.activity.distance_km,
            elevation_gain_per_km=_elevation_per_km(target) if use_actual_elevation else None,
        )
        rows.append(_result_row(target, estimate, fold=fold, train=train))
    return pd.DataFrame.from_records(rows, columns=list(BACKTEST_COLUMNS))


def extrapolation_split(
    runs: Sequence[ClassifiedRun], threshold_km: float
) -> tuple[list[ClassifiedRun], list[ClassifiedRun]]:
    """(train, test): runs shorter than `threshold_km` vs. runs at least that long.

    Answers the question the regression regime exists for: "how well do we predict a distance we have not run?"
    """
    train = [r for r in runs if r.activity.distance_km < threshold_km]
    test = [r for r in runs if r.activity.distance_km >= threshold_km]
    return train, test


def evaluate_extrapolation(
    train: Sequence[ClassifiedRun],
    test: Sequence[ClassifiedRun],
    *,
    estimator: Estimator = "ols",
    config: PredictorConfig | None = None,
    use_actual_elevation: bool = True,
) -> pd.DataFrame:
    """Predict every test run from the training runs that started before it, forcing the REGRESSION regime.

    RECENT_SIMILAR is skipped on purpose (it would answer from the longest short runs instead of extrapolating);
    when the regression cannot be fit, the normal fallback chain is used and `method` says so.
    """
    cfg = _config_for(estimator, config)
    rows: list[dict[str, Any]] = []
    for fold, target in enumerate(sorted(test, key=lambda r: r.activity.start)):
        origin = target.activity.start
        past = [r for r in train if r.activity.start < origin]
        predictor = DurationPredictor(past, now=origin, config=cfg)
        elevation = _elevation_per_km(target) if use_actual_elevation else None
        estimate = predictor.regression_estimate(
            target.run_type, target.activity.distance_km, elevation_gain_per_km=elevation
        ) or predictor.estimate(target.run_type, target.activity.distance_km, elevation_gain_per_km=elevation)
        rows.append(_result_row(target, estimate, fold=fold, train=past))
    return pd.DataFrame.from_records(rows, columns=list(BACKTEST_COLUMNS))
