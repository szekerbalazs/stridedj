from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from stridedj.domain import Activity, ActivityId, ClassifiedRun, EstimateMethod, RunType
from stridedj.runs.analysis import DEFAULT_HR_ZONES, classify_all
from stridedj.runs.predict import (
    BACKTEST_COLUMNS,
    ESTIMATORS,
    DurationPredictor,
    Estimator,
    PredictorConfig,
    estimate_duration,
    evaluate_extrapolation,
    extrapolation_split,
    fit_personal_model,
    race_predictions,
    rolling_origin_backtest,
    slot_minutes,
)
from stridedj.runs.sources.demo import DEMO_NOW, generate_history

NOW = datetime(2026, 10, 17, 16, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def demo_runs() -> list[ClassifiedRun]:
    return classify_all(generate_history(), DEFAULT_HR_ZONES)


@pytest.fixture(scope="module")
def predictor(demo_runs: list[ClassifiedRun]) -> DurationPredictor:
    return DurationPredictor(demo_runs, now=DEMO_NOW)


def run(
    activity_id: str, *, days_ago: float, km: float, minutes: float, run_type: RunType = RunType.Z2
) -> ClassifiedRun:
    activity = Activity(
        activity_id=ActivityId(activity_id),
        source="demo",
        start=NOW - timedelta(days=days_ago),
        distance_m=km * 1000.0,
        elapsed_s=minutes * 60.0,
        moving_s=minutes * 60.0,
    )
    return ClassifiedRun(activity=activity, run_type=run_type, reason="test")


# ─────────────────────────────── regimes ───────────────────────────────
def test_recent_similar_for_a_10k_easy_run(predictor: DurationPredictor) -> None:
    estimate = predictor.estimate(RunType.Z2, 10.0)
    assert estimate.method is EstimateMethod.RECENT_SIMILAR
    assert estimate.n_runs_used >= 3
    assert 50.0 <= estimate.p50_min <= 65.0  # ~5:30–6:30/km
    assert estimate.p50_min <= estimate.p80_min <= estimate.slot_min
    assert estimate.note.startswith(f"Based on {estimate.n_runs_used} z2 runs of 7.5–12.5 km in the last 90 days")


def test_regression_for_a_half_marathon_distance_not_run_recently(predictor: DurationPredictor) -> None:
    estimate = predictor.estimate(RunType.Z2, 21.1)
    assert estimate.method is EstimateMethod.REGRESSION
    assert predictor.model is not None
    assert estimate.n_runs_used == predictor.model.n_recent
    assert 1.0 < predictor.model.exponent < 1.2
    assert 115.0 <= estimate.p50_min <= 145.0  # ~5:30–6:50/km
    low, high = predictor.model.distance_range_km[RunType.Z2]
    verb = "interpolated" if low <= 21.1 <= high else "extrapolated"  # older long runs count as fitted distances
    assert estimate.note.startswith(f"No z2 run of 15.8–26.4 km in the last 90 days → {verb}")


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_every_estimator_fits_and_keeps_p80_above_p50(demo_runs: list[ClassifiedRun], estimator: Estimator) -> None:
    predictor = DurationPredictor(demo_runs, now=DEMO_NOW, config=PredictorConfig(estimator=estimator))
    model = predictor.model
    assert model is not None
    assert model.estimator == estimator
    for run_type, km in [(RunType.Z2, 21.1), (RunType.Z4, 21.0975), (RunType.Z2, 30.0), (RunType.Z1, 3.0)]:
        estimate = predictor.estimate(run_type, km)
        assert estimate.method is EstimateMethod.REGRESSION
        assert estimate.p80_min >= estimate.p50_min
        assert estimate.note.endswith(f"{estimator}).")


def test_coefficients_are_readable(predictor: DurationPredictor) -> None:
    assert predictor.model is not None
    coefficients = predictor.model.coefficients()
    assert coefficients["log_distance_km"] == pytest.approx(predictor.model.exponent)
    assert coefficients["type_z4"] < 0 < coefficients["type_z1"]  # threshold faster, recovery slower than easy
    assert coefficients["elevation_gain_m_per_km"] > 0  # climbing costs time


def test_fitness_trend_option_adds_an_age_feature(demo_runs: list[ClassifiedRun]) -> None:
    model = fit_personal_model(demo_runs, now=DEMO_NOW, config=PredictorConfig(include_fitness_trend=True))
    assert model is not None
    assert "age_days" in model.feature_names
    assert model.coefficients()["age_days"] > 0  # older runs were slower: the athlete got fitter


def test_the_note_says_interpolated_inside_the_fitted_distances() -> None:
    """Older Z2 runs of 5-15 km (none in the last 90 days): 10 km lies inside them, 25 km does not."""
    runs = [
        run(f"h{i}", days_ago=100 + 6 * i, km=km, minutes=6.0 * km * (1 + 0.02 * (i % 3)))
        for i, km in enumerate([5, 7, 9, 11, 13, 15, 6, 8, 12, 14])
    ]
    inside = DurationPredictor(runs, now=NOW).estimate(RunType.Z2, 10.0)
    outside = DurationPredictor(runs, now=NOW).estimate(RunType.Z2, 25.0)
    assert inside.method is outside.method is EstimateMethod.REGRESSION
    assert "→ interpolated with the personal model" in inside.note
    assert "→ extrapolated with the personal model" in outside.note


def test_a_few_identical_runs_still_get_the_personal_spread() -> None:
    """3 similar runs can give p80 = p50 (the 80 % point falls on the same run). The p80 is at least the
    personal model's typical spread above the p50."""
    noise = [0.94, 1.05, 0.99, 1.08, 0.97, 1.03, 0.92, 1.06, 1.01, 0.96, 1.09, 0.98]
    history = [
        run(f"h{i}", days_ago=40 + 9 * i, km=km, minutes=6.0 * km * factor)
        for i, (km, factor) in enumerate(zip([5, 7, 9, 12, 15, 6, 8, 11, 14, 5, 9, 13], noise, strict=True))
    ]
    recent = [run(f"r{i}", days_ago=5 + 7 * i, km=10.0, minutes=60.0) for i in range(3)]
    predictor = DurationPredictor(history + recent, now=NOW)
    estimate = predictor.estimate(RunType.Z2, 10.0)
    assert estimate.method is EstimateMethod.RECENT_SIMILAR
    assert estimate.p50_min == pytest.approx(60.0, abs=0.1)
    assert predictor.model is not None
    spread = math.exp(predictor.model.log_residual_quantile)
    assert spread > 1.01
    assert estimate.p80_min == pytest.approx(60.0 * spread, abs=0.1)
    assert "usual run-to-run spread" in estimate.note


def test_riegel_from_most_recent_run_on_a_tiny_history() -> None:
    runs = [run("old", days_ago=20, km=8.0, minutes=48.0), run("new", days_ago=5, km=6.0, minutes=35.0)]
    estimate = estimate_duration(runs, RunType.Z2, 12.0, now=NOW)
    assert estimate.method is EstimateMethod.RIEGEL
    assert estimate.n_runs_used == 1
    assert estimate.p50_min == pytest.approx(round(35.0 * 2.0**1.06, 1))
    assert estimate.p80_min == pytest.approx(round(35.0 * 2.0**1.06 * 1.10, 1))
    assert "Riegel" in estimate.note
    assert "6 km z2 run 5 days ago" in estimate.note


def test_default_pace_without_history() -> None:
    estimate = estimate_duration([], RunType.Z2, 10.0, now=NOW)
    assert estimate.method is EstimateMethod.DEFAULT
    assert estimate.n_runs_used == 0
    assert estimate.p50_min == 60.0  # 6:00/km
    assert estimate.p80_min == 69.0  # +15 %
    assert estimate.slot_min == 75  # 69 + 5 → 74 → 75
    assert estimate.note == "No usable z2 history → default pace 6:00/km, p80 = p50 + 15%."
    # only other run types in the history: still the default for this one
    other = [run("t", days_ago=3, km=8.0, minutes=40.0, run_type=RunType.Z4)]
    assert estimate_duration(other, RunType.Z5, 5.0, now=NOW).method is EstimateMethod.DEFAULT


def test_demo_has_no_z5_runs_so_z5_uses_the_default(predictor: DurationPredictor) -> None:
    assert predictor.estimate(RunType.Z5, 5.0).method is EstimateMethod.DEFAULT


def test_runs_at_or_after_now_are_ignored() -> None:
    past = [run(f"p{i}", days_ago=3 + i, km=10.0, minutes=60.0 + i) for i in range(3)]
    future = [run("f", days_ago=-2, km=10.0, minutes=30.0), run("at-now", days_ago=0, km=10.0, minutes=30.0)]
    assert estimate_duration(past, RunType.Z2, 10.0, now=NOW) == estimate_duration(
        past + future, RunType.Z2, 10.0, now=NOW
    )


def test_invalid_distance_is_rejected(predictor: DurationPredictor) -> None:
    for km in (0.0, -1.0, 150.0):
        with pytest.raises(ValueError, match="distance_km"):
            predictor.estimate(RunType.Z2, km)


# ─────────────────────────────── slot ───────────────────────────────
@pytest.mark.parametrize(
    ("p80", "buffer", "expected"),
    [(35.0, 5.0, 40), (35.1, 5.0, 45), (68.6, 5.0, 75), (0.5, 0.0, 5), (120.0, 5.0, 125)],
)
def test_slot_is_p80_plus_buffer_rounded_up_to_5_minutes(p80: float, buffer: float, expected: int) -> None:
    assert slot_minutes(p80, buffer) == expected


def test_slots_of_all_estimates_are_valid(predictor: DurationPredictor) -> None:
    for run_type in RunType:
        for km in (3.0, 5.0, 10.0, 16.0, 21.1, 42.195):
            estimate = predictor.estimate(run_type, km)
            assert estimate.slot_min % 5 == 0
            assert estimate.slot_min >= estimate.p80_min + 5.0
            assert estimate.slot_min < estimate.p80_min + 5.0 + 5.0
            assert estimate.p80_min >= estimate.p50_min


# ─────────────────────────────── race predictions ───────────────────────────────
def test_race_predictions_are_monotone_and_plausible(demo_runs: list[ClassifiedRun]) -> None:
    predictions = race_predictions(demo_runs, now=DEMO_NOW)
    assert [p.distance_km for p in predictions] == [5.0, 10.0, 21.0975, 42.195]
    times = [p.predicted_min for p in predictions]
    assert times == sorted(times)
    assert 20.0 <= times[0] <= 30.0
    assert times[3] > 2.0 * times[2]  # the marathon is more than twice the half
    assert all(p.n_runs_used > 0 for p in predictions)


# ─────────────────────────────── evaluation harness ───────────────────────────────
def test_backtest_uses_only_past_data(demo_runs: list[ClassifiedRun]) -> None:
    frame = rolling_origin_backtest(demo_runs, estimator="ols", min_train=30, step=10)
    assert list(frame.columns) == list(BACKTEST_COLUMNS)
    assert len(frame) == math.ceil((len(demo_runs) - 30) / 10)
    assert frame["fold"].tolist() == list(range(len(frame)))
    assert (frame["train_end"] < frame["date"]).all()
    assert frame["date"].is_monotonic_increasing
    assert frame["n_train"].is_monotonic_increasing
    assert (frame["pred_p80_min"] >= frame["pred_p50_min"]).all()
    assert set(frame["method"]) <= {m.value for m in EstimateMethod}


def test_backtest_predictions_do_not_change_when_the_future_changes(demo_runs: list[ClassifiedRun]) -> None:
    cutoff = datetime(2026, 7, 1, tzinfo=UTC)

    def distort(r: ClassifiedRun) -> ClassifiedRun:
        if r.activity.start < cutoff:
            return r
        slower = r.activity.model_copy(
            update={"moving_s": r.activity.duration_s * 2, "elapsed_s": r.activity.elapsed_s * 2}
        )
        return r.model_copy(update={"activity": slower})

    original = rolling_origin_backtest(demo_runs, min_train=30, step=7)
    distorted = rolling_origin_backtest([distort(r) for r in demo_runs], min_train=30, step=7)
    before = original["date"] < pd.Timestamp(cutoff)
    assert before.sum() > 5
    pd.testing.assert_frame_equal(original[before], distorted[before])
    assert not original[~before]["pred_p50_min"].equals(distorted[~before]["pred_p50_min"])


def test_extrapolation_split_and_evaluation(demo_runs: list[ClassifiedRun]) -> None:
    train, test = extrapolation_split(demo_runs, threshold_km=16.0)
    assert len(train) + len(test) == len(demo_runs)
    assert all(r.activity.distance_km < 16.0 for r in train)
    assert all(r.activity.distance_km >= 16.0 for r in test)

    frame = evaluate_extrapolation(train, test, estimator="huber")
    assert list(frame.columns) == list(BACKTEST_COLUMNS)
    assert len(frame) == len(test)
    assert (frame["method"] == EstimateMethod.REGRESSION.value).all()
    assert (frame["train_end"] < frame["date"]).all()
    errors = (frame["pred_p50_min"] / frame["actual_min"] - 1.0).abs()
    assert errors.mean() < 0.10  # extrapolating 16–24 km from shorter runs stays within 10 % on average
