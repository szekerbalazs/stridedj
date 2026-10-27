from __future__ import annotations

import pytest

from stridedj.domain import EstimateMethod, RunType
from stridedj.runs.sessions import build_session_options, interval_session_distance, rest_after


def test_a_missing_recovery_is_a_standing_rest() -> None:
    """Standing rests by default: no distance between the reps unless a jog is named."""
    d = interval_session_distance(reps=4, rep_m=800, warmup_km=2, cooldown_km=1)
    assert (d.total_km, d.work_km, d.recovery_km, d.recovery_assumed) == (6.2, 3.2, 0.0, True)


def test_a_given_recovery_is_used() -> None:
    d = interval_session_distance(reps=5, rep_m=1000, warmup_km=2, cooldown_km=2, recovery_m=400)
    assert (d.total_km, d.recovery_km, d.recovery_assumed) == (10.6, 1.6, False)


def test_a_standing_recovery_counts_as_zero() -> None:
    d = interval_session_distance(reps=6, rep_m=400, warmup_km=1.5, cooldown_km=1.5, recovery_m=0)
    assert (d.total_km, d.recovery_km) == (5.4, 0.0)


def test_a_single_rep_has_no_recovery() -> None:
    d = interval_session_distance(reps=1, rep_m=3000, warmup_km=0, cooldown_km=0)
    assert (d.total_km, d.recovery_km) == (3.0, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reps": 0, "rep_m": 800, "warmup_km": 2, "cooldown_km": 1},
        {"reps": 4, "rep_m": 0, "warmup_km": 2, "cooldown_km": 1},
        {"reps": 4, "rep_m": 800, "warmup_km": -1, "cooldown_km": 1},
        {"reps": 4, "rep_m": 800, "warmup_km": 2, "cooldown_km": 1, "recovery_m": -200},
    ],
)
def test_impossible_sessions_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        interval_session_distance(**kwargs)  # type: ignore[arg-type]


# ─────────────────────────────── session options ───────────────────────────────
TARGETS = {400: 225.0, 800: 250.0}  # 3:45 and 4:10


def _options(**overrides: object) -> list:  # type: ignore[type-arg]
    kwargs: dict[str, object] = {
        "pace_of": lambda d: TARGETS.get(d, 256.0),
        "easy_pace_s_per_km": 370.0,
        "overhead_s": 120.0,
        "spread": 1.06,
        "buffer_min": 5.0,
        "n_history": 12,
    }
    kwargs.update(overrides)
    return build_session_options(**kwargs)  # type: ignore[arg-type]


def test_short_medium_and_pyramid_with_distance_reps_and_standing_rests() -> None:
    short, medium, pyramid = _options()
    assert (short.reps_m, short.target_paces_s_per_km, short.rest_s) == ((400,) * 8, (225.0,) * 8, (75,) * 7)
    assert (medium.reps_m, medium.rest_s[0]) == ((800,) * 6, 90)
    assert pyramid.reps_m == (400, 800, 1200, 800, 400)
    assert short.total_km == 2.0 + 3.2 + 1.5  # standing rests add time, not distance
    assert short.main_set == "8 × 400 m @ 3:45, 75 s rest"
    assert short.outline.startswith("2 km warm-up · 8 × 400 m @ 3:45")


def test_the_duration_comes_from_the_structure() -> None:
    short = _options(measured_rest_s=lambda r: 100.0)[0]  # the rests actually taken, not the planned 75 s
    easy = 3.5 * 370
    work = 8 * 0.4 * 225
    expected = (easy + work + 7 * 100 + 120) / 60
    d = short.duration
    assert d.method is EstimateMethod.SESSION and d.run_type is RunType.INTERVALS
    assert d.p50_min == pytest.approx(expected, abs=0.1)
    assert d.p80_min == pytest.approx(expected * 1.06, abs=0.1)
    assert d.slot_min % 5 == 0 and d.slot_min >= d.p80_min + 5


def test_a_time_limit_shrinks_the_rep_count_or_drops_the_option() -> None:
    full = {o.kind: o for o in _options()}
    limited = {o.kind: o for o in _options(max_minutes=full["short"].duration.slot_min - 5)}
    assert len(limited["short"].reps_m) < 8
    assert all(o.duration.slot_min <= full["short"].duration.slot_min - 5 for o in limited.values())
    assert _options(max_minutes=20) == []


def test_planned_rest_by_rep_distance() -> None:
    assert [rest_after(d) for d in (200, 400, 600, 800, 1000, 1600)] == [75, 75, 90, 90, 120, 120]


def test_sessions_sized_to_a_requested_total() -> None:
    """'9 km in total': the reps stay, warm-up and cool-down (60/40) fill the rest."""
    short, medium, pyramid = _options(total_km=9.0)
    assert [o.total_km for o in (short, medium, pyramid)] == [9.0, 9.0, 9.0]
    assert (short.warmup_km, short.cooldown_km) == (3.5, 2.3)  # 9 - 3.2 = 5.8 km easy
    assert (medium.warmup_km, medium.cooldown_km) == (2.5, 1.7)  # 9 - 4.8 = 4.2 km easy
    assert short.duration.distance_km == 9.0


def test_a_short_total_means_fewer_reps_or_no_option() -> None:
    options = {o.kind: o for o in _options(total_km=5.0)}  # 5 km: 8 × 400 leaves 1.8 km easy, 6 × 800 only 0.2
    assert options["short"].reps_m == (400,) * 8
    assert options["medium"].reps_m == (800,) * 4  # 3.2 km of reps leaves 1.8 km
    assert "pyramid" not in options  # 3.6 km of reps leaves 1.4 km < 1.5
