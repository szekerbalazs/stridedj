from __future__ import annotations

import pytest

from stridedj.runs.sessions import interval_session_distance


def test_a_missing_recovery_defaults_to_half_a_rep_between_reps() -> None:
    d = interval_session_distance(reps=4, rep_m=800, warmup_km=2, cooldown_km=1)
    assert (d.total_km, d.work_km, d.recovery_km, d.recovery_assumed) == (7.4, 3.2, 1.2, True)


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
