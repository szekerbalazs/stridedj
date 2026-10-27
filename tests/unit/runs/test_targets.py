"""Interval pace targets: the runner's starting targets, adapted by each session."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stridedj.runs.intervals import IntervalAnalysis, Rep
from stridedj.runs.targets import Verdict, adapt_targets, judge, pace_for, round_to_5, targets_from_history

SINCE = datetime(2026, 10, 27, tzinfo=UTC)


def _session(rep_m: int, *paces: float) -> IntervalAnalysis:
    reps = tuple(
        Rep(distance_m=rep_m, duration_s=rep_m / 1000 * p, pace_s_per_km=p, recovery_s=90.0, recovery_m=0.0)
        for p in paces
    )
    return IntervalAnalysis(is_intervals=True, reason="test", reps=reps)


@pytest.mark.parametrize(
    ("paces", "verdict"),
    [
        ((224, 225, 223, 226), Verdict.CLEAN),  # within target + 2 s
        ((224, 225, 223, 229), Verdict.MISSED),  # one rep 4 s slow
        ((215, 215, 222, 223), Verdict.FADED),  # on target, but the end is > 3 % slower than the start
    ],
)
def test_judge(paces: tuple[float, ...], verdict: Verdict) -> None:
    assert judge(paces, 225.0) is verdict


def test_clean_sessions_make_the_target_faster_misses_keep_it() -> None:
    sessions = [
        (SINCE + timedelta(days=7), _session(400, 224, 225, 223, 225)),  # clean → 3:43
        (SINCE + timedelta(days=14), _session(400, 224, 228, 226, 230)),  # missed (vs 3:43) → stays
        (SINCE + timedelta(days=21), _session(400, 222, 223, 221, 222)),  # clean → 3:41
    ]
    targets, verdicts = adapt_targets({400: 225.0}, since=SINCE, sessions=sessions, now=SINCE + timedelta(days=22))
    assert [v.verdict for v in verdicts] == [Verdict.CLEAN, Verdict.MISSED, Verdict.CLEAN]
    assert targets[400] == 221.0


def test_six_weeks_without_a_distance_eases_its_target() -> None:
    targets, _ = adapt_targets({800: 250.0}, since=SINCE, sessions=[], now=SINCE + timedelta(days=50))
    assert targets[800] == 255.0


def test_sessions_before_the_targets_were_set_do_not_count() -> None:
    old = [(SINCE - timedelta(days=3), _session(400, 200, 200, 200))]
    targets, verdicts = adapt_targets({400: 225.0}, since=SINCE, sessions=old, now=SINCE)
    assert targets[400] == 225.0 and verdicts == []


def test_history_targets_from_the_last_sessions() -> None:
    """Three 800 m sessions, the latest 19 weeks ago: median of their medians, rounded to 5 s, + 5 s for the gap."""
    now = SINCE
    sessions = [
        (now - timedelta(days=154), _session(800, 241, 241, 241)),
        (now - timedelta(days=147), _session(800, 243, 243, 243)),
        (now - timedelta(days=133), _session(800, 238, 238, 238)),
        (now - timedelta(days=14), _session(400, 210, 210, 210)),  # 3:30, recent: no 6-week ease
        (now - timedelta(days=20), _session(1200, 290, 290, 290)),  # not an anchor distance: ignored
    ]
    targets = targets_from_history(sessions, now=now)
    assert targets == {800: 245.0, 400: 210.0}


def test_other_distances_follow_the_pace_curve_through_the_targets() -> None:
    targets = {400: 225.0, 800: 250.0}
    assert pace_for(400, targets) == 225.0
    paces = [pace_for(d, targets) for d in (200, 600, 1000, 1200)]
    assert paces == sorted(paces)  # longer reps, slower pace
    assert 260 <= pace_for(1200, targets) <= 272  # ~4:26
    assert pace_for(1000, {800: 250.0}) > 250  # one target: Riegel's exponent


def test_round_to_5() -> None:
    assert [round_to_5(x) for x in (212.4, 212.6, 235.4, 237.6)] == [210.0, 215.0, 235.0, 240.0]
