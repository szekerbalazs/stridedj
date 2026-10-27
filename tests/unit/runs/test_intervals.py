"""Interval detection on synthetic series shaped like typical sessions, tempo runs and long runs with a break."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from stridedj.domain import ActivityId, RunSeries
from stridedj.runs.intervals import analyse_series, describe, nominal_distance_m

EASY, FAST = 1000 / 340, 1000 / 225  # 5:40/km and 3:45/km in m/s


@dataclass
class _Builder:
    """Appends 1-s samples. `stop` only moves the wall clock: like Garmin, no sample exists while the watch is
    stopped, and the first sample after it has elapsed time jumped ahead while the timer has not."""

    elapsed: list[float] = field(default_factory=lambda: [0.0])
    timer: list[float] = field(default_factory=lambda: [0.0])
    distance: list[float] = field(default_factory=lambda: [0.0])
    speed: list[float | None] = field(default_factory=lambda: [0.0])
    hr: list[float | None] = field(default_factory=lambda: [100.0])
    cadence: list[float | None] = field(default_factory=lambda: [None])
    paused_s: float = 0.0

    def run(self, seconds: float, mps: float, hr_from: float, hr_to: float, cadence: float = 165) -> _Builder:
        n = int(seconds)
        for k in range(1, n + 1):
            self.elapsed.append(self.elapsed[-1] + 1 + self.paused_s)
            self.paused_s = 0.0
            self.timer.append(self.timer[-1] + 1)
            self.distance.append(self.distance[-1] + mps)
            self.speed.append(mps)
            self.hr.append(hr_from + (hr_to - hr_from) * k / n)
            self.cadence.append(cadence if mps > 1 else None)
        return self

    def stop(self, seconds: float) -> _Builder:
        self.paused_s += seconds
        return self

    def series(self) -> RunSeries:
        return RunSeries(
            activity_id=ActivityId("1"),
            elapsed_s=tuple(self.elapsed),
            timer_s=tuple(self.timer),
            distance_m=tuple(self.distance),
            speed_mps=tuple(self.speed),
            hr_bpm=tuple(self.hr),
            cadence_spm=tuple(self.cadence),
        )


def _reps_with_watch_stops(n: int = 5, rep_s: float = 225, rest_s: float = 135) -> RunSeries:
    b = _Builder().run(660, EASY, 100, 148)  # 2 km warm-up
    b.stop(570)
    for k in range(n):
        b.run(rep_s, FAST, 120, 178, cadence=170)
        if k < n - 1:
            b.stop(rest_s)
    b.stop(400)
    return b.run(450, EASY, 115, 150).series()


def test_reps_with_the_watch_stopped_are_found_with_distances_and_paces() -> None:
    """No samples exist during the rests, so the fast stretches alone would look like one fast run."""
    result = analyse_series(_reps_with_watch_stops())
    assert result.is_intervals
    assert describe(result.reps) == "5 × 1000 m"
    assert all(3 * 60 + 40 <= r.pace_s_per_km <= 3 * 60 + 50 for r in result.reps)
    assert all(r.recovery_s == pytest.approx(136, abs=2) for r in result.reps[:-1])  # 135 s stop + 1 s step
    assert all((r.recovery_hr_drop_bpm or 0) > 40 for r in result.reps[:-1])  # 178 at the stop, 120 at the restart
    assert result.work_cadence_spm == pytest.approx(170)


def test_standing_rests_with_the_timer_running_are_found_by_pace_and_heart_rate() -> None:
    """No watch stop at all; the rests show as no movement and a falling heart rate."""
    b = _Builder().run(540, EASY, 100, 150)
    for _ in range(4):
        b.run(84, 1000 / 210, 155, 185, cadence=175).run(80, 0.0, 185, 160)
    result = analyse_series(b.run(300, EASY, 150, 155).series())
    assert result.is_intervals
    assert describe(result.reps) == "4 × 400 m"


def test_a_traffic_light_inside_a_rep_does_not_split_it() -> None:
    b = _Builder().run(600, EASY, 100, 145)
    for _ in range(4):
        b.run(110, FAST, 120, 170).stop(15).run(115, FAST, 165, 178)  # a 15-s stop: not a rest
        b.stop(120)
    result = analyse_series(b.run(300, EASY, 120, 140).series())
    assert describe(result.reps) == "4 × 1000 m"


@pytest.mark.parametrize(
    ("name", "builder", "reason"),
    [
        ("steady Z2", lambda: _Builder().run(3600, EASY, 120, 150), "steady pace"),
        (
            "tempo run",
            lambda: _Builder().run(540, EASY, 110, 150).run(1000, FAST, 155, 175).run(500, EASY, 170, 150),
            "fast block",
        ),
        (
            "long run with a long break",
            lambda: _Builder().run(3000, EASY, 120, 150).stop(1700).run(2400, EASY, 110, 152),
            "steady pace",
        ),
    ],
)
def test_other_runs_are_not_intervals(name: str, builder: object, reason: str) -> None:
    result = analyse_series(builder().series())  # type: ignore[operator]
    assert not result.is_intervals, name
    assert reason in result.reason


def test_short_activations_are_not_an_interval_session() -> None:
    """4 × 200 m with stops is the right shape, but only 0.8 km fast."""
    b = _Builder().run(360, EASY, 110, 140)
    for _ in range(4):
        b.run(42, 1000 / 210, 140, 170).stop(90)
    result = analyse_series(b.run(330, EASY, 130, 140).series())
    assert not result.is_intervals
    assert "activations" in result.reason


def test_fast_blocks_without_a_heart_rate_drop_are_not_intervals() -> None:
    """Short pick-ups in town: heart rate barely moves between them."""
    b = _Builder().run(600, EASY, 140, 150)
    for _ in range(5):
        b.run(60, FAST, 150, 156).run(60, EASY, 156, 152)
    assert not analyse_series(b.run(600, EASY, 150, 152).series()).is_intervals


@pytest.mark.parametrize(("measured", "nominal"), [(985, 1000), (820, 800), (410, 400), (230, 250), (700, 700)])
def test_gps_distances_snap_to_the_intended_rep(measured: float, nominal: int) -> None:
    assert nominal_distance_m(measured) == nominal
