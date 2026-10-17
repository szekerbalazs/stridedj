from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stridedj.domain import TimeSlot, TrackId
from stridedj.music.demo import build_demo_world
from stridedj.music.plays_logger import PlaysLogger

T0 = datetime(2026, 10, 10, 18, 0, tzinfo=UTC)
A, B, C = TrackId("a" * 22), TrackId("b" * 22), TrackId("c" * 22)


def _at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def _window(start_min: float, end_min: float) -> TimeSlot:
    return TimeSlot(start=_at(start_min), end=_at(end_min))


def test_record_deduplicates_on_track_and_time() -> None:
    log = PlaysLogger(Path(":memory:"))
    assert log.record([(A, _at(0)), (B, _at(4))]) == 2
    assert log.record([(A, _at(0)), (B, _at(4)), (C, _at(8))]) == 1  # a re-sync of overlapping 50 plays
    assert log.count() == 3


def test_plays_within_windows_start_inclusive_end_exclusive() -> None:
    log = PlaysLogger(Path(":memory:"))
    log.record([(A, _at(-1)), (B, _at(0)), (C, _at(30)), (A, _at(60))])
    assert log.plays_within([_window(0, 60)]) == [B, C]
    assert log.plays_within([_window(0, 60)], end_slack=timedelta(minutes=1)) == [B, C, A]


def test_repeat_plays_across_runs_count_but_overlapping_windows_do_not() -> None:
    log = PlaysLogger(Path(":memory:"))
    log.record([(A, _at(10)), (A, _at(1450)), (B, _at(20))])
    assert log.plays_within([_window(0, 60), _window(1440, 1500)]) == [A, B, A]  # A played in two runs
    assert log.plays_within([_window(0, 60), _window(5, 30)]) == [A, B]  # one play, two overlapping windows


def test_timestamps_must_be_aware() -> None:
    log = PlaysLogger(Path(":memory:"))
    with pytest.raises(ValueError, match="timezone-aware"):
        log.record([(A, datetime(2026, 10, 10, 18, 0))])  # noqa: DTZ001 (on purpose)


def test_persists_and_reports_latest_play(tmp_path: Path) -> None:
    path = tmp_path / "plays.db"
    assert PlaysLogger(path).latest_play() is None
    PlaysLogger(path).record([(A, _at(0)), (B, _at(5))])
    assert PlaysLogger(path).latest_play() == _at(5)


async def test_sync_pulls_recently_played_from_the_library() -> None:
    world = build_demo_world(seed=3)
    log = PlaysLogger(Path(":memory:"))
    assert await log.sync(world.library()) == 50  # Spotify keeps the last 50 only
    assert await log.sync(world.library()) == 0
