from __future__ import annotations

import sqlite3
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from stridedj.config import AppSettings
from stridedj.domain import Activity, ActivityId, Lap, LapIntensity
from stridedj.runs.sources.demo import generate_history
from stridedj.runs.store import DB_FILENAME, SqliteRunRepository

T0 = datetime(2026, 10, 1, 7, 30, tzinfo=UTC)


def make_activity(activity_id: str, start: datetime, km: float = 10.0) -> Activity:
    return Activity(
        activity_id=ActivityId(activity_id),
        source="garmin",
        start=start,
        distance_m=km * 1000.0,
        elapsed_s=3700.0,
        moving_s=3600.0,
        elevation_gain_m=55.0,
        avg_hr_bpm=138.0,
        max_hr_bpm=151.0,
        avg_cadence_spm=167.5,
        hr_zone_s=(300.0, 2800.0, 400.0, 100.0, 0.0),
        laps=(
            Lap(start_offset_s=0, duration_s=600, distance_m=1800, avg_hr_bpm=130, intensity=LapIntensity.WARMUP),
            Lap(
                start_offset_s=600, duration_s=180, distance_m=800, avg_cadence_spm=186, intensity=LapIntensity.INTERVAL
            ),
        ),
    )


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    repo = SqliteRunRepository(tmp_path / "db" / "runs.db")
    zurich_start = datetime(2026, 10, 1, 9, 30, tzinfo=ZoneInfo("Europe/Zurich"))
    original = make_activity("g1", zurich_start)
    assert repo.upsert_activities([original]) == 1
    (loaded,) = repo.list_activities()
    assert loaded == original  # aware datetimes compare by instant
    assert loaded.start.utcoffset() == timedelta(0)  # stored and returned as UTC
    assert loaded.laps == original.laps
    assert loaded.hr_zone_s == original.hr_zone_s


def test_upsert_updates_by_activity_id(tmp_path: Path) -> None:
    repo = SqliteRunRepository(tmp_path / "runs.db")
    repo.upsert_activities([make_activity("g1", T0, km=10.0)])
    repo.upsert_activities([make_activity("g1", T0, km=10.4)])
    assert repo.count() == 1
    assert repo.list_activities()[0].distance_km == 10.4
    assert repo.upsert_activities([]) == 0


def test_list_since_is_inclusive_and_newest_first(tmp_path: Path) -> None:
    repo = SqliteRunRepository(tmp_path / "runs.db")
    repo.upsert_activities([make_activity(f"g{i}", T0 + timedelta(days=i)) for i in range(5)])
    since = T0 + timedelta(days=2)
    listed = repo.list_activities(since=since)
    assert [a.activity_id for a in listed] == ["g4", "g3", "g2"]
    # a `since` in another timezone denotes the same instant
    assert len(repo.list_activities(since=since.astimezone(ZoneInfo("America/New_York")))) == 3


def test_demo_history_round_trips(tmp_path: Path) -> None:
    repo = SqliteRunRepository(tmp_path / "runs.db")
    history = generate_history()
    repo.upsert_activities(history)
    assert list(repo.list_activities()) == history


def test_schema_is_the_allowlist_and_file_is_private(tmp_path: Path) -> None:
    repo = SqliteRunRepository(tmp_path / "nested" / "dir" / "runs.db")
    assert repo.path.exists()
    assert stat.S_IMODE(repo.path.stat().st_mode) == 0o600
    with sqlite3.connect(repo.path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(activities)")}
    assert columns == {
        "activity_id",
        "source",
        "start_utc",
        "distance_m",
        "elapsed_s",
        "moving_s",
        "elevation_gain_m",
        "avg_hr_bpm",
        "max_hr_bpm",
        "avg_cadence_spm",
        "hr_zone_s_json",
        "laps_json",
    }
    assert not any(word in " ".join(columns) for word in ("lat", "lon", "name", "polyline", "description"))


def test_from_settings_uses_data_dir(tmp_path: Path) -> None:
    repo = SqliteRunRepository.from_settings(AppSettings(data_dir=tmp_path / "data"))
    assert repo.path == tmp_path / "data" / DB_FILENAME
    assert repo.path.parent.is_dir()
