"""SQLite persistence of activities (allowlisted columns only).

Why SQLite (stdlib `sqlite3`): one user, a few hundred rows per year, no server to run, a single file that is easy
to back up or delete (data minimisation: `rm data/stridedj.db` forgets everything).

Design choices:
* The schema *is* the allowlist: one column per `Activity` field and nothing else, so even a buggy adapter cannot
  persist a GPS track or an activity name; there is nowhere to put it.
* Laps and zone seconds are JSON columns: they are always read together with their activity and never queried.
* Start times are stored as fixed-format UTC ISO-8601 strings, whose lexicographic order equals chronological order,
  so `ORDER BY` / `WHERE start_utc >= ?` work on the text column with an index.
* Per-sample series (`RunSeries`, for interval detection) live in their own table as one zlib-compressed JSON blob
  per run (~2-4 k samples × 6 channels ≈ 30 KB): always read whole, never queried. Again the schema is the allowlist:
  the blob is built from the `RunSeries` fields, which have no GPS.
* A new connection per operation: no connection is shared across threads (the bot runs tool calls in threads), and
  `with conn:` gives one transaction per upsert batch.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import zlib
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

from stridedj.config import AppSettings
from stridedj.domain import Activity, ActivityId, Lap, RunSeries

DB_FILENAME = "stridedj.db"
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS activities (
    activity_id      TEXT PRIMARY KEY,
    source           TEXT NOT NULL,
    start_utc        TEXT NOT NULL,
    distance_m       REAL NOT NULL,
    elapsed_s        REAL NOT NULL,
    moving_s         REAL,
    elevation_gain_m REAL,
    avg_hr_bpm       REAL,
    max_hr_bpm       REAL,
    avg_cadence_spm  REAL,
    hr_zone_s_json   TEXT,
    laps_json        TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_activities_start ON activities (start_utc);
CREATE TABLE IF NOT EXISTS run_series (
    activity_id TEXT PRIMARY KEY,
    payload     BLOB NOT NULL  -- zlib(JSON of the RunSeries channels)
);
"""

_COLUMNS = (
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
)

_UPSERT = (
    f"INSERT INTO activities ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' for _ in _COLUMNS)}) "  # noqa: S608
    "ON CONFLICT(activity_id) DO UPDATE SET " + ", ".join(f"{column} = excluded.{column}" for column in _COLUMNS[1:])
)
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM activities"  # noqa: S608  (constant column list, no user input)

Row = tuple[
    str, str, str, float, float, float | None, float | None, float | None, float | None, float | None, str | None, str
]


def _utc_text(moment: datetime) -> str:
    """Fixed-width UTC text (microsecond precision), so string order is time order."""
    if moment.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware")
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def _to_row(activity: Activity) -> Row:
    return (
        activity.activity_id,
        activity.source,
        _utc_text(activity.start),
        activity.distance_m,
        activity.elapsed_s,
        activity.moving_s,
        activity.elevation_gain_m,
        activity.avg_hr_bpm,
        activity.max_hr_bpm,
        activity.avg_cadence_spm,
        json.dumps(list(activity.hr_zone_s)) if activity.hr_zone_s is not None else None,
        json.dumps([lap.model_dump(mode="json") for lap in activity.laps]),
    )


def _from_row(row: Row) -> Activity:
    (activity_id, source, start, distance, elapsed, moving, gain, avg_hr, max_hr, cadence, zones_json, laps_json) = row
    zones = json.loads(zones_json) if zones_json is not None else None
    return Activity.model_validate(
        {
            "activity_id": ActivityId(activity_id),
            "source": source,
            "start": datetime.fromisoformat(start),
            "distance_m": distance,
            "elapsed_s": elapsed,
            "moving_s": moving,
            "elevation_gain_m": gain,
            "avg_hr_bpm": avg_hr,
            "max_hr_bpm": max_hr,
            "avg_cadence_spm": cadence,
            "hr_zone_s": tuple(zones) if zones is not None else None,
            "laps": tuple(Lap.model_validate(lap) for lap in json.loads(laps_json)),
        }
    )


class SqliteRunRepository:
    """`RunRepository` backed by one SQLite file. The parent directory is created on first use."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        with contextlib.suppress(OSError):
            self._path.chmod(0o600)  # personal data: owner-only

    @classmethod
    def from_settings(cls, app: AppSettings) -> SqliteRunRepository:
        """The default location: `<data_dir>/stridedj.db`."""
        return cls(app.data_dir / DB_FILENAME)

    @property
    def path(self) -> Path:
        return self._path

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        try:
            with conn:  # commit on success, roll back on error
                yield conn
        finally:
            conn.close()

    def upsert_activities(self, activities: Sequence[Activity]) -> int:
        """Insert or update by `activity_id`; returns the number of rows written."""
        if not activities:
            return 0
        with self._connect() as conn:
            conn.executemany(_UPSERT, [_to_row(activity) for activity in activities])
        return len(activities)

    def list_activities(self, *, since: datetime | None = None) -> Sequence[Activity]:
        """Activities that started at or after `since` (all if None), newest first."""
        with self._connect() as conn:
            if since is None:
                cursor = conn.execute(f"{_SELECT} ORDER BY start_utc DESC")
            else:
                cursor = conn.execute(f"{_SELECT} WHERE start_utc >= ? ORDER BY start_utc DESC", (_utc_text(since),))
            rows: list[Row] = cursor.fetchall()
        return [_from_row(row) for row in rows]

    def upsert_series(self, series: RunSeries) -> None:
        payload = zlib.compress(series.model_dump_json(exclude={"activity_id"}).encode(), level=6)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO run_series (activity_id, payload) VALUES (?, ?) "
                "ON CONFLICT (activity_id) DO UPDATE SET payload = excluded.payload",
                (series.activity_id, payload),
            )

    def get_series(self, activity_id: ActivityId) -> RunSeries | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM run_series WHERE activity_id = ?", (activity_id,)).fetchone()
        if row is None:
            return None
        channels = json.loads(zlib.decompress(row[0]))
        return RunSeries.model_validate({"activity_id": activity_id, **channels})

    def series_ids(self) -> set[ActivityId]:
        with self._connect() as conn:
            return {ActivityId(a) for (a,) in conn.execute("SELECT activity_id FROM run_series")}

    def count(self) -> int:
        with self._connect() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM activities").fetchone()
        return int(n)
