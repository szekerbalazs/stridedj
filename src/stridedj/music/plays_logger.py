"""Logs recently-played tracks with timestamps, so plays can be matched to run windows later.

Spotify's `GET /me/player/recently-played` only returns the last 50 plays, so a play from last week's long run is gone
by the time we want it. Calling `sync` regularly (on every planning request, or on a timer) accumulates the history
locally; `plays_within` then finds what the user listened to *while running*, which is the strongest available signal
of "music I run to".

Only (track id, played_at) is stored: no titles, no context URIs. Rows are de-duplicated on that pair, so syncing the
same 50 plays twice is a no-op.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from stridedj.config import AppSettings
from stridedj.domain import TimeSlot, TrackId
from stridedj.ports import MusicLibraryPort

PLAYS_FILENAME = "music_plays.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plays (
    track_id TEXT NOT NULL,
    played_at_ms INTEGER NOT NULL,  -- unix milliseconds, UTC
    PRIMARY KEY (track_id, played_at_ms)
)
"""
_INDEX = "CREATE INDEX IF NOT EXISTS plays_by_time ON plays (played_at_ms)"


def default_plays_path(app: AppSettings) -> Path:
    return app.data_dir / PLAYS_FILENAME


def _to_ms(moment: datetime) -> int:
    if moment.tzinfo is None:
        raise ValueError("played_at must be timezone-aware")
    return round(moment.timestamp() * 1000)


class PlaysLogger:
    """SQLite-backed log of plays. Pass ``Path(":memory:")`` for an in-process log (tests, demo)."""

    def __init__(self, path: Path) -> None:
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute(_SCHEMA)
        self._db.execute(_INDEX)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def record(self, plays: Sequence[tuple[TrackId, datetime]]) -> int:
        """Store plays (timezone-aware timestamps); returns how many were new."""
        rows = [(track_id, _to_ms(played_at)) for track_id, played_at in plays]
        before = self._db.total_changes
        self._db.executemany("INSERT OR IGNORE INTO plays (track_id, played_at_ms) VALUES (?, ?)", rows)
        self._db.commit()
        return self._db.total_changes - before

    async def sync(self, library: MusicLibraryPort) -> int:
        """Pull the last plays from Spotify into the log; returns how many were new."""
        recent = await library.recently_played()
        return self.record([(track.track_id, played_at) for track, played_at in recent])

    def count(self) -> int:
        row = self._db.execute("SELECT COUNT(*) FROM plays").fetchone()
        return int(row[0])

    def plays_within(self, windows: Sequence[TimeSlot], *, end_slack: timedelta = timedelta(0)) -> list[TrackId]:
        """Track ids played inside any of the windows, oldest first.

        A track played during two runs appears twice (more evidence → more weight downstream); a single play inside two
        overlapping windows is counted once. `end_slack` widens each window's end: Spotify's `played_at` is believed
        to be stamped when playback of the track *ends*, so the last track of a run can land just after the watch
        stopped (to be confirmed on live data; the default is no slack).
        """
        found: dict[tuple[str, int], None] = {}
        for window in windows:
            start_ms = _to_ms(window.start)
            end_ms = _to_ms(window.end + end_slack)
            rows = self._db.execute(
                "SELECT track_id, played_at_ms FROM plays WHERE played_at_ms >= ? AND played_at_ms < ?",
                (start_ms, end_ms),
            ).fetchall()
            found.update({(track_id, played_at_ms): None for track_id, played_at_ms in rows})
        ordered = sorted(found, key=lambda row: (row[1], row[0]))
        return [TrackId(track_id) for track_id, _ in ordered]

    def latest_play(self) -> datetime | None:
        """When the most recent logged play happened (useful to tell how fresh the log is)."""
        row = self._db.execute("SELECT MAX(played_at_ms) FROM plays").fetchone()
        return None if row[0] is None else datetime.fromtimestamp(row[0] / 1000, tz=UTC)
