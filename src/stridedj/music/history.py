"""The playlists StrideDJ created and the runner's feedback on them: the memory behind rotation and learning.

Recorded by the executor right after a playlist was created on Spotify (so only approved plans count: re-planning in
a chat keeps its songs, the next run gets different ones). Feedback comes from the bot's buttons:
* per playlist, 👍 / 👎: applies to the playlist's NEW songs (running-playlist favourites are not judged by it);
* per song, 👎 (toggle): that song is never suggested again, favourite or not.
👎 is the signal the style model never had (it learns from positives only: PU learning); 👍 adds positives.
Only ids are stored: draft id, run type, track ids in order, ratings. No titles, no URLs.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Set
from pathlib import Path

from stridedj.config import AppSettings
from stridedj.domain import PlaylistDraft, TrackId

HISTORY_FILENAME = "music_history.db"

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS playlists (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,  -- creation order
        draft_id TEXT NOT NULL UNIQUE,
        run_type TEXT NOT NULL,
        created_at_s INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS playlist_tracks (
        draft_id TEXT NOT NULL REFERENCES playlists (draft_id),
        position INTEGER NOT NULL,
        track_id TEXT NOT NULL,
        PRIMARY KEY (draft_id, position)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS playlist_feedback (
        draft_id TEXT PRIMARY KEY REFERENCES playlists (draft_id),
        rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),  -- the latest tap wins
        at_s INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS song_dislikes (
        track_id TEXT PRIMARY KEY,
        at_s INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
    )
    """,
)


def default_history_path(app: AppSettings) -> Path:
    return app.data_dir / HISTORY_FILENAME


class PlaylistHistory:
    """SQLite-backed `PlaylistHistoryPort`. Pass ``Path(":memory:")`` for an in-process store (tests, demo, evals)."""

    def __init__(self, path: Path) -> None:
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        for statement in _SCHEMA:
            self._db.execute(statement)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def record_created(self, draft: PlaylistDraft) -> None:
        """Remember a created playlist. Recording the same draft twice is a no-op (the executor may retry)."""
        cursor = self._db.execute(
            "INSERT OR IGNORE INTO playlists (draft_id, run_type) VALUES (?, ?)", (draft.draft_id, draft.run_type.value)
        )
        if cursor.rowcount:
            self._db.executemany(
                "INSERT INTO playlist_tracks (draft_id, position, track_id) VALUES (?, ?, ?)",
                [(draft.draft_id, position, track_id) for position, track_id in enumerate(draft.track_ids)],
            )
        self._db.commit()

    def recent_playlists(self, *, last: int) -> list[tuple[TrackId, ...]]:
        """The tracks of the last `last` created playlists (each in order), newest playlist first."""
        drafts = self._db.execute("SELECT draft_id FROM playlists ORDER BY seq DESC LIMIT ?", (max(0, last),))
        return [self.track_ids(draft_id) for (draft_id,) in drafts.fetchall()]

    def track_ids(self, draft_id: str) -> tuple[TrackId, ...]:
        """The tracks of a created playlist in order; empty if StrideDJ never created it."""
        rows = self._db.execute(
            "SELECT track_id FROM playlist_tracks WHERE draft_id = ? ORDER BY position", (draft_id,)
        ).fetchall()
        return tuple(TrackId(track_id) for (track_id,) in rows)

    def rate_playlist(self, draft_id: str, rating: int) -> bool:
        """👍 (+1) or 👎 (-1) for a created playlist; False (and nothing stored) if it is unknown."""
        if rating not in (-1, 1):
            raise ValueError("rating must be -1 or 1")
        if not self._db.execute("SELECT 1 FROM playlists WHERE draft_id = ?", (draft_id,)).fetchone():
            return False
        self._db.execute(
            "INSERT INTO playlist_feedback (draft_id, rating) VALUES (?, ?) "
            "ON CONFLICT (draft_id) DO UPDATE SET rating = excluded.rating, at_s = strftime('%s', 'now')",
            (draft_id, rating),
        )
        self._db.commit()
        return True

    def toggle_song_dislike(self, track_id: str) -> bool:
        """Dislike a song, or undo the dislike; returns whether it is disliked now."""
        if self._db.execute("DELETE FROM song_dislikes WHERE track_id = ?", (track_id,)).rowcount:
            self._db.commit()
            return False
        self._db.execute("INSERT INTO song_dislikes (track_id) VALUES (?)", (track_id,))
        self._db.commit()
        return True

    def disliked_songs(self) -> set[TrackId]:
        return {TrackId(track_id) for (track_id,) in self._db.execute("SELECT track_id FROM song_dislikes")}

    def track_ratings(self, *, favourites: Set[TrackId]) -> dict[TrackId, int]:
        """+1 / -1 per track from all feedback. Playlist ratings apply to the new songs only (not to `favourites`,
        the running playlists); a song dislike always applies and wins over a playlist 👍."""
        ratings: dict[TrackId, int] = {}
        rows = self._db.execute(
            "SELECT t.track_id, f.rating FROM playlist_feedback f JOIN playlist_tracks t ON t.draft_id = f.draft_id "
            "ORDER BY f.at_s"
        ).fetchall()
        for track_id, rating in rows:  # later feedback overrides earlier
            if TrackId(track_id) not in favourites:
                ratings[TrackId(track_id)] = int(rating)
        for track_id in self.disliked_songs():
            ratings[track_id] = -1
        return ratings
