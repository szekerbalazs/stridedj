"""A small SQLite key/value cache for JSON values, with an optional per-entry TTL.

Why a cache at all: ReccoBeats has no SLA and a rate limit it does not publish, and Spotify's development mode removed
the batch GETs (one request per artist/track now). Audio features and artist genres change rarely, so caching them
turns "hundreds of calls per playlist" into "a handful of calls for the new tracks only".

Why SQLite: it is in the standard library, survives restarts, and one file per cache is easy to inspect or delete.
Values are JSON (not pickle), so a corrupted or tampered file cannot execute code when it is read back.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stridedj.config import AppSettings
from stridedj.ports import Clock

CACHE_FILENAME = "music_cache.db"
_SQLITE_MAX_VARIABLES = 500  # stay well below SQLite's bound-parameter limit when reading many keys at once

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at_s REAL  -- unix seconds; NULL = never expires
)
"""


class UtcClock:
    """The real clock (UTC). Tests inject a frozen one instead."""

    def now(self) -> datetime:
        return datetime.now(UTC)


def default_cache_path(app: AppSettings) -> Path:
    """Where the music cache lives by default: next to the other local data (git-ignored)."""
    return app.data_dir / CACHE_FILENAME


class JsonCache:
    """Key → JSON value, with optional expiry.

    Keys are namespaced by the caller (e.g. ``"reccobeats:features:<track id>"``). Expired entries are treated as
    absent on read and removed lazily (or eagerly with `purge_expired`). Pass ``Path(":memory:")`` for a throwaway
    in-process cache (tests, demo mode).
    """

    def __init__(self, path: Path, *, clock: Clock | None = None) -> None:
        self._clock: Clock = clock or UtcClock()
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def get(self, key: str) -> Any | None:
        """The cached value, or None if absent or expired."""
        return self.get_many([key]).get(key)

    def get_many(self, keys: Iterable[str]) -> dict[str, Any]:
        """All present, unexpired entries among `keys` (missing keys are simply absent from the result)."""
        wanted = list(dict.fromkeys(keys))
        now_s = self._now_s()
        found: dict[str, Any] = {}
        for start in range(0, len(wanted), _SQLITE_MAX_VARIABLES):
            chunk = wanted[start : start + _SQLITE_MAX_VARIABLES]
            placeholders = ",".join("?" * len(chunk))
            rows = self._db.execute(
                f"SELECT key, value, expires_at_s FROM kv WHERE key IN ({placeholders})",  # noqa: S608 (placeholders only)
                chunk,
            ).fetchall()
            for key, raw, expires_at_s in rows:
                if expires_at_s is None or expires_at_s > now_s:
                    found[key] = json.loads(raw)
        return found

    def set(self, key: str, value: Any, *, ttl_s: float | None = None) -> None:
        """Store `value` (anything `json.dumps` accepts). `ttl_s=None` keeps it until overwritten or deleted."""
        self.set_many({key: value}, ttl_s=ttl_s)

    def set_many(self, items: Mapping[str, Any], *, ttl_s: float | None = None) -> None:
        if ttl_s is not None and ttl_s <= 0:
            raise ValueError("ttl_s must be > 0 (or None for no expiry)")
        expires_at_s = None if ttl_s is None else self._now_s() + ttl_s
        rows = [(key, json.dumps(value, separators=(",", ":")), expires_at_s) for key, value in items.items()]
        self._db.executemany("INSERT OR REPLACE INTO kv (key, value, expires_at_s) VALUES (?, ?, ?)", rows)
        self._db.commit()

    def delete(self, keys: Sequence[str]) -> None:
        self._db.executemany("DELETE FROM kv WHERE key = ?", [(key,) for key in keys])
        self._db.commit()

    def purge_expired(self) -> int:
        """Delete expired entries; returns how many were removed."""
        cursor = self._db.execute(
            "DELETE FROM kv WHERE expires_at_s IS NOT NULL AND expires_at_s <= ?", (self._now_s(),)
        )
        self._db.commit()
        return cursor.rowcount

    def _now_s(self) -> float:
        now = self._clock.now()
        if now.tzinfo is None:
            raise ValueError("the clock must return timezone-aware datetimes")
        return now.timestamp()
