"""Ports: the interfaces between the core (harness + domain logic) and the outside world (ports-and-adapters).

The core depends only on these Protocols. Each has a real adapter (Garmin, Google, Spotify, ...) and an in-memory
fake (tests/fakes, evals, `--demo`). Swapping an adapter never touches the core, which is what makes the agent
testable offline and resilient to an API breaking (e.g. Garmin's login change in March 2026: switch the RunSource).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import SecretStr

from stridedj.domain import (
    Activity,
    ArtistId,
    AudioFeatures,
    TimeSlot,
    Track,
    TrackId,
    UserId,
)


class Clock(Protocol):
    """Injected time source, so tests and evals can freeze "now"."""

    def now(self) -> datetime: ...


# ─────────────────────────────── runs ───────────────────────────────
class RunSource(Protocol):
    """Where runs come from (live Garmin API, intervals.icu, synthetic demo data)."""

    @property
    def name(self) -> str: ...

    async def fetch_activities(self, *, since: datetime, limit: int) -> Sequence[Activity]:
        """Running activities started at or after `since`, newest first, allowlisted fields only."""
        ...


class RunRepository(Protocol):
    """Local persistence of activities (SQLite)."""

    def upsert_activities(self, activities: Sequence[Activity]) -> int:
        """Insert or update; returns the number of rows written."""
        ...

    def list_activities(self, *, since: datetime | None = None) -> Sequence[Activity]: ...


# ─────────────────────────────── calendar ───────────────────────────────
class CalendarPort(Protocol):
    async def busy_intervals(self, window: TimeSlot) -> Sequence[TimeSlot]:
        """Busy intervals within `window` across the configured calendars (free/busy only: no titles)."""
        ...

    async def insert_event(self, *, event_id: str, slot: TimeSlot, title: str, description: str) -> str:
        """Create an event with a client-chosen id (idempotent: re-inserting the same id is a no-op). Returns the id."""
        ...


# ─────────────────────────────── music ───────────────────────────────
class MusicLibraryPort(Protocol):
    """Read access to the user's Spotify library."""

    async def playlist_tracks(self, playlist_id: str) -> Sequence[Track]: ...

    async def recently_played(self) -> Sequence[tuple[Track, datetime]]:
        """Up to the last 50 plays with their `played_at` timestamps (a Spotify API limit)."""
        ...

    async def saved_tracks(self, *, limit: int) -> Sequence[Track]: ...

    async def artist_genres(self, artist_id: ArtistId) -> Sequence[str]: ...

    async def search_tracks(self, query: str, *, limit: int) -> Sequence[Track]: ...


class AudioFeaturesPort(Protocol):
    """Tempo and audio features by Spotify track id (ReccoBeats)."""

    async def audio_features(self, track_ids: Sequence[TrackId]) -> Mapping[TrackId, AudioFeatures]: ...

    async def recommendations(self, seed_track_ids: Sequence[TrackId], *, limit: int) -> Sequence[TrackId]: ...


class PlaylistWriterPort(Protocol):
    """The only Spotify write. Called by the executor after approval, never by the LLM."""

    async def create_private_playlist(self, *, name: str, description: str, track_ids: Sequence[TrackId]) -> str:
        """Create a private playlist and return its URL."""
        ...


# ─────────────────────────────── credentials (multi-user ready) ───────────────────────────────
class CredentialKey(StrEnum):
    """Names of per-user secrets (the names, not the values)."""

    OPENAI_API_KEY = "openai_api_key"
    SPOTIFY_REFRESH_TOKEN = "spotify_refresh_token"  # noqa: S105
    GOOGLE_REFRESH_TOKEN = "google_refresh_token"  # noqa: S105
    GARMIN_PASSWORD = "garmin_password"  # noqa: S105


class CredentialProvider(Protocol):
    """Per-user secrets: backed by `.env` today (single user); a per-user encrypted store later, no core changes."""

    def get(self, user_id: UserId, key: CredentialKey) -> SecretStr | None: ...
