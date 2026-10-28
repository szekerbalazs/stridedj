"""Last.fm adapter: artist genres from crowd-sourced tags (Spotify returns no genres to development-mode apps).

`GET https://ws.audioscrobbler.com/2.0/?method=artist.gettoptags&artist=<name>&autocorrect=1&api_key=…&format=json`
needs only an API key (no user login). Per the API docs: errors 10 = invalid key, 26 = suspended
key, 29 = rate limit; "several calls per second" can get a key suspended, and requests should carry an identifiable
User-Agent. So calls are spaced ≥ 0.35 s apart and every answer is cached per artist for 90 days (a miss for 30).

Tags are untrusted text (any Last.fm user can tag an artist): they are mapped onto the closed `Genre` vocabulary right
here and never leave this module as free text. An artist whose top tags are on the blocklist (classical, audiobook,
jazz, ...: `genres.is_off_limits`) gets no genre at all, so the planner's running-genres rule leaves its songs out.
The API key travels only as a query parameter, which our HTTP layer never puts into error messages or logs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from pydantic import SecretStr

from stridedj.domain import ArtistId, Genre
from stridedj.errors import AuthError, ExternalAPIError, RateLimitedError
from stridedj.http import RetryPolicy, Sleep, request_with_retries
from stridedj.music.cache import JsonCache
from stridedj.music.genres import genres_from_tags, is_off_limits

logger = logging.getLogger(__name__)

SERVICE = "lastfm"
API_URL = "https://ws.audioscrobbler.com/2.0/"
USER_AGENT = "StrideDJ/0.1 (personal running-playlist bot)"
GENRES_TTL_S = 90 * 24 * 3600.0
MISSING_TTL_S = 30 * 24 * 3600.0
MIN_INTERVAL_S = 0.35
MIN_TAG_COUNT = 10  # Last.fm weights tags 0..100 relative to the artist's top tag; below 10 is noise
_INVALID_KEY_ERRORS = {10, 26}
_RATE_LIMIT_ERROR = 29
_NOT_FOUND_ERRORS = {6, 7}
_GENRE_VALUES = frozenset(g.value for g in Genre)


def parse_top_tags(payload: object) -> list[str]:
    """`{"toptags": {"tag": [{"name": "techno", "count": 100}, ...]}}` → tag names, most important first.

    Tolerates a single tag object instead of a list and a missing `count` (the docs only show the XML form)."""
    toptags = payload.get("toptags") if isinstance(payload, Mapping) else None
    raw = toptags.get("tag") if isinstance(toptags, Mapping) else None
    entries = [raw] if isinstance(raw, Mapping) else raw if isinstance(raw, list) else []
    names: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
            continue
        count = entry.get("count")
        if isinstance(count, int | float) and not isinstance(count, bool) and count < MIN_TAG_COUNT:
            continue
        names.append(entry["name"])
    return names


class LastFmClient:
    """Implements `ArtistGenresPort` with Last.fm tags. Lookups are by artist name; the cache is keyed by the Spotify
    artist id, so a name is sent to Last.fm at most once per artist and TTL."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        http: httpx.AsyncClient,
        cache: JsonCache | None = None,
        policy: RetryPolicy | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._api_key = api_key
        self._http = http
        self._cache = cache
        self._policy = policy or RetryPolicy(max_attempts=2, max_delay_s=5.0)
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._last_call_s = 0.0

    def cached(self, artist_id: ArtistId) -> Sequence[str] | None:
        """The cached genres, or None if this artist was never looked up (no network)."""
        value = self._cache.get(_key(artist_id)) if self._cache else None
        return tuple(g for g in value if g in _GENRE_VALUES) if isinstance(value, list) else None

    async def artist_genres(self, artist_id: ArtistId, artist_name: str) -> Sequence[str]:
        cached = self.cached(artist_id)
        if cached is not None:
            return cached
        name = " ".join(artist_name.split())[:200]
        if not name:
            return ()
        payload = await self._call({"method": "artist.gettoptags", "artist": name, "autocorrect": "1"})
        if payload is None:  # unknown artist
            genres: tuple[Genre, ...] = ()
            ttl_s = MISSING_TTL_S
        else:
            tags = parse_top_tags(payload)
            genres = () if is_off_limits(tags) else genres_from_tags(tags)
            ttl_s = GENRES_TTL_S
        if self._cache:
            self._cache.set(_key(artist_id), [g.value for g in genres], ttl_s=ttl_s)
        return tuple(g.value for g in genres)

    async def _call(self, params: Mapping[str, str]) -> Mapping[str, Any] | None:
        """One throttled GET. None = "not found" (error 6/7); raises on a bad key or a rate limit."""
        async with self._lock:
            wait_s = self._last_call_s + MIN_INTERVAL_S - time.monotonic()
            if wait_s > 0:
                await self._sleep(wait_s)
            self._last_call_s = time.monotonic()
            try:
                response = await request_with_retries(
                    self._http,
                    "GET",
                    API_URL,
                    service=SERVICE,
                    policy=self._policy,
                    sleep=self._sleep,
                    params={**params, "api_key": self._api_key.get_secret_value(), "format": "json"},
                    headers={"User-Agent": USER_AGENT},
                )
                payload: object = response.json()
            except ExternalAPIError as exc:
                if exc.status_code in (400, 404):  # Last.fm also signals some errors via the status code
                    return None
                raise
            except ValueError as exc:  # not JSON
                raise ExternalAPIError(SERVICE, "artist.gettoptags: unreadable response") from exc
        if not isinstance(payload, Mapping):
            raise ExternalAPIError(SERVICE, "artist.gettoptags: unexpected response")
        error = payload.get("error")
        if isinstance(error, int) and not isinstance(error, bool):
            if error in _NOT_FOUND_ERRORS:
                return None
            if error in _INVALID_KEY_ERRORS:
                raise AuthError(SERVICE, "API key rejected: check LASTFM_API_KEY")
            if error == _RATE_LIMIT_ERROR:
                raise RateLimitedError(SERVICE, "artist.gettoptags: rate limited", retry_after_s=None)
            raise ExternalAPIError(SERVICE, f"artist.gettoptags: error {error}")
        return payload


def _key(artist_id: ArtistId) -> str:
    return f"lastfm:genres:v2:{artist_id}"  # v2: with the blocklist (v1 answers are looked up again)
