"""Spotify Web API adapter: library reads, per-item lookups (Feb 2026 dev-mode limits) and private playlist creation.

Shapes coded against (developer.spotify.com/documentation/web-api, checked 2026-10-17):
* `GET /playlists/{id}/items` (renamed from `/tracks`): paging object; each entry has `is_local` and the track under
  `item` (the old `track` key is deprecated, read as a fallback). Only playlists the user owns or collaborates on.
* `GET /me/tracks` and `GET /me/player/recently-played`: entries still carry the track under `track`;
  recently-played adds `played_at` (ISO 8601) and is capped at 50.
* `GET /search?type=track`: `{"tracks": <paging object>}`, `limit` ≤ 10 (was 50), `offset` ≤ 1000.
* `GET /artists/{id}` and `GET /tracks/{id}`: one item per request (the batch GETs were removed), hence the cache.
* `POST /me/playlists` (was `/users/{id}/playlists`) → 201 with `id` and `external_urls.spotify`;
  `PUT|POST /playlists/{id}/items` with `{"uris": [...]}`, at most 100 per request.

Every response is mapped through an explicit allowlist into `Track`: local files, episodes and unplayable tracks are
dropped; everything else (album art, popularity, markets, ...) is never read.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import SecretStr

from stridedj.config import SpotifySettings
from stridedj.domain import ArtistId, Track, TrackId
from stridedj.envfile import store_secret
from stridedj.errors import AuthError, ConfigError, ExternalAPIError, RateLimitedError
from stridedj.http import RetryPolicy, Sleep, request_with_retries
from stridedj.music.cache import JsonCache, UtcClock
from stridedj.music.spotify_auth import REFRESH_TOKEN_ENV_KEY, SERVICE, refresh_access_token
from stridedj.ports import Clock

logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.spotify.com/v1"
PAGE_LIMIT = 50  # max page size of /playlists/{id}/items, /me/tracks, /me/player/recently-played
SEARCH_PAGE_LIMIT = 10  # Feb 2026: search limit max 10 (default 5)
SEARCH_MAX_OFFSET = 1000
PLAYLIST_ITEMS_PER_REQUEST = 100
ACCESS_TOKEN_SKEW_S = 60.0  # refresh a minute early: a token that expires mid-request costs a 401 round trip
ARTIST_GENRES_TTL_S = 30 * 24 * 3600.0
TRACK_TTL_S = 30 * 24 * 3600.0
BLOCKED_KEY = "spotify:blocked_until"  # + ":<client id>": rate limits are per app; survives restarts
MAX_PLAYLIST_NAME_CHARS = 100
MAX_PLAYLIST_DESCRIPTION_CHARS = 300

_ID_RE = re.compile(r"^[0-9A-Za-z]{1,64}$")  # Spotify ids are base62 (22 chars today); never trust them in a path
_PLAYLIST_REF_RE = re.compile(r"(?:spotify:playlist:|open\.spotify\.com/playlist/)?([0-9A-Za-z]+)")

RefreshTokenSink = Callable[[str], None]


# ─────────────────────────────── pure mapping helpers ───────────────────────────────
def normalise_playlist_id(ref: str) -> str:
    """Accept an id, a `spotify:playlist:<id>` URI or an `open.spotify.com/playlist/<id>?si=...` link."""
    candidate = ref.strip()
    if "open.spotify.com/playlist/" in candidate:
        candidate = candidate.split("open.spotify.com/playlist/", 1)[1].split("?", 1)[0].split("/", 1)[0]
    elif candidate.startswith("spotify:playlist:"):
        candidate = candidate.removeprefix("spotify:playlist:")
    if not _ID_RE.fullmatch(candidate):
        raise ConfigError(f"not a Spotify playlist id or link: {ref[:40]!r}")
    return candidate


def track_from_json(obj: object) -> Track | None:
    """Map a Spotify track object to `Track` via an explicit allowlist; None for anything that is not a playable,
    non-local track with an id and a positive duration (episodes, local files, region-locked tracks, nulls)."""
    if not isinstance(obj, Mapping):
        return None
    if obj.get("type", "track") != "track" or obj.get("is_local") is True or obj.get("is_playable") is False:
        return None
    track_id, duration_ms = obj.get("id"), obj.get("duration_ms")
    if not isinstance(track_id, str) or not _ID_RE.fullmatch(track_id):
        return None
    if not isinstance(duration_ms, int) or isinstance(duration_ms, bool) or duration_ms <= 0:
        return None
    artists = [
        artist
        for artist in obj.get("artists") or ()
        if isinstance(artist, Mapping) and isinstance(artist.get("id"), str) and _ID_RE.fullmatch(artist["id"])
    ]
    return Track(
        track_id=TrackId(track_id),
        name=str(obj.get("name") or ""),
        artist_ids=tuple(ArtistId(artist["id"]) for artist in artists),
        artist_names=tuple(str(artist.get("name") or "") for artist in artists),
        duration_ms=duration_ms,
    )


def track_from_playlist_entry(entry: object) -> Track | None:
    """A `/playlists/{id}/items` entry: the track sits under `item` (Feb 2026), formerly `track`."""
    if not isinstance(entry, Mapping) or entry.get("is_local") is True:
        return None
    return track_from_json(entry.get("item") or entry.get("track"))


def play_from_history_entry(entry: object) -> tuple[Track, datetime] | None:
    """A recently-played entry → (track, played_at) with a timezone-aware timestamp."""
    if not isinstance(entry, Mapping):
        return None
    track = track_from_json(entry.get("track"))
    raw = entry.get("played_at")
    if track is None or not isinstance(raw, str):
        return None
    try:
        played_at = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return (track, played_at) if played_at.tzinfo is not None else None


def _single_line(text: str, max_chars: int) -> str:
    return " ".join(text.split())[:max_chars]


def _persist_rotated_refresh_token(value: str) -> None:
    store_secret(REFRESH_TOKEN_ENV_KEY, value)


# ─────────────────────────────── the client ───────────────────────────────
class SpotifyClient:
    """Implements `MusicLibraryPort` and `PlaylistWriterPort` (plus `tracks()`, a per-item lookup with cache).

    The injected `httpx.AsyncClient` is used with absolute URLs only (the API and the token endpoint live on different
    hosts). Pagination `next` links are followed only if they point back at the API: the bearer token must never be
    sent to a host taken from a response body.
    """

    def __init__(
        self,
        *,
        client_id: str,
        refresh_token: SecretStr,
        http: httpx.AsyncClient,
        cache: JsonCache | None = None,
        on_refresh_token_rotated: RefreshTokenSink = _persist_rotated_refresh_token,
        clock: Clock | None = None,
        policy: RetryPolicy | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._client_id = client_id
        self._refresh_token = refresh_token
        self._http = http
        self._cache = cache
        self._on_rotated = on_refresh_token_rotated
        self._clock: Clock = clock or UtcClock()
        self._policy = policy
        self._sleep = sleep
        self._access: SecretStr | None = None
        self._access_expires_at: datetime | None = None
        self._token_lock = asyncio.Lock()
        self._genres_memo: dict[ArtistId, tuple[str, ...]] = {}

    @classmethod
    def from_settings(
        cls, settings: SpotifySettings, *, http: httpx.AsyncClient, cache: JsonCache | None = None
    ) -> SpotifyClient:
        if not settings.client_id:
            raise ConfigError("SPOTIFY_CLIENT_ID is not set (docs/SETUP.md §3)")
        if settings.refresh_token is None:
            raise ConfigError("Spotify is not logged in: run `stridedj auth spotify`")
        return cls(client_id=settings.client_id, refresh_token=settings.refresh_token, http=http, cache=cache)

    # ── MusicLibraryPort ──
    async def playlist_tracks(self, playlist_id: str) -> Sequence[Track]:
        path = f"/playlists/{normalise_playlist_id(playlist_id)}/items"
        params = {"limit": PAGE_LIMIT, "market": "from_token", "additional_types": "track"}
        return [track async for entry in self._paginate(path, params) if (track := track_from_playlist_entry(entry))]

    async def recently_played(self) -> Sequence[tuple[Track, datetime]]:
        page = await self._get_json("/me/player/recently-played", params={"limit": PAGE_LIMIT})
        return [play for entry in page.get("items") or () if (play := play_from_history_entry(entry))]

    async def saved_tracks(self, *, limit: int) -> Sequence[Track]:
        if limit <= 0:
            return []
        params = {"limit": min(PAGE_LIMIT, limit), "market": "from_token"}
        tracks: list[Track] = []
        async for entry in self._paginate("/me/tracks", params):
            track = track_from_json(entry.get("track") if isinstance(entry, Mapping) else None)
            if track is not None:
                tracks.append(track)
            if len(tracks) >= limit:
                break
        return tracks

    async def artist_genres(self, artist_id: ArtistId) -> Sequence[str]:
        """Genres of one artist (often empty: Spotify marks the field deprecated). Memoised and cached for 30 days."""
        if artist_id in self._genres_memo:
            return self._genres_memo[artist_id]
        key = f"spotify:artist_genres:{artist_id}"
        cached = self._cache.get(key) if self._cache else None
        if isinstance(cached, list):
            genres = tuple(str(g) for g in cached)
        else:
            payload = await self._get_json(f"/artists/{_path_id(artist_id)}")
            genres = tuple(str(g).strip().lower() for g in payload.get("genres") or () if isinstance(g, str))
            if self._cache:
                self._cache.set(key, list(genres), ttl_s=ARTIST_GENRES_TTL_S)
        self._genres_memo[artist_id] = genres
        return genres

    async def search_tracks(self, query: str, *, limit: int) -> Sequence[Track]:
        """Track search, paged 10 at a time (the Feb 2026 maximum) up to `limit` results."""
        tracks: list[Track] = []
        offset = 0
        while len(tracks) < limit and offset <= SEARCH_MAX_OFFSET:
            page_size = min(SEARCH_PAGE_LIMIT, limit - len(tracks))
            params = {"q": query, "type": "track", "limit": page_size, "offset": offset, "market": "from_token"}
            container = (await self._get_json("/search", params=params)).get("tracks") or {}
            items = container.get("items") or []
            tracks.extend(track for item in items if (track := track_from_json(item)))
            if not container.get("next") or not items:
                break
            offset += len(items)
        return tracks[:limit]

    # ── TrackLookup (not part of MusicLibraryPort; see service.py) ──
    async def tracks(self, track_ids: Sequence[TrackId]) -> Mapping[TrackId, Track]:
        """Look tracks up one by one (the batch `GET /tracks` is gone), cache-first. Unknown ids are absent."""
        found: dict[TrackId, Track] = {}
        keys = {track_id: f"spotify:track:{track_id}" for track_id in dict.fromkeys(track_ids)}
        cached = self._cache.get_many(keys.values()) if self._cache else {}
        for track_id, key in keys.items():
            if key in cached:
                found[track_id] = Track.model_validate(cached[key])
                continue
            try:
                payload = await self._get_json(f"/tracks/{_path_id(track_id)}", params={"market": "from_token"})
            except ExternalAPIError as exc:
                if exc.status_code in (400, 404):
                    continue
                raise
            track = track_from_json(payload)
            if track is not None:
                found[track_id] = track
                if self._cache:
                    self._cache.set(key, track.model_dump(mode="json"), ttl_s=TRACK_TTL_S)
        return found

    # ── PlaylistWriterPort ──
    async def create_private_playlist(self, *, name: str, description: str, track_ids: Sequence[TrackId]) -> str:
        """Create a new private playlist and fill it; returns its open.spotify.com URL.

        Creation is not retried (a retried POST after a lost response would create a duplicate). The first chunk
        of ≤100 items is written with `PUT` (replace: idempotent, so it is safe to retry), which covers any run
        playlist under ~5 hours; further chunks are appended with `POST`.
        """
        body = {
            "name": _single_line(name, MAX_PLAYLIST_NAME_CHARS),
            "description": _single_line(description, MAX_PLAYLIST_DESCRIPTION_CHARS),
            "public": False,
        }
        created = (await self._api("POST", "/me/playlists", json=body)).json()
        playlist_id = str(created["id"])
        items_path = f"/playlists/{_path_id(playlist_id)}/items"
        uris = [f"spotify:track:{_path_id(track_id)}" for track_id in track_ids]
        for start in range(0, len(uris), PLAYLIST_ITEMS_PER_REQUEST):
            chunk = {"uris": uris[start : start + PLAYLIST_ITEMS_PER_REQUEST]}
            await self._api("PUT" if start == 0 else "POST", items_path, json=chunk)
        url = (created.get("external_urls") or {}).get("spotify")
        return str(url) if url else f"https://open.spotify.com/playlist/{playlist_id}"

    # ── plumbing ──
    async def _access_token(self, *, force: bool = False) -> str:
        """A valid access token, refreshed when (nearly) expired. Persists a rotated refresh token immediately:
        with PKCE, Spotify may invalidate the old one, and losing the new one would force a re-login."""
        async with self._token_lock:
            now = self._clock.now()
            fresh = self._access is not None and self._access_expires_at is not None and now < self._access_expires_at
            if fresh and not force and self._access is not None:
                return self._access.get_secret_value()
            tokens = await refresh_access_token(
                self._http,
                client_id=self._client_id,
                refresh_token=self._refresh_token,
                policy=self._policy,
                sleep=self._sleep,
            )
            rotated = tokens.refresh_token
            if rotated is not None and rotated.get_secret_value() != self._refresh_token.get_secret_value():
                self._refresh_token = rotated
                self._on_rotated(rotated.get_secret_value())
                logger.info("spotify: refresh token rotated and persisted")
            self._access = tokens.access_token
            self._access_expires_at = now + timedelta(seconds=max(0.0, tokens.expires_in_s - ACCESS_TOKEN_SKEW_S))
            return tokens.access_token.get_secret_value()

    async def _api(
        self, method: str, path_or_url: str, *, params: Mapping[str, Any] | None = None, json: Any = None
    ) -> httpx.Response:
        """One authorised request. A 401 (token revoked or expired early) triggers one forced refresh and a retry.

        Circuit breaker: after a 429 with a long Retry-After, every call fails fast without touching the network
        until that time has passed (in dev mode, calling a blocked app again can extend the block).
        """
        url = _api_url(path_or_url)
        self._raise_if_blocked(method)
        for attempt in range(2):
            token = await self._access_token(force=attempt > 0)
            try:
                return await request_with_retries(
                    self._http,
                    method,
                    url,
                    service=SERVICE,
                    policy=self._policy,
                    sleep=self._sleep,
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                    json=json,
                )
            except RateLimitedError as exc:
                self._block(exc.retry_after_s)
                raise
            except AuthError as exc:
                if exc.status_code != 401 or attempt > 0:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    def _block(self, retry_after_s: float | None) -> None:
        if self._cache is None or retry_after_s is None or retry_after_s <= 0:
            return
        until = self._clock.now() + timedelta(seconds=retry_after_s)
        self._cache.set(f"{BLOCKED_KEY}:{self._client_id}", until.isoformat(), ttl_s=retry_after_s)
        logger.warning("spotify: rate limited until %s UTC; not calling Spotify until then", f"{until:%a %H:%M}")

    def _raise_if_blocked(self, method: str) -> None:
        raw = self._cache.get(f"{BLOCKED_KEY}:{self._client_id}") if self._cache else None
        if not isinstance(raw, str):
            return
        until = datetime.fromisoformat(raw)
        remaining_s = (until - self._clock.now()).total_seconds()
        if remaining_s > 0:
            raise RateLimitedError(
                SERVICE, f"{method}: rate limited until {until:%a %H:%M} UTC (not called)", retry_after_s=remaining_s
            )

    async def _get_json(self, path_or_url: str, *, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        payload = (await self._api("GET", path_or_url, params=params)).json()
        if not isinstance(payload, Mapping):
            raise ExternalAPIError(SERVICE, f"GET {httpx.URL(_api_url(path_or_url)).path}: unexpected response shape")
        return payload

    async def _paginate(self, path: str, params: Mapping[str, Any]) -> AsyncIterator[Any]:
        """Yield the `items` of a paging object, following `next` (which already carries the query) to the end."""
        next_url: str | None = path
        next_params: Mapping[str, Any] | None = params
        while next_url:
            page = await self._get_json(next_url, params=next_params)
            for item in page.get("items") or ():
                yield item
            raw_next = page.get("next")
            next_url = raw_next if isinstance(raw_next, str) and raw_next else None
            next_params = None


def _api_url(path_or_url: str) -> str:
    """Absolute API URL. Absolute inputs (pagination links) must point at the API itself."""
    if "://" not in path_or_url:
        return f"{API_BASE_URL}{path_or_url}"
    if not path_or_url.startswith(f"{API_BASE_URL}/"):
        raise ExternalAPIError(SERVICE, "refusing to follow a link outside the Web API")
    return path_or_url


def _path_id(raw: str) -> str:
    """An id that is safe to put in a URL path (ids come from API responses: validate, then quote)."""
    if not _ID_RE.fullmatch(raw):
        raise ExternalAPIError(SERVICE, "invalid Spotify id in a response")
    return quote(raw, safe="")
