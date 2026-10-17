"""ReccoBeats adapter: tempo and audio features by Spotify id, seed-based recommendations; cache-first (no SLA).

Spotify closed `/audio-features` and `/recommendations` to new apps in Nov 2024; ReccoBeats (public, no key) serves
Spotify-style features keyed by its own UUIDs but *accepts Spotify ids* on its batch endpoints. Verified live on
2026-10-17 (fixtures in tests/fixtures/reccobeats/):

* `GET /v1/audio-features?ids=a,b,...` — 1..40 ids (41 → HTTP 400 "size must be between 1 and 40"). Unknown or
  malformed ids are silently dropped and the order is not preserved. Each item's `id` is a ReccoBeats UUID; the
  Spotify id is only in `href` (`https://open.spotify.com/track/<id>`), so results are matched by `href`.
* `GET /v1/track?ids=...` — same 1..40 rule; `trackTitle`, `artists[].href` (Spotify artist URL), `durationMs`.
* `GET /v1/track/recommendation?seeds=...&size=n` — 1..5 seeds (Spotify ids work), `size` ≤ 100, returns *up to*
  `size` tracks (often fewer); if no seed is known → HTTP 400 (body status 4002).
* `GET /v1/track/{id}/audio-features` only takes ReccoBeats UUIDs (a Spotify id gives 404), so it is not used.
* Rate limit: unpublished; 429 with `Retry-After` (handled by `request_with_retries`).

Features of a recording do not change, so they are cached for months; "ReccoBeats does not know this track" is cached
too, but only for a week (their catalogue grows). Missing features are absent from the result, never zero-filled.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from stridedj.domain import ArtistId, AudioFeatures, Track, TrackId
from stridedj.errors import ExternalAPIError
from stridedj.http import RetryPolicy, Sleep, request_with_retries
from stridedj.music.cache import JsonCache

logger = logging.getLogger(__name__)

SERVICE = "reccobeats"
BASE_URL = "https://api.reccobeats.com/v1"
MAX_IDS_PER_REQUEST = 40
MAX_SEEDS = 5
MAX_RECOMMENDATIONS = 100
FEATURES_TTL_S = 180 * 24 * 3600.0
TRACK_TTL_S = 30 * 24 * 3600.0
MISSING_TTL_S = 7 * 24 * 3600.0
RECOMMENDATIONS_TTL_S = 24 * 3600.0

_ID_RE = re.compile(r"^[0-9A-Za-z]{1,64}$")
_TRACK_HREF_RE = re.compile(r"^https://open\.spotify\.com/track/([0-9A-Za-z]{1,64})(?:[/?#].*)?$")
_ARTIST_HREF_RE = re.compile(r"^https://open\.spotify\.com/artist/([0-9A-Za-z]{1,64})(?:[/?#].*)?$")
_UNIT_FEATURES = ("energy", "danceability", "valence", "acousticness", "instrumentalness", "speechiness")
_MISSING = {"missing": True}


# ─────────────────────────────── pure parsing ───────────────────────────────
def spotify_track_id(href: object) -> TrackId | None:
    match = _TRACK_HREF_RE.match(href) if isinstance(href, str) else None
    return TrackId(match.group(1)) if match else None


def spotify_artist_id(href: object) -> ArtistId | None:
    match = _ARTIST_HREF_RE.match(href) if isinstance(href, str) else None
    return ArtistId(match.group(1)) if match else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def parse_audio_features(item: object) -> AudioFeatures | None:
    """One `/audio-features` item → `AudioFeatures` keyed by the *Spotify* id; None if anything required is missing
    or out of range (a tempo of 0 means "no beat detected" and is treated as missing, not as a very slow song)."""
    if not isinstance(item, Mapping):
        return None
    track_id = spotify_track_id(item.get("href"))
    tempo_bpm, loudness_db = _number(item.get("tempo")), _number(item.get("loudness"))
    unit = {name: _number(item.get(name)) for name in _UNIT_FEATURES}
    if track_id is None or tempo_bpm is None or tempo_bpm <= 0 or loudness_db is None:
        return None
    values: dict[str, float] = {}
    for name, value in unit.items():
        if value is None or not 0.0 <= value <= 1.0:
            return None
        values[name] = value
    return AudioFeatures(
        track_id=track_id,
        tempo_bpm=tempo_bpm,
        loudness_db=loudness_db,
        energy=values["energy"],
        danceability=values["danceability"],
        valence=values["valence"],
        acousticness=values["acousticness"],
        instrumentalness=values["instrumentalness"],
        speechiness=values["speechiness"],
    )


def parse_track(item: object) -> Track | None:
    """One `/track` or `/track/recommendation` item → `Track` (allowlist: title, Spotify artist ids/names, duration)."""
    if not isinstance(item, Mapping):
        return None
    track_id = spotify_track_id(item.get("href"))
    duration_ms = item.get("durationMs")
    if track_id is None or not isinstance(duration_ms, int) or isinstance(duration_ms, bool) or duration_ms <= 0:
        return None
    artists: list[tuple[ArtistId, str]] = []
    for artist in item.get("artists") or ():
        artist_id = spotify_artist_id(artist.get("href")) if isinstance(artist, Mapping) else None
        if artist_id is not None and isinstance(artist, Mapping):
            artists.append((artist_id, str(artist.get("name") or "")))
    return Track(
        track_id=track_id,
        name=str(item.get("trackTitle") or ""),
        artist_ids=tuple(artist_id for artist_id, _ in artists),
        artist_names=tuple(name for _, name in artists),
        duration_ms=duration_ms,
    )


def _content(payload: object) -> list[Any]:
    content = payload.get("content") if isinstance(payload, Mapping) else None
    return list(content) if isinstance(content, list) else []


def _valid_unique(ids: Sequence[TrackId]) -> list[TrackId]:
    return [track_id for track_id in dict.fromkeys(ids) if _ID_RE.fullmatch(track_id)]


def _chunks(items: Sequence[TrackId], size: int) -> list[list[TrackId]]:
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


# ─────────────────────────────── the client ───────────────────────────────
class ReccoBeatsClient:
    """Implements `AudioFeaturesPort` (and `tracks()`, a `TrackLookup` that costs no Spotify quota)."""

    def __init__(
        self,
        *,
        http: httpx.AsyncClient,
        cache: JsonCache | None = None,
        policy: RetryPolicy | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._http = http
        self._cache = cache
        self._policy = policy
        self._sleep = sleep

    async def audio_features(self, track_ids: Sequence[TrackId]) -> Mapping[TrackId, AudioFeatures]:
        """Features for the ids ReccoBeats knows; unknown ids are absent. A failed batch is skipped (logged) as long
        as something could be served, so one bad request does not sink a whole playlist."""
        hits, to_fetch = self._from_cache(_valid_unique(track_ids), "features", AudioFeatures)
        found: dict[TrackId, AudioFeatures] = dict(hits)
        await self._fetch_batches(to_fetch, "/audio-features", "features", parse_audio_features, found, FEATURES_TTL_S)
        return found

    async def tracks(self, track_ids: Sequence[TrackId]) -> Mapping[TrackId, Track]:
        hits, to_fetch = self._from_cache(_valid_unique(track_ids), "track", Track)
        found: dict[TrackId, Track] = dict(hits)
        await self._fetch_batches(to_fetch, "/track", "track", parse_track, found, TRACK_TTL_S)
        return found

    async def recommendations(self, seed_track_ids: Sequence[TrackId], *, limit: int) -> Sequence[TrackId]:
        """Up to `limit` (≤ 100) Spotify ids similar to the seeds (the first 5 valid ones are used); seeds excluded.

        Returns [] when no seed is in ReccoBeats' catalogue (their HTTP 400). Track metadata from the response is
        cached, so resolving the ids with `tracks()` afterwards costs no request.
        """
        seeds = _valid_unique(seed_track_ids)[:MAX_SEEDS]
        size = min(limit, MAX_RECOMMENDATIONS)
        if not seeds or size <= 0:
            return []
        key = f"reccobeats:recs:{size}:{','.join(sorted(seeds))}"
        cached = self._cache.get(key) if self._cache else None
        if isinstance(cached, list):
            return [TrackId(str(track_id)) for track_id in cached]
        try:
            response = await self._get("/track/recommendation", {"seeds": ",".join(seeds), "size": size})
        except ExternalAPIError as exc:
            if exc.status_code == 400:
                logger.info("reccobeats: none of the %d seeds is known; no recommendations", len(seeds))
                return []
            raise
        tracks = [track for item in _content(response.json()) if (track := parse_track(item))]
        seed_set = set(seeds)
        ids = [track_id for track_id in dict.fromkeys(t.track_id for t in tracks) if track_id not in seed_set]
        if self._cache:
            self._cache.set_many(
                {f"reccobeats:track:{t.track_id}": t.model_dump(mode="json") for t in tracks}, ttl_s=TRACK_TTL_S
            )
            self._cache.set(key, list(ids), ttl_s=RECOMMENDATIONS_TTL_S)
        return ids[:size]

    # ── plumbing ──
    def _from_cache[M: (AudioFeatures, Track)](
        self, ids: list[TrackId], kind: str, model: type[M]
    ) -> tuple[dict[TrackId, M], list[TrackId]]:
        """(cached hits, ids still to fetch). A cached "missing" marker counts as answered: not refetched."""
        if not self._cache:
            return {}, ids
        cached = self._cache.get_many(f"reccobeats:{kind}:{track_id}" for track_id in ids)
        hits: dict[TrackId, M] = {}
        to_fetch: list[TrackId] = []
        for track_id in ids:
            value = cached.get(f"reccobeats:{kind}:{track_id}")
            if value is None:
                to_fetch.append(track_id)
            elif value != _MISSING:
                hits[track_id] = model.model_validate(value)
        return hits, to_fetch

    async def _fetch_batches[M: (AudioFeatures, Track)](
        self,
        ids: list[TrackId],
        path: str,
        kind: str,
        parse: Callable[[object], M | None],
        found: dict[TrackId, M],
        ttl_s: float,
    ) -> None:
        last_error: ExternalAPIError | None = None
        failed = 0
        batches = _chunks(ids, MAX_IDS_PER_REQUEST)
        for batch in batches:
            try:
                response = await self._get(path, {"ids": ",".join(batch)})
            except ExternalAPIError as exc:
                logger.warning("reccobeats: %s batch of %d ids failed: %s", kind, len(batch), exc)
                last_error, failed = exc, failed + 1
                continue
            wanted = set(batch)
            parsed: dict[TrackId, M] = {}
            for item in _content(response.json()):
                model = parse(item)
                if model is not None and model.track_id in wanted and model.track_id not in parsed:
                    parsed[model.track_id] = model
            found.update(parsed)
            self._remember(kind, parsed, [track_id for track_id in batch if track_id not in parsed], ttl_s)
        if last_error is not None and failed == len(batches) and not found:
            raise last_error

    def _remember(
        self, kind: str, parsed: Mapping[TrackId, AudioFeatures | Track], missing: list[TrackId], ttl_s: float
    ) -> None:
        if not self._cache:
            return
        self._cache.set_many(
            {f"reccobeats:{kind}:{k}": v.model_dump(mode="json") for k, v in parsed.items()}, ttl_s=ttl_s
        )
        if missing:
            self._cache.set_many({f"reccobeats:{kind}:{k}": _MISSING for k in missing}, ttl_s=MISSING_TTL_S)

    async def _get(self, path: str, params: Mapping[str, Any]) -> httpx.Response:
        return await request_with_retries(
            self._http,
            "GET",
            f"{BASE_URL}{path}",
            service=SERVICE,
            policy=self._policy,
            sleep=self._sleep,
            params=params,
        )
