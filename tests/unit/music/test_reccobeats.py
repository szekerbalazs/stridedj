"""ReccoBeats adapter tests. Fixtures are real responses captured on 2026-10-17 (see tests/fixtures/reccobeats)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from stridedj.domain import TrackId
from stridedj.errors import ExternalAPIError
from stridedj.http import RetryPolicy
from stridedj.music.cache import JsonCache
from stridedj.music.reccobeats import (
    BASE_URL,
    MISSING_TTL_S,
    ReccoBeatsClient,
    parse_audio_features,
    parse_track,
    spotify_track_id,
)
from tests.unit.music.support import FrozenClock, load_fixture

RICK = TrackId("4uLU6hMCjMI75M1A2tKUQC")  # not in ReccoBeats' catalogue (verified live)
BLINDING_LIGHTS = TrackId("0VjIjW4GlUZAMYd2vXMi3b")
MR_BRIGHTSIDE = TrackId("3n3Ppam7vgaVa1iaRUc9Lp")

Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    def __init__(self, handler: Handler) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)


async def _no_sleep(_: float) -> None:
    return None


def _client(handler: Handler, *, cache: JsonCache | None = None) -> tuple[ReccoBeatsClient, Recorder]:
    recorder = Recorder(handler)
    client = ReccoBeatsClient(
        http=httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
        cache=cache,
        policy=RetryPolicy(max_attempts=2),
        sleep=_no_sleep,
    )
    return client, recorder


def _json(payload: Any, status: int = 200) -> Handler:
    return lambda _: httpx.Response(status, json=payload)


def _ids_param(request: httpx.Request) -> list[str]:
    return request.url.params["ids"].split(",")


# ─────────────────────────────── parsing (real fixtures) ───────────────────────────────
def test_audio_features_fixture_is_keyed_by_spotify_id_from_href() -> None:
    items = load_fixture("audio_features.json")["content"]
    parsed = {f.track_id: f for item in items if (f := parse_audio_features(item))}
    assert set(parsed) == {BLINDING_LIGHTS, MR_BRIGHTSIDE}  # response order differs from request order
    blinding = parsed[BLINDING_LIGHTS]
    assert blinding.tempo_bpm == pytest.approx(171.001)
    assert blinding.energy == pytest.approx(0.73)
    assert blinding.instrumentalness == pytest.approx(9.54e-5)
    assert blinding.loudness_db == pytest.approx(-5.94)


def test_track_fixture_maps_to_allowlisted_track() -> None:
    items = load_fixture("track.json")["content"]
    tracks = {t.track_id: t for item in items if (t := parse_track(item))}
    assert tracks[MR_BRIGHTSIDE].artist_ids == ("0C0XlULifJtAgn6ZNCW2eu",)  # Spotify id from the artist href
    assert tracks[MR_BRIGHTSIDE].duration_ms == 222200
    assert RICK not in tracks


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"tempo": 0}, "no beat detected"),
        ({"energy": 1.5}, "out of range"),
        ({"valence": None}, "missing"),
        ({"loudness": "loud"}, "not a number"),
        ({"href": "https://evil.example.com/track/abc"}, "not a Spotify track"),
    ],
)
def test_invalid_features_are_treated_as_missing(change: dict[str, Any], reason: str) -> None:
    item = {**load_fixture("audio_features.json")["content"][0], **change}
    assert parse_audio_features(item) is None, reason


def test_spotify_track_id_from_href() -> None:
    assert spotify_track_id("https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b") == BLINDING_LIGHTS
    assert spotify_track_id("https://open.spotify.com/artist/0VjIjW4GlUZAMYd2vXMi3b") is None
    assert spotify_track_id(None) is None


# ─────────────────────────────── client ───────────────────────────────
async def test_audio_features_unknown_ids_are_absent() -> None:
    client, recorder = _client(_json(load_fixture("audio_features.json")))
    found = await client.audio_features([RICK, BLINDING_LIGHTS, MR_BRIGHTSIDE])
    assert set(found) == {BLINDING_LIGHTS, MR_BRIGHTSIDE}
    request = recorder.requests[0]
    assert str(request.url).startswith(f"{BASE_URL}/audio-features")
    assert _ids_param(request) == [RICK, BLINDING_LIGHTS, MR_BRIGHTSIDE]


async def test_audio_features_are_batched_by_40() -> None:
    client, recorder = _client(_json({"content": []}))
    ids = [TrackId(f"t{n:021d}") for n in range(85)]
    await client.audio_features([*ids, ids[0], TrackId("bad/id")])  # duplicates and invalid ids are dropped
    assert [len(_ids_param(r)) for r in recorder.requests] == [40, 40, 5]


async def test_audio_features_are_cache_first_including_known_misses(clock: FrozenClock) -> None:
    cache = JsonCache(Path(":memory:"), clock=clock)
    client, recorder = _client(_json(load_fixture("audio_features.json")), cache=cache)
    ids = [RICK, BLINDING_LIGHTS, MR_BRIGHTSIDE]

    first = await client.audio_features(ids)
    second = await client.audio_features(ids)
    assert first == second
    assert len(recorder.requests) == 1  # the miss (RICK) is remembered too

    clock.advance(seconds=MISSING_TTL_S + 1)  # misses expire after a week: ReccoBeats' catalogue grows
    await client.audio_features(ids)
    assert len(recorder.requests) == 2
    assert _ids_param(recorder.requests[1]) == [RICK]


async def test_a_failed_batch_is_skipped_but_total_failure_raises() -> None:
    features = load_fixture("audio_features.json")

    def flaky(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=features) if BLINDING_LIGHTS in _ids_param(request) else httpx.Response(500)

    client, _ = _client(flaky)
    other = [TrackId(f"t{n:021d}") for n in range(40)]
    found = await client.audio_features([*other, BLINDING_LIGHTS])  # batch 1 fails, batch 2 succeeds
    assert set(found) == {BLINDING_LIGHTS}

    client, _ = _client(_json({}, status=500))
    with pytest.raises(ExternalAPIError):
        await client.audio_features([BLINDING_LIGHTS])


async def test_recommendations_cap_seeds_and_size_and_cache_track_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/track/recommendation"):
            return httpx.Response(200, json=load_fixture("recommendation.json"))
        raise AssertionError("tracks() must be served from the cache")

    cache = JsonCache(Path(":memory:"))
    client, recorder = _client(handler, cache=cache)
    seeds = [BLINDING_LIGHTS, MR_BRIGHTSIDE, *(TrackId(f"s{n:021d}") for n in range(6))]

    ids = await client.recommendations(seeds, limit=500)

    assert ids == ["4bCihe7lYpnerhvpYhTrpx", "0ek2jtO8Ru1YrVImYFLSuW", "6JDLpBIjNMKCHRsw1ffsUy"]
    params = recorder.requests[0].url.params
    assert len(params["seeds"].split(",")) == 5
    assert params["size"] == "100"
    tracks = await client.tracks(ids)
    assert set(tracks) == set(ids)
    assert tracks[TrackId("6JDLpBIjNMKCHRsw1ffsUy")].duration_ms == 258148
    assert await client.recommendations(seeds, limit=500) == ids  # cached for a day
    assert len(recorder.requests) == 1


async def test_recommendations_with_unknown_seeds_return_empty() -> None:
    client, _ = _client(_json(load_fixture("error_unknown_seed.json"), status=400))
    assert await client.recommendations([RICK], limit=10) == []
    assert await client.recommendations([], limit=10) == []


async def test_tracks_lookup_parses_real_fixture() -> None:
    client, recorder = _client(_json(load_fixture("track.json")))
    found = await client.tracks([RICK, BLINDING_LIGHTS, MR_BRIGHTSIDE])
    assert set(found) == {BLINDING_LIGHTS, MR_BRIGHTSIDE}
    assert recorder.requests[0].url.path == "/v1/track"


async def test_batch_limit_error_fixture_is_a_400() -> None:
    """Documents the verified limit: 41 ids → HTTP 400 'size must be between 1 and 40'."""
    body = load_fixture("error_too_many_ids.json")
    assert body["errors"][0]["message"] == "size must be between 1 and 40"
