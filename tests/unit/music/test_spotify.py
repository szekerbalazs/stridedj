from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from pydantic import SecretStr

from stridedj.config import SpotifySettings
from stridedj.domain import ArtistId, TrackId
from stridedj.errors import ConfigError, ExternalAPIError
from stridedj.music.cache import JsonCache
from stridedj.music.spotify import (
    API_BASE_URL,
    SpotifyClient,
    normalise_playlist_id,
    track_from_json,
    track_from_playlist_entry,
)
from tests.unit.music.support import FrozenClock

Route = Callable[[httpx.Request], httpx.Response]
TOKEN_URL = "https://accounts.spotify.com/api/token"


def _track(n: int, **extra: Any) -> dict[str, Any]:
    """A realistic track object, including fields we must ignore (album, popularity, markets, ...)."""
    track_id = f"track{n:017d}"
    return {
        "id": track_id,
        "name": f"Song {n}",
        "type": "track",
        "duration_ms": 200_000 + n,
        "is_local": False,
        "is_playable": True,
        "artists": [{"id": f"artist{n % 3:016d}", "name": f"Artist {n % 3}", "type": "artist"}],
        "album": {"name": "Album", "images": [{"url": "https://i.scdn.co/x"}]},
        "popularity": 99,
        "available_markets": ["CH"],
        "external_ids": {"isrc": "X"},
        "preview_url": None,
        "uri": f"spotify:track:{track_id}",
        **extra,
    }


class FakeSpotify:
    """A MockTransport router: token endpoint + API routes keyed by (method, path)."""

    def __init__(self, *, rotated_refresh_token: str | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: dict[tuple[str, str], Route] = {}
        self.token_calls = 0
        self.rotated = rotated_refresh_token

    def route(self, method: str, path: str, handler: Route | dict[str, Any] | list[dict[str, Any]]) -> None:
        if callable(handler):
            self.routes[(method, path)] = handler
        elif isinstance(handler, list):
            pages = list(handler)
            self.routes[(method, path)] = lambda _: httpx.Response(200, json=pages.pop(0))
        else:
            self.routes[(method, path)] = lambda _: httpx.Response(200, json=handler)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if str(request.url) == TOKEN_URL:
            self.token_calls += 1
            body: dict[str, Any] = {
                "access_token": f"at-{self.token_calls}",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
            if self.rotated:
                body["refresh_token"] = self.rotated
            return httpx.Response(200, json=body)
        assert request.url.host == "api.spotify.com", f"unexpected host {request.url.host}"
        route = self.routes.get((request.method, request.url.path.removeprefix("/v1")))
        return route(request) if route else httpx.Response(404)

    def api_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == "api.spotify.com"]

    async def _no_sleep(self, _: float) -> None:
        return None

    def client(
        self,
        *,
        rotated_sink: list[str] | None = None,
        cache: JsonCache | None = None,
        clock: FrozenClock | None = None,
    ) -> SpotifyClient:
        sink = rotated_sink if rotated_sink is not None else []
        return SpotifyClient(
            client_id="cid",
            refresh_token=SecretStr("old-rt"),
            http=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)),
            cache=cache,
            on_refresh_token_rotated=sink.append,
            clock=clock or FrozenClock(),
            sleep=self._no_sleep,
        )


def _form(request: httpx.Request) -> dict[str, str]:
    return {key: values[0] for key, values in parse_qs(request.content.decode()).items()}


# ─────────────────────────────── tokens ───────────────────────────────
async def test_rotated_refresh_token_is_persisted_once_and_access_token_reused() -> None:
    api = FakeSpotify(rotated_refresh_token="new-rt")
    api.route("GET", "/me/player/recently-played", {"items": []})
    sink: list[str] = []
    client = api.client(rotated_sink=sink)

    await client.recently_played()
    await client.recently_played()

    token_requests = [r for r in api.requests if str(r.url) == TOKEN_URL]
    assert len(token_requests) == 1
    assert _form(token_requests[0]) == {"grant_type": "refresh_token", "refresh_token": "old-rt", "client_id": "cid"}
    assert sink == ["new-rt"]
    assert all(r.headers["authorization"] == "Bearer at-1" for r in api.api_requests())


async def test_unrotated_refresh_token_is_not_persisted() -> None:
    for rotated in (None, "old-rt"):
        api = FakeSpotify(rotated_refresh_token=rotated)
        api.route("GET", "/me/player/recently-played", {"items": []})
        sink: list[str] = []
        await api.client(rotated_sink=sink).recently_played()
        assert sink == []


async def test_expired_access_token_is_refreshed(clock: FrozenClock) -> None:
    api = FakeSpotify()
    api.route("GET", "/me/player/recently-played", {"items": []})
    client = api.client(clock=clock)
    await client.recently_played()
    clock.advance(minutes=59.5)  # inside the last 60 s of a 3600 s token: refresh early
    await client.recently_played()
    assert api.token_calls == 2
    assert api.api_requests()[-1].headers["authorization"] == "Bearer at-2"


async def test_401_forces_one_refresh_and_retry() -> None:
    api = FakeSpotify()
    responses = [httpx.Response(401), httpx.Response(200, json={"items": []})]
    api.route("GET", "/me/player/recently-played", lambda _: responses.pop(0))
    assert await api.client().recently_played() == []
    assert api.token_calls == 2


# ─────────────────────────────── reads ───────────────────────────────
async def test_playlist_items_are_paginated_and_allowlisted() -> None:
    api = FakeSpotify()
    page_1 = {
        "items": [
            {"is_local": False, "item": _track(1)},
            {"is_local": False, "track": _track(2)},  # deprecated key still understood
            {"is_local": True, "item": _track(3, is_local=True, id=None)},  # local file
            {"is_local": False, "item": {"type": "episode", "id": "ep000000000000000000001", "duration_ms": 1}},
            {"is_local": False, "item": _track(4, is_playable=False)},  # region-locked
            {"is_local": False, "item": None},  # removed from Spotify
        ],
        "next": f"{API_BASE_URL}/playlists/pl1/items?offset=6&limit=50",
    }
    page_2 = {"items": [{"is_local": False, "item": _track(5)}], "next": None}
    api.route("GET", "/playlists/pl1/items", [page_1, page_2])

    tracks = await api.client().playlist_tracks("https://open.spotify.com/playlist/pl1?si=abc")

    assert [t.track_id for t in tracks] == [_track(1)["id"], _track(2)["id"], _track(5)["id"]]
    assert tracks[0].model_dump() == {
        "track_id": _track(1)["id"],
        "name": "Song 1",
        "artist_ids": (_track(1)["artists"][0]["id"],),
        "artist_names": ("Artist 1",),
        "duration_ms": 200_001,
    }
    first = api.api_requests()[0]
    assert first.url.params["limit"] == "50"
    assert first.url.params["market"] == "from_token"
    assert api.api_requests()[1].url.params["offset"] == "6"


async def test_next_link_to_another_host_is_refused() -> None:
    api = FakeSpotify()
    api.route("GET", "/playlists/pl1/items", {"items": [], "next": "https://evil.example.com/steal"})
    with pytest.raises(ExternalAPIError, match="outside the Web API"):
        await api.client().playlist_tracks("pl1")
    assert all(r.url.host != "evil.example.com" for r in api.requests)


async def test_recently_played_parses_aware_timestamps() -> None:
    api = FakeSpotify()
    api.route(
        "GET",
        "/me/player/recently-played",
        {
            "items": [
                {"track": _track(1), "played_at": "2026-10-17T07:00:00.123Z", "context": {"uri": "spotify:x"}},
                {"track": _track(2), "played_at": "not a date"},
                {"track": _track(3, is_local=True), "played_at": "2026-10-17T07:04:00Z"},
            ]
        },
    )
    plays = await api.client().recently_played()
    assert len(plays) == 1
    track, played_at = plays[0]
    assert track.track_id == _track(1)["id"]
    assert played_at.tzinfo is not None
    assert played_at.isoformat() == "2026-10-17T07:00:00.123000+00:00"
    assert api.api_requests()[0].url.params["limit"] == "50"


async def test_saved_tracks_stops_at_limit() -> None:
    api = FakeSpotify()
    page = {"items": [{"added_at": "2026-01-01T00:00:00Z", "track": _track(n)} for n in range(5)], "next": None}
    api.route("GET", "/me/tracks", page)
    tracks = await api.client().saved_tracks(limit=3)
    assert len(tracks) == 3
    assert api.api_requests()[0].url.params["limit"] == "3"


async def test_search_pages_by_ten_with_offset() -> None:
    api = FakeSpotify()

    def search(request: httpx.Request) -> httpx.Response:
        offset, limit = int(request.url.params["offset"]), int(request.url.params["limit"])
        items = [_track(offset + i) for i in range(limit)]
        return httpx.Response(200, json={"tracks": {"items": items, "next": "more", "offset": offset}})

    api.route("GET", "/search", search)
    tracks = await api.client().search_tracks('genre:"drum and bass"', limit=25)
    assert len(tracks) == 25
    params = [(r.url.params["limit"], r.url.params["offset"]) for r in api.api_requests()]
    assert params == [("10", "0"), ("10", "10"), ("5", "20")]
    assert api.api_requests()[0].url.params["q"] == 'genre:"drum and bass"'
    assert api.api_requests()[0].url.params["type"] == "track"


async def test_artist_genres_are_memoised_and_cached(tmp_path: Path) -> None:
    api = FakeSpotify()
    api.route("GET", "/artists/artist0000000000000001", {"id": "x", "genres": ["Drum And Bass ", "jungle"]})
    cache = JsonCache(tmp_path / "cache.db")
    client = api.client(cache=cache)
    artist = ArtistId("artist0000000000000001")
    assert await client.artist_genres(artist) == ("drum and bass", "jungle")
    assert await client.artist_genres(artist) == ("drum and bass", "jungle")
    assert len(api.api_requests()) == 1
    assert await api.client(cache=cache).artist_genres(artist) == ("drum and bass", "jungle")  # new process
    assert len(api.api_requests()) == 1


async def test_track_lookup_is_per_item_and_skips_unknown(tmp_path: Path) -> None:
    api = FakeSpotify()
    known = _track(1)
    api.route("GET", f"/tracks/{known['id']}", known)
    client = api.client(cache=JsonCache(tmp_path / "cache.db"))
    found = await client.tracks([TrackId(known["id"]), TrackId("unknown00000000000000x")])
    assert list(found) == [known["id"]]
    await client.tracks([TrackId(known["id"])])
    assert len([r for r in api.api_requests() if r.url.path == f"/v1/tracks/{known['id']}"]) == 1


# ─────────────────────────────── write ───────────────────────────────
async def test_create_private_playlist_chunks_items_and_returns_url() -> None:
    api = FakeSpotify()
    api.route(
        "POST",
        "/me/playlists",
        lambda _: httpx.Response(
            201, json={"id": "newpl", "external_urls": {"spotify": "https://open.spotify.com/playlist/newpl"}}
        ),
    )
    api.route("PUT", "/playlists/newpl/items", lambda _: httpx.Response(200, json={"snapshot_id": "s1"}))
    api.route("POST", "/playlists/newpl/items", lambda _: httpx.Response(201, json={"snapshot_id": "s2"}))
    ids = [TrackId(f"t{n:021d}") for n in range(230)]

    url = await api.client().create_private_playlist(
        name="StrideDJ · Z2 10k", description="line 1\nline 2", track_ids=ids
    )

    assert url == "https://open.spotify.com/playlist/newpl"
    create, *adds = api.api_requests()
    assert json.loads(create.content) == {"name": "StrideDJ · Z2 10k", "description": "line 1 line 2", "public": False}
    assert [(r.method, len(json.loads(r.content)["uris"])) for r in adds] == [("PUT", 100), ("POST", 100), ("POST", 30)]
    assert json.loads(adds[0].content)["uris"][0] == f"spotify:track:{ids[0]}"


async def test_create_playlist_is_not_retried_on_server_error() -> None:
    api = FakeSpotify()
    api.route("POST", "/me/playlists", lambda _: httpx.Response(503))
    with pytest.raises(ExternalAPIError):
        await api.client().create_private_playlist(name="x", description="", track_ids=[])
    assert len(api.api_requests()) == 1


# ─────────────────────────────── pure helpers ───────────────────────────────
@pytest.mark.parametrize(
    "ref",
    [
        "37i9dQZF1DXcBWIGoYBM5M",
        "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
        "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=abc",
    ],
)
def test_playlist_refs_are_normalised(ref: str) -> None:
    assert normalise_playlist_id(ref) == "37i9dQZF1DXcBWIGoYBM5M"


def test_garbage_playlist_ref_is_rejected() -> None:
    with pytest.raises(ConfigError):
        normalise_playlist_id("../../me/player")


def test_track_mapping_rejects_unsafe_or_incomplete_objects() -> None:
    assert track_from_json(_track(1, id="../../me")) is None
    assert track_from_json(_track(1, duration_ms=0)) is None
    assert track_from_json(_track(1, duration_ms=True)) is None
    assert track_from_json("not an object") is None
    no_artist_id = track_from_json(_track(1, artists=[{"id": None, "name": "Local"}]))
    assert no_artist_id is not None and no_artist_id.artist_ids == ()
    assert track_from_playlist_entry({"is_local": True, "item": _track(1)}) is None


def test_from_settings_requires_login() -> None:
    http = httpx.AsyncClient()
    with pytest.raises(ConfigError, match="SPOTIFY_CLIENT_ID"):
        SpotifyClient.from_settings(SpotifySettings(), http=http)
    with pytest.raises(ConfigError, match="stridedj auth spotify"):
        SpotifyClient.from_settings(SpotifySettings(client_id="cid"), http=http)
