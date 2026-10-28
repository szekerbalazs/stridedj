"""Last.fm adapter on a mock transport (no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from stridedj.domain import ArtistId
from stridedj.errors import AuthError
from stridedj.music.cache import JsonCache
from stridedj.music.lastfm import LastFmClient, parse_top_tags

KEY = "0123456789abcdef0123456789abcdef"
ARTIST = ArtistId("artist0000000000000001")


class FakeLastFm:
    def __init__(self, body: dict[str, Any], status: int = 200) -> None:
        self.body, self.status = body, status
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body)

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def client(self, cache: JsonCache | None = None) -> LastFmClient:
        return LastFmClient(
            api_key=SecretStr(KEY),
            http=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)),
            cache=cache,
            sleep=self.sleep,
        )


def _tags(*pairs: tuple[str, int]) -> dict[str, Any]:
    return {"toptags": {"tag": [{"name": n, "count": c, "url": "x"} for n, c in pairs], "@attr": {"artist": "A"}}}


async def test_tags_become_closed_genres_and_are_cached_per_artist(tmp_path: Path) -> None:
    api = FakeLastFm(_tags(("techno", 100), ("seen live", 60), ("house", 40), ("trance", 5)))
    cache = JsonCache(tmp_path / "cache.db")
    assert await api.client(cache).artist_genres(ARTIST, "Some  DJ") == ("techno", "house")  # trance: count < 10
    request = api.requests[0]
    assert request.url.params["method"] == "artist.gettoptags"
    assert request.url.params["artist"] == "Some DJ"
    assert request.headers["user-agent"].startswith("StrideDJ")
    assert await api.client(cache).artist_genres(ARTIST, "Some DJ") == ("techno", "house")  # a new process
    assert len(api.requests) == 1
    assert api.client(cache).cached(ARTIST) == ("techno", "house")


async def test_an_artist_on_the_blocklist_gets_no_genre_and_is_cached(tmp_path: Path) -> None:
    api = FakeLastFm(_tags(("classical", 100), ("piano", 80), ("electronic", 40)))
    cache = JsonCache(tmp_path / "cache.db")
    assert await api.client(cache).artist_genres(ARTIST, "A Composer") == ()
    assert api.client(cache).cached(ARTIST) == ()  # known: not looked up again
    assert len(api.requests) == 1


async def test_unknown_artist_is_an_empty_cached_answer(tmp_path: Path) -> None:
    api = FakeLastFm({"error": 6, "message": "The artist you supplied could not be found", "links": []})
    cache = JsonCache(tmp_path / "cache.db")
    assert await api.client(cache).artist_genres(ARTIST, "Nobody") == ()
    assert api.client(cache).cached(ARTIST) == ()


async def test_a_rejected_key_raises_without_revealing_it() -> None:
    api = FakeLastFm({"error": 10, "message": "Invalid API key - You must be granted a valid key by last.fm"})
    with pytest.raises(AuthError, match="LASTFM_API_KEY") as info:
        await api.client().artist_genres(ARTIST, "Some DJ")
    assert KEY not in str(info.value)


async def test_calls_are_spaced_out() -> None:
    api = FakeLastFm(_tags(("techno", 100)))
    client = api.client()
    await client.artist_genres(ArtistId("artist0000000000000001"), "A")
    await client.artist_genres(ArtistId("artist0000000000000002"), "B")
    assert len(api.sleeps) == 1 and 0 < api.sleeps[0] <= 0.35


def test_parse_top_tags_tolerates_a_single_tag_and_missing_counts() -> None:
    assert parse_top_tags({"toptags": {"tag": {"name": "techno"}}}) == ["techno"]
    assert parse_top_tags({"toptags": {"tag": [{"name": "a"}, {"count": 50}, "junk"]}}) == ["a"]
    assert parse_top_tags({"nope": 1}) == []
