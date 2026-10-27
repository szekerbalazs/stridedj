"""Feedback buttons: strict parsing, storage semantics and the bot handler (fake Telegram objects, no network)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from stridedj.bot.feedback import (
    OpenSongs,
    RatePlaylist,
    ToggleSong,
    parse_feedback,
    playlist_buttons,
    render_song_list,
    song_buttons,
)
from stridedj.bot.telegram_bot import TelegramBot
from stridedj.clock import SystemClock
from stridedj.domain import ArtistId, DraftId, PlaylistDraft, RunType, Track, TrackId
from stridedj.harness.guard import InputGuard
from stridedj.music.history import PlaylistHistory

DRAFT = "0123456789ab"


@pytest.mark.parametrize(
    ("data", "action"),
    [
        (f"f:{DRAFT}:+", RatePlaylist(DRAFT, 1)),
        (f"f:{DRAFT}:-", RatePlaylist(DRAFT, -1)),
        (f"r:{DRAFT}", OpenSongs(DRAFT)),
        (f"s:{DRAFT}:3", ToggleSong(DRAFT, 3)),
        (f"s:{DRAFT}:0", None),
        (f"s:{DRAFT}:41", None),
        (f"s:{DRAFT}:-1", None),
        (f"f:{DRAFT}:++", None),
        ("f:../../etc:+", None),
        (f"f:{DRAFT.upper()}:+", None),
        (f"r:{DRAFT}:x", None),
        ("a:token", None),
    ],
)
def test_callback_data_is_parsed_strictly(data: str, action: object) -> None:
    assert parse_feedback(data) == action


def test_buttons_fit_telegrams_64_bytes_and_parse_back() -> None:
    rows = playlist_buttons(DRAFT) + song_buttons(DRAFT, 40, {2})
    for _, data in (button for row in rows for button in row):
        assert len(data.encode()) <= 64
        assert parse_feedback(data) is not None
    assert song_buttons(DRAFT, 7, {2})[0][1][0] == "🚫 2"
    assert len(song_buttons(DRAFT, 7, set())) == 2  # rows of 5


def test_song_list_escapes_untrusted_titles() -> None:
    track = Track(track_id=TrackId("t1"), name="<b>x</b> & y", artist_ids=(), artist_names=("<i>A</i>",), duration_ms=1)
    text = render_song_list([track], {1})
    assert "&lt;b&gt;x&lt;/b&gt; &amp; y" in text and "&lt;i&gt;A&lt;/i&gt;" in text and "🚫" in text


# ─────────────────────────────── storage semantics ───────────────────────────────
def _created(history: PlaylistHistory, draft_id: str, *tracks: str) -> None:
    history.record_created(
        PlaylistDraft(
            draft_id=DraftId(draft_id),
            run_type=RunType.Z2,
            target_spm=165,
            tolerance_bpm=3,
            track_ids=tuple(TrackId(t) for t in tracks),
            total_min=10,
            mean_abs_bpm_error=1,
            novelty_share=0.5,
        )
    )


def test_playlist_ratings_judge_new_songs_and_song_dislikes_always_win() -> None:
    history = PlaylistHistory(Path(":memory:"))
    _created(history, "aaaaaaaaaaaa", "fav", "new1", "new2")
    assert history.rate_playlist("aaaaaaaaaaaa", 1)
    assert not history.rate_playlist("bbbbbbbbbbbb", 1)  # unknown playlist: nothing stored
    assert history.toggle_song_dislike("new2")
    favourites = {TrackId("fav")}
    assert history.track_ratings(favourites=favourites) == {"new1": 1, "new2": -1}
    assert history.rate_playlist("aaaaaaaaaaaa", -1)  # changed their mind
    assert history.track_ratings(favourites=favourites) == {"new1": -1, "new2": -1}
    assert not history.toggle_song_dislike("new2")  # undo
    assert history.disliked_songs() == set()


# ─────────────────────────────── the bot handler ───────────────────────────────
class FakeQuery:
    def __init__(self, data: str, user_id: int = 1) -> None:
        self.data, self.from_user = data, SimpleNamespace(id=user_id)
        self.message = None
        self.answers: list[str | None] = []
        self.markups: list[Any] = []
        self.texts: list[str] = []

    async def answer(self, text: str | None = None) -> None:
        self.answers.append(text)

    async def edit_message_reply_markup(self, reply_markup: Any = None) -> None:
        self.markups.append(reply_markup)

    async def edit_message_text(self, text: str, **_: Any) -> None:
        self.texts.append(text)


class FakePlaylists:
    async def tracks_of(self, track_ids: Sequence[TrackId]) -> list[Track]:
        return [
            Track(track_id=t, name=f"Song {t}", artist_ids=(ArtistId("a"),), artist_names=("A",), duration_ms=1)
            for t in track_ids
        ]


def _bot(history: PlaylistHistory) -> TelegramBot:
    async def factory():  # never called
        raise AssertionError

    bot = TelegramBot(
        token="123456:TEST", guard=InputGuard(allowed_user_ids={1}, clock=SystemClock()), app_factory=factory
    )
    bot._stride = SimpleNamespace(history=history, playlists=FakePlaylists())  # type: ignore[assignment]
    return bot


async def test_rating_a_created_playlist_is_stored_and_ticked() -> None:
    history = PlaylistHistory(Path(":memory:"))
    _created(history, DRAFT, "t1", "t2")
    query = FakeQuery(f"f:{DRAFT}:-")
    await _bot(history)._on_feedback(query, query.data)  # type: ignore[arg-type]
    assert history.track_ratings(favourites=set()) == {"t1": -1, "t2": -1}
    assert query.answers == ["Noted: its new songs won't be suggested again."]
    labels = [b.text for row in query.markups[0].inline_keyboard for b in row]
    assert "✅ 👎 Not my style" in labels


async def test_toggling_a_song_updates_the_list() -> None:
    history = PlaylistHistory(Path(":memory:"))
    _created(history, DRAFT, "t1", "t2")
    query = FakeQuery(f"s:{DRAFT}:2")
    await _bot(history)._on_feedback(query, query.data)  # type: ignore[arg-type]
    assert history.disliked_songs() == {"t2"}
    assert "2. 🚫 Song t2" in query.texts[0]


@pytest.mark.parametrize("data", [f"f:{'f' * 12}:+", f"s:{DRAFT}:3"])
async def test_unknown_playlists_and_positions_change_nothing(data: str) -> None:
    history = PlaylistHistory(Path(":memory:"))
    _created(history, DRAFT, "t1", "t2")
    query = FakeQuery(data)
    await _bot(history)._on_feedback(query, query.data)  # type: ignore[arg-type]
    assert history.track_ratings(favourites=set()) == {} and query.markups == [] and query.texts == []


async def test_strangers_cannot_leave_feedback() -> None:
    history = PlaylistHistory(Path(":memory:"))
    _created(history, DRAFT, "t1")
    query = FakeQuery(f"f:{DRAFT}:-", user_id=999)
    await _bot(history)._on_feedback(query, query.data)  # type: ignore[arg-type]
    assert history.track_ratings(favourites=set()) == {}
