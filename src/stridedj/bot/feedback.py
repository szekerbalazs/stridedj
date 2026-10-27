"""Feedback buttons under a created playlist: 👍 / 👎 for the playlist, and per-song 👎 toggles.

Pure (no Telegram types), so it is tested without a bot. Callback data (Telegram's limit: 64 bytes) carries only a
draft id and a song position, never titles:

* `f:<draft_id>:+` / `f:<draft_id>:-`  rate the playlist (applies to its new songs, see `music/history.py`)
* `r:<draft_id>`                        show the song list with one 👎 toggle per song
* `s:<draft_id>:<n>`                    toggle the dislike of song n (1-based)

A modified client can send any callback data, so it is parsed strictly here, and the bot additionally checks the
allowlist and that the draft id is a playlist StrideDJ created (`PlaylistHistory.track_ids`).
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence, Set
from dataclasses import dataclass

from stridedj.domain import Track

MAX_SONGS = 40  # buttons per message stay well under Telegram's 100
SONGS_PER_ROW = 5
_DRAFT_ID = re.compile(r"^[0-9a-f]{12}$")  # `assemble_playlist` draft ids: 12 lowercase hex characters

Button = tuple[str, str]  # (label, callback data)


@dataclass(frozen=True, slots=True)
class RatePlaylist:
    draft_id: str
    rating: int  # +1 / -1


@dataclass(frozen=True, slots=True)
class OpenSongs:
    draft_id: str


@dataclass(frozen=True, slots=True)
class ToggleSong:
    draft_id: str
    position: int  # 1-based, as shown


FeedbackAction = RatePlaylist | OpenSongs | ToggleSong


def is_feedback(data: str) -> bool:
    return data[:2] in ("f:", "r:", "s:")


def parse_feedback(data: str) -> FeedbackAction | None:
    """Strict parse of feedback callback data; None for anything malformed."""
    kind, _, rest = data.partition(":")
    draft_id, _, arg = rest.partition(":")
    if not _DRAFT_ID.fullmatch(draft_id):
        return None
    if kind == "f" and arg in ("+", "-"):
        return RatePlaylist(draft_id=draft_id, rating=1 if arg == "+" else -1)
    if kind == "r" and arg == "":
        return OpenSongs(draft_id=draft_id)
    if kind == "s" and arg.isdigit() and 1 <= int(arg) <= MAX_SONGS:
        return ToggleSong(draft_id=draft_id, position=int(arg))
    return None


def playlist_buttons(draft_id: str, *, rated: int | None = None) -> list[list[Button]]:
    """👍 / 👎 (the chosen one is ticked) and the way into per-song feedback."""
    good = "✅ 👍 Good for running" if rated == 1 else "👍 Good for running"
    bad = "✅ 👎 Not my style" if rated == -1 else "👎 Not my style"
    return [[(good, f"f:{draft_id}:+"), (bad, f"f:{draft_id}:-")], [("🎚 Rate songs", f"r:{draft_id}")]]


def song_buttons(draft_id: str, n_songs: int, disliked: Set[int]) -> list[list[Button]]:
    """One toggle per song: "👎 3" → tap → "🚫 3" (tap again to undo)."""
    buttons = [
        (f"{'🚫' if position in disliked else '👎'} {position}", f"s:{draft_id}:{position}")
        for position in range(1, min(n_songs, MAX_SONGS) + 1)
    ]
    return [buttons[start : start + SONGS_PER_ROW] for start in range(0, len(buttons), SONGS_PER_ROW)]


def render_song_list(tracks: Sequence[Track], disliked: Set[int]) -> str:
    """Telegram HTML. Titles and artist names are untrusted: escaped, never formatted."""
    lines = ["<b>Tap 👎 for songs you don't want to run to</b> (tap again to undo):"]
    for position, track in enumerate(tracks[:MAX_SONGS], start=1):
        mark = "🚫 " if position in disliked else ""
        artists = ", ".join(track.artist_names)
        lines.append(f"{position}. {mark}{html.escape(track.name, quote=False)} — {html.escape(artists, quote=False)}")
    return "\n".join(lines)
