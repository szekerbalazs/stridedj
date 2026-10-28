"""Music contracts.

`Track.name` and `Track.artist_names` are **untrusted text** (anyone can publish a song called "Ignore previous
instructions ..."). They are used for display in code-rendered messages only and never placed into the LLM context.
The LLM sees `PlaylistDraftSummary`, which has no titles.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from stridedj.domain.base import ArtistId, DraftId, Frozen, TrackId
from stridedj.domain.runs import RunType


class Mood(StrEnum):
    """How the playlist should feel relative to the user's usual running music (a closed set: no free text)."""

    USUAL = "usual"
    MORE_ENERGETIC = "more_energetic"
    CALMER = "calmer"


class MusicChoice(StrEnum):
    """Where the songs come from (a closed set: the runner's wording is mapped to one of these, never passed on)."""

    USUAL = "usual"  # ≥ 50 % new songs (not in the running playlists); recommendations only to fill gaps
    FAMILIAR = "familiar"  # 0 % new: only songs of the running playlists
    DISCOVER = "discover"  # 100 % new: only songs outside the running playlists, in their style


class Genre(StrEnum):
    """The closed genre vocabulary. Free-form tags (Last.fm tags are written by anyone) are mapped onto it in
    `music/genres.py`; only these values reach the model, the tool schema and the playlist card."""

    # Running genres only: classical, country, jazz, ... are on the blocklist in `music/genres.py` instead. (A comment,
    # not the docstring: the docstring is part of the tool schema the model reads.)

    TECHNO = "techno"
    HOUSE = "house"
    EDM = "edm"
    TRANCE = "trance"
    DRUM_AND_BASS = "drum and bass"
    DUBSTEP = "dubstep"
    ELECTRONIC = "electronic"
    HIP_HOP = "hip hop"
    POP = "pop"
    RNB = "r&b"
    ROCK = "rock"
    INDIE = "indie"
    METAL = "metal"
    PUNK = "punk"
    LATIN = "latin"
    FUNK_DISCO = "funk & disco"
    REGGAE = "reggae"


class Track(Frozen):
    track_id: TrackId
    name: str  # UNTRUSTED
    artist_ids: tuple[ArtistId, ...]
    artist_names: tuple[str, ...]  # UNTRUSTED
    duration_ms: int = Field(gt=0)


class AudioFeatures(Frozen):
    """Spotify-style audio features (from ReccoBeats; Spotify closed its own endpoint to new apps in Nov 2024)."""

    track_id: TrackId
    tempo_bpm: float = Field(gt=0)
    energy: float = Field(ge=0, le=1)
    danceability: float = Field(ge=0, le=1)
    valence: float = Field(ge=0, le=1)
    acousticness: float = Field(ge=0, le=1)
    instrumentalness: float = Field(ge=0, le=1)
    speechiness: float = Field(ge=0, le=1)
    loudness_db: float


class PlaylistDraft(Frozen):
    """A planned playlist (nothing is created on Spotify until the plan is approved)."""

    draft_id: DraftId
    run_type: RunType
    target_spm: float = Field(gt=0)
    tolerance_bpm: float = Field(gt=0)
    track_ids: tuple[TrackId, ...]
    total_min: float = Field(ge=0)
    mean_abs_bpm_error: float = Field(ge=0, description="mean |folded tempo − target| over the chosen tracks")
    novelty_share: float = Field(ge=0, le=1, description="share of tracks not in the running playlists")
    top_genres: tuple[str, ...] = ()
    # what it was drafted with, so the playlist alone can be redrafted (the bot's 0 / 50 / 100 % new buttons)
    requested_min: float | None = Field(default=None, gt=0)
    mood: Mood = Mood.USUAL
    music: MusicChoice = MusicChoice.USUAL
    genre: Genre | None = None


class PlaylistDraftSummary(Frozen):
    """The LLM-facing view of a draft: numbers and sanitised genre tags only, no titles."""

    draft_id: DraftId
    n_tracks: int = Field(ge=0)
    total_min: float = Field(ge=0)
    target_spm: float
    tolerance_bpm: float
    mean_abs_bpm_error: float
    top_genres: tuple[str, ...]
