"""Live wiring of the music side: Spotify (library + playlist writes) + ReccoBeats (tempo/features), one shared cache.

Spotify's recently-played endpoint only returns the last 50 plays (about 3 hours of listening), so the plays log is
synced before every draft AND on a timer while the bot runs (`SyncingPlanner.refresh`): syncing only on drafts loses
the music of a run when no playlist is requested within the next 50 plays. The same timer warms the
artist-genre cache (Last.fm, throttled), so a draft rarely waits for genre lookups.

Genres: Last.fm when `LASTFM_API_KEY` is set, otherwise none. Spotify's artist genres are not used live: they come
back empty for development-mode apps, and one call per artist costs quota (and can trigger a long rate limit).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from stridedj.config import Settings
from stridedj.domain import CadenceTarget, Genre, Mood, MusicChoice, PlaylistDraft, RunType, Track, TrackId
from stridedj.errors import StrideDJError
from stridedj.http import build_client
from stridedj.music.cache import JsonCache, default_cache_path
from stridedj.music.history import PlaylistHistory, default_history_path
from stridedj.music.lastfm import LastFmClient
from stridedj.music.plays_logger import PlaysLogger, default_plays_path
from stridedj.music.reccobeats import ReccoBeatsClient
from stridedj.music.service import (
    ChainedTrackLookup,
    NoGenres,
    PlannerSettings,
    PlaylistPlannerService,
    RunWindows,
)
from stridedj.music.spotify import SpotifyClient
from stridedj.ports import ArtistGenresPort, MusicLibraryPort

log = logging.getLogger(__name__)


class SyncingPlanner:
    """A `PlaylistPlanner` that first pulls the latest plays into the log (best effort), then plans."""

    def __init__(self, planner: PlaylistPlannerService, plays: PlaysLogger, library: MusicLibraryPort) -> None:
        self._planner = planner
        self._plays = plays
        self._library = library

    async def draft(
        self,
        *,
        run_type: RunType,
        cadence: CadenceTarget,
        duration_min: float,
        mood: Mood,
        music: MusicChoice = MusicChoice.USUAL,
        genre: Genre | None = None,
    ) -> PlaylistDraft:
        await self.sync_plays()
        return await self._planner.draft(
            run_type=run_type, cadence=cadence, duration_min=duration_min, mood=mood, music=music, genre=genre
        )

    async def refresh(self) -> int:
        """The background job: sync the plays, then warm the genre cache. Best effort; returns the new plays."""
        new = await self.sync_plays()
        try:
            looked_up = await self._planner.warm_up()
        except StrideDJError as exc:
            log.warning("genre warm-up stopped: %s", exc)
        else:
            if looked_up:
                log.info("genre warm-up: %d artist(s) looked up", looked_up)
        return new

    async def sync_plays(self) -> int:
        """Pull the latest plays into the log (best effort: planning still works without them)."""
        try:
            new = await self._plays.sync(self._library)
        except StrideDJError as exc:
            log.warning("could not sync recently-played: %s", exc)
            return 0
        log.info("plays log: %d new play(s)", new)
        return new

    async def tracks_for(self, draft: PlaylistDraft) -> Sequence[Track]:
        return await self._planner.tracks_for(draft)

    async def tracks_of(self, track_ids: Sequence[TrackId]) -> Sequence[Track]:
        return await self._planner.tracks_of(track_ids)


@dataclass(frozen=True)
class LiveMusic:
    planner: SyncingPlanner  # for the tools
    writer: SpotifyClient  # the playlist writer, for the executor
    history: PlaylistHistory  # created playlists: written by the executor, read by the planner (rotation)


def build_live_music(settings: Settings, *, run_windows: RunWindows) -> LiveMusic:
    cache = JsonCache(default_cache_path(settings.app))
    http = build_client(timeout_s=20.0)
    spotify = SpotifyClient.from_settings(settings.spotify, http=http, cache=cache)
    reccobeats = ReccoBeatsClient(http=http, cache=cache)
    genres: ArtistGenresPort = (
        LastFmClient(api_key=settings.lastfm.api_key, http=http, cache=cache)
        if settings.lastfm.api_key is not None
        else NoGenres()
    )
    plays = PlaysLogger(default_plays_path(settings.app))
    history = PlaylistHistory(default_history_path(settings.app))
    planner = PlaylistPlannerService(
        library=spotify,
        features=reccobeats,
        running_playlist_ids=settings.spotify.running_playlist_ids,
        track_lookup=ChainedTrackLookup(reccobeats, spotify),  # ReccoBeats first: costs no Spotify quota
        plays=plays,
        run_windows=run_windows,
        history=history,
        genres=genres,
        settings=PlannerSettings(
            max_genre_lookups=40,  # Last.fm is throttled (~3/s): the warm-up does the bulk
            saved_tracks_max_age_s=2 * 3600,  # the warm-up (every 30 min) refreshes them; a draft then needs no paging
        ),
    )
    return LiveMusic(planner=SyncingPlanner(planner, plays, spotify), writer=spotify, history=history)
