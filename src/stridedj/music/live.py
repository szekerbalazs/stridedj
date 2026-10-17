"""Live wiring of the music side: Spotify (library + playlist writes) + ReccoBeats (tempo/features), one shared cache.

Spotify's recently-played endpoint only returns the last 50 plays, so the plays log is synced before every draft:
over time it accumulates the history needed to know what was played *during* runs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from stridedj.config import Settings
from stridedj.domain import CadenceTarget, Mood, PlaylistDraft, RunType, Track
from stridedj.errors import StrideDJError
from stridedj.http import build_client
from stridedj.music.cache import JsonCache, default_cache_path
from stridedj.music.plays_logger import PlaysLogger, default_plays_path
from stridedj.music.reccobeats import ReccoBeatsClient
from stridedj.music.service import ChainedTrackLookup, PlaylistPlannerService, RunWindows
from stridedj.music.spotify import SpotifyClient
from stridedj.ports import MusicLibraryPort, PlaylistPlanner

log = logging.getLogger(__name__)


class SyncingPlanner:
    """A `PlaylistPlanner` that first pulls the latest plays into the log (best effort), then plans."""

    def __init__(self, planner: PlaylistPlanner, plays: PlaysLogger, library: MusicLibraryPort) -> None:
        self._planner = planner
        self._plays = plays
        self._library = library

    async def draft(
        self, *, run_type: RunType, cadence: CadenceTarget, duration_min: float, mood: Mood
    ) -> PlaylistDraft:
        try:
            new = await self._plays.sync(self._library)
            log.info("plays log: %d new play(s)", new)
        except StrideDJError as exc:  # planning still works without the newest plays
            log.warning("could not sync recently-played: %s", exc)
        return await self._planner.draft(run_type=run_type, cadence=cadence, duration_min=duration_min, mood=mood)

    async def tracks_for(self, draft: PlaylistDraft) -> Sequence[Track]:
        return await self._planner.tracks_for(draft)


def build_live_music(settings: Settings, *, run_windows: RunWindows) -> tuple[PlaylistPlanner, SpotifyClient]:
    """Returns (planner for the tools, Spotify client as the playlist writer for the executor)."""
    cache = JsonCache(default_cache_path(settings.app))
    http = build_client(timeout_s=20.0)
    spotify = SpotifyClient.from_settings(settings.spotify, http=http, cache=cache)
    reccobeats = ReccoBeatsClient(http=http, cache=cache)
    plays = PlaysLogger(default_plays_path(settings.app))
    planner = PlaylistPlannerService(
        library=spotify,
        features=reccobeats,
        running_playlist_ids=settings.spotify.running_playlist_ids,
        track_lookup=ChainedTrackLookup(reccobeats, spotify),  # ReccoBeats first: costs no Spotify quota
        plays=plays,
        run_windows=run_windows,
    )
    return SyncingPlanner(planner, plays, spotify), spotify
