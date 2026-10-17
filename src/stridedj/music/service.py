"""`PlaylistPlannerService`: the application service behind the agent's playlist tool (implements `PlaylistPlanner`).

Pipeline for one draft:

1. **Positives** — tracks of the running playlists (weight 1) and tracks played *during runs* (+1 per play, capped):
   recently-played is synced into the local log first, then matched against the run windows.
2. **Candidates** — positives + saved tracks + ReccoBeats recommendations seeded with the positives whose tempo is
   closest to the target + Spotify search by the profile's top genres (novelty beyond the user's library).
3. **Enrich** — audio features (ReccoBeats, cached) for everything; artist genres (Spotify, one call per artist,
   cached) for positives and for tempo-eligible candidates only (a track outside ±8 BPM cannot be used anyway).
4. **Score** — fit the style model on positives vs. the rest; mood reference = the positives' mean energy/valence.
5. **Assemble** — `playlist.assemble_playlist` (pure).

Robustness: recommendations, search and genres are optional enrichments; if they fail the draft is built without
them (logged). Missing audio features drop that track, nothing else. The running playlists are only read: this
service has no writer at all (playlist creation is the executor's job, after approval).
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from pydantic import Field

from stridedj.domain import (
    ArtistId,
    AudioFeatures,
    CadenceTarget,
    DraftId,
    Frozen,
    Mood,
    PlaylistDraft,
    RunType,
    TimeSlot,
    Track,
    TrackId,
)
from stridedj.errors import ExternalAPIError
from stridedj.music.playlist import (
    AssemblyParams,
    Candidate,
    MoodReference,
    assemble_playlist,
    fold_tempo,
    sanitise_genre,
)
from stridedj.music.plays_logger import PlaysLogger
from stridedj.music.style import (
    StyleFitter,
    TrackProfile,
    WeightedTrack,
    fit_style_model,
    top_profile_genres,
)
from stridedj.ports import AudioFeaturesPort, MusicLibraryPort

logger = logging.getLogger(__name__)

RunWindows = Callable[[], Sequence[TimeSlot]]
"""Returns the user's past run windows (start → start + elapsed). Provided by the runs module at composition time."""


class TrackLookup(Protocol):
    """Resolve Spotify track ids to `Track` (for recommended ids and logged plays). Not in `MusicLibraryPort`;
    implemented by `ReccoBeatsClient.tracks` (no Spotify quota), `SpotifyClient.tracks` and the demo fakes."""

    async def tracks(self, track_ids: Sequence[TrackId]) -> Mapping[TrackId, Track]: ...


class ChainedTrackLookup:
    """Ask each lookup in turn for the ids still missing (e.g. ReccoBeats first: free; then Spotify: per-item calls)."""

    def __init__(self, *lookups: TrackLookup) -> None:
        self._lookups = lookups

    async def tracks(self, track_ids: Sequence[TrackId]) -> Mapping[TrackId, Track]:
        found: dict[TrackId, Track] = {}
        for lookup in self._lookups:
            missing = [track_id for track_id in dict.fromkeys(track_ids) if track_id not in found]
            if not missing:
                break
            found |= await lookup.tracks(missing)
        return found


class PlannerSettings(Frozen):
    saved_tracks_limit: int = Field(default=300, ge=0)
    recommendation_rounds: int = Field(default=2, ge=0, description="ReccoBeats calls, each with ≤ 5 distinct seeds")
    recommendations_per_round: int = Field(default=100, ge=1, le=100)
    search_genres: int = Field(default=3, ge=0)
    search_tracks_per_genre: int = Field(default=20, ge=1)
    max_genre_lookups: int = Field(default=300, ge=0, description="Spotify artist calls per draft (cached afterwards)")
    play_weight: float = Field(default=1.0, gt=0, description="positive weight added per play during a run")
    max_positive_weight: float = Field(default=3.0, gt=0)
    max_kept_drafts: int = Field(default=64, ge=1)
    assembly: AssemblyParams = AssemblyParams()


class PlaylistPlannerService:
    """Plans (never creates) cadence-matched playlists. All I/O goes through injected ports, so it runs on fakes."""

    def __init__(
        self,
        *,
        library: MusicLibraryPort,
        features: AudioFeaturesPort,
        running_playlist_ids: Sequence[str],
        track_lookup: TrackLookup | None = None,
        plays: PlaysLogger | None = None,
        run_windows: RunWindows | None = None,
        style_fitter: StyleFitter = fit_style_model,
        settings: PlannerSettings | None = None,
        seed: int | None = None,
    ) -> None:
        self._library = library
        self._features = features
        self._playlist_ids = tuple(running_playlist_ids)
        self._lookup = track_lookup
        self._plays = plays
        self._run_windows = run_windows
        self._fit = style_fitter
        self._settings = settings or PlannerSettings()
        self._rng = random.Random(seed)  # noqa: S311 (not crypto: tie-breaking and draft ids)
        self._known: dict[TrackId, Track] = {}
        self._drafts: dict[DraftId, tuple[Track, ...]] = {}

    # ── PlaylistPlanner ──
    async def draft(
        self, *, run_type: RunType, cadence: CadenceTarget, duration_min: float, mood: Mood
    ) -> PlaylistDraft:
        rng = random.Random(self._rng.getrandbits(64))  # noqa: S311 (one reproducible stream per draft)
        target_spm = cadence.spm
        max_tolerance = max(self._settings.assembly.tolerance_ladder_bpm)

        running = await self._running_tracks()
        weights = await self._positive_weights(running)
        saved = self._remember(await self._library.saved_tracks(limit=self._settings.saved_tracks_limit))
        pool = {**saved, **(await self._resolve(list(weights))), **running}
        features = dict(await self._features.audio_features(list(pool)))

        genres: dict[ArtistId, tuple[str, ...]] = {}
        budget = _Budget(self._settings.max_genre_lookups)
        positive_ids = [track_id for track_id in weights if track_id in features and track_id in pool]
        await self._fetch_genres([pool[t] for t in positive_ids], genres, budget)
        positives = [
            WeightedTrack(profile=self._profile(pool[t], features[t], genres), weight=weights[t]) for t in positive_ids
        ]

        extra = await self._recommended(positive_ids, features, target_spm, rng)
        extra |= await self._searched(top_profile_genres(positives, self._settings.search_genres))
        extra = {track_id: track for track_id, track in extra.items() if track_id not in pool}
        features |= await self._features.audio_features([t for t in extra if t not in features])
        pool |= extra

        eligible = [
            track
            for track_id, track in sorted(pool.items())
            if track_id in features and fold_tempo(features[track_id].tempo_bpm, target_spm)[1] <= max_tolerance
        ]
        await self._fetch_genres(eligible, genres, budget)
        unlabeled = [
            self._profile(track, features[track_id], genres)
            for track_id, track in sorted(pool.items())
            if track_id in features and track_id not in weights
        ]
        model = self._fit(positives, unlabeled)
        profiles = [self._profile(track, features[track.track_id], genres) for track in eligible]
        candidates = [
            Candidate(
                track=track,
                features=profile.features,
                genres=profile.genres,
                style_score=score,
                in_running_playlists=track.track_id in running,
            )
            for track, profile, score in zip(eligible, profiles, model.score_many(profiles), strict=True)
        ]
        assembled = assemble_playlist(
            candidates,
            run_type=run_type,
            target_spm=target_spm,
            duration_min=duration_min,
            mood=mood,
            mood_reference=_mood_reference(positives, features),
            rng=rng,
            params=self._settings.assembly,
        )
        self._keep(assembled.draft.draft_id, assembled.tracks)
        logger.info(
            "playlist draft %s: %d tracks, %.1f min, ±%.0f BPM, %d positives, %d candidates",
            assembled.draft.draft_id,
            len(assembled.tracks),
            assembled.draft.total_min,
            assembled.draft.tolerance_bpm,
            len(positives),
            len(candidates),
        )
        return assembled.draft

    async def tracks_for(self, draft: PlaylistDraft) -> Sequence[Track]:
        """The draft's tracks in order. After a restart they are re-resolved via the lookup (unknown ids skipped)."""
        kept = self._drafts.get(draft.draft_id)
        if kept is not None:
            return kept
        resolved = await self._resolve(list(draft.track_ids))
        return [resolved[track_id] for track_id in draft.track_ids if track_id in resolved]

    # ── positives ──
    async def _running_tracks(self) -> dict[TrackId, Track]:
        tracks: dict[TrackId, Track] = {}
        for playlist_id in self._playlist_ids:
            try:
                tracks.update(self._remember(await self._library.playlist_tracks(playlist_id)))
            except ExternalAPIError as exc:  # e.g. a playlist the user does not own (403): skip it, keep the rest
                logger.warning("running playlist skipped: %s", exc)
        return tracks

    async def _positive_weights(self, running: Mapping[TrackId, Track]) -> dict[TrackId, float]:
        """1 per running-playlist track, + `play_weight` per play during a run, capped at `max_positive_weight`."""
        weights: defaultdict[TrackId, float] = defaultdict(float, {track_id: 1.0 for track_id in running})
        if self._plays is not None:
            try:
                recent = await self._library.recently_played()
                self._remember([track for track, _ in recent])
                self._plays.record([(track.track_id, played_at) for track, played_at in recent])
            except ExternalAPIError as exc:
                logger.warning("recently-played sync failed, using the local log only: %s", exc)
            if self._run_windows is not None:
                for track_id in self._plays.plays_within(self._run_windows()):
                    weights[track_id] += self._settings.play_weight
        cap = self._settings.max_positive_weight
        return {track_id: min(weight, cap) for track_id, weight in sorted(weights.items())}

    # ── candidate generation ──
    async def _recommended(
        self,
        positive_ids: Sequence[TrackId],
        features: Mapping[TrackId, AudioFeatures],
        target_spm: float,
        rng: random.Random,
    ) -> dict[TrackId, Track]:
        """ReccoBeats recommendations, seeded with the positives whose folded tempo is closest to the target, so the
        recommendations land near the right tempo. Each round uses 5 different seeds."""
        rounds = self._settings.recommendation_rounds
        if not positive_ids or rounds == 0:
            return {}
        tiebreak = {track_id: rng.random() for track_id in positive_ids}
        ranked = sorted(positive_ids, key=lambda t: (fold_tempo(features[t].tempo_bpm, target_spm)[1], tiebreak[t]))
        ids: list[TrackId] = []
        for round_index in range(rounds):
            seeds = ranked[round_index * 5 : (round_index + 1) * 5]
            if not seeds:
                break
            try:
                ids += await self._features.recommendations(seeds, limit=self._settings.recommendations_per_round)
            except ExternalAPIError as exc:
                logger.warning("recommendations unavailable: %s", exc)
                break
        return await self._resolve(ids)

    async def _searched(self, genres: Sequence[str]) -> dict[TrackId, Track]:
        found: dict[TrackId, Track] = {}
        for genre in genres:
            clean = sanitise_genre(genre)
            if not clean:
                continue
            try:
                tracks = await self._library.search_tracks(
                    f'genre:"{clean}"', limit=self._settings.search_tracks_per_genre
                )
            except ExternalAPIError as exc:
                logger.warning("genre search unavailable: %s", exc)
                break
            found |= self._remember(tracks)
        return found

    # ── enrichment ──
    async def _fetch_genres(
        self, tracks: Sequence[Track], genres: dict[ArtistId, tuple[str, ...]], budget: _Budget
    ) -> None:
        """Genres of each track's lead artist (one Spotify call per new artist). Stops at the budget or at the first
        failure (a 429 storm should not be prolonged by more calls); missing genres are simply unknown."""
        for track in tracks:
            if not track.artist_ids or track.artist_ids[0] in genres:
                continue
            if not budget.take():
                return
            artist_id = track.artist_ids[0]
            try:
                genres[artist_id] = tuple(await self._library.artist_genres(artist_id))
            except ExternalAPIError as exc:
                logger.warning("artist genres unavailable, continuing without: %s", exc)
                budget.exhaust()
                return

    @staticmethod
    def _profile(track: Track, features: AudioFeatures, genres: Mapping[ArtistId, tuple[str, ...]]) -> TrackProfile:
        lead_genres = genres.get(track.artist_ids[0], ()) if track.artist_ids else ()
        return TrackProfile(track_id=track.track_id, features=features, genres=lead_genres)

    # ── track bookkeeping ──
    def _remember(self, tracks: Sequence[Track]) -> dict[TrackId, Track]:
        found = {track.track_id: track for track in tracks}
        self._known.update(found)
        return found

    async def _resolve(self, track_ids: Sequence[TrackId]) -> dict[TrackId, Track]:
        """Known tracks from memory, the rest through the lookup (if any)."""
        found = {track_id: self._known[track_id] for track_id in track_ids if track_id in self._known}
        missing = [track_id for track_id in dict.fromkeys(track_ids) if track_id not in found]
        if missing and self._lookup is not None:
            try:
                found |= self._remember(list((await self._lookup.tracks(missing)).values()))
            except ExternalAPIError as exc:
                logger.warning("track lookup unavailable: %s", exc)
        return {track_id: found[track_id] for track_id in dict.fromkeys(track_ids) if track_id in found}

    def _keep(self, draft_id: DraftId, tracks: tuple[Track, ...]) -> None:
        self._drafts[draft_id] = tracks
        while len(self._drafts) > self._settings.max_kept_drafts:
            self._drafts.pop(next(iter(self._drafts)))


class _Budget:
    """A countdown of allowed external calls within one draft."""

    def __init__(self, calls: int) -> None:
        self._left = calls

    def take(self) -> bool:
        if self._left <= 0:
            return False
        self._left -= 1
        return True

    def exhaust(self) -> None:
        self._left = 0


def _mood_reference(positives: Sequence[WeightedTrack], features: Mapping[TrackId, AudioFeatures]) -> MoodReference:
    """Weighted mean energy/valence of the positives (of everything known, on a cold start)."""
    pairs = [(p.profile.features, p.weight) for p in positives] or [(f, 1.0) for f in features.values()]
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return MoodReference(energy=0.5, valence=0.5)
    return MoodReference(
        energy=sum(f.energy * w for f, w in pairs) / total,
        valence=sum(f.valence * w for f, w in pairs) / total,
    )
