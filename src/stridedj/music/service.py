"""`PlaylistPlannerService`: the application service behind the agent's playlist tool (implements `PlaylistPlanner`).

Pipeline for one draft:

1. **Positives** — tracks of the running playlists (weight 1) and tracks played *during runs* (+1 per play, capped):
   recently-played is synced into the local log first, then matched against the run windows.
2. **Own pool** — positives + saved tracks. Spoken word (audiobook chapters saved as tracks) and very long recordings
   are dropped here (`playlist.is_music`), so they neither get picked nor distort the style model.
3. **Enrich** — audio features (ReccoBeats, cached) for everything; artist genres (Spotify, one call per artist,
   cached) for positives and for tempo-eligible candidates only (a track outside ±8 BPM cannot be used anyway).
4. **Score** — fit the style model on positives vs. the rest. The *style floor* is a low quantile of the positives'
   own scores ("at least as running-like as one of the less typical favourites"); new songs below it are left out.
5. **Assemble, level by level** — `playlist.assemble_playlist` (pure), first on the own pool with the floor; only if
   that cannot fill the run are recommendations fetched (seeded with the most typical favourites that fit the tempo)
   and the floor lowered step by step. The first level that fills the duration wins. The `MusicChoice` picks the
   levels: `familiar` never adds recommendations and keeps to the running music and its artists; `discover` starts
   with recommendations and asks for more new songs. Songs of recently created playlists rank lower (rotation).

Robustness: recommendations, search and genres are optional enrichments; if they fail the draft is built without
them (logged). Missing audio features drop that track, nothing else. The running playlists are only read: this
service has no writer at all (playlist creation is the executor's job, after approval).
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pydantic import Field

from stridedj.domain import (
    ArtistId,
    AudioFeatures,
    CadenceTarget,
    DraftId,
    Frozen,
    Genre,
    Mood,
    MusicChoice,
    PlaylistDraft,
    RunType,
    TimeSlot,
    Track,
    TrackId,
)
from stridedj.errors import ExternalAPIError
from stridedj.music.genres import genres_from_tags
from stridedj.music.playlist import (
    AssembledPlaylist,
    AssemblyParams,
    Candidate,
    MoodReference,
    assemble_playlist,
    fold_tempo,
    is_music,
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
from stridedj.ports import ArtistGenresPort, AudioFeaturesPort, MusicLibraryPort, PlaylistHistoryPort

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
    max_genre_lookups: int = Field(default=300, ge=0, description="new artist genre lookups per draft (then cached)")
    max_warm_up_lookups: int = Field(default=1000, ge=0, description="artist genre lookups per background warm-up")
    play_weight: float = Field(default=1.0, gt=0, description="positive weight added per play during a run")
    max_positive_weight: float = Field(default=3.0, gt=0)
    liked_weight: float = Field(default=1.0, gt=0, description="positive weight of a song the runner liked (👍)")
    max_kept_drafts: int = Field(default=64, ge=1)
    style_floor_quantiles: tuple[float, ...] = Field(
        default=(0.25, 0.10),
        description="style floors tried in order, as quantiles of the positives' scores; after them: no floor",
    )
    rotation_playlists: int = Field(default=3, ge=0, description="created playlists whose tracks rank lower")
    discover_novelty: float = Field(default=0.6, ge=0, le=1, description="minimum share of new songs for 'discover'")
    assembly: AssemblyParams = AssemblyParams()


@dataclass(frozen=True, slots=True)
class _Level:
    """One attempt: with or without recommendations, and which style floor (None = none)."""

    recommendations: bool
    floor_quantile: float | None


@dataclass(frozen=True, slots=True)
class _ScoredPool:
    """The tempo-eligible candidates of one pool, and the positives' own scores under the same fitted model."""

    candidates: list[Candidate]
    positive_scores: dict[TrackId, float]

    def floor(self, quantile: float | None) -> float | None:
        if quantile is None or not self.positive_scores:
            return None  # cold start: nothing to compare with
        return float(np.quantile(list(self.positive_scores.values()), quantile))


class LibraryGenres:
    """`ArtistGenresPort` over the music library's own artist genres (Spotify, the demo world)."""

    def __init__(self, library: MusicLibraryPort) -> None:
        self._library = library

    def cached(self, artist_id: ArtistId) -> Sequence[str] | None:
        return None  # the library caches internally; every lookup counts against the budget

    async def artist_genres(self, artist_id: ArtistId, artist_name: str) -> Sequence[str]:
        return await self._library.artist_genres(artist_id)


class NoGenres:
    """`ArtistGenresPort` when no genre source is configured: no calls, no genres."""

    def cached(self, artist_id: ArtistId) -> Sequence[str] | None:
        return ()

    async def artist_genres(self, artist_id: ArtistId, artist_name: str) -> Sequence[str]:
        return ()


@dataclass(frozen=True, slots=True)
class _DraftContext:
    """What every assembly level of one draft shares."""

    target_spm: float
    positives: list[WeightedTrack]
    running: set[TrackId]
    familiar_artists: set[ArtistId]
    recent_uses: dict[TrackId, int]
    genre: Genre | None  # the runner's genre choice: only songs of it
    excluded: set[TrackId]  # songs the runner disliked (👎): never candidates, seeds or positives


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
        history: PlaylistHistoryPort | None = None,
        genres: ArtistGenresPort | None = None,
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
        self._history = history
        self._genre_source: ArtistGenresPort = genres or LibraryGenres(library)
        self._fit = style_fitter
        self._settings = settings or PlannerSettings()
        self._rng = random.Random(seed)  # noqa: S311 (not crypto: tie-breaking and draft ids)
        self._known: dict[TrackId, Track] = {}
        self._drafts: dict[DraftId, tuple[Track, ...]] = {}

    # ── PlaylistPlanner ──
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
        seed = self._rng.getrandbits(64)  # one reproducible stream per draft
        target_spm = cadence.spm

        running = await self._running_tracks()
        ratings = dict(self._history.track_ratings(favourites=set(running))) if self._history is not None else {}
        excluded = {track_id for track_id, rating in ratings.items() if rating < 0}
        weights = await self._positive_weights(running)
        for track_id, rating in ratings.items():
            if rating > 0:
                weights[track_id] = max(weights.get(track_id, 0.0), self._settings.liked_weight)
        weights = {track_id: w for track_id, w in weights.items() if track_id not in excluded}
        saved = self._remember(await self._library.saved_tracks(limit=self._settings.saved_tracks_limit))
        pool = {**saved, **(await self._resolve(list(weights))), **running}
        pool = {track_id: track for track_id, track in pool.items() if track_id not in excluded}
        features = await self._music_features(pool)

        genres: dict[ArtistId, tuple[str, ...]] = {}
        budget = _Budget(self._settings.max_genre_lookups)
        positive_ids = [track_id for track_id in weights if track_id in features]
        await self._fetch_genres([pool[t] for t in positive_ids], genres, budget)
        positives = [
            WeightedTrack(profile=self._profile(pool[t], features[t], genres), weight=weights[t]) for t in positive_ids
        ]
        context = _DraftContext(
            target_spm=target_spm,
            positives=positives,
            running=set(running),
            familiar_artists={artist for t in positive_ids for artist in pool[t].artist_ids},
            recent_uses=dict(self._history.recent_track_uses(last_playlists=self._settings.rotation_playlists))
            if self._history is not None
            else {},
            genre=genre,
            excluded=excluded,
        )
        mood_reference = _mood_reference(positives, features)
        params = self._params(music)

        scored = await self._score_pool(pool, features, genres, budget, context)
        best: AssembledPlaylist | None = None
        extended, used_level = False, 0
        for index, level in enumerate(self._levels(music, genre)):
            if level.recommendations and not extended:
                extended = True
                extra = await self._extra_candidates(positive_ids, features, scored, context, pool, seed)
                if extra:
                    features |= await self._music_features(extra)
                    pool |= {track_id: track for track_id, track in extra.items() if track_id in features}
                    scored = await self._score_pool(pool, features, genres, budget, context)
            candidates = scored.candidates
            if music is MusicChoice.FAMILIAR:
                candidates = [c for c in candidates if c.in_running_playlists or c.familiar_artist]
            if genre is not None:
                candidates = [c for c in candidates if genre.value in c.genres]
            assembled = assemble_playlist(
                candidates,
                run_type=run_type,
                target_spm=target_spm,
                duration_min=duration_min,
                mood=mood,
                mood_reference=mood_reference,
                rng=random.Random(seed),  # noqa: S311 (same draft id and jitter at every level)
                params=params,
                style_floor=scored.floor(level.floor_quantile),
            )
            if best is None or assembled.draft.total_min > best.draft.total_min:
                best, used_level = assembled, index
            if assembled.draft.total_min >= duration_min:
                break
        if best is None:  # unreachable: every music choice has at least one level
            raise RuntimeError("no assembly level ran")

        self._keep(best.draft.draft_id, best.tracks)
        logger.info(
            "playlist draft %s (%s%s, level %d): %d tracks, %.1f min, ±%.0f BPM, %d positives, %d candidates",
            best.draft.draft_id,
            music.value,
            f", {genre.value}" if genre else "",
            used_level,
            len(best.tracks),
            best.draft.total_min,
            best.draft.tolerance_bpm,
            len(positives),
            len(scored.candidates),
        )
        return best.draft

    async def warm_up(self) -> int:
        """Look up the genres of every library artist not known yet (running playlists first, then saved tracks), so
        drafts rarely wait for the genre source. Meant for a background job; returns how many were looked up."""
        running = await self._running_tracks()
        saved = self._remember(await self._library.saved_tracks(limit=self._settings.saved_tracks_limit))
        budget = _Budget(self._settings.max_warm_up_lookups)
        await self._fetch_genres([*running.values(), *saved.values()], {}, budget)
        return budget.used

    async def tracks_for(self, draft: PlaylistDraft) -> Sequence[Track]:
        """The draft's tracks in order. After a restart they are re-resolved via the lookup (unknown ids skipped)."""
        kept = self._drafts.get(draft.draft_id)
        if kept is not None:
            return kept
        return await self.tracks_of(draft.track_ids)

    async def tracks_of(self, track_ids: Sequence[TrackId]) -> Sequence[Track]:
        resolved = await self._resolve(list(track_ids))
        return [resolved[track_id] for track_id in track_ids if track_id in resolved]

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

    # ── levels ──
    def _levels(self, music: MusicChoice, genre: Genre | None) -> list[_Level]:
        # a chosen genre IS the style: songs of it are ranked by fit, but none is rejected for sounding different
        floors: list[float | None] = [None] if genre else [*self._settings.style_floor_quantiles, None]
        if music is MusicChoice.FAMILIAR:
            return [_Level(recommendations=False, floor_quantile=q) for q in floors]
        with_recommendations = [_Level(recommendations=True, floor_quantile=q) for q in floors]
        if music is MusicChoice.DISCOVER:
            return with_recommendations
        return [_Level(recommendations=False, floor_quantile=floors[0]), *with_recommendations]

    def _params(self, music: MusicChoice) -> AssemblyParams:
        params = self._settings.assembly
        if music is MusicChoice.FAMILIAR:
            return params.model_copy(update={"novelty_target": 0.0})
        if music is MusicChoice.DISCOVER:
            return params.model_copy(
                update={"novelty_target": max(params.novelty_target, self._settings.discover_novelty)}
            )
        return params

    async def _score_pool(
        self,
        pool: Mapping[TrackId, Track],
        features: Mapping[TrackId, AudioFeatures],
        genres: dict[ArtistId, tuple[str, ...]],
        budget: _Budget,
        context: _DraftContext,
    ) -> _ScoredPool:
        """Fit the style model on this pool and turn its tempo-eligible tracks into candidates."""
        max_tolerance = max(self._settings.assembly.tolerance_ladder_bpm)
        eligible = [
            track
            for track_id, track in sorted(pool.items())
            if track_id in features and fold_tempo(features[track_id].tempo_bpm, context.target_spm)[1] <= max_tolerance
        ]
        await self._fetch_genres(eligible, genres, budget)
        positive_ids = {p.profile.track_id for p in context.positives}
        unlabeled = [
            self._profile(track, features[track_id], genres)
            for track_id, track in sorted(pool.items())
            if track_id in features and track_id not in positive_ids
        ]
        model = self._fit(context.positives, unlabeled)
        positive_scores = model.score_many([p.profile for p in context.positives])
        profiles = [self._profile(track, features[track.track_id], genres) for track in eligible]
        candidates = []
        for track, profile, score in zip(eligible, profiles, model.score_many(profiles), strict=True):
            in_running = track.track_id in context.running
            candidates.append(
                Candidate(
                    track=track,
                    features=profile.features,
                    genres=profile.genres,
                    style_score=score,
                    in_running_playlists=in_running,
                    familiar_artist=not in_running and any(a in context.familiar_artists for a in track.artist_ids),
                    recent_uses=context.recent_uses.get(track.track_id, 0),
                )
            )
        return _ScoredPool(
            candidates=candidates,
            positive_scores={p.profile.track_id: s for p, s in zip(context.positives, positive_scores, strict=True)},
        )

    # ── candidates beyond the library ──
    async def _extra_candidates(
        self,
        positive_ids: Sequence[TrackId],
        features: Mapping[TrackId, AudioFeatures],
        scored: _ScoredPool,
        context: _DraftContext,
        pool: Mapping[TrackId, Track],
        seed: int,
    ) -> dict[TrackId, Track]:
        """Recommendations seeded with the most typical favourites that fit the tempo, plus genre search; only
        tracks not already in the pool."""
        rng = random.Random(seed)  # noqa: S311 (tie-breaking only)
        max_tolerance = max(self._settings.assembly.tolerance_ladder_bpm)
        tiebreak = {track_id: rng.random() for track_id in positive_ids}
        genre_of = {p.profile.track_id: p.profile.genres for p in context.positives}
        wanted = context.genre.value if context.genre else None
        seeds = sorted(
            positive_ids,
            key=lambda t: (
                wanted is not None and wanted not in genre_of.get(t, ()),
                fold_tempo(features[t].tempo_bpm, context.target_spm)[1] > max_tolerance,
                -scored.positive_scores.get(t, 0.0),
                tiebreak[t],
            ),
        )
        extra = await self._recommended(seeds)
        search = top_profile_genres(context.positives, self._settings.search_genres)
        if context.genre is not None:
            search = [_SEARCH_TERM.get(context.genre, context.genre.value), *search]
        extra |= await self._searched(list(dict.fromkeys(search)))
        return {t: track for t, track in extra.items() if t not in pool and t not in context.excluded}

    async def _recommended(self, ranked_seeds: Sequence[TrackId]) -> dict[TrackId, Track]:
        """ReccoBeats recommendations, 5 different seeds per round, best-ranked seeds first."""
        ids: list[TrackId] = []
        for round_index in range(self._settings.recommendation_rounds):
            seeds = ranked_seeds[round_index * 5 : (round_index + 1) * 5]
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
    async def _music_features(self, tracks: Mapping[TrackId, Track]) -> dict[TrackId, AudioFeatures]:
        """Audio features of the tracks that are music: spoken word, very long recordings and tracks without
        features are left out (and so are dropped from everything downstream)."""
        found = await self._features.audio_features(list(tracks))
        return {t: f for t, f in found.items() if t in tracks and is_music(tracks[t], f)}

    async def _fetch_genres(
        self, tracks: Sequence[Track], genres: dict[ArtistId, tuple[str, ...]], budget: _Budget
    ) -> None:
        """Genres of each track's lead artist, mapped onto the `Genre` vocabulary. Cached answers are free; the budget
        counts network lookups only. After the first failure (a 429 should not be prolonged by more calls) only cached
        answers are used; missing genres are simply unknown."""
        for track in tracks:
            if not track.artist_ids or track.artist_ids[0] in genres:
                continue
            artist_id = track.artist_ids[0]
            raw = self._genre_source.cached(artist_id)
            if raw is None:
                if not budget.take():
                    continue
                name = track.artist_names[0] if track.artist_names else ""
                try:
                    raw = await self._genre_source.artist_genres(artist_id, name)
                except ExternalAPIError as exc:
                    logger.warning("artist genres unavailable, continuing with cached ones: %s", exc)
                    budget.exhaust()
                    continue
            genres[artist_id] = tuple(g.value for g in genres_from_tags(raw))

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


_SEARCH_TERM: dict[Genre, str] = {Genre.FUNK_DISCO: "disco", Genre.RNB: "r&b"}  # Spotify's own genre names


class _Budget:
    """A countdown of allowed external calls within one draft."""

    def __init__(self, calls: int) -> None:
        self._left = calls
        self.used = 0

    def take(self) -> bool:
        if self._left <= 0:
            return False
        self._left -= 1
        self.used += 1
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
