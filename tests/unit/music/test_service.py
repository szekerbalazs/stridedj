"""End-to-end planner tests on the offline demo world (no network)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from stridedj.domain import (
    ArtistId,
    AudioFeatures,
    CadenceTarget,
    DraftId,
    Genre,
    Mood,
    MusicChoice,
    PlaylistDraft,
    RunType,
    Track,
    TrackId,
)
from stridedj.errors import ExternalAPIError
from stridedj.music.demo import (
    MALICIOUS_TRACK_NAMES,
    DemoMusicWorld,
    FakeAudioFeatures,
    FakeMusicLibrary,
    RecordingPlaylistWriter,
    build_demo_planner,
    build_demo_world,
)
from stridedj.music.genres import genres_from_tags
from stridedj.music.history import PlaylistHistory
from stridedj.music.playlist import fold_tempo, summarise
from stridedj.music.plays_logger import PlaysLogger
from stridedj.music.service import ChainedTrackLookup, PlannerSettings, PlaylistPlannerService


def _cadence(spm: float = 168.0) -> CadenceTarget:
    return CadenceTarget(run_type=RunType.Z2, spm=spm, n_runs=12, lookback_days=60)


@pytest.fixture(scope="module")
def world() -> DemoMusicWorld:
    return build_demo_world(seed=7)


async def test_draft_meets_every_constraint(world: DemoMusicWorld) -> None:
    planner = build_demo_planner(world)
    draft = await planner.draft(run_type=RunType.Z2, cadence=_cadence(), duration_min=62, mood=Mood.USUAL)

    assert draft.run_type is RunType.Z2
    assert draft.total_min >= 62
    assert draft.novelty_share >= 0.3
    assert len(set(draft.track_ids)) == len(draft.track_ids)
    errors = [fold_tempo(world.features[t].tempo_bpm, 168.0)[1] for t in draft.track_ids]
    assert max(errors) <= draft.tolerance_bpm
    artists = Counter(world.tracks[t].artist_ids[0] for t in draft.track_ids)
    assert max(artists.values()) <= 2
    running = set(world.running_track_ids)
    assert any(t in running for t in draft.track_ids)  # familiar favourites ...
    assert any(t not in running for t in draft.track_ids)  # ... and discoveries


async def test_tracks_for_returns_the_draft_in_order(world: DemoMusicWorld) -> None:
    planner = build_demo_planner(world)
    draft = await planner.draft(run_type=RunType.Z2, cadence=_cadence(), duration_min=30, mood=Mood.USUAL)
    tracks = await planner.tracks_for(draft)
    assert tuple(t.track_id for t in tracks) == draft.track_ids
    assert sum(t.duration_ms for t in tracks) / 60_000 == pytest.approx(draft.total_min)

    restarted = build_demo_planner(world, seed=99)  # no in-memory draft: resolved through the lookup
    assert tuple(t.track_id for t in await restarted.tracks_for(draft)) == draft.track_ids


async def test_injection_titles_are_eligible_but_never_in_the_llm_view(world: DemoMusicWorld) -> None:
    planner = build_demo_planner(world)
    draft = await planner.draft(run_type=RunType.Z2, cadence=_cadence(168.0), duration_min=60, mood=Mood.USUAL)
    assert set(world.malicious_track_ids) <= set(draft.track_ids)
    displayed = {t.name for t in await planner.tracks_for(draft)}
    assert set(MALICIOUS_TRACK_NAMES) <= displayed  # shown to the user by code ...

    llm_view = summarise(draft).model_dump_json() + draft.model_dump_json()
    for text in [*MALICIOUS_TRACK_NAMES, "attacker@example.com", "API key", "Demo Track", "Demo Artist"]:
        assert text not in llm_view  # ... never to the model


async def test_mood_shifts_energy(world: DemoMusicWorld) -> None:
    energy = {}
    for mood in (Mood.CALMER, Mood.MORE_ENERGETIC):
        planner = build_demo_planner(world, settings=PlannerSettings())
        draft = await planner.draft(run_type=RunType.Z2, cadence=_cadence(165.0), duration_min=60, mood=mood)
        energy[mood] = sum(world.features[t].energy for t in draft.track_ids) / len(draft.track_ids)
    assert energy[Mood.MORE_ENERGETIC] > energy[Mood.CALMER]


async def test_same_seed_same_drafts(world: DemoMusicWorld) -> None:
    async def two_drafts(seed: int) -> list[tuple[str, tuple[TrackId, ...]]]:
        planner = build_demo_planner(world, seed=seed)
        drafts = [
            await planner.draft(run_type=RunType.Z2, cadence=_cadence(), duration_min=40, mood=Mood.USUAL)
            for _ in range(2)
        ]
        return [(d.draft_id, d.track_ids) for d in drafts]

    assert await two_drafts(5) == await two_drafts(5)
    assert (await two_drafts(5))[0][0] != (await two_drafts(6))[0][0]


async def test_running_playlist_is_never_modified(world: DemoMusicWorld) -> None:
    library = world.library()
    before = library.playlist_snapshot(world.running_playlist_id)
    planner = PlaylistPlannerService(
        library=library, features=world.audio_features(), running_playlist_ids=[world.running_playlist_id], seed=1
    )
    draft = await planner.draft(run_type=RunType.Z2, cadence=_cadence(), duration_min=45, mood=Mood.USUAL)
    writer = RecordingPlaylistWriter()  # creation happens later, in the executor, into a NEW playlist
    url = await writer.create_private_playlist(name="StrideDJ Z2", description="", track_ids=draft.track_ids)
    assert library.playlist_snapshot(world.running_playlist_id) == before
    assert writer.created[0].track_ids == draft.track_ids
    assert url.startswith("https://open.spotify.com/playlist/")


async def test_plays_are_synced_and_used(world: DemoMusicWorld) -> None:
    plays = PlaysLogger(Path(":memory:"))
    library = world.library()
    planner = PlaylistPlannerService(
        library=library,
        features=world.audio_features(),
        running_playlist_ids=[],  # only plays during runs define the profile
        track_lookup=library,
        plays=plays,
        run_windows=lambda: world.run_windows,
        seed=1,
    )
    draft = await planner.draft(run_type=RunType.Z2, cadence=_cadence(), duration_min=30, mood=Mood.USUAL)
    assert plays.count() == 50
    assert draft.total_min >= 30
    assert draft.novelty_share == 1.0  # no running playlists → every track is "novel"


class _BrokenFeatures(FakeAudioFeatures):
    async def recommendations(self, seed_track_ids: Sequence[TrackId], *, limit: int) -> Sequence[TrackId]:
        raise ExternalAPIError("reccobeats", "down", status_code=503, retryable=True)


class _BrokenLibrary(FakeMusicLibrary):
    async def artist_genres(self, artist_id: ArtistId) -> Sequence[str]:
        self.calls["artist_genres"] += 1
        raise ExternalAPIError("spotify", "rate limited", status_code=429, retryable=True)

    async def search_tracks(self, query: str, *, limit: int) -> Sequence[Track]:
        raise ExternalAPIError("spotify", "rate limited", status_code=429, retryable=True)


async def test_optional_enrichments_degrade_gracefully(world: DemoMusicWorld) -> None:
    library = _BrokenLibrary(
        tracks=world.tracks,
        playlists={world.running_playlist_id: world.running_track_ids},
        saved=world.saved_track_ids,
        plays=world.plays,
        artist_genres=world.artist_genres,
    )
    planner = PlaylistPlannerService(
        library=library,
        features=_BrokenFeatures(world.features),
        running_playlist_ids=[world.running_playlist_id],
        seed=1,
    )
    draft = await planner.draft(run_type=RunType.Z2, cadence=_cadence(), duration_min=45, mood=Mood.USUAL)
    assert draft.total_min >= 45
    assert draft.top_genres == ()
    assert library.calls["artist_genres"] == 1  # stops at the first failure instead of hammering a 429


async def test_genre_lookups_respect_the_budget(world: DemoMusicWorld) -> None:
    library = world.library()
    planner = PlaylistPlannerService(
        library=library,
        features=world.audio_features(),
        running_playlist_ids=[world.running_playlist_id],
        settings=PlannerSettings(max_genre_lookups=5),
        seed=1,
    )
    await planner.draft(run_type=RunType.Z2, cadence=_cadence(), duration_min=30, mood=Mood.USUAL)
    assert library.calls["artist_genres"] == 5


async def test_cold_start_without_any_positives(world: DemoMusicWorld) -> None:
    planner = PlaylistPlannerService(
        library=world.library(), features=world.audio_features(), running_playlist_ids=[], seed=1
    )
    draft = await planner.draft(run_type=RunType.Z2, cadence=_cadence(), duration_min=30, mood=Mood.USUAL)
    assert draft.total_min >= 30


async def test_chained_lookup_asks_the_next_lookup_only_for_misses() -> None:
    class Lookup:
        def __init__(self, known: Mapping[TrackId, Track]) -> None:
            self.known, self.asked = known, list[list[TrackId]]()

        async def tracks(self, track_ids: Sequence[TrackId]) -> Mapping[TrackId, Track]:
            self.asked.append(list(track_ids))
            return {t: self.known[t] for t in track_ids if t in self.known}

    def track(n: int) -> Track:
        return Track(track_id=TrackId(f"t{n}"), name="x", artist_ids=(), artist_names=(), duration_ms=1)

    first, second = Lookup({TrackId("t1"): track(1)}), Lookup({TrackId("t2"): track(2)})
    found = await ChainedTrackLookup(first, second).tracks([TrackId("t1"), TrackId("t2"), TrackId("t3")])
    assert set(found) == {"t1", "t2"}
    assert second.asked == [["t2", "t3"]]


def test_demo_world_shape(world: DemoMusicWorld) -> None:
    assert len(world.saved_track_ids) == 300
    assert len(world.tracks) > len(world.saved_track_ids)  # a catalogue beyond the library, for discovery
    assert set(world.running_track_ids) <= set(world.saved_track_ids)
    assert 30 <= len(world.running_track_ids) <= 45
    assert len(world.features) < len(world.tracks)  # ReccoBeats gaps
    assert {world.tracks[t].name for t in world.malicious_track_ids} == set(MALICIOUS_TRACK_NAMES)
    assert all(150_000 <= t.duration_ms <= 300_000 for t in world.tracks.values())
    half_time = [f for f in world.features.values() if 80 <= f.tempo_bpm <= 95]
    assert len(half_time) > 20
    assert all(played_at.tzinfo is not None for _, played_at in world.plays)
    assert build_demo_world(seed=7).plays == world.plays


async def test_fake_library_mimics_spotify_limits(world: DemoMusicWorld) -> None:
    library = world.library()
    recent = await library.recently_played()
    assert len(recent) == 50
    assert recent[0][1] >= recent[-1][1]  # newest first
    hits = await library.search_tracks('genre:"drum and bass"', limit=7)
    assert len(hits) == 7
    features: Mapping[TrackId, AudioFeatures] = await world.audio_features().audio_features(list(world.tracks))
    assert len(features) == len(world.features)


# ─────────────────────────────── where the songs come from (off-style songs) ───────────────
def _planner(
    world: DemoMusicWorld, *, features: FakeAudioFeatures | None = None, history: PlaylistHistory | None = None
) -> tuple[PlaylistPlannerService, FakeAudioFeatures]:
    library = world.library()
    audio = features or world.audio_features()
    plays = PlaysLogger(Path(":memory:"))
    plays.record(list(world.plays))
    planner = PlaylistPlannerService(
        library=library,
        features=audio,
        running_playlist_ids=[world.running_playlist_id],
        track_lookup=library,
        plays=plays,
        run_windows=lambda: world.run_windows,
        history=history,
        seed=3,
    )
    return planner, audio


def _positive_artists(world: DemoMusicWorld) -> set[ArtistId]:
    plays = PlaysLogger(Path(":memory:"))
    plays.record(list(world.plays))
    positives = set(world.running_track_ids) | set(plays.plays_within(world.run_windows))
    return {artist for t in positives for artist in world.tracks[t].artist_ids}


async def test_spoken_word_is_never_drafted(world: DemoMusicWorld) -> None:
    """Spoken word saved as tracks: tempo-perfect, but never music to run to."""
    running = set(world.running_track_ids)
    spoken = {t for t in world.saved_track_ids if t not in running and t in world.features}
    features = {t: f.model_copy(update={"speechiness": 0.94}) if t in spoken else f for t, f in world.features.items()}
    planner, _ = _planner(world, features=FakeAudioFeatures(features))
    for spm in (160.0, 168.0, 176.0):
        draft = await planner.draft(run_type=RunType.Z2, cadence=_cadence(spm), duration_min=60, mood=Mood.USUAL)
        assert not spoken & set(draft.track_ids)


async def test_usual_fills_from_the_own_library_without_recommendations(world: DemoMusicWorld) -> None:
    planner, audio = _planner(world)
    draft = await planner.draft(run_type=RunType.Z2, cadence=_cadence(), duration_min=40, mood=Mood.USUAL)
    assert draft.total_min >= 40
    assert audio.calls["recommendations"] == 0  # own library first: no external suggestions needed


async def test_recommendations_fill_gaps_when_the_library_is_too_short(world: DemoMusicWorld) -> None:
    planner, audio = _planner(world)
    await planner.draft(run_type=RunType.Z2, cadence=_cadence(), duration_min=300, mood=Mood.USUAL)
    assert audio.calls["recommendations"] >= 1


async def test_familiar_keeps_to_the_running_music_and_its_artists(world: DemoMusicWorld) -> None:
    planner, audio = _planner(world)
    draft = await planner.draft(
        run_type=RunType.Z2, cadence=_cadence(), duration_min=300, mood=Mood.USUAL, music=MusicChoice.FAMILIAR
    )
    artists = _positive_artists(world)
    assert draft.track_ids
    assert all(set(world.tracks[t].artist_ids) & artists for t in draft.track_ids)
    assert audio.calls["recommendations"] == 0  # even when the run is far longer than the music


async def test_discover_asks_for_more_new_songs(world: DemoMusicWorld) -> None:
    planner, audio = _planner(world)
    draft = await planner.draft(
        run_type=RunType.Z2, cadence=_cadence(), duration_min=60, mood=Mood.USUAL, music=MusicChoice.DISCOVER
    )
    assert draft.novelty_share >= 0.6
    assert audio.calls["recommendations"] >= 1


async def test_songs_of_created_playlists_rotate(world: DemoMusicWorld) -> None:
    history = PlaylistHistory(Path(":memory:"))
    planner, _ = _planner(world, history=history)

    async def draft() -> tuple[TrackId, ...]:
        made = await planner.draft(run_type=RunType.Z2, cadence=_cadence(), duration_min=40, mood=Mood.USUAL)
        return made.track_ids

    first = await draft()
    assert len(set(first) & set(await draft())) / len(first) > 0.8  # re-planning in a chat keeps its songs
    history.record_created(
        await planner.draft(run_type=RunType.Z2, cadence=_cadence(), duration_min=40, mood=Mood.USUAL)
    )
    # ... the next run gets a clearly different one (the strongest favourites may come back: ~40-55 % here, and
    # 0-30 % between later playlists, measured on this world with the default penalty)
    assert len(set(first) & set(await draft())) / len(first) <= 0.6


# ─────────────────────────────── genres (Last.fm live; the demo world's own tags here) ───────────────
async def test_a_chosen_genre_gives_only_songs_of_that_genre(world: DemoMusicWorld) -> None:
    planner, _ = _planner(world)
    draft = await planner.draft(
        run_type=RunType.Z2, cadence=_cadence(), duration_min=40, mood=Mood.USUAL, genre=Genre.DRUM_AND_BASS
    )
    assert draft.track_ids
    for track_id in draft.track_ids:
        lead = world.tracks[track_id].artist_ids[0]
        assert Genre.DRUM_AND_BASS in genres_from_tags(world.artist_genres.get(lead, ()))
    assert draft.top_genres == ("drum and bass",)  # closed values only: safe for the model


class _CachedGenres:
    """Every artist is already in the cache: no lookup may count against the budget."""

    def __init__(self, world: DemoMusicWorld) -> None:
        self.world, self.calls = world, 0

    def cached(self, artist_id: ArtistId) -> Sequence[str] | None:
        return self.world.artist_genres.get(artist_id, ())

    async def artist_genres(self, artist_id: ArtistId, artist_name: str) -> Sequence[str]:
        self.calls += 1
        return ()


async def test_cached_genres_are_used_beyond_the_lookup_budget(world: DemoMusicWorld) -> None:
    source = _CachedGenres(world)
    library = world.library()
    planner = PlaylistPlannerService(
        library=library,
        features=world.audio_features(),
        running_playlist_ids=[world.running_playlist_id],
        genres=source,
        settings=PlannerSettings(max_genre_lookups=0),
        seed=1,
    )
    draft = await planner.draft(run_type=RunType.Z2, cadence=_cadence(), duration_min=40, mood=Mood.USUAL)
    assert draft.top_genres  # known from the cache although no lookup was allowed
    assert source.calls == 0
    assert library.calls["artist_genres"] == 0


async def test_warm_up_looks_up_each_library_artist_once(world: DemoMusicWorld) -> None:
    library = world.library()
    planner = PlaylistPlannerService(
        library=library, features=world.audio_features(), running_playlist_ids=[world.running_playlist_id], seed=1
    )
    looked_up = await planner.warm_up()
    library_artists = {world.tracks[t].artist_ids[0] for t in (*world.running_track_ids, *world.saved_track_ids)}
    assert looked_up == library.calls["artist_genres"] == len(library_artists)


# ─────────────────────────────── feedback (the bot's 👍 / 👎) ───────────────
async def test_disliked_songs_are_never_drafted_again(world: DemoMusicWorld) -> None:
    history = PlaylistHistory(Path(":memory:"))
    planner, _ = _planner(world, history=history)
    first = await planner.draft(run_type=RunType.Z2, cadence=_cadence(), duration_min=40, mood=Mood.USUAL)
    history.record_created(first)
    for track_id in first.track_ids:
        history.toggle_song_dislike(track_id)
    again = await planner.draft(run_type=RunType.Z2, cadence=_cadence(), duration_min=300, mood=Mood.USUAL)
    assert not set(first.track_ids) & set(again.track_ids)


async def test_a_liked_new_song_counts_as_familiar(world: DemoMusicWorld) -> None:
    """👍 on a playlist turns its new songs into favourites: 'familiar' may then use them."""
    artists = _positive_artists(world)
    stranger = next(
        t
        for t in world.saved_track_ids
        if t in world.features
        and not set(world.tracks[t].artist_ids) & artists
        and fold_tempo(world.features[t].tempo_bpm, 168.0)[1] <= 3
        and world.features[t].speechiness <= 0.66
    )
    history = PlaylistHistory(Path(":memory:"))
    planner, _ = _planner(world, history=history)

    artist = world.tracks[stranger].artist_ids[0]

    async def familiar_artists_used() -> set[ArtistId]:
        made = await planner.draft(
            run_type=RunType.Z2, cadence=_cadence(), duration_min=300, mood=Mood.USUAL, music=MusicChoice.FAMILIAR
        )
        return {a for t in made.track_ids for a in world.tracks[t].artist_ids}

    assert artist not in await familiar_artists_used()
    history.record_created(
        PlaylistDraft(
            draft_id=DraftId("cccccccccccc"),
            run_type=RunType.Z2,
            target_spm=168,
            tolerance_bpm=3,
            track_ids=(stranger,),
            total_min=4,
            mean_abs_bpm_error=0,
            novelty_share=1,
        )
    )
    history.rate_playlist("cccccccccccc", 1)
    # the liked song is a favourite now, so its artist is familiar (which of their songs win is up to the ranking:
    # max 2 per artist, and the liked one was just used, so rotation ranks it lower)
    assert artist in await familiar_artists_used()
