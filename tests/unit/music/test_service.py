"""End-to-end planner tests on the offline demo world (no network)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from stridedj.domain import ArtistId, AudioFeatures, CadenceTarget, Mood, RunType, Track, TrackId
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
