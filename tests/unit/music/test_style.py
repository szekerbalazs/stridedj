from __future__ import annotations

import random
from collections.abc import Sequence
from statistics import mean

import pytest

from stridedj.domain import AudioFeatures, TrackId
from stridedj.music.demo import build_demo_world
from stridedj.music.style import (
    CentroidStyleModel,
    NeutralStyleModel,
    StyleModel,
    TrackProfile,
    WeightedTrack,
    fit_style_model,
    held_out_scores,
    top_profile_genres,
)


def _profile(n: int, *, energy: float, acoustic: float, genres: tuple[str, ...] = ()) -> TrackProfile:
    track_id = TrackId(f"t{n:021d}")
    features = AudioFeatures(
        track_id=track_id,
        tempo_bpm=170,
        energy=energy,
        danceability=0.6,
        valence=0.5,
        acousticness=acoustic,
        instrumentalness=0.1,
        speechiness=0.05,
        loudness_db=-6 - 10 * acoustic,
    )
    return TrackProfile(track_id=track_id, features=features, genres=genres)


RUNNING = [_profile(i, energy=0.85 + 0.01 * i, acoustic=0.05, genres=("drum and bass",)) for i in range(5)]
CALM = [_profile(100 + i, energy=0.2 + 0.01 * i, acoustic=0.8, genres=("folk",)) for i in range(5)]


def test_running_like_tracks_score_higher() -> None:
    model = CentroidStyleModel.fit([WeightedTrack(profile=p) for p in RUNNING], CALM)
    energetic = _profile(200, energy=0.9, acoustic=0.05)
    calm = _profile(201, energy=0.25, acoustic=0.75)
    assert model.score(energetic) > 0.8
    assert model.score(calm) < -0.8
    assert model.score_many([energetic, calm]) == pytest.approx([model.score(energetic), model.score(calm)])


def test_genre_overlap_is_blended_and_missing_genres_are_not_penalised() -> None:
    model = CentroidStyleModel.fit([WeightedTrack(profile=p) for p in RUNNING], CALM)
    same_audio = {"energy": 0.6, "acoustic": 0.3}
    matching = _profile(300, **same_audio, genres=("drum and bass",))
    unknown = _profile(301, **same_audio)
    mismatching = _profile(302, **same_audio, genres=("folk",))
    assert model.score(matching) > model.score(unknown) > model.score(mismatching)


def test_weights_pull_the_centroid() -> None:
    light = CentroidStyleModel.fit([WeightedTrack(profile=RUNNING[0]), WeightedTrack(profile=CALM[0])], [])
    heavy = CentroidStyleModel.fit([WeightedTrack(profile=RUNNING[0], weight=3), WeightedTrack(profile=CALM[0])], [])
    assert heavy.centroid_raw["energy"] > light.centroid_raw["energy"]


def test_cold_start_uses_the_neutral_model() -> None:
    model = fit_style_model([], CALM)
    assert isinstance(model, NeutralStyleModel)
    assert model.score_many(CALM) == [0.0] * len(CALM)
    with pytest.raises(ValueError, match="positive"):
        CentroidStyleModel.fit([], CALM)


def test_top_profile_genres_are_weighted_and_deterministic() -> None:
    positives = [
        WeightedTrack(profile=_profile(1, energy=0.8, acoustic=0.1, genres=("b", "a")), weight=2),
        WeightedTrack(profile=_profile(2, energy=0.8, acoustic=0.1, genres=("c",)), weight=1),
    ]
    assert top_profile_genres(positives, 2) == ["a", "b"]


def _demo_positives_and_library() -> tuple[list[WeightedTrack], list[TrackProfile]]:
    world = build_demo_world(seed=11)
    genres = world.artist_genres
    library = [
        TrackProfile(track_id=t, features=world.features[t], genres=genres[world.tracks[t].artist_ids[0]])
        for t in world.saved_track_ids
        if t in world.features
    ]
    running = set(world.running_track_ids)
    positives = [WeightedTrack(profile=p) for p in library if p.track_id in running]
    return positives, library


def test_held_out_positives_outscore_other_tracks_on_demo_data() -> None:
    positives, library = _demo_positives_and_library()
    result = held_out_scores(fit_style_model, positives, library, holdout_fraction=0.3, rng=random.Random(0))
    assert len(result.positive_scores) == round(0.3 * len(positives))
    assert result.n_train_positives == len(positives) - len(result.positive_scores)
    assert len(result.other_scores) == len(library) - len(positives)
    assert mean(result.positive_scores) > mean(result.other_scores) + 0.3


def test_held_out_scores_are_deterministic_and_model_is_swappable() -> None:
    positives, library = _demo_positives_and_library()
    first = held_out_scores(fit_style_model, positives, library, rng=random.Random(5))
    assert first == held_out_scores(fit_style_model, positives, library, rng=random.Random(5))

    def always_neutral(_: Sequence[WeightedTrack], __: Sequence[TrackProfile]) -> StyleModel:
        return NeutralStyleModel()

    swapped = held_out_scores(always_neutral, positives, library, rng=random.Random(5))
    assert set(swapped.positive_scores) == {0.0}
