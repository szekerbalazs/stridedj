"""The "what I run to" style model: audio features + artist genres, scored against the user's running profile.

Problem framing: we only have **positive** examples (tracks in the running playlists, tracks played during runs) and
a pool of **unlabeled** tracks (saved tracks, recommendations, search results). Nobody tells us "I would never run to
this". That is PU learning; the baseline here is the simplest sound answer to it:

* Represent a track by its 7 audio features, standardised against the whole pool (positives ∪ unlabeled), plus the
  bag of its artist's genres.
* The running profile is the (weighted) centroid of the positives in that standardised space. Because the space is
  centred on the pool, the centroid points in the direction in which running music *differs* from the user's music in
  general; features on which positives are average contribute ~0.
* `score = (1 − β)·cos(z(track), centroid) + β·cos(genres(track), genre profile)`; when either side has no genres
  (Spotify's artist genres are deprecated and often empty) the genre term is dropped, not counted as a mismatch.

Everything downstream depends only on the `StyleModel` / `StyleFitter` protocols, so a PU-learning logistic model
(e.g. Elkan & Noto, or spy-EM) can replace the centroid without touching the planner. `held_out_scores` provides raw
scores for an offline evaluation without baking in a metric choice.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt
from pydantic import Field

from stridedj.domain import AudioFeatures, Frozen, TrackId

STYLE_FEATURES: tuple[str, ...] = (
    "energy",
    "danceability",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "loudness_db",
)
DEFAULT_GENRE_BLEND = 0.3
_MIN_STD = 1e-6  # a feature that is constant over the pool carries no information: its z-score is 0
_EPS = 1e-12

FloatArray = npt.NDArray[np.float64]


class TrackProfile(Frozen):
    """What the style model may see about a track: audio features and genre tags (no title, no artist names)."""

    track_id: TrackId
    features: AudioFeatures
    genres: tuple[str, ...] = ()


class WeightedTrack(Frozen):
    """A positive example with its evidence weight (e.g. 1 per running playlist, +1 per play during a run)."""

    profile: TrackProfile
    weight: float = Field(default=1.0, gt=0)


class StyleModel(Protocol):
    """Higher = more like what the user runs to. Scores are comparable within one fitted model only."""

    def score(self, track: TrackProfile) -> float: ...

    def score_many(self, tracks: Sequence[TrackProfile]) -> list[float]: ...


class StyleFitter(Protocol):
    """Builds a `StyleModel` from positives and unlabeled tracks (the PU-learning interface)."""

    def __call__(self, positives: Sequence[WeightedTrack], unlabeled: Sequence[TrackProfile]) -> StyleModel: ...


# ─────────────────────────────── features ───────────────────────────────
def normalise_genre(raw: str) -> str:
    return " ".join(raw.lower().split())


def feature_matrix(tracks: Sequence[TrackProfile]) -> FloatArray:
    """(n_tracks, 7) matrix of the raw style features, in `STYLE_FEATURES` order."""
    rows = [[float(getattr(track.features, name)) for name in STYLE_FEATURES] for track in tracks]
    return np.asarray(rows, dtype=np.float64).reshape(len(tracks), len(STYLE_FEATURES))


def genre_profile(positives: Sequence[WeightedTrack]) -> dict[str, float]:
    """Weighted share of the positives that carry each genre (a track's genres count once each)."""
    total = sum(item.weight for item in positives)
    weights: defaultdict[str, float] = defaultdict(float)
    for item in positives:
        for genre in {normalise_genre(g) for g in item.profile.genres if g.strip()}:
            weights[genre] += item.weight
    return {genre: weight / total for genre, weight in weights.items()} if total > 0 else {}


def top_profile_genres(positives: Sequence[WeightedTrack], n: int) -> list[str]:
    """The `n` most characteristic genres of the positives (ties broken alphabetically, for determinism)."""
    profile = genre_profile(positives)
    return [genre for genre, _ in sorted(profile.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def _cosine(a: FloatArray, b: FloatArray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / norm if norm > _EPS else 0.0


def _genre_cosine(genres: Sequence[str], profile: Mapping[str, float], profile_norm: float) -> float | None:
    """Cosine between the track's binary genre bag and the profile; None when either side is empty (unknown)."""
    bag = {normalise_genre(g) for g in genres if g.strip()}
    if not bag or profile_norm <= _EPS:
        return None
    return sum(profile.get(genre, 0.0) for genre in bag) / (math.sqrt(len(bag)) * profile_norm)


# ─────────────────────────────── the baseline model ───────────────────────────────
class CentroidStyleModel:
    """Cosine similarity to the positives' centroid in standardised feature space, blended with genre overlap."""

    def __init__(
        self,
        *,
        mean: FloatArray,
        std: FloatArray,
        centroid_z: FloatArray,
        genre_weights: Mapping[str, float],
        genre_blend: float = DEFAULT_GENRE_BLEND,
    ) -> None:
        if not 0.0 <= genre_blend <= 1.0:
            raise ValueError("genre_blend must be in [0, 1]")
        self._mean = mean
        self._std = std
        self._centroid_z = centroid_z
        self._genres = dict(genre_weights)
        self._genre_norm = math.sqrt(sum(w * w for w in self._genres.values()))
        self._genre_blend = genre_blend

    @classmethod
    def fit(
        cls,
        positives: Sequence[WeightedTrack],
        unlabeled: Sequence[TrackProfile],
        *,
        genre_blend: float = DEFAULT_GENRE_BLEND,
    ) -> CentroidStyleModel:
        if not positives:
            raise ValueError("the centroid model needs at least one positive track")
        reference = feature_matrix([p.profile for p in positives] + list(unlabeled))
        mean = reference.mean(axis=0)
        std = np.maximum(reference.std(axis=0), _MIN_STD)
        weights = np.asarray([p.weight for p in positives], dtype=np.float64)
        positives_z = (feature_matrix([p.profile for p in positives]) - mean) / std
        centroid_z = (weights[:, None] * positives_z).sum(axis=0) / weights.sum()
        return cls(
            mean=mean, std=std, centroid_z=centroid_z, genre_weights=genre_profile(positives), genre_blend=genre_blend
        )

    @property
    def centroid_raw(self) -> dict[str, float]:
        """The profile in raw units (e.g. energy 0.82, loudness −5.1 dB), for inspection and notebooks."""
        raw = self._centroid_z * self._std + self._mean
        return {name: float(value) for name, value in zip(STYLE_FEATURES, raw, strict=True)}

    def score(self, track: TrackProfile) -> float:
        return self.score_many([track])[0]

    def score_many(self, tracks: Sequence[TrackProfile]) -> list[float]:
        if not tracks:
            return []
        z = (feature_matrix(tracks) - self._mean) / self._std
        scores: list[float] = []
        for row, track in zip(z, tracks, strict=True):
            audio = _cosine(row, self._centroid_z)
            genre = _genre_cosine(track.genres, self._genres, self._genre_norm)
            scores.append(audio if genre is None else (1 - self._genre_blend) * audio + self._genre_blend * genre)
        return scores


class NeutralStyleModel:
    """Cold start (no running playlists, no plays during runs): every track scores 0, so tempo and mood decide."""

    def score(self, track: TrackProfile) -> float:
        return 0.0

    def score_many(self, tracks: Sequence[TrackProfile]) -> list[float]:
        return [0.0] * len(tracks)


def fit_style_model(positives: Sequence[WeightedTrack], unlabeled: Sequence[TrackProfile]) -> StyleModel:
    """The default `StyleFitter`: the centroid model, or the neutral one when there are no positives."""
    if not positives:
        return NeutralStyleModel()
    return CentroidStyleModel.fit(positives, unlabeled)


# ─────────────────────────────── offline evaluation support ───────────────────────────────
class HeldOutScores(Frozen):
    """Raw scores for a held-out evaluation: compute AUC, precision@k, ... on top (no metric is chosen here)."""

    positive_scores: tuple[float, ...] = Field(description="scores of held-out positives (hidden among unlabeled)")
    other_scores: tuple[float, ...] = Field(description="scores of library tracks that are not positives")
    n_train_positives: int = Field(ge=1)


def held_out_scores(
    fitter: StyleFitter,
    positives: Sequence[WeightedTrack],
    library: Sequence[TrackProfile],
    *,
    holdout_fraction: float = 0.2,
    rng: random.Random,
) -> HeldOutScores:
    """Hide a random share of the positives, fit on the rest, and score the hidden ones vs the other library tracks.

    The hidden positives stay in the unlabeled pool during fitting, exactly as unknown positives do in production
    (PU setting). Deterministic for a given `rng` state; repeat with several seeds for a variance estimate.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    unique = list({item.profile.track_id: item for item in positives}.values())
    if len(unique) < 2:
        raise ValueError("need at least 2 distinct positives to hold some out")
    order = sorted(unique, key=lambda item: item.profile.track_id)
    rng.shuffle(order)
    n_held = min(len(order) - 1, max(1, round(holdout_fraction * len(order))))
    held, train = order[:n_held], order[n_held:]
    train_ids = {item.profile.track_id for item in train}
    positive_ids = train_ids | {item.profile.track_id for item in held}
    library_by_id = {track.track_id: track for track in library}
    unlabeled = {track_id: track for track_id, track in library_by_id.items() if track_id not in train_ids}
    for item in held:  # hidden positives are unlabeled from the model's point of view
        unlabeled.setdefault(item.profile.track_id, item.profile)
    model = fitter(train, list(unlabeled.values()))
    others = [track for track_id, track in library_by_id.items() if track_id not in positive_ids]
    return HeldOutScores(
        positive_scores=tuple(model.score_many([item.profile for item in held])),
        other_scores=tuple(model.score_many(others)),
        n_train_positives=len(train),
    )
