"""Playlist assembly: tempo folding to the cadence target, style ranking, mood, novelty, artist diversity, greedy fill.

Pure and deterministic: no I/O, randomness only from the injected `random.Random` (tie-breaking jitter and the draft
id), so the same inputs and seed always give the same playlist (tests, evals, reproducible bug reports).

Tempo folding: runners lock their steps to the beat at 1×, but also at 2× (a 86 BPM hip-hop track "is" 172 spm,
stepping on every half-beat) and ½× (a 340 BPM reading is a tempo-detection octave error). So each track's tempo is
folded by k ∈ {½, 1, 2} to the value closest to the target cadence.

Tolerance ladder: ±3 BPM is barely noticeable while running, ±8 is still comfortable. We use the tightest level that
can fill the requested duration and record which one was used, so the agent can be honest ("matched within ±5 BPM").
"""

from __future__ import annotations

import random
import re
import unicodedata
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from pydantic import Field, model_validator

from stridedj.domain import (
    AudioFeatures,
    DraftId,
    Frozen,
    Mood,
    PlaylistDraft,
    PlaylistDraftSummary,
    RunType,
    Track,
)

TEMPO_FACTORS: tuple[float, ...] = (1.0, 0.5, 2.0)  # order = preference on an exact tie: unfolded first
DEFAULT_TOLERANCE_LADDER_BPM: tuple[float, ...] = (3.0, 5.0, 8.0)
MAX_TOP_GENRES = 5
MAX_GENRE_CHARS = 24
_GENRE_DISALLOWED = re.compile(r"[^a-z0-9 &-]")


# ─────────────────────────────── tempo ───────────────────────────────
def fold_tempo(bpm: float, target_spm: float) -> tuple[float, float]:
    """Fold `bpm` by ½, 1 or 2 to the value closest to `target_spm`; returns (folded_bpm, |folded − target|)."""
    if bpm <= 0 or target_spm <= 0:
        raise ValueError("bpm and target_spm must be > 0")
    folded = min((factor * bpm for factor in TEMPO_FACTORS), key=lambda value: abs(value - target_spm))
    return folded, abs(folded - target_spm)


# ─────────────────────────────── genres (the only text that reaches the LLM) ───────────────────────────────
def sanitise_genre(raw: str) -> str | None:
    """Lowercase ASCII `[a-z0-9 &-]`, single spaces, ≤ 24 chars; None if nothing is left.

    Genres come from Spotify's artist objects (set by Spotify, not by uploaders), so they are low-risk; this still
    bounds what can ever reach the model through a playlist summary to short, harmless tags.
    """
    ascii_only = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    cleaned = " ".join(_GENRE_DISALLOWED.sub("", ascii_only.lower()).split())
    cleaned = cleaned[:MAX_GENRE_CHARS].strip()
    return cleaned or None


def top_genres(genre_lists: Sequence[Sequence[str]], *, n: int = MAX_TOP_GENRES) -> tuple[str, ...]:
    """Most frequent sanitised genres (each track counts a genre once); ties broken alphabetically."""
    counts: Counter[str] = Counter()
    for genres in genre_lists:
        counts.update({clean for genre in genres if (clean := sanitise_genre(genre))})
    return tuple(genre for genre, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n])


def summarise(draft: PlaylistDraft) -> PlaylistDraftSummary:
    """The LLM-facing view of a draft: counts, minutes, tempo match and genre tags. No titles, no artist names."""
    return PlaylistDraftSummary(
        draft_id=draft.draft_id,
        n_tracks=len(draft.track_ids),
        total_min=round(draft.total_min, 1),
        target_spm=round(draft.target_spm, 1),
        tolerance_bpm=draft.tolerance_bpm,
        mean_abs_bpm_error=round(draft.mean_abs_bpm_error, 2),
        top_genres=tuple(clean for genre in draft.top_genres if (clean := sanitise_genre(genre))),
    )


# ─────────────────────────────── inputs ───────────────────────────────
class Candidate(Frozen):
    """A track that may go into the playlist, with everything the assembler needs to rank it."""

    track: Track
    features: AudioFeatures
    genres: tuple[str, ...] = ()
    style_score: float
    in_running_playlists: bool = Field(description="False = novel: counts towards the novelty quota")


class MoodReference(Frozen):
    """The user's usual running music on the mood axes (weighted mean of the positives)."""

    energy: float = Field(ge=0, le=1)
    valence: float = Field(ge=0, le=1)


class AssemblyParams(Frozen):
    tolerance_ladder_bpm: tuple[float, ...] = DEFAULT_TOLERANCE_LADDER_BPM
    bpm_error_weight: float = Field(default=0.5, ge=0, description="λ: penalty for a full-tolerance tempo error")
    mood_weight: float = Field(default=1.0, ge=0, description="strength of the MORE_ENERGETIC / CALMER shift")
    novelty_target: float = Field(default=0.3, ge=0, le=1, description="minimum share of novel tracks, when possible")
    max_tracks_per_artist: int = Field(default=2, ge=1)
    jitter: float = Field(default=0.02, ge=0, description="random tie-breaking amplitude: variety across drafts")

    @model_validator(mode="after")
    def _ladder(self) -> Self:
        ladder = self.tolerance_ladder_bpm
        if not ladder or any(t <= 0 for t in ladder) or list(ladder) != sorted(set(ladder)):
            raise ValueError("tolerance_ladder_bpm must be strictly increasing positive values")
        return self


class AssembledPlaylist(Frozen):
    """The draft (safe for the LLM after `summarise`) and its tracks (for code-rendered display only)."""

    draft: PlaylistDraft
    tracks: tuple[Track, ...]


# ─────────────────────────────── ranking and greedy fill ───────────────────────────────
@dataclass(frozen=True, slots=True)
class _Scored:
    candidate: Candidate
    folded_bpm: float
    bpm_error: float
    base_rank: float  # style + mood + jitter; the tempo penalty is added per ladder level

    @property
    def novel(self) -> bool:
        return not self.candidate.in_running_playlists


@dataclass(frozen=True, slots=True)
class _Ranked:
    scored: _Scored
    rank: float


def mood_term(features: AudioFeatures, mood: Mood, reference: MoodReference, weight: float) -> float:
    """MORE_ENERGETIC rewards energy/valence above the user's usual, CALMER below; USUAL adds nothing."""
    direction = {Mood.MORE_ENERGETIC: 1.0, Mood.CALMER: -1.0, Mood.USUAL: 0.0}[mood]
    shift = ((features.energy - reference.energy) + (features.valence - reference.valence)) / 2
    return direction * weight * shift


def _song_key(track: Track) -> tuple[str, str]:
    """Same title by the same lead artist (single vs album version) counts as one song."""
    lead = track.artist_ids[0] if track.artist_ids else ""
    return " ".join(track.name.lower().split()), lead


class _GreedyFill:
    """Pick tracks best-first until the duration is covered, under the artist cap and the novelty quota.

    Novelty rule: before each pick, if taking a familiar track would leave the novel share below the target, the best
    remaining novel track is taken instead (when one exists). This keeps the share ≥ target at every prefix, so it
    holds whatever the final length turns out to be.
    """

    def __init__(self, ranked: Sequence[_Ranked], params: AssemblyParams) -> None:
        self._pools: dict[bool, list[_Ranked]] = {
            True: [r for r in ranked if r.scored.novel],
            False: [r for r in ranked if not r.scored.novel],
        }
        self._cursor: dict[bool, int] = {True: 0, False: 0}
        self._params = params
        self._artists: Counter[str] = Counter()
        self._songs: set[tuple[str, str]] = set()
        self.chosen: list[_Ranked] = []
        self.total_ms = 0
        self.n_novel = 0

    def run(self, duration_ms: int) -> Self:
        while self.total_ms < duration_ms:
            pick = self._next()
            if pick is None:
                break
            self._take(pick)
        return self

    def _next(self) -> _Ranked | None:
        novel, familiar = self._head(novel=True), self._head(novel=False)
        need_novel = self.n_novel < self._params.novelty_target * (len(self.chosen) + 1)
        if novel is None or familiar is None:
            return novel or familiar
        if need_novel:
            return novel
        return novel if novel.rank > familiar.rank else familiar

    def _head(self, *, novel: bool) -> _Ranked | None:
        """The best remaining eligible track of one pool; ineligible ones are skipped for good (counts only grow)."""
        pool, index = self._pools[novel], self._cursor[novel]
        while index < len(pool) and not self._eligible(pool[index].scored.candidate.track):
            index += 1
        self._cursor[novel] = index
        return pool[index] if index < len(pool) else None

    def _eligible(self, track: Track) -> bool:
        if _song_key(track) in self._songs:
            return False
        return all(self._artists[artist] < self._params.max_tracks_per_artist for artist in track.artist_ids)

    def _take(self, ranked: _Ranked) -> None:
        track = ranked.scored.candidate.track
        self.chosen.append(ranked)
        self._cursor[ranked.scored.novel] += 1
        self.total_ms += track.duration_ms
        self.n_novel += ranked.scored.novel
        self._songs.add(_song_key(track))
        self._artists.update(set(track.artist_ids))


def _score_candidates(
    candidates: Sequence[Candidate],
    *,
    target_spm: float,
    mood: Mood,
    mood_reference: MoodReference,
    params: AssemblyParams,
    rng: random.Random,
) -> list[_Scored]:
    """Fold tempos and compute the level-independent part of the rank. Candidates are de-duplicated by id and
    processed in id order, so the jitter each one gets does not depend on the input order."""
    unique = sorted({c.track.track_id: c for c in candidates}.values(), key=lambda c: c.track.track_id)
    scored: list[_Scored] = []
    for candidate in unique:
        folded_bpm, bpm_error = fold_tempo(candidate.features.tempo_bpm, target_spm)
        base = (
            candidate.style_score
            + mood_term(candidate.features, mood, mood_reference, params.mood_weight)
            + rng.uniform(-params.jitter, params.jitter)
        )
        scored.append(_Scored(candidate=candidate, folded_bpm=folded_bpm, bpm_error=bpm_error, base_rank=base))
    return scored


def _rank_level(scored: Sequence[_Scored], tolerance_bpm: float, params: AssemblyParams) -> list[_Ranked]:
    """Tracks within the tolerance, best first: rank = base − λ·(bpm error / tolerance); ties broken by id."""
    ranked = [
        _Ranked(scored=s, rank=s.base_rank - params.bpm_error_weight * s.bpm_error / tolerance_bpm)
        for s in scored
        if s.bpm_error <= tolerance_bpm
    ]
    return sorted(ranked, key=lambda r: (-r.rank, r.scored.candidate.track.track_id))


def assemble_playlist(
    candidates: Sequence[Candidate],
    *,
    run_type: RunType,
    target_spm: float,
    duration_min: float,
    mood: Mood,
    mood_reference: MoodReference,
    rng: random.Random,
    params: AssemblyParams | None = None,
) -> AssembledPlaylist:
    """Plan a playlist of at least `duration_min` minutes whose folded tempos match `target_spm`.

    Walks the tolerance ladder and stops at the first level whose greedy fill covers the duration. If even the
    widest level cannot, the widest level's (shorter) playlist is returned: `total_min < duration_min` then tells the
    caller honestly that the library is too small for this cadence. The final order is best-ranked first.
    """
    if duration_min <= 0:
        raise ValueError("duration_min must be > 0")
    params = params or AssemblyParams()
    draft_id = DraftId(uuid.UUID(int=rng.getrandbits(128), version=4).hex[:12])  # uuid4-shaped, seeded
    scored = _score_candidates(
        candidates, target_spm=target_spm, mood=mood, mood_reference=mood_reference, params=params, rng=rng
    )
    duration_ms = round(duration_min * 60_000)
    tolerance_bpm, fill = params.tolerance_ladder_bpm[0], _GreedyFill([], params)
    for tolerance_bpm in params.tolerance_ladder_bpm:
        fill = _GreedyFill(_rank_level(scored, tolerance_bpm, params), params).run(duration_ms)
        if fill.total_ms >= duration_ms:
            break
    chosen = sorted(fill.chosen, key=lambda r: (-r.rank, r.scored.candidate.track.track_id))
    picked = [r.scored for r in chosen]
    draft = PlaylistDraft(
        draft_id=draft_id,
        run_type=run_type,
        target_spm=target_spm,
        tolerance_bpm=tolerance_bpm,
        track_ids=tuple(s.candidate.track.track_id for s in picked),
        total_min=fill.total_ms / 60_000,
        mean_abs_bpm_error=sum(s.bpm_error for s in picked) / len(picked) if picked else 0.0,
        novelty_share=fill.n_novel / len(picked) if picked else 0.0,
        top_genres=top_genres([s.candidate.genres for s in picked]),
    )
    return AssembledPlaylist(draft=draft, tracks=tuple(s.candidate.track for s in picked))
