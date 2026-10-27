"""Playlist quality on the real library (read-only): `uv run python scripts/playlist_quality.py [--show]`.

Drafts a few typical playlists with the live planner and scores them with metrics that do NOT depend on the planner's
own model, so versions can be compared before/after a change:

* spoken       picks that are spoken word (speechiness > 0.66, Spotify's own threshold) or longer than 15 min;
* running      share of picks from the running playlists (familiar favourites);
* run-artist   share of NEW picks (not in a running playlist) by an artist who is in one;
* ref style    median score of the picks under a fixed reference model (centroid of the running playlists vs the
               saved tracks); `off-style` = share of picks scoring below 0 (closer to "my music in general");
* fill         minutes drafted / minutes needed, and the tempo tolerance that was needed.

Nothing is created on Spotify. Default output is numbers only; `--show` also prints the picks (titles stay local).
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from datetime import timedelta

from stridedj.config import load_settings
from stridedj.domain import CadenceTarget, Mood, MusicChoice, RunType, TimeSlot, Track, TrackId
from stridedj.http import build_client
from stridedj.music.cache import JsonCache, default_cache_path
from stridedj.music.live import build_live_music
from stridedj.music.reccobeats import ReccoBeatsClient
from stridedj.music.spotify import SpotifyClient
from stridedj.music.style import CentroidStyleModel, TrackProfile, WeightedTrack
from stridedj.runs.store import SqliteRunRepository

SCENARIOS: tuple[tuple[RunType, float, float], ...] = (
    (RunType.Z2, 160.0, 75.0),
    (RunType.Z3, 165.0, 45.0),
    (RunType.INTERVALS, 180.0, 57.0),
)
SPOKEN_SPEECHINESS = 0.66
LONG_TRACK_MS = 15 * 60_000


async def main(show: bool, music: MusicChoice) -> None:
    settings = load_settings()
    repo = SqliteRunRepository.from_settings(settings.app)

    def run_windows() -> list[TimeSlot]:
        return [TimeSlot(start=a.start, end=a.start + timedelta(seconds=a.elapsed_s)) for a in repo.list_activities()]

    planner = build_live_music(settings, run_windows=run_windows).planner
    cache = JsonCache(default_cache_path(settings.app))
    http = build_client(timeout_s=20.0)
    spotify = SpotifyClient.from_settings(settings.spotify, http=http, cache=cache)
    reccobeats = ReccoBeatsClient(http=http, cache=cache)

    running: dict[TrackId, Track] = {}
    for playlist_id in settings.spotify.running_playlist_ids:
        running |= {t.track_id: t for t in await spotify.playlist_tracks(playlist_id)}
    saved = {t.track_id: t for t in await spotify.saved_tracks(limit=300)}
    running_artists = {artist for t in running.values() for artist in t.artist_ids}
    reference = await _reference_model(reccobeats, running, saved)

    print(f"running playlist tracks: {len(running)} · saved tracks: {len(saved)} · music: {music.value}\n")
    header = (
        f"{'scenario':16s} {'tracks':>6s} {'fill':>6s} {'±BPM':>5s} {'spoken':>6s} {'running':>7s} "
        f"{'run-artist':>10s} {'ref style':>9s} {'off-style':>9s} {'secs':>5s}"
    )
    print(header)
    print("-" * len(header))
    for run_type, spm, minutes in SCENARIOS:
        cadence = CadenceTarget(run_type=run_type, spm=spm, n_runs=8, lookback_days=120)
        started = time.perf_counter()
        draft = await planner.draft(
            run_type=run_type, cadence=cadence, duration_min=minutes, mood=Mood.USUAL, music=music
        )
        elapsed = time.perf_counter() - started
        tracks = list(await planner.tracks_for(draft))
        features = await reccobeats.audio_features([t.track_id for t in tracks])
        ids = [t.track_id for t in tracks]
        spoken = sum(
            t.duration_ms > LONG_TRACK_MS
            or (t.track_id in features and features[t.track_id].speechiness > SPOKEN_SPEECHINESS)
            for t in tracks
        )
        new = [t for t in tracks if t.track_id not in running]
        by_run_artist = sum(any(a in running_artists for a in t.artist_ids) for t in new)
        scores = reference.score_many([TrackProfile(track_id=i, features=features[i]) for i in ids if i in features])
        print(
            f"{run_type.value + f' {spm:.0f}/{minutes:.0f}':16s} {len(tracks):6d} "
            f"{draft.total_min / minutes:6.0%} {draft.tolerance_bpm:5.0f} {spoken:6d} "
            f"{(len(tracks) - len(new)) / max(1, len(tracks)):7.0%} {by_run_artist / max(1, len(new)):10.0%} "
            f"{statistics.median(scores) if scores else float('nan'):+9.2f} "
            f"{sum(s < 0 for s in scores) / max(1, len(scores)):9.0%} {elapsed:5.1f}"
        )
        if show:
            for track in tracks:
                f = features.get(track.track_id)
                tag = "run" if track.track_id in running else "new"
                bpm = f"{f.tempo_bpm:5.1f}" if f else "  ?  "
                print(f"      [{tag}] {bpm} BPM  {track.name[:42]} — {', '.join(track.artist_names)[:30]}")


async def _reference_model(
    reccobeats: ReccoBeatsClient, running: dict[TrackId, Track], saved: dict[TrackId, Track]
) -> CentroidStyleModel:
    """Fixed yardstick: running playlists = positives, saved tracks = unlabeled (spoken word excluded)."""
    features = await reccobeats.audio_features(list(running | saved))
    music = {i: f for i, f in features.items() if f.speechiness <= SPOKEN_SPEECHINESS}
    positives = [WeightedTrack(profile=TrackProfile(track_id=i, features=music[i])) for i in running if i in music]
    unlabeled = [TrackProfile(track_id=i, features=music[i]) for i in saved if i in music and i not in running]
    return CentroidStyleModel.fit(positives, unlabeled, genre_blend=0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--show", action="store_true", help="also print the picks (titles stay on this machine)")
    parser.add_argument("--music", type=MusicChoice, choices=list(MusicChoice), default=MusicChoice.USUAL)
    args = parser.parse_args()
    asyncio.run(main(args.show, args.music))
