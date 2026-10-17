"""Read-only smoke test of every live integration: `uv run python scripts/smoke_live.py`.

Prints counts and coverage only (no names, titles or values) and writes nothing anywhere. Use it before a demo:
it answers "do Garmin, Spotify, ReccoBeats and Google actually return what the agent needs, today?".
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from stridedj.cal.gcal import GoogleCalendarClient
from stridedj.config import load_settings
from stridedj.domain import TimeSlot
from stridedj.errors import StrideDJError
from stridedj.http import build_client
from stridedj.music.reccobeats import ReccoBeatsClient
from stridedj.music.spotify import SpotifyClient
from stridedj.runs.sources.garmin_live import GarminLiveSource


def line(ok: bool, name: str, detail: str) -> None:
    print(f"{'✓' if ok else '✗'} {name:12s} {detail}")


async def main() -> None:
    settings = load_settings()
    tz = ZoneInfo(settings.athlete.timezone)
    now = datetime.now(UTC)

    try:
        runs = await GarminLiveSource(settings.garmin).fetch_activities(since=now - timedelta(days=120), limit=200)
        with_hr = sum(r.avg_hr_bpm is not None for r in runs)
        with_cad = sum(r.avg_cadence_spm is not None for r in runs)
        with_zones = sum(r.hr_zone_s is not None for r in runs)
        with_laps = sum(bool(r.laps) for r in runs)
        km = sum(r.distance_km for r in runs)
        line(
            bool(runs),
            "garmin",
            f"{len(runs)} runs in 120 days ({km:.0f} km); with HR {with_hr}, cadence {with_cad}, "
            f"zone times {with_zones}, laps {with_laps}",
        )
    except StrideDJError as exc:
        line(False, "garmin", str(exc))

    track_ids = []
    async with build_client(timeout_s=20.0) as http:
        try:
            spotify = SpotifyClient.from_settings(settings.spotify, http=http)
            for pid in settings.spotify.running_playlist_ids:
                tracks = await spotify.playlist_tracks(pid)
                track_ids += [t.track_id for t in tracks]
                line(bool(tracks), "spotify", f"running playlist: {len(tracks)} tracks")
            plays = await spotify.recently_played()
            line(True, "spotify", f"recently played: {len(plays)} plays")
            if tracks:
                genres = await spotify.artist_genres(tracks[0].artist_ids[0])
                line(True, "spotify", f"artist genres available: {len(genres)} tag(s) for the first artist")
        except StrideDJError as exc:
            line(False, "spotify", str(exc))

        try:
            features = await ReccoBeatsClient(http=http).audio_features(track_ids[:60])
            share = len(features) / max(1, len(track_ids[:60]))
            line(
                bool(features), "reccobeats", f"tempo/features for {len(features)}/{len(track_ids[:60])} ({share:.0%})"
            )
        except StrideDJError as exc:
            line(False, "reccobeats", str(exc))

    try:
        tomorrow = (now.astimezone(tz) + timedelta(days=1)).date()
        window = TimeSlot(
            start=datetime.combine(tomorrow, time(6), tzinfo=tz), end=datetime.combine(tomorrow, time(22), tzinfo=tz)
        )
        async with GoogleCalendarClient(settings.google, tz=tz) as calendar:
            busy = await calendar.busy_intervals(window)
        line(True, "google", f"free/busy works: {len(busy)} busy block(s) tomorrow 06:00-22:00")
    except StrideDJError as exc:
        line(False, "google", str(exc))


if __name__ == "__main__":
    asyncio.run(main())
