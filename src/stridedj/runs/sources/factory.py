"""Pick the `RunSource` named by `STRIDEDJ_RUN_SOURCE` (demo | garmin | intervals_icu).

This is the single switch that makes the Garmin → intervals.icu fallback a config change: nothing downstream of the
`RunSource` port knows which adapter is in use.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from stridedj.config import Settings
from stridedj.ports import RunSource
from stridedj.runs.sources.demo import DemoRunSource


def _system_now() -> datetime:
    return datetime.now(UTC)


def build_run_source(settings: Settings, *, now: Callable[[], datetime] = _system_now) -> RunSource:
    """The configured source. Adapters are imported lazily so the demo never loads `garminconnect`.

    The demo source is generated up to the current time here (in the live app the history should feel current); tests
    and evals construct `DemoRunSource()` directly, which is pinned to 2026-10-17.
    """
    kind = settings.app.run_source
    if kind == "garmin":
        from stridedj.runs.sources.garmin_live import GarminLiveSource

        return GarminLiveSource(settings.garmin)
    if kind == "intervals_icu":
        from stridedj.runs.sources.intervals_icu import IntervalsIcuSource

        return IntervalsIcuSource(settings.intervals_icu, now=now, local_tz=ZoneInfo(settings.athlete.timezone))
    return DemoRunSource(now=now())
