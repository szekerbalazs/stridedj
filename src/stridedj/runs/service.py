"""`RunInsightsService`: the application service behind the agent's run tools (implements `ports.RunInsights`).

It composes the pieces and adds nothing clever of its own: a `RunSource` (where runs come from), a `RunRepository`
(the local store), the user's heart-rate zones (to type runs) and the pure DS functions in `analysis` / `predict`.
Every answer is computed from the *store*, never from the live API, so the tools stay fast and work offline; only
`refresh()` talks to the source.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from stridedj.config import AthleteSettings
from stridedj.domain import (
    CadenceTarget,
    ClassifiedRun,
    DurationEstimate,
    HeartRateZones,
    RacePrediction,
    RunningProfile,
    RunType,
)
from stridedj.errors import ToolError
from stridedj.ports import Clock, RunRepository, RunSource
from stridedj.runs import analysis
from stridedj.runs.analysis import DEFAULT_HR_ZONES
from stridedj.runs.predict import DurationPredictor, PredictorConfig

if TYPE_CHECKING:
    from stridedj.ports import RunInsights
    from stridedj.runs.sources.demo import DemoRunSource
    from stridedj.runs.sources.garmin_live import GarminLiveSource
    from stridedj.runs.sources.intervals_icu import IntervalsIcuSource
    from stridedj.runs.store import SqliteRunRepository

INITIAL_HISTORY_DAYS = 400
FETCH_LIMIT = 1000
MAX_DISTANCE_KM = 100.0


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


def zones_from_settings(athlete: AthleteSettings) -> HeartRateZones:
    """The athlete's watch zones; the defaults (125, 145, 160, 175) bpm when none are configured.

    The defaults fit a runner with a max HR around 190. With a very different max HR the typing of runs (and so
    every per-type statistic) will be off; `stridedj doctor` should nudge the user to set ATHLETE_HR_ZONE_UPPER_BOUNDS.
    """
    if athlete.hr_zone_upper_bounds is None:
        return DEFAULT_HR_ZONES
    return HeartRateZones(upper_bounds_bpm=athlete.hr_zone_upper_bounds)


class RunInsightsService:
    """Everything the agent can learn about the user's running, as aggregates.

    Wiring (composition root): `RunInsightsService(source, SqliteRunRepository.from_settings(settings.app),
    zones=zones_from_settings(settings.athlete))`.
    """

    def __init__(
        self,
        source: RunSource,
        repository: RunRepository,
        *,
        zones: HeartRateZones = DEFAULT_HR_ZONES,
        config: PredictorConfig | None = None,
        clock: Clock | None = None,
        initial_history_days: int = INITIAL_HISTORY_DAYS,
        profile_lookback_days: int = 120,
        cadence_lookback_days: int = 120,
        cadence_last_n: int = 8,
    ) -> None:
        self._source = source
        self._repository = repository
        self._zones = zones
        self._config = config or PredictorConfig()
        self._clock: Clock = clock or SystemClock()
        self._initial_history_days = initial_history_days
        self._profile_lookback_days = profile_lookback_days
        self._cadence_lookback_days = cadence_lookback_days
        self._cadence_last_n = cadence_last_n

    # ── ingestion ──
    async def refresh(self) -> int:
        """Pull activities since the newest stored one (or the last 400 days) into the store; returns how many are new.

        Re-fetching from the newest stored start (inclusive) re-delivers that run; the upsert makes it a no-op, and it
        also picks up a run that was still syncing (e.g. laps missing) on the previous refresh.
        """
        stored = self._repository.list_activities()
        since = stored[0].start if stored else self._clock.now() - timedelta(days=self._initial_history_days)
        fetched = await self._source.fetch_activities(since=since, limit=FETCH_LIMIT)
        known = {activity.activity_id for activity in stored}
        new = sum(activity.activity_id not in known for activity in fetched)
        self._repository.upsert_activities(fetched)
        return new

    # ── queries (all computed from the store) ──
    def classified_runs(self, *, now: datetime) -> list[ClassifiedRun]:
        """Stored runs within the model's history horizon that started before `now`, typed with the user's zones."""
        since = now - timedelta(days=self._config.history_days)
        activities = [a for a in self._repository.list_activities(since=since) if a.start < now]
        return analysis.classify_all(activities, self._zones)

    def profile(self, run_type: RunType, *, now: datetime) -> RunningProfile:
        return analysis.running_profile(
            self.classified_runs(now=now), run_type, now=now, lookback_days=self._profile_lookback_days
        )

    def estimate_duration(self, run_type: RunType, distance_km: float, *, now: datetime) -> DurationEstimate:
        if not 0.0 < distance_km <= MAX_DISTANCE_KM:
            raise ToolError(f"distance_km must be greater than 0 and at most {MAX_DISTANCE_KM:.0f}; ask the user")
        return DurationPredictor(self.classified_runs(now=now), now=now, config=self._config).estimate(
            run_type, distance_km
        )

    def cadence_target(self, run_type: RunType, *, now: datetime) -> CadenceTarget:
        return analysis.cadence_target(
            self.classified_runs(now=now),
            run_type,
            now=now,
            n_last=self._cadence_last_n,
            lookback_days=self._cadence_lookback_days,
        )

    def race_predictions(self, *, now: datetime) -> Sequence[RacePrediction]:
        return DurationPredictor(self.classified_runs(now=now), now=now, config=self._config).race_predictions()


if TYPE_CHECKING:  # static proof that the adapters satisfy the ports (checked by mypy, never executed)

    def _insights(service: RunInsightsService) -> RunInsights:
        return service

    def _sources(demo: DemoRunSource, garmin: GarminLiveSource, icu: IntervalsIcuSource) -> list[RunSource]:
        return [demo, garmin, icu]

    def _repository(store: SqliteRunRepository) -> RunRepository:
        return store
