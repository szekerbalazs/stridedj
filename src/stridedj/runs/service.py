"""`RunInsightsService`: the application service behind the agent's run tools (implements `ports.RunInsights`).

It composes the pieces and adds nothing clever of its own: a `RunSource` (where runs come from), a `RunRepository`
(the local store), the user's heart-rate zones (to type runs) and the pure DS functions in `analysis` / `predict`.
Every answer is computed from the *store*, never from the live API, so the tools stay fast and work offline; only
`refresh()` talks to the source.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from stridedj.config import AthleteSettings
from stridedj.domain import (
    ActivityId,
    CadenceTarget,
    ClassifiedRun,
    DurationEstimate,
    HeartRateZones,
    IntervalSession,
    RacePrediction,
    RunningProfile,
    RunType,
)
from stridedj.errors import StrideDJError, ToolError
from stridedj.ports import Clock, RunRepository, RunSeriesSource, RunSource
from stridedj.runs import analysis
from stridedj.runs.analysis import DEFAULT_HR_ZONES
from stridedj.runs.intervals import IntervalAnalysis, analyse_series
from stridedj.runs.predict import DurationPredictor, PredictorConfig
from stridedj.runs.sessions import build_session_options, rest_after, typical_spread
from stridedj.runs.targets import (
    SessionVerdict,
    adapt_targets,
    pace_for,
    reps_by_distance,
    round_to_5,
    targets_from_history,
)

if TYPE_CHECKING:
    from stridedj.ports import RunInsights
    from stridedj.runs.sources.demo import DemoRunSource
    from stridedj.runs.sources.garmin_live import GarminLiveSource
    from stridedj.runs.sources.intervals_icu import IntervalsIcuSource
    from stridedj.runs.store import SqliteRunRepository

INITIAL_HISTORY_DAYS = 400
FETCH_LIMIT = 1000
MAX_DISTANCE_KM = 100.0
INTERVAL_CADENCE_LOOKBACK_DAYS = 365
SERIES_PER_REFRESH = 10  # Garmin calls per refresh for per-sample series (one per new run)
FALLBACK_EASY_PACE_S = 360.0
FALLBACK_TARGET_RATIOS = {400: 0.60, 800: 0.65}  # of the easy pace, without any interval history
DEFAULT_OVERHEAD_S = 5 * 60.0  # extra time in a session beyond running and rests (drills, getting to the track)
MAX_OVERHEAD_S = 15 * 60.0
MIN_REPS_PER_GROUP = 3
STANDING_MAX_M = 100.0  # a recovery covering less than this was a standing rest

logger = logging.getLogger(__name__)


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


def zones_from_settings(athlete: AthleteSettings) -> HeartRateZones:
    """The athlete's watch zones; the defaults (125, 145, 160, 175) bpm when none are configured.

    The defaults fit a runner with a max HR around 190. With a very different max HR the typing of runs (and so
    every per-type statistic) will be off. With Garmin as the source, the zones are read from Garmin Connect instead
    (`app.resolve_zones`); this function is the fallback for other sources.
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
        series_source: RunSeriesSource | None = None,
        series_per_refresh: int = SERIES_PER_REFRESH,
        interval_targets: Mapping[int, float] | None = None,
        interval_targets_since: datetime | None = None,
    ) -> None:
        self._source = source
        self._series_source = series_source
        self._series_per_refresh = series_per_refresh
        self._intervals: dict[ActivityId, IntervalAnalysis | None] = {}  # series never change: analyse once
        self._initial_targets = dict(interval_targets or {})
        self._targets_since = interval_targets_since
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
        await self._fetch_missing_series()
        return new

    async def _fetch_missing_series(self) -> None:
        """Per-sample series for runs that have none yet, newest first (best effort, a few per refresh: the bulk of
        an old history is loaded once with `stridedj series`)."""
        if self._series_source is None:
            return
        have = self._repository.series_ids()
        todo = [a.activity_id for a in self._repository.list_activities() if a.activity_id not in have]
        for activity_id in todo[: self._series_per_refresh]:
            try:
                series = await self._series_source.fetch_series(activity_id)
            except StrideDJError as exc:
                logger.warning("run series unavailable, continuing without: %s", exc)
                return
            if series is not None:
                self._repository.upsert_series(series)
                self._intervals.pop(activity_id, None)

    # ── queries (all computed from the store) ──
    def classified_runs(self, *, now: datetime) -> list[ClassifiedRun]:
        """Stored runs within the model's history horizon that started before `now`, typed with the user's zones."""
        since = now - timedelta(days=self._config.history_days)
        activities = [a for a in self._repository.list_activities(since=since) if a.start < now]
        found = {a.activity_id: result for a in activities if (result := self._interval_analysis(a.activity_id))}
        return analysis.classify_all(activities, self._zones, found)

    def _interval_analysis(self, activity_id: ActivityId) -> IntervalAnalysis | None:
        if activity_id not in self._intervals:
            series = self._repository.get_series(activity_id)
            self._intervals[activity_id] = analyse_series(series) if series is not None else None
        return self._intervals[activity_id]

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
        lookback = self._cadence_lookback_days
        if run_type is RunType.INTERVALS:  # sessions come in seasonal blocks: months can pass between two
            lookback = max(lookback, INTERVAL_CADENCE_LOOKBACK_DAYS)
        return analysis.cadence_target(
            self.classified_runs(now=now), run_type, now=now, n_last=self._cadence_last_n, lookback_days=lookback
        )

    # ── interval sessions ──
    def interval_sessions(self, *, now: datetime) -> list[tuple[datetime, IntervalAnalysis]]:
        """Detected interval sessions (start, analysis), oldest first."""
        found = [
            (run.activity.start, result)
            for run in self.classified_runs(now=now)
            if run.run_type is RunType.INTERVALS
            and (result := self._interval_analysis(run.activity.activity_id)) is not None
            and result.is_intervals
        ]
        return sorted(found, key=lambda pair: pair[0])

    def interval_targets(self, *, now: datetime) -> tuple[dict[int, float], list[SessionVerdict], str]:
        """Target pace per rep distance, the verdicts of the sessions since the targets were set, and their source."""
        sessions = self.interval_sessions(now=now)
        if self._initial_targets and self._targets_since is not None:
            targets, verdicts = adapt_targets(
                self._initial_targets, since=self._targets_since, sessions=sessions, now=now
            )
            return targets, verdicts, f"your targets of {self._targets_since:%d %b %Y}, adapted by later sessions"
        if self._initial_targets:
            return dict(self._initial_targets), [], "your targets"
        targets = targets_from_history(sessions, now=now)
        if targets:
            return targets, [], "your last sessions per distance (median rep pace, rounded to 5 s)"
        easy = self.profile(RunType.Z2, now=now).median_pace_s_per_km or FALLBACK_EASY_PACE_S
        fallback = {d: round_to_5(easy * ratio) for d, ratio in FALLBACK_TARGET_RATIOS.items()}
        return fallback, [], "no interval history yet: estimated from your easy pace"

    def interval_options(self, *, now: datetime, max_minutes: int | None = None) -> list[IntervalSession]:
        """2-3 interval sessions (short reps, medium reps, a pyramid) at the runner's target paces."""
        targets, _, _ = self.interval_targets(now=now)
        sessions = self.interval_sessions(now=now)
        easy = self.profile(RunType.Z2, now=now).median_pace_s_per_km or FALLBACK_EASY_PACE_S
        runs = self.classified_runs(now=now)
        model = DurationPredictor(runs, now=now, config=self._config).model
        activity_at = {run.activity.start: run.activity for run in runs}
        typical: dict[int, list[int]] = {}
        rests: dict[int, list[float]] = {}
        overheads: list[float] = []
        for start, session in sessions:
            for distance, paces in reps_by_distance(session).items():
                if len(paces) >= MIN_REPS_PER_GROUP:
                    typical.setdefault(distance, []).append(len(paces))
            for rep in session.reps:
                if rep.recovery_s is not None and (rep.recovery_m or 0.0) <= STANDING_MAX_M:
                    rests.setdefault(rep.nominal_m, []).append(rep.recovery_s)
            activity = activity_at[start]
            reps_s = sum(r.duration_s + (r.recovery_s or 0.0) for r in session.reps)
            moved_m = sum(r.distance_m + (r.recovery_m or 0.0) for r in session.reps)
            easy_s = (activity.distance_m - moved_m) / 1000 * easy
            overheads.append(min(MAX_OVERHEAD_S, max(0.0, activity.elapsed_s - reps_s - easy_s)))

        def measured_rest(rep_m: int) -> float:
            """The rest actually taken after reps of this distance (standing ones), else the planned rest."""
            closest = min(rests, key=lambda d: abs(d - rep_m), default=None)
            if closest is None or abs(closest - rep_m) > rep_m / 2:
                return float(rest_after(rep_m))
            return statistics.median(rests[closest])

        return build_session_options(
            pace_of=lambda distance: pace_for(distance, targets),
            easy_pace_s_per_km=easy,
            overhead_s=statistics.median(overheads) if overheads else DEFAULT_OVERHEAD_S,
            spread=typical_spread(model.log_residual_quantile if model is not None else None),
            buffer_min=self._config.buffer_min,
            n_history=len(sessions),
            measured_rest_s=measured_rest,
            typical_reps={d: round(statistics.median(n)) for d, n in typical.items() if d in (400, 800)},
            max_minutes=max_minutes,
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
