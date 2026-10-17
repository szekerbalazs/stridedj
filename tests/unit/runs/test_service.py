from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from stridedj.config import AthleteSettings
from stridedj.domain import Activity, EstimateMethod, HeartRateZones, RunType
from stridedj.errors import ToolError
from stridedj.runs.analysis import DEFAULT_HR_ZONES
from stridedj.runs.service import RunInsightsService, zones_from_settings
from stridedj.runs.sources.demo import DEMO_NOW, DemoRunSource
from stridedj.runs.store import SqliteRunRepository


class FixedClock:
    def __init__(self, now: datetime) -> None:
        self.current = now

    def now(self) -> datetime:
        return self.current


class SwitchableSource:
    """Delegates to a demo source whose `now` the test moves forward; records the `since` it was asked for."""

    def __init__(self, now: datetime) -> None:
        self.inner = DemoRunSource(now=now)
        self.since_calls: list[datetime] = []

    @property
    def name(self) -> str:
        return "demo"

    async def fetch_activities(self, *, since: datetime, limit: int) -> Sequence[Activity]:
        self.since_calls.append(since)
        return await self.inner.fetch_activities(since=since, limit=limit)


@pytest.fixture
def service(tmp_path: Path) -> RunInsightsService:
    return RunInsightsService(DemoRunSource(), SqliteRunRepository(tmp_path / "runs.db"), clock=FixedClock(DEMO_NOW))


async def test_refresh_imports_history_then_only_new_runs(tmp_path: Path) -> None:
    source = SwitchableSource(now=datetime(2026, 9, 1, tzinfo=UTC))
    repository = SqliteRunRepository(tmp_path / "runs.db")
    clock = FixedClock(datetime(2026, 9, 1, tzinfo=UTC))
    service = RunInsightsService(source, repository, clock=clock)

    first = await service.refresh()
    assert first == repository.count() > 100
    assert source.since_calls[0] == clock.now() - timedelta(days=400)

    assert await service.refresh() == 0  # nothing new: the re-fetched newest run is an idempotent upsert
    newest = repository.list_activities()[0].start
    assert source.since_calls[1] == newest

    source.inner = DemoRunSource(now=DEMO_NOW)  # six weeks later
    clock.current = DEMO_NOW
    added = await service.refresh()
    assert added > 10
    assert repository.count() == first + added


async def test_queries_answer_from_the_store(service: RunInsightsService) -> None:
    assert await service.refresh() > 100

    estimate = service.estimate_duration(RunType.Z2, 21.1, now=DEMO_NOW)
    assert estimate.method is EstimateMethod.REGRESSION
    assert service.estimate_duration(RunType.Z2, 10.0, now=DEMO_NOW).method is EstimateMethod.RECENT_SIMILAR

    profile = service.profile(RunType.Z2, now=DEMO_NOW)
    assert profile.n_runs > 20
    cadence = service.cadence_target(RunType.INTERVALS, now=DEMO_NOW)
    assert cadence.n_runs == 8
    assert cadence.spm > 180
    races = service.race_predictions(now=DEMO_NOW)
    assert [r.distance_km for r in races] == [5.0, 10.0, 21.0975, 42.195]


async def test_queries_never_see_runs_after_now(service: RunInsightsService) -> None:
    await service.refresh()
    early = datetime(2026, 1, 20, tzinfo=UTC)
    runs = service.classified_runs(now=early)
    assert runs
    assert all(r.activity.start < early for r in runs)
    assert service.estimate_duration(RunType.Z2, 10.0, now=datetime(2026, 1, 1, tzinfo=UTC)).method is (
        EstimateMethod.DEFAULT
    )


def test_empty_store_gives_defaults(service: RunInsightsService) -> None:
    assert service.estimate_duration(RunType.Z3, 8.0, now=DEMO_NOW).method is EstimateMethod.DEFAULT
    assert service.cadence_target(RunType.Z3, now=DEMO_NOW).n_runs == 0
    assert service.profile(RunType.Z3, now=DEMO_NOW).n_runs == 0


@pytest.mark.parametrize("km", [0.0, -5.0, 101.0])
def test_invalid_distance_is_an_actionable_tool_error(service: RunInsightsService, km: float) -> None:
    with pytest.raises(ToolError, match="distance_km"):
        service.estimate_duration(RunType.Z2, km, now=DEMO_NOW)


def test_zones_from_settings() -> None:
    assert zones_from_settings(AthleteSettings()) == DEFAULT_HR_ZONES
    configured = zones_from_settings(AthleteSettings(hr_zone_upper_bounds=(130, 150, 165, 178)))
    assert configured == HeartRateZones(upper_bounds_bpm=(130, 150, 165, 178))
