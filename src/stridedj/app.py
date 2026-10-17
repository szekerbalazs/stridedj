"""Composition root: the only module that knows which concrete adapters exist.

`build_demo_app` wires the agent to in-memory fakes (synthetic runs, a fake calendar, a fake music library); it's
what `--demo`, the tests and every eval trial use. `build_live_app` wires the same agent to Garmin, Google and
Spotify. The agent, the harness and the tools are identical in both, which is the point of ports and adapters.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from stridedj.agent.service import StrideAgent
from stridedj.agent.tools import AgentDeps
from stridedj.cal.demo import InMemoryCalendar
from stridedj.cal.service import SlotFinderService
from stridedj.clock import FixedClock, SystemClock
from stridedj.config import Settings
from stridedj.domain import TimeSlot
from stridedj.harness.approvals import ApprovalStore
from stridedj.harness.executor import PlanExecutor
from stridedj.harness.loop import Budget
from stridedj.llm.base import LLMClient
from stridedj.music.demo import RecordingPlaylistWriter, build_demo_planner
from stridedj.ports import CalendarPort, Clock, PlaylistPlanner, PlaylistWriterPort, RunInsights
from stridedj.runs.service import RunInsightsService, zones_from_settings
from stridedj.runs.sources.demo import DemoRunSource
from stridedj.runs.store import SqliteRunRepository

DEFAULT_TZ = "Europe/Zurich"
CalendarKind = Literal["realistic", "empty", "packed_evening"]


@dataclass
class App:
    """A wired agent plus handles on its adapters."""

    agent: StrideAgent
    insights: RunInsights
    calendar: CalendarPort
    playlists: PlaylistPlanner
    writer: PlaylistWriterPort
    clock: Clock
    timezone: str


@dataclass
class DemoApp(App):
    """The offline app; exposes the fakes so evals and tests can inspect the outcome (what was written)."""

    fake_calendar: InMemoryCalendar
    fake_writer: RecordingPlaylistWriter


async def build_demo_app(
    llm: LLMClient,
    *,
    now: datetime,
    timezone: str = DEFAULT_TZ,
    calendar: CalendarKind = "realistic",
    poisoned_calendar: bool = False,
    data_dir: Path | None = None,
    traces_dir: Path | None = None,
    budget: Budget | None = None,
) -> DemoApp:
    """A fully offline app with a frozen clock. Every call builds fresh state (eval trials are isolated)."""
    tz = ZoneInfo(timezone)
    clock = FixedClock(now)
    data_dir = data_dir or Path(tempfile.mkdtemp(prefix="stridedj-demo-"))

    insights = RunInsightsService(DemoRunSource(now=now), SqliteRunRepository(data_dir / "runs.db"), clock=clock)
    await insights.refresh()

    cal = (
        InMemoryCalendar()
        if calendar == "empty"
        else InMemoryCalendar.with_realistic_week(now, tz, include_poisoned=poisoned_calendar)
    )
    if calendar == "packed_evening":
        today = now.astimezone(tz).date()
        start = datetime.combine(today, time(16, 30), tzinfo=tz)
        end = datetime.combine(today, time(22, 0), tzinfo=tz)
        cal.add_busy(TimeSlot(start=start, end=end), title="Concert")

    planner = build_demo_planner()
    writer = RecordingPlaylistWriter()
    agent = StrideAgent(
        llm=llm,
        deps=AgentDeps(insights=insights, slots=SlotFinderService(cal), playlists=planner, clock=clock),
        approvals=ApprovalStore(clock=clock),
        executor=PlanExecutor(calendar=cal, playlists=writer, tz=tz),
        budget=budget,
        traces_dir=traces_dir,
    )
    return DemoApp(
        agent=agent,
        insights=insights,
        calendar=cal,
        playlists=planner,
        writer=writer,
        clock=clock,
        timezone=timezone,
        fake_calendar=cal,
        fake_writer=writer,
    )


async def build_live_app(settings: Settings, llm: LLMClient) -> App:
    """The real thing: run source per STRIDEDJ_RUN_SOURCE, Google Calendar, Spotify + ReccoBeats."""
    from stridedj.cal.gcal import GoogleCalendarClient
    from stridedj.music.live import build_live_music
    from stridedj.runs.sources.factory import build_run_source

    tz = ZoneInfo(settings.athlete.timezone)
    clock = SystemClock()
    settings.app.data_dir.mkdir(parents=True, exist_ok=True)

    repository = SqliteRunRepository.from_settings(settings.app)
    insights = RunInsightsService(
        build_run_source(settings), repository, zones=zones_from_settings(settings.athlete), clock=clock
    )
    await insights.refresh()

    def run_windows() -> list[TimeSlot]:
        """Past runs as time windows, so the music side can tell which plays happened during a run."""
        since = clock.now() - timedelta(days=365)
        return [
            TimeSlot(start=a.start, end=a.start + timedelta(seconds=a.elapsed_s))
            for a in repository.list_activities(since=since)
        ]

    calendar = GoogleCalendarClient(settings.google, tz=tz)
    planner, writer = build_live_music(settings, run_windows=run_windows)
    agent = StrideAgent(
        llm=llm,
        deps=AgentDeps(insights=insights, slots=SlotFinderService(calendar), playlists=planner, clock=clock),
        approvals=ApprovalStore(clock=clock),
        executor=PlanExecutor(calendar=calendar, playlists=writer, tz=tz),
        traces_dir=settings.app.data_dir / "traces",
    )
    return App(
        agent=agent,
        insights=insights,
        calendar=calendar,
        playlists=planner,
        writer=writer,
        clock=clock,
        timezone=settings.athlete.timezone,
    )
