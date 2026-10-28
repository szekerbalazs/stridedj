"""The executor records created playlists (ids only) so the next drafts can rotate songs."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from evals.reference import reference_policy
from evals.tasks import load_tasks
from stridedj.app import build_demo_app
from stridedj.bot.telegram_bot import sync_forever
from stridedj.cal.demo import InMemoryCalendar
from stridedj.domain import RunPlan, TrackId, UserContext, UserId
from stridedj.errors import ExternalAPIError
from stridedj.harness.executor import PlanExecutor
from stridedj.llm.fake import ScriptedLLM
from stridedj.music.demo import RecordingPlaylistWriter
from stridedj.music.history import PlaylistHistory

NOW = datetime.fromisoformat("2026-10-17T15:00:00+02:00")
TZ = ZoneInfo("Europe/Zurich")
USER = UserContext(user_id=UserId("tg:1"), timezone="Europe/Zurich")


async def _plan() -> RunPlan:
    task = next(t for t in load_tasks() if t.id == "r01-10k-z2-tonight")
    app = await build_demo_app(ScriptedLLM(reference_policy(task)), now=NOW)
    response = await app.agent.handle(USER, "10k Z2 tonight")
    assert response.plan is not None and response.plan.playlist is not None
    return response.plan


class _FailingWriter:
    async def create_private_playlist(self, *, name: str, description: str, track_ids: Sequence[TrackId]) -> str:
        raise ExternalAPIError("spotify", "down", status_code=503, retryable=True)


async def test_a_created_playlist_is_recorded_for_rotation() -> None:
    plan, history = await _plan(), PlaylistHistory(Path(":memory:"))
    executor = PlanExecutor(calendar=InMemoryCalendar(), playlists=RecordingPlaylistWriter(), tz=TZ, history=history)
    result = await executor.execute(plan)
    assert result.playlist_url is not None
    assert plan.playlist is not None
    assert history.recent_playlists(last=1) == [plan.playlist.track_ids]


async def test_a_playlist_that_failed_is_not_recorded() -> None:
    plan, history = await _plan(), PlaylistHistory(Path(":memory:"))
    executor = PlanExecutor(calendar=InMemoryCalendar(), playlists=_FailingWriter(), tz=TZ, history=history)
    result = await executor.execute(plan)
    assert result.playlist_url is None and result.errors
    assert history.recent_playlists(last=1) == []


async def test_background_sync_survives_a_failing_round() -> None:
    calls, third_round = 0, asyncio.Event()

    async def sync() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("Spotify hiccup")
        if calls == 3:
            third_round.set()
        return 0

    task = asyncio.create_task(sync_forever(sync, every_s=0))
    await asyncio.wait_for(third_round.wait(), timeout=2)
    task.cancel()


async def test_a_delayed_background_job_waits_for_its_first_round() -> None:
    """Runs were refreshed at start-up, so their background job starts one period later."""
    calls = 0

    async def refresh() -> int:
        nonlocal calls
        calls += 1
        return 0

    task = asyncio.create_task(sync_forever(refresh, every_s=3600, first_after_s=0.2))
    await asyncio.sleep(0.05)
    assert calls == 0
    await asyncio.sleep(0.3)
    task.cancel()
    assert calls == 1
