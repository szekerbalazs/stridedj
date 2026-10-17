"""End-to-end through the real agent, tools, harness and demo adapters, with a scripted model (offline)."""

from __future__ import annotations

from datetime import datetime

import pytest

from evals.reference import reference_policy
from evals.tasks import load_tasks
from stridedj.app import DemoApp, build_demo_app
from stridedj.domain import ReplyKind, UserContext, UserId
from stridedj.errors import ApprovalError
from stridedj.llm.fake import ScriptedLLM, ScriptedTurn

NOW = datetime.fromisoformat("2026-10-17T15:00:00+02:00")
USER = UserContext(user_id=UserId("tg:1"), timezone="Europe/Zurich")


async def _app_for(task_id: str) -> DemoApp:
    task = next(t for t in load_tasks() if t.id == task_id)
    return await build_demo_app(ScriptedLLM(reference_policy(task)), now=NOW)


async def test_proposal_then_approval_writes_exactly_once() -> None:
    app = await _app_for("r01-10k-z2-tonight")
    response = await app.agent.handle(USER, "10k Z2 tonight")
    assert response.reply.kind is ReplyKind.PROPOSAL
    assert response.plan is not None
    assert response.approval_token is not None
    # nothing is written before approval
    assert app.fake_calendar.inserted == []
    assert app.fake_writer.created == []

    result = await app.agent.approve(response.approval_token, user_id=USER.user_id)
    assert result.errors == ()
    assert result.calendar_event_id is not None
    assert result.playlist_url is not None
    assert len(app.fake_calendar.inserted) == 1
    assert len(app.fake_writer.created) == 1

    with pytest.raises(ApprovalError):  # a double tap / replayed callback
        await app.agent.approve(response.approval_token, user_id=USER.user_id)
    assert len(app.fake_calendar.inserted) == 1


async def test_someone_else_cannot_approve() -> None:
    app = await _app_for("r01-10k-z2-tonight")
    response = await app.agent.handle(USER, "10k Z2 tonight")
    assert response.approval_token is not None
    with pytest.raises(ApprovalError, match="only the person"):
        await app.agent.approve(response.approval_token, user_id=UserId("tg:999"))
    assert app.fake_calendar.inserted == []


async def test_plan_numbers_come_from_state_not_the_model() -> None:
    app = await _app_for("r01-10k-z2-tonight")
    response = await app.agent.handle(USER, "10k Z2 tonight")
    plan = response.plan
    assert plan is not None
    assert plan.slot.duration_min >= plan.duration.slot_min
    assert plan.playlist is not None
    assert plan.playlist.total_min >= plan.duration.p80_min


async def test_hallucinated_proposal_id_is_caught() -> None:
    from stridedj.domain import AgentReply

    fake = AgentReply(kind=ReplyKind.PROPOSAL, message="Booked!", proposal_id="invented", missing_fields=[])
    app = await build_demo_app(ScriptedLLM([ScriptedTurn(final=fake), ScriptedTurn(final=fake)]), now=NOW)
    response = await app.agent.handle(USER, "10k Z2 tonight")
    assert response.reply.kind is ReplyKind.ERROR
    assert response.approval_token is None


async def test_track_titles_never_reach_the_model() -> None:
    app = await _app_for("r01-10k-z2-tonight")
    response = await app.agent.handle(USER, "10k Z2 tonight")
    assert response.plan is not None
    assert response.plan.playlist is not None
    tracks = await app.playlists.tracks_for(response.plan.playlist)
    assert tracks
    outputs = " ".join(str(e.data.get("output", "")) for e in response.outcome.transcript.tool_events())
    for track in tracks:
        assert track.name not in outputs


async def test_every_run_is_reported_to_the_usage_sink() -> None:
    from evals.tasks import load_tasks as _load
    from stridedj.harness.tracing import Transcript

    recorded: list[Transcript] = []

    class Sink:
        def record(self, transcript: Transcript) -> None:
            recorded.append(transcript)

    task = next(t for t in _load() if t.id == "r01-10k-z2-tonight")
    app = await build_demo_app(ScriptedLLM(reference_policy(task)), now=NOW, usage_sink=Sink())
    await app.agent.handle(USER, "10k Z2 tonight")
    assert len(recorded) == 1
    assert recorded[0].steps >= 1
