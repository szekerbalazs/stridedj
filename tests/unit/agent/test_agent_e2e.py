"""End-to-end through the real agent, tools, harness and demo adapters, with a scripted model (offline)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime

import pytest

from evals.reference import reference_policy
from evals.tasks import load_tasks
from stridedj.app import DemoApp, build_demo_app
from stridedj.domain import AgentReply, ReplyKind, UserContext, UserId
from stridedj.errors import ApprovalError
from stridedj.llm.base import InputItem
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


async def test_a_proposal_without_a_plan_is_caught() -> None:
    from stridedj.domain import AgentReply

    fake = AgentReply(kind=ReplyKind.PROPOSAL, message="Booked!", missing_fields=[])  # propose_plan never ran
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


async def test_a_named_free_start_is_checked_exactly_and_proposed() -> None:
    app = await _app_for("c13-time-between-day-parts")
    response = await app.agent.handle(USER, "10k Z2 tomorrow at 10:00")
    assert response.reply.kind is ReplyKind.PROPOSAL
    assert response.plan is not None
    assert f"{response.plan.slot.start.astimezone(USER.tz):%a %H:%M}" == "Sun 10:00"


async def test_a_named_busy_start_gets_the_nearest_free_starts_and_no_proposal() -> None:
    app = await _app_for("c14-requested-time-busy")
    response = await app.agent.handle(USER, "10k Z2 on Monday at 9:00")
    assert response.reply.kind is ReplyKind.ANSWER
    assert response.plan is None
    outputs = [
        e.data["output"] for e in response.outcome.transcript.tool_events() if e.data["name"] == "find_free_slots"
    ]
    slots_out = json.loads(outputs[-1])
    assert slots_out["requested_status"] == "busy"
    before, after = slots_out["slots"]
    # work starts 08:30: the latest quarter-hour start that still ends by then (the end depends on the slot length)
    hours, minutes = map(int, before["end_local"].split(":"))
    end_min, work_min = 60 * hours + minutes, 8 * 60 + 30
    assert end_min <= work_min < end_min + 15  # starting a quarter-hour later would not fit
    assert after["start_local"] == "Mon 19 Oct 17:30"  # the first start after work


async def test_the_context_has_a_date_table_so_the_model_never_computes_dates() -> None:
    llm = ScriptedLLM([ScriptedTurn(final=AgentReply(kind=ReplyKind.ANSWER, message="ok", missing_fields=[]))])
    app = await build_demo_app(llm, now=NOW)
    await app.agent.handle(USER, "hi")
    instructions = str(llm.calls[0]["instructions"])
    assert "today Sat 17 Oct = 2026-10-17, tomorrow Sun 18 Oct = 2026-10-18, Mon 19 Oct = 2026-10-19" in instructions
    assert "Sat 31 Oct = 2026-10-31" in instructions  # 14 days ahead, the planning horizon


async def test_a_weekday_that_does_not_match_the_date_is_a_correctable_tool_error() -> None:
    def policy(items: Sequence[InputItem]) -> ScriptedTurn:
        if not any(i.get("type") == "function_call_output" for i in items):
            args = {
                "weekday": "monday",
                "date": "2026-10-20",
                "day_part": "any",
                "duration_min": 60,
                "start_time": None,
            }
            return ScriptedTurn(tool_calls=[("find_free_slots", args)])
        return ScriptedTurn(final=AgentReply(kind=ReplyKind.ANSWER, message="ok", missing_fields=[]))

    app = await build_demo_app(ScriptedLLM(policy), now=NOW)
    response = await app.agent.handle(USER, "10k Z2 on Monday")
    (event,) = response.outcome.transcript.tool_events()
    assert event.data["is_error"] is True
    assert "2026-10-20 is a tuesday, not a monday; the next monday is 2026-10-19" in event.data["output"]


async def test_interval_sessions_are_offered_as_buttons_then_planned_with_the_picked_one() -> None:
    task = next(t for t in load_tasks() if t.id == "c24-intervals-picked-session")
    app = await build_demo_app(ScriptedLLM(reference_policy(task)), now=NOW)

    offer = await app.agent.handle(USER, task.turns[0])
    assert offer.reply.kind is ReplyKind.CLARIFICATION and offer.plan is None
    assert [o.session_id for o in offer.session_options] == ["i1", "i2", "i3"]
    assert not any(ch.isdigit() for ch in offer.text)  # the buttons carry the numbers

    picked = await app.agent.handle(USER, task.turns[1])
    plan = picked.plan
    assert plan is not None and plan.session is not None
    assert plan.session.session_id == "i2" and plan.session.reps_m == (800,) * 6
    assert plan.distance_km == plan.session.total_km
    assert plan.duration == plan.session.duration  # the session's own estimate, not the generic one
    assert picked.session_options == ()
    assert "{" not in picked.text and "800 m" not in picked.reply.message  # the template had no numbers ...
    assert f"{plan.duration.p50_min:.0f} min" in picked.text  # ... code filled them in

    assert picked.approval_token is not None
    await app.agent.approve(picked.approval_token, user_id=USER.user_id)
    assert "Session: 2 km warm-up · 6 × 800 m @" in app.fake_calendar.inserted[0].description
