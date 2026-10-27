"""Reference solutions: an oracle policy that solves each task from its `expect` block.

Anthropic's evals guide: "every task should have a reference solution that passes all graders"; a 0 % pass rate is
usually a broken task, not an incapable agent. Running the suite with this oracle (`run_eval.py --reference`) proves
that every task is solvable in our environment and that the graders accept a correct solution. Only then do model
failures mean something.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from zoneinfo import ZoneInfo

from evals.graders import resolve_date
from evals.tasks import Task, expected_kinds
from stridedj.domain import AgentReply, DayPart, MissingField, Mood, ReplyKind, Weekday
from stridedj.llm.base import InputItem
from stridedj.llm.fake import Policy, ScriptedTurn

QUESTION_TOOLS = ("get_running_profile", "predict_race_times")


def _current_turn_results(items: Sequence[InputItem]) -> dict[str, Any]:
    """Tool name → parsed output, for tool calls made since the last user message."""
    last_user = max((i for i, item in enumerate(items) if item.get("role") == "user"), default=-1)
    turn = items[last_user + 1 :]
    names = {i["call_id"]: i["name"] for i in turn if i.get("type") == "function_call"}
    return {names[i["call_id"]]: json.loads(i["output"]) for i in turn if i.get("type") == "function_call_output"}


def _final(
    kind: ReplyKind, message: str, *, proposal_id: str | None = None, missing: list[MissingField] | None = None
) -> ScriptedTurn:
    return ScriptedTurn(
        final=AgentReply(kind=kind, message=message, proposal_id=proposal_id, missing_fields=missing or [])
    )


def _planning_step(task: Task, tz: ZoneInfo, done: dict[str, Any], *, propose: bool) -> ScriptedTurn | None:
    """The next planning step, or None once propose_plan has run (or, if `propose` is false, once slots were found)."""
    e = task.expect
    run_type = e.run_type.value if e.run_type else "z2"
    distance = e.distance_km or 10.0
    if "estimate_duration" not in done:
        return ScriptedTurn(tool_calls=[("estimate_duration", {"run_type": run_type, "distance_km": distance})])
    estimate = done["estimate_duration"]
    if "find_free_slots" not in done:
        resolved = resolve_date(e.date or "today", task.now, tz)
        day = resolved.isoformat()
        part = (e.day_part or DayPart.ANY).value
        calls: list[tuple[str, dict[str, Any]]] = [
            (
                "find_free_slots",
                {
                    "weekday": Weekday.of(resolved).value,
                    "date": day,
                    "day_part": part,
                    "duration_min": estimate["slot_min"],
                    "start_time": e.start_time,
                },
            )
        ]
        if e.playlist is not False:
            mood = (e.mood or Mood.USUAL).value
            calls.append(("draft_playlist", {"run_type": run_type, "duration_min": estimate["p80_min"], "mood": mood}))
        return ScriptedTurn(tool_calls=calls)  # independent → in parallel, like the prompt asks
    if "propose_plan" in done or not propose:
        return None
    slots = done["find_free_slots"]["slots"]
    if not slots:
        return _final(ReplyKind.ANSWER, "Nothing is free then. Shall I look at another time?")
    draft = done.get("draft_playlist")
    args = {
        "run_type": run_type,
        "distance_km": distance,
        "slot_id": slots[0]["slot_id"],
        "draft_id": draft["draft"]["draft_id"] if draft else None,
    }
    return ScriptedTurn(tool_calls=[("propose_plan", args)])


def reference_policy(task: Task, timezone: str = "Europe/Zurich") -> Policy:
    e = task.expect
    kind = expected_kinds(e)[0]
    tz = ZoneInfo(timezone)

    def policy(items: Sequence[InputItem]) -> ScriptedTurn:
        done = _current_turn_results(items)
        if kind is ReplyKind.CLARIFICATION:
            return _final(kind, "How far would you like to run?", missing=list(e.missing_fields))
        if kind is ReplyKind.REFUSAL:
            return _final(kind, "Sorry, I can only help you plan runs.")
        for name in QUESTION_TOOLS:
            if name in e.must_call and name not in done:
                args = {"run_type": e.run_type.value if e.run_type else "z2"} if name == "get_running_profile" else {}
                return ScriptedTurn(tool_calls=[(name, args)])
        if kind is ReplyKind.PROPOSAL or "find_free_slots" in e.must_call:
            step = _planning_step(task, tz, done, propose=kind is ReplyKind.PROPOSAL)
            if step is not None:
                return step
        if kind is ReplyKind.PROPOSAL:
            return _final(kind, "Planned, see the details below.", proposal_id=done["propose_plan"]["proposal_id"])
        return _final(kind, "Here is what I found.")

    return policy
