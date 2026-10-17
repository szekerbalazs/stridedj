"""The agent loop.

    user message ─► model ─┬─► tool calls ─► run tools (in parallel) ─► append results ─► model ─► ...
                           └─► final answer (strict JSON `AgentReply`) ─► validate ─► check against state ─► done

What makes this a *harness* and not just a while-loop:
* **Budgets**: max steps, max tool calls, max $ and max wall time. An agent must always terminate, and a confused
  model must not burn money in a loop.
* **Structured termination**: the run ends only with a schema-valid `AgentReply` (or a harness-generated fallback),
  so every caller (Telegram, CLI, evals) handles exactly one shape.
* **Verification against state**: the model's *claims* are checked against what actually happened (e.g. a
  "proposal" must reference a plan that `propose_plan` really created). One corrective note is allowed, then the
  harness falls back. In eval terms: we grade the outcome, not what the agent says about it.
* **Tracing**: every step goes into the transcript.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from stridedj.domain import AgentReply, ReplyKind
from stridedj.errors import StrideDJError
from stridedj.harness.tools import ToolRegistry
from stridedj.harness.tracing import Transcript
from stridedj.llm.base import InputItem, LLMClient

StopReason = Literal[
    "final",
    "refusal",
    "max_steps",
    "max_tool_calls",
    "max_cost",
    "max_time",
    "incomplete",
    "invalid_output",
    "llm_error",
]

ReplyCheck = Callable[[AgentReply], str | None]
"""Returns None if the reply is consistent with the harness state, else a correction message for the model."""


class Budget(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=8, ge=1, description="model calls per run")
    max_tool_calls: int = Field(default=16, ge=0)
    max_cost_usd: float = Field(default=0.20, gt=0)
    max_wall_s: float = Field(default=90.0, gt=0)
    max_corrections: int = Field(default=1, ge=0, description="corrective notes after an invalid/inconsistent answer")


class RunOutcome(BaseModel):
    reply: AgentReply
    stop_reason: StopReason
    format_ok: bool = Field(description="the model's final answer parsed and validated as AgentReply on first try")
    transcript: Transcript


FALLBACKS: dict[str, str] = {
    "budget": "Sorry, I couldn't finish planning that. Could you rephrase it a bit more specifically?",
    "error": "Something went wrong on my side. Please try again in a minute.",
}


def _fallback(kind: ReplyKind, key: str) -> AgentReply:
    return AgentReply(kind=kind, message=FALLBACKS[key], proposal_id=None, missing_fields=[])


async def run_agent(
    *,
    llm: LLMClient,
    tools: ToolRegistry,
    instructions: str,
    history: Sequence[InputItem],
    user_message: str,
    user_id: str,
    prompt_version: str,
    budget: Budget | None = None,
    reply_check: ReplyCheck | None = None,
    cache_key: str | None = None,
) -> RunOutcome:
    """Run the model with tools until it produces a valid final answer or a budget is exhausted."""
    budget = budget or Budget()
    transcript = Transcript(
        run_id=uuid.uuid4().hex[:12],
        user_id=user_id,
        model=llm.model,
        prompt_version=prompt_version,
        started_at=datetime.now(UTC),
    )
    transcript.add("user", text=user_message)
    items: list[InputItem] = [*history, llm.user_item(user_message)]
    specs = tools.specs()
    started = time.perf_counter()
    corrections = 0
    format_ok = True

    def finish(reply: AgentReply, reason: StopReason) -> RunOutcome:
        transcript.stop_reason = reason
        transcript.duration_s = round(time.perf_counter() - started, 3)
        transcript.add("final", reply=reply.model_dump(mode="json"), stop_reason=reason)
        return RunOutcome(reply=reply, stop_reason=reason, format_ok=format_ok, transcript=transcript)

    for _step in range(budget.max_steps):
        if time.perf_counter() - started > budget.max_wall_s:
            return finish(_fallback(ReplyKind.ERROR, "budget"), "max_time")
        try:
            turn = await llm.respond(
                instructions=instructions, items=items, tools=specs, output_schema=AgentReply, cache_key=cache_key
            )
        except StrideDJError as exc:
            transcript.add("harness", event="llm_error", error=str(exc))
            return finish(_fallback(ReplyKind.ERROR, "error"), "llm_error")

        transcript.steps += 1
        transcript.usage = transcript.usage + turn.usage
        transcript.cost_usd = round(llm.pricing.cost_usd(transcript.usage), 6)
        transcript.add(
            "llm",
            latency_s=round(turn.latency_s, 3),
            usage=turn.usage.model_dump(),
            tool_calls=[{"name": c.name, "arguments": c.arguments_json} for c in turn.tool_calls],
            text=turn.text,
            refusal=turn.refusal,
            incomplete=turn.incomplete_reason,
        )
        items.extend(turn.output_items)

        if transcript.cost_usd > budget.max_cost_usd:
            return finish(_fallback(ReplyKind.ERROR, "budget"), "max_cost")
        if turn.incomplete_reason is not None:
            return finish(_fallback(ReplyKind.ERROR, "budget"), "incomplete")
        if turn.refusal is not None:
            reply = AgentReply(kind=ReplyKind.REFUSAL, message=turn.refusal[:800], proposal_id=None, missing_fields=[])
            return finish(reply, "refusal")

        if turn.tool_calls:
            if transcript.tool_calls + len(turn.tool_calls) > budget.max_tool_calls:
                return finish(_fallback(ReplyKind.ERROR, "budget"), "max_tool_calls")
            transcript.tool_calls += len(turn.tool_calls)
            results = await asyncio.gather(*(tools.execute(call) for call in turn.tool_calls))
            for result in results:
                transcript.add(
                    "tool",
                    name=result.name,
                    arguments=result.arguments,
                    output=result.output,
                    is_error=result.is_error,
                    latency_s=round(result.latency_s, 3),
                )
                items.append(llm.tool_result_item(result.call_id, result.output))
            continue

        # ── a final answer: parse, validate, check against state ──
        try:
            reply = AgentReply.model_validate_json(turn.text or "")
        except ValidationError as exc:
            format_ok = False
            transcript.add("harness", event="invalid_output", error=str(exc.errors(include_input=False))[:500])
            if corrections >= budget.max_corrections:
                return finish(_fallback(ReplyKind.ERROR, "error"), "invalid_output")
            corrections += 1
            items.append(
                llm.harness_note_item("Your last answer was not a valid AgentReply JSON object. Answer again.")
            )
            continue

        problem = reply_check(reply) if reply_check else None
        if problem is None:
            return finish(reply, "final")
        transcript.add("harness", event="reply_check_failed", problem=problem)
        if corrections >= budget.max_corrections:
            return finish(_fallback(ReplyKind.ERROR, "error"), "invalid_output")
        corrections += 1
        items.append(llm.harness_note_item(problem))

    return finish(_fallback(ReplyKind.ERROR, "budget"), "max_steps")
