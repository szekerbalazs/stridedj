"""`StrideAgent`: the application service every front-end (Telegram, CLI, evals) talks to.

handle(user, text) → run the agent loop with this user's tools → if it proposed a plan: issue an approval token
approve(token, user) → consume the token (bound, single-use, unexpired) → execute the exact plan
cancel(token, user)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from stridedj.agent.placeholders import render_offer_message, render_proposal_message
from stridedj.agent.session import AgentSession
from stridedj.agent.tools import MAX_DAYS_AHEAD, AgentDeps, build_tools
from stridedj.costs.ledger import UsageSink
from stridedj.domain import (
    AgentReply,
    ExecutionResult,
    IntervalSession,
    ReplyKind,
    RunPlan,
    UserContext,
    UserId,
)
from stridedj.harness.approvals import ApprovalStore
from stridedj.harness.context import Prompt, compose_instructions, history_items
from stridedj.harness.executor import PlanExecutor
from stridedj.harness.loop import Budget, RunOutcome, run_agent
from stridedj.harness.tools import ToolRegistry
from stridedj.llm.base import LLMClient

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT = PROMPTS_DIR / "system_v4.md"  # v1-v3 kept for comparison (see docs/PROMPTING.md)


class AgentResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    reply: AgentReply
    text: str = Field(description="what the runner sees: the reply message with its placeholders filled in by code")
    plan: RunPlan | None
    approval_token: str | None
    outcome: RunOutcome
    session_options: tuple[IntervalSession, ...] = ()  # interval sessions offered in this turn: shown as buttons


class StrideAgent:
    def __init__(
        self,
        *,
        llm: LLMClient,
        deps: AgentDeps,
        approvals: ApprovalStore,
        executor: PlanExecutor,
        prompt: Prompt | None = None,
        budget: Budget | None = None,
        traces_dir: Path | None = None,
        tools_factory: Callable[[AgentDeps, AgentSession], ToolRegistry] = build_tools,
        usage_sink: UsageSink | None = None,
    ) -> None:
        self._llm = llm
        self._deps = deps
        self._approvals = approvals
        self._executor = executor
        self._prompt = prompt or Prompt.load(DEFAULT_PROMPT)
        self._budget = budget or Budget()
        self._traces_dir = traces_dir
        self._tools_factory = tools_factory
        self._usage_sink = usage_sink
        self._sessions: dict[UserId, AgentSession] = {}

    @property
    def prompt_version(self) -> str:
        return self._prompt.version

    def session(self, user: UserContext) -> AgentSession:
        if user.user_id not in self._sessions:
            self._sessions[user.user_id] = AgentSession(user=user)
        return self._sessions[user.user_id]

    def _context(self, user: UserContext, now: datetime) -> dict[str, str]:
        local = now.astimezone(user.tz)
        today = local.date()
        # a lookup table, so the model copies "Mon 02 Nov = 2026-11-02" instead of doing weekday arithmetic
        names = {0: "today ", 1: "tomorrow "}
        days = (today + timedelta(days=i) for i in range(MAX_DAYS_AHEAD + 1))
        return {
            "Now": f"{local:%A %d %B %Y, %H:%M} ({user.timezone})",
            "Dates (look them up, never compute them)": ", ".join(
                f"{names.get(i, '')}{d:%a %d %b} = {d.isoformat()}" for i, d in enumerate(days)
            ),
        }

    async def handle(self, user: UserContext, text: str) -> AgentResponse:
        session = self.session(user)
        session.start_run()
        now = self._deps.clock.now()

        def reply_check(reply: AgentReply) -> str | None:
            # Verify the model's claim against state: a "proposal" needs a plan created by propose_plan in THIS run.
            # (The harness attaches that plan itself: the model never copies an id.)
            if reply.kind is ReplyKind.PROPOSAL and not session.run_proposals:
                return (
                    "Your answer has kind 'proposal' but propose_plan did not create a plan in this conversation "
                    "turn. Call propose_plan, or answer with another kind."
                )
            return None

        outcome = await run_agent(
            llm=self._llm,
            tools=self._tools_factory(self._deps, session),
            instructions=compose_instructions(self._prompt, self._context(user, now)),
            history=history_items(self._llm, list(session.history)),
            user_message=text,
            user_id=user.user_id,
            prompt_version=self._prompt.version,
            budget=self._budget,
            reply_check=reply_check,
            cache_key=f"stridedj-{user.user_id}",
        )
        if self._usage_sink is not None:
            try:
                self._usage_sink.record(outcome.transcript)
            except Exception:  # accounting must never break a conversation
                log.exception("could not record usage")
        plan: RunPlan | None = None
        token: str | None = None
        options: tuple[IntervalSession, ...] = ()
        shown, replaced_because = outcome.reply.message, None
        if outcome.reply.kind is ReplyKind.PROPOSAL and session.run_proposals:
            plan = session.state.plans[session.run_proposals[-1]]  # the latest plan of this turn
            token = self._approvals.issue(plan)
            runner_said = " ".join([said for said, _ in session.history] + [text])
            shown, replaced_because = render_proposal_message(
                outcome.reply.message, plan, user.tz, runner_said=runner_said
            )
        elif session.run_options:
            options = tuple(session.state.sessions[i] for i in session.run_options)
            shown, replaced_because = render_offer_message(outcome.reply.message)
        if replaced_because is not None:
            log.info("reply message replaced by the standard one: %s", replaced_because)
            outcome.transcript.add("harness", event="message_replaced", reason=replaced_because)
        if self._traces_dir is not None:
            outcome.transcript.save(self._traces_dir)
        session.history.append((text, shown))
        return AgentResponse(
            reply=outcome.reply, text=shown, plan=plan, approval_token=token, outcome=outcome, session_options=options
        )

    async def approve(self, token: str, *, user_id: UserId) -> ExecutionResult:
        """Raises `ApprovalError` if the token is invalid for this user; otherwise executes the bound plan."""
        plan = self._approvals.consume(token, user_id=user_id)
        result = await self._executor.execute(plan)
        log.info("plan %s executed (errors: %d)", plan.plan_id, len(result.errors))
        return result

    def cancel(self, token: str, *, user_id: UserId) -> None:
        self._approvals.cancel(token, user_id=user_id)
