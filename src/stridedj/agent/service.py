"""`StrideAgent`: the application service every front-end (Telegram, CLI, evals) talks to.

handle(user, text) → run the agent loop with this user's tools → if it proposed a plan: issue an approval token
approve(token, user) → consume the token (bound, single-use, unexpired) → execute the exact plan
redraft(token, user, music) → the same plan with only its playlist rebuilt (no model call) → a new token
cancel(token, user)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from stridedj.agent.placeholders import render_offer_message, render_proposal_message
from stridedj.agent.session import AgentSession
from stridedj.agent.tools import DRAFT_TIMEOUT_S, MAX_DAYS_AHEAD, AgentDeps, build_tools
from stridedj.costs.ledger import UsageSink
from stridedj.domain import (
    AgentReply,
    ExecutionResult,
    IntervalSession,
    MusicChoice,
    PlanId,
    ReplyKind,
    RunPlan,
    UserContext,
    UserId,
)
from stridedj.errors import ApprovalError
from stridedj.harness.approvals import ApprovalStore
from stridedj.harness.context import Prompt, compose_instructions, history_items
from stridedj.harness.executor import PlanExecutor
from stridedj.harness.loop import Budget, RunOutcome, run_agent
from stridedj.harness.tools import ToolRegistry
from stridedj.llm.base import LLMClient

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT = PROMPTS_DIR / "system_v5.md"  # v1-v4 kept for comparison (see docs/PROMPTING.md)


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

    async def redraft(self, token: str, *, user_id: UserId, music: MusicChoice) -> tuple[RunPlan, str]:
        """Rebuild only the playlist of a proposed plan with another music choice (the bot's 0 / 50 / 100 % new
        buttons): same run, slot, cadence, length, mood and genre, no model call. Returns the new plan and its token;
        the old token is cancelled only once the new playlist exists, so a failed draft leaves the old card usable.
        Raises `ApprovalError` (token no longer valid), `TimeoutError` or a `StrideDJError` from the music services."""
        plan = self._approvals.check(token, user_id=user_id)
        old = plan.playlist
        if old is None:
            raise ApprovalError("this plan has no playlist")
        draft = await asyncio.wait_for(
            self._deps.playlists.draft(
                run_type=plan.run_type,
                cadence=plan.cadence,
                duration_min=old.requested_min or plan.duration.p80_min,
                mood=old.mood,
                music=music,
                genre=old.genre,
            ),
            timeout=DRAFT_TIMEOUT_S,
        )
        self._approvals.check(token, user_id=user_id)  # still open? (Approve may have been tapped meanwhile)
        new = plan.model_copy(
            update={"plan_id": PlanId(uuid.uuid4().hex[:12]), "playlist": draft, "created_at": self._deps.clock.now()}
        )
        self._approvals.cancel(token, user_id=user_id)
        new_token = self._approvals.issue(new)
        if (session := self._sessions.get(user_id)) is not None:
            session.state.drafts[draft.draft_id] = draft
            session.state.plans[new.plan_id] = new
        log.info("plan %s: playlist redrafted (%s) as plan %s", plan.plan_id, music.value, new.plan_id)
        return new, new_token

    def cancel(self, token: str, *, user_id: UserId) -> None:
        self._approvals.cancel(token, user_id=user_id)
