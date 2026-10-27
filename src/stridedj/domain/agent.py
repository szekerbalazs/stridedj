"""Agent-level contracts: the user context, the structured final answer, plans and their execution."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from stridedj.domain.base import DraftId, Frozen, PlanId, UserId
from stridedj.domain.calendar import TimeSlot
from stridedj.domain.music import PlaylistDraft
from stridedj.domain.planning import CadenceTarget, DurationEstimate, IntervalSession
from stridedj.domain.runs import RunType

MAX_REPLY_CHARS = 800


class UserContext(Frozen):
    """Who the harness acts for. Passed explicitly (never global): that is what makes multi-user possible later."""

    user_id: UserId
    timezone: str = "Europe/Zurich"

    @field_validator("timezone")
    @classmethod
    def _valid_tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone {v!r}") from exc
        return v

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


# ─────────────────────────────── the agent's final answer (strict structured output) ───────────────────────────────
class ReplyKind(StrEnum):
    """What the answer is. 'proposal': propose_plan created a plan in this turn (the app shows it with Approve/Cancel).
    'clarification': something essential is missing, or the runner has to choose (e.g. an interval session).
    'answer': a question about the runner's own running was answered. 'refusal': out of scope (medical advice,
    other people's data, doing anything directly). 'error': a tool or service failed; say so briefly, no details."""

    PROPOSAL = "proposal"  # a plan was proposed via propose_plan; the bot shows Approve/Cancel
    CLARIFICATION = "clarification"  # something essential is missing or ambiguous
    ANSWER = "answer"  # a question about past runs/predictions was answered
    REFUSAL = "refusal"  # out of scope (medical advice, other users' data, ...)
    ERROR = "error"  # a tool or service failed; tell the user briefly, without technical details


class MissingField(StrEnum):
    """What a clarification asks for."""

    DISTANCE = "distance"
    RUN_TYPE = "run_type"
    DATE = "date"
    DAY_PART = "day_part"


class AgentReply(BaseModel):
    """The ONLY shape the agent may finish with. Enforced with strict structured outputs, validated again by pydantic.

    Kept deliberately flat and free of numeric/length constraints in the JSON schema (strict mode supports only a
    subset of JSON Schema); the length limit is enforced by a validator instead, which runs after parsing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ReplyKind
    message: str = Field(
        description="Short reply to the runner, plain text, no markdown, no URLs. For a proposal: the plan's facts "
        "only as placeholders the app fills in ({start}, {expected}, {session}, {music}, ...), never as numbers."
    )
    missing_fields: list[MissingField] = Field(description="What to ask for when kind is 'clarification', else [].")

    @field_validator("message")
    @classmethod
    def _length(cls, v: str) -> str:
        if len(v) > MAX_REPLY_CHARS:
            raise ValueError(f"message longer than {MAX_REPLY_CHARS} characters")
        return v


# ─────────────────────────────── plans, approvals, execution ───────────────────────────────
class RunPlan(Frozen):
    """Everything an approval is bound to. Created by `propose_plan`; executed only after the user approves it."""

    plan_id: PlanId
    user_id: UserId
    run_type: RunType
    distance_km: float = Field(gt=0, le=100)
    duration: DurationEstimate
    slot: TimeSlot
    cadence: CadenceTarget
    playlist: PlaylistDraft | None
    session: IntervalSession | None = Field(default=None, description="the interval session, if one was chosen")
    created_at: AwareDatetime

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON: any change to the plan invalidates its approval (anti-TOCTOU)."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class ExecutionResult(Frozen):
    plan_id: PlanId
    calendar_event_id: str | None = None
    playlist_url: str | None = None
    playlist_draft_id: DraftId | None = Field(default=None, description="set when a playlist was created")
    errors: tuple[str, ...] = Field(default=(), description="user-safe descriptions of actions that failed")
