"""Human-in-the-loop approvals: the only door from "the agent proposed something" to "something was written".

An approval token is
* **bound** to one plan *and its content hash*: if the plan object changes after it was shown to the user, the
  token no longer matches (TOCTOU protection);
* bound to the **user** it was issued to (another chat member clicking "Approve" is rejected);
* **single use** (a double tap or a replayed callback cannot execute twice);
* **short-lived** (15 minutes by default: a stale proposal is not silently executed tomorrow).

The token is random (`secrets.token_urlsafe`), short enough for Telegram's 64-byte callback data, and reveals
nothing about the plan.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta

from pydantic import AwareDatetime, BaseModel, ConfigDict

from stridedj.domain import PlanId, RunPlan, UserId
from stridedj.errors import ApprovalError
from stridedj.ports import Clock


class Approval(BaseModel):
    model_config = ConfigDict(frozen=True)

    token: str
    plan_id: PlanId
    plan_hash: str
    user_id: UserId
    expires_at: AwareDatetime


class ApprovalStore:
    """In-memory approvals and plans (single process). A multi-instance deployment would back this with Redis/SQL."""

    def __init__(self, *, clock: Clock, ttl: timedelta = timedelta(minutes=15)) -> None:
        self._clock = clock
        self._ttl = ttl
        self._approvals: dict[str, Approval] = {}
        self._plans: dict[PlanId, RunPlan] = {}
        self._used: set[str] = set()

    def issue(self, plan: RunPlan) -> str:
        token = secrets.token_urlsafe(16)
        self._plans[plan.plan_id] = plan
        self._approvals[token] = Approval(
            token=token,
            plan_id=plan.plan_id,
            plan_hash=plan.content_hash(),
            user_id=plan.user_id,
            expires_at=self._clock.now() + self._ttl,
        )
        return token

    def plan_for(self, token: str) -> RunPlan | None:
        approval = self._approvals.get(token)
        return self._plans.get(approval.plan_id) if approval else None

    def consume(self, token: str, *, user_id: UserId) -> RunPlan:
        """Validate and burn the token; return the exact plan it was issued for. Raises `ApprovalError`."""
        approval = self._approvals.get(token)
        if approval is None:
            raise ApprovalError("unknown approval")
        if token in self._used:
            raise ApprovalError("this plan was already approved")
        if self._clock.now() > approval.expires_at:
            raise ApprovalError("this proposal expired; ask again")
        if user_id != approval.user_id:
            raise ApprovalError("only the person who asked can approve")
        plan = self._plans.get(approval.plan_id)
        if plan is None or plan.content_hash() != approval.plan_hash:
            raise ApprovalError("the plan changed after it was proposed; ask again")
        self._used.add(token)
        return plan

    def cancel(self, token: str, *, user_id: UserId) -> None:
        approval = self._approvals.get(token)
        if approval is not None and approval.user_id == user_id:
            self._used.add(token)

    def purge_expired(self, now: datetime | None = None) -> int:
        cutoff = now or self._clock.now()
        stale = [t for t, a in self._approvals.items() if a.expires_at < cutoff]
        for token in stale:
            del self._approvals[token]
        return len(stale)
