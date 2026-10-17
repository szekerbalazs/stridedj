from __future__ import annotations

from datetime import timedelta

import pytest

from stridedj.domain import (
    CadenceTarget,
    DurationEstimate,
    EstimateMethod,
    PlanId,
    RunPlan,
    RunType,
    TimeSlot,
    UserId,
)
from stridedj.errors import ApprovalError
from stridedj.harness.approvals import ApprovalStore
from stridedj.harness.guard import InputGuard, escape_html, looks_like_secret
from stridedj.harness.tracing import redact

from .conftest import FrozenClock


def _plan(clock: FrozenClock, user: str = "tg:1") -> RunPlan:
    t0 = clock.now() + timedelta(hours=2)
    return RunPlan(
        plan_id=PlanId("p1"),
        user_id=UserId(user),
        run_type=RunType.Z2,
        distance_km=10,
        duration=DurationEstimate(
            run_type=RunType.Z2,
            distance_km=10,
            method=EstimateMethod.RECENT_SIMILAR,
            n_runs_used=5,
            p50_min=55,
            p80_min=58,
            slot_min=75,
            note="n",
        ),
        slot=TimeSlot(start=t0, end=t0 + timedelta(minutes=75)),
        cadence=CadenceTarget(run_type=RunType.Z2, spm=168, n_runs=8, lookback_days=120),
        playlist=None,
        created_at=clock.now(),
    )


def test_approval_is_single_use(clock: FrozenClock) -> None:
    store = ApprovalStore(clock=clock)
    token = store.issue(_plan(clock))
    assert store.consume(token, user_id=UserId("tg:1")).plan_id == "p1"
    with pytest.raises(ApprovalError, match="already approved"):
        store.consume(token, user_id=UserId("tg:1"))


def test_approval_expires(clock: FrozenClock) -> None:
    store = ApprovalStore(clock=clock, ttl=timedelta(minutes=15))
    token = store.issue(_plan(clock))
    clock.advance(timedelta(minutes=16))
    with pytest.raises(ApprovalError, match="expired"):
        store.consume(token, user_id=UserId("tg:1"))


def test_only_requester_can_approve(clock: FrozenClock) -> None:
    store = ApprovalStore(clock=clock)
    token = store.issue(_plan(clock))
    with pytest.raises(ApprovalError, match="only the person"):
        store.consume(token, user_id=UserId("tg:2"))


def test_tampered_plan_invalidates_token(clock: FrozenClock) -> None:
    store = ApprovalStore(clock=clock)
    plan = _plan(clock)
    token = store.issue(plan)
    store._plans[plan.plan_id] = plan.model_copy(update={"distance_km": 42.0})  # simulate a swapped plan
    with pytest.raises(ApprovalError, match="changed"):
        store.consume(token, user_id=UserId("tg:1"))


def test_unknown_and_cancelled_tokens(clock: FrozenClock) -> None:
    store = ApprovalStore(clock=clock)
    with pytest.raises(ApprovalError, match="unknown"):
        store.consume("forged", user_id=UserId("tg:1"))
    token = store.issue(_plan(clock))
    store.cancel(token, user_id=UserId("tg:1"))
    with pytest.raises(ApprovalError):
        store.consume(token, user_id=UserId("tg:1"))


def test_guard_allowlist_secret_length_rate(clock: FrozenClock) -> None:
    guard = InputGuard(allowed_user_ids={1}, clock=clock, max_chars=50, max_messages=2, per=timedelta(minutes=1))
    assert guard.check(user_id=2, text="10k z2").reason == "not_allowlisted"
    assert guard.check(user_id=1, text="my key is sk-abcdefghijklmnopqrstuvwx").reason == "secret_detected"
    assert guard.check(user_id=1, text="x" * 51).reason == "too_long"
    assert guard.check(user_id=1, text="10k z2 tonight").allowed
    assert guard.check(user_id=1, text="actually 12k").allowed
    assert guard.check(user_id=1, text="and again").reason == "rate_limited"
    clock.advance(timedelta(minutes=2))
    assert guard.check(user_id=1, text="later").allowed


@pytest.mark.parametrize(
    "text",
    [
        "sk-proj-abcdefghijklmnopqrstuvwxyz123",
        "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw",
        "password: hunter2hunter2",
        "-----BEGIN RSA PRIVATE KEY-----",
    ],
)
def test_secret_detection(text: str) -> None:
    assert looks_like_secret(text)


@pytest.mark.parametrize("text", ["10k Z2 tonight", "intervals 6x800 tomorrow morning", "my token bucket is empty"])
def test_no_false_positive_on_normal_requests(text: str) -> None:
    assert not looks_like_secret(text)


def test_escape_and_redact() -> None:
    assert escape_html("<b>x</b> & y") == "&lt;b&gt;x&lt;/b&gt; &amp; y"
    assert "sk-" not in redact("key sk-abcdefghijklmnopqrstu and bearer abcdefghijklmnopqrstuvwxyz")
    assert "REDACTED" in redact("123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw")
