"""Clocks (the `ports.Clock` implementations). Injected everywhere "now" matters, so tests and evals can freeze time."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    """A frozen clock (evals: every trial sees the same 'now'; tests: deterministic expiry checks)."""

    def __init__(self, now: datetime) -> None:
        if now.tzinfo is None:
            raise ValueError("FixedClock needs a timezone-aware datetime")
        self._now = now

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta
