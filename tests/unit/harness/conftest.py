from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest


class FrozenClock:
    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(datetime(2026, 10, 17, 16, 0, tzinfo=UTC))
