"""Helpers shared by the music tests: a controllable clock and fixture loading."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "reccobeats"


class FrozenClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.current = start or datetime(2026, 10, 17, 12, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self.current

    def advance(self, **kwargs: float) -> None:
        self.current += timedelta(**kwargs)


def load_fixture(name: str) -> Any:
    """A real ReccoBeats response captured on 2026-10-17 (`availableCountries` truncated to keep files small)."""
    return json.loads((FIXTURES / name).read_text())
