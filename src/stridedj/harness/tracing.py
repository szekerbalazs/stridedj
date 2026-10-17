"""Transcripts: the full record of one agent run (in eval terms: the *transcript* or *trajectory* of a *trial*).

Every run writes one JSON document: the user message, each model call (latency, tokens, which tools it called with
which arguments), each tool result, the final answer, the stop reason, total tokens and cost. Reading these is how
you debug an agent ("read the transcripts!", Anthropic's evals guide) and what the eval graders check.

Secrets never get here by construction (tools return aggregates), and `redact` is a second line of defence.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from stridedj.llm.base import Usage

EventKind = Literal["user", "llm", "tool", "harness", "final"]

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),  # OpenAI
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_\-]{30,}\b"),  # Telegram bot token
    re.compile(r"ya29\.[A-Za-z0-9_\-\.]+"),  # Google access token
    re.compile(r"1//[A-Za-z0-9_\-]{20,}"),  # Google refresh token
    re.compile(r"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}"),  # GitHub
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.=]{20,}"),
]


def redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def _redact_any(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _redact_any(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_any(v) for v in value]
    return value


class TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    t_s: float = Field(description="seconds since the start of the run")
    kind: EventKind
    data: dict[str, Any]


class Transcript(BaseModel):
    run_id: str
    user_id: str
    model: str
    prompt_version: str
    started_at: datetime
    events: list[TraceEvent] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    steps: int = 0
    tool_calls: int = 0
    stop_reason: str = "running"
    duration_s: float = 0.0

    def add(self, kind: EventKind, **data: Any) -> None:
        elapsed = (datetime.now(UTC) - self.started_at).total_seconds()
        self.events.append(TraceEvent(t_s=round(elapsed, 3), kind=kind, data=_redact_any(data)))

    def tool_events(self) -> list[TraceEvent]:
        return [e for e in self.events if e.kind == "tool"]

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.started_at:%Y%m%dT%H%M%S}_{self.run_id}.json"
        path.write_text(self.model_dump_json(indent=2))
        return path
