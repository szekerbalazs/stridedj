"""The cost ledger: one row per agent run (one user message), in a local SQLite file.

Why our own ledger next to OpenAI's official numbers: the official Costs API is daily and lags by hours, while every
model response already tells us its exact token counts. The ledger gives live, per-conversation costs (what did
"10k Z2 tonight" cost?), cache hit rates and per-source splits (bot / cli / eval).

API keys are identified by a **fingerprint** (a truncated SHA-256): enough to tell keys apart, useless to an
attacker, and safe to show on screen. The key itself is never stored.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict

from stridedj.harness.tracing import Transcript

Source = Literal["bot", "cli", "eval"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    started_at          TEXT NOT NULL,   -- ISO 8601, UTC
    key_fp              TEXT NOT NULL,
    source              TEXT NOT NULL,
    user_id             TEXT NOT NULL,
    model               TEXT NOT NULL,
    prompt_version      TEXT NOT NULL,
    steps               INTEGER NOT NULL,
    tool_calls          INTEGER NOT NULL,
    input_tokens        INTEGER NOT NULL,
    cached_input_tokens INTEGER NOT NULL,
    output_tokens       INTEGER NOT NULL,
    reasoning_tokens    INTEGER NOT NULL,
    cost_usd            REAL NOT NULL,
    stop_reason         TEXT NOT NULL,
    duration_s          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_by_key_time ON runs (key_fp, started_at);
"""


def key_fingerprint(api_key: str) -> str:
    """A short, stable, non-reversible id for an API key (12 hex chars of SHA-256)."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


class LedgerRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    started_at: AwareDatetime
    key_fp: str
    source: Source
    user_id: str
    model: str
    prompt_version: str
    steps: int
    tool_calls: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: float
    stop_reason: str
    duration_s: float


class UsageSink(Protocol):
    """Where the agent reports a finished run (the ledger, or nothing)."""

    def record(self, transcript: Transcript) -> None: ...


class CostLedger:
    def __init__(self, path: Path) -> None:
        self._path = path
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(_SCHEMA)

    def add(self, transcript: Transcript, *, key_fp: str, source: Source) -> None:
        t = transcript
        self._db.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                t.run_id,
                t.started_at.isoformat(),
                key_fp,
                source,
                t.user_id,
                t.model,
                t.prompt_version,
                t.steps,
                t.tool_calls,
                t.usage.input_tokens,
                t.usage.cached_input_tokens,
                t.usage.output_tokens,
                t.usage.reasoning_tokens,
                t.cost_usd,
                t.stop_reason,
                t.duration_s,
            ),
        )
        self._db.commit()

    def runs(self, *, since: datetime, key_fp: str | None = None) -> Sequence[LedgerRun]:
        """Runs started at or after `since`, newest first (optionally for one key)."""
        query = "SELECT * FROM runs WHERE started_at >= ?"
        params: list[str] = [since.isoformat()]
        if key_fp is not None:
            query += " AND key_fp = ?"
            params.append(key_fp)
        cursor = self._db.execute(query + " ORDER BY started_at DESC", params)
        columns = [c[0] for c in cursor.description]
        return [LedgerRun.model_validate(dict(zip(columns, row, strict=True))) for row in cursor.fetchall()]

    def sink(self, *, key_fp: str, source: Source) -> UsageSink:
        """A `UsageSink` bound to one key and source, for the agent to call after each run."""
        return _BoundSink(self, key_fp=key_fp, source=source)


class _BoundSink:
    def __init__(self, ledger: CostLedger, *, key_fp: str, source: Source) -> None:
        self._ledger = ledger
        self._key_fp = key_fp
        self._source: Source = source

    def record(self, transcript: Transcript) -> None:
        self._ledger.add(transcript, key_fp=self._key_fp, source=self._source)


def default_ledger_path(data_dir: Path) -> Path:
    return data_dir / "costs.db"
