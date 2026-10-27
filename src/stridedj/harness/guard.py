"""Input guardrails, applied BEFORE anything reaches the model.

Order matters and is cheapest-first: who is it (allowlist) → is it sane (length) → is it too much (rate) → is it a
secret (never forward credentials to a third party, and Telegram keeps chat history on its servers).
These are deterministic checks in code. We deliberately do not ask an LLM "is this message malicious?": that check
is itself injectable, costs a call, and is probabilistic. The real protection against injection is architectural
(what the model can see and do), see docs/INJECTION.md.
"""

from __future__ import annotations

import html
import re
from collections import defaultdict, deque
from datetime import timedelta

from pydantic import BaseModel, ConfigDict

from stridedj.ports import Clock

_SECRET_HINTS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),  # OpenAI-style keys
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_\-]{30,}\b"),  # Telegram bot tokens
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),  # Google API keys
    re.compile(r"\bya29\.[A-Za-z0-9_\-\.]{20,}"),  # Google access tokens
    re.compile(r"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}"),  # GitHub tokens
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(password|passwort|api[_ ]?key|secret|token)\s*[:=]\s*\S{6,}"),
]


def looks_like_secret(text: str) -> bool:
    return any(p.search(text) for p in _SECRET_HINTS)


def escape_html(text: str) -> str:
    """Escape text for Telegram's HTML parse mode (we render every message with code, never raw model output)."""
    return html.escape(text, quote=False)


class GuardDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: str | None = None  # for logs
    notice: str | None = None  # what to tell the sender (None = stay silent)


class InputGuard:
    def __init__(
        self,
        *,
        allowed_user_ids: set[int],
        clock: Clock,
        max_chars: int = 500,
        max_messages: int = 30,
        per: timedelta = timedelta(minutes=10),
    ) -> None:
        self._allowed = allowed_user_ids
        self._clock = clock
        self._max_chars = max_chars
        self._max_messages = max_messages
        self._per = per
        self._recent: dict[int, deque[float]] = defaultdict(deque)

    def check(self, *, user_id: int, text: str) -> GuardDecision:
        if user_id not in self._allowed:
            return GuardDecision(allowed=False, reason="not_allowlisted", notice="This is a private bot.")
        if looks_like_secret(text):
            return GuardDecision(
                allowed=False,
                reason="secret_detected",
                notice="That looks like a password or API key. I did not process it. Please delete the message "
                "and never send secrets in a chat; put them into the .env file instead.",
            )
        if len(text) > self._max_chars:
            return GuardDecision(
                allowed=False, reason="too_long", notice=f"Please keep it under {self._max_chars} characters."
            )
        now = self._clock.now().timestamp()
        window = self._recent[user_id]
        while window and now - window[0] > self._per.total_seconds():
            window.popleft()
        if len(window) >= self._max_messages:
            return GuardDecision(
                allowed=False, reason="rate_limited", notice="Slow down a little, try again in a few minutes."
            )
        window.append(now)
        return GuardDecision(allowed=True)
