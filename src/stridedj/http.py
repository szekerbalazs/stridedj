"""One HTTP layer for every REST integration: timeouts, retries with backoff, and errors that never leak secrets.

Retry policy (the standard one for third-party APIs):
* retry only what can succeed on a second try: connection errors/timeouts, HTTP 429 and 5xx;
* never retry other 4xx (a bad request stays bad; 401/403 means re-authenticate);
* honour `Retry-After` (seconds or an HTTP date) when the server sends it, capped at `max_delay_s`;
* otherwise exponential backoff with full jitter: sleep ~ U(0, base · 2^attempt), which spreads out retrying clients;
* only retry idempotent requests by default (GET/PUT/DELETE/HEAD/OPTIONS); a POST is retried only when the caller
  says it is safe (e.g. it carries an idempotency key or a client-chosen id).
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from stridedj.errors import AuthError, ExternalAPIError

IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

Sleep = Callable[[float], Awaitable[None]]


class RetryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=4, ge=1)
    base_delay_s: float = Field(default=0.5, gt=0)
    max_delay_s: float = Field(default=20.0, gt=0)


def retry_after_seconds(headers: Mapping[str, str], *, now: datetime | None = None) -> float | None:
    """Parse a `Retry-After` header (delta-seconds or HTTP-date). Returns None if absent or unparsable."""
    raw = headers.get("retry-after")
    if raw is None:
        return None
    raw = raw.strip()
    if raw.isdigit():
        return float(raw)
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    reference = now or datetime.now(UTC)
    return max(0.0, (when - reference).total_seconds())


_RATE_LIMIT_REASONS = ("ratelimitexceeded", "userratelimitexceeded", "quotaexceeded")


def _is_rate_limit_403(response: httpx.Response) -> bool:
    """True for 403s that are really rate limits (Google APIs: error.errors[].reason = rateLimitExceeded, ...)."""
    try:
        text = response.text.lower()
    except UnicodeDecodeError:  # pragma: no cover
        return False
    return any(reason in text for reason in _RATE_LIMIT_REASONS)


def backoff_delay(attempt: int, policy: RetryPolicy, rng: random.Random | None = None) -> float:
    """Full-jitter exponential backoff for retry number `attempt` (0-based)."""
    cap = min(policy.max_delay_s, policy.base_delay_s * 2**attempt)
    return (rng or random).uniform(0.0, cap)


async def request_with_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    service: str,
    policy: RetryPolicy | None = None,
    retry_unsafe: bool = False,
    label: str | None = None,
    sleep: Sleep = asyncio.sleep,
    **kwargs: Any,
) -> httpx.Response:
    """Send a request; retry transient failures; return the successful (2xx/3xx) response.

    Raises `AuthError` on 401/403, `ExternalAPIError` on any other failure. The error message contains the service
    name, the status code and the path, but never headers, query strings or bodies (which may carry tokens).
    Pass `label` when the path itself is secret (Telegram puts the bot token in the URL path: `/bot<token>/getMe`).
    """
    policy = policy or RetryPolicy()
    method = method.upper()
    may_retry = method in IDEMPOTENT_METHODS or retry_unsafe
    path = label or (httpx.URL(url).path if "://" in url else url.split("?")[0])

    for attempt in range(policy.max_attempts):
        last = attempt == policy.max_attempts - 1
        try:
            response = await client.request(method, url, **kwargs)
        except httpx.TransportError as exc:  # connection refused, DNS, timeouts, ...
            if not may_retry or last:
                error = ExternalAPIError(service, f"{method} {path}: {type(exc).__name__}", retryable=True)
                # with a secret-bearing URL, drop the cause: httpx exception texts can include the request URL
                raise error from (None if label else exc)
            await sleep(backoff_delay(attempt, policy))
            continue

        status = response.status_code
        if status < 400:
            return response
        if status == 403 and _is_rate_limit_403(response):  # Google signals rate limits as 403 + reason
            status = 429
        if status in (401, 403):
            raise AuthError(service, f"{method} {path}: not authorised", status_code=status)
        if status in RETRYABLE_STATUS and may_retry and not last:
            hinted = retry_after_seconds(response.headers)
            delay = min(hinted, policy.max_delay_s) if hinted is not None else backoff_delay(attempt, policy)
            await sleep(delay)
            continue
        raise ExternalAPIError(
            service, f"{method} {path} failed", status_code=status, retryable=status in RETRYABLE_STATUS
        )

    raise AssertionError("unreachable")  # pragma: no cover


def build_client(
    *, base_url: str = "", timeout_s: float = 20.0, headers: Mapping[str, str] | None = None
) -> httpx.AsyncClient:
    """An `httpx.AsyncClient` with explicit timeouts (httpx defaults to 5 s for everything: too short for some APIs)."""
    timeout = httpx.Timeout(timeout_s, connect=10.0)
    return httpx.AsyncClient(base_url=base_url, timeout=timeout, headers=dict(headers or {}), follow_redirects=False)
