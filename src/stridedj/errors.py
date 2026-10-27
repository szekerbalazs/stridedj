"""Exception hierarchy.

Every error raised by StrideDJ code derives from `StrideDJError`, so callers can catch "our" failures without swallowing
programming errors. Messages never contain secrets or raw personal data.
"""

from __future__ import annotations


class StrideDJError(Exception):
    """Base class for all StrideDJ errors."""


class ConfigError(StrideDJError):
    """A required setting is missing or invalid (see `.env.example` / docs/SETUP.md)."""


class ExternalAPIError(StrideDJError):
    """A call to a third-party API failed after retries (or with a non-retryable status)."""

    def __init__(self, service: str, message: str, *, status_code: int | None = None, retryable: bool = False) -> None:
        super().__init__(f"{service}: {message}" + (f" (HTTP {status_code})" if status_code is not None else ""))
        self.service = service
        self.status_code = status_code
        self.retryable = retryable


class RateLimitedError(ExternalAPIError):
    """HTTP 429. `retry_after_s` is the server's wait (None if not given); when it is longer than we would ever
    sleep, the caller must stop calling until it has passed (Spotify can block a development-mode app for hours)."""

    def __init__(self, service: str, message: str, *, retry_after_s: float | None) -> None:
        super().__init__(service, message, status_code=429, retryable=True)
        self.retry_after_s = retry_after_s


class AuthError(ExternalAPIError):
    """Credentials are missing, expired or revoked: re-run `stridedj auth <service>`."""


class ToolError(StrideDJError):
    """A tool call failed in a way the model can fix (bad arguments, no data, ...).

    The message is shown to the LLM, so it must be actionable ("distance_km must be > 0; ask the user") and must not
    leak internals.
    """


class ApprovalError(StrideDJError):
    """An approval token is unknown, expired, already used, or bound to a different plan or user."""


class BudgetExceededError(StrideDJError):
    """The agent loop hit its step, token, cost or time budget."""
