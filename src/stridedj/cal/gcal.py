"""Google Calendar adapter: `freebusy.query` (no titles) and `events.insert` with a client-chosen id.

Why the REST API over httpx instead of `googleapiclient`: that client is synchronous (it would block the bot's event
loop), and going through `stridedj.http` gives the same timeouts, retries and secret-free errors as every other
integration.

* Reads use **only** `POST /freeBusy`, which returns busy intervals and nothing else. Event titles never reach this
  process, so an invite titled "Ignore previous instructions..." cannot reach the LLM. (`events.list` would hand us
  titles, descriptions and attendee e-mails we would then have to strip.)
* The only write is `events.insert`, called by the harness after the user approved a plan. The event id is derived
  from the approval, so a retry after a timeout cannot create a duplicate: Google answers 409 for a known id, which
  we treat as success. This is what makes it safe to retry this POST.
* There is deliberately no attendees parameter: this adapter cannot invite anyone or send e-mail on the user's behalf.

Docs: https://developers.google.com/calendar/api/v3/reference/freebusy/query,
https://developers.google.com/calendar/api/v3/reference/events/insert,
https://developers.google.com/calendar/api/guides/errors,
https://developers.google.com/identity/protocols/oauth2/native-app#offline
"""

from __future__ import annotations

import asyncio
import base64
import re
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any, Self
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, SecretStr, ValidationError

from stridedj.cal.slots import clip_to_window, merge_intervals
from stridedj.config import GoogleSettings
from stridedj.domain.calendar import TimeSlot
from stridedj.errors import AuthError, ConfigError, ExternalAPIError
from stridedj.http import RetryPolicy, Sleep, build_client, request_with_retries
from stridedj.ports import Clock

SERVICE = "google-calendar"
API_BASE = "https://www.googleapis.com/calendar/v3"
TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105 (an endpoint, not a secret)
REAUTH_HINT = "run `stridedj auth google` (in Testing mode Google refresh tokens expire after 7 days)"

TOKEN_EXPIRY_MARGIN_S = 60
"""Refresh this long before Google's stated expiry, so a token never expires between our check and the request."""

MAX_FREEBUSY_CALENDARS = 50  # `calendarExpansionMax` upper bound in the freebusy docs
_RETRYABLE_FREEBUSY_REASONS = frozenset({"internalError", "backendError"})

EVENT_ID_PATTERN = re.compile(r"[a-v0-9]{5,1024}")
"""Client-chosen event ids: base32hex characters (lowercase a-v, 0-9), 5 to 1024 long, unique per calendar."""


def event_id_for(plan_hash_hex: str) -> str:
    """A valid, deterministic Google event id for an approved plan: the base32hex encoding of the plan hash.

    Deterministic is the point: the same approval always maps to the same id, so a retried or replayed insert hits
    Google's duplicate-id check (409) instead of creating a second event. A SHA-256 hex digest gives 52 characters.
    """
    try:
        raw = bytes.fromhex(plan_hash_hex)
    except ValueError as exc:
        raise ValueError("plan_hash_hex must be a hex string with an even number of digits") from exc
    event_id = base64.b32hexencode(raw).decode("ascii").lower().rstrip("=")
    if not is_valid_event_id(event_id):
        raise ValueError("plan hash must be 3 to 640 bytes long to give a 5 to 1024 character event id")
    return event_id


def is_valid_event_id(event_id: str) -> bool:
    return EVENT_ID_PATTERN.fullmatch(event_id) is not None


# ─────────────────────────────── wire models (allowlisted fields only; unknown fields ignored) ───────────────────────
class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _TokenResponse(_Wire):
    access_token: SecretStr
    expires_in: int = Field(default=3600, gt=0)


class _BusyPeriod(_Wire):
    start: AwareDatetime
    end: AwareDatetime


class _FreeBusyError(_Wire):
    domain: str = ""
    reason: str = "unknown"


class _CalendarFreeBusy(_Wire):
    busy: list[_BusyPeriod] = Field(default_factory=list)
    errors: list[_FreeBusyError] = Field(default_factory=list)


class _FreeBusyResponse(_Wire):
    calendars: dict[str, _CalendarFreeBusy] = Field(default_factory=dict)


class _AccessToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: SecretStr
    expires_at: datetime


class _SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


# ─────────────────────────────── the adapter ───────────────────────────────
class GoogleCalendarClient:
    """`CalendarPort` over the Google Calendar REST API, authenticated with the user's OAuth refresh token.

    Scopes needed: `calendar.freebusy` (read availability) and `calendar.events` (insert). The access token is
    cached until shortly before it expires; one refresh is shared by concurrent callers.
    """

    def __init__(
        self,
        settings: GoogleSettings,
        *,
        tz: ZoneInfo,
        client: httpx.AsyncClient | None = None,
        clock: Clock | None = None,
        policy: RetryPolicy | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._tz = tz
        self._owns_client = client is None
        self._client = client or build_client()
        self._clock: Clock = clock or _SystemClock()
        self._policy = policy
        self._sleep = sleep
        self._token: _AccessToken | None = None
        self._token_lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the HTTP client if this adapter created it (an injected client belongs to the caller)."""
        if self._owns_client:
            await self._client.aclose()

    # ── CalendarPort ──
    async def busy_intervals(self, window: TimeSlot) -> list[TimeSlot]:
        """Merged, sorted busy intervals (UTC) within `window` across `settings.busy_calendar_ids`.

        Fails closed: if any calendar reports an error or is missing from the response, this raises instead of
        returning a partial picture, because "no busy data" would look like "free" and lead to double-booking.
        """
        calendar_ids = self._busy_calendar_ids()
        body = {
            "timeMin": _rfc3339_utc(window.start),
            "timeMax": _rfc3339_utc(window.end),
            "timeZone": "UTC",
            "items": [{"id": calendar_id} for calendar_id in calendar_ids],
        }
        # read-only POST: safe to retry
        response = await self._authorized(
            "POST", f"{API_BASE}/freeBusy", label="freeBusy", retry_unsafe=True, json=body
        )
        parsed = _parse(_FreeBusyResponse, response, what="freeBusy response")

        busy: list[TimeSlot] = []
        for position, calendar_id in enumerate(calendar_ids, start=1):
            # calendars are named by position, not id: ids are often e-mail addresses (personal data)
            availability = parsed.calendars.get(calendar_id)
            if availability is None:
                raise ExternalAPIError(SERVICE, f"freeBusy: no data for busy calendar #{position}")
            if availability.errors:
                reasons = sorted({error.reason for error in availability.errors})
                raise ExternalAPIError(
                    SERVICE,
                    f"freeBusy: busy calendar #{position} (GOOGLE_BUSY_CALENDAR_IDS) failed: {', '.join(reasons)}",
                    retryable=bool(_RETRYABLE_FREEBUSY_REASONS.intersection(reasons)),
                )
            busy.extend(TimeSlot(start=p.start, end=p.end) for p in availability.busy if p.end > p.start)
        return merge_intervals(clip_to_window(busy, window))

    async def insert_event(self, *, event_id: str, slot: TimeSlot, title: str, description: str) -> str:
        """Insert a single event with our own id into `settings.calendar_id`; idempotent (409 duplicate → success)."""
        if not is_valid_event_id(event_id):
            raise ValueError("event_id must be 5-1024 base32hex characters (a-v, 0-9); use event_id_for()")
        url = f"{API_BASE}/calendars/{quote(self._settings.calendar_id, safe='@')}/events"
        body = _event_body(event_id=event_id, slot=slot, title=title, description=description, tz=self._tz)
        try:
            # retrying this POST is safe because the id is ours: a replay can only ever produce a 409
            await self._authorized("POST", url, label="events.insert", retry_unsafe=True, json=body)
        except ExternalAPIError as exc:
            if exc.status_code == 409 and not isinstance(exc, AuthError):
                return event_id  # already created (by an earlier attempt whose response we never saw)
            raise
        return event_id

    # ── auth ──
    async def _authorized(
        self, method: str, url: str, *, label: str, retry_unsafe: bool, json: dict[str, Any]
    ) -> httpx.Response:
        """Send with a Bearer token. On 401 refresh once and retry (Google: "get a new access token using the
        long-lived refresh token"); a second 401, or a 403, becomes an `AuthError` with the re-login hint."""
        for attempt in range(2):
            token = await self._access_token()
            try:
                return await request_with_retries(
                    self._client,
                    method,
                    url,
                    service=SERVICE,
                    label=label,
                    retry_unsafe=retry_unsafe,
                    policy=self._policy,
                    sleep=self._sleep,
                    headers={"Authorization": f"Bearer {token}"},
                    json=json,
                )
            except AuthError as exc:
                if exc.status_code == 401 and attempt == 0:
                    self._forget_token(token)
                    continue
                raise _reauth_error(label, exc.status_code) from exc
        raise AssertionError("unreachable")  # pragma: no cover

    async def _access_token(self) -> str:
        async with self._token_lock:  # one refresh shared by concurrent callers
            now = self._clock.now()
            if self._token is None or now >= self._token.expires_at:
                self._token = await self._refresh_access_token(now)
            return self._token.value.get_secret_value()

    def _forget_token(self, token: str) -> None:
        if self._token is not None and self._token.value.get_secret_value() == token:
            self._token = None

    async def _refresh_access_token(self, now: datetime) -> _AccessToken:
        client_id, client_secret, refresh_token = self._credentials()
        form = {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        }
        try:
            # retry-safe: a refresh grant only mints another access token (Google does not rotate the refresh token)
            response = await request_with_retries(
                self._client,
                "POST",
                TOKEN_URL,
                service=SERVICE,
                label="oauth2 token refresh",
                retry_unsafe=True,
                policy=self._policy,
                sleep=self._sleep,
                data=form,
            )
        except ExternalAPIError as exc:
            # 400 invalid_grant: refresh token expired/revoked; 401 invalid_client: wrong client id/secret
            if exc.status_code in (400, 401, 403):
                raise _reauth_error("oauth2 token refresh", exc.status_code) from exc
            raise
        token = _parse(_TokenResponse, response, what="token response")
        lifetime_s = max(0, token.expires_in - TOKEN_EXPIRY_MARGIN_S)
        return _AccessToken(value=token.access_token, expires_at=now + timedelta(seconds=lifetime_s))

    def _credentials(self) -> tuple[str, str, str]:
        settings = self._settings
        client_secret = settings.client_secret.get_secret_value() if settings.client_secret else ""
        if not settings.client_id or not client_secret:
            raise ConfigError("GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set (see docs/SETUP.md §4)")
        refresh_token = settings.refresh_token.get_secret_value() if settings.refresh_token else ""
        if not refresh_token:
            raise AuthError(SERVICE, f"no refresh token: {REAUTH_HINT}")
        return settings.client_id, client_secret, refresh_token

    def _busy_calendar_ids(self) -> list[str]:
        ids = list(dict.fromkeys(self._settings.busy_calendar_ids))  # de-duplicate, keep order
        if not ids:
            # an empty list would make every slot look free: refuse rather than double-book
            raise ConfigError("GOOGLE_BUSY_CALENDAR_IDS is empty; set it to at least `primary`")
        if len(ids) > MAX_FREEBUSY_CALENDARS:
            raise ConfigError(f"GOOGLE_BUSY_CALENDAR_IDS lists more than {MAX_FREEBUSY_CALENDARS} calendars")
        return ids


# ─────────────────────────────── helpers ───────────────────────────────
def _event_body(*, event_id: str, slot: TimeSlot, title: str, description: str, tz: ZoneInfo) -> dict[str, Any]:
    """The complete insert payload. A fixed set of keys: no attendees, no conference data, no recurrence."""
    return {
        "id": event_id,
        "summary": title,
        "description": description,
        "start": {"dateTime": slot.start.astimezone(tz).isoformat(), "timeZone": tz.key},
        "end": {"dateTime": slot.end.astimezone(tz).isoformat(), "timeZone": tz.key},
        "reminders": {"useDefault": True},
    }


def _reauth_error(label: str, status_code: int | None) -> AuthError:
    if status_code == 403:
        problem = "forbidden (a missing scope, or a calendar you cannot write to)"
    else:
        problem = "credentials rejected (expired or revoked refresh token, or wrong client id/secret)"
    return AuthError(SERVICE, f"{label}: {problem}; {REAUTH_HINT}", status_code=status_code)


def _rfc3339_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse[M: BaseModel](model: type[M], response: httpx.Response, *, what: str) -> M:
    """Validate a JSON body; a malformed one becomes an `ExternalAPIError`.

    The cause is dropped on purpose: pydantic's `ValidationError` text echoes input values, which for the token
    response would include the access token.
    """
    try:
        return model.model_validate(response.json())
    except (ValueError, ValidationError):  # JSONDecodeError is a ValueError
        raise ExternalAPIError(SERVICE, f"unexpected {what}", status_code=response.status_code) from None
