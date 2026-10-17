from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from urllib.parse import parse_qs
from zoneinfo import ZoneInfo

import httpx
import pytest
from pydantic import SecretStr

from stridedj.cal.gcal import GoogleCalendarClient, event_id_for, is_valid_event_id
from stridedj.cal.slots import window_for
from stridedj.config import GoogleSettings
from stridedj.domain.calendar import DayPart, TimeSlot
from stridedj.errors import AuthError, ConfigError, ExternalAPIError

ZRH = ZoneInfo("Europe/Zurich")
SAT = date(2026, 10, 24)
SUN = date(2026, 10, 25)
T0 = datetime(2026, 10, 24, 18, 0, tzinfo=UTC)
FREEBUSY_URL = "https://www.googleapis.com/calendar/v3/freeBusy"
EVENT_ID = event_id_for(hashlib.sha256(b"plan-1").hexdigest())


def local(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), tzinfo=ZRH)


class FrozenClock:
    def __init__(self, now: datetime) -> None:
        self.current = now

    def now(self) -> datetime:
        return self.current


class Recorder:
    def __init__(self) -> None:
        self.sleeps: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class FakeGoogle:
    """Routes token requests to an auto-incrementing token issuer and API requests to a scripted queue."""

    def __init__(self, api: list[httpx.Response] | None = None, token: list[httpx.Response] | None = None) -> None:
        self.api = list(api or [])
        self.token = list(token or [])
        self.api_requests: list[httpx.Request] = []
        self.token_requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "oauth2.googleapis.com":
            self.token_requests.append(request)
            if self.token:
                return self.token.pop(0)
            n = len(self.token_requests)
            return httpx.Response(200, json={"access_token": f"at-{n}", "expires_in": 3600, "token_type": "Bearer"})
        self.api_requests.append(request)
        return self.api.pop(0)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def settings(**overrides: Any) -> GoogleSettings:
    values: dict[str, Any] = {
        "client_id": "cid.apps.googleusercontent.com",
        "client_secret": SecretStr("csecret-value"),
        "refresh_token": SecretStr("rtoken-value"),
        "calendar_id": "primary",
        "busy_calendar_ids": ["primary", "work@example.com"],
    }
    values.update(overrides)
    return GoogleSettings(**values)


def make_client(fake: FakeGoogle, clock: FrozenClock | None = None, **overrides: Any) -> GoogleCalendarClient:
    return GoogleCalendarClient(
        settings(**overrides), tz=ZRH, client=fake.client(), clock=clock or FrozenClock(T0), sleep=Recorder().sleep
    )


def freebusy(calendars: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "kind": "calendar#freeBusy",
            "timeMin": "2026-10-24T15:00:00.000Z",
            "timeMax": "2026-10-24T19:30:00.000Z",
            "calendars": calendars,
        },
    )


def busy(*periods: tuple[str, str]) -> dict[str, Any]:
    return {"busy": [{"start": s, "end": e} for s, e in periods]}


def ok_freebusy() -> httpx.Response:
    return freebusy({"primary": busy(), "work@example.com": busy()})


# ─────────────────────────────── freeBusy ───────────────────────────────
async def test_freebusy_request_and_merged_response_across_calendars() -> None:
    fake = FakeGoogle(
        api=[
            freebusy(
                {
                    "primary": busy(
                        ("2026-10-24T16:00:00Z", "2026-10-24T17:00:00Z"),
                        ("2026-10-24T18:30:00Z", "2026-10-24T19:00:00Z"),
                    ),
                    "work@example.com": busy(
                        ("2026-10-24T18:30:00+02:00", "2026-10-24T19:30:00+02:00"),  # 16:30-17:30Z, overlaps
                        ("2026-10-24T15:00:00Z", "2026-10-24T15:15:00Z"),
                    ),
                }
            )
        ]
    )
    window = window_for(SAT, DayPart.EVENING, ZRH)  # 17:00-21:30 CEST = 15:00-19:30Z

    intervals = await make_client(fake).busy_intervals(window)

    request = fake.api_requests[0]
    assert (request.method, str(request.url)) == ("POST", FREEBUSY_URL)
    assert request.headers["authorization"] == "Bearer at-1"
    assert json.loads(request.content) == {
        "timeMin": "2026-10-24T15:00:00Z",
        "timeMax": "2026-10-24T19:30:00Z",
        "timeZone": "UTC",
        "items": [{"id": "primary"}, {"id": "work@example.com"}],
    }
    utc = UTC
    assert [(i.start, i.end) for i in intervals] == [
        (datetime(2026, 10, 24, 15, 0, tzinfo=utc), datetime(2026, 10, 24, 15, 15, tzinfo=utc)),
        (datetime(2026, 10, 24, 16, 0, tzinfo=utc), datetime(2026, 10, 24, 17, 30, tzinfo=utc)),
        (datetime(2026, 10, 24, 18, 30, tzinfo=utc), datetime(2026, 10, 24, 19, 0, tzinfo=utc)),
    ]


async def test_token_refresh_request_is_a_form_post_with_the_refresh_grant() -> None:
    fake = FakeGoogle(api=[ok_freebusy()])
    await make_client(fake).busy_intervals(window_for(SAT, DayPart.EVENING, ZRH))
    request = fake.token_requests[0]
    assert (request.method, str(request.url)) == ("POST", "https://oauth2.googleapis.com/token")
    assert request.headers["content-type"] == "application/x-www-form-urlencoded"
    assert parse_qs(request.content.decode()) == {
        "grant_type": ["refresh_token"],
        "client_id": ["cid.apps.googleusercontent.com"],
        "client_secret": ["csecret-value"],
        "refresh_token": ["rtoken-value"],
    }


@pytest.mark.parametrize(("reason", "retryable"), [("notFound", False), ("internalError", True)])
async def test_per_calendar_errors_fail_closed_without_leaking_the_calendar_id(reason: str, retryable: bool) -> None:
    fake = FakeGoogle(
        api=[freebusy({"primary": busy(), "work@example.com": {"errors": [{"domain": "global", "reason": reason}]}})]
    )
    with pytest.raises(ExternalAPIError) as info:
        await make_client(fake).busy_intervals(window_for(SAT, DayPart.EVENING, ZRH))
    assert reason in str(info.value)
    assert "#2" in str(info.value)
    assert "work@example.com" not in str(info.value)
    assert info.value.retryable is retryable


async def test_calendar_missing_from_response_fails_closed() -> None:
    fake = FakeGoogle(api=[freebusy({"primary": busy()})])
    with pytest.raises(ExternalAPIError, match="no data for busy calendar #2"):
        await make_client(fake).busy_intervals(window_for(SAT, DayPart.EVENING, ZRH))


async def test_empty_busy_calendar_list_is_a_config_error() -> None:
    fake = FakeGoogle()
    with pytest.raises(ConfigError, match="GOOGLE_BUSY_CALENDAR_IDS"):
        await make_client(fake, busy_calendar_ids=[]).busy_intervals(window_for(SAT, DayPart.EVENING, ZRH))
    assert fake.api_requests == fake.token_requests == []


async def test_malformed_freebusy_response_is_an_external_api_error() -> None:
    fake = FakeGoogle(api=[httpx.Response(200, content=b"<html>oops</html>")])
    with pytest.raises(ExternalAPIError, match="unexpected freeBusy response"):
        await make_client(fake).busy_intervals(window_for(SAT, DayPart.EVENING, ZRH))


# ─────────────────────────────── access token ───────────────────────────────
async def test_access_token_is_cached_until_shortly_before_expiry() -> None:
    clock = FrozenClock(T0)
    fake = FakeGoogle(api=[ok_freebusy() for _ in range(4)])
    client = make_client(fake, clock)
    window = window_for(SAT, DayPart.EVENING, ZRH)

    await client.busy_intervals(window)
    clock.current = T0 + timedelta(seconds=3539)  # expires_in 3600 minus the 60 s margin, minus 1 s
    await client.busy_intervals(window)
    assert len(fake.token_requests) == 1

    clock.current = T0 + timedelta(seconds=3540)
    await client.busy_intervals(window)
    await client.busy_intervals(window)
    assert len(fake.token_requests) == 2
    assert [r.headers["authorization"] for r in fake.api_requests] == ["Bearer at-1"] * 2 + ["Bearer at-2"] * 2


async def test_concurrent_calls_share_one_refresh() -> None:
    fake = FakeGoogle(api=[ok_freebusy() for _ in range(3)])
    client = make_client(fake)
    window = window_for(SAT, DayPart.EVENING, ZRH)
    await asyncio.gather(*(client.busy_intervals(window) for _ in range(3)))
    assert len(fake.token_requests) == 1


async def test_401_refreshes_the_token_once_and_retries() -> None:
    fake = FakeGoogle(api=[httpx.Response(401), ok_freebusy()])
    await make_client(fake).busy_intervals(window_for(SAT, DayPart.EVENING, ZRH))
    assert len(fake.token_requests) == 2
    assert [r.headers["authorization"] for r in fake.api_requests] == ["Bearer at-1", "Bearer at-2"]


async def test_repeated_401_raises_auth_error_with_the_relogin_hint() -> None:
    fake = FakeGoogle(api=[httpx.Response(401), httpx.Response(401)])
    with pytest.raises(AuthError) as info:
        await make_client(fake).busy_intervals(window_for(SAT, DayPart.EVENING, ZRH))
    assert "stridedj auth google" in str(info.value)
    assert "7 days" in str(info.value)
    assert info.value.status_code == 401


async def test_invalid_grant_raises_auth_error_with_hint_and_no_secrets() -> None:
    expired = httpx.Response(
        400, json={"error": "invalid_grant", "error_description": "Token has been expired or revoked."}
    )
    fake = FakeGoogle(token=[expired])
    with pytest.raises(AuthError) as info:
        await make_client(fake).busy_intervals(window_for(SAT, DayPart.EVENING, ZRH))
    message = str(info.value)
    assert "stridedj auth google" in message
    assert "Testing mode" in message
    assert "rtoken-value" not in message
    assert "csecret-value" not in message
    assert fake.api_requests == []


async def test_missing_credentials() -> None:
    fake = FakeGoogle()
    with pytest.raises(AuthError, match="stridedj auth google"):
        await make_client(fake, refresh_token=None).busy_intervals(window_for(SAT, DayPart.EVENING, ZRH))
    with pytest.raises(ConfigError, match="GOOGLE_CLIENT_ID"):
        await make_client(fake, client_secret=SecretStr("")).busy_intervals(window_for(SAT, DayPart.EVENING, ZRH))
    assert fake.token_requests == []


# ─────────────────────────────── events.insert ───────────────────────────────
def run_slot() -> TimeSlot:
    return TimeSlot(start=local(SUN, 6), end=local(SUN, 6, 45))


async def test_insert_event_request_shape() -> None:
    fake = FakeGoogle(api=[httpx.Response(200, json={"id": EVENT_ID, "status": "confirmed"})])
    slot = TimeSlot(start=datetime(2026, 10, 25, 5, tzinfo=UTC), end=datetime(2026, 10, 25, 5, 45, tzinfo=UTC))

    returned = await make_client(fake).insert_event(
        event_id=EVENT_ID, slot=slot, title="Run: 10 km Z2", description="Planned by StrideDJ"
    )

    assert returned == EVENT_ID
    request = fake.api_requests[0]
    assert (request.method, str(request.url)) == (
        "POST",
        "https://www.googleapis.com/calendar/v3/calendars/primary/events",
    )
    assert request.url.query == b""  # no sendUpdates or other options
    assert json.loads(request.content) == {
        "id": EVENT_ID,
        "summary": "Run: 10 km Z2",
        "description": "Planned by StrideDJ",
        "start": {"dateTime": "2026-10-25T06:00:00+01:00", "timeZone": "Europe/Zurich"},
        "end": {"dateTime": "2026-10-25T06:45:00+01:00", "timeZone": "Europe/Zurich"},
        "reminders": {"useDefault": True},
    }


async def test_insert_event_encodes_the_calendar_id() -> None:
    fake = FakeGoogle(api=[httpx.Response(200, json={"id": EVENT_ID})])
    client = make_client(fake, calendar_id="abc#runs@group.calendar.google.com")
    await client.insert_event(event_id=EVENT_ID, slot=run_slot(), title="Run", description="")
    assert fake.api_requests[0].url.raw_path == b"/calendar/v3/calendars/abc%23runs@group.calendar.google.com/events"


async def test_duplicate_id_409_is_success() -> None:
    duplicate = httpx.Response(
        409,
        json={
            "error": {
                "errors": [
                    {"domain": "global", "reason": "duplicate", "message": "The requested identifier already exists."}
                ],
                "code": 409,
                "message": "The requested identifier already exists.",
            }
        },
    )
    fake = FakeGoogle(api=[duplicate])
    assert (
        await make_client(fake).insert_event(event_id=EVENT_ID, slot=run_slot(), title="Run", description="")
        == EVENT_ID
    )


async def test_retry_after_a_lost_response_lands_on_409_and_succeeds() -> None:
    # the first POST may have created the event before failing; the retry carries the same id and gets 409
    fake = FakeGoogle(api=[httpx.Response(503), httpx.Response(409)])
    assert (
        await make_client(fake).insert_event(event_id=EVENT_ID, slot=run_slot(), title="Run", description="")
        == EVENT_ID
    )
    assert len(fake.api_requests) == 2
    assert {json.loads(r.content)["id"] for r in fake.api_requests} == {EVENT_ID}


async def test_insert_403_raises_auth_error_without_a_token_refresh() -> None:
    fake = FakeGoogle(api=[httpx.Response(403)])
    with pytest.raises(AuthError, match="stridedj auth google") as info:
        await make_client(fake).insert_event(event_id=EVENT_ID, slot=run_slot(), title="Run", description="")
    assert info.value.status_code == 403
    assert len(fake.token_requests) == 1


async def test_other_insert_errors_propagate() -> None:
    fake = FakeGoogle(api=[httpx.Response(400)])
    with pytest.raises(ExternalAPIError) as info:
        await make_client(fake).insert_event(event_id=EVENT_ID, slot=run_slot(), title="Run", description="")
    assert info.value.status_code == 400


@pytest.mark.parametrize("bad_id", ["abcd", "ABCDE12345", "abcdw12345", "abc-de-123", "a" * 1025])
async def test_insert_rejects_invalid_event_ids_before_any_request(bad_id: str) -> None:
    fake = FakeGoogle()
    with pytest.raises(ValueError, match="base32hex"):
        await make_client(fake).insert_event(event_id=bad_id, slot=run_slot(), title="Run", description="")
    assert fake.api_requests == fake.token_requests == []


def test_insert_event_cannot_take_attendees() -> None:
    parameters = set(inspect.signature(GoogleCalendarClient.insert_event).parameters)
    assert parameters == {"self", "event_id", "slot", "title", "description"}


# ─────────────────────────────── event ids ───────────────────────────────
def test_event_id_for_is_valid_base32hex_and_deterministic() -> None:
    plan_hash = hashlib.sha256(b"plan-1").hexdigest()
    event_id = event_id_for(plan_hash)
    assert event_id == event_id_for(plan_hash) == event_id_for(plan_hash.upper())
    assert len(event_id) == 52  # 256 bits / 5 bits per character, rounded up
    assert set(event_id) <= set("0123456789abcdefghijklmnopqrstuv")
    assert is_valid_event_id(event_id)
    assert event_id != event_id_for(hashlib.sha256(b"plan-2").hexdigest())
    padded = event_id.upper() + "=" * (-len(event_id) % 8)
    assert base64.b32hexdecode(padded) == bytes.fromhex(plan_hash)  # lossless


def test_event_id_for_length_bounds() -> None:
    assert len(event_id_for("abcdef")) == 5  # 3 bytes: the shortest valid id
    assert len(event_id_for("ab" * 640)) == 1024  # 640 bytes: the longest
    for bad in ("abcd", "ab" * 641, "", "xyz1", "abc"):
        with pytest.raises(ValueError):
            event_id_for(bad)


# ─────────────────────────────── lifecycle ───────────────────────────────
async def test_injected_client_is_not_closed_but_an_owned_one_is() -> None:
    injected = FakeGoogle().client()
    async with GoogleCalendarClient(settings(), tz=ZRH, client=injected):
        pass
    assert not injected.is_closed

    owned = GoogleCalendarClient(settings(), tz=ZRH)
    await owned.aclose()
    assert owned._client.is_closed
