from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest

from stridedj.errors import AuthError, ExternalAPIError, RateLimitedError
from stridedj.http import RetryPolicy, backoff_delay, request_with_retries, retry_after_seconds

Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    def __init__(self) -> None:
        self.sleeps: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def _client(responses: list[httpx.Response | Exception]) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.example.com"), seen


async def test_retries_503_then_succeeds() -> None:
    client, seen = _client([httpx.Response(503), httpx.Response(200, json={"ok": True})])
    rec = Recorder()
    r = await request_with_retries(client, "GET", "/x", service="svc", sleep=rec.sleep)
    assert r.json() == {"ok": True}
    assert len(seen) == 2
    assert len(rec.sleeps) == 1


async def test_honours_retry_after_seconds() -> None:
    client, _ = _client([httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200)])
    rec = Recorder()
    await request_with_retries(client, "GET", "/x", service="svc", sleep=rec.sleep)
    assert rec.sleeps == [7.0]


async def test_a_long_retry_after_stops_at_once_instead_of_retrying_early() -> None:
    """A Retry-After of many hours: retrying after 20 s would only hit a blocked API."""
    client, seen = _client([httpx.Response(429, headers={"Retry-After": "60000"}), httpx.Response(200)])
    rec = Recorder()
    with pytest.raises(RateLimitedError) as info:
        await request_with_retries(client, "GET", "/x", service="svc", sleep=rec.sleep)
    assert info.value.retry_after_s == 60000
    assert len(seen) == 1 and rec.sleeps == []


async def test_does_not_retry_400_and_raises() -> None:
    client, seen = _client([httpx.Response(400)])
    with pytest.raises(ExternalAPIError) as info:
        await request_with_retries(client, "GET", "/x", service="svc", sleep=Recorder().sleep)
    assert info.value.status_code == 400
    assert len(seen) == 1


async def test_401_raises_auth_error() -> None:
    client, _ = _client([httpx.Response(401)])
    with pytest.raises(AuthError):
        await request_with_retries(client, "GET", "/x", service="svc", sleep=Recorder().sleep)


async def test_post_is_not_retried_unless_marked_safe() -> None:
    client, seen = _client([httpx.Response(503)])
    with pytest.raises(ExternalAPIError):
        await request_with_retries(client, "POST", "/x", service="svc", sleep=Recorder().sleep)
    assert len(seen) == 1

    client, seen = _client([httpx.Response(503), httpx.Response(201)])
    r = await request_with_retries(client, "POST", "/x", service="svc", retry_unsafe=True, sleep=Recorder().sleep)
    assert r.status_code == 201


async def test_gives_up_after_max_attempts() -> None:
    client, seen = _client([httpx.Response(500)] * 3)
    with pytest.raises(ExternalAPIError) as info:
        await request_with_retries(
            client, "GET", "/x", service="svc", policy=RetryPolicy(max_attempts=3), sleep=Recorder().sleep
        )
    assert len(seen) == 3
    assert info.value.retryable


async def test_transport_errors_are_retried_and_label_hides_secret_path() -> None:
    client, seen = _client([httpx.ConnectError("boom"), httpx.ConnectError("boom")])
    with pytest.raises(ExternalAPIError) as info:
        await request_with_retries(
            client,
            "GET",
            "/botSECRET-TOKEN/getMe",
            service="telegram",
            label="getMe",
            policy=RetryPolicy(max_attempts=2),
            sleep=Recorder().sleep,
        )
    assert len(seen) == 2
    assert "SECRET" not in str(info.value)
    assert info.value.__cause__ is None


def test_retry_after_http_date() -> None:
    now = datetime(2026, 10, 17, 12, 0, 0, tzinfo=UTC)
    assert retry_after_seconds({"retry-after": "Sat, 17 Oct 2026 12:00:30 GMT"}, now=now) == 30
    assert retry_after_seconds({}, now=now) is None
    assert retry_after_seconds({"retry-after": "soon"}, now=now) is None


def test_backoff_is_capped() -> None:
    policy = RetryPolicy(base_delay_s=1, max_delay_s=5)
    assert all(0 <= backoff_delay(attempt, policy) <= 5 for attempt in range(10))


async def test_google_style_403_rate_limit_is_retried() -> None:
    body = {"error": {"code": 403, "errors": [{"reason": "rateLimitExceeded"}]}}
    client, seen = _client([httpx.Response(403, json=body), httpx.Response(200)])
    r = await request_with_retries(client, "GET", "/x", service="google", sleep=Recorder().sleep)
    assert r.status_code == 200
    assert len(seen) == 2
