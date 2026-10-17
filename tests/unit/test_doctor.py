from __future__ import annotations

import httpx
import pytest

from stridedj.config import load_settings
from stridedj.doctor import CheckStatus, run_checks


async def test_offline_reports_missing_without_network() -> None:
    results = await run_checks(load_settings(), live=False)
    by_name = {r.name: r for r in results}
    assert by_name["openai: api key"].status is CheckStatus.MISSING
    assert by_name["spotify: login"].status is CheckStatus.PENDING
    assert by_name["run source"].detail == "demo"


async def test_live_checks_never_print_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-zqx7SENTINEL1")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "999:zqx7SENTINEL2")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.openai.com":
            assert request.headers["Authorization"] == "Bearer sk-zqx7SENTINEL1"
            return httpx.Response(200, json={"id": "gpt-5-mini"})
        return httpx.Response(200, json={"ok": True, "result": {"username": "stride_bot"}})

    results = await run_checks(
        load_settings(), live=True, client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    by_name = {r.name: r for r in results}
    assert by_name["openai: api key"].status is CheckStatus.OK
    assert by_name["telegram: bot token"].detail == "bot @stride_bot"
    assert "zqx7" not in " ".join(r.detail for r in results)


async def test_rejected_keys_are_reported_as_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-bad")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:bad")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    results = await run_checks(
        load_settings(), live=True, client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    by_name = {r.name: r for r in results}
    assert by_name["openai: api key"].status is CheckStatus.FAILED
    assert by_name["telegram: bot token"].status is CheckStatus.FAILED
    assert "bad" not in by_name["telegram: bot token"].detail
