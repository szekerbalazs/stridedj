from __future__ import annotations

from types import SimpleNamespace

import pytest
from telegram.error import Conflict, NetworkError

from stridedj.bot.telegram_bot import TelegramBot
from stridedj.clock import SystemClock
from stridedj.harness.guard import InputGuard


def _bot() -> TelegramBot:
    async def factory():  # never called in these tests
        raise AssertionError

    return TelegramBot(
        token="123456:TEST", guard=InputGuard(allowed_user_ids={1}, clock=SystemClock()), app_factory=factory
    )


class FakeApplication:
    def __init__(self) -> None:
        self.stopped = False

    def stop_running(self) -> None:
        self.stopped = True


async def test_second_instance_conflict_stops_cleanly(caplog: pytest.LogCaptureFixture) -> None:
    app = FakeApplication()
    await _bot()._on_error(
        None, SimpleNamespace(error=Conflict("terminated by other getUpdates request"), application=app)
    )  # type: ignore[arg-type]
    assert app.stopped
    assert "Another copy of this bot is already running" in caplog.text


async def test_network_errors_are_retried_not_fatal(caplog: pytest.LogCaptureFixture) -> None:
    app = FakeApplication()
    await _bot()._on_error(None, SimpleNamespace(error=NetworkError("timeout"), application=app))  # type: ignore[arg-type]
    assert not app.stopped
    assert "retrying automatically" in caplog.text
