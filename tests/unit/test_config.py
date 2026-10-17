from __future__ import annotations

import pytest
from pydantic import ValidationError

from stridedj.config import AthleteSettings, TelegramSettings, load_settings


def test_defaults_load_without_any_env() -> None:
    s = load_settings()
    assert s.app.run_source == "demo"
    assert s.openai.api_key is None
    assert s.google.busy_calendar_ids == ["primary"]


def test_secrets_are_masked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-very-secret")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC-secret")
    s = load_settings()
    dumped = repr(s) + str(s.model_dump())
    assert "very-secret" not in dumped
    assert "ABC-secret" not in dumped
    assert s.openai.api_key is not None
    assert s.openai.api_key.get_secret_value() == "sk-test-very-secret"


def test_comma_separated_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", " 123, 456 ,")
    monkeypatch.setenv("SPOTIFY_RUNNING_PLAYLIST_IDS", "abc,def")
    s = load_settings()
    assert s.telegram.allowed_user_ids == [123, 456]
    assert s.spotify.running_playlist_ids == ["abc", "def"]


def test_empty_list_value_gives_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "")
    assert TelegramSettings().allowed_user_ids == []


def test_hr_zone_bounds_parse_and_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHLETE_HR_ZONE_UPPER_BOUNDS", "125,145,160,175")
    assert AthleteSettings().hr_zone_upper_bounds == (125, 145, 160, 175)
    monkeypatch.setenv("ATHLETE_HR_ZONE_UPPER_BOUNDS", "125,145,140,175")
    with pytest.raises(ValidationError, match="strictly increasing"):
        AthleteSettings()


def test_invalid_timezone_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHLETE_TIMEZONE", "Mars/Olympus")
    with pytest.raises(ValidationError, match="unknown IANA timezone"):
        AthleteSettings()
