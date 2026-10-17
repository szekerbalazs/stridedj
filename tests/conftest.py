"""Shared fixtures. Unit tests never read the real `.env` and never touch the network."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Point every settings class at an empty env file and clear StrideDJ-related environment variables."""
    import os

    prefixes = ("STRIDEDJ_", "OPENAI_", "TELEGRAM_", "SPOTIFY_", "GOOGLE_", "GARMIN_", "INTERVALS_ICU_", "ATHLETE_")
    for key in list(os.environ):
        if key.startswith(prefixes):
            monkeypatch.delenv(key, raising=False)
    empty = tmp_path / "empty.env"
    empty.write_text("")
    from stridedj import config

    for cls in (
        config.AppSettings,
        config.OpenAISettings,
        config.TelegramSettings,
        config.SpotifySettings,
        config.GoogleSettings,
        config.GarminSettings,
        config.IntervalsIcuSettings,
        config.AthleteSettings,
    ):
        monkeypatch.setitem(cls.model_config, "env_file", empty)
    monkeypatch.setenv("GARMIN_TOKEN_DIR", str(tmp_path / "garmin-tokens"))
    yield
