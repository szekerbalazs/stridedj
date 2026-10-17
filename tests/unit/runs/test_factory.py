from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import SecretStr

from stridedj.config import (
    AppSettings,
    AthleteSettings,
    GarminSettings,
    GoogleSettings,
    IntervalsIcuSettings,
    OpenAISettings,
    Settings,
    SpotifySettings,
    TelegramSettings,
)
from stridedj.errors import ConfigError
from stridedj.runs.sources.demo import DemoRunSource
from stridedj.runs.sources.factory import build_run_source
from stridedj.runs.sources.garmin_live import GarminLiveSource
from stridedj.runs.sources.intervals_icu import IntervalsIcuSource


def settings(run_source: str, intervals: IntervalsIcuSettings | None = None) -> Settings:
    return Settings(
        app=AppSettings(run_source=run_source),  # type: ignore[arg-type]
        openai=OpenAISettings(),
        telegram=TelegramSettings(),
        spotify=SpotifySettings(),
        google=GoogleSettings(),
        garmin=GarminSettings(),
        intervals_icu=intervals or IntervalsIcuSettings(),
        athlete=AthleteSettings(),
    )


def test_selects_the_configured_source() -> None:
    now = datetime(2026, 10, 17, 16, 0, tzinfo=UTC)
    assert isinstance(build_run_source(settings("demo"), now=lambda: now), DemoRunSource)
    assert isinstance(build_run_source(settings("garmin")), GarminLiveSource)
    icu = IntervalsIcuSettings(athlete_id="i1", api_key=SecretStr("k"))
    assert isinstance(build_run_source(settings("intervals_icu", icu)), IntervalsIcuSource)


def test_intervals_without_credentials_fails_fast() -> None:
    with pytest.raises(ConfigError):
        build_run_source(settings("intervals_icu"))
