"""Typed configuration loaded from environment variables and `.env`.

* Every secret is a `SecretStr`: `repr()`/logging shows `**********`; the value is only read via `.get_secret_value()`
  at the single place that needs it (the adapter that talks to that API).
* One settings class per integration, each with its own env prefix, so adapters receive only the settings they need
  (least privilege inside the process, and easy to fake in tests).
* `load_settings()` is called once, in the composition root (`app.py`). Nothing else reads the environment.
"""

from __future__ import annotations

import os
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = Path(os.environ.get("STRIDEDJ_ENV_FILE", PROJECT_ROOT / ".env"))

RunSourceName = Literal["demo", "garmin", "intervals_icu"]


def _split_csv(value: object) -> object:
    """Parse "a, b ,c" → ["a", "b", "c"] (an empty string → [])."""
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


class _EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore", frozen=True)


class AppSettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="STRIDEDJ_")

    run_source: RunSourceName = "demo"
    data_dir: Path = Field(default=PROJECT_ROOT / "data", description="SQLite DB, caches and traces (git-ignored)")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("data_dir")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return v if v.is_absolute() else PROJECT_ROOT / v


class OpenAISettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_")

    api_key: SecretStr | None = None
    model: str = "gpt-5-mini"
    request_timeout_s: float = Field(default=60.0, gt=0)


class TelegramSettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="TELEGRAM_")

    bot_token: SecretStr | None = None
    allowed_user_ids: Annotated[list[int], NoDecode] = Field(default_factory=list)

    @field_validator("allowed_user_ids", mode="before")
    @classmethod
    def _split(cls, v: object) -> object:
        return _split_csv(v)


class SpotifySettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="SPOTIFY_")

    client_id: str | None = None  # public identifier: PKCE needs no client secret
    redirect_uri: str = "http://127.0.0.1:8888/callback"
    running_playlist_ids: Annotated[list[str], NoDecode] = Field(default_factory=list)
    refresh_token: SecretStr | None = None

    @field_validator("running_playlist_ids", mode="before")
    @classmethod
    def _split(cls, v: object) -> object:
        return _split_csv(v)


class GoogleSettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="GOOGLE_")

    client_id: str | None = None
    client_secret: SecretStr | None = None
    refresh_token: SecretStr | None = None
    calendar_id: str = "primary"
    busy_calendar_ids: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["primary"])

    @field_validator("busy_calendar_ids", mode="before")
    @classmethod
    def _split(cls, v: object) -> object:
        return _split_csv(v)


class GarminSettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="GARMIN_")

    email: SecretStr | None = None  # PII: treated like a secret so it never shows up in logs
    password: SecretStr | None = None
    token_dir: Path = Path("~/.garminconnect")

    @field_validator("token_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()


class IntervalsIcuSettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="INTERVALS_ICU_")

    athlete_id: str | None = None
    api_key: SecretStr | None = None


class AthleteSettings(_EnvSettings):
    model_config = SettingsConfigDict(env_prefix="ATHLETE_")

    hr_zone_upper_bounds: Annotated[tuple[int, int, int, int] | None, NoDecode] = Field(
        default=None, description="Exclusive upper bounds (bpm) of Z1..Z4, as configured on the watch"
    )
    max_hr: int | None = Field(default=None, ge=120, le=230)
    timezone: str = "Europe/Zurich"

    @field_validator("hr_zone_upper_bounds", mode="before")
    @classmethod
    def _parse_bounds(cls, v: object) -> object:
        if isinstance(v, str):
            parts = _split_csv(v)
            return tuple(int(p) for p in parts) if isinstance(parts, list) and parts else None
        return v

    @field_validator("hr_zone_upper_bounds")
    @classmethod
    def _increasing(cls, v: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
        if v is not None and any(a >= b for a, b in pairwise(v)):
            raise ValueError("HR zone bounds must be strictly increasing (Z1 < Z2 < Z3 < Z4)")
        return v

    @field_validator("timezone")
    @classmethod
    def _valid_tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone {v!r}") from exc
        return v


class Settings(BaseModel):
    """All settings, grouped per integration. Built once by `load_settings()`."""

    model_config = ConfigDict(frozen=True)

    app: AppSettings
    openai: OpenAISettings
    telegram: TelegramSettings
    spotify: SpotifySettings
    google: GoogleSettings
    garmin: GarminSettings
    intervals_icu: IntervalsIcuSettings
    athlete: AthleteSettings


def load_settings() -> Settings:
    """Read every settings group from the environment and `.env` (environment variables win)."""
    return Settings(
        app=AppSettings(),
        openai=OpenAISettings(),
        telegram=TelegramSettings(),
        spotify=SpotifySettings(),
        google=GoogleSettings(),
        garmin=GarminSettings(),
        intervals_icu=IntervalsIcuSettings(),
        athlete=AthleteSettings(),
    )
