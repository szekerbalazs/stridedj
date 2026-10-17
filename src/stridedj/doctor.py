"""`stridedj doctor`: is every integration configured, and do the credentials actually work?

Prints one line per check with ✓ / ✗ / … and a hint. It never prints a secret, not even partially: a "masked" key
(`sk-...abcd`) still leaks information and trains people to paste screenshots of it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum

import httpx
from pydantic import BaseModel, ConfigDict

from stridedj.config import Settings
from stridedj.errors import AuthError, ExternalAPIError
from stridedj.http import RetryPolicy, build_client, request_with_retries

ClientFactory = Callable[[], httpx.AsyncClient]
_NO_RETRY = RetryPolicy(max_attempts=1)


class CheckStatus(StrEnum):
    OK = "ok"
    MISSING = "missing"
    FAILED = "failed"
    PENDING = "pending"  # configured, but the live check needs a later step (e.g. an OAuth login)


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    status: CheckStatus
    detail: str


def _present(name: str, ok: bool, hint: str, ok_detail: str = "set") -> CheckResult:
    return CheckResult(
        name=name, status=CheckStatus.OK if ok else CheckStatus.MISSING, detail=ok_detail if ok else hint
    )


async def _check_openai(settings: Settings, client_factory: ClientFactory) -> CheckResult:
    key = settings.openai.api_key
    if key is None:
        return _present("openai: api key", False, "set OPENAI_API_KEY (docs/SETUP.md §1)")
    model = settings.openai.model
    async with client_factory() as client:
        try:
            await request_with_retries(
                client,
                "GET",
                f"https://api.openai.com/v1/models/{model}",
                service="openai",
                policy=_NO_RETRY,
                headers={"Authorization": f"Bearer {key.get_secret_value()}"},
            )
        except AuthError:
            return CheckResult(name="openai: api key", status=CheckStatus.FAILED, detail="key rejected (401/403)")
        except ExternalAPIError as exc:
            if exc.status_code == 404:
                return CheckResult(
                    name="openai: api key",
                    status=CheckStatus.FAILED,
                    detail=f"key works, but model {model!r} not found",
                )
            return CheckResult(name="openai: api key", status=CheckStatus.FAILED, detail=str(exc))
    return CheckResult(name="openai: api key", status=CheckStatus.OK, detail=f"key valid, model {model!r} available")


async def _check_telegram(settings: Settings, client_factory: ClientFactory) -> CheckResult:
    token = settings.telegram.bot_token
    if token is None:
        return _present("telegram: bot token", False, "set TELEGRAM_BOT_TOKEN (docs/SETUP.md §2)")
    async with client_factory() as client:
        try:
            response = await request_with_retries(
                client,
                "GET",
                f"https://api.telegram.org/bot{token.get_secret_value()}/getMe",
                service="telegram",
                policy=_NO_RETRY,
                label="getMe",  # the token is part of the URL path: never put the path into an error message
            )
        except ExternalAPIError as exc:
            status = f"HTTP {exc.status_code}" if exc.status_code else "network error"
            return CheckResult(name="telegram: bot token", status=CheckStatus.FAILED, detail=f"getMe failed ({status})")
    username = response.json().get("result", {}).get("username", "?")
    return CheckResult(name="telegram: bot token", status=CheckStatus.OK, detail=f"bot @{username}")


def _offline_checks(settings: Settings) -> list[CheckResult]:
    s = settings
    results = [
        _present(
            "telegram: allowed users",
            bool(s.telegram.allowed_user_ids),
            "set TELEGRAM_ALLOWED_USER_IDS (your numeric id, via @userinfobot)",
            ok_detail=f"{len(s.telegram.allowed_user_ids)} user(s)",
        ),
        _present("spotify: client id", s.spotify.client_id is not None, "set SPOTIFY_CLIENT_ID (docs/SETUP.md §3)"),
        _present(
            "spotify: running playlists",
            bool(s.spotify.running_playlist_ids),
            "set SPOTIFY_RUNNING_PLAYLIST_IDS (playlists you own)",
            ok_detail=f"{len(s.spotify.running_playlist_ids)} playlist(s)",
        ),
        _present(
            "google: oauth client",
            s.google.client_id is not None and s.google.client_secret is not None,
            "set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET (docs/SETUP.md §4)",
        ),
        _present(
            "garmin: login",
            s.garmin.email is not None and s.garmin.password is not None,
            "set GARMIN_EMAIL and GARMIN_PASSWORD (docs/SETUP.md §5)",
        ),
        _present(
            "athlete: HR zones",
            s.athlete.hr_zone_upper_bounds is not None,
            "set ATHLETE_HR_ZONE_UPPER_BOUNDS from your watch's zones (docs/SETUP.md §5)",
            ok_detail="Z1..Z4 upper bounds set",
        ),
    ]
    # OAuth refresh tokens are produced by the login commands (later steps); report them as pending, not missing.
    for name, value, command in (
        ("spotify: login", s.spotify.refresh_token, "stridedj auth spotify"),
        ("google: login", s.google.refresh_token, "stridedj auth google"),
    ):
        results.append(
            CheckResult(
                name=name,
                status=CheckStatus.OK if value is not None else CheckStatus.PENDING,
                detail="refresh token stored" if value is not None else f"run `{command}` once the client is set up",
            )
        )
    token_file_present = any(s.garmin.token_dir.glob("*.json")) if s.garmin.token_dir.is_dir() else False
    results.append(
        CheckResult(
            name="garmin: session",
            status=CheckStatus.OK if token_file_present else CheckStatus.PENDING,
            detail="token store found" if token_file_present else "run `stridedj auth garmin` (handles MFA)",
        )
    )
    return results


async def run_checks(
    settings: Settings, *, live: bool = True, client_factory: ClientFactory | None = None
) -> Sequence[CheckResult]:
    """Run all checks. `live=False` only checks presence (no network)."""
    factory = client_factory or (lambda: build_client(timeout_s=10.0))
    results: list[CheckResult] = []
    if live:
        results.append(await _check_openai(settings, factory))
        results.append(await _check_telegram(settings, factory))
    else:
        results.append(_present("openai: api key", settings.openai.api_key is not None, "set OPENAI_API_KEY"))
        results.append(
            _present("telegram: bot token", settings.telegram.bot_token is not None, "set TELEGRAM_BOT_TOKEN")
        )
    results.extend(_offline_checks(settings))
    results.append(CheckResult(name="run source", status=CheckStatus.OK, detail=settings.app.run_source))
    return results
