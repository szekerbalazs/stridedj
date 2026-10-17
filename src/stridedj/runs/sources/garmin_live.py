"""Garmin Connect via `garminconnect` >= 0.3 (unofficial, mobile SSO login). Drops GPS and names at the boundary.

How it fits together:
* `interactive_login` runs once, from the CLI, with email + password (+ MFA code). The library exchanges them for
  OAuth tokens, which are written to `GarminSettings.token_dir` (dir 0700, files 0600). The password is never stored.
* `GarminLiveSource` (the long-running bot) only ever *loads* those tokens: it is constructed without credentials,
  so a leaked bot process cannot re-login, and an expired session surfaces as `AuthError` ("run the login again").
* `garminconnect` is synchronous (requests/curl_cffi); every call runs in `asyncio.to_thread` so it never blocks the
  event loop.

Privacy: `activity_from_garmin` copies an explicit allowlist of fields out of the raw dicts. Everything else (GPS
start/end coordinates, polylines, `activityName`, `description`, `locationName`, device ids, ...) is never read, and
`Activity` (extra="forbid") could not hold it anyway.

ASSUMPTIONS TO VERIFY ON THE FIRST LIVE LOGIN (the library returns raw JSON; these names come from the library's own
typed models where available, otherwise from Garmin Connect web responses):
* activity list (`get_activities_by_date`): `activityId`, `startTimeGMT` ("YYYY-MM-DD HH:MM:SS", UTC; fallback
  `beginTimestamp` in epoch ms), `activityType.typeKey`, `distance` (m), `duration`, `elapsedDuration`,
  `movingDuration` (s), `elevationGain` (m), `averageHR`, `maxHR`, `averageRunningCadenceInStepsPerMinute`
  (confirmed by `garminconnect.typed.Activity`); `hrTimeInZone_1` .. `hrTimeInZone_5` (s) — NOT in the typed model,
  verify; when absent, zone times are fetched per activity.
* `get_activity_hr_in_timezones`: annotated as a dict by the library but expected to be a LIST of
  `{"zoneNumber": 1..5, "secsInZone": float, "zoneLowBoundary": bpm}` — verify.
* `get_activity_splits`: `{"lapDTOs": [{"startTimeGMT", "duration", "distance", "averageHR", "averageRunCadence",
  "intensityType"}]}` — verify `averageRunCadence` is steps/min (not strides/min) and the `intensityType` values
  (ACTIVE, REST, WARMUP, COOLDOWN, RECOVERY, INTERVAL).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, TypeVar

from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

from stridedj.config import GarminSettings
from stridedj.domain import Activity, ActivityId, HeartRateZones, Lap, LapIntensity
from stridedj.errors import AuthError, ConfigError, ExternalAPIError

logger = logging.getLogger(__name__)

SERVICE = "garmin"
RUNNING_TYPE_KEYS = frozenset(
    {
        "running",
        "street_running",
        "track_running",
        "trail_running",
        "treadmill_running",
        "indoor_running",
        "virtual_run",
        "ultra_run",
        "obstacle_run",
    }
)
RUNNING_PARENT_TYPE_ID = 1  # Garmin's "running" parent type
GARMIN_LIST_TYPE_FILTER = "running"  # server-side filter (includes the running sub-types)
ZONE_KEYS = tuple(f"hrTimeInZone_{zone}" for zone in range(1, 6))
_INTENSITY_BY_GARMIN = {member.name: member for member in LapIntensity}  # "ACTIVE" → LapIntensity.ACTIVE, ...

T = TypeVar("T")


class GarminApi(Protocol):
    """The subset of `garminconnect.Garmin` this adapter uses (lets tests pass a fake)."""

    def login(self, /, tokenstore: str | None = None) -> tuple[str | None, str | None]: ...

    def get_activities_by_date(
        self,
        startdate: str,
        enddate: str | None = None,
        activitytype: str | None = None,
        sortorder: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def get_activity_splits(self, activity_id: str) -> Any: ...

    def get_activity_hr_in_timezones(self, activity_id: str) -> Any: ...

    def get_heart_rate_zones(self) -> list[dict[str, Any]]: ...


class TokenStore(Protocol):
    def dump(self, path: str) -> None: ...


class GarminLoginApi(Protocol):
    """What `interactive_login` needs: a credential login and access to the library's token store."""

    @property
    def client(self) -> TokenStore: ...

    def login(self, /, tokenstore: str | None = None) -> tuple[str | None, str | None]: ...


# ─────────────────────────────── mapping (pure, unit-tested) ───────────────────────────────
def _positive(value: Any) -> float | None:
    """A strictly positive float, or None for missing / zero / negative / non-numeric values."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if value > 0 else None


def _non_negative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if value >= 0 else None


def parse_garmin_utc(value: Any) -> datetime | None:
    """Garmin GMT timestamps ("2026-10-15 16:30:12" or "2026-10-15T16:30:12.0") → aware UTC datetime."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _start_utc(summary: Mapping[str, Any]) -> datetime | None:
    start = parse_garmin_utc(summary.get("startTimeGMT"))
    if start is None and isinstance(summary.get("beginTimestamp"), int | float):
        start = datetime.fromtimestamp(summary["beginTimestamp"] / 1000.0, tz=UTC)
    return start


def is_running(summary: Mapping[str, Any]) -> bool:
    activity_type = summary.get("activityType")
    if not isinstance(activity_type, Mapping):
        return False
    return activity_type.get("typeKey") in RUNNING_TYPE_KEYS or activity_type.get("parentTypeId") == (
        RUNNING_PARENT_TYPE_ID
    )


def has_summary_zone_times(summary: Mapping[str, Any]) -> bool:
    return all(_non_negative(summary.get(key)) is not None for key in ZONE_KEYS)


def _zone_seconds(summary: Mapping[str, Any], hr_zones: Any) -> tuple[float, float, float, float, float] | None:
    """Zone seconds from the summary's `hrTimeInZone_n` keys, else from the per-activity zones endpoint."""
    if has_summary_zone_times(summary):
        z = [float(summary[key]) for key in ZONE_KEYS]
    elif isinstance(hr_zones, Sequence) and not isinstance(hr_zones, str):
        z = [0.0] * 5
        for entry in hr_zones:
            if not isinstance(entry, Mapping):
                continue
            zone, seconds = entry.get("zoneNumber"), _non_negative(entry.get("secsInZone"))
            if isinstance(zone, int) and 1 <= zone <= 5 and seconds is not None:
                z[zone - 1] += seconds
    else:
        return None
    return (z[0], z[1], z[2], z[3], z[4]) if sum(z) > 0 else None


def _intensity(value: Any) -> LapIntensity:
    return _INTENSITY_BY_GARMIN.get(value.upper(), LapIntensity.OTHER) if isinstance(value, str) else LapIntensity.OTHER


def laps_from_splits(splits: Any, activity_start: datetime) -> tuple[Lap, ...]:
    """Map `lapDTOs` to `Lap`s (allowlist: timing, distance, HR, cadence, intensity; never the lap coordinates)."""
    if not isinstance(splits, Mapping) or not isinstance(splits.get("lapDTOs"), Sequence):
        return ()
    laps: list[Lap] = []
    running_offset_s = 0.0
    for raw in splits["lapDTOs"]:
        if not isinstance(raw, Mapping):
            continue
        duration_s = _positive(raw.get("duration")) or _positive(raw.get("elapsedDuration"))
        if duration_s is None:
            continue
        lap_start = parse_garmin_utc(raw.get("startTimeGMT"))
        offset_s = max(0.0, (lap_start - activity_start).total_seconds()) if lap_start else running_offset_s
        laps.append(
            Lap(
                start_offset_s=offset_s,
                duration_s=duration_s,
                distance_m=_non_negative(raw.get("distance")) or 0.0,
                avg_hr_bpm=_positive(raw.get("averageHR")),
                avg_cadence_spm=_positive(raw.get("averageRunCadence")),
                intensity=_intensity(raw.get("intensityType")),
            )
        )
        running_offset_s = offset_s + duration_s
    return tuple(laps)


def activity_from_garmin(summary: Mapping[str, Any], *, splits: Any = None, hr_zones: Any = None) -> Activity | None:
    """Map one activity-list entry (+ optional splits and HR-zone payloads) to an `Activity`.

    Returns None for non-running activities and for entries without the fields a run needs (id, start, distance,
    duration), e.g. manually entered activities. Only the fields named below are ever read.
    """
    if not is_running(summary):
        return None
    activity_id = summary.get("activityId")
    start = _start_utc(summary)
    distance_m = _positive(summary.get("distance"))
    elapsed_s = _positive(summary.get("elapsedDuration")) or _positive(summary.get("duration"))
    if activity_id is None or start is None or distance_m is None or elapsed_s is None:
        return None
    moving_s = _positive(summary.get("movingDuration"))
    return Activity(
        activity_id=ActivityId(str(activity_id)),
        source="garmin",
        start=start,
        distance_m=distance_m,
        elapsed_s=elapsed_s,
        moving_s=min(moving_s, elapsed_s) if moving_s is not None else None,
        elevation_gain_m=_non_negative(summary.get("elevationGain")),
        avg_hr_bpm=_positive(summary.get("averageHR")),
        max_hr_bpm=_positive(summary.get("maxHR")),
        avg_cadence_spm=_positive(summary.get("averageRunningCadenceInStepsPerMinute")),
        hr_zone_s=_zone_seconds(summary, hr_zones),
        laps=laps_from_splits(splits, start),
    )


# ─────────────────────────────── the source ───────────────────────────────
SESSION_HINT = "session missing or expired; run `stridedj auth garmin`"
LOGIN_HINT = "login rejected; check GARMIN_EMAIL / GARMIN_PASSWORD and the MFA code"


def zones_from_garmin(profiles: Any) -> HeartRateZones | None:
    """Garmin's per-sport zone settings → our zones. Prefers the RUNNING profile, else DEFAULT.

    Garmin stores the *floor* (inclusive lower bound) of zones 1–5; our model stores the exclusive upper bounds of
    Z1–Z4, i.e. the floors of zones 2–5. (Below zone1Floor Garmin shows "no zone"; we count it as Z1.)
    """
    if not isinstance(profiles, list):
        return None
    by_sport = {str(p.get("sport", "")).upper(): p for p in profiles if isinstance(p, Mapping)}
    profile = by_sport.get("RUNNING") or by_sport.get("DEFAULT")
    if profile is None:
        return None
    raw = [profile.get(f"zone{i}Floor") for i in range(2, 6)]
    floors = [int(f) for f in raw if isinstance(f, int | float) and f > 0]
    if len(floors) != 4:
        return None
    try:
        return HeartRateZones(upper_bounds_bpm=(floors[0], floors[1], floors[2], floors[3]))
    except ValueError:  # not increasing: ignore rather than classify with nonsense zones
        return None


def _translate(exc: Exception, what: str, *, auth_hint: str = SESSION_HINT) -> ExternalAPIError:
    """Library exceptions → ours. Library messages are not forwarded (they may echo request details)."""
    if isinstance(exc, GarminConnectAuthenticationError):
        return AuthError(SERVICE, f"{what}: {auth_hint}")
    if isinstance(exc, GarminConnectTooManyRequestsError):
        return ExternalAPIError(SERVICE, f"{what}: rate limited", status_code=429, retryable=True)
    return ExternalAPIError(SERVICE, f"{what}: {type(exc).__name__}", retryable=True)


_LIBRARY_ERRORS = (GarminConnectAuthenticationError, GarminConnectTooManyRequestsError, GarminConnectConnectionError)


class GarminLiveSource:
    """`RunSource` over the live Garmin Connect API, authenticated with the tokens saved by `interactive_login`.

    Per activity it makes up to two extra calls (laps; zone times only when the list entry lacks them). Set
    `fetch_laps=False` for a fast first import of a long history (intervals are then typed by HR zone only).
    """

    def __init__(
        self,
        settings: GarminSettings,
        *,
        api_factory: Callable[[], GarminApi] | None = None,
        fetch_laps: bool = True,
        fetch_zone_times: bool = True,
    ) -> None:
        self._token_dir = settings.token_dir
        self._api_factory: Callable[[], GarminApi] = api_factory or Garmin  # no credentials: tokens only
        self._fetch_laps = fetch_laps
        self._fetch_zone_times = fetch_zone_times
        self._api: GarminApi | None = None
        self._login_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "garmin"

    async def _session(self) -> GarminApi:
        async with self._login_lock:
            if self._api is None:
                api = self._api_factory()
                await self._call(api.login, str(self._token_dir), what="login")
                self._api = api
            return self._api

    @staticmethod
    async def _call(fn: Callable[..., T], *args: Any, what: str) -> T:
        try:
            return await asyncio.to_thread(fn, *args)
        except _LIBRARY_ERRORS as exc:
            raise _translate(exc, what) from None

    async def _optional_detail(self, fn: Callable[[str], Any], activity_id: str, what: str) -> Any:
        """Per-activity details are best effort: a 404 or a hiccup loses laps/zones, not the whole refresh."""
        try:
            return await self._call(fn, activity_id, what=what)
        except AuthError:
            raise
        except ExternalAPIError as exc:
            if exc.status_code == 429:
                raise
            logger.warning("garmin %s unavailable for one activity; continuing without it", what)
            return None

    async def fetch_activities(self, *, since: datetime, limit: int) -> Sequence[Activity]:
        """Running activities started at or after `since`, newest first, at most `limit`."""
        api = await self._session()
        # the list endpoint filters by *local* date: start one day early, then filter exactly on the UTC start
        start_date = (since.astimezone(UTC) - timedelta(days=1)).date().isoformat()
        summaries = await self._call(
            api.get_activities_by_date, start_date, None, GARMIN_LIST_TYPE_FILTER, what="activity list"
        )
        candidates = [
            (start, summary)
            for summary in summaries
            if is_running(summary) and (start := _start_utc(summary)) is not None and start >= since
        ]
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        activities: list[Activity] = []
        for _, summary in candidates[:limit]:
            activity_id = str(summary.get("activityId", ""))
            splits = (
                await self._optional_detail(api.get_activity_splits, activity_id, "laps") if self._fetch_laps else None
            )
            zones = None
            if self._fetch_zone_times and not has_summary_zone_times(summary):
                zones = await self._optional_detail(api.get_activity_hr_in_timezones, activity_id, "HR zones")
            activity = activity_from_garmin(summary, splits=splits, hr_zones=zones)
            if activity is not None:
                activities.append(activity)
        return activities

    async def heart_rate_zones(self) -> HeartRateZones | None:
        """The zones configured in Garmin Connect (the same ones the watch uses), or None if unavailable."""
        api = await self._session()
        try:
            profiles = await self._call(api.get_heart_rate_zones, what="HR zone settings")
        except ExternalAPIError as exc:
            if isinstance(exc, AuthError):
                raise
            logger.warning("garmin HR zone settings unavailable; falling back to configured zones")
            return None
        return zones_from_garmin(profiles)


# ─────────────────────────────── one-time interactive login ───────────────────────────────
def _private_dir(path: Path) -> None:
    if path.is_symlink():
        raise ConfigError(f"Garmin token dir must not be a symlink: {path}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)  # mkdir's mode is subject to the umask and ignored for an existing dir


def _harden_files(path: Path) -> None:
    for child in path.iterdir():
        if child.is_file() and not child.is_symlink():
            child.chmod(0o600)


def interactive_login(
    settings: GarminSettings,
    prompt_mfa: Callable[[], str],
    *,
    garmin_factory: Callable[..., GarminLoginApi] | None = None,
) -> None:
    """Log in once with email/password (+ MFA via `prompt_mfa`) and persist the tokens to `settings.token_dir`.

    Always a fresh credential login (no tokenstore is passed to `login`, so stale tokens are not reused); the
    library's own `dump` then writes `garmin_tokens.json` atomically with 0600 in a 0700 directory, which is
    enforced again here in case the directory pre-existed with looser permissions.
    Note: if the `GARMINTOKENS` environment variable is set, the library loads tokens from there instead.
    """
    if settings.email is None or settings.password is None:
        raise ConfigError("GARMIN_EMAIL and GARMIN_PASSWORD must be set for the one-time Garmin login")
    token_dir = settings.token_dir
    _private_dir(token_dir)
    factory: Callable[..., GarminLoginApi] = garmin_factory or Garmin
    api = factory(
        email=settings.email.get_secret_value(),
        password=settings.password.get_secret_value(),
        prompt_mfa=prompt_mfa,
    )
    try:
        api.login()
    except _LIBRARY_ERRORS as exc:
        raise _translate(exc, "login", auth_hint=LOGIN_HINT) from None
    api.client.dump(str(token_dir))
    with contextlib.suppress(OSError):
        _harden_files(token_dir)
