"""Plan B: the intervals.icu REST API (Garmin syncs to it officially; personal API key, Basic auth).

Why a plan B: `garminconnect` is unofficial and Garmin has broken third-party logins before (March 2026). Garmin's
*official* sync to intervals.icu keeps working through such changes, and intervals.icu has a documented API with a
personal key. Swapping the `RunSource` is a config change (`STRIDEDJ_RUN_SOURCE=intervals_icu`), nothing else moves.

API: `GET https://intervals.icu/api/v1/athlete/{athlete_id}/activities?oldest=YYYY-MM-DD&newest=YYYY-MM-DD` with HTTP
Basic auth, username literally "API_KEY" and the key as password.

Privacy: as for Garmin, an explicit allowlist (`activity_from_intervals`); `name`, `description`, `start_latlng`,
`polyline`, gear, ... are never read.

ASSUMPTIONS TO VERIFY ON THE FIRST LIVE CALL:
* field names: `id`, `type` ("Run" / "VirtualRun" / "TrailRun"), `start_date` (UTC, "...Z"), `start_date_local`
  (naive local), `distance` (m), `moving_time`, `elapsed_time` (s), `total_elevation_gain` (m),
  `average_heartrate`, `max_heartrate`, `average_cadence`, `icu_hr_zone_times` (list of seconds per zone).
* CADENCE UNIT: for runs, `average_cadence` may be strides/min (one foot, as stored in FIT files: ~85) rather than
  steps/min (~170). `cadence_unit="auto"` doubles values below 120 (no runner's step rate is that low, no stride
  rate that high); pin it to "spm" or "strides_per_min" once verified.
* ZONES: `icu_hr_zone_times` has one entry per zone configured *on intervals.icu*, often 7 (Friel: Z1–Z4, Z5a/b/c).
  Zones beyond the 5th are folded into Z5; check the zone bounds match the watch's.
* activities imported from Strava are returned without data (Strava's API terms): they lack the fields and are
  skipped; this source needs the direct Garmin → intervals.icu sync.
* laps are not part of the list response (they need a per-activity `/intervals` call, not implemented): interval
  sessions are therefore typed by HR zone time only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

import httpx

from stridedj.config import IntervalsIcuSettings
from stridedj.domain import Activity, ActivityId
from stridedj.errors import ConfigError, ExternalAPIError
from stridedj.http import RetryPolicy, Sleep, build_client, request_with_retries

SERVICE = "intervals.icu"
BASE_URL = "https://intervals.icu"
BASIC_AUTH_USER = "API_KEY"
RUN_TYPES = frozenset({"Run", "VirtualRun", "TrailRun"})
MIN_PLAUSIBLE_STEP_RATE_SPM = 120.0

CadenceUnit = Literal["auto", "spm", "strides_per_min"]


def _positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if value > 0 else None


def _non_negative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if value >= 0 else None


def cadence_to_spm(value: float, unit: CadenceUnit) -> float:
    """Convert intervals.icu's run cadence to steps per minute (see the module docstring: verify live)."""
    if unit == "spm":
        return value
    if unit == "strides_per_min":
        return 2.0 * value
    return 2.0 * value if value < MIN_PLAUSIBLE_STEP_RATE_SPM else value


def _start(raw: Mapping[str, Any], local_tz: ZoneInfo) -> datetime | None:
    """`start_date` is UTC; `start_date_local` (naive wall-clock time) is the fallback, localised with `local_tz`."""
    for key, tz in (("start_date", UTC), ("start_date_local", local_tz)):
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            continue
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            continue
        aware = parsed.replace(tzinfo=tz) if parsed.tzinfo is None else parsed
        return aware.astimezone(UTC)
    return None


def _zone_seconds(value: Any) -> tuple[float, float, float, float, float] | None:
    """Per-zone seconds; zones beyond the 5th (e.g. Friel's Z5a/b/c) are folded into Z5."""
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        return None
    seconds = [_non_negative(v) or 0.0 for v in value]
    seconds += [0.0] * (5 - len(seconds))
    z = (seconds[0], seconds[1], seconds[2], seconds[3], sum(seconds[4:]))
    return z if sum(z) > 0 else None


def activity_from_intervals(
    raw: Mapping[str, Any], *, local_tz: ZoneInfo, cadence_unit: CadenceUnit = "auto"
) -> Activity | None:
    """Map one intervals.icu activity to an `Activity`; None for non-runs and entries without the essentials."""
    if raw.get("type") not in RUN_TYPES:
        return None
    activity_id = raw.get("id")
    start = _start(raw, local_tz)
    distance_m = _positive(raw.get("distance"))
    moving_s = _positive(raw.get("moving_time"))
    elapsed_s = _positive(raw.get("elapsed_time")) or moving_s
    if activity_id is None or start is None or distance_m is None or elapsed_s is None:
        return None
    cadence = _positive(raw.get("average_cadence"))
    return Activity(
        activity_id=ActivityId(str(activity_id)),
        source="intervals_icu",
        start=start,
        distance_m=distance_m,
        elapsed_s=elapsed_s,
        moving_s=min(moving_s, elapsed_s) if moving_s is not None else None,
        elevation_gain_m=_non_negative(raw.get("total_elevation_gain")),
        avg_hr_bpm=_positive(raw.get("average_heartrate")),
        max_hr_bpm=_positive(raw.get("max_heartrate")),
        avg_cadence_spm=cadence_to_spm(cadence, cadence_unit) if cadence is not None else None,
        hr_zone_s=_zone_seconds(raw.get("icu_hr_zone_times")),
    )


def _system_now() -> datetime:
    return datetime.now(UTC)


class IntervalsIcuSource:
    """`RunSource` over the intervals.icu activities endpoint."""

    def __init__(
        self,
        settings: IntervalsIcuSettings,
        *,
        client: httpx.AsyncClient | None = None,
        now: Callable[[], datetime] = _system_now,
        local_tz: ZoneInfo | None = None,
        cadence_unit: CadenceUnit = "auto",
        retry_policy: RetryPolicy | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if settings.athlete_id is None or settings.api_key is None:
            raise ConfigError("INTERVALS_ICU_ATHLETE_ID and INTERVALS_ICU_API_KEY must be set")
        self._athlete_id = settings.athlete_id
        self._auth = httpx.BasicAuth(BASIC_AUTH_USER, settings.api_key.get_secret_value())
        self._client = client
        self._now = now
        self._local_tz = local_tz or ZoneInfo("Europe/Zurich")
        self._cadence_unit: CadenceUnit = cadence_unit
        self._policy = retry_policy
        self._sleep = sleep

    @property
    def name(self) -> str:
        return "intervals_icu"

    async def _get_activities(self, client: httpx.AsyncClient, params: Mapping[str, str]) -> Any:
        response = await request_with_retries(
            client,
            "GET",
            f"/api/v1/athlete/{self._athlete_id}/activities",
            service=SERVICE,
            policy=self._policy,
            label="GET /api/v1/athlete/{id}/activities",  # keep the athlete id out of error messages
            sleep=self._sleep,
            auth=self._auth,
            params=dict(params),
        )
        return response.json()

    async def fetch_activities(self, *, since: datetime, limit: int) -> Sequence[Activity]:
        """Running activities started at or after `since`, newest first, at most `limit`."""
        # `oldest`/`newest` are local calendar dates: widen by a day on each side, then filter exactly on UTC start
        params = {
            "oldest": (since.astimezone(self._local_tz) - timedelta(days=1)).date().isoformat(),
            "newest": (self._now().astimezone(self._local_tz) + timedelta(days=1)).date().isoformat(),
        }
        if self._client is not None:
            payload = await self._get_activities(self._client, params)
        else:
            async with build_client(base_url=BASE_URL) as client:
                payload = await self._get_activities(client, params)
        if not isinstance(payload, list):
            raise ExternalAPIError(SERVICE, "activities: unexpected response shape")
        activities = [
            activity
            for raw in payload
            if isinstance(raw, Mapping)
            and (activity := activity_from_intervals(raw, local_tz=self._local_tz, cadence_unit=self._cadence_unit))
            is not None
            and activity.start >= since
        ]
        activities.sort(key=lambda a: a.start, reverse=True)
        return activities[:limit]
