from __future__ import annotations

import base64
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest
from pydantic import SecretStr

from stridedj.config import IntervalsIcuSettings
from stridedj.domain import Activity
from stridedj.errors import AuthError, ConfigError, ExternalAPIError
from stridedj.http import RetryPolicy
from stridedj.runs.sources.intervals_icu import IntervalsIcuSource, activity_from_intervals, cadence_to_spm

ZURICH = ZoneInfo("Europe/Zurich")
NOW = datetime(2026, 10, 17, 16, 0, tzinfo=UTC)
PRIVATE_VALUES = ("Lunch Run with Anna", "felt great", "47.37", "8.54", "polyline-data", "Pegasus 41")

RUN: dict[str, Any] = {
    "id": "i98765",
    "type": "Run",
    "name": "Lunch Run with Anna",
    "description": "felt great",
    "start_latlng": [47.37, 8.54],
    "end_latlng": [47.38, 8.55],
    "polyline": "polyline-data",
    "gear": {"name": "Pegasus 41"},
    "start_date": "2026-10-15T10:30:00Z",
    "start_date_local": "2026-10-15T12:30:00",
    "distance": 8040.0,
    "moving_time": 2880,
    "elapsed_time": 2950,
    "total_elevation_gain": 42.0,
    "average_heartrate": 141,
    "max_heartrate": 155,
    "average_cadence": 84.0,
    "icu_hr_zone_times": [200, 2300, 300, 80, 0, 0, 0],
}


def assert_no_private_data(activity: Activity) -> None:
    dumped = activity.model_dump_json()
    for value in PRIVATE_VALUES:
        assert value not in dumped


# ─────────────────────────────── mapping ───────────────────────────────
def test_mapping_keeps_allowlist_and_drops_gps_and_names() -> None:
    activity = activity_from_intervals(RUN, local_tz=ZURICH)
    assert activity is not None
    assert activity.activity_id == "i98765"
    assert activity.source == "intervals_icu"
    assert activity.start == datetime(2026, 10, 15, 10, 30, tzinfo=UTC)
    assert (activity.distance_m, activity.moving_s, activity.elapsed_s) == (8040.0, 2880.0, 2950.0)
    assert activity.elevation_gain_m == 42.0
    assert (activity.avg_hr_bpm, activity.max_hr_bpm) == (141.0, 155.0)
    assert activity.avg_cadence_spm == 168.0  # 84 strides/min → 168 steps/min (auto)
    assert activity.hr_zone_s == (200.0, 2300.0, 300.0, 80.0, 0.0)
    assert activity.laps == ()
    assert_no_private_data(activity)


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [(84.0, "auto", 168.0), (171.0, "auto", 171.0), (84.0, "spm", 84.0), (86.0, "strides_per_min", 172.0)],
)
def test_cadence_conversion_is_explicit(value: float, unit: Any, expected: float) -> None:
    assert cadence_to_spm(value, unit) == expected


def test_extra_zones_fold_into_z5_and_local_start_fallback() -> None:
    raw = {k: v for k, v in RUN.items() if k != "start_date"}
    raw["icu_hr_zone_times"] = [10, 20, 30, 40, 50, 60, 70]
    activity = activity_from_intervals(raw, local_tz=ZURICH)
    assert activity is not None
    assert activity.hr_zone_s == (10.0, 20.0, 30.0, 40.0, 180.0)
    assert activity.start == datetime(2026, 10, 15, 10, 30, tzinfo=UTC)  # 12:30 CEST


def test_non_runs_and_strava_stubs_are_skipped() -> None:
    assert activity_from_intervals({**RUN, "type": "Ride"}, local_tz=ZURICH) is None
    strava_stub = {"id": "i1", "type": "Run", "source": "STRAVA", "_note": "STRAVA activities are not available"}
    assert activity_from_intervals(strava_stub, local_tz=ZURICH) is None


# ─────────────────────────────── the source over a mock transport ───────────────────────────────
def make_source(
    handler: Callable[[httpx.Request], httpx.Response],
) -> IntervalsIcuSource:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://intervals.icu")
    settings = IntervalsIcuSettings(athlete_id="i12345", api_key=SecretStr("s3cr3t-key"))

    async def no_sleep(_: float) -> None:
        return None

    return IntervalsIcuSource(
        settings,
        client=client,
        now=lambda: NOW,
        local_tz=ZURICH,
        retry_policy=RetryPolicy(max_attempts=2),
        sleep=no_sleep,
    )


async def test_fetch_sends_basic_auth_and_date_window_and_filters() -> None:
    seen: list[httpx.Request] = []
    older = {**RUN, "id": "i1", "start_date": "2026-10-01T06:00:00Z"}
    newer = {**RUN, "id": "i3", "start_date": "2026-10-16T17:00:00Z"}
    ride = {**RUN, "id": "i2", "type": "Ride"}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[older, RUN, ride, newer])

    source = make_source(handler)
    assert source.name == "intervals_icu"
    activities = await source.fetch_activities(since=datetime(2026, 10, 10, tzinfo=UTC), limit=10)

    assert [a.activity_id for a in activities] == ["i3", "i98765"]  # newest first; ride and old run dropped
    (request,) = seen
    assert request.url.path == "/api/v1/athlete/i12345/activities"
    assert dict(request.url.params) == {"oldest": "2026-10-09", "newest": "2026-10-18"}
    expected = "Basic " + base64.b64encode(b"API_KEY:s3cr3t-key").decode()
    assert request.headers["authorization"] == expected
    for activity in activities:
        assert_no_private_data(activity)

    limited = await source.fetch_activities(since=datetime(2026, 10, 10, tzinfo=UTC), limit=1)
    assert [a.activity_id for a in limited] == ["i3"]


async def test_unauthorised_raises_auth_error_without_leaking_ids() -> None:
    source = make_source(lambda request: httpx.Response(401))
    with pytest.raises(AuthError) as info:
        await source.fetch_activities(since=datetime(2026, 10, 10, tzinfo=UTC), limit=10)
    assert "i12345" not in str(info.value)
    assert "s3cr3t" not in str(info.value)


async def test_unexpected_shape_raises() -> None:
    source = make_source(lambda request: httpx.Response(200, json={"error": "nope"}))
    with pytest.raises(ExternalAPIError, match="unexpected response shape"):
        await source.fetch_activities(since=datetime(2026, 10, 10, tzinfo=UTC), limit=10)


def test_missing_settings_raise_config_error() -> None:
    with pytest.raises(ConfigError):
        IntervalsIcuSource(IntervalsIcuSettings())
