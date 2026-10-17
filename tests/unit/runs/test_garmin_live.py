"""Garmin adapter tests: hand-written payloads only, the real API is never called."""

from __future__ import annotations

import copy
import json
import stat
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from garminconnect import GarminConnectAuthenticationError, GarminConnectConnectionError
from pydantic import SecretStr

from stridedj.config import GarminSettings
from stridedj.domain import Activity, LapIntensity
from stridedj.errors import AuthError, ConfigError
from stridedj.runs.sources.garmin_live import (
    GarminLiveSource,
    activity_from_garmin,
    interactive_login,
    laps_from_splits,
)

PRIVATE_VALUES = ("Morning Run by the Lake", "secret notes", "Zürich", "47.3769", "8.5417", "encodedpolyline")

SUMMARY: dict[str, Any] = {
    "activityId": 17234567890,
    "activityName": "Morning Run by the Lake",
    "description": "secret notes",
    "locationName": "Zürich",
    "startLatitude": 47.3769,
    "startLongitude": 8.5417,
    "endLatitude": 47.3770,
    "endLongitude": 8.5418,
    "summaryPolyline": "encodedpolyline",
    "ownerFullName": "Jane Runner",
    "deviceId": 3_400_000_001,
    "startTimeLocal": "2026-10-15 18:30:12",
    "startTimeGMT": "2026-10-15 16:30:12",
    "beginTimestamp": 1_792_082_412_000,
    "activityType": {"typeId": 1, "typeKey": "running", "parentTypeId": 17},
    "distance": 10012.3,
    "duration": 3601.5,
    "elapsedDuration": 3700.0,
    "movingDuration": 3590.2,
    "elevationGain": 54.0,
    "averageHR": 139.0,
    "maxHR": 152.0,
    "averageRunningCadenceInStepsPerMinute": 167.8,
    "hrTimeInZone_1": 300.0,
    "hrTimeInZone_2": 2800.0,
    "hrTimeInZone_3": 400.0,
    "hrTimeInZone_4": 90.0,
    "hrTimeInZone_5": 0.0,
}

SPLITS: dict[str, Any] = {
    "activityId": 17234567890,
    "lapDTOs": [
        {
            "startTimeGMT": "2026-10-15T16:30:12.0",
            "startLatitude": 47.3769,
            "startLongitude": 8.5417,
            "distance": 2000.0,
            "duration": 720.0,
            "averageHR": 131.0,
            "averageRunCadence": 165.0,
            "intensityType": "WARMUP",
        },
        {
            "startTimeGMT": "2026-10-15T16:42:12.0",
            "distance": 800.0,
            "duration": 200.0,
            "averageHR": 168.0,
            "averageRunCadence": 186.0,
            "intensityType": "INTERVAL",
        },
        {"startTimeGMT": "2026-10-15T16:45:32.0", "distance": 0.0, "duration": 90.0, "intensityType": "REST"},
        {"distance": 0.0, "duration": 0.0, "intensityType": "ACTIVE"},  # zero-length: dropped
        {"distance": 800.0, "duration": 198.0, "averageRunCadence": 187.0, "intensityType": "SOMETHING_NEW"},
    ],
}


def assert_no_private_data(activity: Activity) -> None:
    dumped = activity.model_dump_json()
    for value in PRIVATE_VALUES:
        assert value not in dumped
    assert set(activity.model_dump()) == set(Activity.model_fields)


# ─────────────────────────────── mapping ───────────────────────────────
def test_mapping_keeps_allowlisted_fields_and_drops_gps_and_names() -> None:
    activity = activity_from_garmin(SUMMARY, splits=SPLITS)
    assert activity is not None
    assert activity.activity_id == "17234567890"
    assert activity.source == "garmin"
    assert activity.start == datetime(2026, 10, 15, 16, 30, 12, tzinfo=UTC)
    assert activity.distance_m == 10012.3
    assert activity.elapsed_s == 3700.0
    assert activity.moving_s == 3590.2
    assert activity.elevation_gain_m == 54.0
    assert (activity.avg_hr_bpm, activity.max_hr_bpm, activity.avg_cadence_spm) == (139.0, 152.0, 167.8)
    assert activity.hr_zone_s == (300.0, 2800.0, 400.0, 90.0, 0.0)
    assert_no_private_data(activity)


def test_laps_from_splits_maps_intensity_and_offsets() -> None:
    start = datetime(2026, 10, 15, 16, 30, 12, tzinfo=UTC)
    laps = laps_from_splits(SPLITS, start)
    assert [lap.intensity for lap in laps] == [
        LapIntensity.WARMUP,
        LapIntensity.INTERVAL,
        LapIntensity.REST,
        LapIntensity.OTHER,
    ]
    assert [lap.start_offset_s for lap in laps] == [0.0, 720.0, 920.0, 1010.0]  # last one: cumulative fallback
    assert laps[1].avg_cadence_spm == 186.0
    assert laps[2].avg_cadence_spm is None
    assert laps_from_splits(None, start) == ()
    assert laps_from_splits({"lapDTOs": "nope"}, start) == ()


def test_zone_times_fall_back_to_the_zones_endpoint() -> None:
    summary = {k: v for k, v in SUMMARY.items() if not k.startswith("hrTimeInZone")}
    zones = [
        {"zoneNumber": 1, "secsInZone": 100.5, "zoneLowBoundary": 98},
        {"zoneNumber": 2, "secsInZone": 3000.0, "zoneLowBoundary": 125},
        {"zoneNumber": 3, "secsInZone": 200.0, "zoneLowBoundary": 145},
        {"zoneNumber": 7, "secsInZone": 999.0},  # out of range: ignored
    ]
    activity = activity_from_garmin(summary, hr_zones=zones)
    assert activity is not None
    assert activity.hr_zone_s == (100.5, 3000.0, 200.0, 0.0, 0.0)
    no_zones = activity_from_garmin(summary)
    assert no_zones is not None
    assert no_zones.hr_zone_s is None


def test_non_runs_and_incomplete_entries_are_skipped() -> None:
    ride = {**SUMMARY, "activityType": {"typeKey": "cycling", "parentTypeId": 2}}
    assert activity_from_garmin(ride) is None
    for missing in ("activityId", "distance"):
        assert activity_from_garmin({k: v for k, v in SUMMARY.items() if k != missing}) is None
    treadmill = {**SUMMARY, "activityType": {"typeKey": "treadmill_running", "parentTypeId": 1}}
    assert activity_from_garmin(treadmill) is not None


def test_moving_time_is_clamped_and_bad_numbers_become_none() -> None:
    weird = {**SUMMARY, "movingDuration": 9999.0, "averageHR": 0, "maxHR": None, "elevationGain": -3.0}
    activity = activity_from_garmin(weird)
    assert activity is not None
    assert activity.moving_s == activity.elapsed_s
    assert activity.avg_hr_bpm is None
    assert activity.max_hr_bpm is None
    assert activity.elevation_gain_m is None


def test_begin_timestamp_fallback_for_start() -> None:
    summary = {k: v for k, v in SUMMARY.items() if k != "startTimeGMT"}
    activity = activity_from_garmin(summary)
    assert activity is not None
    assert activity.start == datetime.fromtimestamp(SUMMARY["beginTimestamp"] / 1000, tz=UTC)


# ─────────────────────────────── the source with a fake API ───────────────────────────────
class FakeGarmin:
    def __init__(self, summaries: list[dict[str, Any]], *, fail_login: bool = False) -> None:
        self.summaries = summaries
        self.fail_login = fail_login
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def login(self, /, tokenstore: str | None = None) -> tuple[str | None, str | None]:
        self.calls.append(("login", (tokenstore,)))
        if self.fail_login:
            raise GarminConnectAuthenticationError("Username and password are required")
        return None, None

    def get_activities_by_date(
        self,
        startdate: str,
        enddate: str | None = None,
        activitytype: str | None = None,
        sortorder: str | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(("list", (startdate, enddate, activitytype)))
        return copy.deepcopy(self.summaries)

    def get_activity_splits(self, activity_id: str) -> Any:
        self.calls.append(("splits", (activity_id,)))
        if activity_id == "3":
            raise GarminConnectConnectionError("404")
        return SPLITS

    def get_activity_hr_in_timezones(self, activity_id: str) -> Any:
        self.calls.append(("zones", (activity_id,)))
        return [{"zoneNumber": 2, "secsInZone": 3000.0}]


def summary(activity_id: int, start_gmt: str, *, zones: bool = True, type_key: str = "running") -> dict[str, Any]:
    entry = {**SUMMARY, "activityId": activity_id, "startTimeGMT": start_gmt, "activityType": {"typeKey": type_key}}
    if not zones:
        for zone in range(1, 6):
            del entry[f"hrTimeInZone_{zone}"]
    return entry


async def test_fetch_filters_sorts_limits_and_fetches_details(tmp_path: Path) -> None:
    fake = FakeGarmin(
        [
            summary(1, "2026-10-10 07:00:00"),
            summary(2, "2026-10-14 07:00:00", zones=False),
            summary(3, "2026-10-12 07:00:00"),
            summary(4, "2026-10-13 07:00:00", type_key="cycling"),
            summary(5, "2026-10-08 23:00:00"),  # the day margin returns it, the exact filter drops it
        ]
    )
    source = GarminLiveSource(GarminSettings(token_dir=tmp_path / "tokens"), api_factory=lambda: fake)
    assert source.name == "garmin"
    since = datetime(2026, 10, 9, 0, 0, tzinfo=UTC)

    activities = await source.fetch_activities(since=since, limit=10)

    assert [a.activity_id for a in activities] == ["2", "3", "1"]
    assert fake.calls[0] == ("login", (str(tmp_path / "tokens"),))
    assert fake.calls[1] == ("list", ("2026-10-08", None, "running"))
    assert ("zones", ("2",)) in fake.calls
    assert ("zones", ("1",)) not in fake.calls  # the list entry already had zone times
    by_id = {a.activity_id: a for a in activities}
    assert by_id["2"].hr_zone_s == (0.0, 3000.0, 0.0, 0.0, 0.0)
    assert by_id["3"].laps == ()  # its splits call failed: degraded, not fatal
    assert len(by_id["1"].laps) == 4
    for activity in activities:
        assert_no_private_data(activity)

    limited = await source.fetch_activities(since=since, limit=1)
    assert [a.activity_id for a in limited] == ["2"]
    assert sum(name == "login" for name, _ in fake.calls) == 1  # the session is reused


async def test_missing_or_expired_tokens_raise_auth_error(tmp_path: Path) -> None:
    fake = FakeGarmin([], fail_login=True)
    source = GarminLiveSource(GarminSettings(token_dir=tmp_path), api_factory=lambda: fake)
    with pytest.raises(AuthError, match="stridedj auth garmin"):
        await source.fetch_activities(since=datetime(2026, 1, 1, tzinfo=UTC), limit=5)


# ─────────────────────────────── interactive login ───────────────────────────────
class FakeTokenStore:
    def dump(self, path: str) -> None:
        target = Path(path) / "garmin_tokens.json"
        target.write_text(json.dumps({"di_token": "t", "di_refresh_token": "r"}))
        target.chmod(0o644)  # pretend a permissive umask: interactive_login must tighten it


class FakeLoginGarmin:
    instances: list[FakeLoginGarmin] = []  # noqa: RUF012

    def __init__(self, *, email: str, password: str, prompt_mfa: Callable[[], str], reject: bool = False) -> None:
        self.email, self.password, self.prompt_mfa, self.reject = email, password, prompt_mfa, reject
        self.client = FakeTokenStore()
        self.login_args: tuple[str | None, ...] = ()
        FakeLoginGarmin.instances.append(self)

    def login(self, /, tokenstore: str | None = None) -> tuple[str | None, str | None]:
        self.login_args = (tokenstore,)
        if self.reject:
            raise GarminConnectAuthenticationError("401 Unauthorized for hunter2")
        assert self.prompt_mfa() == "123456"
        return None, None


def settings_with_credentials(token_dir: Path) -> GarminSettings:
    return GarminSettings(email=SecretStr("jane@example.com"), password=SecretStr("hunter2"), token_dir=token_dir)


def test_interactive_login_persists_private_tokens(tmp_path: Path) -> None:
    token_dir = tmp_path / "garmin"
    token_dir.mkdir(mode=0o755)
    interactive_login(settings_with_credentials(token_dir), lambda: "123456", garmin_factory=FakeLoginGarmin)

    fake = FakeLoginGarmin.instances[-1]
    assert (fake.email, fake.password) == ("jane@example.com", "hunter2")
    assert fake.login_args == (None,)  # a fresh credential login, never a stale token load
    assert stat.S_IMODE(token_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((token_dir / "garmin_tokens.json").stat().st_mode) == 0o600


def test_interactive_login_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        interactive_login(GarminSettings(token_dir=tmp_path), lambda: "123456", garmin_factory=FakeLoginGarmin)

    def rejecting(**kwargs: Any) -> FakeLoginGarmin:
        return FakeLoginGarmin(**kwargs, reject=True)

    with pytest.raises(AuthError) as info:
        interactive_login(settings_with_credentials(tmp_path / "t"), lambda: "123456", garmin_factory=rejecting)
    assert "hunter2" not in str(info.value)
    assert info.value.__cause__ is None
