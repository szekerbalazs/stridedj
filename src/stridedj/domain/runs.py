"""Run data contracts: what the run sources return, what the store keeps, and how runs are typed.

None of these objects is ever sent to the LLM. The model only sees aggregates computed from them
(durations, cadences, counts), which is the core data-minimisation rule of the project.
"""

from __future__ import annotations

from enum import StrEnum
from itertools import pairwise
from typing import Literal

from pydantic import AwareDatetime, Field, field_validator

from stridedj.domain.base import ActivityId, Frozen

RunSourceKind = Literal["garmin", "intervals_icu", "demo"]


class RunType(StrEnum):
    """What kind of run the user asks for. Zones are the user's own watch zones (not a textbook definition)."""

    Z1 = "z1"
    Z2 = "z2"
    Z3 = "z3"
    Z4 = "z4"
    Z5 = "z5"
    INTERVALS = "intervals"


class LapIntensity(StrEnum):
    ACTIVE = "active"
    REST = "rest"
    WARMUP = "warmup"
    COOLDOWN = "cooldown"
    RECOVERY = "recovery"
    INTERVAL = "interval"
    OTHER = "other"


class Lap(Frozen):
    """One lap/split. Intensity comes from structured workouts (warm-up, work bouts, recoveries)."""

    start_offset_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)
    distance_m: float = Field(ge=0)
    avg_hr_bpm: float | None = Field(default=None, gt=0)
    avg_cadence_spm: float | None = Field(default=None, gt=0)
    intensity: LapIntensity = LapIntensity.OTHER


class Activity(Frozen):
    """One run as stored locally: an allowlist of aggregate fields. No GPS, no polyline, no free-text name."""

    activity_id: ActivityId
    source: RunSourceKind
    start: AwareDatetime
    distance_m: float = Field(gt=0)
    elapsed_s: float = Field(gt=0, description="wall-clock duration including pauses")
    moving_s: float | None = Field(default=None, gt=0)
    elevation_gain_m: float | None = Field(default=None, ge=0)
    avg_hr_bpm: float | None = Field(default=None, gt=0)
    max_hr_bpm: float | None = Field(default=None, gt=0)
    avg_cadence_spm: float | None = Field(default=None, gt=0, description="steps per minute (both feet)")
    hr_zone_s: tuple[float, float, float, float, float] | None = Field(
        default=None, description="seconds spent in Z1..Z5 (the watch's zones)"
    )
    laps: tuple[Lap, ...] = ()

    @property
    def distance_km(self) -> float:
        return self.distance_m / 1000.0

    @property
    def duration_s(self) -> float:
        """The duration used for planning: moving time when known (a calendar slot is about running time + buffer)."""
        return self.moving_s if self.moving_s is not None else self.elapsed_s

    @property
    def pace_s_per_km(self) -> float:
        return self.duration_s / self.distance_km


class HeartRateZones(Frozen):
    """The user's HR zones as exclusive upper bounds of Z1..Z4; everything at or above the last bound is Z5."""

    upper_bounds_bpm: tuple[int, int, int, int]

    @field_validator("upper_bounds_bpm")
    @classmethod
    def _increasing(cls, v: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if any(a >= b for a, b in pairwise(v)):
            raise ValueError("zone bounds must be strictly increasing")
        return v

    def zone_of(self, hr_bpm: float) -> int:
        """Heart rate (bpm) → zone number 1..5."""
        for zone, upper in enumerate(self.upper_bounds_bpm, start=1):
            if hr_bpm < upper:
                return zone
        return 5


class ClassifiedRun(Frozen):
    """An activity plus the run type the classifier assigned, with a short human-readable reason."""

    activity: Activity
    run_type: RunType
    reason: str
