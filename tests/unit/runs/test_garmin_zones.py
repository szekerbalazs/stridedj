from __future__ import annotations

from stridedj.runs.sources.garmin_live import zones_from_garmin


def _profile(sport: str, floors: list[int]) -> dict[str, object]:
    return {
        "sport": sport,
        "trainingMethod": "HR_RESERVE",
        **{f"zone{i}Floor": f for i, f in enumerate(floors, start=1)},
    }


def test_maps_zone_floors_2_to_5_to_upper_bounds_of_z1_to_z4() -> None:
    zones = zones_from_garmin([_profile("DEFAULT", [122, 135, 152, 165, 182])])
    assert zones is not None
    assert zones.upper_bounds_bpm == (135, 152, 165, 182)
    assert zones.zone_of(150) == 2
    assert zones.zone_of(182) == 5


def test_prefers_the_running_profile() -> None:
    zones = zones_from_garmin(
        [_profile("DEFAULT", [120, 130, 140, 150, 160]), _profile("RUNNING", [125, 140, 155, 168, 180])]
    )
    assert zones is not None
    assert zones.upper_bounds_bpm == (140, 155, 168, 180)


def test_unusable_payloads_return_none() -> None:
    assert zones_from_garmin(None) is None
    assert zones_from_garmin([{"sport": "CYCLING"}]) is None
    assert zones_from_garmin([_profile("DEFAULT", [0, 0, 0, 0, 0])]) is None
    assert zones_from_garmin([_profile("DEFAULT", [122, 160, 150, 165, 182])]) is None  # not increasing
