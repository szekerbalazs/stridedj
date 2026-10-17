from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from stridedj.config import AppSettings
from stridedj.music.cache import JsonCache, default_cache_path
from tests.unit.music.support import FrozenClock


def test_roundtrip_of_json_values(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path / "c.db")
    cache.set("a", {"x": [1, 2.5, "s", None, True]})
    assert cache.get("a") == {"x": [1, 2.5, "s", None, True]}
    assert cache.get("missing") is None
    cache.set("a", 3)
    assert cache.get("a") == 3


def test_ttl_expiry(tmp_path: Path, clock: FrozenClock) -> None:
    cache = JsonCache(tmp_path / "c.db", clock=clock)
    cache.set("short", 1, ttl_s=60)
    cache.set("forever", 2)
    clock.advance(seconds=59)
    assert cache.get_many(["short", "forever"]) == {"short": 1, "forever": 2}
    clock.advance(seconds=2)
    assert cache.get_many(["short", "forever"]) == {"forever": 2}
    assert cache.purge_expired() == 1


def test_persists_across_instances_and_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "c.db"
    JsonCache(path).set("k", "v")
    assert JsonCache(path).get("k") == "v"


def test_get_many_handles_more_keys_than_sqlite_variables() -> None:
    cache = JsonCache(Path(":memory:"))
    cache.set_many({f"k{i}": i for i in range(1200)})
    assert len(cache.get_many(f"k{i}" for i in range(1300))) == 1200


def test_delete_and_invalid_ttl() -> None:
    cache = JsonCache(Path(":memory:"))
    cache.set("k", 1)
    cache.delete(["k"])
    assert cache.get("k") is None
    with pytest.raises(ValueError, match="ttl_s"):
        cache.set("k", 1, ttl_s=0)


def test_naive_clock_is_rejected() -> None:
    cache = JsonCache(Path(":memory:"), clock=FrozenClock(datetime(2026, 1, 1)))  # noqa: DTZ001 (on purpose)
    with pytest.raises(ValueError, match="timezone-aware"):
        cache.set("k", 1, ttl_s=60)
    with pytest.raises(ValueError, match="timezone-aware"):
        cache.get("k")


def test_default_path_is_in_data_dir(tmp_path: Path) -> None:
    assert default_cache_path(AppSettings(data_dir=tmp_path)) == tmp_path / "music_cache.db"
