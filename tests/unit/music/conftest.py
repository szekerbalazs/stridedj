from __future__ import annotations

import pytest

from tests.unit.music.support import FrozenClock


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock()
