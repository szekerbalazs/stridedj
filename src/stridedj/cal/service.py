"""`SlotFinder` application service: day part → local window → free/busy → free slots.

The agent's `find_slots` tool calls this; it never sees calendar data beyond the start/end of the offered slots.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from stridedj.cal.slots import DEFAULT_GRID_MIN, DEFAULT_MIN_GAP_MIN, MIN_LEAD_MIN, free_slots, window_for
from stridedj.domain.calendar import SlotRequest, TimeSlot
from stridedj.errors import ToolError
from stridedj.ports import CalendarPort


class SlotFinderService:
    """Implements `ports.SlotFinder` on top of any `CalendarPort` (Google, or the in-memory demo calendar)."""

    def __init__(
        self,
        calendar: CalendarPort,
        *,
        buffer_min: int = 0,
        grid_min: int = DEFAULT_GRID_MIN,
        min_gap_min: int = DEFAULT_MIN_GAP_MIN,
    ) -> None:
        self._calendar = calendar
        self._buffer_min = buffer_min
        self._grid_min = grid_min
        self._min_gap_min = min_gap_min

    async def free_slots(
        self, request: SlotRequest, *, tz: ZoneInfo, now: datetime, max_results: int = 3
    ) -> list[TimeSlot]:
        """Free slots of `request.duration_min` on `request.day` within its day part (local to `tz`), on the grid,
        never in the past. Empty if nothing fits, including when the whole window is already over."""
        if request.duration_min <= 0:
            raise ToolError("duration_min must be > 0 minutes")
        window = window_for(request.day, request.day_part, tz)
        if _too_late(window, now, request.duration_min):
            return []  # e.g. "this morning" asked at 11:00: no need to call the calendar API

        # widen the query by the buffer, so a meeting ending at 05:55 still pushes a 06:00 start back
        pad = timedelta(minutes=self._buffer_min)
        query = TimeSlot(start=window.start.astimezone(UTC) - pad, end=window.end.astimezone(UTC) + pad)
        busy = await self._calendar.busy_intervals(query)
        return free_slots(
            window,
            busy,
            request.duration_min,
            now=now,
            max_results=max_results,
            grid_min=self._grid_min,
            buffer_min=self._buffer_min,
            min_gap_min=self._min_gap_min,
        )


def _too_late(window: TimeSlot, now: datetime, duration_min: float) -> bool:
    """True when even a slot starting at the earliest allowed moment could not end inside the window."""
    earliest_end = now.astimezone(UTC) + timedelta(minutes=MIN_LEAD_MIN + duration_min)
    return earliest_end > window.end.astimezone(UTC)
