"""`SlotFinder` application service: day part → local window → free/busy → free slots.

Two questions, two methods:
* `free_slots`: "when could I run this morning?" → up to 3 examples, spread over the day part;
* `check_start`: "is 08:00 free?" → exactly that start, plus the nearest free starts if it is not. The examples
  alone cannot answer this: a free start that is not among them is still free.

The agent's `find_free_slots` tool calls these; it never sees calendar data beyond the start/end of the slots.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from stridedj.cal.slots import DEFAULT_GRID_MIN, DEFAULT_MIN_GAP_MIN, MIN_LEAD_MIN, free_slots, window_for
from stridedj.domain.calendar import DayPart, SlotRequest, StartCheck, StartStatus, TimeSlot
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

    async def check_start(self, request: SlotRequest, start: time, *, tz: ZoneInfo, now: datetime) -> StartCheck:
        """Is exactly `start` (local, any minute) free for `request.duration_min` on `request.day`?

        The day part is ignored: an explicit time wins over the day-part windows (10:00 belongs to none of them).
        If the start is busy or too soon, the alternatives are the nearest free quarter-hour starts that day (within
        the "any" window), the latest one before and the earliest one after it.
        """
        if request.duration_min <= 0:
            raise ToolError("duration_min must be > 0 minutes")
        begin = datetime.combine(request.day, start, tzinfo=tz)
        wanted = TimeSlot(start=begin, end=begin + timedelta(minutes=request.duration_min))
        day = window_for(request.day, DayPart.ANY, tz)
        pad = timedelta(minutes=self._buffer_min)
        span_start = min(day.start.astimezone(UTC), wanted.start.astimezone(UTC))
        span_end = max(day.end.astimezone(UTC), wanted.end.astimezone(UTC))
        busy = await self._calendar.busy_intervals(TimeSlot(start=span_start - pad, end=span_end + pad))

        if begin.astimezone(UTC) < now.astimezone(UTC) + timedelta(minutes=MIN_LEAD_MIN):
            status = StartStatus.TOO_SOON
        elif free_slots(
            wanted, busy, request.duration_min, now=now, max_results=1, grid_min=1, buffer_min=self._buffer_min
        ):
            return StartCheck(slot=wanted, status=StartStatus.FREE)
        else:
            status = StartStatus.BUSY

        every_start = free_slots(
            day,
            busy,
            request.duration_min,
            now=now,
            max_results=_ALL_STARTS,
            grid_min=self._grid_min,
            buffer_min=self._buffer_min,
            min_gap_min=0,
        )
        earlier = [s for s in every_start if s.start.astimezone(UTC) < begin.astimezone(UTC)]
        later = [s for s in every_start if s.start.astimezone(UTC) > begin.astimezone(UTC)]
        nearest = ([earlier[-1]] if earlier else []) + ([later[0]] if later else [])
        return StartCheck(slot=wanted, status=status, alternatives=tuple(nearest))


_ALL_STARTS = 24 * 60  # an upper bound on the starts in one day: "return every fitting start"


def _too_late(window: TimeSlot, now: datetime, duration_min: float) -> bool:
    """True when even a slot starting at the earliest allowed moment could not end inside the window."""
    earliest_end = now.astimezone(UTC) + timedelta(minutes=MIN_LEAD_MIN + duration_min)
    return earliest_end > window.end.astimezone(UTC)
