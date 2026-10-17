"""Calendar: free/busy-only reads, slot finding in the user's timezone, idempotent event insert.

`google_auth` (the interactive login) is not re-exported: it is only needed by `stridedj auth google`.
"""

from stridedj.cal.demo import POISONED_TITLE, CalendarEvent, InMemoryCalendar, InsertedEvent, realistic_week
from stridedj.cal.gcal import GoogleCalendarClient, event_id_for, is_valid_event_id
from stridedj.cal.service import SlotFinderService
from stridedj.cal.slots import DAY_PART_HOURS, free_slots, round_up_to_grid, window_for

__all__ = [
    "DAY_PART_HOURS",
    "POISONED_TITLE",
    "CalendarEvent",
    "GoogleCalendarClient",
    "InMemoryCalendar",
    "InsertedEvent",
    "SlotFinderService",
    "event_id_for",
    "free_slots",
    "is_valid_event_id",
    "realistic_week",
    "round_up_to_grid",
    "window_for",
]
