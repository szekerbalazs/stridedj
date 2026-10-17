"""Domain contracts: every object that crosses a component or trust boundary is defined in this package."""

from stridedj.domain.agent import (
    AgentReply,
    ExecutionResult,
    MissingField,
    ReplyKind,
    RunPlan,
    UserContext,
)
from stridedj.domain.base import ActivityId, ArtistId, DraftId, Frozen, PlanId, SlotId, TrackId, UserId
from stridedj.domain.calendar import DayPart, SlotOption, SlotRequest, TimeSlot
from stridedj.domain.music import AudioFeatures, Mood, PlaylistDraft, PlaylistDraftSummary, Track
from stridedj.domain.planning import (
    CadenceTarget,
    DurationEstimate,
    EstimateMethod,
    RacePrediction,
    RunningProfile,
)
from stridedj.domain.runs import Activity, ClassifiedRun, HeartRateZones, Lap, LapIntensity, RunType

__all__ = [
    "Activity",
    "ActivityId",
    "AgentReply",
    "ArtistId",
    "AudioFeatures",
    "CadenceTarget",
    "ClassifiedRun",
    "DayPart",
    "DraftId",
    "DurationEstimate",
    "EstimateMethod",
    "ExecutionResult",
    "Frozen",
    "HeartRateZones",
    "Lap",
    "LapIntensity",
    "MissingField",
    "Mood",
    "PlanId",
    "PlaylistDraft",
    "PlaylistDraftSummary",
    "RacePrediction",
    "ReplyKind",
    "RunPlan",
    "RunType",
    "RunningProfile",
    "SlotId",
    "SlotOption",
    "SlotRequest",
    "TimeSlot",
    "Track",
    "TrackId",
    "UserContext",
    "UserId",
]
