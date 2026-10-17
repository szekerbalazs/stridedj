"""Per-chat state.

The model refers to things by **id** (`slot_id`, `draft_id`, `proposal_id`); the data behind the ids lives here, in
code. So the model can't alter a plan's numbers (it never passes them back), and a plan is always assembled from
values our own code computed. "References, not values" is a cheap and strong guardrail.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from stridedj.domain import (
    CadenceTarget,
    DraftId,
    DurationEstimate,
    PlanId,
    PlaylistDraft,
    RunPlan,
    RunType,
    SlotId,
    TimeSlot,
    UserContext,
)

MAX_HISTORY_EXCHANGES = 6


@dataclass
class PlanningState:
    slots: dict[SlotId, TimeSlot] = field(default_factory=dict)
    drafts: dict[DraftId, PlaylistDraft] = field(default_factory=dict)
    estimates: dict[tuple[RunType, float], DurationEstimate] = field(default_factory=dict)
    cadences: dict[RunType, CadenceTarget] = field(default_factory=dict)
    plans: dict[PlanId, RunPlan] = field(default_factory=dict)


@dataclass
class AgentSession:
    user: UserContext
    state: PlanningState = field(default_factory=PlanningState)
    history: deque[tuple[str, str]] = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_EXCHANGES))
    run_proposals: list[PlanId] = field(default_factory=list)  # proposals created during the current run only

    def start_run(self) -> None:
        self.run_proposals.clear()
