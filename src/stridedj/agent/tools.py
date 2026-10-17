"""The tools the model can call. All of them are READ-ONLY with respect to the outside world.

Tool design rules applied here (they are also the answer to "how do you design tools for an agent?"):
* **Descriptions are prompts.** Each says what it does, when to use it, units, and what to pass from other tools.
* **Closed sets as enums** (run type, day part, mood): the model can't invent values, and the schema documents them.
* **References, not values**: plans are assembled from ids (`slot_id`, `draft_id`); numbers come from our state.
* **Aggregates only**: no GPS, no raw series, no activity names, no song titles, no calendar titles.
* **Errors are instructions** the model can act on ("slot too short: call find_free_slots with duration_min=75").
* **Few, task-shaped tools** instead of thin API wrappers: fewer calls, fewer ways to go wrong.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from pydantic import Field, field_validator

from stridedj.agent.session import AgentSession
from stridedj.domain import (
    CadenceTarget,
    DayPart,
    DraftId,
    DurationEstimate,
    Frozen,
    Mood,
    PlanId,
    PlaylistDraftSummary,
    RacePrediction,
    RunningProfile,
    RunPlan,
    RunType,
    SlotId,
    SlotRequest,
)
from stridedj.errors import ToolError
from stridedj.harness.tools import Tool, ToolRegistry
from stridedj.ports import Clock, PlaylistPlanner, RunInsights, SlotFinder

MAX_DAYS_AHEAD = 14
RUN_TYPE_HELP = "z1..z5 = the runner's own heart-rate zones (z2 = easy), 'intervals' = a structured interval session"


# ─────────────────────────────── argument models (→ strict JSON schemas) ───────────────────────────────
class RunTypeArgs(Frozen):
    run_type: RunType = Field(description=RUN_TYPE_HELP)


class NoArgs(Frozen):
    pass


class EstimateDurationArgs(Frozen):
    run_type: RunType = Field(description=RUN_TYPE_HELP)
    distance_km: float = Field(description="planned distance in km, e.g. 10 for '10k', 21.1 for a half marathon")

    @field_validator("distance_km")
    @classmethod
    def _range(cls, v: float) -> float:
        if not 0 < v <= 60:
            raise ValueError("distance_km must be between 0 and 60; ask the runner")
        return v


class FindSlotsArgs(Frozen):
    date: str = Field(description="local date of the run, YYYY-MM-DD (resolve 'tonight'/'tomorrow' from the Context)")
    day_part: DayPart = Field(description="morning 06:00-10:00, midday 11:30-14:00, evening 17:00-21:30, any = all day")
    duration_min: int = Field(description="slot length in minutes: pass slot_min from estimate_duration")

    @field_validator("date")
    @classmethod
    def _iso(cls, v: str) -> str:
        date.fromisoformat(v)  # raises ValueError with a clear message
        return v

    @field_validator("duration_min")
    @classmethod
    def _length(cls, v: int) -> int:
        if not 15 <= v <= 360:
            raise ValueError("duration_min must be between 15 and 360")
        return v


class DraftPlaylistArgs(Frozen):
    run_type: RunType = Field(description=RUN_TYPE_HELP)
    duration_min: float = Field(description="music length in minutes: pass p80_min from estimate_duration")
    mood: Mood = Field(description="'usual' unless the runner asks for more energetic or calmer music")

    @field_validator("duration_min")
    @classmethod
    def _length(cls, v: float) -> float:
        if not 5 <= v <= 360:
            raise ValueError("duration_min must be between 5 and 360")
        return v


class ProposePlanArgs(Frozen):
    run_type: RunType = Field(description=RUN_TYPE_HELP)
    distance_km: float = Field(description="the same distance passed to estimate_duration")
    slot_id: str = Field(description="a slot_id returned by find_free_slots")
    draft_id: str | None = Field(description="a draft_id returned by draft_playlist, or null for no playlist")


# ─────────────────────────────── result models (what the model reads) ───────────────────────────────
class SlotOut(Frozen):
    slot_id: str
    start_local: str = Field(description="e.g. 'Sat 17 Oct 18:30'")
    end_local: str = Field(description="e.g. '19:45'")


class SlotsOut(Frozen):
    date: str
    day_part: DayPart
    timezone: str
    slots: list[SlotOut]
    note: str


class DraftOut(Frozen):
    draft: PlaylistDraftSummary
    cadence: CadenceTarget


class RacePredictionsOut(Frozen):
    predictions: list[RacePrediction]
    note: str


class ProposalOut(Frozen):
    proposal_id: str
    summary: str
    next_step: str


# ─────────────────────────────── the tools ───────────────────────────────
@dataclass(frozen=True)
class AgentDeps:
    """Everything the tools need, injected (fakes in tests/evals/demo, real adapters in production)."""

    insights: RunInsights
    slots: SlotFinder
    playlists: PlaylistPlanner
    clock: Clock


def _fmt_local(dt: datetime, *, with_day: bool) -> str:
    return dt.strftime("%a %d %b %H:%M") if with_day else dt.strftime("%H:%M")


def build_tools(deps: AgentDeps, session: AgentSession) -> ToolRegistry:
    """Bind the tools to one user's session. Called per run, so every closure sees the right user and state."""
    tz = session.user.tz
    state = session.state

    def now() -> datetime:
        return deps.clock.now().astimezone(tz)

    def estimate(run_type: RunType, distance_km: float) -> DurationEstimate:
        key = (run_type, round(distance_km, 2))
        if key not in state.estimates:
            state.estimates[key] = deps.insights.estimate_duration(run_type, distance_km, now=now())
        return state.estimates[key]

    def cadence(run_type: RunType) -> CadenceTarget:
        if run_type not in state.cadences:
            state.cadences[run_type] = deps.insights.cadence_target(run_type, now=now())
        return state.cadences[run_type]

    async def get_running_profile(args: RunTypeArgs) -> RunningProfile:
        return deps.insights.profile(args.run_type, now=now())

    async def predict_race_times(_: NoArgs) -> RacePredictionsOut:
        preds = list(deps.insights.race_predictions(now=now()))
        return RacePredictionsOut(
            predictions=preds, note="estimates from training runs, not races; real race results are usually faster"
        )

    async def estimate_duration(args: EstimateDurationArgs) -> DurationEstimate:
        return estimate(args.run_type, args.distance_km)

    async def find_free_slots(args: FindSlotsArgs) -> SlotsOut:
        day = date.fromisoformat(args.date)
        today = now().date()
        if day < today:
            raise ToolError(f"{args.date} is in the past; today is {today.isoformat()}")
        if day > today + timedelta(days=MAX_DAYS_AHEAD):
            raise ToolError(f"can only plan up to {MAX_DAYS_AHEAD} days ahead")
        request = SlotRequest(day=day, day_part=args.day_part, duration_min=args.duration_min)
        found = await deps.slots.free_slots(request, tz=tz, now=now())
        out: list[SlotOut] = []
        for slot in found:
            slot_id = SlotId(f"s{len(state.slots) + 1}")
            state.slots[slot_id] = slot
            out.append(
                SlotOut(
                    slot_id=slot_id,
                    start_local=_fmt_local(slot.start.astimezone(tz), with_day=True),
                    end_local=_fmt_local(slot.end.astimezone(tz), with_day=False),
                )
            )
        note = "no free slot of that length in this day part" if not out else f"{len(out)} free slot(s)"
        return SlotsOut(date=args.date, day_part=args.day_part, timezone=session.user.timezone, slots=out, note=note)

    async def draft_playlist(args: DraftPlaylistArgs) -> DraftOut:
        target = cadence(args.run_type)
        draft = await deps.playlists.draft(
            run_type=args.run_type, cadence=target, duration_min=args.duration_min, mood=args.mood
        )
        state.drafts[draft.draft_id] = draft
        summary = PlaylistDraftSummary(
            draft_id=draft.draft_id,
            n_tracks=len(draft.track_ids),
            total_min=round(draft.total_min, 1),
            target_spm=draft.target_spm,
            tolerance_bpm=draft.tolerance_bpm,
            mean_abs_bpm_error=round(draft.mean_abs_bpm_error, 2),
            top_genres=draft.top_genres,
        )
        return DraftOut(draft=summary, cadence=target)

    async def propose_plan(args: ProposePlanArgs) -> ProposalOut:
        slot = state.slots.get(SlotId(args.slot_id))
        if slot is None:
            raise ToolError("unknown slot_id; call find_free_slots first and use one of its slot_ids")
        duration = estimate(args.run_type, args.distance_km)
        if slot.duration_min < duration.slot_min:
            raise ToolError(
                f"slot is {slot.duration_min:.0f} min but this run needs {duration.slot_min} min; "
                f"call find_free_slots with duration_min={duration.slot_min}"
            )
        draft = None
        if args.draft_id is not None:
            draft = state.drafts.get(DraftId(args.draft_id))
            if draft is None:
                raise ToolError("unknown draft_id; call draft_playlist first, or pass null for no playlist")
            if draft.run_type != args.run_type:
                raise ToolError("this playlist was drafted for another run type; call draft_playlist again")
        plan = RunPlan(
            plan_id=PlanId(uuid.uuid4().hex[:12]),
            user_id=session.user.user_id,
            run_type=args.run_type,
            distance_km=args.distance_km,
            duration=duration,
            slot=slot,
            cadence=cadence(args.run_type),
            playlist=draft,
            created_at=now(),
        )
        state.plans[plan.plan_id] = plan
        session.run_proposals.append(plan.plan_id)
        start = slot.start.astimezone(tz)
        music = f", playlist {len(draft.track_ids)} tracks / {draft.total_min:.0f} min" if draft else ", no playlist"
        summary = (
            f"{args.distance_km:g} km {args.run_type.value} on {_fmt_local(start, with_day=True)}, "
            f"expected {duration.p50_min:.0f} min (p80 {duration.p80_min:.0f}), slot {duration.slot_min} min{music}"
        )
        return ProposalOut(
            proposal_id=plan.plan_id,
            summary=summary,
            next_step="The runner now sees Approve/Cancel buttons. Finish with kind 'proposal' and this proposal_id.",
        )

    return ToolRegistry(
        [
            Tool(
                "estimate_duration",
                "Estimate how long a run will take from the runner's own history. Returns p50/p80 minutes, slot_min "
                "(calendar length incl. buffer, already rounded) and how the estimate was made. Call this first "
                "when planning a run.",
                EstimateDurationArgs,
                estimate_duration,
            ),
            Tool(
                "find_free_slots",
                "Find up to 3 free calendar slots of duration_min minutes on a date and part of the day. Returns "
                "slot_ids to use in propose_plan. Use slot_min from estimate_duration as duration_min.",
                FindSlotsArgs,
                find_free_slots,
            ),
            Tool(
                "draft_playlist",
                "Draft (not create) a playlist of at least duration_min minutes whose tempo matches the runner's "
                "cadence for this run type, in the style of their running music. Use p80_min from "
                "estimate_duration as duration_min. Returns a draft_id, stats and the cadence target.",
                DraftPlaylistArgs,
                draft_playlist,
            ),
            Tool(
                "propose_plan",
                "Create the proposal the runner will approve or cancel with buttons. Nothing is booked or created "
                "until they approve. Requires a slot_id from find_free_slots and optionally a draft_id.",
                ProposePlanArgs,
                propose_plan,
            ),
            Tool(
                "get_running_profile",
                "Summary of the runner's recent runs of one type (count, typical distance, median pace in s/km, "
                "cadence, days since the last one, pace trend). Use it to answer questions about their running.",
                RunTypeArgs,
                get_running_profile,
            ),
            Tool(
                "predict_race_times",
                "Predicted finishing times for 5 km, 10 km, half marathon and marathon from recent training runs.",
                NoArgs,
                predict_race_times,
            ),
        ]
    )
