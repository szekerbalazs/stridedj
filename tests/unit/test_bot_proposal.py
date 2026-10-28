"""The proposal card's buttons and text (code-rendered, no Telegram needed)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from evals.reference import reference_policy
from evals.tasks import load_tasks
from stridedj.app import build_demo_app
from stridedj.bot.render import render_proposal
from stridedj.bot.telegram_bot import proposal_buttons
from stridedj.domain import MusicChoice, RunPlan, UserContext, UserId
from stridedj.llm.fake import ScriptedLLM

NOW = datetime.fromisoformat("2026-10-17T15:00:00+02:00")
USER = UserContext(user_id=UserId("tg:1"), timezone="Europe/Zurich")
TOKEN = "x" * 22  # the length of `secrets.token_urlsafe(16)`


async def _plan() -> RunPlan:
    task = next(t for t in load_tasks() if t.id == "r01-10k-z2-tonight")
    app = await build_demo_app(ScriptedLLM(reference_policy(task)), now=NOW)
    response = await app.agent.handle(USER, "10k Z2 tonight")
    assert response.plan is not None and response.plan.playlist is not None
    return response.plan


async def test_a_plan_with_a_playlist_offers_0_50_100_percent_new() -> None:
    plan = await _plan()
    music_row, decision_row = proposal_buttons(plan, TOKEN)
    assert [label for label, _ in music_row] == ["0% new", "✓ 50% new", "100% new"]  # usual is ticked
    assert [data for _, data in music_row] == [f"m:{TOKEN}:f", f"m:{TOKEN}:u", f"m:{TOKEN}:d"]
    assert decision_row == [("✅ Approve", f"a:{TOKEN}"), ("✖ Cancel", f"c:{TOKEN}")]
    assert all(len(data.encode()) <= 64 for row in (music_row, decision_row) for _, data in row)  # Telegram's limit

    assert plan.playlist is not None
    discover = plan.model_copy(update={"playlist": plan.playlist.model_copy(update={"music": MusicChoice.DISCOVER})})
    assert proposal_buttons(discover, TOKEN)[0][2][0] == "✓ 100% new"


async def test_a_plan_without_a_playlist_has_only_approve_and_cancel() -> None:
    plan = (await _plan()).model_copy(update={"playlist": None})
    assert proposal_buttons(plan, TOKEN) == [[("✅ Approve", f"a:{TOKEN}"), ("✖ Cancel", f"c:{TOKEN}")]]


async def test_the_card_says_how_many_songs_are_new() -> None:
    plan = await _plan()
    assert plan.playlist is not None
    n_new = round(plan.playlist.novelty_share * len(plan.playlist.track_ids))
    assert f"{len(plan.playlist.track_ids)} tracks · {n_new} new · " in render_proposal(
        plan, [], ZoneInfo("Europe/Zurich")
    )
