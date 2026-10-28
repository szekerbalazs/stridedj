"""Render agent results for humans. Code, not the LLM, formats everything the user sees around a proposal.

Why: the model's `message` is short free text (escaped before display); the facts (time, duration, cadence, tracks)
are rendered from the plan object, so what the user approves is exactly what will be executed. Untrusted strings
(track titles, artist names) are HTML-escaped: a song called "<a href=...>" can't inject markup into the chat.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from zoneinfo import ZoneInfo

from stridedj.domain import ExecutionResult, IntervalSession, RunPlan, Track
from stridedj.harness.executor import RUN_TYPE_LABEL

MAX_TRACKS_SHOWN = 8


def _b(text: str, rich: bool) -> str:
    return f"<b>{text}</b>" if rich else text


def _esc(text: str, rich: bool) -> str:
    return html.escape(text, quote=False) if rich else text


def render_proposal(plan: RunPlan, tracks: Sequence[Track], tz: ZoneInfo, *, rich: bool = True) -> str:
    """The proposal card. `rich=True` → Telegram HTML; `rich=False` → plain text (CLI)."""
    start = plan.slot.start.astimezone(tz)
    end = plan.slot.end.astimezone(tz)
    d = plan.duration
    lines = [
        _b(f"🏃 {plan.distance_km:g} km {RUN_TYPE_LABEL[plan.run_type]} run", rich),
        f"📅 {start:%a %d %b}, {start:%H:%M}–{end:%H:%M}",
        f"⏱ ~{d.p50_min:.0f} min typical, ≤ {d.p80_min:.0f} min in 4 of 5 runs",
        f"   {_esc(d.note, rich)}",
        f"👟 cadence target {plan.cadence.spm:.0f} spm"
        + (f" ({plan.cadence.n_runs} recent runs)" if plan.cadence.n_runs else ""),
    ]
    if plan.session is not None:
        lines.insert(1, f"🔁 {plan.session.outline}")
    p = plan.playlist
    if p is None:
        lines.append("🎵 no playlist")
    else:
        genres = f" · {', '.join(p.top_genres)}" if p.top_genres else ""
        n_new = round(p.novelty_share * len(p.track_ids))
        stats = f"{len(p.track_ids)} tracks · {n_new} new · {p.total_min:.0f} min · ±{p.mean_abs_bpm_error:.1f} BPM"
        lines.append(f"🎵 {stats}{_esc(genres, rich)}")
        for i, track in enumerate(tracks[:MAX_TRACKS_SHOWN], start=1):
            artists = ", ".join(track.artist_names)
            lines.append(f"   {i}. {_esc(track.name, rich)} — {_esc(artists, rich)}")
        if len(tracks) > MAX_TRACKS_SHOWN:
            lines.append(f"   … and {len(tracks) - MAX_TRACKS_SHOWN} more")
    return "\n".join(lines)


def session_button_label(session: IntervalSession) -> str:
    """ "6 × 800 m @ 4:10 · 8.3 km · ~52′" (a pyramid: "Pyramid 400-1200 m · 7.1 km · ~46′")."""
    if len(set(session.reps_m)) == 1:
        head = session.main_set.split(",")[0]
    else:
        head = f"Pyramid {min(session.reps_m)}-{max(session.reps_m)} m"
    return f"{head} · {session.total_km:g} km · ~{session.duration.p50_min:.0f}′"


def render_execution(result: ExecutionResult, *, rich: bool = True) -> str:
    lines: list[str] = []
    if result.calendar_event_id:
        lines.append("✅ Added to your calendar.")
    if result.playlist_url:
        url = html.escape(result.playlist_url, quote=True) if rich else result.playlist_url
        lines.append(f'🎵 <a href="{url}">Your playlist</a>' if rich else f"🎵 Playlist: {url}")
    lines.extend(f"⚠️ {_esc(e, rich)}" for e in result.errors)
    return "\n".join(lines) or "Nothing to do."
