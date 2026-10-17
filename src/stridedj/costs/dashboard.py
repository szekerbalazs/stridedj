"""Cost dashboard: merge the ledger (live, per conversation) with OpenAI's official costs (exact, per key, daily).

One section per API key:
* official keys come from the Costs API (named, exact billed $);
* the key StrideDJ is configured with is linked to its official id by matching OpenAI's masked value locally;
* ledger fingerprints without an official match (no Admin key, or a rotated key) get their own section.
Rendered as terminal tables (`render_terminal`) and as a self-contained HTML page (`render_html`).
"""

from __future__ import annotations

import html
import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from importlib.resources import files
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr
from rich.console import Console
from rich.table import Table

from stridedj.costs.ledger import LedgerRun, key_fingerprint
from stridedj.costs.openai_costs import OfficialCosts, redacted_matches

DailySource = Literal["official", "ledger"]


class RunRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    when: datetime
    source: str
    model: str
    steps: int
    tool_calls: int
    tokens: int
    cost_usd: float


class KeyView(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    detail: str
    is_current: bool
    official_period_usd: float | None
    official_month_usd: float | None
    ledger_period_usd: float
    ledger_runs: int
    ledger_avg_usd: float | None
    cache_hit_rate: float | None
    by_source: dict[str, float]
    daily: list[tuple[date, float]]
    daily_source: DailySource
    line_items: list[tuple[str, float]]
    recent: list[RunRow]


class DashboardView(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    days: int
    has_official: bool
    keys: list[KeyView]
    total_official_month_usd: float | None
    total_ledger_period_usd: float


def _days(now: datetime, days: int) -> list[date]:
    today = now.astimezone(UTC).date()
    return [today - timedelta(days=i) for i in range(days - 1, -1, -1)]


def _ledger_daily(runs: Sequence[LedgerRun], axis: list[date]) -> list[tuple[date, float]]:
    per_day: dict[date, float] = defaultdict(float)
    for r in runs:
        per_day[r.started_at.astimezone(UTC).date()] += r.cost_usd
    return [(d, per_day.get(d, 0.0)) for d in axis]


def _ledger_stats(runs: Sequence[LedgerRun]) -> tuple[float, int, float | None, float | None, dict[str, float]]:
    total = sum(r.cost_usd for r in runs)
    inputs = sum(r.input_tokens for r in runs)
    cached = sum(r.cached_input_tokens for r in runs)
    by_source: dict[str, float] = defaultdict(float)
    for r in runs:
        by_source[r.source] += r.cost_usd
    avg = total / len(runs) if runs else None
    hit = cached / inputs if inputs else None
    return total, len(runs), avg, hit, dict(by_source)


def _recent(runs: Sequence[LedgerRun], n: int = 12) -> list[RunRow]:
    return [
        RunRow(
            when=r.started_at,
            source=r.source,
            model=r.model,
            steps=r.steps,
            tool_calls=r.tool_calls,
            tokens=r.input_tokens + r.output_tokens,
            cost_usd=r.cost_usd,
        )
        for r in runs[:n]
    ]


def build_view(
    *,
    ledger_runs: Sequence[LedgerRun],
    official: OfficialCosts | None,
    current_key: SecretStr | None,
    days: int,
    now: datetime | None = None,
) -> DashboardView:
    now = now or datetime.now(UTC)
    axis = _days(now, days)
    month_start = now.astimezone(UTC).date().replace(day=1)
    current_fp = key_fingerprint(current_key.get_secret_value()) if current_key else None

    runs_by_fp: dict[str, list[LedgerRun]] = defaultdict(list)
    for r in ledger_runs:
        runs_by_fp[r.key_fp].append(r)

    # official key id ↔ the configured key (matched locally on OpenAI's masked value; nothing is sent)
    linked: dict[str, str] = {}
    if official is not None and current_key is not None and current_fp is not None:
        for key_id, info in official.keys.items():
            if redacted_matches(info.redacted_value, current_key.get_secret_value()):
                linked[key_id] = current_fp

    views: list[KeyView] = []
    used_fps: set[str] = set()
    if official is not None:
        by_key: dict[str, list[tuple[date, str, float]]] = defaultdict(list)
        for c in official.costs:
            by_key[c.api_key_id].append((c.day, c.line_item, c.amount_usd))
        for key_id in sorted(set(by_key) | set(linked), key=lambda k: -sum(v for _, _, v in by_key.get(k, []))):
            entries = by_key.get(key_id, [])
            key_info = official.keys.get(key_id)
            fp = linked.get(key_id)
            runs = runs_by_fp.get(fp, []) if fp else []
            if fp:
                used_fps.add(fp)
            per_day: dict[date, float] = defaultdict(float)
            per_item: dict[str, float] = defaultdict(float)
            for day, item, value in entries:
                per_day[day] += value
                per_item[item] += value
            total, n, avg, hit, by_source = _ledger_stats(runs)
            views.append(
                KeyView(
                    label=key_info.name if key_info else key_id,
                    detail=(
                        f"{key_info.project_name} · {key_info.redacted_value}"
                        if key_info
                        else "key not found in any project"
                    ),
                    is_current=fp is not None and fp == current_fp,
                    official_period_usd=sum(per_day.values()),
                    official_month_usd=sum(v for d, v in per_day.items() if d >= month_start),
                    ledger_period_usd=total,
                    ledger_runs=n,
                    ledger_avg_usd=avg,
                    cache_hit_rate=hit,
                    by_source=by_source,
                    daily=[(d, per_day.get(d, 0.0)) for d in axis],
                    daily_source="official",
                    line_items=sorted(per_item.items(), key=lambda kv: -kv[1]),
                    recent=_recent(runs),
                )
            )
    for fp, runs in sorted(runs_by_fp.items(), key=lambda kv: -sum(r.cost_usd for r in kv[1])):
        if fp in used_fps:
            continue
        total, n, avg, hit, by_source = _ledger_stats(runs)
        views.append(
            KeyView(
                label=f"key ·{fp[:8]}",
                detail="StrideDJ's own ledger" + ("" if official is None else " · not matched to an official key"),
                is_current=fp == current_fp,
                official_period_usd=None,
                official_month_usd=None,
                ledger_period_usd=total,
                ledger_runs=n,
                ledger_avg_usd=avg,
                cache_hit_rate=hit,
                by_source=by_source,
                daily=_ledger_daily(runs, axis),
                daily_source="ledger",
                line_items=[],
                recent=_recent(runs),
            )
        )
    views.sort(key=lambda v: (not v.is_current,))
    month_values = [v.official_month_usd for v in views if v.official_month_usd is not None]
    return DashboardView(
        generated_at=now,
        days=days,
        has_official=official is not None,
        keys=views,
        total_official_month_usd=sum(month_values) if official is not None else None,
        total_ledger_period_usd=sum(v.ledger_period_usd for v in views),
    )


# ─────────────────────────────── formatting ───────────────────────────────
def usd(value: float | None) -> str:
    if value is None:
        return "–"
    if value == 0:
        return "$0"
    if abs(value) < 0.01:
        return f"${value:.4f}"
    if abs(value) < 1:
        return f"${value:.3f}"
    return f"${value:,.2f}"


def pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


# ─────────────────────────────── terminal ───────────────────────────────
def render_terminal(view: DashboardView, console: Console) -> None:
    summary = Table(title=f"OpenAI costs · last {view.days} days", title_justify="left")
    for col in (
        "API key",
        "billed (OpenAI)",
        "this month",
        "StrideDJ (ledger)",
        "conversations",
        "avg / conv.",
        "cache hits",
    ):
        summary.add_column(col, justify="left" if col == "API key" else "right")
    for k in view.keys:
        name = ("● " if k.is_current else "  ") + k.label
        summary.add_row(
            name,
            usd(k.official_period_usd),
            usd(k.official_month_usd),
            usd(k.ledger_period_usd),
            str(k.ledger_runs),
            usd(k.ledger_avg_usd),
            pct(k.cache_hit_rate),
        )
    console.print(summary)
    if not view.has_official:
        console.print(
            "[dim]Official billing not shown: set OPENAI_ADMIN_KEY for exact per-key costs (docs/SETUP.md §1).[/]"
        )
    for k in view.keys:
        if k.recent:
            t = Table(title=f"Recent conversations · {k.label}", title_justify="left")
            for col in ("when (UTC)", "source", "model", "steps", "tools", "tokens", "cost"):
                t.add_column(col, justify="left" if col in ("when (UTC)", "source", "model") else "right")
            for r in k.recent[:8]:
                t.add_row(
                    f"{r.when:%d %b %H:%M}",
                    r.source,
                    r.model,
                    str(r.steps),
                    str(r.tool_calls),
                    f"{r.tokens:,}",
                    usd(r.cost_usd),
                )
            console.print(t)


# ─────────────────────────────── html ───────────────────────────────
def _ticks(vmax: float) -> tuple[float, int]:
    """A round tick step (1, 2, 2.5 or 5 × 10^k) and how many steps cover `vmax` (2–5 steps)."""
    if vmax <= 0:
        return 0.001, 1
    raw = vmax / 4
    scale = 10.0 ** math.floor(math.log10(raw))
    step = next(m * scale for m in (1.0, 2.0, 2.5, 5.0, 10.0) if m * scale >= raw - 1e-12)
    return step, max(1, math.ceil(vmax / step - 1e-9))


def _bar_chart(daily: list[tuple[date, float]], *, title: str) -> str:
    width, height = 760, 240
    left, right, top, bottom = 56, 12, 16, 30
    plot_w, plot_h = width - left - right, height - top - bottom
    step, n_steps = _ticks(max((v for _, v in daily), default=0.0))
    vmax = step * n_steps
    band = plot_w / max(1, len(daily))
    bar_w = min(24.0, max(2.0, band - 2))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title, quote=True)}">']
    for i in range(n_steps + 1):
        value = step * i
        y = top + plot_h - plot_h * i / n_steps
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{html.escape(usd(value))}</text>'
        )
    parts.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}"/>')
    for i, (day, value) in enumerate(daily):
        x0 = left + i * band
        x = x0 + (band - bar_w) / 2
        h = plot_h * value / vmax if vmax else 0
        y = top + plot_h - h
        if h > 0:
            r = min(4.0, h, bar_w / 2)
            path = (
                f"M{x:.1f},{top + plot_h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
                f"H{x + bar_w - r:.1f} Q{x + bar_w:.1f},{y:.1f} {x + bar_w:.1f},{y + r:.1f} V{top + plot_h:.1f} Z"
            )
            parts.append(f'<path class="bar" d="{path}"/>')
        label = html.escape(f"{day:%a %d %b}", quote=True)
        parts.append(
            f'<rect class="hit" x="{x0:.1f}" y="{top}" width="{band:.1f}" height="{plot_h}" tabindex="0" '
            f'data-label="{label}" data-value="{html.escape(usd(value), quote=True)}"/>'
        )
        if (len(daily) - 1 - i) % 7 == 0:  # weekly ticks anchored on the latest day: no end-of-axis collision
            parts.append(
                f'<text class="tick" x="{x0 + band / 2:.1f}" y="{height - 10}" text-anchor="middle">{day:%d %b}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _tile(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="tile-note">{html.escape(note)}</div>' if note else ""
    return (
        f'<div class="tile"><div class="tile-label">{html.escape(label)}</div>'
        f'<div class="tile-value">{html.escape(value)}</div>{note_html}</div>'
    )


def _key_section(k: KeyView, days: int) -> str:
    official = k.official_period_usd is not None
    tiles = [
        _tile(f"Billed, last {days} days", usd(k.official_period_usd), "OpenAI Costs API")
        if official
        else _tile(f"StrideDJ, last {days} days", usd(k.ledger_period_usd), "own ledger (estimate)"),
        _tile("Billed this month", usd(k.official_month_usd), "OpenAI Costs API")
        if official
        else _tile("Conversations", str(k.ledger_runs), f"last {days} days"),
        _tile("StrideDJ conversations", str(k.ledger_runs), f"{usd(k.ledger_period_usd)} in total")
        if official
        else _tile("Avg per conversation", usd(k.ledger_avg_usd)),
        _tile("Cache hit rate", pct(k.cache_hit_rate), "input tokens served from the prompt cache"),
    ]
    chart_title = (
        f"Daily cost, last {days} days ({'billed by OpenAI' if k.daily_source == 'official' else 'StrideDJ ledger'})"
    )
    daily_rows = (
        "".join(
            f"<tr><td>{d:%a %d %b}</td><td class='num'>{html.escape(usd(v))}</td></tr>" for d, v in k.daily if v > 0
        )
        or "<tr><td colspan='2'>no costs in this period</td></tr>"
    )
    items = "".join(
        f"<tr><td>{html.escape(item)}</td><td class='num'>{html.escape(usd(v))}</td></tr>" for item, v in k.line_items
    )
    items_html = (
        f"<h3>By line item</h3><table><thead><tr><th>line item</th><th class='num'>billed</th></tr></thead>"
        f"<tbody>{items}</tbody></table>"
        if items
        else ""
    )
    sources = " · ".join(f"{html.escape(s)} {html.escape(usd(v))}" for s, v in sorted(k.by_source.items()))
    recent = "".join(
        f"<tr><td>{r.when:%d %b %H:%M}</td><td>{html.escape(r.source)}</td><td>{html.escape(r.model)}</td>"
        f"<td class='num'>{r.steps}</td><td class='num'>{r.tool_calls}</td><td class='num'>{r.tokens:,}</td>"
        f"<td class='num'>{html.escape(usd(r.cost_usd))}</td></tr>"
        for r in k.recent
    )
    recent_html = (
        "<h3>Recent conversations (StrideDJ ledger)</h3><table><thead><tr><th>when (UTC)</th><th>source</th>"
        "<th>model</th><th class='num'>steps</th><th class='num'>tool calls</th><th class='num'>tokens</th>"
        f"<th class='num'>cost</th></tr></thead><tbody>{recent}</tbody></table>"
        f"<p class='muted'>Ledger cost by source: {sources or '–'}</p>"
        if recent
        else ""
    )
    badge = '<span class="badge">configured in .env</span>' if k.is_current else ""
    return (
        f'<section class="key"><h2>{html.escape(k.label)} {badge}</h2><p class="muted">{html.escape(k.detail)}</p>'
        f'<div class="tiles">{"".join(tiles)}</div>'
        f"<figure><figcaption>{html.escape(chart_title)}</figcaption>{_bar_chart(k.daily, title=chart_title)}"
        f"<details><summary>Table view</summary><table><thead><tr><th>day</th><th class='num'>cost</th></tr></thead>"
        f"<tbody>{daily_rows}</tbody></table></details></figure>"
        f"{items_html}{recent_html}</section>"
    )


_ASSETS = files("stridedj.costs") / "assets"


def _asset(name: str) -> str:
    return (_ASSETS / name).read_text(encoding="utf-8")


def render_html(view: DashboardView) -> str:
    if view.total_official_month_usd is not None:
        hero_value, hero_label = usd(view.total_official_month_usd), "billed by OpenAI this month, all API keys"
    else:
        hero_value, hero_label = (
            usd(view.total_ledger_period_usd),
            f"StrideDJ usage, last {view.days} days (own ledger)",
        )
    note = (
        "Official numbers come from OpenAI's Costs API (daily, can lag by a few hours). Ledger numbers are StrideDJ's "
        "own estimates from token counts, live and per conversation."
        if view.has_official
        else "Showing StrideDJ's own ledger only (estimates from token counts). Set OPENAI_ADMIN_KEY for exact "
        "billed costs per API key."
    )
    empty = (
        '<section class="key"><p class="muted">No usage recorded yet. Chat with the bot, then run this again.</p>'
        "</section>"
    )
    sections = "".join(_key_section(k, view.days) for k in view.keys) or empty
    generated = f"Generated {view.generated_at:%d %b %Y %H:%M} UTC"
    hero = (
        f"<div class='hero'><div class='hero-value'>{html.escape(hero_value)}</div>"
        f"<div class='hero-label'>{html.escape(hero_label)}</div></div>"
    )
    head = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'><title>StrideDJ costs</title>"
        f"<style>{_asset('dashboard.css')}</style></head>"
    )
    return (
        f"{head}<body><div class='viz-root'><main>"
        f"<h1>StrideDJ · costs</h1><p class='muted'>{generated} · {html.escape(note)}</p>{hero}"
        f"{sections}</main><div class='tooltip' role='status'></div></div>"
        f"<script>{_asset('dashboard.js')}</script></body></html>"
    )
