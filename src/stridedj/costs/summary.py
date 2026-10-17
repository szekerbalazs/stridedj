"""A short cost summary for chat (`/costs` in Telegram): ledger only, so the bot never needs the Admin key."""

from __future__ import annotations

from datetime import UTC, datetime

from stridedj.costs.dashboard import usd
from stridedj.costs.ledger import CostLedger


def month_summary(ledger: CostLedger, *, key_fp: str, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    month_start = now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    day_start = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    runs = ledger.runs(since=month_start, key_fp=key_fp)
    total = sum(r.cost_usd for r in runs)
    today = sum(r.cost_usd for r in runs if r.started_at >= day_start)
    avg = total / len(runs) if runs else 0.0
    by_source = {s: sum(r.cost_usd for r in runs if r.source == s) for s in ("bot", "cli", "eval")}
    parts = ", ".join(f"{s} {usd(v)}" for s, v in by_source.items() if v > 0) or "none yet"
    return (
        f"This month: {usd(total)} over {len(runs)} conversation(s), avg {usd(avg)} each.\n"
        f"Today: {usd(today)}. By source: {parts}.\n"
        f"API key ·{key_fp[:8]} · StrideDJ's own estimate. "
        "Run `stridedj costs` on your computer for OpenAI's billed numbers."
    )
