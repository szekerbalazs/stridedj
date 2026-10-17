"""A short cost summary for chat (`/costs` in Telegram): ledger only, so the bot never needs the Admin key.

It shows amounts only. No key, no fingerprint, no account details: a chat message can be forwarded or screenshotted.
"""

from __future__ import annotations

from datetime import UTC, datetime

from stridedj.costs.dashboard import SOURCE_NAMES, usd
from stridedj.costs.ledger import CostLedger


def month_summary(ledger: CostLedger, *, key_fp: str, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    month_start = now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    day_start = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    runs = ledger.runs(since=month_start, key_fp=key_fp)
    if not runs:
        return (
            f"💸 StrideDJ costs, {now:%B}\n"
            "No conversations recorded this month yet.\n"
            "Exact billing: run `stridedj costs` on your computer."
        )
    total = sum(r.cost_usd for r in runs)
    today = sum(r.cost_usd for r in runs if r.started_at >= day_start)
    lines = [
        f"💸 StrideDJ costs, {now:%B}",
        f"Total: {usd(total)} for {len(runs)} conversation(s), about {usd(total / len(runs))} each",
        f"Today: {usd(today)}",
    ]
    for source, name in SOURCE_NAMES.items():
        subset = [r for r in runs if r.source == source]
        if subset:
            lines.append(f"• {name}: {usd(sum(r.cost_usd for r in subset))} ({len(subset)})")
    lines.append("Estimated from token counts. Exact billing: run `stridedj costs` on your computer.")
    return "\n".join(lines)
