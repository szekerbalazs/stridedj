"""Command-line entry point (`stridedj ...`). Thin: parses arguments and delegates to the composition root."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated
from zoneinfo import ZoneInfo

import typer
from rich.console import Console
from rich.table import Table

from stridedj.app import App, build_demo_app, build_live_app
from stridedj.auth import Service, login
from stridedj.bot.render import render_execution, render_proposal
from stridedj.config import Settings, load_settings
from stridedj.costs.ledger import CostLedger, Source, UsageSink, default_ledger_path, key_fingerprint
from stridedj.costs.summary import month_summary
from stridedj.doctor import CheckStatus, run_checks
from stridedj.domain import ReplyKind, UserContext, UserId
from stridedj.errors import ApprovalError
from stridedj.llm.openai_client import OpenAIResponsesClient


class DataMode(StrEnum):
    DEMO = "demo"  # synthetic runs, fake calendar and music; only the OpenAI key is needed
    LIVE = "live"  # Garmin/intervals.icu, Google Calendar, Spotify


app = typer.Typer(help="StrideDJ: plan a run, book the slot, match the music.", no_args_is_help=True)
console = Console()

_ICON = {
    CheckStatus.OK: "[green]✓[/]",
    CheckStatus.MISSING: "[red]✗[/]",
    CheckStatus.FAILED: "[red]✗[/]",
    CheckStatus.PENDING: "[yellow]…[/]",
}


@app.command()
def doctor(
    offline: Annotated[
        bool, typer.Option("--offline", help="Only check that values are present (no network).")
    ] = False,
) -> None:
    """Check every credential in .env. Prints ✓/✗ and hints, never the values."""
    settings = load_settings()
    results = asyncio.run(run_checks(settings, live=not offline))
    table = Table(show_header=True, header_style="bold")
    table.add_column("")
    table.add_column("check")
    table.add_column("detail")
    for r in results:
        table.add_row(_ICON[r.status], r.name, r.detail)
    console.print(table)
    if any(r.status in (CheckStatus.MISSING, CheckStatus.FAILED) for r in results):
        raise typer.Exit(code=1)


@app.command()
def auth(service: Annotated[Service, typer.Argument(help="garmin | spotify | google")]) -> None:
    """One-time login for a service (opens the browser for Spotify/Google; asks for an MFA code for Garmin)."""
    login(service)


def _ledger_sink(settings: Settings, source: Source) -> UsageSink | None:
    """Every conversation is recorded in the cost ledger, under a fingerprint of the API key in use."""
    if settings.openai.api_key is None:
        return None
    ledger = CostLedger(default_ledger_path(settings.app.data_dir))
    return ledger.sink(key_fp=key_fingerprint(settings.openai.api_key.get_secret_value()), source=source)


async def _build(settings: Settings, data: DataMode, model: str | None, *, source: Source) -> App:
    openai_settings = settings.openai.model_copy(update={"model": model}) if model else settings.openai
    llm = OpenAIResponsesClient.from_settings(openai_settings)
    sink = _ledger_sink(settings, source)
    if data is DataMode.DEMO:
        return await build_demo_app(
            llm,
            now=datetime.now(UTC),
            timezone=settings.athlete.timezone,
            traces_dir=settings.app.data_dir / "traces",
            usage_sink=sink,
        )
    return await build_live_app(settings, llm, usage_sink=sink)


async def _chat_loop(app_: App) -> None:
    user = UserContext(user_id=UserId("cli"), timezone=app_.timezone)
    tz = ZoneInfo(app_.timezone)
    console.print("[bold]StrideDJ[/] (empty line to quit). Try: [italic]10k Z2 tonight[/]")
    while True:
        text = console.input("[bold cyan]you>[/] ").strip()
        if not text:
            return
        response = await app_.agent.handle(user, text)
        t = response.outcome.transcript
        console.print(f"[bold magenta]stridedj>[/] {response.reply.message}")
        console.print(
            f"[dim]({response.reply.kind.value} · {t.steps} steps · {t.tool_calls} tool calls · "
            f"${t.cost_usd:.4f} · {t.duration_s:.1f}s)[/]"
        )
        if response.reply.kind is ReplyKind.PROPOSAL and response.plan and response.approval_token:
            playlist = response.plan.playlist
            tracks = await app_.playlists.tracks_for(playlist) if playlist else []
            console.print(render_proposal(response.plan, tracks, tz, rich=False), markup=False)
            if typer.confirm("Approve?", default=False):
                try:
                    result = await app_.agent.approve(response.approval_token, user_id=user.user_id)
                except ApprovalError as exc:
                    console.print(f"[red]{exc}[/]")
                    continue
                console.print(render_execution(result, rich=False), markup=False)


@app.command()
def chat(
    data: Annotated[
        DataMode, typer.Option(help="demo: synthetic data + fakes; live: your real accounts")
    ] = DataMode.DEMO,
    model: Annotated[str | None, typer.Option(help="override OPENAI_MODEL")] = None,
) -> None:
    """Talk to the agent in the terminal (same agent as the Telegram bot)."""
    settings = load_settings()

    async def main() -> None:
        await _chat_loop(await _build(settings, data, model, source="cli"))

    asyncio.run(main())


@app.command()
def bot(
    data: Annotated[
        DataMode, typer.Option(help="demo: synthetic data + fakes; live: your real accounts")
    ] = DataMode.LIVE,
    model: Annotated[str | None, typer.Option(help="override OPENAI_MODEL")] = None,
) -> None:
    """Run the Telegram bot (long polling)."""
    from stridedj.bot.telegram_bot import TelegramBot
    from stridedj.clock import SystemClock
    from stridedj.harness.guard import InputGuard

    settings = load_settings()
    logging.basicConfig(level=settings.app.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx logs request URLs, and Telegram's contain the token
    if settings.telegram.bot_token is None or not settings.telegram.allowed_user_ids:
        console.print("[red]Set TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_USER_IDS first (docs/SETUP.md §2).[/]")
        raise typer.Exit(code=1)
    costs: Callable[[], str] | None = None
    if settings.openai.api_key is not None:
        ledger = CostLedger(default_ledger_path(settings.app.data_dir))
        fingerprint = key_fingerprint(settings.openai.api_key.get_secret_value())
        costs = lambda: month_summary(ledger, key_fp=fingerprint)  # noqa: E731
    TelegramBot(
        token=settings.telegram.bot_token.get_secret_value(),
        guard=InputGuard(
            allowed_user_ids=set(settings.telegram.allowed_user_ids),
            clock=SystemClock(),
            max_messages=settings.telegram.max_messages_per_10_min,
        ),
        app_factory=lambda: _build(settings, data, model, source="bot"),
        costs_summary=costs,
    ).run()


@app.command()
def costs(
    days: Annotated[int, typer.Option(help="how many days back")] = 30,
    html_out: Annotated[bool, typer.Option("--html/--no-html", help="write the HTML dashboard")] = True,
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="open the dashboard in the browser")] = True,
) -> None:
    """Cost dashboard: OpenAI's billed costs per API key (with OPENAI_ADMIN_KEY) + StrideDJ's own ledger."""
    import webbrowser

    from stridedj.costs.dashboard import build_view, render_html, render_terminal
    from stridedj.costs.openai_costs import OfficialCosts, fetch_official_costs
    from stridedj.errors import StrideDJError

    settings = load_settings()
    ledger = CostLedger(default_ledger_path(settings.app.data_dir))
    now = datetime.now(UTC)
    runs = ledger.runs(since=now - timedelta(days=days))
    official: OfficialCosts | None = None
    if settings.openai.admin_key is not None:
        try:
            official = asyncio.run(fetch_official_costs(settings.openai.admin_key, days=days, now=now))
        except StrideDJError as exc:
            console.print(f"[yellow]Official costs unavailable ({exc}); showing the ledger only.[/]")
    view = build_view(ledger_runs=runs, official=official, current_key=settings.openai.api_key, days=days, now=now)
    render_terminal(view, console)
    if html_out:
        path = settings.app.data_dir / "costs.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_html(view), encoding="utf-8")
        console.print(f"Dashboard: {path}")
        if open_browser:
            webbrowser.open(path.as_uri())


@app.command()
def version() -> None:
    """Print the installed version."""
    from importlib.metadata import version as pkg_version

    console.print(pkg_version("stridedj"))


if __name__ == "__main__":  # pragma: no cover
    app()
