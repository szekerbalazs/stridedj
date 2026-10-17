"""Command-line entry point (`stridedj ...`). Thin: parses arguments and delegates to the composition root."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
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


async def _build(settings: Settings, data: DataMode, model: str | None) -> App:
    openai_settings = settings.openai.model_copy(update={"model": model}) if model else settings.openai
    llm = OpenAIResponsesClient.from_settings(openai_settings)
    if data is DataMode.DEMO:
        return await build_demo_app(
            llm, now=datetime.now(UTC), timezone=settings.athlete.timezone, traces_dir=settings.app.data_dir / "traces"
        )
    return await build_live_app(settings, llm)


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
        await _chat_loop(await _build(settings, data, model))

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
    TelegramBot(
        token=settings.telegram.bot_token.get_secret_value(),
        guard=InputGuard(allowed_user_ids=set(settings.telegram.allowed_user_ids), clock=SystemClock()),
        app_factory=lambda: _build(settings, data, model),
    ).run()


@app.command()
def version() -> None:
    """Print the installed version."""
    from importlib.metadata import version as pkg_version

    console.print(pkg_version("stridedj"))


if __name__ == "__main__":  # pragma: no cover
    app()
