"""Telegram front-end (python-telegram-bot, long polling).

Long polling means the bot opens the connection to Telegram (outbound only): no public URL, no TLS certificate, no
webhook secret to manage, and the container needs no inbound port. A webhook is the better fit for serverless
hosting or many users (see docs/HOSTING.md).

Per message: guard (private chat, allowlist, secret detection, length, rate) → agent → reply rendered by code.
Buttons carry only a random approval token (`a:<token>` / `c:<token>`, under Telegram's 64-byte limit), the token and
a music choice (`m:<token>:d`: 0 / 50 / 100 % new songs, the playlist is redrafted in code, no model call), an interval
session id (`x:i2`: a tap becomes a code-made message to the agent), or, under a created playlist, a draft id and a
song position for feedback (`bot/feedback.py`).

In the background, every 30 minutes: the latest Spotify plays are pulled into the plays log (Spotify only remembers
the last 50, and the music played during a run is the strongest signal of what the runner runs to), the artist
genre cache is warmed, and new runs are pulled from Garmin (before, only at start-up).
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from collections.abc import Set as AbstractSet
from typing import Any

from telegram import CallbackQuery, Chat, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatAction, ChatType, ParseMode
from telegram.error import Conflict, InvalidToken, NetworkError
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from stridedj.app import App
from stridedj.bot.feedback import (
    Button,
    OpenSongs,
    RatePlaylist,
    ToggleSong,
    is_feedback,
    parse_feedback,
    playlist_buttons,
    render_song_list,
    song_buttons,
)
from stridedj.bot.render import render_execution, render_proposal, session_button_label
from stridedj.domain import MusicChoice, ReplyKind, RunPlan, TrackId, UserContext, UserId
from stridedj.errors import ApprovalError, StrideDJError
from stridedj.harness.guard import InputGuard

log = logging.getLogger(__name__)

_SESSION_ID = re.compile(r"^i\d{1,3}$")  # interval session ids are code-made: i1, i2, ...
PLAYS_SYNC_EVERY_S = 30 * 60  # 50 plays ≈ 3 hours of listening: every 30 min loses nothing
RUNS_REFRESH_EVERY_S = 30 * 60  # a finished run is known within half an hour (one Garmin list call per round)
MUSIC_BUTTONS: tuple[tuple[str, MusicChoice, str], ...] = (  # callback code, choice, label
    ("f", MusicChoice.FAMILIAR, "0% new"),
    ("u", MusicChoice.USUAL, "50% new"),
    ("d", MusicChoice.DISCOVER, "100% new"),
)

WELCOME = (
    "Hi! I'm StrideDJ. Tell me what you want to run, e.g. <i>10k Z2 tonight</i> or "
    "<i>intervals tomorrow morning, 8 km</i>. I'll find a slot in your calendar and a playlist matched to your "
    "cadence. Nothing is booked until you tap Approve."
)


class TelegramBot:
    def __init__(
        self,
        *,
        token: str,
        guard: InputGuard,
        app_factory: Callable[[], Awaitable[App]],
        costs_summary: Callable[[], str] | None = None,
    ) -> None:
        self._guard = guard
        self._app_factory = app_factory
        self._costs_summary = costs_summary
        self._stride: App | None = None
        self._background: list[asyncio.Task[None]] = []
        # The agent is built inside PTB's own event loop (post_init), so its async clients belong to that loop.
        self._app = (
            Application.builder().token(token).post_init(self._post_init).post_shutdown(self._post_shutdown).build()
        )
        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(CommandHandler("costs", self._on_costs))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text))
        self._app.add_handler(CallbackQueryHandler(self._on_button))
        self._app.add_error_handler(self._on_error)

    async def _post_init(self, _: Application[Any, Any, Any, Any, Any, Any]) -> None:
        self._stride = await self._app_factory()
        jobs = [(self._stride.refresh_music, PLAYS_SYNC_EVERY_S, 0.0)]
        # runs were refreshed while the app was built: their first background round comes one period later
        jobs.append((self._stride.refresh_runs, RUNS_REFRESH_EVERY_S, RUNS_REFRESH_EVERY_S))
        for job, every_s, first_after_s in jobs:
            if job is not None:
                self._background.append(
                    asyncio.create_task(sync_forever(job, every_s=every_s, first_after_s=first_after_s))
                )
        log.info("StrideDJ ready (timezone %s)", self._stride.timezone)

    async def _post_shutdown(self, _: Application[Any, Any, Any, Any, Any, Any]) -> None:
        for task in self._background:
            task.cancel()

    @property
    def _s(self) -> App:
        if self._stride is None:
            raise RuntimeError("bot not initialised")
        return self._stride

    def run(self) -> None:
        """Blocking: poll Telegram until interrupted."""
        self._app.run_polling(allowed_updates=[Update.MESSAGE, Update.CALLBACK_QUERY], drop_pending_updates=True)

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """One readable line per problem instead of a traceback loop; stop when polling can't work at all."""
        error = context.error
        if isinstance(error, Conflict):
            log.error(
                "Another copy of this bot is already running: Telegram delivers a bot's messages to one program only. "
                "Stop the other copy (Ctrl+C in its terminal, or `pkill -f 'stridedj bot'`), then start this one again."
            )
            context.application.stop_running()
            return
        if isinstance(error, InvalidToken):
            log.error("Telegram rejected the bot token: check TELEGRAM_BOT_TOKEN in .env (docs/SETUP.md §2).")
            context.application.stop_running()
            return
        if isinstance(error, NetworkError):
            log.warning("Telegram is unreachable right now (%s); retrying automatically.", type(error).__name__)
            return
        log.error("Unexpected error while handling a message", exc_info=error)
        if isinstance(update, Update) and update.effective_chat is not None:
            await update.effective_chat.send_message("Something went wrong on my side. Please try again in a minute.")

    def _user(self, telegram_user_id: int) -> UserContext:
        return UserContext(user_id=UserId(f"tg:{telegram_user_id}"), timezone=self._s.timezone)

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None or update.message is None:
            return
        decision = self._guard.check(user_id=update.effective_user.id, text="/start")
        await update.message.reply_text(
            WELCOME if decision.allowed else (decision.notice or ""), parse_mode=ParseMode.HTML
        )

    async def _on_costs(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """`/costs`: this month's StrideDJ spend from the ledger (allowlisted users only)."""
        if update.effective_user is None or update.message is None:
            return
        decision = self._guard.check(user_id=update.effective_user.id, text="/costs")
        if not decision.allowed:
            if decision.notice:
                await update.message.reply_text(decision.notice)
            return
        text = self._costs_summary() if self._costs_summary else "Cost tracking is not configured."
        await update.message.reply_text(text)

    async def _on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message, sender, chat = update.message, update.effective_user, update.effective_chat
        if message is None or sender is None or chat is None or message.text is None:
            return
        if chat.type != ChatType.PRIVATE:
            return  # never act in groups: other members' messages would be untrusted input
        decision = self._guard.check(user_id=sender.id, text=message.text)
        if not decision.allowed:
            log.info("blocked message: %s", decision.reason)
            if decision.reason == "secret_detected":
                try:
                    await message.delete()
                except Exception:
                    log.warning("could not delete a message containing a secret")
            if decision.notice:
                await chat.send_message(decision.notice)
            return

        await self._converse(chat, sender.id, message.text)

    async def _converse(self, chat: Chat, sender_id: int, text: str) -> None:
        """One agent turn and its code-rendered reply: a proposal card with Approve/Cancel, interval sessions as
        buttons, or just the (escaped) message."""
        await chat.send_action(ChatAction.TYPING)
        user = self._user(sender_id)
        response = await self._s.agent.handle(user, text)
        reply = html.escape(response.text, quote=False)  # placeholders already filled in by code
        if response.reply.kind is ReplyKind.PROPOSAL and response.plan is not None and response.approval_token:
            tracks = await self._s.playlists.tracks_for(response.plan.playlist) if response.plan.playlist else []
            card = render_proposal(response.plan, tracks, user.tz, rich=True)
            keyboard = _markup(proposal_buttons(response.plan, response.approval_token))
            await chat.send_message(f"{reply}\n\n{card}", parse_mode=ParseMode.HTML, reply_markup=keyboard)
        elif response.session_options:
            options = [[(session_button_label(o), f"x:{o.session_id}")] for o in response.session_options]
            await chat.send_message(reply, parse_mode=ParseMode.HTML, reply_markup=_markup(options))
        else:
            await chat.send_message(reply, parse_mode=ParseMode.HTML)

    async def _on_session_pick(self, query: CallbackQuery, chat: Chat) -> None:
        """A tapped interval session becomes a code-made message to the agent ("I choose interval session i2: ...").
        It goes through the same guard as typed text (allowlist, rate limit): it costs a model call."""
        session_id = query.data.removeprefix("x:") if query.data else ""
        user = self._user(query.from_user.id)
        picked = (
            self._s.agent.session(user).state.sessions.get(session_id) if _SESSION_ID.fullmatch(session_id) else None
        )
        if not self._guard.is_allowed(query.from_user.id) or picked is None:
            await query.answer("That option is no longer available." if picked is None else None)
            return
        text = f"I choose interval session {picked.session_id}: {picked.main_set}, {picked.total_km:g} km in total."
        decision = self._guard.check(user_id=query.from_user.id, text=text)
        if not decision.allowed:
            await query.answer(decision.notice)
            return
        await query.answer()
        await query.edit_message_reply_markup(reply_markup=None)
        await chat.send_message(f"➡️ {html.escape(session_button_label(picked), quote=False)}")
        await self._converse(chat, query.from_user.id, text)

    async def _on_music_pick(self, query: CallbackQuery, chat: Chat) -> None:
        """0 / 50 / 100 % new songs: the same plan with only its playlist redrafted (no model call), sent as a new card
        with new buttons. The old card keeps its buttons until the new one exists (a failed draft changes nothing)."""
        head, _, code = (query.data or "").rpartition(":")
        token = head.removeprefix("m:")
        picked = next(((choice, label) for c, choice, label in MUSIC_BUTTONS if c == code), None)
        if not self._guard.is_allowed(query.from_user.id) or picked is None:
            await query.answer()
            return
        music, label = picked
        await query.answer(f"Building a playlist with {label} songs…")
        await chat.send_action(ChatAction.TYPING)
        user = self._user(query.from_user.id)
        try:
            plan, new_token = await self._s.agent.redraft(token, user_id=user.user_id, music=music)
        except ApprovalError as exc:
            await chat.send_message(f"Can't change the playlist: {exc}")
            return
        except (TimeoutError, StrideDJError) as exc:
            log.warning("playlist redraft failed: %r", exc)
            await chat.send_message("The music services are slow right now. The plan above still stands; try again.")
            return
        await query.edit_message_reply_markup(reply_markup=None)
        tracks = await self._s.playlists.tracks_for(plan.playlist) if plan.playlist else []
        card = render_proposal(plan, tracks, user.tz, rich=True)
        await chat.send_message(
            f"Same run, new playlist ({label} songs).\n\n{card}",
            parse_mode=ParseMode.HTML,
            reply_markup=_markup(proposal_buttons(plan, new_token)),
        )

    async def _on_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query, chat = update.callback_query, update.effective_chat
        if query is None or query.data is None or query.from_user is None or chat is None:
            return
        if is_feedback(query.data):
            await self._on_feedback(query, query.data)
            return
        if query.data.startswith("x:"):
            await self._on_session_pick(query, chat)
            return
        if query.data.startswith("m:"):
            await self._on_music_pick(query, chat)
            return
        await query.answer()
        action, _, token = query.data.partition(":")
        user_id = self._user(query.from_user.id).user_id
        if action == "c":
            self._s.agent.cancel(token, user_id=user_id)
            await query.edit_message_reply_markup(reply_markup=None)
            await chat.send_message("Cancelled. Nothing was booked.")
            return
        if action != "a":
            return
        try:
            result = await self._s.agent.approve(token, user_id=user_id)
        except ApprovalError as exc:
            await chat.send_message(f"Can't do that: {exc}")
            return
        await query.edit_message_reply_markup(reply_markup=None)
        text, markup = render_execution(result, rich=True), None
        if result.playlist_draft_id is not None and self._s.history is not None:
            text += "\n\nAfter your run: how was the music?"
            markup = _markup(playlist_buttons(result.playlist_draft_id))
        await chat.send_message(text, parse_mode=ParseMode.HTML, reply_markup=markup)

    async def _on_feedback(self, query: CallbackQuery, data: str) -> None:
        """👍 / 👎 for a created playlist, the song list, and per-song 👎 toggles. Only ids are stored."""
        history = self._s.history
        action = parse_feedback(data)
        if not self._guard.is_allowed(query.from_user.id) or history is None or action is None:
            await query.answer()
            return
        track_ids = history.track_ids(action.draft_id)
        if not track_ids:  # not a playlist StrideDJ created (or a forged button)
            await query.answer("I don't know this playlist.")
            return
        if isinstance(action, RatePlaylist):
            history.rate_playlist(action.draft_id, action.rating)
            await query.answer(
                "Noted: more like this." if action.rating > 0 else "Noted: its new songs won't be suggested again."
            )
            await query.edit_message_reply_markup(_markup(playlist_buttons(action.draft_id, rated=action.rating)))
            return
        if isinstance(action, OpenSongs):
            await query.answer()
            tracks = await self._s.playlists.tracks_of(track_ids)
            disliked = _disliked_positions(track_ids, history.disliked_songs())
            if isinstance(query.message, Message):  # None/inaccessible: too old to reply to
                await query.message.reply_text(
                    render_song_list(tracks, disliked),
                    parse_mode=ParseMode.HTML,
                    reply_markup=_markup(song_buttons(action.draft_id, len(tracks), disliked)),
                )
            return
        if isinstance(action, ToggleSong):
            if action.position > len(track_ids):
                await query.answer()
                return
            now_disliked = history.toggle_song_dislike(track_ids[action.position - 1])
            await query.answer(
                f"Song {action.position} won't be suggested again."
                if now_disliked
                else f"Song {action.position}: undone."
            )
            tracks = await self._s.playlists.tracks_of(track_ids)
            disliked = _disliked_positions(track_ids, history.disliked_songs())
            await query.edit_message_text(
                render_song_list(tracks, disliked),
                parse_mode=ParseMode.HTML,
                reply_markup=_markup(song_buttons(action.draft_id, len(tracks), disliked)),
            )


async def sync_forever(sync: Callable[[], Awaitable[int]], *, every_s: float, first_after_s: float = 0.0) -> None:
    """Call `sync` (after `first_after_s`) and then every `every_s` seconds until cancelled; a failed round is
    logged, never fatal."""
    await asyncio.sleep(first_after_s)
    while True:
        try:
            await sync()
        except Exception:
            log.exception("background refresh failed; retrying at the next round")
        await asyncio.sleep(every_s)


def proposal_buttons(plan: RunPlan, token: str) -> list[list[Button]]:
    """Approve / Cancel, and above them for a plan with a playlist: 0 / 50 / 100 % new. The current choice is ticked;
    tapping it again draws another playlist of the same kind."""
    rows: list[list[Button]] = []
    if plan.playlist is not None:
        current = plan.playlist.music
        rows.append(
            [(("✓ " if choice is current else "") + label, f"m:{token}:{c}") for c, choice, label in MUSIC_BUTTONS]
        )
    rows.append([("✅ Approve", f"a:{token}"), ("✖ Cancel", f"c:{token}")])
    return rows


def _markup(rows: list[list[Button]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(label, callback_data=data) for label, data in row] for row in rows]
    )


def _disliked_positions(track_ids: Sequence[TrackId], disliked: AbstractSet[TrackId]) -> set[int]:
    return {position for position, track_id in enumerate(track_ids, start=1) if track_id in disliked}
