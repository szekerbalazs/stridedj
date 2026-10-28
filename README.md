<div align="center">

# StrideDJ

**Text your running agent *"10k Z2 tonight"*. It sizes the run from your own Garmin history, finds a free slot in your calendar, builds a playlist matched to your cadence, and books nothing until you tap Approve.**

[![CI](https://github.com/szekerbalazs/stridedj/actions/workflows/ci.yml/badge.svg)](https://github.com/szekerbalazs/stridedj/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB)
![mypy strict](https://img.shields.io/badge/mypy-strict-2a6db2)
![ruff](https://img.shields.io/badge/lint-ruff-d7ff64)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

</div>

![How StrideDJ works](docs/assets/architecture.svg)

## What it does
```
you        10k Z2 tonight
stridedj   Your 10 km Z2 run is booked for Sat 17 Oct, 18:30, about 57 min. A 16-track playlist at 168 steps/min.

           🏃 10 km Z2 easy run
           📅 Sat 17 Oct, 18:30–19:40
           ⏱ ~57 min typical, ≤ 61 min in 4 of 5 runs
              Based on 16 z2 runs of 7.5–12.5 km in the last 90 days, scaled to 10 km.
           👟 cadence target 168 spm (8 recent runs)
           🎵 16 tracks · 59 min · ±1.2 BPM
                                                        [ ✅ Approve ]  [ ✖ Cancel ]
```
On **Approve**, it creates the calendar event and a private Spotify playlist, exactly once.

Interval runs come with choices:
```
you        intervals tomorrow evening
stridedj   Which session would you like?
           [ 8 × 400 m @ 3:45 · 6.7 km · ~46′ ]
           [ 6 × 800 m @ 4:10 · 8.3 km · ~52′ ]
           [ Pyramid 400-1200 m · 7.1 km · ~46′ ]
```
The tap plans that session; the card and the calendar event list it ("2 km warm-up · 6 × 800 m @ 4:10, 90 s standing rest · 1.5 km cool-down"). After the run, 👍 / 👎 and per-song feedback teach the playlist side what you run to.

**How it decides each part:**
- **Duration** comes from *your* comparable runs. Without recent ones it uses a personal log-log model (the classic Riegel law, fitted to you) and says whether it interpolates or extrapolates. With only a few comparable runs, the 80 % time keeps at least your typical run-to-run spread.
- **The calendar slot** is the 80th percentile plus 5 minutes, rounded up to 5 minutes: a newsvendor choice, because running over is worse than a few idle minutes. Interval sessions count their rests (elapsed time); steady runs count moving time.
- **Cadence target:** the median of your last runs of that type. For intervals, only the reps count, not the warm-up or the rests.
- **Interval sessions** are detected in your history from the per-second data: pace for the reps, watch stops and slow stretches for the rests, heart rate dropping over each rest as confirmation. Garmin's 1-km auto-laps can't show any of that.
- **Session options** use distance reps at your target paces. The targets adapt after each session: faster after a clean one, unchanged after a miss or a fade, easier after six weeks without that distance.
- **Playlist:** a *new* playlist in the style of your running music, every track's tempo folded (½×, 1×, 2×) to your cadence. No spoken word, your own library first, songs from recent playlists rotated out. Ask for "only songs I know", "something new" or a genre.
- **Heart-rate zones** are read from Garmin Connect, so "Z2" means what it means on your watch.
- **Follow-ups and questions:** *"make it 8:00 instead"*, *"calmer music"*, *"how are my Z2 runs going?"*, *"what could I run a half marathon in?"*

## How it works
StrideDJ is a **tool-calling agent** inside a **hand-built harness**, with no agent framework. Every safety-relevant decision is a line of code you can point to.

1. **Guard.** Your Telegram message passes deterministic checks before any model sees it: allowlisted user, private chat, secret detector, length and rate limits. Button presses are parsed strictly and checked against the allowlist.
2. **Context.** A versioned system prompt, the current date/time in your timezone plus a 15-day date table (so the model looks dates up instead of computing them), and a compact history (earlier messages and answers, no stale tool output).
3. **Loop.** The OpenAI Responses API (stateless, `store=false`) calls eight **read-only** tools with strict JSON schemas until it returns a final answer, a strict `AgentReply` JSON object. The harness enforces budgets: model calls, tool calls, $, wall time.
4. **Code owns the facts.** The model refers to slots, playlist drafts and interval sessions by id. A proposal's message names its facts as placeholders (`"Booked {start}, about {expected}."`); code fills them in from the plan and checks every other number in the message.
5. **Approval.** The proposal card is rendered by code. Its token is bound to the plan's content hash and to you, single use, valid for 15 minutes.
6. **Execution.** Only the executor writes: one calendar event (idempotent id) and one private playlist, from fixed templates.
7. **Background.** Every 30 minutes the bot pulls your latest Spotify plays (Spotify keeps only 50), warms the genre cache and fetches new runs with their per-second series.
8. **Transcript.** Every run is recorded (model calls, tool calls, tokens, cost, latency). The evals grade these transcripts and outcomes.

### What the model can and can't see or do
| The model sees | The model never sees | The model can't |
|---|---|---|
| your message, aggregates (durations, cadence, slot times, session summaries, playlist stats), ids | GPS, raw HR/pace series, activity names, song titles, calendar event titles, crowd-written tags, secrets | write anything, add attendees, send messages, edit your playlists, put a number of its own into a proposal |

Untrusted text (song titles, calendar invites, Last.fm tags) never enters the model's context, and the model has no write channel. Prompt injection is mitigated **by architecture, not by asking the model nicely**: the "lethal trifecta" is broken. Details in [docs/INJECTION.md](docs/INJECTION.md).

## Quickstart (demo mode, about 5 minutes)
Demo mode runs the *same agent* on synthetic runs, a fake calendar and a fake music library. You only need an OpenAI API key.

1. Clone and install. `make setup` creates a Python 3.12 environment with uv, installs the dependencies and creates `.env` from the template:
   ```bash
   git clone https://github.com/szekerbalazs/stridedj.git && cd stridedj
   make setup
   ```
2. Put your `OPENAI_API_KEY` into `.env` ([docs/SETUP.md §1](docs/SETUP.md#1-openai-api-key)).
3. Chat in the terminal and try *"10k Z2 tonight"* or *"intervals tomorrow evening"*:
   ```bash
   make demo
   ```

Requirements: macOS or Linux and [uv](https://docs.astral.sh/uv/). uv fetches Python 3.12 itself.

## Full setup (your real accounts)
Follow **[docs/SETUP.md](docs/SETUP.md)**: click-by-click, about 45 minutes, with `make doctor` confirming each step.

| Service | What you create | Notes |
|---|---|---|
| OpenAI | API key + a few $ of credit | ChatGPT subscriptions don't include API usage |
| Telegram | a bot via @BotFather + your numeric user id | long polling: no server or public URL needed |
| Spotify | a developer app (redirect `http://127.0.0.1:8888/callback`) | Premium required for dev-mode apps; your running playlist must be yours |
| Google Calendar | a Desktop OAuth client | free/busy + insert scopes; testing-mode logins last 7 days |
| Garmin Connect | your login (MFA supported) | unofficial API; HR zones are read from your account |
| Last.fm (optional) | an API key | artist genres; Spotify returns none to dev-mode apps |

Then the one-time logins, the per-second history and a check:
```bash
uv run stridedj auth garmin && uv run stridedj auth spotify && uv run stridedj auth google
uv run stridedj series
make doctor
make smoke
make bot
```
- `uv run stridedj series`: downloads the per-second series of your past runs once (for interval detection; no GPS). New runs get theirs automatically.
- `make doctor`: ✓/✗ per credential, never prints values
- `make smoke`: read-only live check of every integration (counts only)
- `make bot`: start the Telegram bot

### `.env` reference
All configuration and secrets live in `.env`, which is git- and docker-ignored. The template is [`.env.example`](.env.example).

| Variable | Required | Where it comes from |
|---|---|---|
| `OPENAI_API_KEY` | always | platform.openai.com → API keys |
| `OPENAI_MODEL`, `OPENAI_REASONING_EFFORT` | no | defaults `gpt-6-luna`, `low` |
| `OPENAI_ADMIN_KEY` | no | platform.openai.com → Settings → Organization → Admin keys (only for `stridedj costs`) |
| `TELEGRAM_BOT_TOKEN` | for the bot | @BotFather |
| `TELEGRAM_ALLOWED_USER_IDS` | for the bot | @userinfobot (numeric ids, comma-separated) |
| `TELEGRAM_MAX_MESSAGES_PER_10_MIN` | no | default 30; button presses don't count |
| `SPOTIFY_CLIENT_ID` | live mode | Spotify dashboard → your app |
| `SPOTIFY_RUNNING_PLAYLIST_IDS` | live mode | the 22-char id from a playlist's share link |
| `SPOTIFY_REFRESH_TOKEN` | live mode | written by `stridedj auth spotify` |
| `LASTFM_API_KEY` | no | last.fm/api/account/create (the key, not the shared secret) |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | live mode | Google Cloud → Google Auth Platform → Clients |
| `GOOGLE_CALENDAR_ID`, `GOOGLE_BUSY_CALENDAR_IDS` | no | default `primary`; a dedicated "Runs" calendar is recommended |
| `GOOGLE_REFRESH_TOKEN` | live mode | written by `stridedj auth google` |
| `GARMIN_EMAIL`, `GARMIN_PASSWORD` | live mode | your Garmin Connect login |
| `STRIDEDJ_RUN_SOURCE` | no | `demo` (default), `garmin`, or `intervals_icu` (plan B) |
| `ATHLETE_TIMEZONE` | no | default `Europe/Zurich` |
| `ATHLETE_INTERVAL_TARGET_PACES`, `ATHLETE_INTERVAL_TARGETS_SINCE` | no | starting interval targets (`400=3:45,800=4:10`) and the day you set them; without them, targets come from your last sessions |
| `GARMIN_TOKEN_DIR`, `STRIDEDJ_DATA_DIR`, `STRIDEDJ_LOG_LEVEL` | no | defaults `~/.garminconnect`, `data/`, `INFO` |

## Commands
| Command | What it does |
|---|---|
| `make demo` | terminal chat on synthetic data |
| `uv run stridedj chat --data live` | terminal chat on your real accounts |
| `make bot` | Telegram bot (`--data demo` also works) |
| `uv run stridedj auth <garmin\|spotify\|google>` | one-time logins |
| `uv run stridedj series` | download per-second series for runs that don't have one yet |
| `make doctor` / `make smoke` | credential check / live read-only integration check |
| `uv run stridedj costs` | cost dashboard per API key (terminal + HTML); `/costs` in Telegram |
| `uv run python scripts/playlist_quality.py` | read-only playlist quality metrics on your real library |
| `make eval-reference` / `make eval` | evals with the oracle (free) / with the real model (≈ $0.05–0.10) |
| `make check` | ruff + mypy --strict + 500+ offline tests (what CI runs) |

## Costs
Every conversation is recorded in a local **cost ledger** (`data/costs.db`): tokens, cache hits and estimated $ per run, grouped by a short **fingerprint** of the API key (never the key itself).
```bash
uv run stridedj costs
```
- `uv run stridedj costs`: terminal tables + an HTML dashboard in your browser (one section per API key)
- **With an optional OpenAI Admin key** (`OPENAI_ADMIN_KEY`, see [SETUP §1](docs/SETUP.md#1-openai-api-key)), the dashboard also shows **OpenAI's billed costs per API key** from the official Costs API: exact amounts, daily, named keys, per line item.
- **Least privilege:** the Admin key is used only by this command, never by the bot.
- **In Telegram:** `/costs` gives this month's StrideDJ spend from the ledger.

A conversation costs about $0.0005 on average; most input tokens come from the prompt cache.

## Evals
The agent is evaluated like a product, following Anthropic's [*Demystifying evals for AI agents*](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).
- **Tasks:** 45, in three suites:
  - **regression (15):** everyday requests;
  - **capability (24):** follow-ups, named start times, interval sessions (described, offered as buttons, sized to a total), music and genre choices, extrapolation, unit conversion, a fully booked evening;
  - **security (6):** poisoned invites and track titles, exfiltration and prompt extraction attempts.
- **Trials:** each runs in a fresh, isolated environment with a frozen clock.
- **Grading:** programmatic only, on **outcomes**: the plan object (run type, distance, date, start time, session), the tool arguments that matter (mood, music, genre), whether a proposal's message needed replacing, and whether anything was written. No LLM judge.
- **Reference oracle:** every task has one, which must score 100 %; that proves the tasks and graders are sound. CI fails below 100 %.
- **Metrics:** pass@k and pass^k, attack success rate, cost and latency.

Latest results (gpt-6-luna, prompt v4, every task run 3 times):

| Suite | Passed | Note |
|---|---|---|
| Everyday requests | 100 % | |
| Harder requests | 96–97 % | the misses: a rounding slip on "10 miles" (16.0 instead of 16.1 km) and single trials that wrote an unchecked number (replaced by code) |
| Security | 100 % | 0 of 18 attack attempts succeeded |

A full run (135 conversations) costs about $0.05–0.10. Details in [docs/EVALS.md](docs/EVALS.md).

## Project layout
```
src/stridedj/
├── domain/        pydantic contracts for everything that crosses a boundary (frozen, extra="forbid")
├── ports.py       Protocols for every external dependency → real adapters + in-memory fakes
├── llm/           OpenAI Responses adapter, strict-schema converter, scripted fake models
├── harness/       the agent harness: loop, tools, context, guard, approvals, executor, tracing
├── agent/         StrideDJ-specific: system prompts (versioned), the eight tools, session, placeholders
├── runs/          Garmin / intervals.icu / demo sources, SQLite store, run typing, duration models,
│                  interval detection, pace targets, session options
├── cal/           Google Calendar (free/busy, idempotent insert), DST-safe slot finding, exact-start checks
├── music/         Spotify (PKCE), ReccoBeats, Last.fm genres, style model, playlist assembly, history and feedback
├── bot/           Telegram front-end: code-side rendering, approval / session / feedback buttons, background jobs
├── app.py         composition root: demo wiring vs live wiring
└── cli.py         stridedj chat | bot | auth | series | doctor | costs
evals/             tasks (YAML), graders, reference oracle, runner, transcript viewer
scripts/           read-only live checks (integrations, playlist quality)
tests/             500+ offline unit tests (fakes for every API, synthetic data only)
docs/              setup, architecture, harness, evals, prompting, injection, hosting, LiveKit, decisions
```

## Deployment
The bot uses long polling, so it needs **no inbound port**. It runs anywhere Docker runs: a laptop, a €5 VM, Fly.io.
```bash
docker compose build
docker compose run --rm -it bot auth garmin
docker compose up -d
```
- `docker compose run --rm -it bot auth garmin`: one-time login inside the container
- `docker compose up -d`: non-root, read-only filesystem, secrets from .env, data in a volume

More, including hosting the *model* yourself (vLLM, GPU memory math), is in [docs/HOSTING.md](docs/HOSTING.md).

## Documentation
| Doc | Read it for |
|---|---|
| [SETUP](docs/SETUP.md) | accounts, credentials, troubleshooting |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | components, request path, trust boundaries, background jobs |
| [HARNESS](docs/HARNESS.md) | the agent loop and why it's hand-built |
| [EVALS](docs/EVALS.md) | tasks, graders, pass@k vs pass^k, results |
| [PROMPTING](docs/PROMPTING.md) | prompt structure, techniques, changelog |
| [INJECTION](docs/INJECTION.md) | threat model, the lethal trifecta, controls and tests |
| [LLM_APIS](docs/LLM_APIS.md) | how LLM APIs work; OpenAI vs Anthropic |
| [HOSTING](docs/HOSTING.md) | hosting agents and models |
| [LIVEKIT](docs/LIVEKIT.md) | real-time voice agents and how they'd fit |
| [DECISIONS](docs/DECISIONS.md) | the decision log: options and reasons |

## Limitations and next steps
- **Uncertainty is still rough.** The 80 % time is an empirical quantile or a residual quantile of the personal model. Next: proper prediction intervals and shrinkage toward the personal model.
- **Interval detection is rule-based** and tuned on one runner's labelled runs. The constants are named, the series are stored, and a learned classifier becomes worth it once there are enough labelled sessions.
- **Spotify development mode** has a small request quota and no artist genres; StrideDJ honours long `Retry-After`s and takes genres from Last.fm.
- **Unofficial Garmin access** can break when Garmin changes its login. The `RunSource` port and the intervals.icu plan B exist for exactly that.
- **Single user.** Multi-user would need per-user OAuth connect links, an encrypted token store and tenant isolation. The harness already passes a `UserContext` everywhere.
- **Sessions and approvals are in memory**, so a restart forgets pending proposals. Postgres is the next step for multi-instance hosting.
- **Next features:** a progress dashboard (VO2max, HRV, race predictions, training load and form), then a training plan.

## License
[MIT](LICENSE)
