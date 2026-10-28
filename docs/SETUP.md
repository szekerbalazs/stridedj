# Setup guide

Everything StrideDJ needs, step by step: about 45 minutes the first time. You can stop after **step 1** and run the full agent in **demo mode** (synthetic runs, a fake calendar and music library). Every further step connects one real service.

| Step | Service | Needed for | Time |
|---|---|---|---|
| 0 | Install | everything | 2 min |
| 1 | OpenAI API key | the agent (also demo mode) | 10 min |
| 2 | Telegram bot | chatting from your phone | 5 min |
| 3 | Spotify app | real playlists | 10 min |
| 4 | Google OAuth client | real calendar slots | 15 min |
| 5 | Garmin login | your real runs | 5 min |
| 6 | One-time logins | Spotify, Google, Garmin sessions | 5 min |
| 7 | Last.fm API key (optional) | genres for playlists | 3 min |

After each step run `make doctor`: it checks every credential (live where possible) and prints only ✓/✗/… and a hint, **never the values**.

**How secrets are handled:**
- **One file:** all secrets live in `.env` in the project root, created from the template. `.env` is git- and docker-ignored.
- **Never paste a key, token or password into a chat**, whether Telegram, an AI assistant or an issue. If one leaks, revoke it and create a new one.
- Tokens created by the logins in step 6 are written into `.env` for you. The only exception is Garmin's session file, see step 5.

---

## 0. Install
Requirements: macOS or Linux, [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). uv installs Python 3.12 by itself.
```bash
git clone https://github.com/szekerbalazs/stridedj.git && cd stridedj
make setup
```
- `make setup`: creates .venv with Python 3.12, installs dependencies, copies .env.example → .env

## 1. OpenAI API key
ChatGPT subscriptions do **not** include API usage; the API is billed separately with prepaid credit.
1. Sign in at <https://platform.openai.com>.
2. **Settings → Billing**: add a payment method and about **$5–10** of credit. Leave auto-recharge off.
3. **Settings → Limits:** set a monthly budget (e.g. $15) with an email alert.
4. **API keys → Create new secret key**, in the default project. *Restricted* permissions are enough: *Responses: Write*, *Models: Read*. Copy it now; it's shown only once.
5. In `.env`: `OPENAI_API_KEY=sk-...`

✅ `make doctor` → `openai: api key ✓`. **You can now run the demo:** `make demo`.

**Optional: an Admin key for exact per-key costs** (`uv run stridedj costs`).
1. platform.openai.com → **Settings → Organization → Admin keys → Create**. Only organization **owners** can create one; on a personal account, that's you. If it offers permissions, choose read-only / *Usage*.
2. `.env`: `OPENAI_ADMIN_KEY=sk-admin-...`
3. Treat it as the most sensitive secret here: it can manage your organization. StrideDJ uses it only in the `costs` command, never in the bot.

Without it, the dashboard shows StrideDJ's own ledger (estimates from token counts).

## 2. Telegram bot
1. In Telegram, open **@BotFather** → `/newbot` → pick a name and a username ending in `bot`.
2. Copy the token (`1234567890:AA...`) → `.env`: `TELEGRAM_BOT_TOKEN=...`
3. Open **@userinfobot** → it replies with your numeric id → `.env`: `TELEGRAM_ALLOWED_USER_IDS=123456789`. Bots are publicly findable; this allowlist is what keeps strangers out.
4. Open your bot and press **Start**. Bots can't message you first.

✅ `make doctor` → `telegram: bot token ✓ bot @yourbot`, `telegram: allowed users ✓`.

## 3. Spotify developer app
Requirements:
- **Spotify Premium:** since Feb 2026, development-mode apps need a Premium owner.
- **Running playlists you own:** the API only lets apps read playlists you own or collaborate on.

1. <https://developer.spotify.com/dashboard> → **Create app**.
   - **Redirect URI:** `http://127.0.0.1:8888/callback` (exactly this; `localhost` is rejected).
   - **API:** *Web API* → Save.
2. App **Settings** → copy the **Client ID** → `.env`: `SPOTIFY_CLIENT_ID=...`. No client secret is needed: the login uses PKCE.
3. **User Management:** add your name and your Spotify account's email (dev-mode apps work only for listed users, max 5).
4. Your running playlist(s):
   1. In Spotify: playlist → **⋯ → Share → Copy link**. The ID is the 22 characters between `/playlist/` and `?`.
   2. `.env`: `SPOTIFY_RUNNING_PLAYLIST_IDS=id1,id2`.
   3. If it's a Spotify-made playlist, copy it first (⋯ → *Add to other playlist → New playlist*) and use the copy's ID.

✅ `make doctor` → `spotify: client id ✓`, `spotify: running playlists ✓`.

## 4. Google Calendar OAuth client
1. <https://console.cloud.google.com> → **New project** (e.g. `stridedj`) → select it.
2. **APIs & Services → Library** → *Google Calendar API* → **Enable**.
3. **Google Auth Platform** (older UI: *OAuth consent screen*) → **Get started**: app name, your email, Audience **External** → Create.
4. **Audience → Test users → Add users** → your Google account.
   - In *Testing* mode Google expires the login after **7 days**. Re-run `stridedj auth google` before a demo, or publish the app (you'll see an "unverified app" warning, which is fine for personal use).
5. **Clients → Create client** → type **Desktop app** → copy the **Client ID** and **Client secret** → `.env`: `GOOGLE_CLIENT_ID=...`, `GOOGLE_CLIENT_SECRET=...`. You don't need the JSON download.
6. Recommended: a dedicated calendar for runs.
   1. <https://calendar.google.com> → *Other calendars* **+** → *Create new calendar* → `Runs`.
   2. Open its settings → *Integrate calendar* → **Calendar ID**.
   3. `.env`: `GOOGLE_CALENDAR_ID=<that id>` and `GOOGLE_BUSY_CALENDAR_IDS=primary,<that id>`. Availability is checked on both.

✅ `make doctor` → `google: oauth client ✓`.

Scopes requested:
- `calendar.freebusy`: busy/free times only, never event titles;
- `calendar.events`: to insert the run.

## 5. Garmin
1. `.env`: `GARMIN_EMAIL=...`, `GARMIN_PASSWORD=...`, `STRIDEDJ_RUN_SOURCE=garmin`. If your password contains `#`, wrap it in single quotes.
2. That's all. Your **heart-rate zones are read from Garmin Connect** at startup (the RUNNING profile, otherwise DEFAULT), so "Z2" means exactly what it means on your watch.

Notes:
- `garminconnect` is **unofficial**: Garmin's official API is for business partners only. It logs in like the Garmin Connect app.
- Garmin sometimes rate-limits logins (HTTP 429). The library then tries several login methods. Log in once and reuse the session.
- The session lives in `GARMIN_TOKEN_DIR` (default `~/.garminconnect/garmin_tokens.json`, mode 0600), outside `.env`, because the library manages and refreshes it. Treat it like a password.

**Plan B (if the Garmin login doesn't work):** [intervals.icu](https://intervals.icu).
1. Create a free account and connect Garmin there (Garmin syncs to it officially).
2. **Settings → Developer Settings** → create an API key.
3. `.env`: `INTERVALS_ICU_API_KEY=...`, `INTERVALS_ICU_ATHLETE_ID=i12345`, `STRIDEDJ_RUN_SOURCE=intervals_icu`.

## 6. One-time logins
```bash
uv run stridedj auth garmin
uv run stridedj auth spotify
uv run stridedj auth google
make doctor
make smoke
```
- `auth garmin`: asks for the MFA code if you use one.
- `auth spotify`: opens the browser. Approve, and the refresh token is written to `.env`.
- `auth google`: opens the browser. At *"Google hasn't verified this app"*, click **Advanced → Continue** and tick **both** permissions; the refresh token is written to `.env`.
- `make doctor`: everything should be ✓.
- `make smoke`: a read-only live check that prints counts only.

Then download the per-second series of your past runs once (pace, heart rate, cadence; Garmin's position fields are never read or stored). They let StrideDJ find your interval sessions, which Garmin's 1-km auto-laps hide:
```bash
uv run stridedj series
```
It makes one Garmin call per run, 1.5 s apart, and can be re-run any time (it only fetches what's missing). New runs get their series automatically while the bot runs.

**Optional, interval targets:** StrideDJ derives a target pace per rep distance from your last sessions. To start from your own targets instead, set `ATHLETE_INTERVAL_TARGET_PACES=400=3:45,800=4:10` and `ATHLETE_INTERVAL_TARGETS_SINCE=` (the day you set them, `YYYY-MM-DD`). Sessions after that day adapt them.

## 7. Last.fm API key (optional)
Spotify doesn't return artist genres to development-mode apps, so genres come from Last.fm tags. Without a key, playlists still work; the genre option ("Z2 tonight with techno") then finds nothing.
1. Create a free account at [last.fm/join](https://www.last.fm/join).
2. Open [last.fm/api/account/create](https://www.last.fm/api/account/create): any application name and a short description; **leave the callback URL and the homepage empty**.
3. Copy only the **API key** into `.env`: `LASTFM_API_KEY=...`. The shared secret is not needed (StrideDJ never logs in as you; it only reads public artist tags).

## Run it
```bash
make demo
```
Terminal chat on synthetic data (only the OpenAI key needed).

```bash
uv run stridedj chat --data live
```
Terminal chat on your real accounts.

```bash
make bot
```
The Telegram bot. Keep the terminal open; **only one copy can run per bot token**, so stop the old one first (Ctrl+C).

For an always-on bot, see [HOSTING.md](HOSTING.md) (`docker compose up -d`).

## Troubleshooting
| Symptom | Fix |
|---|---|
| `garmin: session …` in doctor | run `uv run stridedj auth garmin` |
| Garmin login: 429 | wait some minutes and retry once; don't loop. Plan B: intervals.icu |
| Google: `invalid_grant` / "run stridedj auth google" | the 7-day testing-mode expiry: run `uv run stridedj auth google` again |
| Spotify: 403 on a playlist | you don't own it: copy it into your own playlist and use the copy's ID |
| Spotify: `INVALID_CLIENT: Invalid redirect URI` | the dashboard and `.env` must both say exactly `http://127.0.0.1:8888/callback` |
| The bot doesn't answer | is `make bot` running? Is your numeric id in `TELEGRAM_ALLOWED_USER_IDS`? Did you press Start? |
| `Conflict: terminated by other getUpdates request` | two copies of the bot are running (e.g. one in another terminal). Stop the old one: Ctrl+C there, or `pkill -f "stridedj bot"` |
| `make: No rule to make target '#'` | a pasted line had a `# comment`; zsh passes it on as arguments. Run the command without the comment |
