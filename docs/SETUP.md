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
make setup        # creates .venv with Python 3.12, installs dependencies, copies .env.example → .env
```

## 1. OpenAI API key
ChatGPT subscriptions do **not** include API usage; the API is billed separately with prepaid credit.
1. Sign in at <https://platform.openai.com>.
2. **Settings → Billing**: add a payment method and about **$5–10** of credit. Leave auto-recharge off.
3. **Settings → Limits:** set a monthly budget (e.g. $15) with an email alert.
4. **API keys → Create new secret key**, in the default project. *Restricted* permissions are enough: *Responses: Write*, *Models: Read*. Copy it now; it's shown only once.
5. In `.env`: `OPENAI_API_KEY=sk-...`

✅ `make doctor` → `openai: api key ✓`. **You can now run the demo:** `make demo`.

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
uv run stridedj auth garmin     # asks for the MFA code if you use one
uv run stridedj auth spotify    # opens the browser → approve → refresh token written to .env
uv run stridedj auth google     # opens the browser → "Google hasn't verified this app" → Advanced → Continue,
                                #   tick both permissions → refresh token written to .env
make doctor                     # everything ✓
make smoke                      # read-only live check: prints counts only
```

## Run it
```bash
make demo                        # terminal chat on synthetic data (only OpenAI needed)
uv run stridedj chat --data live # terminal chat on your real accounts
make bot                         # the Telegram bot (keep the terminal open)
```
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
