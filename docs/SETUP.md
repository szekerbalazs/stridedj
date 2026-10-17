# Setup: accounts and credentials

Six accounts, about an hour in total. Work through them in order; after each one run `make doctor` and that line turns ✓.

**How secrets are handled:**
- **One file:** every value goes into `.env` in the project root (create it with `cp .env.example .env`). It's git-ignored and docker-ignored.
- **Never paste a key or token into a chat:** not Telegram, not an AI assistant, not a screenshot. If one leaks, revoke it and create a new one. That takes a minute.
- **`make doctor` never prints values:** it only shows ✓/✗.

---

## 1. OpenAI API key (≈ 10 min)
A ChatGPT Plus subscription does **not** include API usage. The API is billed separately with prepaid credits.

1. Go to <https://platform.openai.com> and sign in, with the same account as ChatGPT if you like.
2. **Add credit:**
   1. Open **Settings** (gear icon, top right) → **Billing** → **Add payment details**.
   2. Then **Add to credit balance** → **$10**. That is far more than this project needs; a full eval run costs cents.
   3. Leave **auto-recharge off**.
3. **Set a spending limit:** Settings → **Limits** (or *Usage limits / Budgets*) → set a **monthly budget**, e.g. $15, and an email alert at 50 %.
4. **Create the key:**
   1. Go to **API keys** (left sidebar, or <https://platform.openai.com/api-keys>) → **Create new secret key**.
   2. Name: `stridedj-laptop`. Project: *Default project*.
   3. **Permissions:** *All* is simplest. *Restricted* is the least-privilege option: allow **Model capabilities / Responses: Write** and **Models: Read**.
   4. **Copy the key now.** It's shown only once.
5. **Store it:** in `.env`, set `OPENAI_API_KEY=sk-...` and leave `OPENAI_MODEL` as it is (we pick the model together in the next step).
6. **Check:** run `make doctor`. The line `openai: api key` shows ✓.

If the dashboard asks you to **verify your organization**, do it (ID check). Some newer models and streaming require it.

## 2. Telegram bot + your user ID (≈ 5 min)
Install Telegram on your phone (and optionally on the desktop).

1. **Create the bot with @BotFather:**
   1. In Telegram, search for **@BotFather** (blue check mark) → **Start**.
   2. Send `/newbot`.
   3. Name: `StrideDJ`. Username: it must end in `bot` and be globally unique, e.g. `your_stridedj_bot`.
   4. BotFather replies with a **token** like `1234567890:AAH...`. Put it into `.env` as `TELEGRAM_BOT_TOKEN=...`.
2. **Your numeric user ID:** search for **@userinfobot** → **Start**. It replies with `Id: 123456789`. Put it into `.env` as `TELEGRAM_ALLOWED_USER_IDS=123456789`.
   - Your bot is publicly findable by its username. This allowlist is what stops strangers from using it (and your OpenAI credit).
3. **Open your bot** (search its username) and press **Start**. Bots can't message you first.
4. **Check:** run `make doctor`. You should see `telegram: bot token ✓ bot @your_stridedj_bot` and `telegram: allowed users ✓`.

## 3. Spotify developer app (≈ 10 min)
Requirements:
- **Spotify Premium:** since Feb 2026, development-mode apps need a Premium owner.
- Your **running playlists must be owned by you**: apps can only read playlists you own or collaborate on.

1. **Create the app:**
   1. Go to <https://developer.spotify.com/dashboard> → log in → accept the developer terms → **Create app**.
   2. App name: `StrideDJ`. Description: `Personal running playlist planner`.
   3. **Redirect URI:** `http://127.0.0.1:8888/callback` → click **Add**. Exactly this: `localhost` is rejected, and the loopback IP is allowed over http.
   4. **Which API/SDKs:** tick **Web API** → accept → **Save**.
2. **Client ID:** in the app, open **Settings** → copy the **Client ID** → `.env`: `SPOTIFY_CLIENT_ID=...`. You do **not** need the client secret: we use Authorization Code with **PKCE**, the flow for apps that can't keep a secret.
3. **User Management** (in the app settings) → add your own name and the **email of your Spotify account**. Dev-mode apps only work for listed users (max 5).
4. **Running playlist IDs:**
   1. In the Spotify app, open your running playlist → **⋯** → **Share** → **Copy link to playlist**. The link looks like `https://open.spotify.com/playlist/37i9dQZF1DX...?si=...`; the ID is the part between `/playlist/` and `?`.
   2. Check it says *"by <your name>"*. If it's a Spotify-made playlist: **⋯ → Add to other playlist → New playlist**, then use the copy's ID.
   3. `.env`: `SPOTIFY_RUNNING_PLAYLIST_IDS=id1,id2`.
5. **Check:** run `make doctor`. You should see `spotify: client id ✓` and `spotify: running playlists ✓`. The login itself (`stridedj auth spotify`) comes later; it opens your browser once.

## 4. Google Calendar OAuth client (≈ 20 min, the fiddliest one)
1. **Project:** go to <https://console.cloud.google.com> → project picker (top left) → **New project** → name `stridedj` → **Create** → select it.
2. **Enable the API:** ☰ → **APIs & Services → Library** → search **Google Calendar API** → **Enable**.
3. **Consent screen:** ☰ → **Google Auth Platform** (older UI: *APIs & Services → OAuth consent screen*) → **Get started**.
   1. App name: `StrideDJ`. User support email: yours.
   2. Audience: **External**.
   3. Contact email: yours → agree → **Create**.
4. **Test user:** Google Auth Platform → **Audience** → **Test users** → **Add users** → your Google email → Save.
   - While the app is in *Testing* mode, logins expire after **7 days**. Plan a re-login on Sun 25 Oct, the day before the presentation.
5. **OAuth client:** Google Auth Platform → **Clients** → **Create client**.
   1. Application type: **Desktop app**. Name `stridedj-cli` → **Create**.
   2. Copy the **Client ID** and **Client secret** → `.env`: `GOOGLE_CLIENT_ID=...`, `GOOGLE_CLIENT_SECRET=...`.
   3. Don't download the JSON file. If you did, delete it, or at least keep it out of the repo (it's git-ignored).
6. **A "Runs" calendar (recommended):** this makes demo events easy to find and clean up.
   1. At <https://calendar.google.com> → **Other calendars +** → **Create new calendar** → `Runs` → Create.
   2. Open its **Settings → Integrate calendar** → copy the **Calendar ID** (`...@group.calendar.google.com`).
   3. `.env`: `GOOGLE_CALENDAR_ID=<that id>` and `GOOGLE_BUSY_CALENDAR_IDS=primary,<that id>` (free/busy is checked on both).
7. **Check:** run `make doctor`. You should see `google: oauth client ✓`. The browser login (`stridedj auth google`) comes later.

**Scopes we will request:**
- `calendar.freebusy`: busy/free times only, never event titles;
- `calendar.events`: to insert the run event.

## 5. Garmin login + your heart-rate zones (≈ 5 min)
1. **Login:** in `.env`, set `GARMIN_EMAIL=...` and `GARMIN_PASSWORD=...`.
   - `garminconnect` is **unofficial**: Garmin's official API is for business partners only. It logs in like the Garmin Connect phone app.
   - If you use two-factor authentication, `stridedj auth garmin` asks for the code once.
   - The session then lives in `~/.garminconnect/garmin_tokens.json` (mode 0600). Treat that file like a password.
2. **Your HR zones** ("Z2" must mean what it means on your watch):
   1. Find them in the Garmin Connect app (☰ / *More* → **Garmin Devices** → your watch → **User Settings → Heart Rate Zones**) or on connect.garmin.com (profile → *Settings* → *User Settings* / *Heart Rate Zones*).
   2. Note the **upper limit of zones 1, 2, 3 and 4** in bpm, e.g. Z1 ≤ 124, Z2 125–144, Z3 145–159, Z4 160–174, Z5 ≥ 175.
   3. Put the **lower** bound of the next zone into `.env`: `ATHLETE_HR_ZONE_UPPER_BOUNDS=125,145,160,175`. The values are exclusive upper bounds of Z1..Z4.
   4. Also set `ATHLETE_MAX_HR=...`, shown on the same screen.
3. **Check:** run `make doctor`. You should see `garmin: login ✓` and `athlete: HR zones ✓`.

**Plan B, only if the Garmin login fails:** intervals.icu.
1. Create a free account at <https://intervals.icu> and connect Garmin (Settings → Connections). Garmin syncs to it officially.
2. In intervals.icu **Settings → Developer Settings**, create an **API key**.
3. `.env`: `INTERVALS_ICU_API_KEY=...`, `INTERVALS_ICU_ATHLETE_ID=i12345` (your ID is shown in the same place), and `STRIDEDJ_RUN_SOURCE=intervals_icu`.

## 6. GitHub repository (≈ 2 min)
1. Go to <https://github.com/new>. Repository name: `stridedj`. **Private**. **Don't** add a README, .gitignore or license: the repo already has them.
2. **Create repository** → copy the HTTPS URL (`https://github.com/<you>/stridedj.git`) and tell me the URL (it isn't a secret).
3. Make sure the email of your commits (`git config user.email`) is added to your GitHub account (Settings → Emails). Otherwise the commits won't show as yours.
4. Pushing over HTTPS asks for a **personal access token** instead of a password: GitHub → Settings → Developer settings → *Fine-grained tokens* → access to this repo only, *Contents: Read and write*. macOS Keychain remembers it after the first push.
