<div align="center">

# StrideDJ

A Telegram running agent. It plans a run from your Garmin history, finds a free slot in your calendar and builds a playlist matched to your cadence. Nothing is booked until you approve.

[![CI](https://github.com/szekerbalazs/stridedj/actions/workflows/ci.yml/badge.svg)](https://github.com/szekerbalazs/stridedj/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB)
![mypy strict](https://img.shields.io/badge/mypy-strict-2a6db2)
![ruff](https://img.shields.io/badge/lint-ruff-d7ff64)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

</div>

![Architecture](docs/assets/architecture.svg)

## Example
```
you        10k Z2 tonight
stridedj   Booked Sat 17 Oct, 18:30, about 57 min. A 16-track playlist at 168 steps/min.

           10 km Z2 easy run
           Sat 17 Oct, 18:30–19:45
           ~57 min typical, ≤ 61 min in 4 of 5 runs
           cadence target 168 spm
           16 tracks · 59 min · ±1.2 BPM
                                                        [ Approve ]  [ Cancel ]
```
```
you        intervals tomorrow evening
stridedj   Which session would you like?
           [ 8 × 400 m @ 3:45 · 6.7 km ]  [ 6 × 800 m @ 4:10 · 8.3 km ]  [ Pyramid 400-1200 m · 7.1 km ]
```
Approving creates the calendar event and a private Spotify playlist.

## Features
- Duration estimates from your own runs, with a personal log-log model when you haven't run a distance recently
- Calendar slots sized to the 80th percentile of your running time, from free/busy only
- Interval sessions detected in your history from per-second pace, watch stops and heart rate
- Session options at target paces that adapt to your results
- Playlists from your own library, tempo-matched to your cadence, with genre, novelty and feedback options
- Read-only tools for the model; writes only after an explicit approval
- 45 programmatic eval tasks, including prompt-injection attacks

## Quickstart
Demo mode runs the same agent on synthetic data. You only need an OpenAI API key.
```bash
git clone https://github.com/szekerbalazs/stridedj.git && cd stridedj
make setup                # Python 3.12 via uv, dependencies, .env from the template
# add OPENAI_API_KEY to .env
make demo                 # terminal chat, try "10k Z2 tonight"
```
Requires macOS or Linux and [uv](https://docs.astral.sh/uv/).

## Setup
Connecting Telegram, Garmin, Google Calendar and Spotify takes about 45 minutes: see [docs/SETUP.md](docs/SETUP.md).
```bash
uv run stridedj auth garmin && uv run stridedj auth spotify && uv run stridedj auth google
uv run stridedj series    # per-second history for interval detection
make doctor               # checks every credential, prints no values
make bot
```

## Usage
| Command | |
|---|---|
| `make bot` | Telegram bot |
| `make demo` | terminal chat on synthetic data |
| `uv run stridedj chat --data live` | terminal chat on your accounts |
| `uv run stridedj series` | download per-second series for new runs |
| `uv run stridedj costs` | cost per API key (terminal and HTML) |
| `make doctor`, `make smoke` | credential check, read-only integration check |

With Docker: `docker compose up -d` (see [docs/HOSTING.md](docs/HOSTING.md)).

## Development
```bash
make check                # ruff, mypy --strict, 500+ offline tests
make eval-reference       # evals with the reference oracle, must be 100 %
make eval                 # evals with the real model, k = 3
```

## Documentation
- [Setup](docs/SETUP.md): accounts, credentials, configuration
- [Architecture](docs/ARCHITECTURE.md): components, request path, trust boundaries
- [Harness](docs/HARNESS.md): the agent loop and why it's hand-built
- [Evals](docs/EVALS.md): tasks, graders, results
- [Prompting](docs/PROMPTING.md): prompt structure and changelog
- [Prompt injection](docs/INJECTION.md): threat model and controls
- [Decisions](docs/DECISIONS.md): options considered and why
- [Hosting](docs/HOSTING.md), [LLM APIs](docs/LLM_APIS.md), [LiveKit](docs/LIVEKIT.md)

## License
[MIT](LICENSE)
