# StrideDJ

A tool-calling running agent. Message it on Telegram, e.g. *"10k Z2 tonight"*, and it does four things:

1. Estimates how long the run will take, from **your own** Garmin history. If you haven't run that distance recently, it extrapolates from your recent runs.
2. Finds a free slot in your Google Calendar, using free/busy information only.
3. Builds a **new** Spotify playlist in the style of what you run to, at a tempo matched to your recent cadence for that kind of run.
4. Proposes the plan with **Approve / Cancel** buttons. Nothing is created until you approve.

**What the LLM never sees:** your GPS, raw heart-rate series, activity names, song titles or calendar event titles. It works only with aggregates returned by tools, and the only writes happen in code after your approval.

## Quickstart
```bash
make install              # uv: Python 3.12 venv + dependencies
cp .env.example .env      # then follow docs/SETUP.md
make doctor               # ✓/✗ per credential (never prints values)
make check                # ruff + mypy --strict + pytest
```

## Layout
| Path | What |
|---|---|
| `src/stridedj/domain/` | pydantic contracts for everything that crosses a boundary |
| `src/stridedj/ports.py` | interfaces to the outside world (Protocols); each has a real adapter and a fake |
| `src/stridedj/llm/` | the OpenAI adapter + deterministic fakes |
| `src/stridedj/harness/` | the agent harness: loop, tools, context, guardrails, approvals, executor, tracing |
| `src/stridedj/runs/` | Garmin / intervals.icu / demo sources, the store, run typing, duration and race-time models |
| `src/stridedj/music/` | Spotify, ReccoBeats, the style model, playlist assembly |
| `src/stridedj/cal/` | Google Calendar free/busy + slot finding |
| `src/stridedj/bot/` | the Telegram front-end |
| `evals/` | agent evals: tasks, trials, programmatic graders, pass@k / pass^k |
| `docs/` | setup, architecture, decisions |
