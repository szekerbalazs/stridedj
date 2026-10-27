# Decision log

Each row: what was decided, the options considered, and why. Interviewers ask "why X and not Y?" more than "what is X?".

| # | Decision | Options | Choice and why |
|---|---|---|---|
| 1 | LLM provider | OpenAI · Anthropic · self-hosted open model | **OpenAI**: prior experience with the SDK, and a mature Responses API with strict structured outputs and function calling. The provider sits behind an `LLMClient` port, so switching costs one adapter. |
| 2 | Agent framework | LangGraph / OpenAI Agents SDK / LiveKit Agents · hand-built | **Hand-built harness**: the loop is ~150 lines, and owning it means I can explain and test every decision (budgets, approvals, tracing). Frameworks hide exactly the parts interviewers ask about. |
| 3 | Agent shape | LLM-only intent parser + fixed workflow · autonomous agent · tool-calling agent with gated writes | **Tool-calling agent with read-only tools; writes only after human approval.** Flexible ("make it tomorrow morning instead") without giving the model the power to act. |
| 4 | Final answer format | free text · JSON in the prompt · strict structured outputs | **Strict structured output (`AgentReply`)**: the transport renders it with code, and evals grade it programmatically. |
| 5 | Chat transport | Telegram · Slack · WhatsApp | **Telegram**: a bot in 2 minutes, inline buttons, long polling (no public endpoint to secure). WhatsApp needs Meta business verification plus a public webhook. |
| 6 | Run data | Strava API · Garmin official API · `garminconnect` · intervals.icu · manual export | **`garminconnect` (live, unofficial)**, with intervals.icu as plan B and a synthetic source offline. Garmin's official API is partner-only. Strava's terms ban AI training on its data (2024) and reportedly its use in LLM context (2026). Unofficial APIs break (Garmin's March 2026 auth change broke `garth`), hence the `RunSource` port and the fallbacks. |
| 7 | Calendar reads | `events.list` · `freebusy.query` | **freebusy**: busy intervals without titles. Titles are private (who I meet) and untrusted (anyone can send an invite carrying instructions). |
| 8 | Secrets | `.env` · OS keychain · secret manager | **`.env` + `SecretStr`** for a single-user project; never logged, never shown to the LLM. A product would use a secret manager plus per-user encrypted token storage. |
| 9 | Project tooling | pip/venv · poetry · uv | **uv** (fast, a lockfile), **ruff**, **mypy --strict**, **pytest**, src layout. |
| 10 | HR zones | typed into `.env` · read from Garmin | **Read from Garmin Connect at startup** (RUNNING profile, else DEFAULT): one source of truth, so "Z2" always matches the watch. Garmin also returns time-in-zone per run, computed with the same zones. `.env` is only an override for non-Garmin sources. |
| 11 | Prompt changes | edit in place · versioned files + evals | **Versioned prompts** (`system_v1.md` … `system_v3.md`), compared with `run_eval.py --prompt`. v2 fixed one ambiguity found in transcripts ("6x800"); v3 (with new tool arguments) fixed named start times and interval totals found in live use. No regressions either time. |
