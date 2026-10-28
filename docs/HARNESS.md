# The agent harness

> "Evaluating an agent means evaluating the harness and the model together." (Anthropic, *Demystifying evals for AI agents*)

The **model** decides what to do next. The **harness** is everything around it: what the model sees, what it may do, when it must stop, what happens to its output, and what gets recorded. StrideDJ's harness is hand-built (≈ 600 lines in `src/stridedj/harness/`) so every one of these decisions is visible, testable and explainable.

## The loop (`harness/loop.py`)
```
            ┌───────────────────────────── budgets: steps · tool calls · $ · wall time ─────────────────────────────┐
user msg ─► │ context ─► model ─┬─► tool calls ─► ToolRegistry.execute (parallel, validated, timed) ─► results ─┐ │
            │                   │                                                                              │ │
            │                   └─► final JSON (AgentReply) ─► pydantic validate ─► reply_check vs. state ─► done│ │
            │                         ▲                         │ invalid             │ inconsistent             │ │
            │                         └──── one corrective developer note ◄──────────┴──────────────────────────┘ │
            └────────────────────────────────────────────────────────────────────────────────────────────────────────┘
done ─► the harness attaches the plan of this turn ─► code fills the message's {placeholders} and checks every other number
     ─► a wrong number or an unknown placeholder: a standard message is shown instead (recorded in the transcript)
```

| Concern | Where | Decision |
|---|---|---|
| **Model access** | `llm/base.py`, `llm/openai_client.py` | one `LLMClient` Protocol. Conversation items stay provider-native, appended unchanged (reasoning items included). `store=False` + `reasoning.encrypted_content`: stateless, nothing kept at OpenAI |
| **Context** | `harness/context.py` | static versioned prompt first (cacheable prefix), dynamic "Context" (date/time) last. History = previous user messages + final answers only, never old tool calls (stale, expensive) |
| **Tools** | `harness/tools.py`, `agent/tools.py` | pydantic args → strict JSON schema → constrained decoding. Args are re-validated (semantics), errors are actionable messages, results capped at 6k chars, a 30 s timeout per call |
| **Termination** | `loop.py` | only via a schema-valid `AgentReply` or a harness fallback. Budgets: 8 steps, 16 tool calls, $0.20, 90 s |
| **Verification** | `agent/service.py` `reply_check` | the model's claims are checked against state: a `proposal` needs a plan that `propose_plan` created *in this run*. One correction, then fallback |
| **Facts** | `agent/placeholders.py` | the model never copies a value code already knows: the harness attaches the plan (no id in the answer), and a proposal's message names its facts as placeholders (`{start}`, `{expected}`, `{session}`, `{music}` …) that code fills in. Any other number must be a fact of the plan or something the runner wrote; otherwise a standard message is shown and the transcript records `message_replaced` (an eval grader counts it). No retry: a stray digit must not cost the runner the plan |
| **Permissions** | `approvals.py`, `executor.py` | tools are read-only. Writes are not tools: plan → approval token (bound to plan hash + user, single use, 15 min TTL) → executor (fixed templates, idempotent) |
| **Input guard** | `guard.py` | allowlist → secret detector → length → rate limit, in code, before the model |
| **Observability** | `tracing.py` | one JSON transcript per run: every model call (latency, tokens), tool call (args, output), final reply, stop reason, $ |
| **Background work** | `bot/telegram_bot.py` `sync_forever` | jobs outside the loop (plays sync, genre warm-up, run refresh) every 30 min; a failed round is logged and retried next round, never fatal |
| **External rate limits** | `http.py`, `music/spotify.py` | retries with backoff, honouring `Retry-After`; a wait longer than we would ever sleep stops at once (`RateLimitedError`), and the Spotify client remembers "blocked until" so it fails fast instead of extending the block |

## Why hand-built, not a framework?
- **Frameworks considered:** LangGraph, the OpenAI Agents SDK, LiveKit Agents (voice). They are useful for big graphs, handoffs and voice pipelines.
- **What this agent needs:** one loop, eight tools and an approval gate.
- **What the hand-built version gives:**
  1. every safety-relevant decision is a line of my code that I can point to;
  2. the fakes (`ScriptedLLM`) plug in at the `LLMClient` port, so the whole agent is testable offline;
  3. no framework churn.
- **When I'd switch:** multi-agent handoffs, durable long-running workflows (resume after a crash), or voice.

## Design rules worth saying out loud
1. **Workflow where possible, agency where useful.** Plan → approve → execute is fixed. The model only decides *how* to fill the plan (which tools, which arguments, when to ask).
2. **References, not values.** The model passes `slot_id`/`draft_id`/`session_id`; the numbers come from our state. It can't "adjust" the duration on its way to the calendar, and in its own message it names facts as placeholders instead of typing them.
3. **The model never writes to the world.** The only exfiltration channels (calendar text, playlist name) are fixed templates.
4. **Untrusted text never enters the context** (song titles, calendar titles, activity names). That makes it architecture, not "please ignore injections".
5. **Every failure has a defined shape**: tool errors → model-fixable messages; model failures → fallback replies; budget exhaustion → fallback. The user always gets *an* answer.
6. **Everything is recorded** in a transcript, which is what the evals grade and what I read when something goes wrong.
