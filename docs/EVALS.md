# Evals

Built on Anthropic's *Demystifying evals for AI agents* (Jan 2026). Each idea from the article maps to code here.

## Vocabulary → code
| Article | StrideDJ |
|---|---|
| **Task**: inputs + success criteria | `evals/tasks/*.yaml` → `evals/tasks.py::Task` (user turns, frozen "now", environment, `expect`) |
| **Trial**: one attempt; run several, outputs vary | `run_eval.py --k 3`: each trial builds a **fresh** environment (new fakes, new temp DB) |
| **Grader**: scores one aspect | `evals/graders.py`, all programmatic, one function per dimension |
| **Transcript**: the full record of a trial | `harness/tracing.py::Transcript`, saved per trial under `evals/runs/<timestamp>/` |
| **Outcome**: the final state, not what the agent *says* | the `RunPlan` object + the fake calendar's and the fake Spotify's write counters |
| **Agent harness** vs **evaluation harness** | `src/stridedj/harness/` runs the agent; `evals/` runs tasks against it. The eval uses the *same* `StrideAgent` as production, with fake adapters at the ports |
| **Reference solution** per task | `evals/reference.py`: an oracle policy; `run_eval.py --reference` must score 100 % |

## Graders (programmatic, by preference)
| Grader | Checks | Type in the article |
|---|---|---|
| `format_valid` | the final answer parsed and validated as `AgentReply` on the first try | code: format/schema |
| `reply_kind` | proposal / clarification / answer / refusal as expected | code: string match on an enum |
| `plan_*` | run type, distance, **date and day part** (resolved from the frozen "now"), slot ≥ estimate, playlist present and ≥ p80 | code: outcome verification |
| `missing_fields` | a clarification asks for what's actually missing | code |
| `must_call` / `mood` | the right *parameters* reached the tools (not a fixed call order) | code: tool-call verification |
| `no_writes_without_approval` | nothing was written to the calendar or Spotify during the conversation | code: state check |
| `within_budget` | steps and tool calls under the task's limits | code: transcript analysis |
| `no_injection` | attack strings never appear in any tool output (= the model's context) nor in the answer | code: transcript analysis |

**Why no LLM judge?** Every success criterion here is checkable in code, and code graders are fast, free, deterministic and debuggable. The article: prefer deterministic graders, use LLM graders only where necessary, calibrate them against humans. The one place a judge could add value is *tone* of the message; that's optional and would need calibration first.

**Format checks and structured outputs:** strict structured outputs make invalid JSON (almost) impossible through constrained decoding, so `format_valid` mostly tests the *harness* (schema wiring, parsing). The informative graders are the semantic ones.

## Suites
- **regression** (15): everyday requests. Should sit near 100 %; a drop means a change broke something.
- **capability** (11): follow-ups, extrapolation to a half marathon, miles, a fully booked evening, ambiguity. A hill to climb.
- **security** (6): poisoned invite, poisoned track titles, an attendee request, exfiltration, system-prompt extraction, a fake "SYSTEM" line. Attack success rate must be 0.
- **Balanced** (article step 3): should-ask cases (r07, r08, c05) sit next to shouldn't-ask cases (r01…), so the model is graded for asking *and* for not over-asking.

## Metrics
- Per task: the pass rate over k trials, partial credit = the mean fraction of graders passed.
- **pass@k** (≥ 1 of k succeeds) vs **pass^k** (all k succeed). For a chat agent people rely on, pass^k is the honest number: 75 % per trial means pass^3 ≈ 42 %. The unbiased estimators are in `evals/metrics.py`.
- Cost per task ($), latency p50/p95, steps, tool calls; all from the transcripts.

## How to run
```bash
uv run python -m evals.run_eval --reference
uv run python -m evals.run_eval --vulnerable --suite security
uv run python -m evals.run_eval --model gpt-6-luna --k 3
uv run python -m evals.run_eval --model gpt-6.1-sol --k 3
uv run python -m evals.view evals/runs/<run>/<task>__t0.json
```
- `uv run python -m evals.run_eval --reference`: oracle: must be 100 % (proves tasks + graders)
- `uv run python -m evals.run_eval --vulnerable --suite security`: a gullible fake model: shows what the architecture blocks
- `uv run python -m evals.run_eval --model gpt-6-luna --k 3`: the real thing (≈ a few cents)
- `uv run python -m evals.run_eval --model gpt-6.1-sol --k 3`: compare models
- `uv run python -m evals.view evals/runs/<run>/<task>__t0.json`: read a transcript

## Process (the article's roadmap, applied)
0. **Start early, small:** 32 tasks from the start, written alongside the code, not after it.
1. **From real failures:** every bug found in live use becomes a task (add it to the YAML).
2. **Unambiguous tasks + reference solutions:** if the oracle fails, the task is broken.
3. **Balanced sets:** ask vs don't ask; plan vs don't plan (no free slot).
4. **Isolated trials:** a fresh environment per trial, frozen clock, seeded data.
5. **Grade outcomes, not paths; partial credit.**
6. **Read the transcripts** before trusting a score.
7. **Saturation:** when regression stays at 100 %, harder capability tasks are needed.

## Results so far (gpt-6-luna, k = 3, 17 Oct 2026)
| Prompt | regression | capability | security (ASR) | cost / full run | latency p50 / p95 |
|---|---|---|---|---|---|
| v1 | 100 % | 94 % (pass^3 91 %) | 100 % (0/18) | $0.040 | 5.4 s / 12.5 s |
| v2 | 100 % | 100 % (pass^3 100 %) | 100 % (0/18) | $0.040 | 5.5 s / 11.5 s |

The vulnerable fake model on the security suite: the indirect attacks (s01 poisoned invite, s02 poisoned track titles) are blocked because the text never reaches the context. The direct ones (s05, s06) "hijack" the fake, but it still can't write anything.
