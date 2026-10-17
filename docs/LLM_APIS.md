# How LLM APIs work, and why OpenAI here

## What a call actually is
An LLM API is an HTTPS endpoint. You `POST` JSON, authenticated with a secret key in a header, and get JSON back (or a stream).
```
POST https://api.openai.com/v1/responses
Authorization: Bearer $OPENAI_API_KEY
{ "model": "gpt-6-luna", "instructions": "...", "input": [ ...conversation items... ], "tools": [...], "text": {"format": {...}} }
```
- **Stateless by default.** The model remembers nothing between calls; *you* send the conversation every time. That is why context management is the harness's job.
  - OpenAI can keep state for you (`store=true` + `previous_response_id`).
  - We use `store=false`: our data isn't kept server-side for that purpose, we can trim and inspect the context, and we stay provider-portable.
- **Tokens** are the unit of everything: context-window limits, latency (output tokens are generated one at a time) and price (input and output priced separately per million tokens; output costs more).
  - Reasoning models also produce hidden *reasoning tokens*, billed as output.
- **The SDK** (`openai` Python package) is a typed wrapper around those HTTP calls. It adds retries with backoff on 429/5xx/connection errors, timeouts, pydantic response models and helpers such as `responses.parse`. Compare `doctor.py` (a raw `httpx` call to `/v1/models`) with `llm/openai_client.py` (the SDK).

## The request knobs that matter
| Knob | What it does | Here |
|---|---|---|
| `model` | capability/price/latency trade-off | `gpt-6-luna` default; `gpt-6.1-sol` compared in evals |
| `instructions` / system prompt | the highest-priority instructions | `agent/prompts/system_v1.md` + dynamic context |
| `input` | the conversation: user/assistant/developer messages, function calls and their outputs, reasoning items | built by the harness |
| `tools` | JSON-schema'd functions the model may call (`strict: true` → constrained decoding) | 6 read-only tools |
| `text.format` | structured outputs: the final answer must match a JSON schema | `AgentReply` |
| `reasoning.effort` | how much the model thinks before answering (none/low/medium/high) | `low`: the task is simple; more effort = more latency + cost |
| `max_output_tokens` | a hard cap on the answer; hitting it gives `status: incomplete` | 4000; handled as a failure |
| `temperature`/`top_p` | sampling randomness (classic models; reasoning models largely manage it themselves) | default |
| `prompt_cache_key` | routes similar requests to the same cache | per user |
| `store`, `include` | server-side state; return the encrypted reasoning so it can be passed back statelessly | `false`, `["reasoning.encrypted_content"]` |

## Function calling (the "tool use" loop)
1. You send tools: name, description, JSON schema of the arguments.
2. The model replies with `function_call` items (name + JSON arguments + `call_id`) instead of, or alongside, text. It **does not run anything**: it asks *you* to.
3. Your code runs the function and appends a `function_call_output` item with the same `call_id`.
4. You call the model again with the extended conversation. Repeat until it answers without tool calls.

Reasoning models also return `reasoning` items that must be passed back with the tool outputs, so the model keeps its chain of thought between steps.

## Structured outputs and strict mode
With `strict: true` the API compiles the JSON schema into a grammar and **masks every token that would break it** while decoding. The output always parses and always has the required fields. Requirements:
- `additionalProperties: false` on every object;
- all properties `required` (optional = `["string", "null"]`);
- only a subset of JSON Schema.

Validation of *values* (ranges, lengths) still happens in pydantic after parsing. Refusals come back as a separate `refusal` content part, not as schema-shaped JSON.

## Operating an LLM API in production
- **Rate limits** are per organisation and model, in requests/min and tokens/min. You get `429` plus `x-ratelimit-*` headers. Retry with exponential backoff and jitter, honour `Retry-After`, and cap total retries.
- **Errors:**
  - `400`: your request is wrong (e.g. an invalid schema); don't retry.
  - `401`/`403`: key or permission; don't retry.
  - `404`: the model is unknown to your org.
  - `429`/`5xx`: retry.
  - `incomplete`: you hit `max_output_tokens`.
  - `refusal`: the model declined.
- **Cost control:**
  - pick the smallest model that passes your evals;
  - **prompt caching** (OpenAI: automatic for identical prefixes ≥ ~1k tokens, a large discount on cached input; keep static content first);
  - the **Batch API** (about half price, results within 24 h) for offline jobs like large eval runs;
  - budgets in the harness.
- **Latency:**
  - fewer round trips (parallel tool calls, task-shaped tools);
  - smaller models;
  - lower reasoning effort;
  - **streaming** (server-sent events) so users see tokens early;
  - caching.
- **Data handling:**
  - API data is not used for training by default;
  - abuse-monitoring retention applies, and Zero Data Retention is available for eligible organisations;
  - `store=false` avoids server-side conversation storage;
  - the real minimisation is sending less: we send aggregates, not raw health data.
- **Secrets:** the key lives in `.env`/a secret manager, never in code, logs or the frontend. Use project-scoped, least-privilege keys and spending limits.

## Chat Completions vs the Responses API
Chat Completions (`/v1/chat/completions`, a list of role/content messages) is the older, widely copied interface. Many "OpenAI-compatible" servers (vLLM, Ollama, OpenRouter) implement it.

The Responses API is OpenAI's current primary API:
- typed output items (message, function_call, reasoning, web_search_call, …);
- built-in tools;
- optional server-side state;
- first-class reasoning items.

I use Responses for the OpenAI adapter. A local/open model would get a second adapter on the Chat Completions shape behind the same `LLMClient` port.

## OpenAI vs Anthropic (Claude): same concepts, different shapes
| | OpenAI Responses | Anthropic Messages |
|---|---|---|
| Endpoint | `POST /v1/responses` | `POST /v1/messages` (+ `anthropic-version` header) |
| System prompt | `instructions` (or developer messages) | top-level `system` |
| Conversation | heterogeneous *items* | `messages` with typed *content blocks* (`text`, `tool_use`, `tool_result`, `thinking`) |
| Tool call | `function_call` item → `function_call_output` item | `tool_use` block in the assistant turn → `tool_result` block in the next *user* turn |
| Output length | `max_output_tokens` (optional) | `max_tokens` (required) |
| Reasoning | `reasoning.effort`; encrypted reasoning items | (adaptive) thinking; thinking blocks passed back with tool results |
| Prompt caching | automatic prefix caching | explicit `cache_control` breakpoints (you choose what's cached); cache reads are heavily discounted |
| Server-side state | optional (`store`, `previous_response_id`) | stateless |
| Structured output | `text.format` json_schema, strict tools | JSON-schema outputs / strict tool use on current models (check the docs for the exact parameters) |

Moving StrideDJ to Claude = one new adapter implementing `LLMClient` (≈ 150 lines). The harness, tools, evals and prompts stay; the evals would tell me whether the prompt needs changes for that model.

## Why OpenAI for this project
1. **Experience:** I've used the OpenAI SDK before. In a one-week build, familiarity lowers the risk.
2. **Fit:** strict function calling + strict structured outputs are exactly what the harness relies on (schema-valid tool args and final answers).
3. **Cost for evals:** `gpt-6-luna` ($0.10 / $0.50 per M tokens) makes running the whole eval suite with k=3 trials cost cents, so I can measure instead of guess, and then check whether `gpt-6.1-sol` is worth 20× the price.
4. **Not locked in:** the provider sits behind a port. The decision is reversible by design, and the evals tell me if a switch is safe.

Billing note: ChatGPT Plus and Claude Pro subscriptions don't include API usage; both APIs bill separately (prepaid credits).
