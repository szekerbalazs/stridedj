# Hosting: the agent, and the models

## Part 1: hosting the agent
**What runs:** one long-running Python process (`stridedj bot`). It polls Telegram, runs the agent loop, and calls OpenAI, Google, Spotify and Garmin. It is I/O-bound (waiting on APIs), so one small container handles a single user easily. `asyncio` lets it serve several conversations concurrently.

### Polling vs webhooks
| | Long polling (used) | Webhook |
|---|---|---|
| Direction | the bot connects out to Telegram | Telegram calls your HTTPS URL |
| Needs | outbound internet only | a public domain + TLS + a secret token check |
| Fits | a laptop, a VM, a container; one instance | serverless (Cloud Run, Lambda), many instances behind a load balancer |
| Attack surface | none inbound | a public endpoint (validate `X-Telegram-Bot-Api-Secret-Token`) |

### The container (`Dockerfile`, `compose.yaml`)
- **Multi-stage build with uv:** dependencies are cached in their own layer; the runtime image holds only the venv + code.
- **Non-root user**, `read_only` root filesystem, `cap_drop: ALL`, `no-new-privileges`, a memory limit: a compromised dependency can do little.
- **Secrets are never baked into the image.** `.env` is bind-mounted; in the cloud you'd use a secret manager (GCP Secret Manager, AWS Secrets Manager, Doppler) injected as env vars.
- **State** (SQLite, caches, transcripts, the Garmin session) lives in a volume. The container itself is disposable.
- `restart: unless-stopped` plus a healthcheck.

```bash
docker compose build
docker compose run --rm -it bot auth garmin
docker compose up -d
docker compose logs -f bot
```
- `docker compose run --rm -it bot auth garmin`: one-time interactive login (MFA) inside the container

### Where to run it
| Option | Pros | Cons |
|---|---|---|
| Laptop | zero cost, easy debugging | stops when the lid closes |
| Small VM (Hetzner, a GCP e2-small, …) + Docker | cheap (~€5/mo), always on, polling works | you patch the OS |
| Fly.io / Railway / Render | `git push` deploys, volumes, secrets | another platform to learn |
| Cloud Run / Lambda | scale to zero, managed | needs webhook mode; keep state in a DB, not a local volume |
| Kubernetes | standard at scale | overkill for one user |

For the demo: local Docker. The first upgrade: a small VM with the same compose file.

### Making it production-grade (what I'd add next)
- **State in a database** (Postgres) instead of in-memory sessions and approvals, so the bot can restart without losing pending proposals and can run multiple replicas.
- **Observability:**
  - structured logs (JSON) to a log service;
  - transcripts to an LLM-tracing tool (Langfuse self-hosted, LangSmith, Braintrust, Phoenix);
  - dashboards for latency p50/p95, cost per conversation, tool error rates, budget stops.
- **CI/CD:** GitHub Actions runs `make check` plus the regression eval with the fake model on every push, and nightly a small real-model eval for regressions and model drift.
- **Rollouts:** pin the model version and the prompt version; change one at a time behind the evals; keep the previous image for rollback.
- **Rate limiting and cost alerts** at the provider (OpenAI usage limits) and in the app.

## Part 2: hosting models
### API vs self-hosting
| | Hosted API (OpenAI, Anthropic, Gemini; or via Azure/Bedrock/Vertex) | Self-hosted open model (Llama, Qwen, Mistral, gpt-oss, …) |
|---|---|---|
| Quality | frontier | good and improving; behind on hard agentic tasks |
| Ops | none | GPUs, drivers, an inference server, autoscaling, monitoring |
| Cost | per token; cheap at low volume | per GPU-hour; cheaper only at high, steady utilisation |
| Data | leaves your infra (contracts, retention options, regional endpoints) | stays in your VPC |
| Latency | network + queueing | controllable, can be very low |
| Customisation | prompts, some fine-tuning | full fine-tuning, LoRA, quantisation |

**Self-host when:** data can't leave (regulation, client contracts), volume is high and steady, you need a fine-tuned model, or offline/edge operation. **Otherwise:** an API, often through a cloud provider for enterprise contracts and regions.

### Inference servers
- **vLLM:** high-throughput serving. *PagedAttention* stores the KV cache in pages (less fragmentation), and *continuous batching* adds and removes requests from the running batch every step. It exposes an **OpenAI-compatible API**.
- **SGLang, TGI** (Hugging Face), **TensorRT-LLM** (NVIDIA): similar role, different trade-offs.
- **Ollama / llama.cpp:** local, quantised models on laptops and CPUs; great for development, not for high throughput.
- **OpenAI-compatible endpoints** are why a `base_url` swap often suffices. Our `LLMClient` port would take a Chat-Completions adapter pointing at vLLM/Ollama.

### GPU memory: back-of-the-envelope
- **Weights:** params × bytes per param. fp16/bf16 = 2 bytes, int8 = 1, int4 ≈ 0.5.
  - 8B model in bf16 ≈ **16 GB**.
  - 70B in bf16 ≈ **140 GB** (2× 80 GB GPUs); in int4 ≈ 35–40 GB (1 GPU).
- **KV cache** per token = 2 (K and V) × layers × KV heads × head dim × bytes.
  - Example, Llama-3.1-8B (32 layers, 8 KV heads with GQA, head dim 128, bf16): 2·32·8·128·2 ≈ **131 KB/token**, so an 8k-token sequence ≈ 1 GB.
  - Batch × context drives memory: **long contexts and many concurrent users are what fill a GPU**, not just the weights.
- **Throughput levers:** batching (more tokens/s per GPU at some latency cost), quantisation (fits more and is faster, with a small quality loss), speculative decoding (a small draft model proposes tokens, the big model verifies them), prefix caching (shared system prompts).
- **Latency metrics:** time-to-first-token (prefill, ∝ prompt length) vs inter-token latency (decode, memory-bandwidth-bound).

### Hosting the agent vs hosting the model
They're independent decisions:
- **StrideDJ:** a self-hosted agent (my container) + a hosted model (the OpenAI API).
- **A bank:** might self-host both.
- **A startup:** might host neither (OpenAI's or Anthropic's managed agent offerings, LiveKit Cloud for voice agents).

The harness design (ports) keeps all of these options open.
