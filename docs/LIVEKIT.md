# LiveKit (what it is, and how it would fit StrideDJ)

## In one sentence
LiveKit is an open-source (Apache 2.0) **real-time audio/video/data platform**, and **LiveKit Agents** is its framework for **voice AI agents** that join those real-time sessions as participants.

## The server: `livekit/livekit`
- A **WebRTC SFU** (Selective Forwarding Unit) written in Go, on the Pion WebRTC stack, shipped as a single binary.
- **SFU vs MCU:** an SFU doesn't mix or transcode media. It *forwards* each participant's streams to the others, choosing per subscriber which quality layer (simulcast) to send. That keeps server CPU low and latency minimal.
- **Concepts:**
  - **Room:** a session.
  - **Participant:** a human client, a phone call via SIP, or an **agent**.
  - **Track:** an audio, video or data stream a participant *publishes*; others *subscribe*.
  - **Access token:** a JWT signed with your API secret, encoding identity and room permissions; issued by *your* backend.
- **Connectivity:** WebRTC over UDP, with TCP and TURN fallbacks for restrictive networks.
- **Ecosystem:**
  - client SDKs (JS, Swift, Android, Flutter, React Native, Unity, Rust, Python, ESP32, …);
  - server SDKs;
  - Egress (recording/streaming), Ingress (RTMP/WHIP in), SIP (telephony);
  - the CLI.
- **Hosting:** LiveKit Cloud (managed, multi-region, agent hosting and inference), or self-host (binary, Docker, Helm); `livekit-server --dev` for local development.

## The agents framework: `livekit/agents`
- **Agent:** instructions + tools (async functions decorated with `@function_tool`; the docstring is the tool description the LLM sees).
- **AgentSession:** the runtime for one conversation: VAD (voice activity detection), STT, LLM, TTS, turn detection, state (`userdata`).
- **Entrypoint and server:** an `AgentServer` registers with LiveKit, which *dispatches* a job (an entrypoint call with a `JobContext`) whenever a room needs an agent. Workers scale horizontally.
- **Two pipelines:**
  1. **Cascaded:** STT → LLM → TTS. You mix providers (Deepgram STT, any LLM, Cartesia/ElevenLabs/OpenAI TTS); each stage is swappable and the text is inspectable.
  2. **Realtime / speech-to-speech:** e.g. OpenAI Realtime or Gemini Live. Lower latency and more natural prosody, but less control over the text in between.
- **Turn detection:** a small transformer model decides whether the user has *finished* speaking (not just paused), which reduces interruptions. VAD alone cuts people off mid-thought.
- **Handoffs:** a tool can return another `Agent` (multi-agent flows: intake → specialist).
- **Testing:** pytest integration (run a session with text input, assert on function-call events, optionally use an LLM judge for intent).
- **Modes:** `console` (local mic/speaker, no server), `dev` (hot reload, connects to LiveKit), `start` (production).
- **Telephony:** inbound/outbound phone calls through LiveKit SIP.

## The voice-agent latency budget (why this needs special infrastructure)
A natural conversation tolerates about **≤ 1 s** from the end of the user's speech to the start of the reply. The parts:
- end-of-turn detection (~200–500 ms);
- STT finalisation;
- LLM time-to-first-token;
- TTS time-to-first-audio;
- network.

Hence streaming at every stage, regional media servers, and WebRTC (UDP, jitter buffers) instead of HTTP request/response.

## How it would fit StrideDJ (not built, a deliberate scope decision)
- **The idea:** "Hey StrideDJ, plan me 10k Z2 tonight" by voice, hands-free on the way out the door.
- **Reuse:** the same `StrideAgent` tools. Each of our `Tool`s becomes a `@function_tool` wrapper calling the same handlers. The harness ideas transfer: read-only tools, budgets, transcripts.
- **What changes:**
  1. **Approvals stay on the phone button.** Voice "yes" is easy to trigger accidentally or by someone else in the room. Writes keep a deliberate, authenticated confirmation.
  2. **Shorter, spoken replies:** a separate prompt variant, no lists, numbers rounded.
  3. **Latency:** fewer tool round trips (or a realtime model), streaming TTS.
  4. **Evals:** the same tasks, but with a text-in session first (LiveKit's test framework), then audio robustness (accents, noise).
- **Why I didn't build it:** the product is asynchronous by nature (plan a run in the evening). Text plus buttons is the right UX, and the week was better spent on evals and security. Voice is the natural next interface, and the architecture (ports, tool surface, transport-independent harness) makes it an additive change.
