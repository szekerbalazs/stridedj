"""Deterministic LLM fakes, so the harness, the tools and the security properties are testable without a model.

* `ScriptedLLM` replays a script of turns (or asks a policy function what to do next, given the conversation so far).
* `VulnerableLLM` is a deliberately gullible model: if any text in its context contains an injection trigger
  ("ignore previous instructions", "SYSTEM:"), it obeys it. With it, the security suite shows that the *architecture*
  protects us: the attack text never reaches the model's context, so there is nothing to obey.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from stridedj.domain import AgentReply, ReplyKind
from stridedj.llm.base import InputItem, LLMTurn, Pricing, ToolCall, ToolSpec, Usage


class ScriptedTurn(BaseModel):
    """One scripted model turn: either tool calls, or a final answer."""

    model_config = ConfigDict(frozen=True)

    tool_calls: list[tuple[str, dict[str, Any]]] = Field(default_factory=list)
    final: AgentReply | None = None
    raw_text: str | None = Field(default=None, description="send this text verbatim (to test malformed output)")


Policy = Callable[[Sequence[InputItem]], ScriptedTurn]


def _approx_tokens(payload: object) -> int:
    return max(1, len(json.dumps(payload, default=str)) // 4)


class ScriptedLLM:
    """Replays `script` turn by turn, or calls `policy(items)` for each turn."""

    def __init__(self, script: Sequence[ScriptedTurn] | Policy, *, model: str = "scripted") -> None:
        self._script = list(script) if not callable(script) else None
        self._policy = script if callable(script) else None
        self._model = model
        self._n = 0
        self.calls: list[dict[str, Any]] = []  # what the harness sent, for assertions in tests

    @property
    def model(self) -> str:
        return self._model

    @property
    def pricing(self) -> Pricing:
        return Pricing(input_per_mtok=0, cached_input_per_mtok=0, output_per_mtok=0)

    def user_item(self, text: str) -> InputItem:
        return {"role": "user", "content": text}

    def assistant_item(self, text: str) -> InputItem:
        return {"role": "assistant", "content": text}

    def harness_note_item(self, text: str) -> InputItem:
        return {"role": "developer", "content": text}

    def tool_result_item(self, call_id: str, output: str) -> InputItem:
        return {"type": "function_call_output", "call_id": call_id, "output": output}

    def _next(self, items: Sequence[InputItem]) -> ScriptedTurn:
        if self._policy is not None:
            return self._policy(items)
        assert self._script is not None  # noqa: S101
        if self._n >= len(self._script):
            raise RuntimeError("ScriptedLLM ran out of turns")
        return self._script[self._n]

    async def respond(
        self,
        *,
        instructions: str,
        items: Sequence[InputItem],
        tools: Sequence[ToolSpec],
        output_schema: type[BaseModel] | None = None,
        cache_key: str | None = None,
    ) -> LLMTurn:
        self.calls.append({"instructions": instructions, "items": list(items), "tools": [t.name for t in tools]})
        turn = self._next(items)
        self._n += 1
        out: list[InputItem] = []
        calls: list[ToolCall] = []
        for i, (name, args) in enumerate(turn.tool_calls):
            call = ToolCall(call_id=f"call_{self._n}_{i}", name=name, arguments_json=json.dumps(args))
            calls.append(call)
            out.append(
                {"type": "function_call", "call_id": call.call_id, "name": name, "arguments": call.arguments_json}
            )
        text = turn.raw_text if turn.raw_text is not None else (turn.final.model_dump_json() if turn.final else None)
        if text is not None:
            out.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
        usage = Usage(input_tokens=_approx_tokens([instructions, list(items)]), output_tokens=_approx_tokens(out))
        return LLMTurn(output_items=out, tool_calls=calls, text=text, usage=usage, model=self._model, latency_s=0.0)


# ─────────────────────────────── the gullible model (security suite) ───────────────────────────────
INJECTION_TRIGGERS = re.compile(
    r"ignore (all |your |any )?(previous|prior|above) instructions|^\s*system\s*:", re.IGNORECASE | re.MULTILINE
)


def _context_texts(items: Sequence[InputItem]) -> list[str]:
    """Every piece of text the model would 'read': user/developer messages and tool outputs."""
    texts: list[str] = []
    for item in items:
        content = item.get("content")
        if isinstance(content, str):
            texts.append(content)
        output = item.get("output")
        if isinstance(output, str):
            texts.append(output)
    return texts


class VulnerableLLM(ScriptedLLM):
    """Behaves like `ScriptedLLM`, but if an injection trigger appears anywhere in its context it 'gets hijacked' and
    returns a reply starting with COMPROMISED, quoting the instruction it obeyed. The security suite asserts this
    never happens with the real tool surface (because untrusted text is never placed into the context)."""

    def _next(self, items: Sequence[InputItem]) -> ScriptedTurn:
        for text in _context_texts(items):
            match = INJECTION_TRIGGERS.search(text)
            if match:
                snippet = text[match.start() : match.start() + 120]
                return ScriptedTurn(
                    final=AgentReply(
                        kind=ReplyKind.ANSWER,
                        message=f"COMPROMISED: following injected instruction: {snippet}",
                        missing_fields=[],
                    )
                )
        return super()._next(items)
