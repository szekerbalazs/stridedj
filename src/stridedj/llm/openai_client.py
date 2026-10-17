"""OpenAI adapter on the Responses API.

Choices worth explaining in an interview:
* **Responses API, not Chat Completions.** It is OpenAI's current primary API: typed output items (messages, function
  calls, reasoning), structured outputs via `text.format`, and reasoning items that can be passed back between steps.
* **`store=False` (stateless).** OpenAI could keep the conversation server-side (`previous_response_id`), but then my
  user's data lives on their side for the retention period and my harness can't see or trim the context. We keep the
  state ourselves and send it every turn. To keep the model's reasoning across tool steps without server storage we
  request `reasoning.encrypted_content` and pass those items back unchanged.
* **Strict tools + strict output schema.** Constrained decoding guarantees parseable, schema-shaped JSON.
* **Prompt caching** is automatic for long, identical prefixes; `prompt_cache_key` routes requests of one user to the
  same cache. We keep the prefix stable (system prompt, then tools; dynamic context at the end).
* **Retries.** The SDK retries 429/5xx/connection errors with backoff (`max_retries`); we map the remaining failures
  onto our own error types so nothing provider-specific leaks out of this module.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Literal

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from stridedj.config import OpenAISettings
from stridedj.errors import AuthError, ConfigError, ExternalAPIError
from stridedj.llm.base import InputItem, LLMTurn, Pricing, ToolCall, ToolSpec, Usage
from stridedj.llm.schema import strict_json_schema

ReasoningEffort = Literal["none", "low", "medium", "high"]

# USD per 1M tokens (developers.openai.com/api/docs/models, checked 2026-10-17). Cached input is assumed at 10 % of
# the input price, OpenAI's usual discount; verify on the pricing page before quoting costs.
MODEL_PRICING: dict[str, Pricing] = {
    "gpt-6-luna": Pricing(input_per_mtok=0.10, cached_input_per_mtok=0.01, output_per_mtok=0.50),
    "gpt-6.1-sol": Pricing(input_per_mtok=2.00, cached_input_per_mtok=0.20, output_per_mtok=10.00),
    "gpt-6-astra": Pricing(input_per_mtok=10.00, cached_input_per_mtok=1.00, output_per_mtok=50.00),
}
_FALLBACK_PRICING = MODEL_PRICING["gpt-6.1-sol"]  # unknown model: assume mid-tier prices rather than "free"


class OpenAIResponsesClient:
    """`LLMClient` implementation for OpenAI's Responses API."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        reasoning_effort: ReasoningEffort | None = "low",
        max_output_tokens: int = 4000,
        pricing: Pricing | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._max_output_tokens = max_output_tokens
        self._pricing = pricing or MODEL_PRICING.get(model, _FALLBACK_PRICING)

    @classmethod
    def from_settings(cls, settings: OpenAISettings) -> OpenAIResponsesClient:
        if settings.api_key is None:
            raise ConfigError("OPENAI_API_KEY is not set (docs/SETUP.md §1)")
        client = AsyncOpenAI(
            api_key=settings.api_key.get_secret_value(),
            timeout=settings.request_timeout_s,
            max_retries=settings.max_retries,
        )
        return cls(
            client=client,
            model=settings.model,
            reasoning_effort=settings.reasoning_effort,
            max_output_tokens=settings.max_output_tokens,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def pricing(self) -> Pricing:
        return self._pricing

    # ── item builders (the only provider-specific shapes the harness needs) ──
    def user_item(self, text: str) -> InputItem:
        return {"role": "user", "content": text}

    def assistant_item(self, text: str) -> InputItem:
        return {"role": "assistant", "content": text}

    def harness_note_item(self, text: str) -> InputItem:
        # "developer" messages carry instructions from the application, ranked above user messages.
        return {"role": "developer", "content": text}

    def tool_result_item(self, call_id: str, output: str) -> InputItem:
        return {"type": "function_call_output", "call_id": call_id, "output": output}

    # ── the call ──
    def _request(
        self,
        *,
        instructions: str,
        items: Sequence[InputItem],
        tools: Sequence[ToolSpec],
        output_schema: type[BaseModel] | None,
        cache_key: str | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._model,
            "instructions": instructions,
            "input": list(items),
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": self._max_output_tokens,
        }
        if tools:
            params["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "strict": True,
                }
                for t in tools
            ]
            params["parallel_tool_calls"] = True
        if output_schema is not None:
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": output_schema.__name__,
                    "schema": strict_json_schema(output_schema),
                    "strict": True,
                }
            }
        if self._reasoning_effort is not None:
            params["reasoning"] = {"effort": self._reasoning_effort}
        if cache_key is not None:
            params["prompt_cache_key"] = cache_key
        return params

    async def respond(
        self,
        *,
        instructions: str,
        items: Sequence[InputItem],
        tools: Sequence[ToolSpec],
        output_schema: type[BaseModel] | None = None,
        cache_key: str | None = None,
    ) -> LLMTurn:
        params = self._request(
            instructions=instructions, items=items, tools=tools, output_schema=output_schema, cache_key=cache_key
        )
        started = time.perf_counter()
        try:
            response = await self._client.responses.create(**params)
        except openai.AuthenticationError as exc:
            raise AuthError("openai", "API key rejected", status_code=exc.status_code) from None
        except openai.PermissionDeniedError as exc:
            raise AuthError("openai", "no access to this model or feature", status_code=exc.status_code) from None
        except openai.RateLimitError as exc:
            raise ExternalAPIError("openai", "rate limited or out of credit", status_code=429, retryable=True) from exc
        except openai.APIConnectionError as exc:  # includes timeouts
            raise ExternalAPIError("openai", f"connection failed ({type(exc).__name__})", retryable=True) from exc
        except openai.APIStatusError as exc:
            # 400s carry a useful, secret-free message (e.g. an invalid schema); keep its first line
            detail = str(exc.message).splitlines()[0][:300]
            raise ExternalAPIError("openai", detail, status_code=exc.status_code) from exc
        latency = time.perf_counter() - started
        return _to_turn(response, model=self._model, latency_s=latency)


def _to_turn(response: Any, *, model: str, latency_s: float) -> LLMTurn:
    """Normalise a Responses API `Response` into an `LLMTurn`."""
    output_items: list[InputItem] = [item.to_dict() for item in response.output]
    tool_calls = [
        ToolCall(call_id=item.call_id, name=item.name, arguments_json=item.arguments)
        for item in response.output
        if item.type == "function_call"
    ]
    refusal: str | None = None
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "refusal":
                    refusal = part.refusal
    text = response.output_text or None
    incomplete = (
        response.incomplete_details.reason if response.status == "incomplete" and response.incomplete_details else None
    )
    u = response.usage
    usage = (
        Usage(
            input_tokens=u.input_tokens,
            cached_input_tokens=u.input_tokens_details.cached_tokens if u.input_tokens_details else 0,
            output_tokens=u.output_tokens,
            reasoning_tokens=u.output_tokens_details.reasoning_tokens if u.output_tokens_details else 0,
        )
        if u is not None
        else Usage()
    )
    return LLMTurn(
        output_items=output_items,
        tool_calls=tool_calls,
        text=text,
        refusal=refusal,
        incomplete_reason=incomplete,
        usage=usage,
        model=model,
        latency_s=latency_s,
    )
