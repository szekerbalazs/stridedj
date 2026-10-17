"""Provider-neutral LLM types.

The harness talks to every model through `LLMClient`. Conversation items are provider-native dicts (an OpenAI
Responses item, an Anthropic content block, ...): the harness never inspects them, it only appends what the client
returns and asks the client to build the two item kinds it needs (a user message, a tool result). That keeps the loop
provider-neutral without a lossy translation layer, and it keeps provider details (reasoning items, encrypted
reasoning content) intact between steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

InputItem = dict[str, Any]


class ToolSpec(BaseModel):
    """A function the model may call. `parameters` is a strict JSON schema (see `harness.schema`)."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    name: str
    arguments_json: str


class Usage(BaseModel):
    """Token counts of one or more model calls. `cached_input_tokens` is a subset of `input_tokens`."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0  # a subset of output_tokens (billed as output)

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


class Pricing(BaseModel):
    """USD per million tokens. Cached input is billed at a discount; reasoning tokens are billed as output."""

    model_config = ConfigDict(frozen=True)

    input_per_mtok: float = Field(ge=0)
    cached_input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)

    def cost_usd(self, usage: Usage) -> float:
        uncached = usage.input_tokens - usage.cached_input_tokens
        return (
            uncached * self.input_per_mtok
            + usage.cached_input_tokens * self.cached_input_per_mtok
            + usage.output_tokens * self.output_per_mtok
        ) / 1e6


class LLMTurn(BaseModel):
    """One model response, normalised.

    `output_items` must be appended to the conversation UNCHANGED before the tool results (reasoning models require
    their reasoning items to be passed back with the tool outputs).
    """

    model_config = ConfigDict(frozen=True)

    output_items: list[InputItem]
    tool_calls: list[ToolCall]
    text: str | None = Field(description="concatenated assistant text (the JSON answer when an output schema is set)")
    refusal: str | None = None
    incomplete_reason: str | None = Field(default=None, description="e.g. 'max_output_tokens'")
    usage: Usage
    model: str
    latency_s: float


class LLMClient(Protocol):
    """The one interface the harness uses to talk to a model."""

    @property
    def model(self) -> str: ...

    @property
    def pricing(self) -> Pricing: ...

    async def respond(
        self,
        *,
        instructions: str,
        items: Sequence[InputItem],
        tools: Sequence[ToolSpec],
        output_schema: type[BaseModel] | None = None,
        cache_key: str | None = None,
    ) -> LLMTurn:
        """Send the conversation; return the model's next turn (tool calls and/or a final message)."""
        ...

    def user_item(self, text: str) -> InputItem:
        """A user message in the provider's format."""
        ...

    def assistant_item(self, text: str) -> InputItem:
        """A plain assistant message (used to carry earlier final answers in the chat history)."""
        ...

    def harness_note_item(self, text: str) -> InputItem:
        """A message from the application itself (e.g. "your answer referenced an unknown proposal id"),
        ranked above user messages."""
        ...

    def tool_result_item(self, call_id: str, output: str) -> InputItem:
        """The result of a tool call, linked to the call by id."""
        ...
