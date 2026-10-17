"""Context building: what the model sees at the start of every run.

Layout, chosen for prompt caching and clarity:
1. the static system prompt (versioned file; identical across requests → cacheable prefix);
2. a short dynamic "Context" block at the END of the instructions (current date/time, timezone);
3. a compact chat history: only previous user messages and final answers, not old tool calls. Old tool results are
   stale (slots fill up, new runs arrive) and expensive; the state that matters lives in the session and is reachable
   through tools. This is context management by design rather than by truncation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from stridedj.llm.base import InputItem, LLMClient


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str
    text: str

    @classmethod
    def load(cls, path: Path) -> Prompt:
        """Load `system_vN.md`; the version is the file stem (logged in every transcript, compared in evals)."""
        return cls(version=path.stem, text=path.read_text(encoding="utf-8").strip())


def compose_instructions(prompt: Prompt, context: Mapping[str, str]) -> str:
    lines = "\n".join(f"- {key}: {value}" for key, value in context.items())
    return f"{prompt.text}\n\n# Context\n{lines}"


def history_items(llm: LLMClient, exchanges: Sequence[tuple[str, str]]) -> list[InputItem]:
    items: list[InputItem] = []
    for user_text, reply_text in exchanges:
        items.append(llm.user_item(user_text))
        items.append(llm.assistant_item(reply_text))
    return items
