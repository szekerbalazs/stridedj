"""Tool registry: typed tools in, model-friendly results out.

A tool = name + description written *for the model* + a pydantic argument model + an async handler returning a
pydantic model. The registry
* turns the argument model into a strict JSON schema (so the model's arguments always have the right shape);
* validates the arguments again with pydantic (semantic checks the schema cannot express, e.g. 0 < km ≤ 100);
* converts every failure into a short message the model can act on ("distance_km must be ≤ 100; ask the user"),
  never a stack trace (internals and data must not leak into the context);
* enforces a timeout per call and caps the size of results (context is a budget).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from stridedj.errors import ExternalAPIError, StrideDJError, ToolError
from stridedj.llm.base import ToolCall, ToolSpec
from stridedj.llm.schema import strict_json_schema

log = logging.getLogger(__name__)

MAX_RESULT_CHARS = 6000


@dataclass(frozen=True)
class Tool[ArgsT: BaseModel]:
    name: str
    description: str
    args_model: type[ArgsT]
    handler: Callable[[ArgsT], Awaitable[BaseModel]]
    timeout_s: float = 30.0

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=strict_json_schema(self.args_model))


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    name: str
    arguments: dict[str, Any]
    output: str
    is_error: bool
    latency_s: float


def _error(message: str) -> str:
    return json.dumps({"error": message})


def _validation_message(exc: ValidationError) -> str:
    """Summarise pydantic errors as 'field: problem' without echoing the (possibly huge) input values."""
    parts = []
    for err in exc.errors(include_input=False, include_url=False):
        loc = ".".join(str(p) for p in err["loc"]) or "arguments"
        parts.append(f"{loc}: {err['msg']}")
    return "invalid arguments: " + "; ".join(parts)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool[Any]]) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            self._tools[tool.name] = tool

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> Sequence[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        try:
            raw_args: dict[str, Any] = json.loads(call.arguments_json or "{}")
        except json.JSONDecodeError:
            raw_args = {}
        output, is_error = await self._run(call, raw_args)
        if len(output) > MAX_RESULT_CHARS:
            output = output[:MAX_RESULT_CHARS] + '..."[truncated]"'
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            arguments=raw_args,
            output=output,
            is_error=is_error,
            latency_s=time.perf_counter() - started,
        )

    async def _run(self, call: ToolCall, raw_args: dict[str, Any]) -> tuple[str, bool]:
        tool = self._tools.get(call.name)
        if tool is None:
            return _error(f"unknown tool {call.name!r}; available: {', '.join(self._tools)}"), True
        try:
            args = tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            return _error(_validation_message(exc)), True
        try:
            result = await asyncio.wait_for(tool.handler(args), timeout=tool.timeout_s)
        except ToolError as exc:  # expected, model-fixable
            return _error(str(exc)), True
        except TimeoutError:
            return _error(f"{call.name} timed out; tell the user the service is slow and to try again"), True
        except ExternalAPIError as exc:
            log.warning("tool %s: external API failure: %s", call.name, exc)
            return _error(f"the {exc.service} service is unavailable right now; tell the user, do not retry"), True
        except StrideDJError as exc:
            log.warning("tool %s failed: %s", call.name, exc)
            return _error(f"{call.name} failed; tell the user something went wrong"), True
        except Exception:  # a bug: log it fully, show the model nothing internal
            log.exception("tool %s crashed", call.name)
            return _error(f"{call.name} failed with an internal error; tell the user something went wrong"), True
        return result.model_dump_json(), False
