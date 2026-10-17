from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

from stridedj.domain import AgentReply, DayPart, Frozen
from stridedj.errors import ExternalAPIError, ToolError
from stridedj.harness.tools import Tool, ToolRegistry
from stridedj.llm.base import ToolCall
from stridedj.llm.schema import strict_json_schema


def _walk_objects(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for v in node.values():
            found.extend(_walk_objects(v))
    elif isinstance(node, list):
        for v in node:
            found.extend(_walk_objects(v))
    return found


class Inner(Frozen):
    day_part: DayPart = Field(description="part of the day")
    note: str | None = Field(default=None, description="optional")


class Outer(Frozen):
    name: str
    inner: Inner
    many: list[Inner]


def test_strict_schema_rules() -> None:
    schema = strict_json_schema(Outer)
    objects = _walk_objects(schema)
    assert objects, "expected object schemas"
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"])
    text = json.dumps(schema)
    assert '"default"' not in text
    assert '"title"' not in text
    # a $ref with a sibling description is inlined (strict mode forbids siblings next to $ref)
    for obj in objects:
        for prop in obj["properties"].values():
            assert not ("$ref" in prop and len(prop) > 1)


class HasKeywordNamedFields(Frozen):
    title: str
    default: int


def test_fields_named_like_keywords_survive() -> None:
    schema = strict_json_schema(HasKeywordNamedFields)
    assert set(schema["properties"]) == {"title", "default"}
    assert set(schema["required"]) == {"title", "default"}


def test_agent_reply_schema_is_strict() -> None:
    schema = strict_json_schema(AgentReply)
    assert set(schema["required"]) == {"kind", "message", "proposal_id", "missing_fields"}
    assert schema["additionalProperties"] is False


class KmArgs(Frozen):
    km: float = Field(description="distance")

    @field_validator("km")
    @classmethod
    def _pos(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("km must be positive; ask the runner")
        return v


class KmOut(BaseModel):
    double: float


async def _double(args: KmArgs) -> KmOut:
    return KmOut(double=args.km * 2)


async def _fails_tool(args: KmArgs) -> KmOut:
    raise ToolError("no runs of that type yet; tell the runner")


async def _fails_api(args: KmArgs) -> KmOut:
    raise ExternalAPIError("spotify", "boom", status_code=503)


async def _crashes(args: KmArgs) -> KmOut:
    raise RuntimeError("secret internal detail")


def _registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool("double", "doubles", KmArgs, _double),
            Tool("fails_tool", "x", KmArgs, _fails_tool),
            Tool("fails_api", "x", KmArgs, _fails_api),
            Tool("crashes", "x", KmArgs, _crashes),
        ]
    )


async def test_tool_success_returns_json() -> None:
    result = await _registry().execute(ToolCall(call_id="c1", name="double", arguments_json='{"km": 5}'))
    assert not result.is_error
    assert json.loads(result.output) == {"double": 10.0}


async def test_validation_errors_are_actionable_and_do_not_echo_input() -> None:
    result = await _registry().execute(ToolCall(call_id="c1", name="double", arguments_json='{"km": -3}'))
    assert result.is_error
    assert "km must be positive" in result.output
    assert "-3" not in result.output


async def test_unknown_tool_lists_available() -> None:
    result = await _registry().execute(ToolCall(call_id="c1", name="nope", arguments_json="{}"))
    assert result.is_error
    assert "double" in result.output


async def test_failures_map_to_safe_messages() -> None:
    reg = _registry()
    tool_err = await reg.execute(ToolCall(call_id="1", name="fails_tool", arguments_json='{"km": 1}'))
    api_err = await reg.execute(ToolCall(call_id="2", name="fails_api", arguments_json='{"km": 1}'))
    crash = await reg.execute(ToolCall(call_id="3", name="crashes", arguments_json='{"km": 1}'))
    assert "no runs of that type" in tool_err.output
    assert "spotify service is unavailable" in api_err.output
    assert crash.is_error
    assert "secret internal detail" not in crash.output
