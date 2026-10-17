from __future__ import annotations

from pydantic import BaseModel

from stridedj.domain import AgentReply, Frozen, ReplyKind
from stridedj.harness.loop import Budget, run_agent
from stridedj.harness.tools import Tool, ToolRegistry
from stridedj.llm.fake import ScriptedLLM, ScriptedTurn


class EchoArgs(Frozen):
    text: str


class EchoOut(BaseModel):
    echoed: str


async def _echo(args: EchoArgs) -> EchoOut:
    return EchoOut(echoed=args.text)


def _tools() -> ToolRegistry:
    return ToolRegistry([Tool("echo", "echo text", EchoArgs, _echo)])


def _answer(msg: str = "done", kind: ReplyKind = ReplyKind.ANSWER, proposal_id: str | None = None) -> AgentReply:
    return AgentReply(kind=kind, message=msg, proposal_id=proposal_id, missing_fields=[])


async def _run(llm: ScriptedLLM, **kw: object) -> object:
    return await run_agent(
        llm=llm,
        tools=_tools(),
        instructions="sys",
        history=[],
        user_message="hi",
        user_id="u1",
        prompt_version="test",
        **kw,  # type: ignore[arg-type]
    )


async def test_tool_call_then_final_answer() -> None:
    llm = ScriptedLLM(
        [
            ScriptedTurn(tool_calls=[("echo", {"text": "a"}), ("echo", {"text": "b"})]),
            ScriptedTurn(final=_answer("ok")),
        ]
    )
    outcome = await run_agent(
        llm=llm, tools=_tools(), instructions="sys", history=[], user_message="hi", user_id="u1", prompt_version="t"
    )
    assert outcome.stop_reason == "final"
    assert outcome.reply.message == "ok"
    assert outcome.format_ok
    assert outcome.transcript.tool_calls == 2
    assert outcome.transcript.steps == 2
    # both tool results were sent back to the model, linked by call id
    second_call_items = llm.calls[1]["items"]
    outputs = [i for i in second_call_items if i.get("type") == "function_call_output"]
    assert [o["call_id"] for o in outputs] == ["call_1_0", "call_1_1"]


async def test_invalid_json_gets_one_correction() -> None:
    llm = ScriptedLLM([ScriptedTurn(raw_text="not json"), ScriptedTurn(final=_answer("fixed"))])
    outcome = await run_agent(
        llm=llm, tools=_tools(), instructions="s", history=[], user_message="hi", user_id="u", prompt_version="t"
    )
    assert outcome.stop_reason == "final"
    assert outcome.reply.message == "fixed"
    assert outcome.format_ok is False
    assert any(i.get("role") == "developer" for i in llm.calls[1]["items"])


async def test_reply_check_failure_then_fallback() -> None:
    bad = _answer("booked!", kind=ReplyKind.PROPOSAL, proposal_id="made-up")
    llm = ScriptedLLM([ScriptedTurn(final=bad), ScriptedTurn(final=bad)])
    outcome = await run_agent(
        llm=llm,
        tools=_tools(),
        instructions="s",
        history=[],
        user_message="hi",
        user_id="u",
        prompt_version="t",
        reply_check=lambda r: "unknown proposal" if r.kind is ReplyKind.PROPOSAL else None,
    )
    assert outcome.stop_reason == "invalid_output"
    assert outcome.reply.kind is ReplyKind.ERROR


async def test_step_budget_terminates_a_looping_model() -> None:
    llm = ScriptedLLM(lambda items: ScriptedTurn(tool_calls=[("echo", {"text": "again"})]))
    outcome = await run_agent(
        llm=llm,
        tools=_tools(),
        instructions="s",
        history=[],
        user_message="hi",
        user_id="u",
        prompt_version="t",
        budget=Budget(max_steps=3),
    )
    assert outcome.stop_reason == "max_steps"
    assert outcome.transcript.steps == 3
    assert outcome.reply.kind is ReplyKind.ERROR


async def test_tool_call_budget() -> None:
    llm = ScriptedLLM(lambda items: ScriptedTurn(tool_calls=[("echo", {"text": "x"})] * 3))
    outcome = await run_agent(
        llm=llm,
        tools=_tools(),
        instructions="s",
        history=[],
        user_message="hi",
        user_id="u",
        prompt_version="t",
        budget=Budget(max_tool_calls=4),
    )
    assert outcome.stop_reason == "max_tool_calls"
    assert outcome.transcript.tool_calls == 3
