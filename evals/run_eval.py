"""Run the eval suites: `python -m evals.run_eval [--reference | --vulnerable | --model M] [--k 3] [--suite S]`.

The evaluation harness: for each task × trial it builds a FRESH offline app (same `StrideAgent` as production, fake
adapters, frozen clock), plays the user turns, grades the outcome + transcript, and aggregates pass rates, pass@k,
pass^k, attack success rate, cost and latency into `evals/runs/<timestamp>/report.md`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from evals.graders import Grade, TrialRecord, grade
from evals.metrics import pass_at_k, pass_hat_k, percentile
from evals.reference import reference_policy
from evals.tasks import Task, load_tasks
from stridedj.agent.service import PROMPTS_DIR
from stridedj.app import build_demo_app
from stridedj.config import load_settings
from stridedj.costs.ledger import CostLedger, UsageSink, default_ledger_path, key_fingerprint
from stridedj.domain import UserContext, UserId
from stridedj.harness.context import Prompt
from stridedj.llm.base import LLMClient
from stridedj.llm.fake import ScriptedLLM, VulnerableLLM
from stridedj.llm.openai_client import OpenAIResponsesClient

RUNS_DIR = Path(__file__).parent / "runs"
LLMFactory = Callable[[Task], LLMClient]


class TrialResult(BaseModel):
    task_id: str
    suite: str
    trial: int
    passed: bool
    partial: float
    grades: list[Grade]
    cost_usd: float
    duration_s: float
    steps: int
    tool_calls: int
    reply_kind: str
    reply_message: str


async def run_trial(
    task: Task,
    trial: int,
    llm_factory: LLMFactory,
    out_dir: Path,
    prompt: Prompt | None = None,
    usage_sink: UsageSink | None = None,
) -> TrialResult:
    app = await build_demo_app(
        llm_factory(task),
        now=task.now,
        calendar=task.env.calendar,
        poisoned_calendar=task.env.poisoned_calendar,
        prompt=prompt,
        usage_sink=usage_sink,
    )
    user = UserContext(user_id=UserId(f"eval-{task.id}-{trial}"), timezone=app.timezone)
    responses = [await app.agent.handle(user, turn) for turn in task.turns]
    record = TrialRecord(
        task=task,
        trial=trial,
        responses=responses,
        calendar_writes=len(app.fake_calendar.inserted),
        playlist_writes=len(app.fake_writer.created),
        timezone=app.timezone,
    )
    grades = grade(record)
    for i, response in enumerate(responses):
        path = out_dir / "transcripts" / f"{task.id}__t{trial}__turn{i}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.outcome.transcript.model_dump_json(indent=2))
    transcripts = [r.outcome.transcript for r in responses]
    return TrialResult(
        task_id=task.id,
        suite=task.suite,
        trial=trial,
        passed=all(g.passed for g in grades),
        partial=sum(g.passed for g in grades) / len(grades),
        grades=grades,
        cost_usd=sum(t.cost_usd for t in transcripts),
        duration_s=sum(t.duration_s for t in transcripts),
        steps=sum(t.steps for t in transcripts),
        tool_calls=sum(t.tool_calls for t in transcripts),
        reply_kind=record.final.reply.kind.value,
        reply_message=record.final.reply.message,
    )


async def run_all(
    tasks: list[Task],
    k: int,
    llm_factory: LLMFactory,
    out_dir: Path,
    concurrency: int,
    prompt: Prompt | None = None,
    usage_sink: UsageSink | None = None,
) -> list[TrialResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(task: Task, trial: int) -> TrialResult:
        async with semaphore:
            return await run_trial(task, trial, llm_factory, out_dir, prompt, usage_sink)

    return list(await asyncio.gather(*(guarded(t, i) for t in tasks for i in range(k))))


def report(results: list[TrialResult], *, k: int, label: str) -> str:
    by_task: dict[str, list[TrialResult]] = defaultdict(list)
    for r in results:
        by_task[r.task_id].append(r)
    lines = [
        f"# Eval report: {label}",
        "",
        f"{len(by_task)} tasks × {k} trials, {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        "",
    ]
    lines += [
        "| suite | tasks | trial pass rate | pass@k | pass^k | mean partial credit |",
        "|---|---|---|---|---|---|",
    ]
    suites = sorted({r.suite for r in results})
    for suite in suites:
        tasks = [t for t, rs in by_task.items() if rs[0].suite == suite]
        rs = [r for r in results if r.suite == suite]
        pk = sum(pass_at_k(k, sum(r.passed for r in by_task[t]), k) for t in tasks) / len(tasks)
        phk = sum(pass_hat_k(k, sum(r.passed for r in by_task[t]), k) for t in tasks) / len(tasks)
        rate = sum(r.passed for r in rs) / len(rs)
        partial = sum(r.partial for r in rs) / len(rs)
        lines.append(f"| {suite} | {len(tasks)} | {rate:.0%} | {pk:.0%} | {phk:.0%} | {partial:.0%} |")
    sec = [r for r in results if r.suite == "security"]
    if sec:
        attacked = [
            r
            for r in sec
            if any(g.name in ("no_injection", "no_writes_without_approval") and not g.passed for g in r.grades)
        ]
        lines += ["", f"**Attack success rate (security suite): {len(attacked)}/{len(sec)} trials**"]
    durations = [r.duration_s for r in results]
    cost = sum(r.cost_usd for r in results)
    lines += [
        "",
        f"Cost: ${cost:.4f} total, ${cost / len(results):.5f} per trial · "
        f"latency per trial p50 {percentile(durations, 50):.1f}s, p95 {percentile(durations, 95):.1f}s · "
        f"mean steps {sum(r.steps for r in results) / len(results):.1f}, "
        f"mean tool calls {sum(r.tool_calls for r in results) / len(results):.1f}",
        "",
        "## Per task",
        "| task | passed | failing graders (trial: detail) |",
        "|---|---|---|",
    ]
    for task_id, rs in by_task.items():
        failures = [f"t{r.trial} {g.name}: {g.detail}" for r in rs for g in r.grades if not g.passed]
        detail = "<br>".join(f.replace("|", "/") for f in failures[:4]) or ""
        lines.append(f"| {task_id} | {sum(r.passed for r in rs)}/{len(rs)} | {detail} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reference", action="store_true", help="oracle policy: must score 100%% (tests the tasks)")
    mode.add_argument("--vulnerable", action="store_true", help="gullible fake model (security demonstration)")
    mode.add_argument("--model", help="OpenAI model id (default: OPENAI_MODEL)")
    parser.add_argument("--k", type=int, default=1, help="trials per task")
    parser.add_argument("--suite", action="append", choices=["regression", "capability", "security"])
    parser.add_argument("--task", action="append", help="only these task ids")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--prompt", help="system prompt version, e.g. system_v1 (default: the agent's current one)")
    parser.add_argument("--fail-under", type=float, default=None, help="exit 1 if the trial pass rate is below this")
    args = parser.parse_args()

    tasks = load_tasks(suites=set(args.suite) if args.suite else None)
    if args.task:
        tasks = [t for t in tasks if t.id in set(args.task)]

    factory: LLMFactory
    sink: UsageSink | None = None  # real-model eval runs cost money: record them in the cost ledger
    if args.reference:
        factory, label = (lambda task: ScriptedLLM(reference_policy(task))), "reference"
    elif args.vulnerable:
        factory, label = (lambda task: VulnerableLLM(reference_policy(task))), "vulnerable"
    else:
        settings = load_settings().openai
        settings = settings.model_copy(update={"model": args.model}) if args.model else settings
        client = OpenAIResponsesClient.from_settings(settings)
        factory, label = (lambda task: client), settings.model
        if settings.api_key is not None:
            ledger = CostLedger(default_ledger_path(load_settings().app.data_dir))
            sink = ledger.sink(key_fp=key_fingerprint(settings.api_key.get_secret_value()), source="eval")

    prompt = Prompt.load(PROMPTS_DIR / f"{args.prompt}.md") if args.prompt else None
    label = f"{label}_{args.prompt}" if args.prompt else label
    out_dir = RUNS_DIR / f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{label}"
    results = asyncio.run(run_all(tasks, args.k, factory, out_dir, args.concurrency, prompt, sink))
    text = report(results, k=args.k, label=label)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(text)
    (out_dir / "results.json").write_text(json.dumps([r.model_dump() for r in results], indent=2))
    print(text)
    print(f"→ {out_dir}")
    rate = sum(r.passed for r in results) / len(results)
    if args.fail_under is not None and rate < args.fail_under:
        raise SystemExit(f"pass rate {rate:.0%} < {args.fail_under:.0%}")


if __name__ == "__main__":
    main()
