"""Eval tasks: one YAML file per task in `evals/tasks/`.

Vocabulary (Anthropic, "Demystifying evals for AI agents", 2026):
* **task**: inputs + success criteria (this file's `Task`);
* **trial**: one attempt at a task; agents are non-deterministic, so we run k trials per task;
* **grader**: logic that scores one aspect of a trial (`graders.py`);
* **transcript**: the full record of a trial (our `Transcript`);
* **outcome**: the final state of the environment (the proposal, and what was or wasn't written to the fake APIs).

Suites: `regression` (should stay ~100 %; catches backsliding), `capability` (hard; a hill to climb),
`security` (prompt-injection and abuse cases; the attack success rate must be 0).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from stridedj.domain import DayPart, MissingField, Mood, ReplyKind, RunType

TASKS_DIR = Path(__file__).parent / "tasks"  # one YAML file per suite, a list of tasks each
DEFAULT_NOW = datetime.fromisoformat("2026-10-17T15:00:00+02:00")  # Saturday afternoon in Zurich

Suite = Literal["regression", "capability", "security"]


class EnvSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    calendar: Literal["realistic", "empty", "packed_evening"] = "realistic"
    poisoned_calendar: bool = False  # an invite whose title carries an injection
    # (the demo music library always contains two tracks whose titles carry injections)


class Expect(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ReplyKind | list[ReplyKind] = Field(description="the expected reply kind, or a list of acceptable kinds")
    run_type: RunType | None = None
    distance_km: float | None = None
    date: str | None = Field(default=None, description="'today', 'tomorrow', a weekday ('monday') or YYYY-MM-DD")
    day_part: DayPart | None = None
    playlist: bool | None = None
    mood: Mood | None = None
    missing_fields: list[MissingField] = Field(default_factory=list)
    must_call: list[str] = Field(default_factory=list)
    max_steps: int = 8
    max_tool_calls: int = 12


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    suite: Suite
    description: str
    turns: list[str] = Field(min_length=1, description="user messages; the last one is graded")
    now: AwareDatetime = DEFAULT_NOW
    env: EnvSpec = Field(default_factory=EnvSpec)
    expect: Expect
    attack_markers: list[str] = Field(
        default_factory=list, description="strings that must never reach the model's context or its answer"
    )


def load_tasks(directory: Path = TASKS_DIR, *, suites: set[str] | None = None) -> list[Task]:
    tasks = []
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for entry in data if isinstance(data, list) else [data]:
            task = Task.model_validate(entry)
            if suites is None or task.suite in suites:
                tasks.append(task)
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids")
    return tasks


def expected_kinds(expect: Expect) -> list[ReplyKind]:
    return expect.kind if isinstance(expect.kind, list) else [expect.kind]
