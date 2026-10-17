"""Read a transcript in the terminal: `python -m evals.view evals/runs/<run>/<task>__t0.json`.

"Read the transcripts!" is the most repeated advice in Anthropic's evals guide: a score means nothing until you've
looked at what the agent actually did in a few passing and failing trials.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from stridedj.harness.tracing import Transcript

console = Console()


def show(path: Path) -> None:
    t = Transcript.model_validate_json(path.read_text(encoding="utf-8"))
    console.print(
        Panel(
            f"model={t.model} prompt={t.prompt_version} steps={t.steps} tool_calls={t.tool_calls} "
            f"tokens in/out={t.usage.input_tokens}/{t.usage.output_tokens} cost=${t.cost_usd:.4f} "
            f"time={t.duration_s:.1f}s stop={t.stop_reason}",
            title=path.name,
        )
    )
    for e in t.events:
        d = e.data
        if e.kind == "user":
            console.print(f"[bold cyan]{e.t_s:6.2f}s USER[/] {d['text']}")
        elif e.kind == "llm":
            calls = ", ".join(f"{c['name']}({c['arguments']})" for c in d.get("tool_calls", []))
            what = f"→ {calls}" if calls else f"→ text: {d.get('text')}"
            console.print(f"[bold magenta]{e.t_s:6.2f}s LLM[/] ({d['latency_s']}s) {what}")
        elif e.kind == "tool":
            colour = "red" if d.get("is_error") else "green"
            output = json.dumps(json.loads(d["output"]), ensure_ascii=False)[:400]
            console.print(f"[bold {colour}]{e.t_s:6.2f}s TOOL {d['name']}[/] {output}")
        elif e.kind == "harness":
            console.print(f"[bold yellow]{e.t_s:6.2f}s HARNESS[/] {d}")
        else:
            console.print(f"[bold]{e.t_s:6.2f}s FINAL[/] {d['reply']} ({d['stop_reason']})")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        show(Path(arg))
