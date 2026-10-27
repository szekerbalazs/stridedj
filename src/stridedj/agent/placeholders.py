"""Placeholders: the model names the facts of a proposal, code fills them in, and checks every other number.

The model copying numbers is where it can slip (a time from another slot, a rounded duration): "only state numbers a
tool returned" was a prompt rule nothing enforced. So a proposal's message is a template, "Booked {start}, about
{expected}: {session}. {music}.", filled from the plan in code. A number the model still writes itself ("your 10 km
run") must be one of the plan's own facts or a number the runner wrote in this chat ("5x1000 with 400 m jogs",
"zone 2"): an echo of the runner's own words is fine, a mis-copied one ("6x1000") is not. Zone names like "Z2" are
words. A message with a wrong number or an unknown placeholder is never shown: it is replaced by a standard message
built from the same placeholders, and the trace records it, so the evals measure how often that happens. (A first
version banned every digit; the evals showed that mostly flagged "Z2" and correct echoes of the distance.) Session
offers carry no numbers: the buttons do.

Answers to questions ("what is my median pace?") keep plain numbers from the tools; the prompt rule covers them.
"""

from __future__ import annotations

import re
from zoneinfo import ZoneInfo

from stridedj.domain import RunPlan
from stridedj.harness.executor import RUN_TYPE_LABEL

PROPOSAL_PLACEHOLDERS: tuple[str, ...] = (
    "start",
    "date",
    "time",
    "slot",
    "expected",
    "p80",
    "distance",
    "run_type",
    "session",
    "music",
)
DEFAULT_PROPOSAL = "Planned: {distance} {run_type} on {start}, about {expected}. {music}."
DEFAULT_SESSION_OFFER = "Which session would you like? Tap one below."

_PLACEHOLDER = re.compile(r"\{([a-z_0-9]*)\}")
_THOUSANDS = re.compile(r"\d{1,3}\.\d{3}")
_NUMBER = re.compile(r"(?<![Zz])\d+(?:[.,:]\d+)?")  # "Z2" is a zone name; "5x1000" has 5 and 1000; "10k" has 10


def numbers_in(text: str) -> set[str]:
    """The numbers in a text, normalised: "8,3" → "8.3"; "1,000" / "1.000" (a thousands separator) → "1000"."""
    found = set()
    for raw in _NUMBER.findall(text):
        number = raw.replace(",", ".")
        found.add(number.replace(".", "") if _THOUSANDS.fullmatch(number) else number)
    return found


def problem(template: str, allowed: tuple[str, ...], facts: set[str] | None = None) -> str | None:
    """Why `template` can't be shown as is (an unknown placeholder, or a number that is not one of `facts`), or None."""
    unknown = sorted({name for name in _PLACEHOLDER.findall(template) if name not in allowed})
    if unknown:
        return f"unknown placeholder(s) {', '.join('{' + n + '}' for n in unknown)}"
    wrong = sorted(numbers_in(_PLACEHOLDER.sub("", template)) - (facts or set()))
    if wrong:
        return f"number(s) that are not facts of the plan: {', '.join(wrong)}"
    return None


def plan_values(plan: RunPlan, tz: ZoneInfo) -> dict[str, str]:
    start, end = plan.slot.start.astimezone(tz), plan.slot.end.astimezone(tz)
    d = plan.duration
    if plan.playlist is not None and plan.playlist.track_ids:
        music = f"a {len(plan.playlist.track_ids)}-track playlist at {plan.cadence.spm:.0f} steps/min"
    else:
        music = "no playlist"
    return {
        "start": f"{start:%a %d %b}, {start:%H:%M}",
        "date": f"{start:%a %d %b}",
        "time": f"{start:%H:%M}",
        "slot": f"{start:%H:%M}–{end:%H:%M}",
        "expected": f"{d.p50_min:.0f} min",
        "p80": f"{d.p80_min:.0f} min",
        "distance": f"{plan.distance_km:g} km",
        "run_type": RUN_TYPE_LABEL[plan.run_type],
        "session": plan.session.main_set if plan.session is not None else "",
        "music": music,
    }


def fill(template: str, values: dict[str, str]) -> str:
    """Substitute the placeholders (unknown ones must have been rejected by `problem` before)."""
    text = _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), template)
    text = re.sub(r"\s+([.,:;!?])", r"\1", " ".join(text.split()))  # an empty {session} leaves no " ." behind
    return re.sub(r"[,:;]+([.!?])", r"\1", text)  # ... nor a dangling ":" before the full stop


def render_proposal_message(
    template: str, plan: RunPlan, tz: ZoneInfo, *, runner_said: str = ""
) -> tuple[str, str | None]:
    """(text to show, why the model's template was replaced or None). `runner_said`: the runner's messages."""
    values = plan_values(plan, tz)
    zone = {plan.run_type.value[1:]} if plan.run_type.value[:1] == "z" else set()  # "zone 2" for a z2 run
    facts = set().union(zone, numbers_in(runner_said), *(numbers_in(v) for v in values.values()))
    reason = problem(template, PROPOSAL_PLACEHOLDERS, facts)
    return fill(DEFAULT_PROPOSAL if reason else template, values), reason


def render_offer_message(template: str) -> tuple[str, str | None]:
    """A session offer: no numbers and no placeholders (the buttons carry them)."""
    reason = problem(template, ())
    return (DEFAULT_SESSION_OFFER if reason else template), reason
