from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from pydantic import SecretStr

from stridedj.costs.dashboard import build_view, render_html, usd
from stridedj.costs.ledger import CostLedger, key_fingerprint
from stridedj.costs.openai_costs import ApiKeyInfo, DailyKeyCost, OfficialCosts, _parse_buckets, redacted_matches
from stridedj.costs.summary import month_summary
from stridedj.harness.tracing import Transcript
from stridedj.llm.base import Usage

NOW = datetime(2026, 10, 17, 15, 0, tzinfo=UTC)
KEY = "sk-proj-zz9SECRETVALUEabcd"


def _transcript(run_id: str, *, cost: float, when: datetime, cached: int = 0) -> Transcript:
    t = Transcript(run_id=run_id, user_id="tg:1", model="gpt-6-luna", prompt_version="system_v2", started_at=when)
    t.usage = Usage(input_tokens=1000, cached_input_tokens=cached, output_tokens=100)
    t.cost_usd, t.steps, t.tool_calls, t.stop_reason, t.duration_s = cost, 4, 4, "final", 5.0
    return t


def test_fingerprint_is_stable_short_and_not_the_key() -> None:
    fp = key_fingerprint(KEY)
    assert fp == key_fingerprint(KEY)
    assert len(fp) == 12
    assert "SECRET" not in fp
    assert key_fingerprint("sk-other") != fp


def test_ledger_records_and_filters(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "costs.db")
    sink = ledger.sink(key_fp="aaa", source="bot")
    sink.record(_transcript("r1", cost=0.001, when=NOW - timedelta(days=2)))
    sink.record(_transcript("r2", cost=0.002, when=NOW - timedelta(days=40)))
    ledger.sink(key_fp="bbb", source="eval").record(_transcript("r3", cost=0.5, when=NOW))
    recent = ledger.runs(since=NOW - timedelta(days=30))
    assert {r.run_id for r in recent} == {"r1", "r3"}
    assert [r.run_id for r in ledger.runs(since=NOW - timedelta(days=30), key_fp="aaa")] == ["r1"]
    assert recent[0].run_id == "r3"  # newest first


def test_redacted_value_matching() -> None:
    assert redacted_matches("sk-proj-****abcd", KEY)
    assert redacted_matches("sk-pr...abcd", KEY)
    assert not redacted_matches("sk-proj-****wxyz", KEY)
    assert not redacted_matches("sk-proj-abcd", KEY)  # not a masked value


def test_parse_cost_buckets() -> None:
    day = int(datetime(2026, 10, 16, tzinfo=UTC).timestamp())
    page = SimpleNamespace(
        data=[
            SimpleNamespace(
                start_time=day,
                results=[
                    SimpleNamespace(
                        amount=SimpleNamespace(value=0.12, currency="usd"),
                        api_key_id="key_1",
                        line_item="gpt-6-luna, input",
                    ),
                    SimpleNamespace(amount=None, api_key_id="key_1", line_item="x"),
                    SimpleNamespace(
                        amount=SimpleNamespace(value=0.03, currency="usd"), api_key_id=None, line_item=None
                    ),
                ],
            )
        ]
    )
    costs = _parse_buckets([page])
    assert costs[0] == DailyKeyCost(
        day=date(2026, 10, 16), api_key_id="key_1", line_item="gpt-6-luna, input", amount_usd=0.12
    )
    assert costs[1].api_key_id == "unattributed"
    assert len(costs) == 2


def test_view_links_ledger_to_the_official_key_and_html_hides_secrets(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "costs.db")
    fp = key_fingerprint(KEY)
    ledger.sink(key_fp=fp, source="bot").record(
        _transcript("r1", cost=0.004, when=NOW - timedelta(hours=3), cached=800)
    )
    ledger.sink(key_fp="feedbeef0000", source="cli").record(_transcript("r2", cost=0.001, when=NOW - timedelta(days=1)))
    official = OfficialCosts(
        costs=[
            DailyKeyCost(day=date(2026, 10, 16), api_key_id="key_1", line_item="gpt-6-luna, input", amount_usd=0.02),
            DailyKeyCost(day=date(2026, 10, 1), api_key_id="key_2", line_item="gpt-6.1-sol, output", amount_usd=1.5),
        ],
        keys={
            "key_1": ApiKeyInfo(
                api_key_id="key_1", name="stridedj-laptop", project_name="Default", redacted_value="sk-proj-****abcd"
            ),
            "key_2": ApiKeyInfo(
                api_key_id="key_2", name="other-app", project_name="Default", redacted_value="sk-proj-****9999"
            ),
        },
        fetched_at=NOW,
    )
    view = build_view(
        ledger_runs=ledger.runs(since=NOW - timedelta(days=30)),
        official=official,
        current_key=SecretStr(KEY),
        days=30,
        now=NOW,
    )
    labels = [k.label for k in view.keys]
    assert labels[0] == "stridedj-laptop"  # the configured key comes first
    current = view.keys[0]
    assert current.is_current
    assert current.official_period_usd == 0.02
    assert current.ledger_runs == 1
    assert current.cache_hit_rate == 0.8
    assert "other-app" in labels
    assert any(label.startswith("key ·feedbeef") for label in labels)  # unmatched ledger key gets its own section
    assert view.total_official_month_usd == 1.52
    page = render_html(view)
    assert "SECRETVALUE" not in page
    assert "<svg" in page and "stridedj-laptop" in page


def test_view_without_admin_key_uses_the_ledger(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "costs.db")
    ledger.sink(key_fp=key_fingerprint(KEY), source="bot").record(_transcript("r1", cost=0.004, when=NOW))
    view = build_view(
        ledger_runs=ledger.runs(since=NOW - timedelta(days=7)),
        official=None,
        current_key=SecretStr(KEY),
        days=7,
        now=NOW,
    )
    assert not view.has_official
    assert view.keys[0].daily_source == "ledger"
    assert view.keys[0].daily[-1][1] == 0.004
    assert "OPENAI_ADMIN_KEY" in render_html(view)


def test_month_summary_and_formatting(tmp_path: Path) -> None:
    ledger = CostLedger(tmp_path / "costs.db")
    sink = ledger.sink(key_fp="abcdef123456", source="bot")
    sink.record(_transcript("r1", cost=0.0004, when=NOW - timedelta(hours=1)))
    sink.record(_transcript("r2", cost=0.0006, when=NOW - timedelta(days=3)))
    text = month_summary(ledger, key_fp="abcdef123456", now=NOW)
    assert "2 conversation(s)" in text
    assert "$0.0010" in text
    assert usd(0) == "$0" and usd(0.5) == "$0.500" and usd(12.3456) == "$12.35"
