.DEFAULT_GOAL := help
.PHONY: help setup install fmt lint typecheck test cov check doctor smoke demo bot eval eval-reference

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

# ── getting started ───────────────────────────────────────────────────────────
setup: install  ## first-time setup: venv + dependencies + .env from the template
	@test -f .env || (cp .env.example .env && chmod 600 .env && echo "created .env from .env.example; now follow docs/SETUP.md")

install:  ## create .venv (Python 3.12) and install all dependencies
	uv sync

doctor:  ## check every credential in .env (prints ✓/✗ only, never the values)
	uv run stridedj doctor

smoke:  ## read-only live check of Garmin, Spotify, ReccoBeats and Google (prints counts only)
	uv run python scripts/smoke_live.py

# ── running ───────────────────────────────────────────────────────────────────
demo:  ## chat in the terminal on synthetic data (needs only OPENAI_API_KEY)
	uv run stridedj chat --data demo

bot:  ## run the Telegram bot on your real accounts
	uv run stridedj bot --data live

# ── evals ─────────────────────────────────────────────────────────────────────
eval-reference:  ## evals with the oracle policy (free; must be 100%)
	uv run python -m evals.run_eval --reference --fail-under 1.0

eval:  ## evals with the configured OpenAI model, 3 trials per task (≈ $0.05)
	uv run python -m evals.run_eval --k 3

# ── quality ───────────────────────────────────────────────────────────────────
fmt:  ## auto-format and auto-fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

typecheck:  ## mypy --strict
	uv run mypy

test:  ## unit tests (offline)
	uv run pytest

cov:  ## unit tests with coverage
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test  ## everything CI runs
