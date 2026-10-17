.DEFAULT_GOAL := help
.PHONY: help install fmt lint typecheck test cov check doctor

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## create .venv (Python 3.12) and install all dependencies
	uv sync

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

doctor:  ## check every credential in .env (prints ✓/✗ only, never the values)
	uv run stridedj doctor
