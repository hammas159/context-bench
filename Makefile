.DEFAULT_GOAL := help

help:  ## Show this help
	@grep -E "^[a-zA-Z_-]+:.*?## " $(MAKEFILE_LIST) | awk "BEGIN{FS=\":.*?## \"}{printf \"  [36m%-10s[0m %s\n\", $$1, $$2}"

install:  ## Create .venv and install
	uv sync --all-groups

test:  ## 26 tests, no network
	uv run pytest -q

bench:  ## Run the full sweep
	uv run python run_bench.py

ui:  ## Serve the UI on :8000
	uv run uvicorn web.app:app --reload --port 8000

lint:  ## Lint and format check
	uv run ruff check src tests web run_bench.py
	uv run ruff format --check src tests web run_bench.py

.PHONY: help install test bench ui lint
