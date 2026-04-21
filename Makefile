.PHONY: install format lint test dev run

install:
	uv sync --extra dev

format:
	uv run ruff format .

lint:
	uv run ruff check .
	uv run mypy .

test:
	uv run pytest

dev: format lint test

run:
	@if [ -z "$(intent)" ]; then \
		echo "Usage: make run intent=\"your intent here\""; \
	else \
		uv run python run.py run "$(intent)"; \
	fi
