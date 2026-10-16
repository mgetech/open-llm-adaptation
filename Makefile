.DEFAULT_GOAL := help
.PHONY: help setup lint format test requirements

help:  ## Show the available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup:  ## Create the virtual environment and install dependencies
	uv sync

lint:  ## Check style and formatting
	uv run ruff check .
	uv run ruff format --check .

format:  ## Fix style issues and format the code
	uv run ruff check --fix .
	uv run ruff format .

test:  ## Run the tests
	uv run pytest

requirements:  ## Export pinned requirements for GPU machines from uv.lock
	uv export --frozen --group gpu --no-dev --no-emit-project --no-hashes -o requirements-gpu.txt
