.DEFAULT_GOAL := help
PLATFORM ?= tax-booking
SEED ?= 0

.PHONY: help setup data validate-gold lint format test requirements

help:  ## Show the available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup:  ## Create the virtual environment and install dependencies
	uv sync

data:  ## Generate the data splits and the manifest (PLATFORM=, SEED=)
	uv run python -m adapt.data.split --platform $(PLATFORM) --seed $(SEED)

validate-gold:  ## Check the gold file: schema, amounts and accounts (PLATFORM=)
	uv run python -m adapt.data.gold --platform $(PLATFORM)

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
