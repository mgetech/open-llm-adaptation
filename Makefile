.DEFAULT_GOAL := help
PLATFORM ?= tax-booking
SEED ?= 0
SPLIT ?= dev
ENDPOINT ?= http://localhost:8000/v1

.PHONY: help setup install-vllm data validate-gold models-check baseline eval gate lint format test requirements

help:  ## Show the available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup:  ## Create the virtual environment and install dependencies
	uv sync

install-vllm:  ## Install the vLLM version pinned in configs/models.yaml (Linux GPU machines)
	uv pip install --python .venv/bin/python "vllm==$$(uv run python -m adapt.models --vllm-version)"

data:  ## Generate the data splits and the manifest (PLATFORM=, SEED=)
	uv run python -m adapt.data.split --platform $(PLATFORM) --seed $(SEED)

validate-gold:  ## Check the gold file: schema, amounts and accounts (PLATFORM=)
	uv run python -m adapt.data.gold --platform $(PLATFORM)

models-check:  ## Check the model gates and compare the pinned revisions with the Hub (OFFLINE=1, PIN=1)
	uv run python -m adapt.models $(if $(OFFLINE),--offline) $(if $(PIN),--pin)

eval:  ## Evaluate a system on a split (SYSTEM=, SPLIT=, ENDPOINT=; optional MODEL=, THRESHOLD=, FEW_SHOT=1, OUT_DIR=)
	uv run python -m adapt.eval.runner --platform $(PLATFORM) --split $(SPLIT) --endpoint $(ENDPOINT) \
		$(if $(SYSTEM),--system $(SYSTEM)) $(if $(MODEL),--model $(MODEL)) \
		$(if $(THRESHOLD),--threshold $(THRESHOLD)) $(if $(WORKERS),--workers $(WORKERS)) \
		$(if $(MODEL_REVISION),--model-revision $(MODEL_REVISION)) \
		$(if $(ADAPTER_PATH),--adapter-path $(ADAPTER_PATH)) $(if $(FEW_SHOT),--few-shot) 		$(if $(OUT_DIR),--out-dir $(OUT_DIR))

baseline:  ## Few-shot baseline of one model on all splits, with its own server (MODEL=; optional OUT_DIR=, DRY_RUN=1)
	uv run python -m adapt.baseline --platform $(PLATFORM) --model $(MODEL) 		$(if $(OUT_DIR),--out-dir $(OUT_DIR)) $(if $(DRY_RUN),--dry-run)

gate:  ## Check the results of a system against the gate (SYSTEM=, optional BASELINE=, OUT_DIR=)
	uv run python -m adapt.eval.gate --platform $(PLATFORM) $(if $(SYSTEM),--system $(SYSTEM)) $(if $(BASELINE),--baseline $(BASELINE)) 		$(if $(OUT_DIR),--out-dir $(OUT_DIR))

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
