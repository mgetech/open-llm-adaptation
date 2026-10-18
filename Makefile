.DEFAULT_GOAL := help
PLATFORM ?= tax-booking
SEED ?= 0
NUM_GPUS ?= 1
SPLIT ?= dev
ENDPOINTS ?= http://localhost:8000/v1

.PHONY: help setup install-vllm data keep-round validate-gold models-check baseline train eval-sft eval gate lint format test requirements

help:  ## Show the available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup:  ## Create the virtual environment and install dependencies
	uv sync

install-vllm:  ## Install the vLLM version pinned in configs/models.yaml (Linux GPU machines)
	uv pip install --python .venv/bin/python "vllm==$$(uv run python -m adapt.models --vllm-version)"

data:  ## Generate the data splits and the manifest (PLATFORM=, SEED=)
	uv run python -m adapt.data.split --platform $(PLATFORM) --seed $(SEED)

keep-round:  ## Copy the current data to round<N>/ and record it in the manifest (ROUND=, optional NOTE=)
	uv run python -m adapt.data.split --platform $(PLATFORM) --keep-round $(ROUND) $(if $(NOTE),--note "$(NOTE)")

validate-gold:  ## Check the gold file: schema, amounts and accounts (PLATFORM=)
	uv run python -m adapt.data.gold --platform $(PLATFORM)

models-check:  ## Check the model gates and compare the pinned revisions with the Hub (OFFLINE=1, PIN=1)
	uv run python -m adapt.models $(if $(OFFLINE),--offline) $(if $(PIN),--pin)

train:  ## QLoRA fine-tuning of one model (MODEL=, SEED=; optional ROUND=, NUM_GPUS=2, SMOKE=1, RESUME=1, MAX_STEPS=, ADAPTERS_DIR=, RUNS_DIR=)
	uv run --group gpu $(if $(filter-out 1,$(NUM_GPUS)),accelerate launch --multi_gpu --num_processes $(NUM_GPUS) -m adapt.train,python -m adapt.train) \
		--platform $(PLATFORM) --seed $(SEED) --num-gpus $(NUM_GPUS) \
		$(if $(SMOKE),--smoke,--model $(MODEL)) $(if $(RESUME),--resume) \
		$(if $(MAX_STEPS),--max-steps $(MAX_STEPS)) $(if $(ROUND),--round $(ROUND)) \
		$(if $(ADAPTERS_DIR),--adapters-dir $(ADAPTERS_DIR)) $(if $(RUNS_DIR),--runs-dir $(RUNS_DIR))

eval-sft:  ## Evaluate a trained adapter on all splits, with its own server (MODEL=, SEED=; optional ROUND=, SMOKE=1, MAX_STEPS=, NUM_GPUS=, SERVERS=2, SPLITS=, ADAPTERS_DIR=, OUT_DIR=, DRY_RUN=1)
	uv run python -m adapt.train.evaluate --platform $(PLATFORM) --seed $(SEED) \
		$(if $(SMOKE),--smoke,--model $(MODEL)) $(if $(MAX_STEPS),--max-steps $(MAX_STEPS) --num-gpus $(NUM_GPUS)) \
		$(if $(SERVERS),--servers $(SERVERS)) $(if $(ROUND),--round $(ROUND)) $(if $(SPLITS),--splits $(SPLITS)) $(if $(ADAPTERS_DIR),--adapters-dir $(ADAPTERS_DIR)) \
		$(if $(OUT_DIR),--out-dir $(OUT_DIR)) $(if $(DRY_RUN),--dry-run)

eval:  ## Evaluate a system on a split (SYSTEM=, SPLIT=, ENDPOINTS=; optional ROUND=, MODEL=, THRESHOLD=, FEW_SHOT=1, OUT_DIR=)
	uv run python -m adapt.eval.runner --platform $(PLATFORM) --split $(SPLIT) --endpoints $(ENDPOINTS) \
		$(if $(SYSTEM),--system $(SYSTEM)) $(if $(MODEL),--model $(MODEL)) \
		$(if $(THRESHOLD),--threshold $(THRESHOLD)) $(if $(WORKERS),--workers $(WORKERS)) \
		$(if $(MODEL_REVISION),--model-revision $(MODEL_REVISION)) \
		$(if $(ADAPTER_PATH),--adapter-path $(ADAPTER_PATH)) $(if $(ROUND),--round $(ROUND)) $(if $(FEW_SHOT),--few-shot) 		$(if $(OUT_DIR),--out-dir $(OUT_DIR))

baseline:  ## Few-shot baseline of one model, with its own server (MODEL=; optional ROUND=, SPLITS=, OUT_DIR=, DRY_RUN=1)
	uv run python -m adapt.baseline --platform $(PLATFORM) --model $(MODEL) 		$(if $(ROUND),--round $(ROUND)) $(if $(SPLITS),--splits $(SPLITS)) $(if $(OUT_DIR),--out-dir $(OUT_DIR)) $(if $(DRY_RUN),--dry-run)

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
