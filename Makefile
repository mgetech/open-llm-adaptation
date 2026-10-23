.DEFAULT_GOAL := help
PLATFORM ?= tax-booking
SEED ?= 0
NUM_GPUS ?= 1
SPLIT ?= dev
ENDPOINTS ?= http://localhost:8000/v1

.PHONY: help setup install-vllm fetch-weights fetch-adapters serve-local gateway-key compose-env compose-config compose-up data rewrite-export rewrite-check audit-export audit-compare keep-round validate-gold models-check baseline train select eval-sft eval pair rescore threshold-report stage-gate post-hoc post-hoc-gold gate seal declare-claim lint format test requirements

help:  ## Show the available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup:  ## Create the virtual environment and install dependencies
	uv sync

install-vllm:  ## Install the vLLM version pinned in configs/models.yaml (Linux GPU machines)
	uv pip install --python .venv/bin/python "vllm==$$(uv run python -m adapt.models --vllm-version)"

fetch-weights:  ## Download the base model of configs/serve.yaml once, so that serving is offline (optional WEIGHTS_DIR=weights for the containers)
	uv run python -m adapt.serve --fetch-weights $(if $(WEIGHTS_DIR),--weights-dir $(WEIGHTS_DIR))

fetch-adapters:  ## Download the adapters of configs/serve.yaml from the Hub at the pinned revision and check their hash (needs HF_TOKEN in the environment or in .env; optional ADAPTERS_DIR=adapters)
	uv run python -m adapt.serve --fetch-adapters $(if $(ADAPTERS_DIR),--adapters-dir $(ADAPTERS_DIR))

serve-local:  ## Start vLLM and the gateway from configs/serve.yaml and configs/keys.yaml (optional PRINT=1 shows the commands only, NO_GATEWAY=1 starts vLLM alone)
	uv run python -m adapt.serve $(if $(PRINT),--print) $(if $(NO_GATEWAY),--no-gateway)

compose-env:  ## Write deploy/vllm.env, the arguments and environment of the vLLM container, from configs/serve.yaml
	uv run python -m adapt.serve --compose-env deploy/vllm.env

compose-config:  ## Check deploy/compose.yaml and print it with its values filled in
	docker compose --env-file deploy/vllm.env -f deploy/compose.yaml config

compose-up:  ## Build the gateway image and start vLLM and the gateway (needs a GPU, the weights, the adapter and configs/keys.yaml)
	docker compose --env-file deploy/vllm.env -f deploy/compose.yaml up --build -d

gateway-key:  ## Make an API key and print the lines to add to configs/keys.yaml (ID=name; optional KEY_PLATFORMS="tax-booking")
	uv run python -m adapt.gateway.keys --id $(ID) $(if $(KEY_PLATFORMS),--platforms $(KEY_PLATFORMS))

data:  ## Generate the data splits and the manifest (PLATFORM=, SEED=, optional TRAIN_SIZE=, REWRITES=folder)
	uv run python -m adapt.data.split --platform $(PLATFORM) --seed $(SEED) $(if $(TRAIN_SIZE),--train-size $(TRAIN_SIZE)) $(if $(REWRITES),--rewrites $(REWRITES))

rewrite-export:  ## Make the tasks and the folder (and zip) for the rewriter (SEED=, optional SHARE=0.5, TRAIN_SIZE=, OUT=)
	uv run python -m adapt.data.rewrite_export --platform $(PLATFORM) --seed $(SEED) $(if $(SHARE),--share $(SHARE)) $(if $(TRAIN_SIZE),--train-size $(TRAIN_SIZE)) $(if $(OUT),--out $(OUT))

audit-export:  ## Make the folder (and zip) with the three jobs for a second author (GUIDE=file, optional OUT=, TRAIN_SIZE=, DATA_SEED=)
	uv run python -m adapt.data.audit export --platform $(PLATFORM) --guide $(GUIDE) $(if $(OUT),--out $(OUT)) $(if $(TRAIN_SIZE),--train-size $(TRAIN_SIZE)) $(if $(DATA_SEED),--data-seed $(DATA_SEED))

audit-compare:  ## Compare the labeller's answers with our labels (ANSWERS=file, optional REPORT=)
	uv run python -m adapt.data.audit compare --platform $(PLATFORM) --answers $(ANSWERS) $(if $(REPORT),--report $(REPORT))

rewrite-check:  ## Check the replies of the rewriter and write accepted.jsonl (optional DIR=data/rewrite/round3)
	uv run python -m adapt.data.rewrite_check check --dir $(if $(DIR),$(DIR),data/rewrite/round3)

keep-round:  ## Copy the current data to round<N>/ and record it in the manifest (ROUND=, optional NOTE=)
	uv run python -m adapt.data.split --platform $(PLATFORM) --keep-round $(ROUND) $(if $(NOTE),--note "$(NOTE)")

validate-gold:  ## Check the gold, gold_v2, gold_v3 and dev_real files: schema, amounts, accounts (PLATFORM=; optional DETAILS=1 shows the problems of gold_v3 in full)
	uv run python -m adapt.data.gold --platform $(PLATFORM) $(if $(DETAILS),--details)

models-check:  ## Check the model gates and compare the pinned revisions with the Hub (OFFLINE=1, PIN=1)
	uv run python -m adapt.models $(if $(OFFLINE),--offline) $(if $(PIN),--pin)

train:  ## QLoRA fine-tuning of one model (MODEL=, SEED=; optional ROUND=, EPOCHS=2, NUM_GPUS=2, SMOKE=1, RESUME=1, MAX_STEPS=, ADAPTERS_DIR=, RUNS_DIR=)
	uv run --group gpu $(if $(filter-out 1,$(NUM_GPUS)),accelerate launch --multi_gpu --num_processes $(NUM_GPUS) -m adapt.train,python -m adapt.train) \
		--platform $(PLATFORM) --seed $(SEED) --num-gpus $(NUM_GPUS) \
		$(if $(SMOKE),--smoke,--model $(MODEL)) $(if $(RESUME),--resume) $(if $(EPOCHS),--epochs $(EPOCHS)) \
		$(if $(MAX_STEPS),--max-steps $(MAX_STEPS)) $(if $(ROUND),--round $(ROUND)) \
		$(if $(ADAPTERS_DIR),--adapters-dir $(ADAPTERS_DIR)) $(if $(RUNS_DIR),--runs-dir $(RUNS_DIR))

select:  ## Score the checkpoints of a run on dev_real and keep the best as its adapter (MODEL=, SEED=; optional ROUND=, EPOCHS=2, STEPS="150 225", SERVERS=2, SMOKE=1, ADAPTERS_DIR=, RUNS_DIR=, OUT_DIR=, DRY_RUN=1)
	uv run python -m adapt.train.selection --platform $(PLATFORM) --seed $(SEED) \
		$(if $(SMOKE),--smoke,--model $(MODEL)) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(STEPS),--steps $(STEPS)) \
		$(if $(SERVERS),--servers $(SERVERS)) $(if $(ROUND),--round $(ROUND)) \
		$(if $(ADAPTERS_DIR),--adapters-dir $(ADAPTERS_DIR)) $(if $(RUNS_DIR),--runs-dir $(RUNS_DIR)) \
		$(if $(OUT_DIR),--out-dir $(OUT_DIR)) $(if $(DRY_RUN),--dry-run)

eval-sft:  ## Evaluate a trained adapter on all splits, with its own server (MODEL=, SEED=; optional ROUND=, EPOCHS=2, SMOKE=1, MAX_STEPS=, NUM_GPUS=, SERVERS=2, SPLITS=, ADAPTERS_DIR=, OUT_DIR=, DRY_RUN=1, THRESHOLD=, OVERRIDE_FINAL_LOCK=1)
	uv run python -m adapt.train.evaluate --platform $(PLATFORM) --seed $(SEED) \
		$(if $(SMOKE),--smoke,--model $(MODEL)) $(if $(EPOCHS),--epochs $(EPOCHS)) $(if $(MAX_STEPS),--max-steps $(MAX_STEPS) --num-gpus $(NUM_GPUS)) \
		$(if $(SERVERS),--servers $(SERVERS)) $(if $(ROUND),--round $(ROUND)) $(if $(SPLITS),--splits $(SPLITS)) $(if $(ADAPTERS_DIR),--adapters-dir $(ADAPTERS_DIR)) \
		$(if $(OUT_DIR),--out-dir $(OUT_DIR)) $(if $(DRY_RUN),--dry-run) $(if $(THRESHOLD),--threshold $(THRESHOLD)) $(if $(OVERRIDE_FINAL_LOCK),--override-final-lock)

eval:  ## Evaluate a system on a split (SYSTEM=, SPLIT=, ENDPOINTS=; optional ROUND=, MODEL=, THRESHOLD=, FEW_SHOT=1, OUT_DIR=, OVERRIDE_FINAL_LOCK=1)
	uv run python -m adapt.eval.runner --platform $(PLATFORM) --split $(SPLIT) --endpoints $(ENDPOINTS) \
		$(if $(SYSTEM),--system $(SYSTEM)) $(if $(MODEL),--model $(MODEL)) \
		$(if $(THRESHOLD),--threshold $(THRESHOLD)) $(if $(WORKERS),--workers $(WORKERS)) \
		$(if $(MODEL_REVISION),--model-revision $(MODEL_REVISION)) \
		$(if $(ADAPTER_PATH),--adapter-path $(ADAPTER_PATH)) $(if $(ROUND),--round $(ROUND)) $(if $(FEW_SHOT),--few-shot) 		$(if $(OUT_DIR),--out-dir $(OUT_DIR)) $(if $(OVERRIDE_FINAL_LOCK),--override-final-lock)

baseline:  ## Few-shot baseline of one model, with its own server (MODEL=; optional ROUND=, SPLITS=, OUT_DIR=, DRY_RUN=1)
	uv run python -m adapt.baseline --platform $(PLATFORM) --model $(MODEL) 		$(if $(ROUND),--round $(ROUND)) $(if $(SPLITS),--splits $(SPLITS)) $(if $(OUT_DIR),--out-dir $(OUT_DIR)) $(if $(DRY_RUN),--dry-run)

pair:  ## Combine the saved results of two systems into one pair configuration (FIRST=, SECOND=, NAME=; optional CONFIDENCE=each|ignore, SPLITS=, ROUND=, OUT_DIR=)
	uv run python -m adapt.eval.configuration --platform $(PLATFORM) --first $(FIRST) --second $(SECOND) --name $(NAME) 		$(if $(CONFIDENCE),--confidence $(CONFIDENCE)) $(if $(SPLITS),--splits $(SPLITS)) $(if $(ROUND),--round $(ROUND)) $(if $(OUT_DIR),--out-dir $(OUT_DIR))

rescore:  ## Score saved results again with the newer pipeline checks, no model run (SYSTEMS="a b"; optional SPLITS=, ROUND=, RETUNE=1, TUNE_ON="dev dev_real")
	uv run python -m adapt.eval.rescore --platform $(PLATFORM) --systems $(SYSTEMS) $(if $(SPLITS),--splits $(SPLITS)) $(if $(ROUND),--round $(ROUND)) $(if $(RETUNE),--retune $(if $(TUNE_ON),--tune-on $(TUNE_ON)))

post-hoc:  ## Re-score the saved outputs of a claim run with later changes, labelled post-hoc, not a claim (SYSTEM=, BASELINE=; optional ROUND=, IN_DIR=, OUT=)
	uv run python -m adapt.eval.post_hoc --platform $(PLATFORM) --system $(SYSTEM) --baseline $(BASELINE) $(if $(ROUND),--round $(ROUND)) $(if $(IN_DIR),--in-dir $(IN_DIR)) $(if $(OUT),--out $(OUT))

post-hoc-gold:  ## Re-score the shipped configuration on gold and gold_v2 from saved outputs, labelled post-hoc, not a claim (SYSTEM=; optional ROUND=, IN_DIR=, OUT=)
	uv run python -m adapt.eval.post_hoc_gold --platform $(PLATFORM) --system $(SYSTEM) $(if $(ROUND),--round $(ROUND)) $(if $(IN_DIR),--in-dir $(IN_DIR)) $(if $(OUT),--out $(OUT))

stage-gate:  ## Gate checks on dev and dev_real of a system, a pass or fail for each, not the claim (SYSTEM=; optional ROUND=, OUT_DIR=, MARGIN=1)
	uv run python -m adapt.eval.stage_gate --platform $(PLATFORM) --system $(SYSTEM) $(if $(ROUND),--round $(ROUND)) $(if $(OUT_DIR),--out-dir $(OUT_DIR)) $(if $(MARGIN),--margin)

threshold-report:  ## Coverage and critical-error curve of a system on dev and dev_real, threshold chosen on dev (SYSTEM=; optional ROUND=, SPLITS=, IN_DIR=, TUNE_ON="dev dev_real")
	uv run python -m adapt.eval.threshold_report --platform $(PLATFORM) --system $(SYSTEM) $(if $(ROUND),--round $(ROUND)) $(if $(SPLITS),--splits $(SPLITS)) $(if $(IN_DIR),--in-dir $(IN_DIR)) $(if $(TUNE_ON),--tune-on $(TUNE_ON))

gate:  ## Check the results of a system against the gate (SYSTEM=, optional BASELINE=, GOLD=gold_v2, ROUND=, OUT_DIR=)
	uv run python -m adapt.eval.gate --platform $(PLATFORM) $(if $(SYSTEM),--system $(SYSTEM)) $(if $(BASELINE),--baseline $(BASELINE)) $(if $(GOLD),--gold $(GOLD)) $(if $(ROUND),--round $(ROUND)) 		$(if $(OUT_DIR),--out-dir $(OUT_DIR))

seal:  ## Record the SHA-256 and the date of a file in the lock, to show it was fixed before later decisions (FILE=)
	uv run python -m adapt.eval.declare_claim --platform $(PLATFORM) --seal $(FILE)

declare-claim:  ## Declare the configurations of the final claim before the final run (CONFIGS="name=system:folder ...", SHIPPED=; optional ROUND=; STATUS=1 shows it)
	uv run python -m adapt.eval.declare_claim --platform $(PLATFORM) $(if $(ROUND),--round $(ROUND)) 		$(if $(STATUS),--status,--config $(CONFIGS) --shipped $(SHIPPED))

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
