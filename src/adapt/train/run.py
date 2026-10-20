"""Fine-tune one model with QLoRA and write its adapter and a run manifest.

    adapters/<platform>/<model>-<seed>/            the adapter (not committed)
    adapters/<platform>/<model>-<seed>-checkpoints/  checkpoints, for `--resume`
    results/round<N>/runs/<platform>-<model>-<seed>.json    the run manifest

`--round N` trains on the data of round N (default: the current round). A run on a later round
than the first has `-r<N>` after the model, as in `qwen3-4b-r2-0`.

If `dev_real.jsonl` exists, the loss on its documents is measured each time a checkpoint is saved.
The run stops when it has not improved for `early_stopping.patience` checkpoints, and the adapter
is the one with the lowest loss. The documents are never trained on. The manifest has the losses.

`--smoke` trains a tiny random model with a tiny tokenizer on the CPU for two steps. It checks the
code path. It writes under `smoke-<seed>` and its manifest goes next to the adapter, not into
`results/`.

`--max-steps N` stops after N steps, for a short check on the GPU. Its output has
`-steps<N>-gpus<G>` in its name, so it never replaces a full run, and a run on one GPU does not
replace one on two.

`--num-gpus N` is the number of processes, one per GPU. Several processes need the launcher, which
`make train NUM_GPUS=N` runs. Each process holds the whole 4-bit model on its own GPU and trains on
its share of every batch. The gradient accumulation is divided by N, so the effective batch size is
the same on any number of GPUs. Only the first process saves the adapter and writes the manifest.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform as host
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapt.config import load_platform
from adapt.data.rounds import generator_version, resolve_round, round_tag
from adapt.data.rounds import runs_dir as round_runs_dir
from adapt.eval.runner import git_state, gpu_name, hash_directory, hash_file, split_path
from adapt.models import ModelEntry, Registry, check_model, load_registry
from adapt.prompting import system_prompt
from adapt.train.config import TrainConfig, load_train_config
from adapt.train.data import encode_records, load_dev_real_records, load_train_records

ADAPTERS_DIR = Path("adapters")
PACKAGES = ("torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "tokenizers")


@dataclass(frozen=True)
class TrainSettings:
    platform: str
    seed: int
    model: str | None = None  # a key of configs/models.yaml, not needed for a smoke run
    smoke: bool = False
    max_steps: int | None = None
    resume: bool = False
    num_gpus: int = 1  # processes: one per GPU (on the CPU in a smoke run)
    adapters_dir: Path = ADAPTERS_DIR
    runs_dir: Path | None = None  # default: results/round<N>/runs, or the adapter folder (smoke)
    data_round: int | None = None  # None: the current round


def run_name(settings: TrainSettings) -> str:
    if settings.smoke:
        return f"smoke-{settings.seed}"
    name = f"{settings.model}{round_tag(resolve_round(settings.data_round))}-{settings.seed}"
    if settings.max_steps:
        return f"{name}-steps{settings.max_steps}-gpus{settings.num_gpus}"
    return name


def process_ranks() -> tuple[int, int]:
    """The rank of this process on its machine and the number of processes, as the launcher sets
    them. A run started without the launcher is one process."""
    return int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def check_processes(num_gpus: int) -> None:
    """The launcher must have started as many processes as the run was told to expect."""
    if num_gpus < 1:
        raise ValueError(f"the number of GPUs must be at least 1, not {num_gpus}")
    _, world_size = process_ranks()
    if world_size != num_gpus:
        raise ValueError(
            f"{world_size} processes were started for {num_gpus} GPUs. "
            "Start the run with `make train NUM_GPUS=...`"
        )


def accumulation_steps(total: int, num_gpus: int) -> int:
    """The accumulation steps of each process. The batch size of a step must not change with the
    number of GPUs, because the learning rate and the number of steps depend on it."""
    if total % num_gpus:
        raise ValueError(
            f"{total} accumulation steps cannot be shared by {num_gpus} GPUs "
            "without changing the effective batch size"
        )
    return total // num_gpus


def trainable_entry(registry: Registry, key: str) -> ModelEntry:
    """The entry of a model that may be fine-tuned. A model that fails a gate is refused."""
    if key not in registry.models:
        raise ValueError(f"unknown model {key!r}, expected one of {', '.join(registry.models)}")
    entry = registry.models[key]
    failed = [r.name for r in check_model(entry, registry.gates) if not r.passed]
    if failed:
        raise ValueError(f"{key} fails the model gates: " + ", ".join(failed))
    return entry


def find_checkpoint(directory: Path) -> Path | None:
    """The checkpoint with the highest step, or None."""
    steps = {
        int(path.name.removeprefix("checkpoint-")): path
        for path in directory.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    }
    return steps[max(steps)] if steps else None


def prepare_output(adapter_dir: Path, checkpoint_dir: Path, *, resume: bool, smoke: bool) -> None:
    """Refuse to mix a new run with the files of an earlier one. A smoke run replaces its own."""
    if resume:
        return
    if smoke:
        shutil.rmtree(adapter_dir, ignore_errors=True)
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    elif adapter_dir.exists() or find_checkpoint(checkpoint_dir):
        raise ValueError(
            f"{adapter_dir} or its checkpoints exist. Use --resume to continue that run, "
            "or remove them to start again"
        )


def selection_summary(
    history: Sequence[Mapping[str, Any]],
    best_checkpoint: str | None,
    best_loss: float | None,
    steps: int,
    planned_steps: int,
) -> dict[str, Any]:
    """What the dev_real loss did during a run, for the manifest."""
    best_step = None
    if best_checkpoint:
        best_step = int(Path(best_checkpoint).name.removeprefix("checkpoint-"))
    return {
        "eval_losses": [
            {"step": entry["step"], "loss": entry["eval_loss"]}
            for entry in history
            if "eval_loss" in entry
        ],
        "best_step": best_step,
        "best_loss": best_loss,
        "stopped_early": steps < planned_steps,
    }


def lora_to_float32(model: Any) -> None:
    """TRL converts the LoRA weights of a quantised model to bf16. A T4 has no bf16, and fp16
    training needs fp32 weights, so they go back to fp32 before the optimiser sees them."""
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()


def load_real(entry: ModelEntry, config: TrainConfig, num_gpus: int = 1) -> tuple[Any, Any]:
    """The 4-bit base model on the GPU of this process and its tokenizer, both at the pinned
    revision.

    The run uses the first `num_gpus` GPUs, as the memory check assumes. Kaggle shows two T4 to a
    notebook, and a run on one GPU must not use both. A choice that is already set is kept."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(num_gpus)))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("no GPU found. Use --smoke to check the code on the CPU")
    if torch.cuda.device_count() < num_gpus:
        raise RuntimeError(f"{num_gpus} GPUs were asked for, {torch.cuda.device_count()} found")
    local_rank, _ = process_ranks()
    torch.cuda.set_device(local_rank)
    quantisation = config.quantisation
    tokenizer = AutoTokenizer.from_pretrained(entry.hf_id, revision=entry.revision)
    model = AutoModelForCausalLM.from_pretrained(
        entry.hf_id,
        revision=entry.revision,
        dtype=torch.float16,
        device_map={"": local_rank},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quantisation.quant_type,
            bnb_4bit_use_double_quant=quantisation.double_quant,
            bnb_4bit_compute_dtype=getattr(torch, quantisation.compute_dtype),
        ),
    )
    return model, tokenizer


def load_smoke(
    config: TrainConfig, records: Sequence[dict[str, Any]], seed: int
) -> tuple[Any, Any]:
    from adapt.train.smoke import tiny_model, tiny_tokenizer

    smoke = config.smoke
    platform = load_platform(records[0]["platform"])
    texts = [
        r["document_text"] + "\n" + json.dumps(r["target"])
        for r in records[: smoke.tokenizer_examples]
    ]
    texts.append(system_prompt(platform.config.name, platform.rules.chart, platform.rules))
    tokenizer = tiny_tokenizer(texts, smoke.vocab_size)
    return tiny_model(tokenizer, smoke, seed), tokenizer


def versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {"python": host.python_version()}
    for name in PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def run(
    settings: TrainSettings,
    config: TrainConfig | None = None,
    registry: Registry | None = None,
) -> dict[str, Any] | None:
    """Train, save the adapter, write the manifest, and return the manifest. With several
    processes, only the first one saves and returns it. The others return None."""
    check_processes(settings.num_gpus)
    config = config or load_train_config()
    platform = load_platform(settings.platform)
    entry = None
    if not settings.smoke:
        if not settings.model:
            raise ValueError("a model is needed unless this is a smoke run")
        entry = trainable_entry(registry or load_registry(), settings.model)

    name = run_name(settings)
    run_id = f"{settings.platform}-{name}"
    adapter_dir = settings.adapters_dir / settings.platform / name
    checkpoint_dir = settings.adapters_dir / settings.platform / f"{name}-checkpoints"
    data_round = resolve_round(settings.data_round)
    runs_dir = settings.runs_dir or (
        adapter_dir.parent if settings.smoke else round_runs_dir(data_round)
    )
    prepare_output(adapter_dir, checkpoint_dir, resume=settings.resume, smoke=settings.smoke)

    records = load_train_records(settings.platform, data_round=data_round)
    if settings.smoke:
        records = records[: max(config.smoke.examples, config.smoke.tokenizer_examples)]
    model, tokenizer = (
        load_smoke(config, records, settings.seed)
        if settings.smoke
        else load_real(entry, config, settings.num_gpus)
    )
    smoke = config.smoke
    max_length = smoke.max_length if settings.smoke else config.data.max_length
    examples = encode_records(
        tokenizer,
        settings.platform,
        platform.rules,
        records[: smoke.examples] if settings.smoke else records,
        enable_thinking=entry.chat_template.enable_thinking if entry else False,
        max_length=max_length,
    )

    selection = load_dev_real_records(settings.platform)
    if settings.smoke:
        selection = selection[: smoke.eval_examples]
    selection_examples = encode_records(
        tokenizer,
        settings.platform,
        platform.rules,
        selection,
        enable_thinking=entry.chat_template.enable_thinking if entry else False,
        max_length=max_length,
    )

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    lora = config.lora
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    optimiser = config.optimiser
    max_steps = settings.max_steps or (smoke.steps if settings.smoke else -1)
    per_device_batch = smoke.batch_size if settings.smoke else optimiser.batch_size
    accumulation = (
        1
        if settings.smoke
        else accumulation_steps(optimiser.gradient_accumulation_steps, settings.num_gpus)
    )
    effective_batch = per_device_batch * accumulation * settings.num_gpus
    save_steps = smoke.save_steps if settings.smoke else config.checkpoint.save_steps
    stopping = config.early_stopping
    # The loss is measured when a checkpoint is saved, so that the best one can be kept.
    selecting: dict[str, Any] = {}
    if selection_examples:
        selecting = {
            "eval_strategy": "steps",
            "eval_steps": save_steps,
            "per_device_eval_batch_size": per_device_batch,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }
    callbacks = []
    if selection_examples:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=stopping.patience,
                early_stopping_threshold=stopping.min_delta,
            )
        )
    arguments = SFTConfig(
        output_dir=str(checkpoint_dir),
        seed=settings.seed,
        data_seed=settings.seed,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=accumulation,
        learning_rate=optimiser.learning_rate,
        lr_scheduler_type=optimiser.lr_scheduler,
        warmup_steps=optimiser.warmup_ratio,
        num_train_epochs=optimiser.epochs,
        max_steps=max_steps,
        optim="adamw_torch" if settings.smoke else optimiser.optim,
        fp16=not settings.smoke,
        use_cpu=settings.smoke,
        gradient_checkpointing=False if settings.smoke else optimiser.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=config.checkpoint.keep,
        logging_steps=config.logging_steps,
        report_to="none",
        max_length=max_length,
        packing=False,
        **selecting,
    )
    trainer = SFTTrainer(
        model=model,
        args=arguments,
        train_dataset=Dataset.from_list(examples),
        eval_dataset=Dataset.from_list(selection_examples) if selection_examples else None,
        callbacks=callbacks,
        processing_class=tokenizer,
    )
    lora_to_float32(trainer.model)

    checkpoint = find_checkpoint(checkpoint_dir) if settings.resume else None
    resumed_from = int(checkpoint.name.removeprefix("checkpoint-")) if checkpoint else None
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    wall_seconds = time.perf_counter() - started
    if not trainer.is_world_process_zero():
        return None
    trainer.model.save_pretrained(adapter_dir)

    import torch

    gpus = torch.cuda.device_count() if torch.cuda.is_available() and not settings.smoke else 0
    gpus_used = 0 if settings.smoke else settings.num_gpus
    peak_memory = None if settings.smoke else torch.cuda.max_memory_reserved() / 2**30
    commit, dirty = git_state()
    train_file = split_path(settings.platform, "train", data_round=data_round)
    selection_file = split_path(settings.platform, "dev_real")
    state = trainer.state
    manifest = {
        "run_id": run_id,
        "platform": settings.platform,
        "model": None if settings.smoke else settings.model,
        "model_hf_id": entry.hf_id if entry else None,
        "model_revision": entry.revision if entry else None,
        "seed": settings.seed,
        "smoke": settings.smoke,
        "max_steps": settings.max_steps,
        "config": config.model_dump(),
        "prompt_sha256": hashlib.sha256(
            system_prompt(settings.platform, platform.rules.chart, platform.rules).encode()
        ).hexdigest(),
        "data": {
            "round": data_round,
            "generator": generator_version(data_round),
            "train_file": train_file.name,
            "train_sha256": hash_file(train_file),
            "examples": len(examples),
            "longest_example_tokens": max(len(e["input_ids"]) for e in examples),
            "max_length": max_length,
        },
        "selection": {
            "split": "dev_real",
            "file": selection_file.name,
            "sha256": hash_file(selection_file),
            "examples": len(selection_examples),
            "eval_steps": save_steps,
            **selection_summary(
                state.log_history,
                state.best_model_checkpoint,
                state.best_metric,
                result.global_step,
                state.max_steps,
            ),
        }
        if selection_examples
        else None,
        "git_commit": commit,
        "git_dirty": dirty,
        "gpu": gpu_name(),
        "gpu_count": gpus,
        "num_gpus": settings.num_gpus,
        "per_device_batch_size": per_device_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch,
        "peak_gpu_memory_gib": peak_memory,
        "versions": versions(),
        "steps": result.global_step,
        "resumed_from_step": resumed_from,
        "train_loss": result.training_loss,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "wall_seconds": wall_seconds,
        "gpu_hours": wall_seconds * gpus_used / 3600,
        "adapter_dir": adapter_dir.as_posix(),
        "adapter_sha256": hash_directory(adapter_dir),
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run_id}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    manifest["manifest_file"] = path.as_posix()
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune one model with QLoRA.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--model", help="key in configs/models.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="a tiny random model on the CPU")
    parser.add_argument("--max-steps", type=int, help="stop after this many steps")
    parser.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="processes started by the launcher, one per GPU"
    )
    parser.add_argument("--adapters-dir", type=Path, default=ADAPTERS_DIR)
    parser.add_argument("--runs-dir", type=Path, help="default: results/round<N>/runs")
    parser.add_argument("--round", type=int, help="round of the data (default: the current one)")
    args = parser.parse_args(argv)
    if not args.smoke and not args.model:
        parser.error("--model is required unless --smoke is given")

    settings = TrainSettings(
        platform=args.platform,
        seed=args.seed,
        model=args.model,
        smoke=args.smoke,
        max_steps=args.max_steps,
        resume=args.resume,
        num_gpus=args.num_gpus,
        adapters_dir=args.adapters_dir,
        runs_dir=args.runs_dir,
        data_round=args.round,
    )
    try:
        manifest = run(settings)
    except (RuntimeError, ValueError) as error:
        sys.exit(f"error: {error}")
    if manifest is None:  # not the first process
        return 0
    print(
        f"{manifest['run_id']}: {manifest['steps']} steps, num_gpus {manifest['num_gpus']}, "
        f"loss {manifest['train_loss']:.4f}, {manifest['wall_seconds']:.0f} seconds"
    )
    print(f"adapter: {manifest['adapter_dir']}")
    print(f"manifest: {manifest['manifest_file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
