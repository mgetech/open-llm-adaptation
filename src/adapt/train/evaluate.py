"""Evaluate a trained adapter on the splits, with its own vLLM server.

The server loads the base model in fp16 and the adapter as a LoRA module. The requests ask for
the adapter by its system name, with the zero-shot prompt it was trained on. The results go to
`<out-dir>/<platform>/<system>/<split>.json`, where `make gate` reads them.

`--dry-run` starts the fake server from the tests instead of vLLM. `--smoke` selects the adapter
of a smoke run, which is random, so it can only be used with `--dry-run`. Results from either
come from a fake model and are never real results.
"""

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from adapt.baseline import (
    START_TIMEOUT_SECONDS,
    fake_command,
    free_port,
    serve_command,
    stop,
    wait_until_ready,
)
from adapt.eval.runner import (
    RESULTS_DIR,
    SPLITS,
    RunSettings,
    evaluate,
    results_file,
    server_version,
    write_results,
)
from adapt.models import Registry, load_registry
from adapt.train.config import load_train_config
from adapt.train.run import ADAPTERS_DIR, TrainSettings, run_name, trainable_entry


def system_name(
    key: str | None, seed: int, *, smoke: bool = False, max_steps: int | None = None
) -> str:
    """The name of the system in the results: `qwen3-4b-sft-0`, or `-steps3` after a short check."""
    if smoke:
        return f"smoke-sft-{seed}"
    name = f"{key}-sft-{seed}"
    return f"{name}-steps{max_steps}" if max_steps else name


def lora_command(
    registry: Registry, key: str, port: int, system: str, adapter_dir: Path, rank: int
) -> list[str]:
    """The base model command of the baseline, with the adapter loaded as a LoRA module."""
    return [
        *serve_command(registry, key, port),
        "--enable-lora",
        "--max-lora-rank",
        str(rank),
        "--lora-modules",
        f"{system}={adapter_dir.resolve()}",
    ]


def evaluate_adapter(
    registry: Registry,
    key: str | None,
    seed: int,
    out_dir: Path,
    *,
    platform: str = "tax-booking",
    adapters_dir: Path = ADAPTERS_DIR,
    max_steps: int | None = None,
    num_gpus: int = 1,
    splits: Sequence[str] = SPLITS,
    smoke: bool = False,
    port: int = 8000,
    dry_run: bool = False,
    workers: int = 8,
    timeout: float = START_TIMEOUT_SECONDS,
) -> list[Path]:
    """Evaluate the adapter of a run on `splits`. Returns the results files."""
    if smoke and not dry_run:
        raise ValueError("the adapter of a smoke run is random, evaluate it with --dry-run")
    unknown = [split for split in splits if split not in SPLITS]
    if unknown:
        raise ValueError(f"unknown split {unknown[0]!r}, expected one of {', '.join(SPLITS)}")
    entry = None if smoke else trainable_entry(registry, key or "")

    train = TrainSettings(
        platform, seed, model=key, smoke=smoke, max_steps=max_steps, num_gpus=num_gpus
    )
    adapter_dir = adapters_dir / platform / run_name(train)
    if not (adapter_dir / "adapter_config.json").exists():
        raise ValueError(f"no adapter in {adapter_dir}, train it first")
    system = system_name(key, seed, smoke=smoke, max_steps=max_steps)

    if dry_run:
        command = fake_command(port)
    else:
        rank = load_train_config().lora.r
        command = lora_command(registry, key or "", port, system, adapter_dir, rank)
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(command)
    written = []
    try:
        wait_until_ready(base_url, process, timeout)
        endpoint = f"{base_url}/v1"
        version = server_version(endpoint)
        pinned = registry.gates.vllm.version
        if not dry_run and version != pinned:
            raise ValueError(f"the server runs vllm {version}, the registry pins {pinned}")
        for split in splits:
            settings = RunSettings(
                platform,
                system,
                split,
                endpoint,
                system,
                workers=workers,
                model_revision=entry.revision if entry else None,
                adapter_path=adapter_dir,
                out_dir=out_dir,
            )
            results = evaluate(settings)
            path = results_file(out_dir, platform, system, split)
            write_results(results, path)
            written.append(path)
            print(f"{system} on {split}: {len(results['documents'])} documents, results in {path}")
    finally:
        stop(process)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained adapter on the splits.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--model", help="key in configs/models.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="the adapter of a smoke run")
    parser.add_argument("--max-steps", type=int, help="the adapter of a run with this limit")
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="GPUs of that run (only its folder name has it)"
    )
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), help="default: all splits")
    parser.add_argument("--adapters-dir", type=Path, default=ADAPTERS_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--workers", type=int, default=8, help="requests in flight")
    parser.add_argument(
        "--port", type=int, help="port of the server (default: 8000, or a free one)"
    )
    parser.add_argument("--dry-run", action="store_true", help="use the fake server, not vLLM")
    args = parser.parse_args(argv)
    if not args.smoke and not args.model:
        parser.error("--model is required unless --smoke is given")

    port = args.port or (free_port() if args.dry_run else 8000)
    try:
        evaluate_adapter(
            load_registry(),
            args.model,
            args.seed,
            args.out_dir,
            platform=args.platform,
            adapters_dir=args.adapters_dir,
            max_steps=args.max_steps,
            num_gpus=args.num_gpus,
            splits=args.splits,
            smoke=args.smoke,
            port=port,
            dry_run=args.dry_run,
            workers=args.workers,
        )
    except (RuntimeError, ValueError) as error:
        sys.exit(f"error: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
