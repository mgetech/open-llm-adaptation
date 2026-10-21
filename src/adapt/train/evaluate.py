"""Evaluate a trained adapter on the splits, with its own vLLM server.

The server loads the base model in fp16 and the adapter as a LoRA module. The requests ask for
the adapter by its system name, with the zero-shot prompt it was trained on. The results go to
`<out-dir>/<platform>/<system>/<split>.json`, where `make gate` reads them.

`--servers N` starts N servers, one per GPU (`CUDA_VISIBLE_DEVICES` 0 to N-1, on consecutive
ports), each with the same base model and adapter. The documents of a split are shared between
them and the results are merged in document order.

Without `--splits`, every split is evaluated, and gold_v2 and dev_real too once their files
exist.

`--round N` is the round of the data (default: the current round). The adapter and the system
name carry it, as in `qwen3-4b-sft-r2-0`, and the splits are the ones of that round.

`--dry-run` starts the fake server from the tests instead of vLLM. `--smoke` selects the adapter
of a smoke run, which is random, so it can only be used with `--dry-run`. Results from either
come from a fake model and are never real results.
"""

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from adapt.baseline import (
    START_TIMEOUT_SECONDS,
    fake_command,
    free_port,
    serve_command,
    stop,
    wait_until_ready,
)
from adapt.data.rounds import eval_dir, resolve_round, round_tag
from adapt.eval.runner import (
    ALL_SPLITS,
    RunSettings,
    claim_final_split,
    default_splits,
    evaluate,
    results_file,
    server_version,
    write_results,
)
from adapt.models import Registry, load_registry
from adapt.train.config import load_train_config
from adapt.train.run import ADAPTERS_DIR, TrainSettings, run_name, trainable_entry


def system_name(
    key: str | None,
    seed: int,
    *,
    smoke: bool = False,
    max_steps: int | None = None,
    data_round: int = 1,
) -> str:
    """The name of the system in the results: `qwen3-4b-sft-0`, `qwen3-4b-sft-r2-0` for round 2,
    and `-steps3` after a short check."""
    if smoke:
        return f"smoke-sft-{seed}"
    name = f"{key}-sft{round_tag(data_round)}-{seed}"
    return f"{name}-steps{max_steps}" if max_steps else name


def lora_modules_command(
    registry: Registry, key: str, port: int, modules: Mapping[str, Path], rank: int
) -> list[str]:
    """The base model command of the baseline, with the adapters loaded as LoRA modules. A server
    that holds several adapters has a slot for each, so a request for any of them runs at once."""
    command = [*serve_command(registry, key, port), "--enable-lora", "--max-lora-rank", str(rank)]
    if len(modules) > 1:
        command += ["--max-loras", str(len(modules))]
    return [
        *command,
        "--lora-modules",
        *(f"{name}={path.resolve()}" for name, path in modules.items()),
    ]


def lora_command(
    registry: Registry, key: str, port: int, system: str, adapter_dir: Path, rank: int
) -> list[str]:
    """The base model command of the baseline, with the adapter loaded as a LoRA module."""
    return lora_modules_command(registry, key, port, {system: adapter_dir}, rank)


def evaluate_adapters(
    registry: Registry,
    key: str | None,
    adapters: Mapping[str, Path],
    out_dir: Path,
    *,
    platform: str = "tax-booking",
    splits: Sequence[str],
    smoke: bool = False,
    port: int = 8000,
    servers: int = 1,
    dry_run: bool = False,
    workers: int = 8,
    timeout: float = START_TIMEOUT_SECONDS,
    data_round: int | None = None,
    threshold: float | None = None,
    override_final_lock: bool = False,
) -> dict[str, list[Path]]:
    """Serve the base model with every adapter in `adapters` (system name to folder) and
    evaluate each system on `splits`. Returns the results files of each system.

    `servers` servers are started, on `port` and the ports after it. A dry run takes a free
    port for each of the others, so that it never meets a port that is in use. A `threshold` is
    used as given, and without one it is tuned on dev, as for any system.

    A run on test_ood or a gold set makes the final claim on those files, or is refused when
    another system holds it (see `adapt.eval.final_lock`). The claim is made before any server
    starts."""
    if smoke and not dry_run:
        raise ValueError("the adapter of a smoke run is random, evaluate it with --dry-run")
    if servers < 1:
        raise ValueError(f"the number of servers must be at least 1, not {servers}")
    unknown = [split for split in splits if split not in ALL_SPLITS]
    if unknown:
        raise ValueError(f"unknown split {unknown[0]!r}, expected one of {', '.join(ALL_SPLITS)}")
    entry = None if smoke else trainable_entry(registry, key or "")
    for system, adapter_dir in adapters.items():
        if not (adapter_dir / "adapter_config.json").exists():
            raise ValueError(f"no adapter in {adapter_dir}, train it first ({system})")

    data_round = resolve_round(data_round)
    for system, adapter_dir in adapters.items():
        for split in splits:
            claim_final_split(
                platform,
                system,
                split,
                out_dir,
                adapter_path=adapter_dir,
                data_round=data_round,
                override=override_final_lock,
            )
    if dry_run:
        ports = [port, *(free_port() for _ in range(servers - 1))]
    else:
        ports = [port + index for index in range(servers)]
    rank = load_train_config().lora.r
    processes: list[subprocess.Popen] = []
    written: dict[str, list[Path]] = {system: [] for system in adapters}
    try:
        for index, server_port in enumerate(ports):
            if dry_run:
                command = fake_command(server_port, data_round)
            else:
                command = lora_modules_command(registry, key or "", server_port, adapters, rank)
            # With several servers, each one has a GPU of its own.
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(index)} if servers > 1 else None
            processes.append(subprocess.Popen(command, env=env))
        endpoints = []
        for process, server_port in zip(processes, ports, strict=True):
            base_url = f"http://127.0.0.1:{server_port}"
            wait_until_ready(base_url, process, timeout)
            endpoint = f"{base_url}/v1"
            version = server_version(endpoint)
            pinned = registry.gates.vllm.version
            if not dry_run and version != pinned:
                raise ValueError(f"the server runs vllm {version}, the registry pins {pinned}")
            endpoints.append(endpoint)
        for system, adapter_dir in adapters.items():
            for split in splits:
                settings = RunSettings(
                    platform,
                    system,
                    split,
                    tuple(endpoints),
                    system,
                    threshold=threshold,
                    workers=workers,
                    model_revision=entry.revision if entry else None,
                    adapter_path=adapter_dir,
                    out_dir=out_dir,
                    data_round=data_round,
                )
                results = evaluate(settings)
                path = results_file(out_dir, platform, system, split)
                write_results(results, path)
                written[system].append(path)
                print(
                    f"{system} on {split}: {len(results['documents'])} documents, results in {path}"
                )
    finally:
        for process in processes:
            stop(process)
    return written


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
    splits: Sequence[str] | None = None,
    smoke: bool = False,
    port: int = 8000,
    servers: int = 1,
    dry_run: bool = False,
    workers: int = 8,
    timeout: float = START_TIMEOUT_SECONDS,
    data_round: int | None = None,
    override_final_lock: bool = False,
) -> list[Path]:
    """Evaluate the adapter of a run on `splits`. Returns the results files."""
    splits = default_splits(platform) if splits is None else splits
    data_round = resolve_round(data_round)
    train = TrainSettings(
        platform,
        seed,
        model=key,
        smoke=smoke,
        max_steps=max_steps,
        num_gpus=num_gpus,
        data_round=data_round,
    )
    adapter_dir = adapters_dir / platform / run_name(train)
    system = system_name(key, seed, smoke=smoke, max_steps=max_steps, data_round=data_round)
    written = evaluate_adapters(
        registry,
        key,
        {system: adapter_dir},
        out_dir,
        platform=platform,
        splits=splits,
        smoke=smoke,
        port=port,
        servers=servers,
        dry_run=dry_run,
        workers=workers,
        timeout=timeout,
        data_round=data_round,
        override_final_lock=override_final_lock,
    )
    return written[system]


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
    parser.add_argument(
        "--splits",
        nargs="+",
        help="default: all splits, and gold_v2 and dev_real if their files exist",
    )
    parser.add_argument("--adapters-dir", type=Path, default=ADAPTERS_DIR)
    parser.add_argument("--out-dir", type=Path, help="default: results/round<N>/eval")
    parser.add_argument("--workers", type=int, default=8, help="requests in flight")
    parser.add_argument(
        "--port", type=int, help="port of the server (default: 8000, or a free one)"
    )
    parser.add_argument("--servers", type=int, default=1, help="servers, one per GPU")
    parser.add_argument("--round", type=int, help="round of the data (default: the current one)")
    parser.add_argument("--dry-run", action="store_true", help="use the fake server, not vLLM")
    parser.add_argument(
        "--override-final-lock",
        action="store_true",
        help="run on test_ood or gold files that already back a final claim",
    )
    args = parser.parse_args(argv)
    if not args.smoke and not args.model:
        parser.error("--model is required unless --smoke is given")

    port = args.port or (free_port() if args.dry_run else 8000)
    try:
        evaluate_adapter(
            load_registry(),
            args.model,
            args.seed,
            args.out_dir or eval_dir(args.round),
            platform=args.platform,
            adapters_dir=args.adapters_dir,
            max_steps=args.max_steps,
            num_gpus=args.num_gpus,
            splits=args.splits,
            smoke=args.smoke,
            port=port,
            servers=args.servers,
            dry_run=args.dry_run,
            workers=args.workers,
            data_round=args.round,
            override_final_lock=args.override_final_lock,
        )
    except (RuntimeError, ValueError) as error:
        sys.exit(f"error: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
