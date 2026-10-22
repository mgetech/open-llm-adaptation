"""Serve the shipped model with vLLM from `configs/serve.yaml`.

Run with `make serve-local`. It checks that every adapter has the hash in the file, then starts
vLLM with the base model at its pinned revision and the adapters loaded as LoRA modules, offline,
and the gateway in front of it. Kaggle has no Docker, so this is how the two run there. `--print`
shows the commands and the environment without starting anything, and `--no-gateway` starts vLLM
alone.

`make fetch-weights` downloads the base model once, so that serving needs no network.
"""

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from adapt.baseline import stop
from adapt.eval.runner import ADAPTER_FILES, ROOT, hash_directory
from adapt.gateway.keys import load_keys
from adapt.models import CONFIGS_DIR, Registry, load_registry

SERVE_FILE = CONFIGS_DIR / "serve.yaml"
WEIGHT_FILES = ("*.safetensors", "*.json", "*.txt")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Lora(_Model):
    max_loras: int = Field(ge=1)
    max_lora_rank: int = Field(ge=1)


class Adapter(_Model):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    threshold: float = Field(ge=0, le=1)


class ServeConfig(_Model):
    model: str
    port: int = Field(ge=1, le=65535)
    dtype: str
    max_model_len: int = Field(gt=0)
    gpu_memory_utilization: float = Field(gt=0, le=1)
    prefix_caching: bool
    lora: Lora
    adapters: list[Adapter] = Field(min_length=1)
    environment: dict[str, str]

    @model_validator(mode="after")
    def _adapters_fit_the_slots(self) -> "ServeConfig":
        names = [adapter.name for adapter in self.adapters]
        if len(set(names)) != len(names):
            raise ValueError("an adapter name is used twice")
        if len(names) > self.lora.max_loras:
            raise ValueError(f"{len(names)} adapters do not fit in {self.lora.max_loras} slots")
        return self


def load_serve_config(path: Path = SERVE_FILE) -> ServeConfig:
    try:
        return ServeConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as error:
        raise ValueError(f"{path.name} is not valid: {error}") from error


def adapter_path(adapter: Adapter) -> Path:
    return adapter.path if adapter.path.is_absolute() else ROOT / adapter.path


def check_adapters(config: ServeConfig) -> None:
    """Every adapter must be on disk with the hash in the file."""
    for adapter in config.adapters:
        path = adapter_path(adapter)
        if not (path / "adapter_config.json").is_file():
            raise ValueError(f"no adapter {adapter.name} in {path}")
        found = hash_directory(path, ADAPTER_FILES)
        if found != adapter.sha256:
            raise ValueError(
                f"adapter {adapter.name} in {path} has hash {found}, "
                f"the file expects {adapter.sha256}"
            )


def serve_command(config: ServeConfig, registry: Registry, python: str | None = None) -> list[str]:
    """The vLLM command. It runs with the Python of this process, so it uses the vLLM installed
    next to this project and never another `vllm` on the PATH."""
    if config.model not in registry.models:
        known = ", ".join(registry.models)
        raise ValueError(f"unknown model {config.model!r}, expected one of {known}")
    entry = registry.models[config.model]
    command = [
        python or sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        entry.hf_id,
        "--revision",
        entry.revision,
        "--served-model-name",
        config.model,
        "--dtype",
        config.dtype,
        "--max-model-len",
        str(config.max_model_len),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--seed",
        "0",
        "--port",
        str(config.port),
        "--enable-lora",
        "--max-loras",
        str(config.lora.max_loras),
        "--max-lora-rank",
        str(config.lora.max_lora_rank),
    ]
    if config.prefix_caching:
        command.append("--enable-prefix-caching")
    command += [
        "--lora-modules",
        *(f"{adapter.name}={adapter_path(adapter).resolve()}" for adapter in config.adapters),
    ]
    return command


def gateway_command(file: Path, port: int, python: str | None = None) -> list[str]:
    """The gateway command. It reaches vLLM on the port of the serve file."""
    return [
        python or sys.executable,
        "-m",
        "adapt.gateway",
        "--serve-file",
        str(file),
        "--port",
        str(port),
    ]


def environment(config: ServeConfig, base: Mapping[str, str] | None = None) -> dict[str, str]:
    return {**(os.environ if base is None else base), **config.environment}


def run_together(commands: Sequence[tuple[Sequence[str], Mapping[str, str]]]) -> int:
    """Run the commands side by side until one ends or the user stops them, then stop them all."""
    processes = []
    try:
        for command, env in commands:
            processes.append(subprocess.Popen(list(command), env=dict(env)))
        while True:
            for process in processes:
                code = process.poll()
                if code is not None:
                    return code
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 130
    finally:
        for process in processes:
            stop(process)


def fetch_weights(registry: Registry, key: str) -> Path:
    """Download the base model at its pinned revision into the Hugging Face cache."""
    from huggingface_hub import snapshot_download

    entry = registry.models[key]
    return Path(
        snapshot_download(
            entry.hf_id,
            revision=entry.revision,
            allow_patterns=list(WEIGHT_FILES),
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the shipped model with vLLM.")
    parser.add_argument("--file", type=Path, default=SERVE_FILE, help="default: configs/serve.yaml")
    parser.add_argument("--print", action="store_true", help="show the commands, do not start them")
    parser.add_argument("--no-gateway", action="store_true", help="start vLLM alone")
    parser.add_argument("--gateway-port", type=int, default=8080)
    parser.add_argument("--fetch-weights", action="store_true", help="download the base model")
    args = parser.parse_args(argv)
    try:
        config = load_serve_config(args.file)
        registry = load_registry()
        if args.fetch_weights:
            print(f"base model {config.model}: {fetch_weights(registry, config.model)}")
            return 0
        commands = [(serve_command(config, registry), environment(config))]
        if not args.no_gateway:
            gateway = gateway_command(args.file, args.gateway_port)
            commands.append((gateway, environment(config)))
        if args.print:
            for name, value in config.environment.items():
                print(f"{name}={value}")
            for command, _ in commands:
                print(" ".join(command))
            return 0
        check_adapters(config)
        if not args.no_gateway:
            load_keys()
    except (ValueError, OSError) as error:
        sys.exit(f"error: {error}")
    return run_together(commands)


if __name__ == "__main__":
    sys.exit(main())
