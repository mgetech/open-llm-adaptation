"""Serve the shipped model with vLLM from `configs/serve.yaml`.

Run with `make serve-local`. It checks that every adapter has the hash in the file, then starts
vLLM with the base model at its pinned revision and the adapters loaded as LoRA modules, offline,
and the gateway in front of it. Kaggle has no Docker, so this is how the two run there. `--print`
shows the commands and the environment without starting anything, and `--no-gateway` starts vLLM
alone.

`make fetch-weights` downloads the base model once, so that serving needs no network. With
`WEIGHTS_DIR=weights` it downloads into the folder the containers mount. `make fetch-adapters`
downloads the adapters from the Hub at the revision in the file and checks their hash. The repo is
gated, so it needs a token: `HF_TOKEN` in the environment or in the `.env` file. `make compose-env`
writes `deploy/vllm.env`, the arguments and environment of the vLLM container, from the same file.
"""

import argparse
import os
import shlex
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


class Hub(_Model):
    repo: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class Adapter(_Model):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    threshold: float = Field(ge=0, le=1)
    hub: Hub | None = None


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


def adapter_target(adapter: Adapter, adapters_dir: Path | None = None) -> Path:
    """Where `fetch_adapters` puts an adapter: its path in the file, or the same path below
    `adapters_dir` in place of the `adapters` folder."""
    if adapters_dir is None:
        return adapter_path(adapter)
    if adapter.path.is_absolute() or adapter.path.parts[0] != "adapters":
        raise ValueError(f"adapter {adapter.name} is not in the adapters folder: {adapter.path}")
    return adapters_dir.joinpath(*adapter.path.parts[1:])


def _hash_if_there(path: Path) -> str | None:
    if not all((path / name).is_file() for name in ADAPTER_FILES):
        return None
    return hash_directory(path, ADAPTER_FILES)


def hub_token(environ: Mapping[str, str] | None = None, file: Path = ROOT / ".env") -> str | None:
    """The token for the Hub: `HF_TOKEN` of the environment, or else the line `HF_TOKEN=...` of
    the `.env` file. Without either, the login of the Hub library applies."""
    token = (os.environ if environ is None else environ).get("HF_TOKEN", "").strip()
    if token:
        return token
    if file.is_file():
        for line in file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "HF_TOKEN" and value.strip().strip("'\""):
                return value.strip().strip("'\"")
    return None


def fetch_adapters(
    config: ServeConfig, adapters_dir: Path | None = None, token: str | None = None
) -> list[Path]:
    """Download each adapter from the Hub at the revision in the file and check its hash. An
    adapter that is already there with the right hash is not downloaded again."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    folders = []
    for adapter in config.adapters:
        if adapter.hub is None:
            raise ValueError(f"adapter {adapter.name} has no hub entry in the serve file")
        target = adapter_target(adapter, adapters_dir)
        if _hash_if_there(target) != adapter.sha256:
            try:
                snapshot_download(
                    adapter.hub.repo,
                    revision=adapter.hub.revision,
                    allow_patterns=list(ADAPTER_FILES),
                    local_dir=target,
                    token=token,
                )
            except HfHubHTTPError as error:
                raise ValueError(
                    f"cannot download {adapter.hub.repo} at {adapter.hub.revision}: {error}. "
                    "The repo is gated: accept its terms with the account of the token, and "
                    "set HF_TOKEN or put it in the .env file."
                ) from error
            found = _hash_if_there(target)
            if found != adapter.sha256:
                raise ValueError(
                    f"adapter {adapter.name} from {adapter.hub.repo} has hash {found}, "
                    f"the file expects {adapter.sha256}; check the files in {target}"
                )
        folders.append(target)
    return folders


def vllm_arguments(
    config: ServeConfig, registry: Registry, adapter_paths: Mapping[str, str]
) -> list[str]:
    """What follows `vllm serve`: the model at its pinned revision, the flags of the file, and the
    adapters at `adapter_paths` (name to the folder as the server sees it)."""
    if config.model not in registry.models:
        known = ", ".join(registry.models)
        raise ValueError(f"unknown model {config.model!r}, expected one of {known}")
    entry = registry.models[config.model]
    arguments = [
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
        arguments.append("--enable-prefix-caching")
    return [
        *arguments,
        "--lora-modules",
        *(f"{adapter.name}={adapter_paths[adapter.name]}" for adapter in config.adapters),
    ]


def serve_command(config: ServeConfig, registry: Registry, python: str | None = None) -> list[str]:
    """The vLLM command. It runs with the Python of this process, so it uses the vLLM installed
    next to this project and never another `vllm` on the PATH."""
    paths = {adapter.name: str(adapter_path(adapter).resolve()) for adapter in config.adapters}
    return [
        python or sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        *vllm_arguments(config, registry, paths),
    ]


def container_paths(config: ServeConfig) -> dict[str, str]:
    """Where each adapter is inside the vLLM container: the folder of the file, at the root."""
    return {adapter.name: "/" + adapter.path.as_posix().lstrip("/") for adapter in config.adapters}


def container_environment(config: ServeConfig, registry: Registry) -> str:
    """The text of `deploy/vllm.env`: the arguments of the vLLM image and its environment, from the
    same file as the local launcher uses."""
    arguments = [*vllm_arguments(config, registry, container_paths(config)), "--host", "0.0.0.0"]
    lines = [
        "# Written by `make compose-env` from configs/serve.yaml. Do not edit it.",
        f"VLLM_ARGS={shlex.join(arguments)}",
        *(f"{name}={value}" for name, value in config.environment.items()),
    ]
    return "\n".join(lines) + "\n"


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


def fetch_weights(registry: Registry, key: str, weights_dir: Path | None = None) -> Path:
    """Download the base model at its pinned revision into the Hugging Face cache, or into the
    `hub` folder of `weights_dir`, which is the cache folder a container mounts."""
    from huggingface_hub import snapshot_download

    entry = registry.models[key]
    return Path(
        snapshot_download(
            entry.hf_id,
            revision=entry.revision,
            allow_patterns=list(WEIGHT_FILES),
            cache_dir=None if weights_dir is None else weights_dir / "hub",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the shipped model with vLLM.")
    parser.add_argument("--file", type=Path, default=SERVE_FILE, help="default: configs/serve.yaml")
    parser.add_argument("--print", action="store_true", help="show the commands, do not start them")
    parser.add_argument("--no-gateway", action="store_true", help="start vLLM alone")
    parser.add_argument("--gateway-port", type=int, default=8080)
    parser.add_argument("--fetch-weights", action="store_true", help="download the base model")
    parser.add_argument("--weights-dir", type=Path, help="download into this folder's hub folder")
    parser.add_argument("--fetch-adapters", action="store_true", help="download the adapters")
    parser.add_argument("--adapters-dir", type=Path, help="download into this folder, not adapters")
    parser.add_argument("--compose-env", type=Path, help="write the vLLM container's env file")
    args = parser.parse_args(argv)
    try:
        config = load_serve_config(args.file)
        registry = load_registry()
        if args.fetch_weights:
            weights = fetch_weights(registry, config.model, args.weights_dir)
            print(f"base model {config.model}: {weights}")
            return 0
        if args.fetch_adapters:
            folders = fetch_adapters(config, args.adapters_dir, hub_token())
            for adapter, folder in zip(config.adapters, folders, strict=True):
                print(f"adapter {adapter.name}: {folder} (hash {adapter.sha256[:12]}... matches)")
            return 0
        if args.compose_env:
            text = container_environment(config, registry)
            args.compose_env.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {args.compose_env}")
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
