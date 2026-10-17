"""The model registry, and the checks a model must pass before I spend any GPU time on it.

Run with `make models-check`. The gates that need only the file are always checked. Unless
`--offline` is given, the file is also compared with the Hugging Face Hub: the pinned revision
must exist, and its licence and parameter count must match. `--pin` replaces a revision that is
a branch or tag name, such as `main`, with the commit SHA it points to.
"""

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
MODELS_FILE = CONFIGS_DIR / "models.yaml"
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChatTemplate(_Model):
    enable_thinking: bool


class ModelEntry(_Model):
    hf_id: str
    revision: str
    licence: str
    parameters: int = Field(gt=0)
    architecture: str
    role: Literal["candidate", "reference"]
    chat_template: ChatTemplate


class QloraGate(_Model):
    gpu_memory_gb: float = Field(gt=0)
    usable_fraction: float = Field(gt=0, le=1)
    sequence_length: int = Field(gt=0)
    bytes_per_weight: float = Field(gt=0)
    trainable_fraction: float = Field(gt=0, le=1)
    bytes_per_trainable: float = Field(gt=0)
    activation_gb_per_1k_tokens: float = Field(ge=0)


class VllmGate(_Model):
    version: str
    lora_architectures: list[str] = Field(min_length=1)


class Gates(_Model):
    licences: list[str] = Field(min_length=1)
    qlora: QloraGate
    vllm: VllmGate


class Registry(_Model):
    gates: Gates
    models: dict[str, ModelEntry] = Field(min_length=1)


@dataclass(frozen=True)
class HubInfo:
    """What the Hub says about one revision of a model. None if it does not say."""

    sha: str
    licence: str | None
    parameters: int | None


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


Hub = Callable[[str, str], HubInfo]


def load_registry(path: Path = MODELS_FILE) -> Registry:
    with path.open(encoding="utf-8") as file:
        return Registry.model_validate(yaml.safe_load(file))


def hub_info(hf_id: str, revision: str) -> HubInfo:
    from huggingface_hub import HfApi  # only the online check needs it

    info = HfApi().model_info(hf_id, revision=revision)
    licence = info.card_data.license if info.card_data else None
    parameters = info.safetensors.total if info.safetensors else None
    return HubInfo(info.sha, licence, parameters)


def training_memory_gb(parameters: int, gate: QloraGate) -> float:
    """Estimated GPU memory of QLoRA training: weights, trainable states and activations."""
    weights = parameters * gate.bytes_per_weight
    trainable = parameters * gate.trainable_fraction * gate.bytes_per_trainable
    activations = gate.activation_gb_per_1k_tokens * 1e9 * gate.sequence_length / 1024
    return (weights + trainable + activations) / 1e9


def check_model(entry: ModelEntry, gates: Gates) -> list[GateResult]:
    """The gates that need only the registry. Training and serving apply to candidates."""
    allowed = [licence.lower() for licence in gates.licences]
    results = [
        GateResult(
            "licence",
            entry.licence.lower() in allowed,
            f"{entry.licence}, allowed: {', '.join(gates.licences)}",
        ),
        GateResult(
            "revision pinned",
            bool(COMMIT_SHA.match(entry.revision)),
            entry.revision
            if COMMIT_SHA.match(entry.revision)
            else f"{entry.revision} is not a SHA",
        ),
    ]
    if entry.role != "candidate":
        return results

    limit = gates.qlora.gpu_memory_gb * gates.qlora.usable_fraction
    needed = training_memory_gb(entry.parameters, gates.qlora)
    results.append(
        GateResult(
            "qlora fits one gpu",
            round(needed - limit, 9) <= 0,  # rounded, so the limit itself passes
            f"about {needed:.1f} GB needed, {limit:.1f} GB usable at length "
            f"{gates.qlora.sequence_length}",
        )
    )
    results.append(
        GateResult(
            "vllm serves it with lora",
            entry.architecture in gates.vllm.lora_architectures,
            f"{entry.architecture} in vllm {gates.vllm.version}",
        )
    )
    return results


def check_hub(entry: ModelEntry, hub: Hub) -> list[GateResult]:
    """Compare a pinned entry with the Hub. A revision that is not a SHA cannot be compared."""
    if not COMMIT_SHA.match(entry.revision):
        return []
    try:
        info = hub(entry.hf_id, entry.revision)
    except Exception as error:
        return [GateResult("hub lookup", False, f"{type(error).__name__}: {error}")]
    return [
        GateResult("hub revision", info.sha == entry.revision, f"hub has {info.sha}"),
        GateResult(
            "hub licence",
            info.licence is not None and info.licence.lower() == entry.licence.lower(),
            f"hub has {info.licence}",
        ),
        GateResult(
            "hub parameters",
            info.parameters == entry.parameters,
            f"hub has {info.parameters}",
        ),
    ]


def pin_revisions(path: Path, hub: Hub) -> dict[str, str]:
    """Replace each revision that is not a SHA with the SHA it resolves to. Returns the changes."""
    registry = load_registry(path)
    pinned = {}
    for key, entry in registry.models.items():
        if not COMMIT_SHA.match(entry.revision):
            pinned[key] = hub(entry.hf_id, entry.revision).sha
    if not pinned:
        return pinned

    # Bytes, not text mode, so that the line endings of the file are kept on every system.
    lines = path.read_bytes().decode("utf-8").splitlines(keepends=True)
    current = None
    for number, line in enumerate(lines):
        entry_start = re.match(r"^  ([\w.-]+):\s*$", line)
        if entry_start:
            current = entry_start.group(1)
        elif current in pinned and re.match(r"^    revision:", line):
            ending = line[len(line.rstrip("\r\n")) :]
            lines[number] = f"    revision: {pinned[current]}{ending}"
    path.write_bytes("".join(lines).encode("utf-8"))
    return pinned


def run_checks(registry: Registry, hub: Hub | None) -> dict[str, list[GateResult]]:
    """All results for each model. `hub` is None to skip the comparison with the Hub."""
    results = {}
    for key, entry in registry.models.items():
        found = check_model(entry, registry.gates)
        if hub is not None:
            found += check_hub(entry, hub)
        results[key] = found
    return results


def format_results(registry: Registry, results: dict[str, list[GateResult]]) -> str:
    lines = []
    for key, found in results.items():
        entry = registry.models[key]
        lines.append(f"{key}  {entry.hf_id}  ({entry.role})")
        lines.extend(
            f"  {'pass' if result.passed else 'FAIL'}  {result.name}: {result.detail}"
            for result in found
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the model registry against its gates.")
    parser.add_argument("--file", type=Path, default=MODELS_FILE)
    parser.add_argument("--offline", action="store_true", help="do not ask the Hub")
    parser.add_argument("--pin", action="store_true", help="pin revisions that are not SHAs")
    parser.add_argument("--vllm-version", action="store_true", help="print the pinned vLLM version")
    args = parser.parse_args(argv)

    if args.vllm_version:
        print(load_registry(args.file).gates.vllm.version)
        return 0

    try:
        if args.pin:
            if args.offline:
                parser.error("--pin needs the Hub, so it cannot be used with --offline")
            for key, sha in pin_revisions(args.file, hub_info).items():
                print(f"pinned {key} to {sha}")
        registry = load_registry(args.file)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        print(f"cannot read {args.file}: {error}", file=sys.stderr)
        return 2

    results = run_checks(registry, None if args.offline else hub_info)
    print(format_results(registry, results))
    failed = [r for found in results.values() for r in found if not r.passed]
    total = sum(len(found) for found in results.values())
    print()
    if failed:
        print(f"models: FAIL ({len(failed)} of {total} checks)")
        return 1
    print(f"models: PASS ({total} of {total} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
