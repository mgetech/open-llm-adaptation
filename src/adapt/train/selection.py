"""Choose the checkpoint of a run that becomes its adapter.

A run keeps a checkpoint every `checkpoint.save_steps` steps. The loss on the answers does not say
which of them books best, because most of the answer is wording and only the account and the VAT
rate decide the result. So this scores the checkpoints of `selection.steps` on the dev_real
documents, with the whole pipeline and no confidence limit, and keeps the one with the lowest
critical error rate. A tie goes to the higher document accuracy, then to the later step. The
documents are never trained on.

One server per GPU holds the base model and all the checkpoints as LoRA modules, so the model is
loaded once. Each checkpoint is copied without its optimiser state to
`adapters/<platform>/<run>-ckpt<step>`, and its results go to
`<out-dir>/<platform>/<system>-ckpt<step>`. The chosen checkpoint replaces the adapter of the run,
and the choice and the scores are written into the run manifest. Run it once, after training and
before the evaluation of the adapter.
"""

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from adapt.baseline import START_TIMEOUT_SECONDS, free_port
from adapt.data.rounds import eval_dir, resolve_round
from adapt.data.rounds import runs_dir as round_runs_dir
from adapt.eval.runner import hash_directory, hash_file, results_file, split_path
from adapt.models import Registry, load_registry
from adapt.train.config import load_train_config
from adapt.train.evaluate import evaluate_adapters, system_name
from adapt.train.run import ADAPTERS_DIR, TrainSettings, run_name

SPLIT = "dev_real"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
METHOD = "lowest critical error rate, then highest document accuracy, then the later step"
NO_LIMIT = 0.0  # every answer that no rule and not the model sent to review is automated


def candidate_system(system: str, step: int) -> str:
    return f"{system}-ckpt{step}"


def export_adapter(checkpoint: Path, target: Path) -> Path:
    """Copy the adapter files of a checkpoint, and not its optimiser state, to a clean folder."""
    missing = [name for name in ADAPTER_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"{checkpoint} has no {missing[0]}")
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for name in ADAPTER_FILES:
        shutil.copy2(checkpoint / name, target / name)
    return target


def scores(results: Mapping[str, Any]) -> dict[str, float | None]:
    metrics = results["metrics"]
    return {
        name: metrics[name]["value"]
        for name in ("critical_error_rate", "document_accuracy", "automation_rate")
    }


def choose(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The candidate with the lowest critical error rate, then the highest document accuracy, then
    the later step. A rate that cannot be computed, because nothing was automated, counts as 1."""

    def order(candidate: Mapping[str, Any]) -> tuple[float, float, int]:
        rate = candidate["critical_error_rate"]
        accuracy = candidate["document_accuracy"]
        return (
            1.0 if rate is None else rate,
            -(0.0 if accuracy is None else accuracy),
            -candidate["step"],
        )

    return min(candidates, key=order)


def select_checkpoint(
    registry: Registry,
    key: str | None,
    seed: int,
    out_dir: Path,
    *,
    platform: str = "tax-booking",
    adapters_dir: Path = ADAPTERS_DIR,
    runs_dir: Path | None = None,
    steps: Sequence[int] | None = None,
    smoke: bool = False,
    port: int = 8000,
    servers: int = 1,
    dry_run: bool = False,
    workers: int = 8,
    timeout: float = START_TIMEOUT_SECONDS,
    data_round: int | None = None,
) -> dict[str, Any]:
    """Score the checkpoints, put the best one in the place of the adapter, write the choice into
    the manifest of the run, and return the `selection` entry."""
    steps = tuple(steps) if steps else tuple(load_train_config().selection.steps)
    data_round = resolve_round(data_round)
    train = TrainSettings(platform, seed, model=key, smoke=smoke, data_round=data_round)
    name = run_name(train)
    system = system_name(key, seed, smoke=smoke, data_round=data_round)
    adapter_dir = adapters_dir / platform / name
    checkpoint_dir = adapters_dir / platform / f"{name}-checkpoints"
    runs_dir = runs_dir or (adapter_dir.parent if smoke else round_runs_dir(data_round))
    manifest_path = runs_dir / f"{platform}-{name}.json"
    if not manifest_path.exists():
        raise ValueError(f"no manifest at {manifest_path}, train the run first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selection"):
        raise ValueError(
            f"{manifest_path.name} has a selection already (step "
            f"{manifest['selection']['chosen_step']}). Remove the entry to choose again"
        )
    missing = [step for step in steps if not (checkpoint_dir / f"checkpoint-{step}").is_dir()]
    if missing:
        raise ValueError(f"no checkpoint for step {missing[0]} in {checkpoint_dir}")

    exports = {
        step: export_adapter(
            checkpoint_dir / f"checkpoint-{step}", adapters_dir / platform / f"{name}-ckpt{step}"
        )
        for step in steps
    }
    systems = {step: candidate_system(system, step) for step in steps}
    if dry_run:
        port = free_port()
    evaluate_adapters(
        registry,
        key,
        {systems[step]: exports[step] for step in steps},
        out_dir,
        platform=platform,
        splits=(SPLIT,),
        smoke=smoke,
        port=port,
        servers=servers,
        dry_run=dry_run,
        workers=workers,
        timeout=timeout,
        data_round=data_round,
        threshold=NO_LIMIT,
    )

    candidates = []
    examples = 0
    for step in steps:
        path = results_file(out_dir, platform, systems[step], SPLIT)
        results = json.loads(path.read_text(encoding="utf-8"))
        examples = len(results["documents"])
        candidates.append(
            {
                "step": step,
                "system": systems[step],
                **scores(results),
                "adapter_sha256": hash_directory(exports[step]),
            }
        )
    chosen = choose(candidates)

    # The chosen checkpoint takes the place of the adapter the run wrote at its last step.
    shutil.rmtree(adapter_dir)
    shutil.copytree(exports[chosen["step"]], adapter_dir)

    selection_file = split_path(platform, SPLIT)
    selection = {
        "split": SPLIT,
        "file": selection_file.name,
        "sha256": hash_file(selection_file),
        "examples": examples,
        "method": METHOD,
        "confidence_limit": NO_LIMIT,
        "candidates": candidates,
        "chosen_step": chosen["step"],
    }
    manifest["selection"] = selection
    manifest["adapter_sha256"] = hash_directory(adapter_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return selection


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Choose the checkpoint that becomes the adapter.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--model", help="key in configs/models.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="the checkpoints of a smoke run")
    parser.add_argument(
        "--steps", type=int, nargs="+", help="default: selection.steps of configs/train.yaml"
    )
    parser.add_argument("--adapters-dir", type=Path, default=ADAPTERS_DIR)
    parser.add_argument("--runs-dir", type=Path, help="default: results/round<N>/runs")
    parser.add_argument("--out-dir", type=Path, help="default: results/round<N>/eval")
    parser.add_argument("--workers", type=int, default=8, help="requests in flight")
    parser.add_argument("--port", type=int, default=8000, help="port of the first server")
    parser.add_argument("--servers", type=int, default=1, help="servers, one per GPU")
    parser.add_argument("--round", type=int, help="round of the data (default: the current one)")
    parser.add_argument("--dry-run", action="store_true", help="use the fake server, not vLLM")
    args = parser.parse_args(argv)
    if not args.smoke and not args.model:
        parser.error("--model is required unless --smoke is given")

    try:
        selection = select_checkpoint(
            load_registry(),
            args.model,
            args.seed,
            args.out_dir or eval_dir(args.round),
            platform=args.platform,
            adapters_dir=args.adapters_dir,
            runs_dir=args.runs_dir,
            steps=args.steps,
            smoke=args.smoke,
            port=args.port,
            servers=args.servers,
            dry_run=args.dry_run,
            workers=args.workers,
            data_round=args.round,
        )
    except (RuntimeError, ValueError) as error:
        sys.exit(f"error: {error}")
    print(f"{selection['split']} ({selection['examples']} documents), {selection['method']}:")
    for candidate in selection["candidates"]:
        rate, accuracy = candidate["critical_error_rate"], candidate["document_accuracy"]
        mark = "  <- chosen" if candidate["step"] == selection["chosen_step"] else ""
        print(
            f"  step {candidate['step']:4}  critical error rate {rate}  "
            f"document accuracy {accuracy}{mark}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
