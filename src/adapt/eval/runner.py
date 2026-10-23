"""Run a system on one split and write the results file.

Run with `make eval PLATFORM=... SYSTEM=... SPLIT=... ENDPOINTS=...`. Every document goes
through the same pipeline the gateway uses. The results go to
`results/round<N>/eval/<platform>/<system>/<split>.json` (N is the round of the data, the
current round unless `--round` says another) and hold the metrics with confidence
intervals, the slices, one row per document, and a manifest of what was run.

`ENDPOINTS` can name several servers that hold the same model and adapter. Document `i` of the
split goes to server `i % n`, each server gets its own share of the requests in flight, and the
results are put back in document order. At temperature 0 this gives the results of one server.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np

from adapt.client import ModelClient, ssl_context
from adapt.config import PLATFORMS_DIR, Platform, load_platform
from adapt.data.gold import gold_path
from adapt.data.rounds import eval_dir, generator_version, resolve_round, round_folder
from adapt.eval.final_lock import CLAIM_SPLITS, lock_file, start_run
from adapt.eval.metrics import account_groups
from adapt.eval.scoring import (
    EvalDocument,
    compute_metrics,
    compute_slices,
    document_rows,
    score,
)
from adapt.eval.stats import SEED
from adapt.eval.threshold import (
    apply_threshold,
    read_threshold,
    tune_threshold,
    tuning_record,
)
from adapt.pipeline import Pipeline, Result
from adapt.prompting import system_prompt

ROOT = PLATFORMS_DIR.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = eval_dir()  # the results of the current round
RESULTS_ROOT = ROOT / "results"
SPLITS = ("dev", "test_iid", "test_ood", "edge", "gold")
# gold_v2 was generated separately from the training generator after the first round. It is
# reported next to gold, and the gate does not read it. dev_real is a separate set too. It is used
# to choose a configuration and is never part of the gate or of a report.
FIXED_SPLITS = ("gold", "gold_v2", "dev_real")
ALL_SPLITS = (*SPLITS, "gold_v2", "dev_real")
# The files that make an adapter. A folder with more files (a readme, a tokenizer) has the same
# adapter, so the final claim hashes only these.
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
SYSTEM_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# A request waits for its turn on a busy server. The 30 seconds of the gateway are too short
# for 8 requests in flight on a T4, and a timeout would count as an error of the model.
REQUEST_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class Canary:
    name: str
    text: str
    chart: str


def _read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def default_splits(platform: str, root: Path = PLATFORMS_DIR) -> tuple[str, ...]:
    """The splits to evaluate when none are named: all of them, and gold_v2 and dev_real once
    their files exist."""
    extra = tuple(
        name for name in ("gold_v2", "dev_real") if gold_path(platform, root, name).exists()
    )
    return (*SPLITS, *extra)


def split_path(
    platform: str, split: str, data_dir: Path = DATA_DIR, data_round: int | None = None
) -> Path:
    """The file of a split. The evaluation sets of `FIXED_SPLITS` do not depend on the round.
    `data_round` None is the current round."""
    if split in FIXED_SPLITS:
        return gold_path(platform, data_dir.parent / "platforms", split)
    return round_folder(platform, data_round, data_dir) / f"{split}.jsonl"


def load_documents(
    platform: str, split: str, data_dir: Path = DATA_DIR, data_round: int | None = None
) -> list[EvalDocument]:
    """The documents of one split. The documents of the fixed sets (gold, gold_v2 and dev_real)
    have no category or noise level, except that the supplier is compared with the training
    names."""
    if split not in ALL_SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {', '.join(ALL_SPLITS)}")
    records = _read_lines(split_path(platform, split, data_dir, data_round))
    if split not in FIXED_SPLITS:
        return [
            EvalDocument(
                r["id"],
                r["document_text"],
                r["chart"],
                r["target"],
                r["expected_review"],
                r["category"],
                r["layout"],
                r["noise"],
                r["seen_supplier"],
            )
            for r in records
        ]
    train = {
        r["supplier"] for r in _read_lines(split_path(platform, "train", data_dir, data_round))
    }
    return [
        EvalDocument(
            r["id"],
            r["document_text"],
            r["chart"],
            r["target"],
            r["target"]["needs_review"],
            layout=r.get("layout"),
            seen_supplier=r["target"]["supplier"] in train,
        )
        for r in records
    ]


def load_canaries(
    platform: str, data_dir: Path = DATA_DIR, data_round: int | None = None
) -> list[Canary]:
    """The training documents with a canary name, as they are sent at eval time: name removed."""
    train = _read_lines(split_path(platform, "train", data_dir, data_round))
    return [
        Canary(r["canary"]["name"], r["canary"]["text_without_name"], r["chart"])
        for r in train
        if r["canary"]
    ]


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_directory(path: Path, names: Sequence[str] | None = None) -> str:
    """One hash for all files below `path`, from their relative names and contents. With `names`,
    only the files with those relative names.

    The files are sorted by the text of their names. Sorting the paths themselves is not the
    same on every system: Windows ignores case, so "README.md" would come last."""
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")
    digest = hashlib.sha256()
    files = [
        p
        for p in path.rglob("*")
        if p.is_file() and (names is None or p.relative_to(path).as_posix() in names)
    ]
    for file in sorted(files, key=lambda p: p.relative_to(path).parts):
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(hash_file(file).encode() + b"\n")
    return digest.hexdigest()


def git_state(root: Path = ROOT) -> tuple[str | None, bool | None]:
    """The current commit and whether tracked files have changes, or None without git.

    Files that git does not track are not changes. Results and adapters of earlier runs are such
    files, and they would mark every later run as dirty."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(changes)


def gpu_name() -> str | None:
    """The name of the first GPU, or None if there is no NVIDIA driver."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return output.splitlines()[0].strip() if output.strip() else None


def server_version(endpoint: str) -> str | None:
    """The vLLM version, from the server's /version page, or None if it does not answer."""
    parts = urlsplit(endpoint)
    try:
        url = f"{parts.scheme}://{parts.netloc}/version"
        response = httpx.get(url, timeout=5.0, verify=ssl_context())
        response.raise_for_status()
        return str(response.json()["version"])
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def run_shard(
    pipeline: Pipeline, documents: Sequence[EvalDocument], workers: int, label: str = ""
) -> tuple[list[Result], list[float]]:
    """Process the documents on one server. Returns the results and the time per document."""

    def one(document: EvalDocument) -> tuple[Result, float]:
        start = time.perf_counter()
        result = pipeline.process(document.text, document.chart)
        return result, time.perf_counter() - start

    results, latencies = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (result, latency) in enumerate(pool.map(one, documents), start=1):
            results.append(result)
            latencies.append(latency)
            if done % 100 == 0:
                print(f"{label}{done}/{len(documents)} documents", file=sys.stderr)
    return results, latencies


def run_documents(
    pipelines: Sequence[Pipeline], documents: Sequence[EvalDocument], workers: int
) -> tuple[list[Result], list[float], float]:
    """Process the documents, one pipeline per server. Document `i` goes to server `i % n`.

    Each server works on its share at the same time with `workers` requests in flight. Returns the
    results and the time per document, both in document order, and the wall time."""
    count = len(pipelines)
    started = time.perf_counter()
    if count == 1:
        results, latencies = run_shard(pipelines[0], documents, workers)
        return results, latencies, time.perf_counter() - started

    def shard(index: int) -> tuple[list[Result], list[float]]:
        label = f"server {index + 1}: "
        return run_shard(pipelines[index], documents[index::count], workers, label)

    with ThreadPoolExecutor(max_workers=count) as servers:
        shards = list(servers.map(shard, range(count)))
    results: list[Any] = [None] * len(documents)
    latencies = [0.0] * len(documents)
    for index, (shard_results, shard_latencies) in enumerate(shards):
        results[index::count] = shard_results
        latencies[index::count] = shard_latencies
    return results, latencies, time.perf_counter() - started


def canary_pairs(pipeline: Pipeline, canaries: Sequence[Canary]) -> list[tuple[str, str]]:
    """Send each canary document with its name removed. Return the name and the supplier found."""
    pairs = []
    for canary in canaries:
        result = pipeline.process(canary.text, canary.chart)
        pairs.append((canary.name, (result.output or {}).get("supplier", "")))
    return pairs


@dataclass(frozen=True)
class RunSettings:
    platform: str
    system: str
    split: str
    endpoint: str | tuple[str, ...]  # one server, or several that hold the same model and adapter
    model: str
    threshold: float | None = None
    workers: int = 8
    model_revision: str | None = None
    adapter_path: Path | None = None
    few_shot: bool = False
    data_dir: Path = DATA_DIR
    out_dir: Path = RESULTS_DIR
    data_round: int | None = None  # None: the current round

    def __post_init__(self) -> None:
        if not self.endpoints:
            raise ValueError("at least one endpoint is needed")

    @property
    def endpoints(self) -> tuple[str, ...]:
        return (self.endpoint,) if isinstance(self.endpoint, str) else tuple(self.endpoint)


def resolve_threshold(settings: RunSettings) -> tuple[float | None, str]:
    """The confidence threshold for a run, and where it came from.

    A threshold that is given is used. On dev with none given, it is tuned after the run, so
    None is returned. On any other split it is read from the dev results of the same system.
    """
    if settings.threshold is not None:
        return settings.threshold, "given"
    if settings.split == "dev":
        return None, "tuned on dev"
    dev_file = results_path(replace(settings, split="dev"), settings.out_dir)
    if not dev_file.exists():
        raise ValueError(
            f"no threshold for {settings.system}: run the dev split first or give a threshold"
        )
    value = read_threshold(dev_file)
    if value is None:
        return 0.0, "dev target not reached, no confidence limit"
    return value, f"tuned on dev ({dev_file.name})"


def evaluate(settings: RunSettings, platform: Platform | None = None) -> dict[str, Any]:
    """Run the split and return the results as a dictionary."""
    if not SYSTEM_NAME.match(settings.system):
        raise ValueError(f"system name {settings.system!r} has characters that are not allowed")
    threshold, source = resolve_threshold(settings)
    platform = platform or load_platform(settings.platform)
    data_round = resolve_round(settings.data_round, settings.data_dir)
    documents = load_documents(settings.platform, settings.split, settings.data_dir, data_round)
    canaries = load_canaries(settings.platform, settings.data_dir, data_round)
    adapter_hash = (
        hash_directory(settings.adapter_path, ADAPTER_FILES) if settings.adapter_path else None
    )
    groups = account_groups(platform.rules)
    versions = [server_version(endpoint) for endpoint in settings.endpoints]
    if len(set(versions)) > 1:
        raise ValueError(f"the servers do not run the same vllm version: {versions}")

    with ExitStack() as stack:
        clients = [
            stack.enter_context(ModelClient(endpoint, timeout=REQUEST_TIMEOUT_SECONDS))
            for endpoint in settings.endpoints
        ]
        # When the threshold is still to be tuned, run without one and apply it afterwards.
        pipelines = [
            Pipeline(
                platform,
                client,
                adapter=settings.model,
                threshold=0.0 if threshold is None else threshold,
                adapter_version=adapter_hash,
                few_shot=settings.few_shot,
            )
            for client in clients
        ]
        results, latencies, wall = run_documents(pipelines, documents, settings.workers)
        pairs = canary_pairs(pipelines[0], canaries)

    confidence_threshold: dict[str, Any] = {"value": threshold, "source": source}
    if threshold is None:
        gate = platform.config.gate
        tuned = tune_threshold(
            document_rows(score(documents, results, groups)),
            gate.accuracy_automated_min,
            gate.critical_error_rate_max,
        )
        threshold = tuned.threshold if tuned.threshold is not None else 0.0
        if tuned.threshold is None:
            confidence_threshold["source"] = "dev target not reached, no confidence limit"
        elif not tuned.critical_reached:
            confidence_threshold["source"] = "tuned on dev, critical-error limit not reached"
        elif not tuned.accuracy_reached:
            confidence_threshold["source"] = "tuned on dev, accuracy target not reached"
        confidence_threshold |= tuning_record(tuned)
        results = [apply_threshold(result, threshold) for result in results]

    scored = score(documents, results, groups)
    data_file = split_path(settings.platform, settings.split, settings.data_dir, data_round)
    commit, dirty = git_state()
    prompt = system_prompt(
        settings.platform, platform.rules.chart, platform.rules, few_shot=settings.few_shot
    )
    p50, p95 = np.percentile(latencies, [50, 95]) if latencies else (None, None)
    return {
        "platform": settings.platform,
        "system": settings.system,
        "split": settings.split,
        "manifest": {
            "model": settings.model,
            "model_revision": settings.model_revision,
            "adapter_hash": adapter_hash,
            "vllm_version": versions[0],
            "servers": len(settings.endpoints),
            "gpu": gpu_name(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "few_shot": settings.few_shot,
            "data_round": data_round,
            "data_generator": generator_version(data_round, settings.data_dir),
            "data_sha256": hash_file(data_file),
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": SEED,
            "threshold": threshold,
            "documents": len(documents),
        },
        "confidence_threshold": confidence_threshold,
        "performance": {
            "workers": settings.workers,
            "wall_seconds": wall,
            "documents_per_second": len(documents) / wall if wall else None,
            "latency_p50_seconds": None if p50 is None else float(p50),
            "latency_p95_seconds": None if p95 is None else float(p95),
        },
        "metrics": compute_metrics(scored, pairs),
        "slices": compute_slices(scored),
        "documents": document_rows(scored),
    }


def final_files(
    platform: str, data_dir: Path = DATA_DIR, data_round: int | None = None
) -> dict[str, Any]:
    """The test splits the gate reads and the gold sets, by hash: the files of a final claim."""
    data_round = resolve_round(data_round, data_dir)
    platforms = data_dir.parent / "platforms"
    return {
        **{
            f"{name}_sha256": hash_file(split_path(platform, name, data_dir, data_round))
            for name in ("test_iid", "test_ood", "edge")
        },
        "gold_sha256": {
            name: hash_file(gold_path(platform, platforms, name))
            for name in ("gold", "gold_v2")
            if gold_path(platform, platforms, name).exists()
        },
    }


def claim_final_split(
    platform: str,
    system: str,
    split: str,
    out_dir: Path,
    *,
    adapter_path: Path | None = None,
    data_dir: Path = DATA_DIR,
    data_round: int | None = None,
    few_shot: bool = False,
    override: bool = False,
    results_root: Path = RESULTS_ROOT,
) -> None:
    """Allow a run on test_ood or a gold set, or refuse it (see final_lock). Other splits, the
    few-shot baseline and runs that write outside the results folder are not part of a claim."""
    path = lock_file(out_dir, results_root)
    if split not in CLAIM_SPLITS or few_shot or path is None:
        return
    adapter_hash = hash_directory(adapter_path, ADAPTER_FILES) if adapter_path else None
    files = final_files(platform, data_dir, data_round)
    start_run(path, files, system, adapter_hash, override)


def results_file(out_dir: Path, platform: str, system: str, split: str) -> Path:
    return out_dir / platform / system / f"{split}.json"


def results_path(settings: RunSettings, out_dir: Path = RESULTS_DIR) -> Path:
    return results_file(out_dir, settings.platform, settings.system, settings.split)


def write_results(results: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


def _summary(results: Mapping[str, Any]) -> str:
    def show(name: str) -> str:
        value = results["metrics"][name]["value"]
        return f"{name} {'not defined' if value is None else f'{value:.3f}'}"

    lines = [
        f"  {show(name)}"
        for name in ("document_accuracy", "automation_rate", "accuracy_on_automated")
    ]
    threshold = results["manifest"]["threshold"]
    source = results["confidence_threshold"]["source"]
    lines.append(f"  confidence threshold {threshold:.4f} ({source})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a system on one split.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--system", required=True, help="name for the results folder")
    parser.add_argument("--split", default="dev", choices=ALL_SPLITS)
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=["http://localhost:8000/v1"],
        help="one or more servers with the same model and adapter",
    )
    parser.add_argument("--model", help="model or adapter name on the server (default: system)")
    parser.add_argument(
        "--threshold",
        type=float,
        help="confidence threshold (default: tuned on dev, or read from the dev results)",
    )
    parser.add_argument("--workers", type=int, default=8, help="requests in flight")
    parser.add_argument("--model-revision", help="revision of the base model, for the manifest")
    parser.add_argument("--adapter-path", type=Path, help="adapter folder, hashed for the manifest")
    parser.add_argument("--few-shot", action="store_true", help="add the examples to the prompt")
    parser.add_argument("--round", type=int, help="round of the data (default: the current one)")
    parser.add_argument("--out-dir", type=Path, help="default: results/round<N>/eval")
    parser.add_argument(
        "--override-final-lock",
        action="store_true",
        help="run on test_ood or gold files that already back a final claim (written to the lock)",
    )
    args = parser.parse_args(argv)
    try:
        out_dir = args.out_dir or eval_dir(args.round)
    except ValueError as error:
        sys.exit(f"error: {error}")

    settings = RunSettings(
        args.platform,
        args.system,
        args.split,
        tuple(args.endpoints),
        args.model or args.system,
        args.threshold,
        args.workers,
        args.model_revision,
        args.adapter_path,
        few_shot=args.few_shot,
        out_dir=out_dir,
        data_round=args.round,
    )
    try:
        claim_final_split(
            args.platform,
            args.system,
            args.split,
            out_dir,
            adapter_path=args.adapter_path,
            data_round=args.round,
            few_shot=args.few_shot,
            override=args.override_final_lock,
        )
        results = evaluate(settings)
    except ValueError as error:
        sys.exit(f"error: {error}")
    path = results_path(settings, out_dir)
    write_results(results, path)
    print(f"{args.system} on {args.split}: {len(results['documents'])} documents")
    print(_summary(results))
    print(f"results: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
