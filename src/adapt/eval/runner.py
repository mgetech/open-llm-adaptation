"""Run a system on one split and write the results file.

Run with `make eval PLATFORM=... SYSTEM=... SPLIT=... ENDPOINT=...`. Every document goes
through the same pipeline the gateway uses. The results go to
`results/eval/<platform>/<system>/<split>.json` and hold the metrics with confidence
intervals, the slices, one row per document, and a manifest of what was run.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np

from adapt.client import ModelClient, ssl_context
from adapt.config import PLATFORMS_DIR, Platform, load_platform
from adapt.data.gold import gold_path
from adapt.eval.metrics import account_groups
from adapt.eval.scoring import (
    EvalDocument,
    compute_metrics,
    compute_slices,
    document_rows,
    score,
)
from adapt.eval.stats import SEED
from adapt.pipeline import Pipeline, Result
from adapt.prompting import system_prompt

ROOT = PLATFORMS_DIR.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = Path("results/eval")
SPLITS = ("dev", "test_iid", "test_ood", "edge", "gold")
SYSTEM_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class Canary:
    name: str
    text: str
    chart: str


def _read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def split_path(platform: str, split: str, data_dir: Path = DATA_DIR) -> Path:
    if split == "gold":
        return gold_path(platform, data_dir.parent / "platforms")
    return data_dir / platform / f"{split}.jsonl"


def load_documents(platform: str, split: str, data_dir: Path = DATA_DIR) -> list[EvalDocument]:
    """The documents of one split. Gold documents have no category, noise level or flag
    for a seen supplier, except that the supplier is compared with the training names."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {', '.join(SPLITS)}")
    records = _read_lines(split_path(platform, split, data_dir))
    if split != "gold":
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
    train = {r["supplier"] for r in _read_lines(split_path(platform, "train", data_dir))}
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


def load_canaries(platform: str, data_dir: Path = DATA_DIR) -> list[Canary]:
    """The training documents with a canary name, as they are sent at eval time: name removed."""
    train = _read_lines(split_path(platform, "train", data_dir))
    return [
        Canary(r["canary"]["name"], r["canary"]["text_without_name"], r["chart"])
        for r in train
        if r["canary"]
    ]


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_directory(path: Path) -> str:
    """One hash for all files below `path`, from their relative names and contents."""
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(hash_file(file).encode() + b"\n")
    return digest.hexdigest()


def git_state(root: Path = ROOT) -> tuple[str | None, bool | None]:
    """The current commit and whether the working tree has changes, or None without git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
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


def run_documents(
    pipeline: Pipeline, documents: Sequence[EvalDocument], workers: int
) -> tuple[list[Result], list[float], float]:
    """Process the documents. Returns the results, the time per document and the wall time."""

    def one(document: EvalDocument) -> tuple[Result, float]:
        start = time.perf_counter()
        result = pipeline.process(document.text, document.chart)
        return result, time.perf_counter() - start

    started = time.perf_counter()
    results, latencies = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (result, latency) in enumerate(pool.map(one, documents), start=1):
            results.append(result)
            latencies.append(latency)
            if done % 100 == 0:
                print(f"{done}/{len(documents)} documents", file=sys.stderr)
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
    endpoint: str
    model: str
    threshold: float = 0.0
    workers: int = 8
    model_revision: str | None = None
    adapter_path: Path | None = None
    data_dir: Path = DATA_DIR


def evaluate(settings: RunSettings, platform: Platform | None = None) -> dict[str, Any]:
    """Run the split and return the results as a dictionary."""
    if not SYSTEM_NAME.match(settings.system):
        raise ValueError(f"system name {settings.system!r} has characters that are not allowed")
    platform = platform or load_platform(settings.platform)
    documents = load_documents(settings.platform, settings.split, settings.data_dir)
    canaries = load_canaries(settings.platform, settings.data_dir)
    adapter_hash = hash_directory(settings.adapter_path) if settings.adapter_path else None
    groups = account_groups(platform.rules)

    with ModelClient(settings.endpoint) as client:
        pipeline = Pipeline(
            platform,
            client,
            adapter=settings.model,
            threshold=settings.threshold,
            adapter_version=adapter_hash,
        )
        results, latencies, wall = run_documents(pipeline, documents, settings.workers)
        pairs = canary_pairs(pipeline, canaries)

    scored = score(documents, results, groups)
    data_file = split_path(settings.platform, settings.split, settings.data_dir)
    commit, dirty = git_state()
    prompt = system_prompt(settings.platform, platform.rules.chart, platform.rules)
    p50, p95 = np.percentile(latencies, [50, 95]) if latencies else (None, None)
    return {
        "platform": settings.platform,
        "system": settings.system,
        "split": settings.split,
        "manifest": {
            "model": settings.model,
            "model_revision": settings.model_revision,
            "adapter_hash": adapter_hash,
            "vllm_version": server_version(settings.endpoint),
            "gpu": gpu_name(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "data_sha256": hash_file(data_file),
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": SEED,
            "threshold": settings.threshold,
            "documents": len(documents),
        },
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


def results_path(settings: RunSettings, out_dir: Path = RESULTS_DIR) -> Path:
    return out_dir / settings.platform / settings.system / f"{settings.split}.json"


def write_results(results: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


def _summary(results: Mapping[str, Any]) -> str:
    def show(name: str) -> str:
        value = results["metrics"][name]["value"]
        return f"{name} {'not defined' if value is None else f'{value:.3f}'}"

    return "\n".join(
        f"  {show(name)}"
        for name in ("document_accuracy", "automation_rate", "accuracy_on_automated")
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a system on one split.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--system", required=True, help="name for the results folder")
    parser.add_argument("--split", default="dev", choices=SPLITS)
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--model", help="model or adapter name on the server (default: system)")
    parser.add_argument("--threshold", type=float, default=0.0, help="confidence threshold")
    parser.add_argument("--workers", type=int, default=8, help="requests in flight")
    parser.add_argument("--model-revision", help="revision of the base model, for the manifest")
    parser.add_argument("--adapter-path", type=Path, help="adapter folder, hashed for the manifest")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    settings = RunSettings(
        args.platform,
        args.system,
        args.split,
        args.endpoint,
        args.model or args.system,
        args.threshold,
        args.workers,
        args.model_revision,
        args.adapter_path,
    )
    results = evaluate(settings)
    path = results_path(settings, args.out_dir)
    write_results(results, path)
    print(f"{args.system} on {args.split}: {len(results['documents'])} documents")
    print(_summary(results))
    print(f"results: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
