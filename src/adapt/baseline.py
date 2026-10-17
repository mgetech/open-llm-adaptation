"""Run the few-shot baseline of one model on every split.

I start the server for the model, wait until it answers, run each split with the few-shot
prompt, and stop the server. The results go to `<out-dir>/<platform>/<model>-fewshot/`.

`--dry-run` starts the fake server from the tests instead of vLLM, so the whole path can be
checked without a GPU. Its results come from a fake model. They are never real results.
"""

import argparse
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import httpx

from adapt.eval.runner import (
    RESULTS_DIR,
    ROOT,
    SPLITS,
    RunSettings,
    evaluate,
    results_file,
    server_version,
    write_results,
)
from adapt.models import Registry, check_model, load_registry

# The few-shot prompt is about 1,500 tokens. A document is at most 6,000 characters, and the
# answer is short.
MAX_MODEL_LEN = 4096
START_TIMEOUT_SECONDS = 1200.0


def system_name(key: str) -> str:
    return f"{key}-fewshot"


def serve_command(registry: Registry, key: str, port: int) -> list[str]:
    """The vLLM command for a base model. A T4 has no bf16, so the weights are fp16."""
    entry = registry.models[key]
    return [
        "vllm",
        "serve",
        entry.hf_id,
        "--revision",
        entry.revision,
        "--served-model-name",
        key,
        "--dtype",
        "half",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        "0.9",
        "--seed",
        "0",
        "--port",
        str(port),
    ]


def fake_command(port: int) -> list[str]:
    return [sys.executable, str(ROOT / "tests" / "fake_serve.py"), "--port", str(port)]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until_ready(
    base_url: str,
    process: subprocess.Popen,
    timeout: float = START_TIMEOUT_SECONDS,
    interval: float = 1.0,
) -> None:
    """Wait until the server answers on /version. Stop at once if its process has ended."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the server stopped with exit code {process.returncode}")
        try:
            if httpx.get(f"{base_url}/version", timeout=5.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(interval)
    raise RuntimeError(f"the server was not ready after {timeout:.0f} seconds")


def stop(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_baseline(
    registry: Registry,
    key: str,
    out_dir: Path,
    *,
    platform: str = "tax-booking",
    port: int = 8000,
    dry_run: bool = False,
    workers: int = 8,
    timeout: float = START_TIMEOUT_SECONDS,
) -> list[Path]:
    """Evaluate `key` with the few-shot prompt on every split. Returns the results files."""
    entry = registry.models[key]
    failed = [r for r in check_model(entry, registry.gates) if not r.passed]
    if failed:
        raise ValueError(f"{key} fails the model gates: " + ", ".join(r.name for r in failed))

    command = fake_command(port) if dry_run else serve_command(registry, key, port)
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
        for split in SPLITS:
            settings = RunSettings(
                platform,
                system_name(key),
                split,
                endpoint,
                key,
                workers=workers,
                model_revision=entry.revision,
                few_shot=True,
                out_dir=out_dir,
            )
            results = evaluate(settings)
            path = results_file(out_dir, platform, settings.system, split)
            write_results(results, path)
            written.append(path)
            print(f"{key} on {split}: {len(results['documents'])} documents, results in {path}")
    finally:
        stop(process)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the few-shot baseline of one model.")
    parser.add_argument("--platform", default="tax-booking")
    parser.add_argument("--model", required=True, help="key in configs/models.yaml")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--workers", type=int, default=8, help="requests in flight")
    parser.add_argument(
        "--port", type=int, help="port of the server (default: 8000, or a free one)"
    )
    parser.add_argument("--dry-run", action="store_true", help="use the fake server, not vLLM")
    args = parser.parse_args(argv)

    registry = load_registry()
    if args.model not in registry.models:
        known = ", ".join(registry.models)
        sys.exit(f"error: unknown model {args.model!r}, expected one of {known}")
    port = args.port or (free_port() if args.dry_run else 8000)
    try:
        run_baseline(
            registry,
            args.model,
            args.out_dir,
            platform=args.platform,
            port=port,
            dry_run=args.dry_run,
            workers=args.workers,
        )
    except (RuntimeError, ValueError) as error:
        sys.exit(f"error: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
