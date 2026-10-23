"""Smoke test of the serving stack: vLLM with the adapter, behind the gateway.

`make serve-check` runs against a stack that `make serve-local` started. It waits for `/readyz`,
checks that vLLM is the pinned version and serves the base model and the adapter, checks the
answers to a missing key, a wrong key and a key without the platform, sends a few documents of
the dev split, and reads the metrics and the audit log. The result goes to a JSON file with no
document text in it. It shows that the stack starts and answers. It does not measure the model.

`--prepare` makes the two keys of the test first: it writes `configs/keys.yaml` and keeps the
keys themselves in `logs/`, which git ignores. It refuses to replace a key file.
"""

import argparse
import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from adapt.eval.runner import RESULTS_ROOT, ROOT, git_state, gpu_name, load_documents
from adapt.eval.scoring import EvalDocument
from adapt.gateway.keys import KEYS_FILE, hash_key, new_key
from adapt.models import Registry, load_registry
from adapt.serve import ServeConfig, load_serve_config

PLATFORM = "tax-booking"
KEY_SECRETS = ROOT / "logs" / "serve-check-keys.json"
AUDIT_LOG = ROOT / "logs" / "audit.jsonl"
RESULT_FILE = RESULTS_ROOT / "serving" / "smoke-test.json"
DOCUMENTS = 6
REQUEST_PREFIX = "smoke-"
OWN_KEY, OTHER_KEY = "smoke-check", "smoke-other"
NO_TOTAL_TEXT = "Delivery note 4411\nTwo boxes of printer paper, no prices on this slip."
AUDIT_FIELDS = {
    "timestamp",
    "request_id",
    "platform",
    "key_id",
    "input_sha256",
    "adapter",
    "adapter_sha256",
    "base_revision",
    "prompt_sha256",
    "outcome",
    "review_reasons",
    "latency_ms",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def prepare_keys(keys_file: Path = KEYS_FILE, secrets_file: Path = KEY_SECRETS) -> None:
    """Write a key file with the two keys of the test, and the keys themselves to `secrets_file`."""
    if keys_file.exists():
        raise ValueError(f"{keys_file} exists; remove it or add the two keys of the test yourself")
    keys = {OWN_KEY: new_key(), OTHER_KEY: new_key()}
    platforms = {OWN_KEY: [PLATFORM], OTHER_KEY: ["other"]}
    entries = [
        {"id": name, "sha256": hash_key(key), "platforms": platforms[name]}
        for name, key in keys.items()
    ]
    keys_file.write_text(yaml.safe_dump({"keys": entries}, sort_keys=False), encoding="utf-8")
    secrets_file.parent.mkdir(parents=True, exist_ok=True)
    secrets_file.write_text(json.dumps(keys), encoding="utf-8")


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def wait_until_ready(
    gateway: httpx.Client,
    timeout: float,
    interval: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[float, Any]:
    """Poll `/readyz` until it answers 200. Returns the seconds waited and the last answer."""
    start = clock()
    answer: Any = "no answer"
    while True:
        try:
            response = gateway.get("/readyz")
            answer = response.json()
            if response.status_code == 200:
                return clock() - start, answer
        except (httpx.HTTPError, ValueError) as error:
            answer = repr(error)
        if clock() - start >= timeout:
            return clock() - start, answer
        sleep(interval)


def read_audit(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def run_checks(
    gateway: httpx.Client,
    vllm: httpx.Client | None,
    keys: Mapping[str, str],
    serve: ServeConfig,
    registry: Registry,
    documents: Sequence[EvalDocument],
    audit_file: Path,
    *,
    ready_timeout: float,
    interval: float = 5.0,
) -> dict[str, Any]:
    checks: list[Check] = []
    rows: list[dict[str, Any]] = []
    sent = 0
    url = f"/v1/{PLATFORM}/process"
    adapter = serve.adapters[0]

    def step(name: str, test: Callable[[], tuple[bool, str]]) -> None:
        try:
            passed, detail = test()
        except Exception as error:  # a check that cannot run is a failed check
            passed, detail = False, repr(error)
        checks.append(Check(name, passed, detail))

    seconds, answer = wait_until_ready(gateway, ready_timeout, interval)
    ready = isinstance(answer, dict) and answer.get("ready") is True
    checks.append(Check("the gateway is ready", ready, f"{seconds:.0f} s, {answer}"))
    vllm_info: dict[str, Any] = {"pinned": registry.gates.vllm.version}
    if vllm is not None:

        def version() -> tuple[bool, str]:
            vllm_info["version"] = vllm.get("/version").json()["version"]
            return vllm_info["version"] == vllm_info["pinned"], str(vllm_info["version"])

        def models() -> tuple[bool, str]:
            listed = [m["id"] for m in vllm.get("/v1/models").json()["data"]]
            vllm_info["models"] = listed
            wanted = {serve.model, *(a.name for a in serve.adapters)}
            return wanted <= set(listed), ", ".join(listed)

        step("vLLM is the pinned version", version)
        step("vLLM serves the base model and the adapter", models)

    def status_of(headers: Mapping[str, str]) -> int:
        nonlocal sent
        sent += 1
        body = {"document_text": NO_TOTAL_TEXT, "chart": "SKR03"}
        headers = {**headers, "X-Request-Id": f"{REQUEST_PREFIX}{sent}"}
        return gateway.post(url, json=body, headers=headers).status_code

    step("no key gives 401", lambda: (status_of({}) == 401, ""))
    step("a wrong key gives 401", lambda: (status_of(bearer("wrong")) == 401, ""))
    step(
        "a key without the platform gives 403",
        lambda: (status_of(bearer(keys[OTHER_KEY])) == 403, ""),
    )

    def post(text: str, chart: str) -> tuple[httpx.Response, float]:
        nonlocal sent
        sent += 1
        headers = {**bearer(keys[OWN_KEY]), "X-Request-Id": f"{REQUEST_PREFIX}{sent}"}
        started = time.perf_counter()
        response = gateway.post(url, json={"document_text": text, "chart": chart}, headers=headers)
        return response, (time.perf_counter() - started) * 1000

    def documents_answer() -> tuple[bool, str]:
        for document in documents:
            response, milliseconds = post(document.text, document.chart)
            data = response.json() if response.status_code == 200 else {}
            rows.append(
                {
                    "id": document.id,
                    "status": response.status_code,
                    "expected_review": document.expected_review,
                    "needs_review": data.get("needs_review"),
                    "reasons": data.get("reasons"),
                    "confidence": data.get("confidence"),
                    "latency_ms": round(milliseconds),
                }
            )
        good = sum(row["status"] == 200 for row in rows)
        return good == len(documents) > 0, f"{good} of {len(documents)} answered with 200"

    step("the documents of the dev split are answered", documents_answer)

    def no_total() -> tuple[bool, str]:
        response, _ = post(NO_TOTAL_TEXT, "SKR03")
        data = response.json()
        return (
            response.status_code == 200
            and data["needs_review"] is True
            and "no_total" in data["reasons"],
            str(data.get("reasons")),
        )

    step("a document without a total goes to review", no_total)

    def metrics() -> tuple[bool, str]:
        text = gateway.get("/metrics").text
        seen = [
            o
            for o in ("automated", "review", "unauthorized", "forbidden")
            if f'outcome="{o}"' in text
        ]
        return {"review", "unauthorized", "forbidden"} <= set(seen), ", ".join(seen)

    step("the metrics count the requests", metrics)

    def audit() -> tuple[bool, str]:
        records = [
            r
            for r in read_audit(audit_file)
            if str(r.get("request_id", "")).startswith(REQUEST_PREFIX)
        ]
        answered = [r for r in records if r["outcome"] in ("automated", "review")]
        problems = []
        if len(records) < sent:
            problems.append(f"{len(records)} records for {sent} requests")
        for record in records:
            if set(record) != AUDIT_FIELDS:
                problems.append(f"fields of {record.get('request_id')}")
        for record in answered:
            if record["adapter_sha256"] != adapter.sha256:
                problems.append(f"adapter hash of {record['request_id']}")
            if record["base_revision"] != registry.models[serve.model].revision:
                problems.append(f"base revision of {record['request_id']}")
            if not SHA256.match(record["prompt_sha256"] or "") or not SHA256.match(
                record["input_sha256"] or ""
            ):
                problems.append(f"hashes of {record['request_id']}")
        return not problems, "; ".join(problems) or f"{len(records)} records"

    step("the audit log has a record for every request, with hashes and no text", audit)

    commit, dirty = git_state()
    return {
        "kind": "serving smoke test",
        "note": "Shows that the stack starts and answers. It is not a measurement of the model.",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "gpu": gpu_name(),
        "vllm": vllm_info,
        "base_model": serve.model,
        "base_revision": registry.models[serve.model].revision,
        "adapter": adapter.name,
        "adapter_sha256": adapter.sha256,
        "ready_after_seconds": round(seconds),
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
        "documents": rows,
        "review_decision_matches_label": sum(
            row["needs_review"] == row["expected_review"] for row in rows
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test of the gateway and vLLM.")
    parser.add_argument("--prepare", action="store_true", help="write the keys of the test")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--vllm-url", help="default: the port of configs/serve.yaml on this machine"
    )
    parser.add_argument("--audit-log", type=Path, default=AUDIT_LOG)
    parser.add_argument("--wait", type=float, default=1500.0, help="seconds to wait for /readyz")
    parser.add_argument("--documents", type=int, default=DOCUMENTS)
    parser.add_argument("--out", type=Path, default=RESULT_FILE)
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            prepare_keys()
            print(f"wrote {KEYS_FILE} and {KEY_SECRETS}")
            return 0
        serve = load_serve_config()
        registry = load_registry()
        keys = json.loads(KEY_SECRETS.read_text(encoding="utf-8"))
        documents = load_documents(PLATFORM, "dev")[: args.documents]
    except (ValueError, OSError) as error:
        sys.exit(f"error: {error}")
    vllm_url = args.vllm_url or f"http://127.0.0.1:{serve.port}"
    with (
        httpx.Client(base_url=args.gateway_url, timeout=120.0) as gateway,
        httpx.Client(base_url=vllm_url, timeout=30.0) as vllm,
    ):
        report = run_checks(
            gateway,
            vllm,
            keys,
            serve,
            registry,
            documents,
            args.audit_log,
            ready_timeout=args.wait,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    for check in report["checks"]:
        print(f"{'ok  ' if check['passed'] else 'FAIL'} {check['name']} {check['detail']}")
    print(f"wrote {args.out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
