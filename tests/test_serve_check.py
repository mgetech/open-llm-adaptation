"""The smoke test of the serving stack, run against the gateway and the fake model server."""

import json

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from adapt import serve_check
from adapt.eval.runner import load_canaries, load_documents
from adapt.gateway.app import create_app
from adapt.gateway.audit import AuditLog
from adapt.gateway.keys import load_keys
from adapt.models import load_registry
from adapt.serve import load_serve_config
from adapt.serve_check import (
    OTHER_KEY,
    OWN_KEY,
    PLATFORM,
    prepare_keys,
    run_checks,
    wait_until_ready,
)
from fake_model import Model
from test_gateway_health import adapter_folder, serve  # noqa: F401  (fixtures)

PINNED = load_registry().gates.vllm.version
ROW_FIELDS = {
    "id",
    "status",
    "expected_review",
    "needs_review",
    "reasons",
    "confidence",
    "latency_ms",
}


@pytest.fixture(scope="module")
def documents():
    return load_documents(PLATFORM, "dev")[:4]


@pytest.fixture
def stack(server, serve, documents, tmp_path):  # noqa: F811
    keys_file, secrets_file = tmp_path / "keys.yaml", tmp_path / "keys.json"
    prepare_keys(keys_file, secrets_file)
    server.version = PINNED
    server.models = [serve.model, "tax-booking"]
    server.handler = Model(documents, load_canaries(PLATFORM))
    audit_file = tmp_path / "audit.jsonl"
    app = create_app(serve, load_keys(keys_file), server.url, AuditLog(audit_file), timeout=5.0)
    with TestClient(app) as gateway, httpx.Client(base_url=server.url.removesuffix("/v1")) as vllm:
        yield {
            "gateway": gateway,
            "vllm": vllm,
            "keys": json.loads(secrets_file.read_text(encoding="utf-8")),
            "serve": serve,
            "audit": audit_file,
            "documents": documents,
        }


def check(stack, **changes):
    arguments = {**stack, **changes}
    return run_checks(
        arguments["gateway"],
        arguments["vllm"],
        arguments["keys"],
        arguments["serve"],
        load_registry(),
        arguments["documents"],
        arguments["audit"],
        ready_timeout=2.0,
        interval=0.1,
    )


def passed(report):
    return {c["name"]: c["passed"] for c in report["checks"]}


def test_a_working_stack_passes_every_check(stack):
    report = check(stack)
    assert [c["name"] for c in report["checks"] if not c["passed"]] == []
    assert report["passed"] is True
    assert len(report["documents"]) == 4
    assert all(row["status"] == 200 for row in report["documents"])


def test_the_report_names_the_versions_and_the_hashes(stack):
    report = check(stack)
    assert report["vllm"]["version"] == PINNED and report["vllm"]["pinned"] == PINNED
    assert report["adapter_sha256"] == stack["serve"].adapters[0].sha256
    assert report["base_revision"] == load_registry().models[stack["serve"].model].revision
    assert report["ready_after_seconds"] >= 0 and "not a measurement" in report["note"]


def test_the_report_holds_no_document_text(stack):
    text = json.dumps(check(stack))
    for document in stack["documents"]:
        assert document.text[:30] not in text
    assert stack["keys"][OWN_KEY] not in text and stack["keys"][OTHER_KEY] not in text


def test_a_wrong_vllm_version_fails_only_that_check(stack, server):
    server.version = "0.0.1"
    report = check(stack)
    assert [name for name, ok in passed(report).items() if not ok] == ["vLLM is the pinned version"]
    assert report["passed"] is False


def test_a_missing_adapter_in_the_model_list_fails(stack, server):
    server.models = [stack["serve"].model]
    assert passed(check(stack))["vLLM serves the base model and the adapter"] is False


def test_a_stack_that_is_not_ready_fails_and_still_reports(stack, server):
    server.models = None
    report = check(stack)
    assert passed(report)["the gateway is ready"] is False and report["passed"] is False


def test_a_key_without_the_platform_is_not_enough(stack):
    keys = {**stack["keys"], OWN_KEY: stack["keys"][OTHER_KEY]}
    report = check(stack, keys=keys)
    assert passed(report)["the documents of the dev split are answered"] is False


def test_an_audit_log_with_text_in_it_fails(stack):
    first = check(stack)
    assert first["passed"]
    lines = stack["audit"].read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1]) | {"document_text": "Nordpaper GmbH, 4 boxes"}
    stack["audit"].write_text("\n".join([*lines[:-1], json.dumps(record)]) + "\n", encoding="utf-8")
    second = check(stack, audit=stack["audit"])
    assert (
        passed(second)["the audit log has a record for every request, with hashes and no text"]
        is False
    )


def test_the_committed_result_is_of_the_shipped_stack_and_holds_no_text():
    path = serve_check.RESULT_FILE
    report = json.loads(path.read_text(encoding="utf-8"))
    (adapter,) = load_serve_config().adapters
    assert report["kind"] == "serving smoke test" and report["passed"] is True
    assert all(check["passed"] for check in report["checks"])
    assert report["adapter_sha256"] == adapter.sha256
    assert report["vllm"]["version"] == report["vllm"]["pinned"] == PINNED
    assert report["base_revision"] == load_registry().models[load_serve_config().model].revision
    assert report["gpu"] and report["git_dirty"] is False
    assert {"document_text", "text", "key"}.isdisjoint(report)
    assert all(set(row) <= ROW_FIELDS for row in report["documents"])


def test_the_keys_of_the_test_are_written_once(tmp_path):
    keys_file, secrets_file = tmp_path / "keys.yaml", tmp_path / "logs" / "keys.json"
    prepare_keys(keys_file, secrets_file)
    entries = yaml.safe_load(keys_file.read_text(encoding="utf-8"))["keys"]
    secrets = json.loads(secrets_file.read_text(encoding="utf-8"))
    assert {e["id"]: e["platforms"] for e in entries} == {OWN_KEY: [PLATFORM], OTHER_KEY: ["other"]}
    assert secrets[OWN_KEY] not in keys_file.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="exists"):
        prepare_keys(keys_file, secrets_file)


def test_waiting_for_the_gateway_stops_at_the_timeout():
    class Never:
        def get(self, path):
            raise httpx.ConnectError("refused")

    now = [0.0]
    seconds, answer = wait_until_ready(
        Never(),
        timeout=10,
        interval=4,
        sleep=lambda s: now.__setitem__(0, now[0] + s),
        clock=lambda: now[0],
    )
    assert seconds >= 10 and "refused" in answer


@pytest.mark.parametrize(("passes", "code"), [(True, 0), (False, 1)])
def test_the_entry_point_writes_the_result_file_and_exits_by_the_result(
    passes, code, monkeypatch, tmp_path
):
    out = tmp_path / "out" / "smoke.json"
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({OWN_KEY: "a", OTHER_KEY: "b"}), encoding="utf-8")
    monkeypatch.setattr(serve_check, "KEY_SECRETS", keys)
    report = {"passed": passes, "checks": [{"name": "x", "passed": passes, "detail": ""}]}
    calls = []
    monkeypatch.setattr(serve_check, "run_checks", lambda *a, **k: calls.append(k) or report)
    assert serve_check.main(["--out", str(out), "--documents", "2", "--wait", "7"]) == code
    assert json.loads(out.read_text(encoding="utf-8")) == report
    assert calls[0]["ready_timeout"] == 7.0


def test_without_the_keys_of_the_test_the_entry_point_says_what_to_do(monkeypatch, tmp_path):
    monkeypatch.setattr(serve_check, "KEY_SECRETS", tmp_path / "none.json")
    with pytest.raises(SystemExit, match="error"):
        serve_check.main(["--out", str(tmp_path / "out.json")])
