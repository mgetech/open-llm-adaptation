"""Health, readiness and metrics of the gateway."""

import copy
import re

import pytest
import yaml
from fastapi.testclient import TestClient

from adapt.client import ModelClient, ModelError
from adapt.eval.runner import ADAPTER_FILES, hash_directory
from adapt.gateway import health
from adapt.gateway.app import create_app
from adapt.gateway.audit import AuditLog
from adapt.gateway.health import AdapterHashes
from adapt.serve import SERVE_FILE, ServeConfig
from test_gateway import ALPHA, BETA, BODY, KEYS, URL, bearer
from test_pipeline import OUTPUT


@pytest.fixture
def adapter_folder(tmp_path):
    folder = tmp_path / "adapter"
    folder.mkdir()
    (folder / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(b"weights")
    return folder


@pytest.fixture
def serve(adapter_folder):
    raw = yaml.safe_load(SERVE_FILE.read_text(encoding="utf-8"))
    raw["adapters"][0].update(
        path=str(adapter_folder), sha256=hash_directory(adapter_folder, ADAPTER_FILES)
    )
    return ServeConfig.model_validate(copy.deepcopy(raw))


def make_client(serve, url, tmp_path):
    app = create_app(serve, KEYS, url, AuditLog(tmp_path / "audit.jsonl"), timeout=2.0)
    return TestClient(app)


@pytest.fixture
def client(server, serve, tmp_path):
    server.models = [serve.model, "tax-booking"]
    with make_client(serve, server.url, tmp_path) as client:
        yield client


def test_healthz_answers_without_a_key_and_without_calling_the_model(client, server):
    response = client.get("/healthz")
    assert response.status_code == 200 and response.json() == {"status": "ok"}
    assert server.requests == []


def test_readyz_is_ready_when_vllm_serves_everything_and_the_hashes_match(client):
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"ready": True, "problems": []}


def test_readyz_needs_no_key(client):
    assert client.get("/readyz", headers={}).status_code == 200


def test_readyz_fails_when_vllm_is_down(serve, tmp_path):
    with make_client(serve, "http://127.0.0.1:1/v1", tmp_path) as client:
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["ready"] is False
        assert response.json()["problems"] == ["vLLM cannot be reached (unavailable)"]
        assert client.get("/healthz").status_code == 200


def test_readyz_fails_when_vllm_does_not_list_models_at_all(server, serve, tmp_path):
    server.models = None  # the fake answers 404
    with make_client(serve, server.url, tmp_path) as client:
        assert client.get("/readyz").status_code == 503


def test_readyz_fails_when_an_adapter_is_not_loaded(server, serve, tmp_path):
    server.models = [serve.model]
    with make_client(serve, server.url, tmp_path) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["problems"] == ["tax-booking is not served by vLLM"]


def test_readyz_fails_when_the_base_model_is_not_served(server, serve, tmp_path):
    server.models = ["tax-booking"]
    with make_client(serve, server.url, tmp_path) as client:
        assert client.get("/readyz").json()["problems"] == ["qwen3-4b is not served by vLLM"]


def test_readyz_fails_when_an_adapter_hash_does_not_match(client, adapter_folder):
    assert client.get("/readyz").status_code == 200
    (adapter_folder / "adapter_model.safetensors").write_bytes(b"other weights!")
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["problems"] == [
        "adapter tax-booking does not have the hash in serve.yaml"
    ]


def test_readyz_fails_when_an_adapter_is_not_on_disk(client, adapter_folder):
    (adapter_folder / "adapter_model.safetensors").unlink()
    assert client.get("/readyz").json()["problems"] == ["adapter tax-booking is not on disk"]


def test_the_hash_is_worked_out_again_only_when_a_file_changes(adapter_folder, monkeypatch):
    calls = []
    real = health.hash_directory
    monkeypatch.setattr(health, "hash_directory", lambda *a: calls.append(a) or real(*a))
    hashes = AdapterHashes()
    first = hashes.of(adapter_folder)
    assert hashes.of(adapter_folder) == first and len(calls) == 1
    (adapter_folder / "adapter_config.json").write_text('{"r": 8, "changed": true}')
    assert hashes.of(adapter_folder) != first and len(calls) == 2


def metric_value(text, name, **labels):
    pairs = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
    match = re.search(rf"^{name}\{{{re.escape(pairs)}\}} ([0-9.e+-]+)$", text, re.MULTILINE)
    return float(match.group(1)) if match else None


def test_metrics_count_requests_by_platform_and_outcome(client, server):
    server.reply_with_output(OUTPUT)
    client.post(URL, json=BODY, headers=bearer(ALPHA))
    client.post(URL, json={**BODY, "chart": "SKR04"}, headers=bearer(ALPHA))
    client.post(URL, json=BODY, headers=bearer("wrong"))
    client.post(URL, json=BODY, headers=bearer(BETA))
    text = client.get("/metrics").text
    for outcome in ("automated", "review", "unauthorized", "forbidden"):
        assert (
            metric_value(text, "gateway_requests_total", platform="tax-booking", outcome=outcome)
            == 1.0
        )
    assert (
        metric_value(
            text, "gateway_review_reasons_total", platform="tax-booking", reason="unsupported_chart"
        )
        == 1.0
    )
    assert (
        metric_value(text, "gateway_request_duration_seconds_count", platform="tax-booking") == 4.0
    )


def test_the_latency_is_a_histogram(client, server):
    server.reply_with_output(OUTPUT)
    client.post(URL, json=BODY, headers=bearer(ALPHA))
    text = client.get("/metrics").text
    assert "gateway_request_duration_seconds_bucket" in text
    assert 'le="+Inf"' in text and "gateway_request_duration_seconds_sum" in text


def test_a_caller_cannot_make_new_labels_with_a_platform_name(client):
    client.post("/v1/made-up-name/process", json=BODY, headers=bearer(ALPHA))
    client.post("/v1/another-one/process", json=BODY, headers=bearer("wrong"))
    text = client.get("/metrics").text
    assert "made-up-name" not in text and "another-one" not in text
    assert (
        metric_value(text, "gateway_requests_total", platform="unknown", outcome="forbidden") == 1
    )
    assert (
        metric_value(text, "gateway_requests_total", platform="unknown", outcome="unauthorized")
        == 1
    )


def test_metrics_hold_no_key_and_no_document_text(client, server):
    server.reply_with_output(OUTPUT)
    client.post(URL, json=BODY, headers=bearer(ALPHA))
    text = client.get("/metrics").text
    assert ALPHA not in text and "Nordpaper" not in text


def test_every_app_counts_from_zero(server, serve, tmp_path):
    server.reply_with_output(OUTPUT)
    with make_client(serve, server.url, tmp_path) as first:
        first.post(URL, json=BODY, headers=bearer(ALPHA))
    with make_client(serve, server.url, tmp_path) as second:
        text = second.get("/metrics").text
    assert (
        metric_value(text, "gateway_requests_total", platform="tax-booking", outcome="automated")
        is None
    )


def test_a_request_whose_record_cannot_be_written_is_counted_as_an_error(server, serve, tmp_path):
    server.reply_with_output(OUTPUT)
    app = create_app(serve, KEYS, server.url, AuditLog(tmp_path), timeout=2.0)
    with TestClient(app) as client:
        assert client.post(URL, json=BODY, headers=bearer(ALPHA)).status_code == 500
        text = client.get("/metrics").text
    assert (
        metric_value(text, "gateway_requests_total", platform="tax-booking", outcome="error") == 1
    )


def test_the_client_lists_the_names_the_server_serves(server):
    server.models = ["qwen3-4b", "tax-booking"]
    with ModelClient(server.url) as client:
        assert client.models() == ["qwen3-4b", "tax-booking"]


def test_the_client_says_unavailable_when_the_server_has_no_model_list(server):
    with ModelClient(server.url) as client, pytest.raises(ModelError) as raised:
        client.models()
    assert raised.value.kind == "unavailable"


def test_the_client_says_unavailable_when_nothing_listens():
    with ModelClient("http://127.0.0.1:1/v1") as client, pytest.raises(ModelError) as raised:
        client.models(timeout=1.0)
    assert raised.value.kind == "unavailable"
