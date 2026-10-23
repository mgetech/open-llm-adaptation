"""The audit log of the gateway: the fields of the record, and that no document text is in it."""

import hashlib
import json
import threading

import pytest
from fastapi.testclient import TestClient

from adapt.config import load_platform
from adapt.data.pseudonymise import pseudonymise
from adapt.gateway.app import create_app
from adapt.gateway.audit import AuditLog, AuditRecord, now
from adapt.models import load_registry
from adapt.prompting import system_prompt
from adapt.serve import load_serve_config
from test_gateway import ALPHA, BETA, BODY, KEYS, URL, bearer
from test_pipeline import OUTPUT, TEXT

FIELDS = {
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


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "logs" / "audit.jsonl")


@pytest.fixture
def client(server, audit):
    with TestClient(create_app(load_serve_config(), KEYS, server.url, audit, timeout=2.0)) as c:
        yield c


def test_a_request_writes_one_record_with_the_fields_of_the_spec(client, server, audit):
    server.reply_with_output(OUTPUT)
    response = client.post(URL, json=BODY, headers=bearer(ALPHA))
    (record,) = audit.records()
    assert set(record) == FIELDS
    assert record["request_id"] == response.json()["request_id"]
    assert record["platform"] == "tax-booking" and record["key_id"] == "alpha"
    assert record["outcome"] == "automated" and record["review_reasons"] == []
    assert isinstance(record["latency_ms"], int) and record["latency_ms"] >= 0
    assert record["timestamp"].endswith("+00:00")


def test_the_record_names_the_adapter_the_base_revision_and_the_prompt(client, server, audit):
    server.reply_with_output(OUTPUT)
    client.post(URL, json=BODY, headers=bearer(ALPHA))
    record = audit.records()[0]
    config = load_serve_config()
    platform = load_platform("tax-booking")
    prompt = system_prompt("tax-booking", platform.rules.chart, platform.rules, few_shot=False)
    assert record["adapter"] == "tax-booking"
    assert record["adapter_sha256"] == config.adapters[0].sha256
    assert record["base_revision"] == load_registry().models[config.model].revision
    assert record["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()


def test_the_input_is_recorded_as_the_hash_of_the_pseudonymised_text(client, server, audit):
    text = TEXT + "\nPay to DE89 3704 0044 0532 0130 00, mail anna@example.com"
    other = TEXT + "\nPay to DE02 1203 0000 0000 2020 51, mail bernd@example.org"
    for document in (text, other):
        server.reply_with_output(OUTPUT)
        client.post(URL, json={**BODY, "document_text": document}, headers=bearer(ALPHA))
    first, second = audit.records()
    assert first["input_sha256"] == hashlib.sha256(pseudonymise(text).encode()).hexdigest()
    assert first["input_sha256"] == second["input_sha256"]


def test_a_review_is_recorded_with_its_reasons(client, server, audit):
    client.post(URL, json={**BODY, "chart": "SKR04"}, headers=bearer(ALPHA))
    (record,) = audit.records()
    assert record["outcome"] == "review" and "unsupported_chart" in record["review_reasons"]


def test_no_part_of_the_document_is_in_the_log(client, server, audit):
    text = (
        "Zebrafink Quarzwerk GmbH, Rechnung 4711-AX\n"
        "Lieferung von Granitplatten 'Nordlicht' an Frau Musterfrau, Kastanienallee 12\n"
        "Pay to DE89 3704 0044 0532 0130 00 or anna.musterfrau@example.com\n"
        "Total EUR 1190.00"
    )
    server.reply_with_output(OUTPUT)
    client.post(URL, json={"document_text": text, "chart": "SKR03"}, headers=bearer(ALPHA))
    client.post(URL, json={"document_text": text, "chart": "SKR04"}, headers=bearer(ALPHA))
    logged = audit.path.read_text(encoding="utf-8")
    assert logged.count("\n") == 2
    for source in (text, pseudonymise(text)):
        for start in range(len(source) - 8):
            window = source[start : start + 9]
            assert not window.strip() or window not in logged, window


def test_a_rejected_key_is_recorded_without_the_key(client, audit):
    client.post(URL, json=BODY, headers=bearer("wrong-key-value"))
    (record,) = audit.records()
    assert record["outcome"] == "unauthorized" and record["key_id"] is None
    assert record["input_sha256"] is None and record["adapter"] is None
    assert "wrong-key-value" not in audit.path.read_text(encoding="utf-8")


def test_a_forbidden_platform_is_recorded_with_the_key_id(client, audit):
    client.post(URL, json=BODY, headers=bearer(BETA))
    (record,) = audit.records()
    assert record["outcome"] == "forbidden" and record["key_id"] == "beta"
    assert record["platform"] == "tax-booking"


def test_a_record_cannot_carry_a_field_the_spec_does_not_name():
    fields = {name: None for name in FIELDS} | {
        "timestamp": now(),
        "request_id": "r",
        "platform": "p",
        "outcome": "review",
        "review_reasons": [],
        "latency_ms": 1,
    }
    AuditRecord.model_validate(fields)
    with pytest.raises(ValueError):
        AuditRecord.model_validate({**fields, "document_text": "an invoice"})


def test_an_answer_is_withheld_when_the_log_cannot_be_written(server, tmp_path):
    broken = AuditLog(tmp_path)  # a folder, so the file cannot be opened
    server.reply_with_output(OUTPUT)
    with TestClient(create_app(load_serve_config(), KEYS, server.url, broken, timeout=2.0)) as c:
        response = c.post(URL, json=BODY, headers=bearer(ALPHA))
    assert response.status_code == 500
    assert "items" not in response.json() and "booking_lines" not in response.json()


def test_records_written_at_the_same_time_stay_whole_lines(audit):
    def write(number):
        audit.write(
            AuditRecord(
                timestamp=now(),
                request_id=f"r{number}",
                platform="tax-booking",
                key_id="alpha",
                input_sha256=None,
                adapter=None,
                adapter_sha256=None,
                base_revision=None,
                prompt_sha256=None,
                outcome="review",
                review_reasons=["low_confidence"] * 5,
                latency_ms=number,
            )
        )

    threads = [threading.Thread(target=write, args=(n,)) for n in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 50
    assert sorted(json.loads(line)["latency_ms"] for line in lines) == list(range(50))
