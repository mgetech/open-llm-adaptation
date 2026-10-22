"""The gateway against the fake model server: keys, platforms, the answer, and failing closed."""

import json

import pytest
import yaml
from fastapi.testclient import TestClient

from adapt.eval.runner import RESULTS_ROOT
from adapt.gateway import __main__ as gateway_main
from adapt.gateway.app import MAX_BODY_CHARACTERS, create_app
from adapt.gateway.keys import (
    EXAMPLE_FILE,
    KeyFile,
    find_key,
    hash_key,
    load_keys,
)
from adapt.gateway.keys import (
    main as keys_main,
)
from adapt.pipeline import MAX_TEXT_LENGTH
from adapt.serve import load_serve_config
from test_pipeline import OUTPUT, TEXT

ALPHA, BETA = "alpha-secret", "beta-secret"
KEYS = KeyFile.model_validate(
    {
        "keys": [
            {"id": "alpha", "sha256": hash_key(ALPHA), "platforms": ["tax-booking"]},
            {"id": "beta", "sha256": hash_key(BETA), "platforms": ["other"]},
        ]
    }
)
URL = "/v1/tax-booking/process"
BODY = {"document_text": TEXT, "chart": "SKR03"}


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def client(server):
    with TestClient(create_app(load_serve_config(), KEYS, server.url, timeout=2.0)) as client:
        yield client


def test_a_valid_request_returns_the_items_and_the_booking_lines(client, server):
    server.reply_with_output(OUTPUT)
    response = client.post(URL, json=BODY, headers=bearer(ALPHA))
    assert response.status_code == 200
    data = response.json()
    assert data["needs_review"] is False and data["reasons"] == []
    assert data["items"] == OUTPUT["items"]
    assert [
        (line["account"], line["vat_rate"], line["net"], line["vat"])
        for line in data["booking_lines"]
    ] == [("4930", "19", "20.00", "3.80"), ("4940", "7", "20.00", "1.40")]
    assert data["document"]["supplier"] == "Nordpaper GmbH" and data["document"]["total"] == 45.2
    assert data["adapter"] == "tax-booking"
    assert data["adapter_version"] == load_serve_config().adapters[0].sha256
    assert data["request_id"]


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Bearer "}, {"Authorization": ALPHA}],
)
def test_a_missing_or_wrong_key_gives_401_and_the_model_is_not_called(client, server, headers):
    response = client.post(URL, json=BODY, headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert server.requests == []


def test_a_different_scheme_with_the_right_key_gives_401(client):
    response = client.post(URL, json=BODY, headers={"Authorization": f"Basic {ALPHA}"})
    assert response.status_code == 401


def test_the_key_is_checked_before_the_body(client):
    response = client.post(URL, json={"unexpected": 1}, headers=bearer("wrong"))
    assert response.status_code == 401


def test_a_key_without_access_to_the_platform_gives_403(client, server):
    response = client.post(URL, json=BODY, headers=bearer(BETA))
    assert response.status_code == 403
    assert server.requests == []


def test_an_unknown_platform_gives_the_same_403_as_a_forbidden_one(client):
    forbidden = client.post(URL, json=BODY, headers=bearer(BETA))
    unknown = client.post("/v1/nothing/process", json=BODY, headers=bearer(ALPHA))
    assert unknown.status_code == 403
    assert unknown.json() == forbidden.json()


def test_a_request_cannot_choose_the_adapter(client, server):
    server.reply_with_output(OUTPUT)
    response = client.post(URL, json={**BODY, "model": "other"}, headers=bearer(ALPHA))
    assert response.status_code == 200
    assert server.requests[0].body["model"] == "tax-booking"


def test_the_request_id_of_the_caller_is_used_when_it_is_plain(client, server):
    server.reply_with_output(OUTPUT)
    response = client.post(URL, json=BODY, headers={**bearer(ALPHA), "X-Request-ID": "order-17"})
    assert response.json()["request_id"] == "order-17"


def test_a_request_id_with_odd_characters_is_replaced(client, server):
    server.reply_with_output(OUTPUT)
    headers = {**bearer(ALPHA), "X-Request-ID": "a b\tc"}
    request_id = client.post(URL, json=BODY, headers=headers).json()["request_id"]
    assert request_id != "a b\tc" and len(request_id) == 32


def test_a_model_that_is_down_ends_in_review_never_in_a_booking(client, server):
    response = client.post(URL, json=BODY, headers=bearer(ALPHA))
    data = response.json()
    assert response.status_code == 200
    assert data["needs_review"] is True and "model_unavailable" in data["reasons"]
    assert data["booking_lines"] == []


def test_an_unsupported_chart_is_reviewed_without_calling_the_model(client, server):
    data = client.post(URL, json={**BODY, "chart": "SKR04"}, headers=bearer(ALPHA)).json()
    assert data["needs_review"] is True and "unsupported_chart" in data["reasons"]
    assert server.requests == []


def test_a_text_that_is_too_long_for_the_pipeline_is_reviewed(client, server):
    text = "x" * (MAX_TEXT_LENGTH + 1)
    data = client.post(
        URL, json={"document_text": text, "chart": "SKR03"}, headers=bearer(ALPHA)
    ).json()
    assert data["needs_review"] is True and "text_too_long" in data["reasons"]
    assert server.requests == []


def test_a_body_over_the_limit_is_refused(client):
    text = "x" * (MAX_BODY_CHARACTERS + 1)
    response = client.post(
        URL, json={"document_text": text, "chart": "SKR03"}, headers=bearer(ALPHA)
    )
    assert response.status_code == 422


def test_a_body_without_the_document_is_refused(client):
    assert client.post(URL, json={"chart": "SKR03"}, headers=bearer(ALPHA)).status_code == 422


def test_there_is_no_chat_endpoint_and_no_docs(client):
    for path in ("/v1/chat/completions", "/docs", "/openapi.json"):
        assert client.post(path, json={}, headers=bearer(ALPHA)).status_code in (404, 405)


def test_the_threshold_of_the_serve_file_is_the_one_chosen_on_dev_and_dev_real():
    saved = json.loads(
        (
            RESULTS_ROOT / "round3" / "rescored" / "tax-booking" / "qwen3-4b-sft-r3-0" / "dev.json"
        ).read_text("utf-8")
    )
    assert load_serve_config().adapters[0].threshold == saved["confidence_threshold"]["value"]


def test_the_example_key_file_loads_and_holds_the_documented_key():
    keys = load_keys(EXAMPLE_FILE)
    assert find_key(keys, "example-key-do-not-use").id == "example"


def test_a_key_is_found_by_its_hash_only():
    assert find_key(KEYS, ALPHA).id == "alpha"
    assert find_key(KEYS, hash_key(ALPHA)) is None
    assert find_key(KEYS, "") is None and find_key(KEYS, None) is None


def test_a_missing_key_file_says_what_to_copy(tmp_path):
    with pytest.raises(ValueError, match="copy keys.example.yaml"):
        load_keys(tmp_path / "keys.yaml")


def test_a_key_file_with_the_same_id_twice_is_refused(tmp_path):
    entry = {"id": "a", "sha256": "0" * 64, "platforms": ["tax-booking"]}
    path = tmp_path / "keys.yaml"
    path.write_text(yaml.safe_dump({"keys": [entry, {**entry, "sha256": "1" * 64}]}), "utf-8")
    with pytest.raises(ValueError, match="id is used twice"):
        load_keys(path)


def test_a_key_file_with_a_plain_key_in_it_is_refused(tmp_path):
    entry = {"id": "a", "sha256": "not-a-hash", "platforms": ["tax-booking"]}
    path = tmp_path / "keys.yaml"
    path.write_text(yaml.safe_dump({"keys": [entry]}), "utf-8")
    with pytest.raises(ValueError, match="keys.yaml is not valid"):
        load_keys(path)


def test_a_new_key_is_printed_once_with_the_line_for_the_file(capsys):
    assert keys_main(["--id", "partner", "--platforms", "tax-booking"]) == 0
    lines = capsys.readouterr().out.splitlines()
    key = lines[0].removeprefix("key (shown once): ")
    entry = yaml.safe_load("\n".join(lines[2:]))[0]
    assert entry["id"] == "partner" and entry["sha256"] == hash_key(key)


def test_the_gateway_stops_with_a_message_when_there_is_no_key_file(tmp_path):
    with pytest.raises(SystemExit, match="copy keys.example.yaml"):
        gateway_main.main(["--keys-file", str(tmp_path / "keys.yaml")])
