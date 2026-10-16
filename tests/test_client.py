import json
import socket

import pytest

from adapt.client import ModelClient, ModelError
from adapt.config import load_platform
from adapt.schema import build_output_schema
from fake_server import completion_body, split_tokens

MESSAGES = [
    {"role": "system", "content": "You book invoices."},
    {"role": "user", "content": "Paper 23.80"},
]
OUTPUT = {
    "supplier": "Nordpaper GmbH",
    "document_date": "2026-09-30",
    "document_type": "invoice",
    "currency": "EUR",
    "total": 23.8,
    "items": [{"description": "Paper", "amount": 23.8, "account": "4930", "vat_rate": "19"}],
    "reverse_charge": False,
    "needs_review": False,
    "reason": "Office supplies at 19%",
}


@pytest.fixture(scope="module")
def schema():
    return build_output_schema(load_platform("tax-booking").rules)


@pytest.fixture
def client(server):
    with ModelClient(server.url, timeout=2.0) as client:
        yield client


def test_the_fake_server_pieces_join_back_to_the_text():
    text = json.dumps(OUTPUT)
    pieces = split_tokens(text)
    assert "".join(pieces) == text
    assert "4930" in pieces


def test_a_valid_answer_is_parsed(server, client, schema):
    server.reply_with_output(OUTPUT)
    result = client.complete("tax-booking", MESSAGES, schema)
    assert result.output == OUTPUT
    assert json.loads(result.content) == OUTPUT
    assert len(server.requests) == 1


def test_the_request_carries_the_documented_parameters(server, client, schema):
    server.reply_with_output(OUTPUT)
    client.complete("tax-booking", MESSAGES, schema)
    request = server.requests[0]
    assert request.path == "/v1/chat/completions"
    body = request.body
    assert body["model"] == "tax-booking"
    assert body["messages"] == MESSAGES
    assert body["temperature"] == 0
    assert body["logprobs"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == schema


def test_an_api_key_is_sent_as_a_bearer_token(server, schema):
    server.reply_with_output(OUTPUT)
    with ModelClient(server.url, api_key="secret") as client:
        client.complete("tax-booking", MESSAGES, schema)
    assert server.requests[0].headers["authorization"] == "Bearer secret"


def test_no_authorization_header_without_a_key(server, client, schema):
    server.reply_with_output(OUTPUT)
    client.complete("tax-booking", MESSAGES, schema)
    assert "authorization" not in server.requests[0].headers


def test_token_logprobs_are_returned_in_order(server, client, schema):
    server.reply_with_output(OUTPUT, logprob_for=lambda token: -2.0 if token == "4930" else -0.1)
    result = client.complete("tax-booking", MESSAGES, schema)
    assert "".join(t.token for t in result.tokens) == result.content
    assert [t.logprob for t in result.tokens if t.token == "4930"] == [-2.0]
    assert all(isinstance(t.logprob, float) for t in result.tokens)


def test_an_answer_without_logprobs_gives_no_tokens(server, client, schema):
    server.reply_with_output(OUTPUT, with_logprobs=False)
    assert client.complete("tax-booking", MESSAGES, schema).tokens == []


def test_a_server_error_is_retried_once(server, client, schema):
    server.reply(status=500, body={"error": "busy"}).reply_with_output(OUTPUT)
    assert client.complete("tax-booking", MESSAGES, schema).output == OUTPUT
    assert len(server.requests) == 2


def test_invalid_json_is_retried_once(server, client, schema):
    server.reply(completion_body("{not json")).reply_with_output(OUTPUT)
    assert client.complete("tax-booking", MESSAGES, schema).output == OUTPUT
    assert len(server.requests) == 2


def test_a_timeout_is_retried_once(server, schema):
    server.reply_with_output(OUTPUT, delay=1.0).reply_with_output(OUTPUT)
    with ModelClient(server.url, timeout=0.2) as client:
        assert client.complete("tax-booking", MESSAGES, schema).output == OUTPUT
    server.wait_for_requests(2)
    assert len(server.requests) == 2


def test_two_timeouts_raise_an_unavailable_error(server, schema):
    server.reply_with_output(OUTPUT, delay=1.0).reply_with_output(OUTPUT, delay=1.0)
    with ModelClient(server.url, timeout=0.2) as client, pytest.raises(ModelError) as caught:
        client.complete("tax-booking", MESSAGES, schema)
    assert caught.value.kind == "unavailable"
    assert "in time" in str(caught.value)
    server.wait_for_requests(2)
    assert len(server.requests) == 2


def test_two_server_errors_raise_an_unavailable_error(server, client, schema):
    server.reply(status=503, body="down").reply(status=500, body="down")
    with pytest.raises(ModelError) as caught:
        client.complete("tax-booking", MESSAGES, schema)
    assert caught.value.kind == "unavailable"
    assert "500" in str(caught.value)
    assert len(server.requests) == 2


def test_a_dropped_connection_is_retried_and_then_raises_an_unavailable_error(
    server, client, schema
):
    server.drop().drop()
    with pytest.raises(ModelError) as caught:
        client.complete("tax-booking", MESSAGES, schema)
    assert caught.value.kind == "unavailable"
    assert len(server.requests) == 2


def test_a_dropped_connection_followed_by_a_good_answer_succeeds(server, client, schema):
    server.drop().reply_with_output(OUTPUT)
    assert client.complete("tax-booking", MESSAGES, schema).output == OUTPUT


def test_a_server_that_is_not_running_raises_an_unavailable_error(schema):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # No retry here: a refused connection takes a couple of seconds on Windows.
    with (
        ModelClient(f"http://127.0.0.1:{port}/v1", timeout=1.0, retries=0) as client,
        pytest.raises(ModelError) as caught,
    ):
        client.complete("tax-booking", MESSAGES, schema)
    assert caught.value.kind == "unavailable"


def test_invalid_json_twice_raises_an_invalid_output_error(server, client, schema):
    server.reply(completion_body("{not json")).reply(completion_body("Sure, here it is"))
    with pytest.raises(ModelError) as caught:
        client.complete("tax-booking", MESSAGES, schema)
    assert caught.value.kind == "invalid_output"
    assert "not valid JSON" in str(caught.value)
    assert len(server.requests) == 2


def test_json_that_is_not_an_object_is_invalid(server, client, schema):
    server.reply(completion_body("[1, 2]")).reply(completion_body("[1, 2]"))
    with pytest.raises(ModelError, match="not a JSON object"):
        client.complete("tax-booking", MESSAGES, schema)


def test_a_missing_field_is_invalid(server, client, schema):
    broken = {k: v for k, v in OUTPUT.items() if k != "reason"}
    server.reply_with_output(broken).reply_with_output(broken)
    with pytest.raises(ModelError) as caught:
        client.complete("tax-booking", MESSAGES, schema)
    assert caught.value.kind == "invalid_output"
    assert "reason" in str(caught.value)


def test_an_account_outside_the_chart_is_invalid(server, client, schema):
    broken = {**OUTPUT, "items": [{**OUTPUT["items"][0], "account": "9999"}]}
    server.reply_with_output(broken).reply_with_output(broken)
    with pytest.raises(ModelError, match="does not fit the schema"):
        client.complete("tax-booking", MESSAGES, schema)


def test_an_answer_that_was_cut_off_is_invalid(server, client, schema):
    cut = completion_body(json.dumps(OUTPUT), finish_reason="length")
    server.reply(cut).reply(cut)
    with pytest.raises(ModelError, match="cut off"):
        client.complete("tax-booking", MESSAGES, schema)


@pytest.mark.parametrize(
    "body",
    [{}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": {"content": None}}]}, "x"],
)
def test_an_unexpected_response_shape_is_invalid(server, client, schema, body):
    server.reply(body).reply(body)
    with pytest.raises(ModelError) as caught:
        client.complete("tax-booking", MESSAGES, schema)
    assert caught.value.kind == "invalid_output"


def test_malformed_logprobs_are_invalid(server, client, schema):
    body = completion_body(json.dumps(OUTPUT))
    body["choices"][0]["logprobs"] = {"content": [{"token": "x"}]}
    server.reply(body).reply(body)
    with pytest.raises(ModelError, match="logprobs"):
        client.complete("tax-booking", MESSAGES, schema)


def test_without_retries_a_failure_is_not_repeated(server, schema):
    server.reply(status=500).reply_with_output(OUTPUT)
    with ModelClient(server.url, retries=0) as client, pytest.raises(ModelError):
        client.complete("tax-booking", MESSAGES, schema)
    assert len(server.requests) == 1


def test_negative_retries_are_refused(server):
    with pytest.raises(ValueError, match="retries"):
        ModelClient(server.url, retries=-1)


def test_the_default_timeout_is_thirty_seconds_and_one_retry():
    with ModelClient("http://localhost:1/v1") as client:
        assert client._timeout == 30.0
        assert client._retries == 1
