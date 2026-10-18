import copy
import json
import math
from decimal import Decimal

import pytest

import adapt.pipeline as pipeline_module
from adapt.client import ModelClient, TokenLogprob
from adapt.config import load_platform
from adapt.data.split import build_splits
from adapt.pipeline import MAX_TEXT_LENGTH, Pipeline, account_confidences, document_problems
from fake_server import completion_body

TEXT = (
    "Nordpaper GmbH  Invoice 2026-1184  30.09.2026\n"
    "2x Printer paper A4 ... 23.80\n"
    "1x Tax handbook 2026 ... 21.40\n"
    "Net 19%: 20.00  VAT 19%: 3.80\n"
    "Net 7%: 20.00  VAT 7%: 1.40\n"
    "Total EUR 45.20"
)
OUTPUT = {
    "supplier": "Nordpaper GmbH",
    "document_date": "2026-09-30",
    "document_type": "invoice",
    "currency": "EUR",
    "total": 45.2,
    "items": [
        {"description": "Printer paper A4", "amount": 23.8, "account": "4930", "vat_rate": "19"},
        {"description": "Tax handbook 2026", "amount": 21.4, "account": "4940", "vat_rate": "7"},
    ],
    "reverse_charge": False,
    "needs_review": False,
    "reason": "Office supplies at 19%, book at 7%",
}
FLAGGED = {
    **OUTPUT,
    "items": [],
    "total": 0.0,
    "needs_review": True,
    "reason": "The amounts are illegible",
}


@pytest.fixture(scope="module")
def platform():
    return load_platform("tax-booking")


@pytest.fixture
def pipe(server, platform):
    with ModelClient(server.url, timeout=2.0) as client:
        yield Pipeline(platform, client, adapter="tax-booking", threshold=0.5, adapter_version="v1")


def output_with(**changes):
    return {**copy.deepcopy(OUTPUT), **changes}


def run(pipe, server, output=OUTPUT, text=TEXT, chart="SKR03", **reply):
    server.reply_with_output(output, **reply)
    return pipe.process(text, chart)


def test_a_clean_document_comes_out_automated(pipe, server):
    result = run(pipe, server)
    assert not result.needs_review
    assert result.reasons == []
    assert result.items == OUTPUT["items"]
    assert [(line.account, line.vat_rate) for line in result.booking_lines] == [
        ("4930", "19"),
        ("4940", "7"),
    ]
    assert result.booking_lines[0].net == Decimal("20.00")
    assert result.booking_lines[1].vat == Decimal("1.40")
    assert result.output == OUTPUT
    assert result.adapter == "tax-booking"
    assert result.adapter_version == "v1"
    assert result.request_id


def test_the_model_is_called_with_the_adapter_name_and_the_documented_settings(pipe, server):
    run(pipe, server)
    body = server.requests[0].body
    assert len(server.requests) == 1
    assert body["model"] == "tax-booking"
    assert body["temperature"] == 0
    assert body["logprobs"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][0]["role"] == "system"
    assert "SKR03" in body["messages"][0]["content"]
    assert body["messages"][1] == {"role": "user", "content": TEXT}


def test_personal_identifiers_are_replaced_before_the_model_sees_them(pipe, server):
    text = TEXT + "\nIBAN DE89 3704 0044 0532 0130 00  mail anna@example.com  Tel. 030 23125 123"
    run(pipe, server, text=text)
    sent = server.requests[0].body["messages"][1]["content"]
    assert "<IBAN>" in sent and "<EMAIL>" in sent and "<PHONE>" in sent
    assert "DE89" not in sent and "anna@example.com" not in sent and "23125" not in sent


def test_a_request_id_is_kept_or_made_up(pipe, server):
    server.reply_with_output(OUTPUT).reply_with_output(OUTPUT).reply_with_output(OUTPUT)
    assert pipe.process(TEXT, "SKR03", "abc").request_id == "abc"
    first, second = pipe.process(TEXT, "SKR03"), pipe.process(TEXT, "SKR03")
    assert first.request_id != second.request_id


@pytest.mark.parametrize(
    ("text", "chart", "reasons"),
    [
        ("", "SKR03", ["empty_text"]),
        ("   \n ", "SKR03", ["empty_text"]),
        (None, "SKR03", ["empty_text"]),
        (TEXT + "x" * MAX_TEXT_LENGTH, "SKR03", ["text_too_long"]),
        (TEXT, "SKR04", ["unsupported_chart"]),
        (TEXT, None, ["unsupported_chart"]),
        ("Nordpaper GmbH\nPaper 23.80", "SKR03", ["no_total"]),
        ("Nordpaper GmbH\nTotal EUR 4", "SKR03", ["no_total"]),
        ("Nordpaper GmbH\nTotal EUR 2?4.0?", "SKR03", ["no_total"]),
        ("Nordpaper GmbH\nPaper 23.80", "SKR04", ["unsupported_chart", "no_total"]),
    ],
)
def test_a_document_that_fails_the_first_checks_never_reaches_the_model(
    pipe, server, text, chart, reasons
):
    result = pipe.process(text, chart)
    assert result.needs_review
    assert result.reasons == reasons
    assert result.items == [] and result.booking_lines == []
    assert server.requests == []


def test_the_text_length_limit_is_inclusive(platform):
    text = TEXT + " " * (MAX_TEXT_LENGTH - len(TEXT))
    assert len(text) == MAX_TEXT_LENGTH
    assert document_problems(text, "SKR03", platform) == []
    assert document_problems(text + " ", "SKR03", platform) == ["text_too_long"]


@pytest.mark.parametrize(
    "line",
    [
        "Total EUR 45.20",
        "Total due: 1,533.13 EUR",
        "Grand total € 69,10",
        "Tota1 due: 412.06 EUR",
        "Total EUR 5O4,79",
        "Total: 22l.4SEUR",
        "Total credit: -59.50 EUR",
        "The total is EUR 975.80",
        "Amount due 85.40",
    ],
)
def test_totals_that_are_present_are_found(platform, line):
    assert document_problems(f"Supplier\n{line}", "SKR03", platform) == []


def test_the_model_being_unavailable_needs_review(pipe, server):
    server.reply(status=500).reply(status=500)
    result = pipe.process(TEXT, "SKR03")
    assert result.needs_review and result.reasons == ["model_unavailable"]
    assert len(server.requests) == 2


def test_a_slow_model_needs_review(server, platform):
    server.reply_with_output(OUTPUT, delay=1.0).reply_with_output(OUTPUT, delay=1.0)
    with ModelClient(server.url, timeout=0.2) as client:
        pipe = Pipeline(platform, client, adapter="tax-booking", threshold=0.5)
        result = pipe.process(TEXT, "SKR03")
    assert result.needs_review and result.reasons == ["model_unavailable"]


def test_invalid_json_needs_review(pipe, server):
    server.reply(completion_body("not json")).reply(completion_body("still not json"))
    result = pipe.process(TEXT, "SKR03")
    assert result.needs_review and result.reasons == ["invalid_output"]
    assert result.output is None and result.items == []


def test_an_answer_outside_the_schema_needs_review(pipe, server):
    broken = output_with(items=[{**OUTPUT["items"][0], "account": "9999"}])
    server.reply_with_output(broken).reply_with_output(broken)
    assert pipe.process(TEXT, "SKR03").reasons == ["invalid_output"]


def test_a_retry_that_works_gives_a_normal_result(pipe, server):
    server.reply(status=500).reply_with_output(OUTPUT)
    assert not pipe.process(TEXT, "SKR03").needs_review


def test_a_document_the_model_flagged_needs_review(pipe, server):
    result = run(pipe, server, output=FLAGGED)
    assert result.needs_review and result.reasons == ["model_flagged"]
    assert result.items == [] and result.confidence is None
    assert result.item_confidences is None
    assert result.output == FLAGGED


def test_a_flagged_answer_with_items_still_gets_booking_lines_and_stays_in_review(pipe, server):
    result = run(pipe, server, output=output_with(needs_review=True, reason="unsure"))
    assert result.needs_review and result.reasons == ["model_flagged"]
    assert len(result.booking_lines) == 2


def test_an_answer_with_no_items_that_is_not_flagged_needs_review(pipe, server):
    result = run(pipe, server, output=output_with(items=[], total=0.0))
    assert result.needs_review and result.reasons == ["no_items"]


def test_a_total_that_does_not_match_needs_review(pipe, server):
    result = run(pipe, server, output=output_with(total=50.0))
    assert result.needs_review and result.reasons == ["total_mismatch"]
    assert len(result.booking_lines) == 2


def test_a_net_amount_near_the_asset_limit_needs_review(pipe, server):
    item = {"description": "Laptop", "amount": 975.8, "account": "0410", "vat_rate": "19"}
    result = run(pipe, server, output=output_with(items=[item], total=975.8))
    assert result.needs_review and result.reasons == ["near_asset_limit"]


def test_an_asset_booked_as_expense_needs_review(pipe, server):
    item = {"description": "Desk", "amount": 1785.0, "account": "4930", "vat_rate": "19"}
    result = run(pipe, server, output=output_with(items=[item], total=1785.0))
    assert result.needs_review and result.reasons == ["asset_as_expense"]


def test_a_credit_note_comes_out_automated(pipe, server):
    item = {"description": "Toner, returned", "amount": -59.5, "account": "4930", "vat_rate": "19"}
    output = output_with(items=[item], total=-59.5, document_type="credit_note")
    result = run(pipe, server, output=output)
    assert not result.needs_review
    assert result.booking_lines[0].net == Decimal("-50.00")


def test_a_reverse_charge_document_gets_a_liability(pipe, server):
    item = {"description": "Hosting", "amount": 100.0, "account": "4964", "vat_rate": "19"}
    output = output_with(items=[item], total=100.0, reverse_charge=True)
    result = run(pipe, server, output=output)
    assert not result.needs_review
    assert result.booking_lines[0].liability == Decimal("19.00")


def test_confidence_is_the_probability_of_the_least_certain_account(pipe, server):
    logprobs = {"4930": math.log(0.9), "4940": math.log(0.6)}
    result = run(pipe, server, logprob_for=lambda token: logprobs.get(token, -0.001))
    assert result.confidence == pytest.approx(0.6)
    assert result.item_confidences == [pytest.approx(0.9), pytest.approx(0.6)]
    assert not result.needs_review


def test_a_confidence_below_the_threshold_needs_review(pipe, server):
    logprobs = {"4930": math.log(0.9), "4940": math.log(0.4)}
    result = run(pipe, server, logprob_for=lambda token: logprobs.get(token, -0.001))
    assert result.needs_review and result.reasons == ["low_confidence"]
    assert result.confidence == pytest.approx(0.4)
    assert len(result.booking_lines) == 2


def test_a_confidence_equal_to_the_threshold_is_accepted(server, platform):
    def logprob_for(token):
        return math.log(0.7) if token == "4930" else -0.1

    results = []
    with ModelClient(server.url, timeout=2.0) as client:
        probe = Pipeline(platform, client, adapter="tax-booking", threshold=0.0)
        confidence = run(probe, server, logprob_for=logprob_for).confidence
        for threshold in (confidence, math.nextafter(confidence, 1.0)):
            pipe = Pipeline(platform, client, adapter="tax-booking", threshold=threshold)
            results.append(run(pipe, server, logprob_for=logprob_for))
    at, above = results
    assert not at.needs_review
    assert above.needs_review and above.reasons == ["low_confidence"]


def test_without_logprobs_the_confidence_is_unknown_and_the_document_needs_review(pipe, server):
    result = run(pipe, server, with_logprobs=False)
    assert result.needs_review and result.reasons == ["no_confidence"]
    assert result.confidence is None


def test_tokens_that_do_not_add_up_to_the_answer_need_review(pipe, server):
    body = completion_body(json.dumps(OUTPUT))
    body["choices"][0]["logprobs"]["content"][3]["token"] = "changed"
    server.reply(body)
    result = pipe.process(TEXT, "SKR03")
    assert result.needs_review and result.reasons == ["no_confidence"]


def test_an_exception_anywhere_ends_in_review(pipe, server, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline_module, "compute_booking", broken)
    result = run(pipe, server)
    assert result.needs_review and result.reasons == ["internal_error"]
    assert result.items == [] and result.output is None


def test_an_exception_in_the_client_ends_in_review(pipe, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk")

    monkeypatch.setattr(pipe.client, "complete", broken)
    result = pipe.process(TEXT, "SKR03")
    assert result.needs_review and result.reasons == ["internal_error"]


def test_an_exception_in_the_document_checks_ends_in_review(pipe, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("bad")

    monkeypatch.setattr(pipeline_module, "document_problems", broken)
    assert pipe.process(TEXT, "SKR03").reasons == ["internal_error"]


def test_no_failure_path_is_automated(pipe, server):
    """Every scenario above that is not the clean one ends in review with a reason."""
    scenarios = [
        dict(text="", chart="SKR03"),
        dict(text=TEXT, chart="SKR04"),
        dict(text="Paper 23.80", chart="SKR03"),
        dict(output=FLAGGED),
        dict(output=output_with(total=1.0)),
        dict(output=output_with(items=[], total=0.0)),
        dict(with_logprobs=False),
    ]
    for scenario in scenarios:
        server.reset()
        text = scenario.pop("text", TEXT)
        chart = scenario.pop("chart", "SKR03")
        output = scenario.pop("output", OUTPUT)
        server.reply_with_output(output, **scenario)
        result = pipe.process(text, chart)
        assert result.needs_review and result.reasons, scenario


def test_account_confidence_sums_the_tokens_of_each_account_value():
    content = '{"items":[{"account":"4930"},{"account":"4940"}]}'
    pieces = ['{"items":[{"account":"', "4", "9", "3", "0", '"},{"account":"', "49", "40", '"}]}']
    logprobs = [-9.0, -0.1, -0.2, -0.3, -0.4, -9.0, -0.5, -0.25, -9.0]
    tokens = [TokenLogprob(p, lp) for p, lp in zip(pieces, logprobs, strict=True)]
    first, second = account_confidences(content, tokens, 2)
    assert first == pytest.approx(math.exp(-1.0))
    assert second == pytest.approx(math.exp(-0.75))


def test_a_token_that_covers_a_quote_and_a_digit_counts_for_that_account():
    content = '{"account":"4930"}'
    pieces = ['{"account":', '"4', "9", "3", "0", '"}']
    tokens = [TokenLogprob(p, -0.5) for p in pieces]
    assert account_confidences(content, tokens, 1) == [pytest.approx(math.exp(-2.0))]


def test_account_confidence_gives_none_when_it_cannot_be_matched():
    content = '{"account":"4930"}'
    tokens = [TokenLogprob(content, -0.1)]
    assert account_confidences(content, [], 1) is None
    assert account_confidences(content, tokens, 2) is None  # the item count differs
    assert account_confidences(content, [TokenLogprob("other", -0.1)], 1) is None


def test_tokens_after_the_end_of_the_text_are_ignored():
    """vLLM can list the end-of-text token after the last character of the answer."""
    content = '{"account":"4930"}'
    pieces = ['{"account":"', "4930", '"}']
    tokens = [TokenLogprob(p, -0.5) for p in pieces]
    plain = account_confidences(content, tokens, 1)
    for end in ["<|im_end|>", ""]:
        with_end = account_confidences(content, [*tokens, TokenLogprob(end, -0.01)], 1)
        assert with_end == plain == [pytest.approx(math.exp(-0.5))]


def test_account_confidence_gives_none_when_the_text_is_longer_than_the_tokens():
    content = '{"account":"4930"}'
    tokens = [TokenLogprob('{"account":"', -0.5), TokenLogprob("4930", -0.5)]
    assert account_confidences(content, tokens, 1) is None


def test_an_account_named_inside_a_description_is_not_counted():
    content = json.dumps({"items": [{"description": 'say "account": "1"', "account": "4930"}]})
    tokens = [TokenLogprob(content, -0.1)]
    assert len(account_confidences(content, tokens, 1)) == 1


def test_documents_from_the_dev_split_end_as_the_data_says(pipe, server, platform):
    """Send each dev target back as the model answer. It is automated if no review is expected."""
    records = build_splits(platform, 0)["dev"]
    for r in records:
        server.reset()
        server.reply_with_output(r["target"])
        result = pipe.process(r["document_text"], r["chart"])
        assert result.needs_review == r["expected_review"], (r["id"], result.reasons)
        if result.needs_review:
            assert result.reasons, r["id"]
    assert len(records) > 250
