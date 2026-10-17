import hashlib
import json
import re
from decimal import Decimal

import pytest
from jsonschema import Draft202012Validator

from adapt.booking import compute_booking
from adapt.client import ModelClient
from adapt.config import load_platform
from adapt.data.gold import gold_path
from adapt.data.pseudonymise import pseudonymise
from adapt.eval.runner import DATA_DIR, RunSettings, evaluate, load_canaries, load_documents
from adapt.pipeline import Pipeline, document_problems
from adapt.prompting import build_messages, system_prompt
from adapt.schema import MAX_REASON_LENGTH, build_output_schema
from fake_model import Model

PLATFORM = load_platform("tax-booking")
EXAMPLE = re.compile(r"Document:\n(.*?)\nAnswer:\n(\{[^\n]*\})\n", re.DOTALL)


@pytest.fixture(scope="module")
def zero_shot() -> str:
    return system_prompt("tax-booking", "SKR03", PLATFORM.rules)


@pytest.fixture(scope="module")
def few_shot() -> str:
    return system_prompt("tax-booking", "SKR03", PLATFORM.rules, few_shot=True)


@pytest.fixture(scope="module")
def examples(few_shot) -> list[tuple[str, dict]]:
    return [(text, json.loads(answer)) for text, answer in EXAMPLE.findall(few_shot)]


def test_the_prompt_is_zero_shot_unless_examples_are_asked_for(zero_shot):
    assert zero_shot == system_prompt("tax-booking", "SKR03", PLATFORM.rules, few_shot=False)
    assert "Examples" not in zero_shot and "Answer:" not in zero_shot
    assert EXAMPLE.findall(zero_shot) == []


def test_the_few_shot_prompt_is_the_zero_shot_prompt_plus_the_examples(zero_shot, few_shot):
    assert few_shot.startswith(zero_shot)
    assert few_shot.count("Document:") == 3 and few_shot.count("Answer:") == 3


def test_there_are_three_examples_in_the_order_simple_mixed_and_review(examples):
    assert len(examples) == 3
    simple, mixed, review = (answer for _, answer in examples)
    assert not simple["needs_review"] and len({i["vat_rate"] for i in simple["items"]}) == 1
    assert not mixed["needs_review"] and len({i["vat_rate"] for i in mixed["items"]}) == 2
    assert review["needs_review"] and review["items"] == []


def test_every_example_answer_follows_the_output_schema(examples):
    validator = Draft202012Validator(build_output_schema(PLATFORM.rules))
    for _, answer in examples:
        assert list(validator.iter_errors(answer)) == [], answer["supplier"]
        assert len(answer["reason"]) <= MAX_REASON_LENGTH


def test_the_answers_copy_the_document_and_book_without_a_review(examples):
    for text, answer in examples[:2]:
        assert document_problems(text, "SKR03", PLATFORM) == []
        for item in answer["items"]:
            # The amount is printed in the document, with a point or with a comma.
            amount = f"{item['amount']:.2f}"
            assert amount in text or amount.replace(".", ",") in text, item
        assert sum(Decimal(str(i["amount"])) for i in answer["items"]) == Decimal(
            str(answer["total"])
        )
        assert compute_booking(answer, "SKR03", PLATFORM).reasons == []


def test_the_mixed_rates_example_splits_the_vat_as_printed(examples):
    text, answer = examples[1]
    lines = {line.vat_rate: line for line in compute_booking(answer, "SKR03", PLATFORM).lines}
    assert (lines["19"].net, lines["19"].vat) == (Decimal("40.00"), Decimal("7.60"))
    assert (lines["7"].net, lines["7"].vat) == (Decimal("20.00"), Decimal("1.40"))
    assert "incl. VAT 19%: 7.60" in text and "incl. VAT 7%: 1.40" in text


def test_the_review_example_gives_a_reason_and_the_total_as_printed(examples):
    text, answer = examples[2]
    assert document_problems(text, "SKR03", PLATFORM) == []
    assert answer["total"] == 66.1 and "66.10" in text
    assert answer["reason"] and answer["reverse_charge"] is False


def test_the_examples_are_the_same_text_the_model_gets_for_a_real_document(examples):
    for text, _ in examples:
        assert pseudonymise(text) == text


def test_the_examples_are_not_documents_or_suppliers_of_any_split(examples):
    files = [*DATA_DIR.joinpath("tax-booking").glob("*.jsonl"), gold_path("tax-booking")]
    records = [
        json.loads(line) for path in files for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) > 4000
    texts = [r["document_text"].lower() for r in records]
    suppliers = {r["target"]["supplier"].lower() for r in records}
    for text, answer in examples:
        assert answer["supplier"].lower() not in suppliers
        assert text.lower() not in texts
        # The first line of an example is its supplier name. No document mentions it.
        name = text.splitlines()[0].lower()
        assert not any(name in other for other in texts), name


def test_the_examples_have_different_suppliers_and_totals(examples):
    assert len({answer["supplier"] for _, answer in examples}) == 3
    assert len({answer["total"] for _, answer in examples}) == 3


def test_the_examples_come_after_the_rules_and_the_accounts(zero_shot, few_shot):
    assert few_shot.index("Rules:") < few_shot.index("Accounts:") < few_shot.index("Examples")
    for account in PLATFORM.rules.accounts:
        assert few_shot.index(f"- {account.number} {account.name}") < few_shot.index("Examples")


def test_the_few_shot_messages_keep_the_document_as_the_user_message(few_shot):
    messages = build_messages(few_shot, "Some document")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "Some document"


@pytest.fixture
def pipeline_requests(server):
    def messages_sent(few_shot):
        server.reset()
        server.reply_with_output(
            {
                "supplier": "S",
                "document_date": "2026-01-01",
                "document_type": "invoice",
                "currency": "EUR",
                "total": 59.5,
                "items": [],
                "reverse_charge": False,
                "needs_review": True,
                "reason": "x",
            }
        )
        with ModelClient(server.url, timeout=2.0) as client:
            pipeline = Pipeline(PLATFORM, client, adapter="m", threshold=0.0, few_shot=few_shot)
            pipeline.process("Total due: 59,50 EUR", "SKR03")
        return server.requests[-1].body["messages"]

    return messages_sent


def test_the_pipeline_sends_the_examples_only_when_asked(pipeline_requests, zero_shot, few_shot):
    assert pipeline_requests(False)[0]["content"] == zero_shot
    assert pipeline_requests(True)[0]["content"] == few_shot


def test_the_runner_records_and_uses_the_few_shot_prompt(server, few_shot, zero_shot, tmp_path):
    def run(**changes):
        server.reset()
        server.handler = Model(load_documents("tax-booking", "dev"), load_canaries("tax-booking"))
        values = dict(
            platform="tax-booking",
            system="demo",
            split="dev",
            endpoint=server.url,
            model="tax-booking",
            workers=4,
            threshold=0.0,
            out_dir=tmp_path,
        )
        results = evaluate(RunSettings(**{**values, **changes}))
        return results["manifest"], {r.body["messages"][0]["content"] for r in server.requests}

    manifest, prompts = run(few_shot=True)
    assert manifest["few_shot"] is True
    assert manifest["prompt_sha256"] == hashlib.sha256(few_shot.encode()).hexdigest()
    assert prompts == {few_shot}

    manifest, prompts = run()
    assert manifest["few_shot"] is False
    assert manifest["prompt_sha256"] == hashlib.sha256(zero_shot.encode()).hexdigest()
    assert prompts == {zero_shot}


def test_the_command_line_has_a_few_shot_flag(monkeypatch, tmp_path, capsys):
    from adapt.eval import runner

    seen = []
    monkeypatch.setattr(
        runner,
        "evaluate",
        lambda settings: (
            seen.append(settings)
            or {
                "documents": [],
                "confidence_threshold": {"value": 0.0, "source": "test"},
                "metrics": {},
                "performance": {},
            }
        ),
    )
    monkeypatch.setattr(runner, "write_results", lambda results, path: None)
    monkeypatch.setattr(runner, "_summary", lambda results: "")
    runner.main(["--system", "a", "--out-dir", str(tmp_path)])
    runner.main(["--system", "a", "--few-shot", "--out-dir", str(tmp_path)])
    assert [s.few_shot for s in seen] == [False, True]
