import copy
import json

import pytest
from jsonschema import Draft202012Validator

from adapt.config import load_platform
from adapt.data.gold import check_record, gold_path, main, validate_gold
from adapt.data.split import build_splits
from adapt.schema import build_output_schema


@pytest.fixture(scope="module")
def platform():
    return load_platform("tax-booking")


@pytest.fixture(scope="module")
def validator(platform):
    return Draft202012Validator(build_output_schema(platform.rules))


@pytest.fixture(scope="module")
def examples():
    lines = gold_path("tax-booking").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def problems_after(examples, validator, platform, change, index=0):
    record = copy.deepcopy(examples[index])
    change(record)
    return check_record(record, validator, platform)


def test_the_committed_examples_are_valid(platform):
    assert validate_gold(gold_path("tax-booking"), platform) == []


def test_the_examples_cover_a_credit_note_a_review_and_the_held_out_layout(examples):
    assert len(examples) == 3
    assert {r["target"]["document_type"] for r in examples} >= {"invoice", "credit_note"}
    assert any(r["target"]["needs_review"] for r in examples)
    assert any(r["layout"] == "letter_invoice" for r in examples)


def test_make_target_entry_point_passes_on_the_examples(capsys):
    assert main(["--platform", "tax-booking"]) == 0
    assert "3 records, all valid" in capsys.readouterr().out


def test_every_generated_target_passes_the_same_checks(platform, validator):
    splits = build_splits(platform, 0)
    for records in splits.values():
        for r in records:
            gold_like = {k: r[k] for k in ("id", "platform", "chart", "layout", "document_text")}
            gold_like["target"] = r["target"]
            expected = ["chart must be SKR03"] if r["category"] == "unsupported_chart" else []
            assert check_record(gold_like, validator, platform) == expected, r["id"]


def test_fails_when_the_items_do_not_add_up_to_the_total(examples, validator, platform):
    problems = problems_after(
        examples, validator, platform, lambda r: r["target"].update(total=85.41)
    )
    assert problems == ["the items add up to 85.40, but the total is 85.41"]


def test_fails_for_an_account_that_is_not_in_the_chart(examples, validator, platform):
    def change(record):
        record["target"]["items"][1]["account"] = "9999"

    problems = problems_after(examples, validator, platform, change)
    assert problems == ["item 2: account 9999 is not in the chart"]


def test_fails_for_a_missing_field_in_the_target(examples, validator, platform):
    def change(record):
        del record["target"]["reason"]

    problems = problems_after(examples, validator, platform, change)
    assert len(problems) == 1 and "reason" in problems[0]


def test_fails_for_an_unknown_vat_rate(examples, validator, platform):
    def change(record):
        record["target"]["items"][0]["vat_rate"] = "16"

    problems = problems_after(examples, validator, platform, change)
    assert len(problems) == 1 and problems[0].startswith("items.0.vat_rate")


def test_fails_for_the_wrong_sign_on_a_credit_note(examples, validator, platform):
    def change(record):
        record["target"]["items"][0]["amount"] = 59.5
        record["target"]["total"] = 59.5

    problems = problems_after(examples, validator, platform, change, index=1)
    assert problems == ["the sign of the total does not fit the document type"]


def test_fails_for_a_negative_total_on_an_invoice(examples, validator, platform):
    def change(record):
        record["target"]["items"] = [{**record["target"]["items"][0], "amount": -5.0}]
        record["target"]["total"] = -5.0

    problems = problems_after(examples, validator, platform, change)
    assert problems == ["the sign of the total does not fit the document type"]


def test_fails_for_no_items_without_a_review(examples, validator, platform):
    def change(record):
        record["target"]["needs_review"] = False

    problems = problems_after(examples, validator, platform, change, index=2)
    assert problems == ["a document without items must have needs_review set"]


def test_fails_for_a_review_without_a_reason(examples, validator, platform):
    def change(record):
        record["target"]["reason"] = " "

    problems = problems_after(examples, validator, platform, change, index=2)
    assert problems == ["a document that needs review must give a reason"]


def test_fails_for_a_wrong_chart_platform_layout_or_text(examples, validator, platform):
    def change(record):
        record.update(chart="DATEV", platform="other", layout="poster", document_text="")

    problems = problems_after(examples, validator, platform, change)
    assert problems == [
        "platform must be tax-booking",
        "chart must be SKR03",
        "unknown layout 'poster'",
        "document_text must be a non-empty string",
    ]


def test_fails_for_a_missing_or_unknown_field(examples, validator, platform):
    def change(record):
        del record["document_text"]
        record["note"] = "x"

    problems = problems_after(examples, validator, platform, change)
    assert problems == ["missing field document_text", "unknown field note"]


def test_the_layout_field_is_optional(examples, validator, platform):
    problems = problems_after(examples, validator, platform, lambda r: r.pop("layout"))
    assert problems == []


def test_file_check_names_the_line_and_the_id(tmp_path, examples, platform):
    broken = copy.deepcopy(examples)
    broken[1]["target"]["total"] = -1.0
    problems = validate_gold(write(tmp_path / "gold.jsonl", broken), platform)
    assert problems == [
        "line 2 (gold-example-02): the items add up to -59.50, but the total is -1.0"
    ]


def test_file_check_finds_bad_json_empty_lines_and_repeated_ids(tmp_path, examples, platform):
    path = tmp_path / "gold.jsonl"
    lines = [json.dumps(examples[0]), "{not json", "", json.dumps(examples[0])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    problems = validate_gold(path, platform)
    assert len(problems) == 3
    assert problems[0].startswith("line 2: not valid JSON")
    assert problems[1] == "line 3: empty line"
    assert problems[2] == "line 4 (gold-example-01): id already used on line 1"


def test_file_check_fails_for_a_missing_or_empty_file(tmp_path, platform):
    assert "does not exist" in validate_gold(tmp_path / "none.jsonl", platform)[0]
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert "has no records" in validate_gold(empty, platform)[0]


def test_entry_point_returns_one_and_prints_problems_for_a_broken_file(
    tmp_path, examples, platform, capsys
):
    broken = copy.deepcopy(examples)
    broken[0]["target"]["items"][0]["account"] = "9999"
    path = write(tmp_path / "gold.jsonl", broken)
    assert main(["--path", str(path)]) == 1
    err = capsys.readouterr().err
    assert "line 1 (gold-example-01): item 1: account 9999 is not in the chart" in err
    assert "1 problem(s)" in err
