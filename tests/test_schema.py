import copy
import json

import pytest
from jsonschema import Draft202012Validator

from adapt.config import load_platform
from adapt.schema import build_output_schema

EXAMPLE = {
    "supplier": "Nordpaper GmbH",
    "document_date": "2026-09-30",
    "document_type": "invoice",
    "currency": "EUR",
    "total": 45.20,
    "items": [
        {"description": "Printer paper A4", "amount": 23.80, "account": "4930", "vat_rate": "19"},
        {"description": "Tax handbook 2026", "amount": 21.40, "account": "4940", "vat_rate": "7"},
    ],
    "reverse_charge": False,
    "needs_review": False,
    "reason": "Office supplies at 19%, book at 7%",
}


@pytest.fixture(scope="module")
def rules():
    return load_platform("tax-booking").rules


@pytest.fixture(scope="module")
def validator(rules):
    return Draft202012Validator(build_output_schema(rules))


@pytest.fixture
def example():
    return copy.deepcopy(EXAMPLE)


def test_schema_is_valid_json_schema(rules):
    schema = build_output_schema(rules)
    Draft202012Validator.check_schema(schema)
    assert json.loads(json.dumps(schema)) == schema


def test_example_from_the_task_is_valid(validator, example):
    assert validator.is_valid(example)


def test_account_enum_comes_from_the_rules(rules):
    schema = build_output_schema(rules)
    enum = schema["properties"]["items"]["items"]["properties"]["account"]["enum"]
    assert enum == [account.number for account in rules.accounts]


def test_account_enum_follows_changed_rules(rules):
    smaller = rules.model_copy(update={"accounts": rules.accounts[:3]})
    schema = build_output_schema(smaller)
    enum = schema["properties"]["items"]["items"]["properties"]["account"]["enum"]
    assert enum == ["0410", "0480", "0650"]


def test_all_fields_are_required_and_extra_fields_are_not_allowed(rules):
    schema = build_output_schema(rules)
    assert schema["required"] == list(schema["properties"])
    assert schema["additionalProperties"] is False
    assert schema["properties"]["items"]["items"]["additionalProperties"] is False


def test_review_document_may_have_no_items(validator, example):
    example["items"] = []
    example["needs_review"] = True
    assert validator.is_valid(example)


def test_negative_amounts_are_valid(validator, example):
    example["items"][0]["amount"] = -5.0
    example["document_type"] = "credit_note"
    assert validator.is_valid(example)


def test_each_document_type_is_valid(validator, example):
    for document_type in ("invoice", "receipt", "credit_note"):
        example["document_type"] = document_type
        assert validator.is_valid(example)


def test_unknown_account_is_invalid(validator, example):
    example["items"][0]["account"] = "9999"
    assert not validator.is_valid(example)


@pytest.mark.parametrize("vat_rate", ["16", "19.0", 19, ""])
def test_bad_vat_rate_is_invalid(validator, example, vat_rate):
    example["items"][0]["vat_rate"] = vat_rate
    assert not validator.is_valid(example)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_type", "offer"),
        ("document_date", "30.09.2026"),
        ("currency", "eur"),
        ("total", "45.20"),
        ("reverse_charge", "false"),
        ("needs_review", None),
        ("reason", "x" * 121),
    ],
)
def test_bad_field_values_are_invalid(validator, example, field, value):
    example[field] = value
    assert not validator.is_valid(example)


def test_reason_of_120_characters_is_valid(validator, example):
    example["reason"] = "x" * 120
    assert validator.is_valid(example)


@pytest.mark.parametrize("field", list(EXAMPLE))
def test_missing_field_is_invalid(validator, example, field):
    del example[field]
    assert not validator.is_valid(example)


def test_missing_item_field_is_invalid(validator, example):
    del example["items"][0]["account"]
    assert not validator.is_valid(example)


def test_extra_fields_are_invalid(validator, example):
    assert not validator.is_valid({**example, "comment": "hello"})
    example["items"][0]["note"] = "hello"
    assert not validator.is_valid(example)
