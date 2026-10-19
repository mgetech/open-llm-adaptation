import json
from decimal import Decimal

import pytest

from adapt.config import load_platform
from adapt.data.split import build_splits
from adapt.eval.runner import load_documents, split_path
from adapt.vat_check import parse_amount, vat_mismatch

GENERATED = ("dev", "test_iid", "test_ood", "edge")
ITEMS = [
    {"description": "Printer paper A4", "amount": 23.8, "account": "4930", "vat_rate": "19"},
    {"description": "Tax handbook 2026", "amount": 21.4, "account": "4940", "vat_rate": "7"},
]
TABLE = (
    "Nordpaper GmbH  Invoice 2026-1184\n"
    "Printer paper A4 ... 23.80\n"
    "Tax handbook 2026 ... 21.40\n"
    "Net 19%: 20.00  VAT 19%: 3.80\n"
    "Net 7%: 20.00  VAT 7%: 1.40\n"
    "Total EUR 45.20"
)


@pytest.fixture(scope="module")
def platform():
    return load_platform("tax-booking")


def at_rate(items, index, rate):
    return [{**item, "vat_rate": rate} if i == index else item for i, item in enumerate(items)]


@pytest.mark.parametrize(
    ("text", "value"),
    [("1,234.56", "1234.56"), ("1.234,56", "1234.56"), ("3.80", "3.80"), ("-59,50", "-59.50")],
)
def test_amounts_in_both_number_styles_are_read(text, value):
    assert parse_amount(text) == Decimal(value)


def test_a_booking_that_matches_the_printed_net_and_vat_is_not_a_mismatch():
    assert not vat_mismatch(TABLE, ITEMS)


def test_an_item_booked_at_the_wrong_rate_is_a_mismatch():
    assert vat_mismatch(TABLE, at_rate(ITEMS, 0, "7"))
    assert vat_mismatch(TABLE, at_rate(ITEMS, 1, "19"))


def test_a_whole_document_moved_to_another_rate_is_a_mismatch():
    text = "Shop\nTea 10.70\nincl. VAT 7%: 0.70\nTotal 10.70"
    item = {"description": "Tea", "amount": 10.7, "account": "3300", "vat_rate": "7"}
    assert not vat_mismatch(text, [item])
    assert vat_mismatch(text, [{**item, "vat_rate": "19"}])


def test_the_included_vat_lines_of_a_receipt_are_compared():
    text = "Shop\nTea 10.70\nCake 11.90\nTotal 22.60\nincl. VAT 19%: 1.90\nincl. VAT 7%: 0.70"
    items = [
        {"description": "Tea", "amount": 10.7, "account": "3300", "vat_rate": "7"},
        {"description": "Cake", "amount": 11.9, "account": "3400", "vat_rate": "19"},
    ]
    assert not vat_mismatch(text, items)
    assert vat_mismatch(text, at_rate(items, 1, "7"))


def test_the_vat_sentence_of_a_letter_is_compared():
    text = (
        "Dear Sir,\nTea: 10.70. Cake: 11.90.\n"
        "The invoice includes VAT of 19% (1.90), 7% (0.70).\nTotal 22.60"
    )
    items = [
        {"description": "Tea", "amount": 10.7, "account": "3300", "vat_rate": "7"},
        {"description": "Cake", "amount": 11.9, "account": "3400", "vat_rate": "19"},
    ]
    assert not vat_mismatch(text, items)
    assert vat_mismatch(text, at_rate(items, 0, "19"))


def test_german_number_style_and_negative_credit_notes_are_read():
    text = "Shop\nBooks -1.021,40\nNet 7%: -954,58  VAT 7%: -66,82\nCredit total -1.021,40"
    item = {"description": "Books", "amount": -1021.4, "account": "4940", "vat_rate": "7"}
    assert not vat_mismatch(text, [item])
    assert vat_mismatch(text, [{**item, "vat_rate": "19"}])


def test_a_word_merged_into_the_number_before_it_is_still_read():
    text = "Shop\nTea ...10,70Net 7%: 10,00  VAT 7%:0,70\nTotal 10,70"
    item = {"description": "Tea", "amount": 10.7, "account": "3300", "vat_rate": "7"}
    assert not vat_mismatch(text, [item])
    assert vat_mismatch(text, [{**item, "vat_rate": "19"}])


@pytest.mark.parametrize("items", [ITEMS, []])
def test_a_document_without_printed_vat_is_not_checked(items):
    assert not vat_mismatch("Shop\nTea 10.70\nTotal 10.70", items)


def test_a_reverse_charge_document_is_not_checked():
    assert not vat_mismatch(TABLE, at_rate(ITEMS, 0, "7"), reverse_charge=True)


def test_a_line_that_cannot_be_read_is_ignored():
    text = TABLE.replace("Net 19%: 20.00", "Net 19%: 2O.OO")
    assert not vat_mismatch(text, ITEMS)
    assert not vat_mismatch(text.replace("VAT 7%: 1.40", "VAT l%: 1.40"), ITEMS)


def test_a_booked_rate_that_is_not_printed_is_a_mismatch_only_if_every_line_was_read():
    text = "Shop\nTea 10.70\nCake 11.90\nTotal 22.60\nincl. VAT 19%: 1.90"
    items = [
        {"description": "Tea", "amount": 10.7, "account": "3300", "vat_rate": "7"},
        {"description": "Cake", "amount": 11.9, "account": "3400", "vat_rate": "19"},
    ]
    assert vat_mismatch(text, items)
    unreadable = text + "\nincl. VAT 7%: O.70"
    assert not vat_mismatch(unreadable, items)


def test_a_zero_rate_that_is_not_printed_is_not_looked_for():
    text = "Shop\nTea 10.70\nStamps 5.00\nTotal 15.70\nNet 7%: 10.00  VAT 7%: 0.70"
    items = [
        {"description": "Tea", "amount": 10.7, "account": "3300", "vat_rate": "7"},
        {"description": "Stamps", "amount": 5.0, "account": "4910", "vat_rate": "0"},
    ]
    assert not vat_mismatch(text, items)


def test_the_amounts_may_differ_by_a_cent_from_rounding():
    item = {"description": "Tea", "amount": 10.7, "account": "3300", "vat_rate": "7"}
    assert not vat_mismatch("Shop\nTea 10.70\nNet 7%: 10.01  VAT 7%: 0.70", [item])
    assert vat_mismatch("Shop\nTea 10.70\nNet 7%: 10.10  VAT 7%: 0.70", [item])


def test_the_reference_answers_never_differ_from_the_printed_vat():
    """Over every generated document, noisy ones included, the correct answer is never flagged.

    Only the generated splits are read here. The gold sets are not."""
    train = [
        (json.loads(line)["document_text"], json.loads(line)["target"])
        for line in split_path("tax-booking", "train").read_text("utf-8").splitlines()
    ]
    others = [
        (d.text, d.target) for split in GENERATED for d in load_documents("tax-booking", split)
    ]
    documents = [pair for pair in train + others if pair[1]["items"]]
    assert len(documents) > 3000
    flagged = [
        text
        for text, target in documents
        if vat_mismatch(text, target["items"], target["reverse_charge"])
    ]
    assert flagged == []


def test_the_check_works_on_a_fresh_set_of_documents(platform):
    records = build_splits(platform, 5)["dev"]
    for record in records:
        target = record["target"]
        if target["items"]:
            assert not vat_mismatch(
                record["document_text"], target["items"], target["reverse_charge"]
            )
