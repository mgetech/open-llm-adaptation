import re
from decimal import Decimal

import pytest
from jsonschema import Draft202012Validator

from adapt.config import load_platform
from adapt.data.edge import (
    AMBIGUOUS_ITEMS,
    CATEGORIES,
    FOREIGN_CURRENCIES,
    FREE_TEXT_NOTES,
    INJECTIONS,
    MONEY,
    SMUDGE_CHARS,
    UNSUPPORTED_CHART,
    generate_edge_documents,
)
from adapt.data.generate import make_suppliers
from adapt.data.layouts import REVERSE_CHARGE_NOTE
from adapt.schema import build_output_schema

SEED = 21
CODES = dict(FOREIGN_CURRENCIES)

# category: (expected outcome of the whole system, needs_review in the model output)
EXPECTED = {
    "ambiguous_purpose": (True, True),
    "business_private_mixed": (True, True),
    "missing_total": (True, True),
    "unsupported_chart": (True, True),
    "foreign_currency": (True, True),
    "heavy_noise": (True, True),
    "near_asset_limit": (True, False),
    "reverse_charge_free_text": (False, False),
    "credit_or_discount": (False, False),
}


@pytest.fixture(scope="module")
def platform():
    return load_platform("tax-booking")


@pytest.fixture(scope="module")
def suppliers(platform):
    return make_suppliers(platform.rules, SEED)


@pytest.fixture(scope="module")
def documents(platform, suppliers):
    return generate_edge_documents(platform, suppliers, SEED)


@pytest.fixture(scope="module")
def many(platform, suppliers):
    return generate_edge_documents(platform, suppliers, SEED, per_category=60)


def in_category(documents, category):
    return [d for d in documents if d.category == category]


def test_default_size_and_order(documents):
    assert len(documents) == 150
    assert [d.category for d in documents] == [c for c in CATEGORIES for _ in range(15)]


def test_categories_match_the_spec_table():
    assert len(CATEGORIES) == 10
    assert set(EXPECTED) | {"prompt_injection"} == set(CATEGORIES)


@pytest.mark.parametrize("category", list(EXPECTED))
def test_each_category_has_its_expected_label(documents, category):
    expected_review, model_review = EXPECTED[category]
    found = in_category(documents, category)
    assert len(found) == 15
    for document in found:
        assert document.expected_review is expected_review
        assert document.target["needs_review"] is model_review


def test_documents_the_model_flags_have_no_items_and_a_reason(documents):
    for document in documents:
        if document.target["needs_review"]:
            assert document.target["items"] == []
            assert document.target["reason"]


def test_targets_follow_the_schema(platform, documents):
    validator = Draft202012Validator(build_output_schema(platform.rules))
    for document in documents:
        assert validator.is_valid(document.target), document.target


def test_text_is_not_empty_and_short(documents):
    assert all(0 < len(d.text) < 6000 for d in documents)


def test_letter_invoice_is_never_used(documents):
    assert "letter_invoice" not in {d.layout for d in documents}


def test_suppliers_come_from_the_pool(suppliers, documents):
    names = {s.name for s in suppliers}
    assert all(d.supplier in names for d in documents)


def test_only_the_unsupported_chart_category_changes_the_chart(platform, documents):
    for document in documents:
        if document.category == "unsupported_chart":
            assert document.chart == UNSUPPORTED_CHART != platform.rules.chart
        else:
            assert document.chart == platform.rules.chart


def test_ambiguous_documents_show_an_ambiguous_item(documents):
    for document in in_category(documents, "ambiguous_purpose"):
        assert any(item in document.text for item in AMBIGUOUS_ITEMS)


def test_mixed_documents_show_private_and_business_items(platform, documents):
    private_accounts = {a.number for a in platform.rules.accounts if a.group == "private"}
    items = [i for t in platform.rules.supplier_types for i in t.items]
    private = {i.description for i in items if i.account in private_accounts}
    business = {i.description for i in items if i.account not in private_accounts}
    for document in in_category(documents, "business_private_mixed"):
        assert any(description in document.text for description in private)
        assert any(description in document.text for description in business)


def test_missing_total_documents_have_no_complete_total(documents):
    found = in_category(documents, "missing_total")
    for document in found:
        after_label = document.text[document.text.lower().rfind("total") :]
        assert not MONEY.search(after_label), document.text
        assert document.target["total"] == 0.0
    # Both kinds appear: the total line is gone, or it is cut in the middle of the number.
    assert any("total" not in d.text.lower() for d in found)
    assert any("total" in d.text.lower() for d in found)


def test_foreign_currency_documents_do_not_use_euro(documents):
    for document in in_category(documents, "foreign_currency"):
        code = document.target["currency"]
        assert code in CODES
        assert "EUR" not in document.text
        assert "€" not in document.text
        assert code in document.text or CODES[code] in document.text


def test_heavy_noise_documents_have_illegible_amounts(documents):
    for document in in_category(documents, "heavy_noise"):
        assert any(char in document.text for char in SMUDGE_CHARS)
        assert document.target["total"] == 0.0


def test_near_limit_documents_have_an_item_close_to_the_limit(platform, documents):
    limit = platform.config.asset_limit
    low, high = limit.net_eur - limit.margin_eur, limit.net_eur + limit.margin_eur
    for document in in_category(documents, "near_asset_limit"):
        nets = [
            Decimal(str(item["amount"])) / (1 + Decimal(item["vat_rate"]) / 100)
            for item in document.target["items"]
        ]
        assert any(low <= net <= high for net in nets), nets


def test_reverse_charge_note_appears_only_in_free_text(documents):
    for document in in_category(documents, "reverse_charge_free_text"):
        assert document.target["reverse_charge"] is True
        assert REVERSE_CHARGE_NOTE not in document.text
        assert any(note in document.text for note in FREE_TEXT_NOTES)


def test_credit_and_discount_documents_have_negative_amounts(documents):
    found = in_category(documents, "credit_or_discount")
    for document in found:
        amounts = [Decimal(str(i["amount"])) for i in document.target["items"]]
        assert any(amount < 0 for amount in amounts)
        assert sum(amounts) == Decimal(str(document.target["total"]))
    layouts = {d.layout for d in found}
    assert "credit_note" in layouts
    assert layouts - {"credit_note"}


def test_injection_documents_keep_the_correct_label(platform, many):
    found = in_category(many, "prompt_injection")
    flagged = [d for d in found if d.target["needs_review"]]
    normal = [d for d in found if not d.target["needs_review"]]
    assert flagged
    assert normal
    for document in found:
        assert any(phrase in document.text for phrase in INJECTIONS)
        assert document.expected_review is document.target["needs_review"]
    # No injected account (1000 is not in the chart, 1800 is a private account) is booked.
    for document in normal:
        accounts = {item["account"] for item in document.target["items"]}
        assert accounts
        assert not accounts & {"1000", "1800"}


def test_injection_text_appears_only_in_injection_documents(documents):
    for document in documents:
        has_injection = any(phrase in document.text for phrase in INJECTIONS)
        assert has_injection == (document.category == "prompt_injection")


def test_same_seed_gives_same_documents(platform, suppliers, documents):
    assert generate_edge_documents(platform, suppliers, SEED) == documents
    assert generate_edge_documents(platform, suppliers, SEED + 1) != documents


def test_per_category_can_be_changed(platform, suppliers):
    documents = generate_edge_documents(platform, suppliers, SEED, per_category=3)
    assert len(documents) == 30


def test_pool_without_a_suitable_supplier_is_an_error(platform, suppliers):
    banks = [s for s in suppliers if s.type_key == "bank"]
    with pytest.raises(ValueError, match="no supplier in the pool"):
        generate_edge_documents(platform, banks, SEED, per_category=1)


def test_money_pattern_matches_both_styles():
    assert re.findall(MONEY, "Total 1,234.56 and 1.234,56 and 4.50, date 30.09.2026") == [
        "1,234.56",
        "1.234,56",
        "4.50",
    ]


def test_mixed_documents_appear_on_receipts_and_invoices(many):
    layouts = {d.layout for d in in_category(many, "business_private_mixed")}
    assert {"till_receipt", "table_invoice"} <= layouts
    assert layouts <= {"till_receipt", "short_receipt", "table_invoice"}


def test_mixed_documents_name_their_private_items(platform, many):
    private_accounts = {a.number for a in platform.rules.accounts if a.group == "private"}
    private = {
        i.description
        for t in platform.rules.supplier_types
        for i in t.items
        if i.account in private_accounts
    }
    for document in in_category(many, "business_private_mixed"):
        assert any(description in document.text for description in private)
