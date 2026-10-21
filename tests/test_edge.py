import re
from decimal import Decimal

import pytest
from jsonschema import Draft202012Validator

from adapt.config import load_platform
from adapt.data.edge import (
    AMBIGUOUS_ITEMS,
    CATEGORIES,
    EDGE_WEIGHTS,
    FOREIGN_CURRENCIES,
    FREE_TEXT_NOTES,
    INJECTIONS,
    MIXED_LAYOUTS,
    MONEY,
    NUMBERED_AMBIGUOUS,
    SMUDGE_CHARS,
    UNSUPPORTED_CHART,
    edge_counts,
    generate_edge_documents,
)
from adapt.data.generate import make_suppliers
from adapt.data.layouts import REVERSE_CHARGE_NOTES
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


NUMBERED = re.compile(
    "|".join(re.escape(pattern).replace(r"\{n\}", r"\d+") for pattern in NUMBERED_AMBIGUOUS),
    re.IGNORECASE,
)


def test_ambiguous_documents_show_an_ambiguous_item(documents):
    for document in in_category(documents, "ambiguous_purpose"):
        text = document.text.lower()
        assert any(item.lower() in text for item in AMBIGUOUS_ITEMS) or NUMBERED.search(text)


def test_mixed_documents_show_private_and_business_items(platform, documents):
    private_accounts = {a.number for a in platform.rules.accounts if a.group == "private"}
    items = [i for t in platform.rules.supplier_types for i in t.items]
    private = {i.description for i in items if i.account in private_accounts}
    business = {i.description for i in items if i.account not in private_accounts}
    for document in in_category(documents, "business_private_mixed"):
        text = document.text.lower()
        assert any(description.lower() in text for description in private)
        assert any(description.lower() in text for description in business)


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
        assert not any(note in document.text for note in REVERSE_CHARGE_NOTES)
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
    assert {"till_receipt", "table_invoice", "tax_code_receipt", "quantity_receipt"} <= layouts
    assert layouts <= set(MIXED_LAYOUTS)


def test_mixed_documents_name_their_private_items(platform, many):
    private_accounts = {a.number for a in platform.rules.accounts if a.group == "private"}
    private = {
        i.description.lower()
        for t in platform.rules.supplier_types
        for i in t.items
        if i.account in private_accounts
    }
    for document in in_category(many, "business_private_mixed"):
        text = document.text.lower()
        assert any(description in text for description in private)


def test_mixed_documents_come_from_business_shops_and_from_private_shops(platform, suppliers, many):
    private_accounts = {a.number for a in platform.rules.accounts if a.group == "private"}
    type_of = {s.name: s.type_key for s in suppliers}
    types = {t.key: t for t in platform.rules.supplier_types}
    kinds = set()
    for document in in_category(many, "business_private_mixed"):
        shop = types[type_of[document.supplier]]
        kinds.add(all(i.account in private_accounts for i in shop.items))
        assert shop.country == "DE"
    assert kinds == {True, False}


def test_a_private_shop_receipt_with_a_business_item_names_paper_or_a_book(
    platform, suppliers, many
):
    private_accounts = {a.number for a in platform.rules.accounts if a.group == "private"}
    type_of = {s.name: s.type_key for s in suppliers}
    types = {t.key: t for t in platform.rules.supplier_types}
    business = {
        i.description.lower()
        for t in platform.rules.supplier_types
        for i in t.items
        if i.account in ("4930", "4940")
    }
    found = 0
    for document in in_category(many, "business_private_mixed"):
        shop = types[type_of[document.supplier]]
        if all(i.account in private_accounts for i in shop.items):
            found += 1
            assert any(name in document.text.lower() for name in business)
    assert found > 5


def test_free_text_reverse_charge_notes_use_many_phrasings(many):
    found = in_category(many, "reverse_charge_free_text")
    used = {note for document in found for note in FREE_TEXT_NOTES if note in document.text}
    assert len(FREE_TEXT_NOTES) >= 40
    assert len(used) >= 20


def test_free_text_reverse_charge_documents_use_several_layouts(many):
    layouts = {d.layout for d in in_category(many, "reverse_charge_free_text")}
    assert "eu_invoice" in layouts
    assert len(layouts) >= 4


def test_injection_documents_use_many_phrasings(many):
    found = in_category(many, "prompt_injection")
    used = {phrase for document in found for phrase in INJECTIONS if phrase in document.text}
    assert len(INJECTIONS) >= 40
    assert len(used) >= 20


def test_ambiguous_documents_use_many_phrasings(many):
    found = in_category(many, "ambiguous_purpose")
    used = {
        item
        for document in found
        for item in AMBIGUOUS_ITEMS
        if item.lower() in document.text.lower()
    }
    assert len(AMBIGUOUS_ITEMS) >= 80
    assert len(used) >= 30
    assert any(NUMBERED.search(document.text) for document in found)


def test_the_hard_categories_get_more_documents(platform, suppliers):
    counts = edge_counts(3, EDGE_WEIGHTS)
    assert counts["business_private_mixed"] == 9 and counts["ambiguous_purpose"] == 6
    assert counts["missing_total"] == 3
    assert set(EDGE_WEIGHTS) <= set(CATEGORIES)
    documents = generate_edge_documents(platform, suppliers, SEED, 2, weights=EDGE_WEIGHTS)
    assert len(documents) == sum(edge_counts(2, EDGE_WEIGHTS).values())
    assert len(in_category(documents, "business_private_mixed")) == 6
    plain = generate_edge_documents(platform, suppliers, SEED, 2)
    assert in_category(documents, "missing_total") == in_category(plain, "missing_total")


def test_a_discount_sits_below_the_item_it_reduces_and_shares_its_account(many):
    named = 0
    for document in in_category(many, "credit_or_discount"):
        items = document.target["items"]
        if document.target["document_type"] == "credit_note":
            continue
        position = next(n for n, item in enumerate(items) if item["amount"] < 0)
        above = items[position - 1]
        assert items[position]["account"] == above["account"]
        assert items[position]["vat_rate"] == above["vat_rate"]
        assert -items[position]["amount"] <= above["amount"] * 0.2 + 0.01
        if items[position]["description"].endswith(f" on {above['description']}"):
            named += 1
    assert named > 0


def test_an_ambiguous_name_is_never_the_name_of_a_real_item(platform):
    real = {
        item.description.lower().rstrip("s")
        for supplier_type in platform.rules.supplier_types
        for item in supplier_type.items
    }
    for name in AMBIGUOUS_ITEMS:
        assert name.lower().rstrip("s") not in real, name
    assert len(set(AMBIGUOUS_ITEMS)) == len(AMBIGUOUS_ITEMS)
