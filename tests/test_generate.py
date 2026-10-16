from collections import Counter
from decimal import Decimal

import pytest
from jsonschema import Draft202012Validator

from adapt.config import load_platform
from adapt.data.generate import generate_invoices, make_reason, make_suppliers
from adapt.schema import MAX_REASON_LENGTH, build_output_schema

SEED = 7
COUNT = 2000


@pytest.fixture(scope="module")
def rules():
    return load_platform("tax-booking").rules


@pytest.fixture(scope="module")
def suppliers(rules):
    return make_suppliers(rules, SEED)


@pytest.fixture(scope="module")
def invoices(rules, suppliers):
    return generate_invoices(rules, suppliers, SEED, COUNT)


def test_suppliers_have_distinct_names(suppliers):
    names = [s.name for s in suppliers]
    assert len(names) == 200
    assert len(set(names)) == 200
    assert len({name.split()[0] for name in names}) == 200


def test_suppliers_are_spread_over_all_types(rules, suppliers):
    counts = Counter(s.type_key for s in suppliers)
    assert set(counts) == {t.key for t in rules.supplier_types}
    assert min(counts.values()) >= 4


def test_supplier_suffix_fits_the_country(rules, suppliers):
    types = {t.key: t for t in rules.supplier_types}
    for supplier in suppliers:
        suffix = supplier.name.split(" ", 1)[1]
        assert suffix in rules.supplier_names.suffixes[types[supplier.type_key].country]


def test_suppliers_are_deterministic(rules, suppliers):
    assert make_suppliers(rules, SEED) == suppliers
    assert make_suppliers(rules, SEED + 1) != suppliers


def test_too_many_suppliers_are_rejected(rules):
    with pytest.raises(ValueError, match="not enough syllables"):
        make_suppliers(rules, SEED, count=10_000)


def test_same_seed_gives_same_invoices(rules, suppliers, invoices):
    assert generate_invoices(rules, suppliers, SEED, COUNT) == invoices
    assert generate_invoices(rules, suppliers, SEED + 1, COUNT) != invoices


def test_a_document_does_not_depend_on_the_count(rules, suppliers, invoices):
    assert generate_invoices(rules, suppliers, SEED, 10) == invoices[:10]


def test_items_sum_to_total(invoices):
    for invoice in invoices:
        assert sum(item.amount for item in invoice.items) == invoice.total
        target = invoice.target
        assert sum(Decimal(str(i["amount"])) for i in target["items"]) == Decimal(
            str(target["total"])
        )


def test_item_count_is_between_one_and_six(invoices):
    counts = Counter(len(invoice.items) for invoice in invoices)
    assert set(counts) == {1, 2, 3, 4, 5, 6}


def test_amounts_have_two_decimals_and_stay_in_the_price_range(rules, invoices):
    types = {t.key: t for t in rules.supplier_types}
    for invoice in invoices:
        offers = {
            (i.description, i.account, i.vat_rate): i.price
            for i in types[invoice.supplier.type_key].items
        }
        for item in invoice.items:
            assert item.amount.as_tuple().exponent == -2
            low, high = offers[(item.description, item.account, item.vat_rate)]
            assert Decimal(str(low)) <= abs(item.amount) <= Decimal(str(high))


def test_mixed_rate_share_is_about_forty_percent(invoices):
    mixed = sum(len({item.vat_rate for item in invoice.items}) > 1 for invoice in invoices)
    assert 0.35 <= mixed / len(invoices) <= 0.45


def test_mixed_share_can_be_changed(rules, suppliers):
    none = generate_invoices(rules, suppliers, SEED, 300, mixed_share=0)
    assert all(len({i.vat_rate for i in inv.items}) == 1 for inv in none)
    every = generate_invoices(rules, suppliers, SEED, 300, mixed_share=1)
    assert all(len({i.vat_rate for i in inv.items}) > 1 for inv in every)


def test_document_type_follows_the_layout(invoices):
    for invoice in invoices:
        expected = {
            "credit_note": "credit_note",
            "till_receipt": "receipt",
            "short_receipt": "receipt",
        }.get(invoice.layout, "invoice")
        assert invoice.document_type == expected
        assert invoice.target["document_type"] == expected


def test_credit_notes_have_negative_amounts(invoices):
    notes = [i for i in invoices if i.layout == "credit_note"]
    assert notes
    for invoice in notes:
        assert all(item.amount < 0 for item in invoice.items)
        assert invoice.total < 0
    others = [i for i in invoices if i.layout != "credit_note"]
    assert all(item.amount > 0 for invoice in others for item in invoice.items)


def test_reverse_charge_only_on_foreign_invoices(rules, invoices):
    types = {t.key: t for t in rules.supplier_types}
    for invoice in invoices:
        foreign = types[invoice.supplier.type_key].country != "DE"
        assert invoice.reverse_charge == (invoice.layout == "eu_invoice") == foreign
        assert invoice.target["reverse_charge"] == invoice.reverse_charge


def test_every_layout_is_generated(invoices):
    assert {i.layout for i in invoices} == {
        "table_invoice",
        "till_receipt",
        "letter_invoice",
        "credit_note",
        "eu_invoice",
        "short_receipt",
    }


def test_layouts_can_be_limited(rules, suppliers):
    invoices = generate_invoices(
        rules,
        suppliers,
        SEED,
        500,
        layouts={"table_invoice", "till_receipt", "credit_note", "eu_invoice", "short_receipt"},
    )
    assert "letter_invoice" not in {i.layout for i in invoices}


def test_no_usable_supplier_is_an_error(rules, suppliers):
    with pytest.raises(ValueError, match="no supplier"):
        generate_invoices(rules, suppliers, SEED, 5, layouts=set())


def test_dates_are_in_2026(invoices):
    assert all(i.date.year == 2026 for i in invoices)
    assert all(i.target["document_date"] == i.date.isoformat() for i in invoices)


def test_targets_follow_the_schema(rules, invoices):
    validator = Draft202012Validator(build_output_schema(rules))
    for invoice in invoices:
        assert validator.is_valid(invoice.target), invoice.target


def test_targets_match_the_invoice(invoices):
    for invoice in invoices:
        target = invoice.target
        assert target["supplier"] == invoice.supplier.name
        assert target["needs_review"] is False
        assert target["currency"] == "EUR"
        assert [(i["description"], i["account"], i["vat_rate"]) for i in target["items"]] == [
            (i.description, i.account, i.vat_rate) for i in invoice.items
        ]


def test_reason_describes_the_account_groups(rules, suppliers):
    invoices = generate_invoices(rules, suppliers, SEED, 300)
    names = {a.number: a.name.lower() for a in rules.accounts}
    for invoice in invoices:
        reason = invoice.target["reason"]
        assert len(reason) <= MAX_REASON_LENGTH
        assert reason[0].isupper()
        first = invoice.items[0]
        if len({(i.account, i.vat_rate) for i in invoice.items}) == 1:
            assert names[first.account] in reason.lower()
            assert f"at {first.vat_rate}%" in reason
        assert ("reverse charge" in reason) == invoice.reverse_charge
        assert reason.startswith("Credit note: ") == (invoice.document_type == "credit_note")


def test_reason_for_a_mixed_document():
    reason = make_reason(["office supplies at 19%", "books and journals at 7%"], False, False)
    assert reason == "Office supplies at 19%, books and journals at 7%"


def test_reason_is_shortened_when_too_long():
    parts = [f"account number {n} with a long name at 19%" for n in range(6)]
    reason = make_reason(parts, True, True)
    assert len(reason) <= MAX_REASON_LENGTH
    assert "more" in reason
    assert reason.endswith("; reverse charge")
