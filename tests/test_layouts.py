import random
from datetime import date
from decimal import Decimal

import pytest

from adapt.config import load_platform
from adapt.data.generate import Invoice, Item, Supplier, generate_invoices, make_suppliers
from adapt.data.layouts import (
    DATE_FORMATS,
    NUMBER_STYLES,
    REVERSE_CHARGE_NOTE,
    format_money,
    render_document,
    vat_summary,
)

SEED = 11
LAYOUTS = [
    "table_invoice",
    "till_receipt",
    "letter_invoice",
    "credit_note",
    "eu_invoice",
    "short_receipt",
]


@pytest.fixture(scope="module")
def invoices():
    rules = load_platform("tax-booking").rules
    return generate_invoices(rules, make_suppliers(rules, SEED), SEED, 1500)


@pytest.fixture(scope="module")
def rendered(invoices):
    return [
        (i, render_document(i, random.Random(f"{SEED}/render/{n}"))) for n, i in enumerate(invoices)
    ]


def by_layout(rendered, layout):
    return [(invoice, text) for invoice, text in rendered if invoice.layout == layout]


def spec_invoice(layout="table_invoice") -> Invoice:
    items = (
        Item("Printer paper A4", Decimal("23.80"), "4930", "19"),
        Item("Tax handbook 2026", Decimal("21.40"), "4940", "7"),
    )
    return Invoice(
        supplier=Supplier("Nordpaper GmbH", "stationery_store"),
        layout=layout,
        date=date(2026, 9, 30),
        number=1184,
        document_type="invoice",
        items=items,
        total=Decimal("45.20"),
        reverse_charge=False,
        target={},
    )


def test_format_money():
    amount = Decimal("1234.56")
    assert format_money(amount, "en") == "1,234.56"
    assert format_money(amount, "de") == "1.234,56"
    assert format_money(Decimal("-1234567.80"), "en") == "-1,234,567.80"
    assert format_money(Decimal("-1234567.80"), "de") == "-1.234.567,80"
    assert format_money(Decimal("4.50"), "en") == "4.50"
    assert format_money(Decimal("4.50"), "de") == "4,50"


def test_vat_summary_of_the_task_example():
    assert vat_summary(spec_invoice()) == [
        ("19", Decimal("20.00"), Decimal("3.80")),
        ("7", Decimal("20.00"), Decimal("1.40")),
    ]


def test_vat_summary_adds_up_to_the_gross_amounts(invoices):
    for invoice in invoices:
        for rate, net, vat in vat_summary(invoice):
            gross = sum(i.amount for i in invoice.items if i.vat_rate == rate)
            assert net + vat == gross
            expected = gross / (1 + Decimal(rate) / 100)
            assert abs(net - expected) <= Decimal("0.005")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_amount_and_the_total_appear(rendered, layout):
    documents = by_layout(rendered, layout)
    assert len(documents) >= 50
    for invoice, text in documents:
        amounts = [item.amount for item in invoice.items] + [invoice.total]
        assert any(
            all(format_money(a, style) in text for a in amounts) for style in NUMBER_STYLES
        ), text


@pytest.mark.parametrize("layout", LAYOUTS)
def test_name_and_date_appear(rendered, layout):
    for invoice, text in by_layout(rendered, layout):
        assert invoice.supplier.name.lower() in text.lower() or (
            invoice.supplier.name.split()[0].lower() in text.lower()
        )
        assert any(fmt(invoice.date) in text for fmt in DATE_FORMATS.values()), text


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_document_has_a_total_label(rendered, layout):
    for _, text in by_layout(rendered, layout):
        assert "total" in text.lower()


def test_documents_stay_short(rendered):
    assert all(0 < len(text) < 2000 for _, text in rendered)


def test_documents_vary_in_number_and_date_format(rendered):
    texts = [text for _, text in rendered]
    assert any("," in text and ".00" not in text for text in texts)
    assert any(".00" in text for text in texts)
    formats_seen = {
        name
        for name, fmt in DATE_FORMATS.items()
        if any(fmt(invoice.date) in text for invoice, text in rendered)
    }
    assert formats_seen == set(DATE_FORMATS)


def test_reverse_charge_note_only_on_eu_invoices(rendered):
    for invoice, text in rendered:
        assert (REVERSE_CHARGE_NOTE in text) == (invoice.layout == "eu_invoice")


def test_credit_notes_print_negative_amounts(rendered):
    for invoice, text in by_layout(rendered, "credit_note"):
        assert "credit" in text.lower()
        for item in invoice.items:
            assert any(format_money(item.amount, style) in text for style in NUMBER_STYLES)
            assert item.amount < 0


def test_net_and_vat_lines_in_table_invoice(rendered):
    for invoice, text in by_layout(rendered, "table_invoice"):
        for rate, _, _ in vat_summary(invoice):
            assert f"Net {rate}%" in text
            assert f"VAT {rate}%" in text


def test_receipts_show_included_vat(rendered):
    for layout in ("till_receipt",):
        for _, text in by_layout(rendered, layout):
            assert "incl. VAT" in text


def test_letter_invoice_is_prose(rendered):
    for invoice, text in by_layout(rendered, "letter_invoice"):
        assert text.startswith(("Dear Sir or Madam,", "Hello,"))
        assert f"Kind regards\n{invoice.supplier.name}" in text


def test_same_rng_gives_same_text(invoices):
    for invoice in invoices[:50]:
        first = render_document(invoice, random.Random(1))
        assert render_document(invoice, random.Random(1)) == first


def test_different_rng_gives_different_texts():
    invoice = spec_invoice()
    texts = {render_document(invoice, random.Random(n)) for n in range(30)}
    assert len(texts) > 10


def test_task_example_is_rendered_in_table_layout():
    text = render_document(spec_invoice(), random.Random(3))
    assert "Printer paper A4" in text
    assert "Tax handbook 2026" in text
    assert "45.20" in text or "45,20" in text
