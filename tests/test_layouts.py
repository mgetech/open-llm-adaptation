import random
import re
from datetime import date
from decimal import Decimal

import pytest

from adapt.config import load_platform
from adapt.data.generate import Invoice, Item, Supplier, generate_invoices, make_suppliers
from adapt.data.layouts import (
    DATE_FORMATS,
    LETTER_OPENINGS,
    NUMBER_STYLES,
    REVERSE_CHARGE_NOTES,
    format_money,
    render_document,
    split_quantity,
    vat_summary,
)
from adapt.pipeline import document_problems

SEED = 11
LAYOUTS = [
    "table_invoice",
    "till_receipt",
    "letter_invoice",
    "credit_note",
    "eu_invoice",
    "short_receipt",
    "column_invoice",
    "service_invoice",
    "email_invoice",
    "statement",
    "tax_code_receipt",
    "quantity_receipt",
]


@pytest.fixture(scope="module")
def invoices():
    rules = load_platform("tax-booking").rules
    return generate_invoices(rules, make_suppliers(rules, SEED), SEED, 3000)


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


@pytest.fixture(scope="module")
def platform():
    return load_platform("tax-booking")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_document_has_a_total_the_pipeline_can_find(rendered, platform, layout):
    for _, text in by_layout(rendered, layout):
        assert "total" in text.lower()
        assert "no_total" not in document_problems(text, "SKR03", platform)


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


def test_a_reverse_charge_note_only_on_reverse_charge_invoices(rendered):
    for invoice, text in rendered:
        has_note = any(note in text for note in REVERSE_CHARGE_NOTES)
        assert has_note == invoice.reverse_charge, text


def test_reverse_charge_invoices_are_net_and_print_no_vat_lines(rendered):
    foreign = [(invoice, text) for invoice, text in rendered if invoice.reverse_charge]
    assert len({invoice.layout for invoice, _ in foreign}) >= 4
    for _, text in foreign:
        assert "VAT 19%" not in text and "VAT 7%" not in text and "Net 19%" not in text


def test_the_note_moves_around_inside_a_document(rendered):
    positions = set()
    for _, text in by_layout(rendered, "eu_invoice"):
        lines = text.split("\n")
        for number, line in enumerate(lines):
            if any(note in line for note in REVERSE_CHARGE_NOTES):
                positions.add("start" if number < len(lines) // 3 else "later")
    assert positions == {"start", "later"}


def test_credit_notes_print_negative_amounts(rendered):
    for invoice, text in by_layout(rendered, "credit_note"):
        assert "credit" in text.lower()
        for item in invoice.items:
            assert any(format_money(item.amount, style) in text for style in NUMBER_STYLES)
            assert item.amount < 0


@pytest.mark.parametrize(
    "layout", ["table_invoice", "column_invoice", "service_invoice", "statement", "credit_note"]
)
def test_invoices_print_the_vat_of_every_rate(rendered, layout):
    shown = [(i, t) for i, t in by_layout(rendered, layout) if not i.reverse_charge]
    assert shown
    for invoice, text in shown:
        for rate, _, vat in vat_summary(invoice):
            assert f"VAT {rate}%" in text
            assert any(format_money(vat, style) in text for style in NUMBER_STYLES), text


def test_the_vat_lines_have_several_wordings(rendered):
    texts = [t for i, t in by_layout(rendered, "table_invoice") if not i.reverse_charge]
    assert any("(net " in t for t in texts)
    assert any("Net 19%:" in t and "  VAT" in t for t in texts)
    assert any("\nVAT 19%" in t for t in texts)
    assert any("| VAT" in t for t in texts)


@pytest.mark.parametrize("layout", ["till_receipt", "short_receipt", "quantity_receipt"])
def test_receipts_show_the_vat_inside_the_amounts(rendered, layout):
    for invoice, text in by_layout(rendered, layout):
        if "Total" in text.splitlines()[0]:
            continue  # the one-line receipt prints no VAT
        assert "vat" in text.lower(), text
        for _, _, vat in vat_summary(invoice):
            assert any(format_money(vat, style) in text for style in NUMBER_STYLES), text


def test_letter_invoice_is_prose(rendered):
    for invoice, text in by_layout(rendered, "letter_invoice"):
        assert text.startswith(LETTER_OPENINGS)
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


def test_split_quantity_multiplies_back_to_the_amount():
    rng = random.Random(1)
    for cents in range(100, 4000, 7):
        amount = Decimal(cents).scaleb(-2)
        quantity, unit = split_quantity(amount, rng)
        assert quantity * unit == amount
        assert unit.as_tuple().exponent >= -2
    assert split_quantity(Decimal("7.01"), rng) == (1, Decimal("7.01"))
    assert split_quantity(Decimal("-12.00"), rng)[1] < 0


def test_column_invoice_cuts_the_table_into_single_values(rendered):
    for invoice, text in by_layout(rendered, "column_invoice"):
        lines = text.split("\n")
        heading = next(i for i, line in enumerate(lines) if line in ("Qty", "Pos", "Quantity"))
        assert lines[heading + 1] in ("Item", "Description")
        values = lines[heading + 4 : heading + 8]
        amount = invoice.items[0].amount
        assert any(format_money(amount, style) in values for style in NUMBER_STYLES), text


def test_service_invoice_has_rows_with_pipes(rendered):
    for _, text in by_layout(rendered, "service_invoice"):
        rows = [line for line in text.split("\n") if line.startswith(("1 |", "Pos |"))]
        assert len(rows) == 2, text


def test_email_invoice_reads_as_a_message(rendered):
    for invoice, text in by_layout(rendered, "email_invoice"):
        assert text.startswith("Subject: Invoice ")
        assert invoice.supplier.name in text.splitlines()[0]
        assert re.search(rf"[Rr]egards\n{re.escape(invoice.supplier.name)}", text), text


def test_statement_lines_start_with_a_day_and_a_month(rendered):
    for invoice, text in by_layout(rendered, "statement"):
        dated = [line for line in text.split("\n") if re.match(r"\d\d\.\d\d\. ", line)]
        assert len(dated) == len(invoice.items), text
        assert all(int(line[3:5]) == invoice.date.month for line in dated)


def test_tax_code_receipt_gives_every_item_a_code_and_explains_the_codes(rendered):
    for invoice, text in by_layout(rendered, "tax_code_receipt"):
        assert "Tax  Rate  Net  VAT" in text
        coded = [line for line in text.split("\n") if re.search(r" [ABC]$", line)]
        assert len(coded) == len(invoice.items), text
        codes = {line[-1] for line in coded}
        legend = re.findall(r"^([ABC])  (\d+)%", text, re.M)
        assert {code for code, _ in legend} == codes
        assert {rate for _, rate in legend} == {rate for rate, _, _ in vat_summary(invoice)}


def test_quantity_receipt_prints_the_unit_price_under_the_item(rendered):
    for invoice, text in by_layout(rendered, "quantity_receipt"):
        rows = re.findall(r"^\s+(\d) x ([\d.,]+)", text, re.M)
        assert len(rows) == len(invoice.items), text


def parse_amount(text, style):
    return Decimal(
        text.replace(".", "").replace(",", ".") if style == "de" else text.replace(",", "")
    )


def test_a_payment_block_after_the_total_adds_up(rendered):
    seen = 0
    for invoice, text in rendered:
        if "Cash " not in text or "Change " not in text:
            continue
        seen += 1
        cash = re.search(r"Cash ([\d.,]+)", text).group(1)
        change = re.search(r"Change ([\d.,]+)", text).group(1)
        style = "de" if re.search(r",\d\d$", cash) else "en"
        assert parse_amount(cash, style) - parse_amount(change, style) == invoice.total, text
    assert seen > 50


def test_some_receipts_print_the_item_names_in_capitals(rendered):
    upper = [
        text
        for invoice, text in by_layout(rendered, "till_receipt")
        if invoice.items[0].description.upper() in text
    ]
    assert upper


def test_article_numbers_and_quantity_rows_appear(rendered):
    texts = [text for _, text in rendered]
    assert sum(bool(re.search(r"(Art\.|SKU|Item no\.|Ref) \d{3,5}", t)) for t in texts) > 200
    assert sum(bool(re.search(r"\d x [\d.,]+ ", t)) for t in texts) > 200
