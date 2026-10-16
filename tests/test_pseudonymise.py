import random
import re

import pytest

from adapt.config import load_platform
from adapt.data.edge import generate_edge_documents
from adapt.data.generate import generate_invoices, make_suppliers
from adapt.data.layouts import (
    IDENTIFIER_LABELS,
    IDENTIFIER_SHARE,
    NUMBER_STYLES,
    format_money,
    render_document,
)
from adapt.data.noise import NOISE_LEVELS, add_noise
from adapt.data.pseudonymise import PLACEHOLDERS, pseudonymise

SEED = 17


@pytest.mark.parametrize(
    "text",
    [
        "DE89 3704 0044 0532 0130 00",
        "DE89370400440532013000",
        "GB82 WEST 1234 5698 7654 32",
        "NL91 ABNA 0417 1643 00",
    ],
)
def test_iban_is_replaced(text):
    assert pseudonymise(f"Bank: IBAN {text}.") == "Bank: IBAN <IBAN>."


@pytest.mark.parametrize(
    "text",
    ["info@nordpaper.example", "max.muster+billing@mail-server.example.org", "a_b@x.co"],
)
def test_email_is_replaced(text):
    assert pseudonymise(f"Write to {text}.") == "Write to <EMAIL>."


@pytest.mark.parametrize(
    "text",
    [
        "+49 30 23125 123",
        "+49 151 2345678",
        "+4930231250",
        "+49 (0) 30 23125123",
        "030 23125 123",
        "030/2312512",
        "030-23125-123",
        "(030) 23125 123",
    ],
)
def test_phone_is_replaced(text):
    assert pseudonymise(f"Phone {text}.") == "Phone <PHONE>."


@pytest.mark.parametrize(
    "text", ["12345678901", "12 345 678 901", "14 034 567 890", "14034567890", "90 000 000 000"]
)
def test_tax_id_is_replaced(text):
    assert pseudonymise(f"Tax ID {text}.") == "Tax ID <TAX_ID>."


def test_every_kind_in_one_text():
    text = (
        "IBAN DE89 3704 0044 0532 0130 00, info@nordpaper.example, "
        "Tel. +49 30 23125 123, Tax ID 12 345 678 901"
    )
    assert pseudonymise(text) == "IBAN <IBAN>, <EMAIL>, Tel. <PHONE>, Tax ID <TAX_ID>"


def test_several_of_the_same_kind():
    assert pseudonymise("a@x.example and b@y.example") == "<EMAIL> and <EMAIL>"


def test_a_clean_document_is_unchanged():
    text = (
        "NORDPAPER GmbH  Invoice INV-00397  2026-09-30\n"
        "2x Printer paper A4 ... 1,234.56\n"
        "1x Tax handbook 2026 ... 1.234,56\n"
        "Receipt 006134  30.09.26  30/09/2026  30 September 2026\n"
        "Net 19%: 20.00  VAT 19%: 3.80\n"
        "Total EUR 45.20"
    )
    assert pseudonymise(text) == text


@pytest.mark.parametrize(
    "text",
    ["1234567890", "123456789012", "12345678901234", "2026-09-30", "0,50", "0.00"],
)
def test_numbers_of_the_wrong_length_are_kept(text):
    assert pseudonymise(f"Amount {text}") == f"Amount {text}"


def test_pseudonymise_twice_changes_nothing_more():
    once = pseudonymise("info@x.example, +49 30 23125 123, 12345678901")
    assert pseudonymise(once) == once


def test_placeholders():
    assert set(PLACEHOLDERS) == {"<EMAIL>", "<IBAN>", "<PHONE>", "<TAX_ID>"}


# The generated documents show that real-looking text is not changed by mistake and that
# every fake identifier is found.


@pytest.fixture(scope="module")
def platform():
    return load_platform("tax-booking")


@pytest.fixture(scope="module")
def documents(platform):
    suppliers = make_suppliers(platform.rules, SEED)
    invoices = generate_invoices(platform.rules, suppliers, SEED, 1500)
    texts = [
        (invoice, render_document(invoice, random.Random(f"{SEED}/render/{n}")))
        for n, invoice in enumerate(invoices)
    ]
    edge = generate_edge_documents(platform, suppliers, SEED)
    return texts, edge


def has_identifiers(text):
    return any(label in text for label in IDENTIFIER_LABELS)


def test_about_a_third_of_the_documents_carry_identifiers(documents):
    texts, _ = documents
    share = sum(has_identifiers(text) for _, text in texts) / len(texts)
    assert abs(share - IDENTIFIER_SHARE) < 0.05


def test_documents_without_identifiers_are_unchanged(documents):
    texts, edge = documents
    clean = [text for _, text in texts if not has_identifiers(text)]
    clean += [d.text for d in edge if not has_identifiers(d.text)]
    assert len(clean) > 1000
    for text in clean:
        assert pseudonymise(text) == text


@pytest.mark.parametrize("level", list(NOISE_LEVELS))
def test_noisy_documents_without_identifiers_are_unchanged(documents, level):
    texts, _ = documents
    for n, (_, text) in enumerate(texts[:600]):
        if not has_identifiers(text):
            noisy = add_noise(text, level, random.Random(n))
            assert pseudonymise(noisy) == noisy


def test_every_identifier_in_a_document_is_found(documents):
    texts, _ = documents
    found = {"IBAN": 0, "Tax": 0, "Email": 0, "Phone": 0}
    for _, text in texts:
        result = pseudonymise(text)
        if "IBAN " in text:
            assert "<IBAN>" in result
            found["IBAN"] += 1
        if re.search(r"Tax (ID|number) ", text):
            assert "<TAX_ID>" in result
            found["Tax"] += 1
        if "Email " in text:
            assert "<EMAIL>" in result
            found["Email"] += 1
        if re.search(r"(Phone|Tel\.) ", text):
            assert "<PHONE>" in result
            found["Phone"] += 1
        assert "@" not in result
        assert not re.search(r"\b[A-Z]{2}\d{2} ?\d{4}", result)
        assert not re.search(r"\b\d{2}(?: ?\d{3}){3}\b", result)
        assert not re.search(r"\b0?30 ?23125", result)
    assert all(count > 100 for count in found.values()), found


def test_amounts_survive_pseudonymisation(documents):
    texts, _ = documents
    for invoice, text in texts:
        result = pseudonymise(text)
        amounts = [item.amount for item in invoice.items] + [invoice.total]
        assert any(
            all(format_money(a, style) in result for a in amounts) for style in NUMBER_STYLES
        ), result
