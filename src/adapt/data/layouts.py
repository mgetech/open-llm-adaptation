"""Render an invoice as document text in one of six layouts.

The text is what the model reads. Number format, date format, spacing and line
breaks change from document to document, but every item amount and the total
always appear in the text.
"""

import random
from collections.abc import Callable
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from adapt.data.generate import Invoice, Item

NUMBER_STYLES = ("en", "de")
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
DATE_FORMATS: dict[str, Callable[[date], str]] = {
    "dotted": lambda d: f"{d.day:02d}.{d.month:02d}.{d.year}",
    "dotted_short": lambda d: f"{d.day:02d}.{d.month:02d}.{d.year % 100:02d}",
    "iso": lambda d: d.isoformat(),
    "slashed": lambda d: f"{d.day:02d}/{d.month:02d}/{d.year}",
    "long": lambda d: f"{d.day} {MONTHS[d.month - 1]} {d.year}",
}
NUMBER_PATTERNS: tuple[Callable[[Invoice], str], ...] = (
    lambda i: f"{i.date.year}-{i.number:04d}",
    lambda i: f"INV-{i.number:05d}",
    lambda i: f"{i.number:06d}",
)
TOTAL_LABELS = ("Total", "Total amount", "Grand total", "Total due")
CREDIT_TOTAL_LABELS = ("Credit total", "Total credit", "Total credit amount")
NET_TOTAL_LABELS = ("Total", "Total net", "Total net amount")
REVERSE_CHARGE_NOTE = "Reverse charge: tax liability of the recipient"

CENT = Decimal("0.01")


def format_money(amount: Decimal, style: str) -> str:
    """Format an amount as `1,234.56` (style "en") or `1.234,56` (style "de")."""
    text = f"{abs(amount):,.2f}"
    if style == "de":
        text = text.translate(str.maketrans(",.", ".,"))
    return f"-{text}" if amount < 0 else text


def vat_summary(invoice: Invoice) -> list[tuple[str, Decimal, Decimal]]:
    """Return (rate, net, vat) for each VAT rate, highest rate first. Amounts are gross."""
    gross: dict[str, Decimal] = {}
    for item in invoice.items:
        gross[item.vat_rate] = gross.get(item.vat_rate, Decimal(0)) + item.amount
    summary = []
    for rate in sorted(gross, key=int, reverse=True):
        net = (gross[rate] / (1 + Decimal(rate) / 100)).quantize(CENT, ROUND_HALF_UP)
        summary.append((rate, net, gross[rate] - net))
    return summary


class _Writer:
    """The choices that stay the same inside one document."""

    def __init__(self, invoice: Invoice, rng: random.Random):
        self.invoice = invoice
        self.rng = rng
        self.style = rng.choice(NUMBER_STYLES)
        self.date = rng.choice(list(DATE_FORMATS.values()))(invoice.date)
        self.currency = rng.choice(("EUR", "€"))
        self.fill = rng.choice(("spaces", "dots", "colon"))
        self.number = rng.choice(NUMBER_PATTERNS)(invoice)
        self.quantities = rng.random() < 0.5
        base, _, suffix = invoice.supplier.name.partition(" ")
        upper = rng.random() < 0.5
        self.name = f"{base.upper()} {suffix}" if upper else invoice.supplier.name

    def money(self, amount: Decimal) -> str:
        return format_money(amount, self.style)

    def line(self, text: str, amount: Decimal) -> str:
        value = self.money(amount)
        if self.fill == "dots":
            return f"{text} ... {value}"
        if self.fill == "colon":
            return f"{text}: {value}"
        return f"{text}{' ' * self.rng.randint(2, 6)}{value}"

    def item_line(self, item: Item) -> str:
        text = item.description
        if self.quantities:
            text = f"{self.rng.randint(1, 3)}x {text}"
        return self.line(text, item.amount)

    def total_line(self, labels: tuple[str, ...]) -> str:
        label = self.rng.choice(labels)
        value = self.money(self.invoice.total)
        if self.rng.random() < 0.5:
            return f"{label} {self.currency} {value}"
        return f"{label}: {value} {self.currency}"

    def join(self, sections: list[str]) -> str:
        text = sections[0]
        for section in sections[1:]:
            text += ("\n\n" if self.rng.random() < 0.2 else "\n") + section
        return text


def _net_and_vat_lines(w: _Writer) -> str:
    return "\n".join(
        f"Net {rate}%: {w.money(net)}  VAT {rate}%: {w.money(vat)}"
        for rate, net, vat in vat_summary(w.invoice)
    )


def _included_vat_lines(w: _Writer) -> str:
    return "\n".join(
        f"incl. VAT {rate}%: {w.money(vat)}" for rate, _, vat in vat_summary(w.invoice)
    )


def _table_invoice(w: _Writer) -> str:
    word = w.rng.choice(("Invoice", "Invoice no.", "Invoice number", "Inv."))
    if w.rng.random() < 0.5:
        head = f"{w.name}  {word} {w.number}  {w.date}"
    else:
        head = f"{w.name}\n{word} {w.number}\nDate: {w.date}"
    items = "\n".join(w.item_line(item) for item in w.invoice.items)
    return w.join([head, items, _net_and_vat_lines(w), w.total_line(TOTAL_LABELS)])


def _till_receipt(w: _Writer) -> str:
    head = f"{w.name}\nReceipt {w.number}  {w.date}"
    items = "\n".join(w.item_line(item) for item in w.invoice.items)
    total = w.total_line(TOTAL_LABELS)
    footer = w.rng.choice(("Thank you for your visit", "Have a nice day", "See you again soon"))
    return w.join([head, items, "-" * 24, total, _included_vat_lines(w), footer])


def _letter_invoice(w: _Writer) -> str:
    rng = w.rng
    sentences = []
    for item in w.invoice.items:
        value = w.money(item.amount)
        sentences.append(
            rng.choice(
                (
                    f"{item.description}: {value}.",
                    f"For {item.description} we charge {value}.",
                    f"{item.description} comes to {value}.",
                )
            )
        )
    vat = ", ".join(f"{rate}% ({w.money(amount)})" for rate, _, amount in vat_summary(w.invoice))
    total = w.money(w.invoice.total)
    notes = rng.sample(
        (
            "Payment is due within 14 days without deduction.",
            "Please quote the invoice number with your payment.",
            "Thank you for your order.",
        ),
        k=2,
    )
    return w.join(
        [
            rng.choice(("Dear Sir or Madam,", "Hello,")),
            f"with this letter we send you our invoice {w.number} dated {w.date}.",
            " ".join(sentences),
            f"The invoice includes VAT of {vat}.",
            f"The {rng.choice(('total', 'total amount'))} is {w.currency} {total}.",
            " ".join(notes),
            f"Kind regards\n{w.invoice.supplier.name}",
        ]
    )


def _credit_note(w: _Writer) -> str:
    head = f"{w.name}  Credit note {w.number}  {w.date}"
    intro = "We credit you for the following items:"
    items = "\n".join(w.item_line(item) for item in w.invoice.items)
    return w.join([head, intro, items, _net_and_vat_lines(w), w.total_line(CREDIT_TOTAL_LABELS)])


def _eu_invoice(w: _Writer) -> str:
    head = f"{w.name}  Invoice {w.number}  {w.date}"
    items = "\n".join(w.item_line(item) for item in w.invoice.items)
    ending = [REVERSE_CHARGE_NOTE, w.total_line(NET_TOTAL_LABELS)]
    if w.rng.random() < 0.5:
        ending.reverse()
    return w.join([head, items, *ending])


def _short_receipt(w: _Writer) -> str:
    invoice = w.invoice
    if len(invoice.items) == 1 and w.rng.random() < 0.5:
        item = invoice.items[0]
        value = w.money(item.amount)
        total = w.money(invoice.total)
        line = f"{item.description} {value} {w.currency}"
        return f"{w.name} {w.date}  {line}  Total {total} {w.currency}"
    items = "\n".join(w.line(item.description, item.amount) for item in invoice.items)
    return w.join(
        [f"{w.name}\n{w.date}", items, w.total_line(TOTAL_LABELS), _included_vat_lines(w)]
    )


_LAYOUTS: dict[str, Callable[[_Writer], str]] = {
    "table_invoice": _table_invoice,
    "till_receipt": _till_receipt,
    "letter_invoice": _letter_invoice,
    "credit_note": _credit_note,
    "eu_invoice": _eu_invoice,
    "short_receipt": _short_receipt,
}


def render_document(invoice: Invoice, rng: random.Random) -> str:
    """Render the invoice in its own layout. All variation comes from `rng`."""
    return _LAYOUTS[invoice.layout](_Writer(invoice, rng))
