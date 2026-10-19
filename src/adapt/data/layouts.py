"""Render an invoice as document text in one of twelve layouts.

The text is what the model reads. Number format, date format, spacing and line
breaks change from document to document, but every item amount and the total
always appear in the text. An invoice of a supplier in the EU or outside it is
net, prints no VAT lines and carries a reverse-charge note in any layout.
"""

import random
from collections.abc import Callable
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from adapt.data.generate import RECEIPT_LAYOUTS, Invoice, Item

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
    lambda i: f"R{i.number:05d}",
    lambda i: f"{i.date.year % 100:02d}/{i.number:04d}",
)
TOTAL_LABELS = (
    "Total",
    "Total amount",
    "Grand total",
    "Total due",
    "Invoice total",
    "Total payment",
    "Total amount due",
)
RECEIPT_TOTAL_LABELS = ("Total", "Total", "Total sum", "Total to pay", "Total due", "Grand total")
CREDIT_TOTAL_LABELS = ("Credit total", "Total credit", "Total credit amount", "Credit note total")
NET_TOTAL_LABELS = ("Total", "Total net", "Total net amount", "Net total", "Total net due")
REVERSE_CHARGE_NOTE = "Reverse charge: tax liability of the recipient"
REVERSE_CHARGE_NOTES = (
    REVERSE_CHARGE_NOTE,
    REVERSE_CHARGE_NOTE,
    REVERSE_CHARGE_NOTE,
    "Reverse charge",
    "VAT reverse charge applies",
    "Reverse charge - VAT to be accounted for by the recipient",
    "Reverse-charge procedure: the recipient is liable for the VAT",
    "Reverse charge according to Art. 196 of the VAT Directive",
    "Tax liability of the recipient (reverse charge)",
    "Reverse charge supply, no VAT charged",
    "Reverse charge: the customer accounts for the VAT",
    "VAT reverse charge: customer to account for the tax",
)
IDENTIFIER_SHARE = 0.3  # share of documents that carry a line of fake identifiers
IDENTIFIER_LABELS = ("IBAN", "Tax ID", "Tax number", "Email", "Phone", "Tel.")
FOOTERS = (
    "Thank you for your visit",
    "Have a nice day",
    "See you again soon",
    "Thank you for shopping with us",
    "Goods are not returnable without this receipt",
    "Open Monday to Saturday",
)
PAYMENT_TERMS = (
    "Payment terms: 14 days net",
    "Payment terms: 30 days net",
    "Payable within 14 days without deduction",
    "Please transfer the amount within 10 days",
    "Payment due on receipt",
    "2% discount if paid within 7 days",
)
SHOP_LINES = ("Cashier 3", "Store 0412", "Till 2", "Terminal 7", "Served by Anna", "Self checkout")
LETTER_NOTES = (
    "Payment is due within 14 days without deduction.",
    "Please quote the invoice number with your payment.",
    "Thank you for your order.",
    "We appreciate your business.",
    "Please contact us if you have any questions about this invoice.",
    "Delivery was made as agreed.",
)
LETTER_OPENINGS = ("Dear Sir or Madam,", "Hello,", "Dear customer,", "Good day,")

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


def split_quantity(amount: Decimal, rng: random.Random) -> tuple[int, Decimal]:
    """A quantity and a unit price whose product is `amount`. The quantity is 1 if none fits."""
    cents = abs(int(amount * 100))
    options = [q for q in range(2, 7) if cents % q == 0 and cents // q >= 20]
    if not options:
        return 1, amount
    quantity = rng.choice(options)
    return quantity, amount / quantity


class _Writer:
    """The choices that stay the same inside one document."""

    def __init__(self, invoice: Invoice, rng: random.Random, note: str | None = None):
        self.invoice = invoice
        self.rng = rng
        self.foreign = invoice.reverse_charge
        self.note = note or rng.choice(REVERSE_CHARGE_NOTES)
        self.style = rng.choice(NUMBER_STYLES)
        self.date = rng.choice(list(DATE_FORMATS.values()))(invoice.date)
        self.currency = rng.choice(("EUR", "€"))
        self.fill = rng.choice(("spaces", "dots", "colon"))
        self.number = rng.choice(NUMBER_PATTERNS)(invoice)
        self.quantities = rng.random() < 0.3
        self.articles = rng.random() < 0.25
        self.upper = invoice.layout in RECEIPT_LAYOUTS and rng.random() < 0.35
        self.vat_style = rng.randrange(4)
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

    def article(self) -> str:
        label = self.rng.choice(("Art.", "SKU", "Item no.", "Ref"))
        digits = "".join(self.rng.choice("0123456789") for _ in range(self.rng.randint(3, 5)))
        return f"{label} {digits}"

    def label(self, item: Item) -> str:
        """The item text, with an article number in some documents."""
        text = item.description.upper() if self.upper else item.description
        return f"{self.article()} {text}" if self.articles else text

    def item_line(self, item: Item) -> str:
        text = self.label(item)
        if self.quantities:
            text = f"{self.rng.randint(1, 3)}x {text}"
        return self.line(text, item.amount)

    def quantity_line(self, item: Item) -> str:
        """`desc  3 x 4.50  13.50`: the quantity and the unit price, with the line amount."""
        quantity, unit = split_quantity(item.amount, self.rng)
        text = f"{self.label(item)}  {quantity} x {self.money(unit)}"
        return self.line(text, item.amount)

    def total_text(self, labels: tuple[str, ...]) -> str:
        if self.foreign:
            labels = NET_TOTAL_LABELS
        return self.rng.choice(labels)

    def total_line(self, labels: tuple[str, ...]) -> str:
        label = self.total_text(labels)
        value = self.money(self.invoice.total)
        if self.rng.random() < 0.5:
            return f"{label} {self.currency} {value}"
        return f"{label}: {value} {self.currency}"

    def vat_lines(self) -> str:
        """The net and the VAT per rate, in one of four wordings. Nothing for a foreign supplier."""
        if self.foreign:
            return ""
        rows = vat_summary(self.invoice)
        if self.vat_style == 0:
            return "\n".join(
                f"Net {rate}%: {self.money(net)}  VAT {rate}%: {self.money(vat)}"
                for rate, net, vat in rows
            )
        if self.vat_style == 1:
            return "\n".join(
                f"Net {rate}%: {self.money(net)}\nVAT {rate}%: {self.money(vat)}"
                for rate, net, vat in rows
            )
        if self.vat_style == 2:
            return "\n".join(
                f"VAT {rate}%: {self.money(vat)} (net {self.money(net)})" for rate, net, vat in rows
            )
        return "\n".join(
            f"Net {rate}% {self.money(net)} | VAT {rate}% {self.money(vat)}"
            for rate, net, vat in rows
        )

    def included_vat_lines(self) -> str:
        """The VAT inside the gross amounts, as a receipt prints it."""
        rows = vat_summary(self.invoice)
        if self.vat_style == 0:
            return "\n".join(f"incl. VAT {rate}%: {self.money(vat)}" for rate, _, vat in rows)
        if self.vat_style == 1:
            return "\n".join(f"incl. {rate}% VAT: {self.money(vat)}" for rate, _, vat in rows)
        if self.vat_style == 2:
            parts = ", ".join(f"{rate}% {self.money(vat)}" for rate, _, vat in rows)
            return f"Incl. VAT: {parts}"
        return "\n".join(f"VAT {rate}% incl.: {self.money(vat)}" for rate, _, vat in rows)

    def payment_lines(self) -> str:
        """How the receipt was paid: nothing, a card line, or cash with the change."""
        total = self.invoice.total
        choice = self.rng.random()
        if choice < 0.4 or total <= 0:
            return ""
        if choice < 0.6:
            return self.rng.choice(("Paid by card", "Card payment", "Contactless"))
        step = self.rng.choice((5, 10, 20, 50))
        cash = (total / step).to_integral_value(rounding="ROUND_CEILING") * step
        if cash == total:
            cash += step
        return f"Cash {self.money(cash)}\nChange {self.money(cash - total)}"

    def due_line(self) -> str:
        """An extra date on some invoices, so the model must find the invoice date itself."""
        if self.rng.random() < 0.7:
            return ""
        due = self.invoice.date + timedelta(days=self.rng.choice((7, 14, 30)))
        formatted = self.rng.choice(list(DATE_FORMATS.values()))(due)
        return f"Due date: {formatted}"

    def terms(self) -> str:
        return self.rng.choice(PAYMENT_TERMS) if self.rng.random() < 0.35 else ""

    def join(self, sections: list[str]) -> str:
        sections = [section for section in sections if section]
        text = sections[0]
        for section in sections[1:]:
            text += ("\n\n" if self.rng.random() < 0.2 else "\n") + section
        return text


def _items(w: _Writer) -> str:
    rng = w.rng
    lines = []
    quantity_rows = rng.random() < 0.2
    for item in w.invoice.items:
        lines.append(w.quantity_line(item) if quantity_rows else w.item_line(item))
    return "\n".join(lines)


def _table_invoice(w: _Writer) -> list[str]:
    word = w.rng.choice(("Invoice", "Invoice no.", "Invoice number", "Inv.", "Invoice No"))
    if w.rng.random() < 0.5:
        head = f"{w.name}  {word} {w.number}  {w.date}"
    else:
        head = f"{w.name}\n{word} {w.number}\nDate: {w.date}"
    return [head, w.due_line(), _items(w), w.vat_lines(), w.total_line(TOTAL_LABELS), w.terms()]


def _column_invoice(w: _Writer) -> list[str]:
    """The columns of a table cut apart: the headings, then every value on its own line."""
    rng = w.rng
    head = f"{w.name}\nInvoice {w.number}\nDate: {w.date}"
    headings = rng.choice(
        (
            ("Qty", "Item", "Unit price", "Amount"),
            ("Pos", "Description", "Price", "Total"),
            ("Quantity", "Description", "Unit price", "Line amount"),
        )
    )
    cells = list(headings)
    for position, item in enumerate(w.invoice.items, start=1):
        quantity, unit = split_quantity(item.amount, rng)
        first = str(position) if headings[0] == "Pos" else str(quantity)
        cells += [first, w.label(item), w.money(unit), w.money(item.amount)]
    return [head, "\n".join(cells), w.vat_lines(), w.total_line(TOTAL_LABELS), w.terms()]


def _service_invoice(w: _Writer) -> list[str]:
    """Rows with pipes between the cells, as a service invoice with positions."""
    rng = w.rng
    head = rng.choice(
        (
            f"{w.name}\nInvoice {w.number} | {w.date}",
            f"{w.name} - Invoice {w.number}\nDate: {w.date}",
        )
    )
    rows = ["Pos | Description | Qty | Amount"]
    for position, item in enumerate(w.invoice.items, start=1):
        quantity, _ = split_quantity(item.amount, rng)
        rows.append(f"{position} | {w.label(item)} | {quantity} | {w.money(item.amount)}")
    return ["\n".join([head, *rows]), w.vat_lines(), w.total_line(TOTAL_LABELS), w.terms()]


def _email_invoice(w: _Writer) -> list[str]:
    rng = w.rng
    bullet = rng.choice(("- ", "* ", ""))
    lines = [f"{bullet}{w.line(w.label(item), item.amount)}" for item in w.invoice.items]
    vat = ""
    if not w.foreign and rng.random() < 0.6:
        rates = ", ".join(f"{rate}% ({w.money(v)})" for rate, _, v in vat_summary(w.invoice))
        vat = f"The invoice includes VAT of {rates}."
    return [
        f"Subject: Invoice {w.number} from {w.invoice.supplier.name}\nDate: {w.date}",
        f"{rng.choice(LETTER_OPENINGS)}\nplease find our invoice below.",
        "\n".join(lines),
        vat,
        w.total_line(TOTAL_LABELS),
        f"{rng.choice(('Best regards', 'Kind regards', 'Regards'))}\n{w.invoice.supplier.name}",
    ]


def _statement(w: _Writer) -> list[str]:
    """A statement with a short date before each line, like a monthly bill."""
    rng = w.rng
    day = w.invoice.date.day
    rows = []
    for item in w.invoice.items:
        when = date(w.invoice.date.year, w.invoice.date.month, rng.randint(1, day))
        rows.append(w.line(f"{when.day:02d}.{when.month:02d}. {w.label(item)}", item.amount))
    title = rng.choice(("Statement", "Monthly statement", "Account statement"))
    head = f"{w.name}\n{title} {w.number}  {w.date}"
    return [head, "\n".join(rows), w.vat_lines(), w.total_line(TOTAL_LABELS), w.terms()]


def _shop_head(w: _Writer, label: str = "Receipt") -> str:
    rng = w.rng
    when = w.date
    if rng.random() < 0.4:
        when = f"{when} {rng.randint(7, 21):02d}:{rng.randint(0, 59):02d}"
    extra = f"\n{rng.choice(SHOP_LINES)}" if rng.random() < 0.3 else ""
    return f"{w.name}\n{label} {w.number}  {when}{extra}"


def _till_receipt(w: _Writer) -> list[str]:
    footer = w.rng.choice(FOOTERS)
    return [
        _shop_head(w),
        _items(w),
        "-" * 24,
        w.total_line(RECEIPT_TOTAL_LABELS),
        w.included_vat_lines(),
        w.payment_lines(),
        footer,
    ]


def _tax_code_receipt(w: _Writer) -> list[str]:
    """Each item ends with a tax code. A table at the bottom says what each code means."""
    rng = w.rng
    rows = vat_summary(w.invoice)
    codes = dict(zip((rate for rate, _, _ in rows), rng.sample("ABC", len(rows)), strict=True))
    lines = [
        w.line(w.label(item), item.amount) + f" {codes[item.vat_rate]}" for item in w.invoice.items
    ]
    table = ["Tax  Rate  Net  VAT"]
    for rate, net, vat in sorted(rows, key=lambda r: codes[r[0]]):
        table.append(f"{codes[rate]}  {rate}%  {w.money(net)}  {w.money(vat)}")
    return [
        _shop_head(w, "Receipt"),
        "\n".join(lines),
        "-" * 24,
        w.total_line(RECEIPT_TOTAL_LABELS),
        "\n".join(table),
        w.payment_lines(),
        rng.choice(FOOTERS),
    ]


def _quantity_receipt(w: _Writer) -> list[str]:
    """Two lines for each item: the item, then the quantity times the unit price."""
    rng = w.rng
    lines = []
    for item in w.invoice.items:
        quantity, unit = split_quantity(item.amount, rng)
        lines.append(w.label(item))
        lines.append(w.line(f"  {quantity} x {w.money(unit)}", item.amount))
    return [
        _shop_head(w),
        "\n".join(lines),
        w.total_line(RECEIPT_TOTAL_LABELS),
        w.included_vat_lines(),
        w.payment_lines(),
        rng.choice(FOOTERS),
    ]


def _short_receipt(w: _Writer) -> list[str]:
    invoice = w.invoice
    if len(invoice.items) == 1 and w.rng.random() < 0.5:
        item = invoice.items[0]
        value = w.money(item.amount)
        total = w.money(invoice.total)
        line = f"{w.label(item)} {value} {w.currency}"
        return [f"{w.name} {w.date}  {line}  Total {total} {w.currency}"]
    items = "\n".join(w.line(w.label(item), item.amount) for item in invoice.items)
    return [
        f"{w.name}\n{w.date}",
        items,
        w.total_line(RECEIPT_TOTAL_LABELS),
        w.included_vat_lines(),
        w.payment_lines(),
    ]


def _letter_invoice(w: _Writer) -> list[str]:
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
                    f"{item.description} is invoiced at {value}.",
                    f"The charge for {item.description} is {value}.",
                )
            )
        )
    vat = ""
    if not w.foreign:
        rates = ", ".join(
            f"{rate}% ({w.money(amount)})" for rate, _, amount in vat_summary(w.invoice)
        )
        vat = f"The invoice includes VAT of {rates}."
    total = w.money(w.invoice.total)
    kind = "total net amount" if w.foreign else rng.choice(("total", "total amount"))
    return [
        rng.choice(LETTER_OPENINGS),
        f"with this letter we send you our invoice {w.number} dated {w.date}.",
        " ".join(sentences),
        vat,
        f"The {kind} is {w.currency} {total}.",
        " ".join(rng.sample(LETTER_NOTES, k=2)),
        f"Kind regards\n{w.invoice.supplier.name}",
    ]


def _credit_note(w: _Writer) -> list[str]:
    rng = w.rng
    head = f"{w.name}  Credit note {w.number}  {w.date}"
    reference = ""
    if rng.random() < 0.4:
        earlier = w.invoice.date - timedelta(days=rng.randint(5, 60))
        shown = rng.choice(list(DATE_FORMATS.values()))(earlier)
        reference = f"Reference: invoice {rng.randint(1, 9999):04d} dated {shown}"
    intro = rng.choice(
        (
            "We credit you for the following items:",
            "Credit for returned goods:",
            "Price correction, we credit:",
        )
    )
    return [head, reference, intro, _items(w), w.vat_lines(), w.total_line(CREDIT_TOTAL_LABELS)]


def _eu_invoice(w: _Writer) -> list[str]:
    head = f"{w.name}  Invoice {w.number}  {w.date}"
    return [head, _items(w), w.total_line(NET_TOTAL_LABELS)]


_LAYOUTS: dict[str, Callable[[_Writer], list[str]]] = {
    "table_invoice": _table_invoice,
    "till_receipt": _till_receipt,
    "letter_invoice": _letter_invoice,
    "credit_note": _credit_note,
    "eu_invoice": _eu_invoice,
    "short_receipt": _short_receipt,
    "column_invoice": _column_invoice,
    "service_invoice": _service_invoice,
    "email_invoice": _email_invoice,
    "statement": _statement,
    "tax_code_receipt": _tax_code_receipt,
    "quantity_receipt": _quantity_receipt,
}


def _digits(rng: random.Random, count: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(count))


def _identifier_line(w: _Writer) -> str:
    """A line of fake identifiers. The email domain and phone numbers are reserved for examples."""
    rng = w.rng
    iban = (
        f"DE{_digits(rng, 2)} "
        + " ".join(_digits(rng, 4) for _ in range(4))
        + f" {_digits(rng, 2)}"
    )
    tax_id = rng.choice("123456789") + _digits(rng, 10)
    if rng.random() < 0.5:
        tax_id = f"{tax_id[:2]} {tax_id[2:5]} {tax_id[5:8]} {tax_id[8:]}"
    domain = w.invoice.supplier.name.split()[0].lower()
    email = f"{rng.choice(('info', 'office', 'billing'))}@{domain}.example"
    phone = rng.choice(("+49 30 23125 ", "030 23125 ")) + _digits(rng, 3)
    parts = [
        f"IBAN {iban}",
        f"{rng.choice(('Tax ID', 'Tax number'))} {tax_id}",
        f"Email {email}",
        f"{rng.choice(('Phone', 'Tel.'))} {phone}",
    ]
    if w.foreign:
        country = rng.choice(("FR", "NL", "IT", "ES", "PL", "CZ", "BE", "AT", "GB", "US"))
        parts.append(f"VAT ID {country} {_digits(rng, rng.choice((8, 9)))}")
    chosen = [part for part in parts if rng.random() < 0.6] or [parts[0]]
    return rng.choice((" | ", "  ", "\n")).join(chosen)


def render_document(invoice: Invoice, rng: random.Random, note: str | None = None) -> str:
    """Render the invoice in its own layout. All variation comes from `rng`.

    The invoice of a supplier outside Germany carries a reverse-charge note, `note` or a
    standard one, at a random place between its parts. Some documents end with a line of fake
    identifiers (IBAN, tax ID, email, phone).
    """
    writer = _Writer(invoice, rng, note)
    sections = [section for section in _LAYOUTS[invoice.layout](writer) if section]
    if invoice.reverse_charge:
        sections.insert(rng.randint(1, len(sections)), writer.note)
    text = writer.join(sections)
    if rng.random() < IDENTIFIER_SHARE:
        text = writer.join([text, _identifier_line(writer)])
    return text
