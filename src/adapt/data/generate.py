"""Generate invoices together with the output the model must produce.

Every random choice comes from a `random.Random` seeded with the run seed and
the document index, so a document does not depend on how many others are made.
"""

import random
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, get_args

from adapt.config import Layout, Rules, SupplierItem, SupplierType
from adapt.schema import MAX_REASON_LENGTH

CURRENCY = "EUR"
MAX_ITEMS = 6
MIXED_SHARE = 0.4
SUPPLIER_COUNT = 200
YEAR = 2026

CREDIT_NOTE_LAYOUT = "credit_note"
REVERSE_CHARGE_LAYOUT = "eu_invoice"
RECEIPT_LAYOUTS = ("till_receipt", "short_receipt")


@dataclass(frozen=True)
class Supplier:
    name: str
    type_key: str


@dataclass(frozen=True)
class Item:
    description: str
    amount: Decimal
    account: str
    vat_rate: str


@dataclass(frozen=True)
class Invoice:
    supplier: Supplier
    layout: str
    date: date
    number: int
    document_type: str
    items: tuple[Item, ...]
    total: Decimal
    reverse_charge: bool
    target: dict[str, Any]


def make_suppliers(rules: Rules, seed: int, count: int = SUPPLIER_COUNT) -> list[Supplier]:
    """Make `count` suppliers with distinct fictional names, spread over the supplier types."""
    names = rules.supplier_names
    if count > len(names.syllables) * (len(names.syllables) - 1):
        raise ValueError("not enough syllables for that many distinct names")
    rng = random.Random(f"{seed}/suppliers")
    suppliers: list[Supplier] = []
    used: set[str] = set()
    while len(suppliers) < count:
        supplier_type = rules.supplier_types[len(suppliers) % len(rules.supplier_types)]
        base = "".join(rng.sample(names.syllables, rng.choice((2, 3)))).capitalize()
        if base in used:
            continue
        used.add(base)
        suffix = rng.choice(names.suffixes[supplier_type.country])
        suppliers.append(Supplier(f"{base} {suffix}", supplier_type.key))
    return suppliers


def generate_invoices(
    rules: Rules,
    suppliers: Sequence[Supplier],
    seed: int,
    count: int,
    *,
    layouts: Collection[str] | None = None,
    mixed_share: float = MIXED_SHARE,
) -> list[Invoice]:
    """Generate `count` invoices.

    `mixed_share` is the share of documents with more than one VAT rate. Only layouts
    in `layouts` are used (all layouts when it is None).
    """
    types = {t.key: t for t in rules.supplier_types}
    allowed = set(layouts) if layouts is not None else set(get_args(Layout))
    usable = [s for s in suppliers if allowed & set(types[s.type_key].layouts)]
    if not usable:
        raise ValueError("no supplier can use the allowed layouts")
    mixable = [s for s in usable if len(_rates(types[s.type_key])) > 1]
    if mixed_share > 0 and not mixable:
        raise ValueError("no supplier can have mixed VAT rates")
    return [
        _generate_one(rules, usable, mixable, allowed, seed, index, mixed_share)
        for index in range(count)
    ]


def _generate_one(
    rules: Rules,
    usable: list[Supplier],
    mixable: list[Supplier],
    allowed: Collection[str],
    seed: int,
    index: int,
    mixed_share: float,
) -> Invoice:
    rng = random.Random(f"{seed}/invoice/{index}")
    mixed = rng.random() < mixed_share
    return draw_invoice(rules, mixable if mixed else usable, allowed, rng, mixed=mixed)


def draw_invoice(
    rules: Rules,
    suppliers: Sequence[Supplier],
    allowed: Collection[str],
    rng: random.Random,
    *,
    mixed: bool = False,
) -> Invoice:
    """Draw one invoice for one of `suppliers`. Every supplier needs an allowed layout.

    With `mixed` the items have more than one VAT rate, so every supplier needs several rates.
    """
    types = {t.key: t for t in rules.supplier_types}
    weights = [types[s.type_key].weight for s in suppliers]
    supplier = rng.choices(suppliers, weights)[0]
    supplier_type = types[supplier.type_key]
    layout = rng.choice([x for x in supplier_type.layouts if x in allowed])

    sign = -1 if layout == CREDIT_NOTE_LAYOUT else 1
    items = tuple(
        Item(item.description, sign * draw_amount(rng, item.price), item.account, item.vat_rate)
        for item in _pick_items(rng, supplier_type, mixed)
    )
    invoice_date = date(YEAR, 1, 1) + timedelta(days=rng.randrange(365))
    return make_invoice(rules, supplier, layout, invoice_date, rng.randint(1, 9999), items)


def document_type(layout: str) -> str:
    if layout == CREDIT_NOTE_LAYOUT:
        return "credit_note"
    if layout in RECEIPT_LAYOUTS:
        return "receipt"
    return "invoice"


def make_invoice(
    rules: Rules,
    supplier: Supplier,
    layout: str,
    invoice_date: date,
    number: int,
    items: tuple[Item, ...],
) -> Invoice:
    """Build an invoice from its items. Total, reverse charge and target follow from them."""
    doc_type = document_type(layout)
    reverse_charge = layout == REVERSE_CHARGE_LAYOUT
    total = sum((item.amount for item in items), Decimal(0))
    return Invoice(
        supplier=supplier,
        layout=layout,
        date=invoice_date,
        number=number,
        document_type=doc_type,
        items=items,
        total=total,
        reverse_charge=reverse_charge,
        target=_target(rules, supplier, invoice_date, doc_type, items, total, reverse_charge),
    )


def _rates(supplier_type: SupplierType) -> list[str]:
    return sorted({item.vat_rate for item in supplier_type.items})


def _pick_items(rng: random.Random, supplier_type: SupplierType, mixed: bool) -> list[SupplierItem]:
    by_rate = {
        rate: [item for item in supplier_type.items if item.vat_rate == rate]
        for rate in _rates(supplier_type)
    }
    if not mixed:
        rate = rng.choice(list(by_rate))
        return rng.choices(by_rate[rate], k=rng.randint(1, MAX_ITEMS))
    first, second = rng.sample(list(by_rate), 2)
    picked = [rng.choice(by_rate[first]), rng.choice(by_rate[second])]
    picked += rng.choices(supplier_type.items, k=rng.randint(0, MAX_ITEMS - 2))
    rng.shuffle(picked)
    return picked


def draw_amount(rng: random.Random, price: tuple[float, float]) -> Decimal:
    cents = rng.randint(round(price[0] * 100), round(price[1] * 100))
    return Decimal(cents).scaleb(-2)


def _target(
    rules: Rules,
    supplier: Supplier,
    invoice_date: date,
    document_type: str,
    items: tuple[Item, ...],
    total: Decimal,
    reverse_charge: bool,
) -> dict[str, Any]:
    account_names = {account.number: account.name for account in rules.accounts}
    groups = list(dict.fromkeys((item.account, item.vat_rate) for item in items))
    parts = [f"{account_names[account].lower()} at {rate}%" for account, rate in groups]
    return {
        "supplier": supplier.name,
        "document_date": invoice_date.isoformat(),
        "document_type": document_type,
        "currency": CURRENCY,
        "total": float(total),
        "items": [
            {
                "description": item.description,
                "amount": float(item.amount),
                "account": item.account,
                "vat_rate": item.vat_rate,
            }
            for item in items
        ],
        "reverse_charge": reverse_charge,
        "needs_review": False,
        "reason": make_reason(parts, document_type == "credit_note", reverse_charge),
    }


def make_reason(parts: list[str], credit_note: bool, reverse_charge: bool) -> str:
    """Build the reason line from one phrase per (account, rate) group, within the length limit."""
    prefix = "Credit note: " if credit_note else ""
    suffix = "; reverse charge" if reverse_charge else ""
    for kept in range(len(parts), 0, -1):
        omitted = len(parts) - kept
        text = (
            prefix + ", ".join(parts[:kept]) + (f" and {omitted} more" if omitted else "") + suffix
        )
        if len(text) <= MAX_REASON_LENGTH:
            return text[0].upper() + text[1:]
    return (prefix + "several accounts" + suffix)[:MAX_REASON_LENGTH]
