"""Booking lines and arithmetic checks, computed in code from the model output.

The model copies amounts and picks accounts. All calculation happens here, in
`Decimal`, with halves rounded away from zero.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal, get_args

from adapt.config import Platform, VatRate

ReasonCode = Literal[
    "total_mismatch",
    "near_asset_limit",
    "asset_as_expense",
    "unsupported_chart",
    "unknown_account",
]

CENT = Decimal("0.01")
TOTAL_TOLERANCE = Decimal("0.01")


@dataclass(frozen=True)
class BookingLine:
    """All items with the same account and VAT rate.

    `vat` is the VAT inside the gross amount. On a reverse-charge invoice no VAT is
    printed, so `vat` is zero and `liability` is the VAT the recipient owes.
    """

    account: str
    vat_rate: str
    gross: Decimal
    net: Decimal
    vat: Decimal
    liability: Decimal


@dataclass(frozen=True)
class Booking:
    lines: list[BookingLine]
    reasons: list[ReasonCode]

    @property
    def needs_review(self) -> bool:
        return bool(self.reasons)


def to_decimal(value: Any) -> Decimal:
    """Convert a JSON number to Decimal through its printed form, so 0.1 stays 0.1."""
    return Decimal(str(value))


def round_cents(value: Decimal) -> Decimal:
    return value.quantize(CENT, ROUND_HALF_UP)


def _rate(vat_rate: str) -> Decimal:
    if vat_rate not in get_args(VatRate):
        raise ValueError(f"unknown VAT rate {vat_rate!r}")
    return Decimal(vat_rate) / 100


def net_amount(amount: Decimal, vat_rate: str, reverse_charge: bool) -> Decimal:
    """The net part of a printed amount. On reverse-charge invoices the amount is net."""
    if reverse_charge:
        return amount
    return round_cents(amount / (1 + _rate(vat_rate)))


def booking_lines(items: Sequence[Mapping[str, Any]], reverse_charge: bool) -> list[BookingLine]:
    """Group items by (account, VAT rate) and split each sum into net and VAT."""
    groups: dict[tuple[str, str], Decimal] = {}
    for item in items:
        key = (item["account"], item["vat_rate"])
        groups[key] = groups.get(key, Decimal(0)) + to_decimal(item["amount"])

    lines = []
    for (account, vat_rate), gross in groups.items():
        net = net_amount(gross, vat_rate, reverse_charge)
        if reverse_charge:
            vat, liability = Decimal(0), round_cents(net * _rate(vat_rate))
        else:
            vat, liability = gross - net, Decimal(0)
        lines.append(BookingLine(account, vat_rate, gross, net, vat, liability))
    return lines


def check_booking(output: Mapping[str, Any], chart: str, platform: Platform) -> list[ReasonCode]:
    """Run the arithmetic and rule checks. Each reason code appears once, in a fixed order."""
    reasons: list[ReasonCode] = []
    items = output["items"]
    reverse_charge = output["reverse_charge"]
    accounts = {account.number: account for account in platform.rules.accounts}
    limit = Decimal(str(platform.config.asset_limit.net_eur))
    margin = Decimal(str(platform.config.asset_limit.margin_eur))

    printed = sum((to_decimal(item["amount"]) for item in items), Decimal(0))
    if abs(to_decimal(output["total"]) - printed) > TOTAL_TOLERANCE:
        reasons.append("total_mismatch")

    # Credit notes and discount lines are negative, so the size of the net amount counts.
    nets = [
        (item, abs(net_amount(to_decimal(item["amount"]), item["vat_rate"], reverse_charge)))
        for item in items
    ]
    if any(abs(net - limit) <= margin for _, net in nets):
        reasons.append("near_asset_limit")
    if any(
        net > limit and item["account"] in accounts and not accounts[item["account"]].asset
        for item, net in nets
    ):
        reasons.append("asset_as_expense")
    if chart not in platform.config.charts:
        reasons.append("unsupported_chart")
    if any(item["account"] not in accounts for item in items):
        reasons.append("unknown_account")
    return reasons


def compute_booking(output: Mapping[str, Any], chart: str, platform: Platform) -> Booking:
    """Booking lines and review reasons for one model output."""
    return Booking(
        lines=booking_lines(output["items"], output["reverse_charge"]),
        reasons=check_booking(output, chart, platform),
    )
