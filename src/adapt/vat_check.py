"""Compare the VAT printed on a document with the VAT of the booked items.

Many documents print their VAT: a net amount and a VAT amount for each rate, a line "incl. VAT
19%: 3.80", or a sentence "The invoice includes VAT of 19% (3.80), 7% (1.40)". The amounts of
the items are gross, so the booked net and VAT of a rate follow from the items. If the two do not
agree, an item was booked at the wrong rate.

The check only says "different" when it read a number it can trust. A line it cannot read, such
as one with a letter in the amount, is ignored, and so is the check for a rate that is not printed.
"""

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

TOLERANCE = Decimal("0.02")

# An amount with two decimals, in either number style: 1,234.56 or 1.234,56.
_AMOUNT = r"-?\d{1,3}(?:[.,]\d{3})*[.,]\d{2}(?!\d)|-?\d+[.,]\d{2}(?!\d)"
# Noise can merge a word into the number before it ("514,63Net 19%"), so a keyword may follow a
# digit, but not a letter.
_NET = re.compile(rf"(?<![a-z])Net\s*(\d{{1,2}})\s*%\s*:?\s*({_AMOUNT})", re.IGNORECASE)
_VAT = re.compile(rf"(?<![a-z])VAT\s*(\d{{1,2}})\s*%\s*:?\s*({_AMOUNT})", re.IGNORECASE)
_VAT_OF_RATE = re.compile(rf"(?<![\dOolIS])(\d{{1,2}})\s*%\s*\(\s*({_AMOUNT})\s*\)")
_VAT_SENTENCE = re.compile(r"includes\s+VAT\s+of\b", re.IGNORECASE)
# The same places with look-alike characters for the digits. They show that a line was printed
# even when its numbers cannot be read.
_LOOSE_LINE = re.compile(r"(?<![a-z])(?:Net|VAT)\s*[\dOolIS]{1,2}\s*%", re.IGNORECASE)
_LOOSE_RATE = re.compile(r"(?<![\dOolIS])[\dOolIS]{1,2}\s*%\s*\(")


def parse_amount(text: str) -> Decimal:
    """`1.234,56` and `1,234.56` both give 1234.56."""
    separator = max(text.rfind("."), text.rfind(","))
    digits = re.sub(r"[.,]", "", text[:separator]) + "." + text[separator + 1 :]
    return Decimal(digits)


def _printed(text: str) -> tuple[dict[str, Decimal], dict[str, Decimal], bool]:
    """The net and the VAT printed for each rate, and whether every such line could be read."""
    net_lines, vat_lines = _NET.findall(text), _VAT.findall(text)
    nets = {rate: parse_amount(amount) for rate, amount in net_lines}
    vats = {rate: parse_amount(amount) for rate, amount in vat_lines}
    expected, read = len(_LOOSE_LINE.findall(text)), len(net_lines) + len(vat_lines)
    sentence = _VAT_SENTENCE.search(text)
    if sentence:
        rest = text[sentence.end() :]
        sentence_rates = _VAT_OF_RATE.findall(rest)
        for rate, amount in sentence_rates:
            vats.setdefault(rate, parse_amount(amount))
        expected += len(_LOOSE_RATE.findall(rest))
        read += len(sentence_rates)
    return nets, vats, expected == read


def _booked(items: Sequence[Mapping[str, Any]]) -> dict[str, tuple[Decimal, Decimal]]:
    """The net and the VAT of the items for each rate. The item amounts are gross."""
    gross: dict[str, Decimal] = defaultdict(Decimal)
    for item in items:
        gross[str(item["vat_rate"])] += Decimal(str(item["amount"]))
    booked = {}
    for rate, amount in gross.items():
        net = amount / (1 + Decimal(rate) / 100)
        booked[rate] = (net, amount - net)
    return booked


def vat_mismatch(
    text: str, items: Sequence[Mapping[str, Any]], reverse_charge: bool = False
) -> bool:
    """True when the VAT printed on the document differs from the VAT of the booked items.

    A reverse-charge document is not checked: the amounts are net and no VAT is charged.
    """
    if reverse_charge or not items:
        return False
    nets, vats, readable = _printed(text)
    if not nets and not vats:
        return False
    booked = _booked(items)
    zero = (Decimal(0), Decimal(0))
    for printed, position in ((nets, 0), (vats, 1)):
        for rate, amount in printed.items():
            if abs(amount - booked.get(rate, zero)[position]) > TOLERANCE:
                return True
        if printed and readable:
            # Every printed line was read, so a booked rate that is not printed is a mistake.
            # A 0% rate may be left out of a document, so it is not looked for.
            for rate, values in booked.items():
                if rate != "0" and rate not in printed and abs(values[position]) > TOLERANCE:
                    return True
    return False
