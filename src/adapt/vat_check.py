"""Compare the VAT printed on a document with the VAT of the booked items.

Many documents print their VAT: a net amount and a VAT amount for each rate, a line "incl. VAT
19%: 3.80" (also "VAT 19% incl.: 3.80", "incl. 19% VAT: 3.80" and "Incl. VAT: 19% 3.80, 7% 1.40"),
a list "VAT 19% 5.29 and 7% 1.24", or a sentence "The invoice includes VAT of 19% (3.80), 7%
(1.40)". The amounts of
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
_AMOUNT = r"-?\d{1,3}(?:[.,]\d{3})*[.,]\d{2}(?![\dOolIS])|-?\d+[.,]\d{2}(?![\dOolIS])"
# Noise can merge a word into the number before it ("514,63Net 19%"), so a keyword may follow a
# digit, but not a letter. "and" may be merged into a keyword too ("2,87 andVAT 7%").
_NET = re.compile(rf"(?<![a-z])Net\s*(\d{{1,2}})\s*%\s*:?\s*({_AMOUNT})", re.IGNORECASE)
_VAT = re.compile(
    rf"(?:(?<![a-z])|(?<=and))VAT\s*(\d{{1,2}})\s*%\s*(?:incl\.?)?\s*:?\s*({_AMOUNT})",
    re.IGNORECASE,
)
# A receipt may print the included VAT as "incl. 19% VAT: 3.80" or as one line "Incl. VAT: 19% 3.80,
# 7% 1.40".
_VAT_RATE_FIRST = re.compile(
    rf"(?<![a-z])incl\.?\s*(\d{{1,2}})\s*%\s*VAT\s*:?\s*({_AMOUNT})", re.IGNORECASE
)
_VAT_LIST = re.compile(r"(?<![a-z])incl\.?\s*VAT\s*:", re.IGNORECASE)
# After a VAT line, the net of the same rate ("VAT 19% 85.84 on net 451.78") or the next rates
# without the word VAT ("VAT 19% 5.29 and 7% 1.24", or "7% 1.24" on the next line).
_ON_NET = re.compile(rf"\s*(?:on|of)\s+net\s*:?\s*({_AMOUNT})", re.IGNORECASE)
_NEXT_RATE = re.compile(rf"\s*(?:,|;|/|and)?\s*(\d{{1,2}})\s*%\s*:?\s*({_AMOUNT})", re.IGNORECASE)
_NEXT_RATE_LOOSE = re.compile(
    r"\s*(?:,|;|/|and)?\s*[\dOolIS]{1,2}\s*%\s*:?\s*-?[\dOolIS][\dOolIS.,]*", re.IGNORECASE
)
_VAT_OF_RATE = re.compile(rf"(?<![\dOolIS])(\d{{1,2}})\s*%\s*\(\s*({_AMOUNT})\s*\)")
_VAT_SENTENCE = re.compile(r"includes\s+VAT\s+of\b", re.IGNORECASE)
# The same places with look-alike characters for the digits. They show that a line was printed
# even when its numbers cannot be read.
_LOOSE_LINE = re.compile(
    r"(?<![a-z])(?:(?:Net|VAT)\s*[\dOolIS]{1,2}\s*%|inc[l1I]\.?\s*[\dOolIS]{1,2}\s*%\s*\W{0,3}VAT)",
    re.IGNORECASE,
)
_LOOSE_RATE = re.compile(r"(?<![\dOolIS])[\dOolIS]{1,2}\s*%\s*\(")


def parse_amount(text: str) -> Decimal:
    """`1.234,56` and `1,234.56` both give 1234.56."""
    separator = max(text.rfind("."), text.rfind(","))
    digits = re.sub(r"[.,]", "", text[:separator]) + "." + text[separator + 1 :]
    return Decimal(digits)


def _next_rates(text: str, position: int, vats: dict[str, Decimal]) -> tuple[int, int]:
    """Read the rates that follow without the word VAT, and return how many were there and read."""
    expected = read = 0
    while loose := _NEXT_RATE_LOOSE.match(text, position):
        expected += 1
        strict = _NEXT_RATE.match(text, position)
        if strict:
            vats.setdefault(strict.group(1), parse_amount(strict.group(2)))
            read += 1
            position = strict.end()
        else:
            position = loose.end()
    return expected, read


def _printed(text: str) -> tuple[dict[str, Decimal], dict[str, Decimal], bool]:
    """The net and the VAT printed for each rate, and whether every such line could be read."""
    net_lines, vat_lines = _NET.findall(text), _VAT.findall(text)
    rate_first = _VAT_RATE_FIRST.findall(text)
    nets = {rate: parse_amount(amount) for rate, amount in net_lines}
    vats = {rate: parse_amount(amount) for rate, amount in vat_lines + rate_first}
    expected = len(_LOOSE_LINE.findall(text))
    read = len(net_lines) + len(vat_lines) + len(rate_first)
    for line in _VAT.finditer(text):
        position = line.end()
        on_net = _ON_NET.match(text, position)
        if on_net:
            nets.setdefault(line.group(1), parse_amount(on_net.group(1)))
            position = on_net.end()
        more_expected, more_read = _next_rates(text, position, vats)
        expected, read = expected + more_expected, read + more_read
    for line in _VAT_LIST.finditer(text):
        more_expected, more_read = _next_rates(text, line.end(), vats)
        expected, read = expected + more_expected, read + more_read
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
