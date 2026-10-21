"""Checks on the answer of the model that send a document to review.

Each check looks at the document text and at the answer, never at the model itself, so it can be
applied again to saved answers (see `adapt.eval.rescore`). They are pipeline rules, reported apart
from the results of the model:

- `vat_mismatch`: the VAT printed on the document differs from the VAT of the booked items.
- `mixed_private`: the answer books a private account and a business account on one document.
  A receipt that mixes private and business items is a review case, not something to split.
- `reverse_charge_cue`: the text names the reverse charge, and the answer says there is none.
"""

import re
from collections.abc import Collection, Mapping
from typing import Any

from adapt.vat_check import vat_mismatch

ANSWER_RULES = ("vat_mismatch", "mixed_private", "reverse_charge_cue")

# Wordings that a supplier uses when the buyer owes the VAT. They are written for any supplier,
# not taken from the generated documents.
_REVERSE_CHARGE = re.compile(
    "|".join(
        [
            r"reverse[\s-]+charge",
            r"tax\s+liability\s+(?:of|rests\s+with|lies\s+with)\s+the\s+"
            r"(?:recipient|customer|buyer|purchaser)",
            r"§\s*13b",
            r"\b13b\s+UStG",
            # An article number on an item line also looks like "Art. 196", so the number counts
            # only when a legal reference follows.
            r"\bart(?:icle)?\.?\s*196\b(?=\s*(?:of\b|VAT\b|Directive\b|Council\b|EU\b|EC\b|MwSt|2006))",
            r"2006/112",
            r"intra[\s-]*community\s+(?:supply|acquisition|delivery)",
            r"steuerschuldnerschaft",
            r"(?:customer|recipient|buyer|purchaser)\s+(?:must\s+|has\s+to\s+)?accounts?\s+for\s+"
            r"(?:the\s+)?(?:value[\s-]*added\s+)?(?:tax|VAT)",
            r"VAT\s+(?:is\s+|to\s+be\s+)?self[\s-]*assessed",
        ]
    ),
    re.IGNORECASE,
)


def private_accounts(accounts: Collection[Any]) -> frozenset[str]:
    """The numbers of the accounts in the group `private`."""
    return frozenset(a.number for a in accounts if a.group == "private")


def mixed_private(output: Mapping[str, Any], private: Collection[str]) -> bool:
    """True when the items use a private account and an account that is not private."""
    kinds = {item["account"] in private for item in output["items"]}
    return kinds == {True, False}


def reverse_charge_cue(text: str, output: Mapping[str, Any]) -> bool:
    """True when the text names the reverse charge and the answer has `reverse_charge` false."""
    return not output.get("reverse_charge") and _REVERSE_CHARGE.search(text) is not None


def answer_flags(
    text: str,
    output: Mapping[str, Any],
    private: Collection[str],
    rules: Collection[str] = ANSWER_RULES,
) -> list[str]:
    """The rules in `rules` that send this answer to review, in the order of `ANSWER_RULES`.

    An answer without items is a review case already and has nothing to check."""
    if not output["items"]:
        return []
    fired = {
        "vat_mismatch": lambda: vat_mismatch(
            text, output["items"], bool(output.get("reverse_charge"))
        ),
        "mixed_private": lambda: mixed_private(output, private),
        "reverse_charge_cue": lambda: reverse_charge_cue(text, output),
    }
    return [rule for rule in ANSWER_RULES if rule in rules and fired[rule]()]
