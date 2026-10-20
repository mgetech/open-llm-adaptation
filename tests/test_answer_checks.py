import pytest

from adapt.answer_checks import (
    ANSWER_RULES,
    answer_flags,
    mixed_private,
    private_accounts,
    reverse_charge_cue,
)
from adapt.config import load_platform


@pytest.fixture(scope="module")
def private():
    return private_accounts(load_platform("tax-booking").rules.accounts)


def answer(*accounts, reverse_charge=False):
    items = [
        {"description": "x", "amount": 10.0, "account": account, "vat_rate": "19"}
        for account in accounts
    ]
    return {"items": items, "reverse_charge": reverse_charge}


def test_the_private_accounts_are_the_group_private(private):
    assert private == {"1800"}


@pytest.mark.parametrize(
    ("accounts", "expected"),
    [
        (("1800", "4930"), True),
        (("4930", "1800", "4940"), True),
        (("1800", "1800"), False),
        (("4930", "4940"), False),
        (("4930",), False),
        ((), False),
    ],
)
def test_a_private_and_a_business_account_together_are_mixed(private, accounts, expected):
    assert mixed_private(answer(*accounts), private) is expected


@pytest.mark.parametrize(
    "text",
    [
        "Reverse charge: tax liability of the recipient",
        "REVERSE-CHARGE applies",
        "Tax liability rests with the customer.",
        "Steuerschuldnerschaft des Leistungsempfängers nach § 13b UStG",
        "Supply under §13b UStG",
        "Pursuant to Article 196 of Directive 2006/112/EC",
        "See art. 196",
        "Intra-community supply, VAT exempt",
        "Intracommunity acquisition",
        "VAT is not charged; the customer accounts for the tax in its own country.",
        "The recipient must account for the VAT.",
        "VAT is self-assessed by the buyer",
    ],
)
def test_the_reverse_charge_wordings_are_found(text):
    assert reverse_charge_cue(text, answer("4930"))


@pytest.mark.parametrize(
    "text",
    [
        "Net 19%: 20.00  VAT 19%: 3.80",
        "Invoice 196 for the order, part 196-B",
        "Charge for the card terminal",
        "Total EUR 45.20",
    ],
)
def test_ordinary_documents_have_no_cue(text):
    assert not reverse_charge_cue(text, answer("4930"))


def test_a_cue_is_no_flag_when_the_answer_says_reverse_charge():
    text = "Reverse charge: tax liability of the recipient"
    assert not reverse_charge_cue(text, answer("4930", reverse_charge=True))


def test_answer_flags_name_each_rule_that_fires(private):
    text = "Reverse charge\nTotal EUR 20.00"
    assert answer_flags(text, answer("1800", "4930"), private) == [
        "mixed_private",
        "reverse_charge_cue",
    ]
    assert answer_flags("Total EUR 20.00", answer("4930", "4940"), private) == []


def test_answer_flags_apply_only_the_rules_asked_for(private):
    text = "Reverse charge\nTotal EUR 20.00"
    both = answer("1800", "4930")
    assert answer_flags(text, both, private, ("mixed_private",)) == ["mixed_private"]
    assert answer_flags(text, both, private, ("reverse_charge_cue",)) == ["reverse_charge_cue"]
    assert answer_flags(text, both, private, ("vat_mismatch",)) == []


def test_an_answer_without_items_has_nothing_to_check(private):
    assert answer_flags("Reverse charge", answer(), private) == []


def test_the_rules_are_listed_in_one_order():
    assert ANSWER_RULES == ("vat_mismatch", "mixed_private", "reverse_charge_cue")
