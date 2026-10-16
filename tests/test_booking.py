import json
from decimal import Decimal

import pytest

from adapt.booking import (
    booking_lines,
    check_booking,
    compute_booking,
    round_cents,
    to_decimal,
)
from adapt.config import load_platform
from adapt.data.gold import gold_path
from adapt.data.split import build_splits


@pytest.fixture(scope="module")
def platform():
    return load_platform("tax-booking")


def item(amount, account="4930", vat_rate="19", description="Item"):
    return {"description": description, "amount": amount, "account": account, "vat_rate": vat_rate}


def output(items, total=None, reverse_charge=False):
    if total is None:
        total = float(sum(to_decimal(i["amount"]) for i in items))
    return {"items": items, "total": total, "reverse_charge": reverse_charge}


def reasons(platform, items, total=None, reverse_charge=False, chart="SKR03"):
    return check_booking(output(items, total, reverse_charge), chart, platform)


def test_decimal_conversion_keeps_the_printed_value():
    assert to_decimal(0.1) == Decimal("0.1")
    assert to_decimal(19.99) == Decimal("19.99")


def test_net_and_vat_for_one_rate():
    lines = booking_lines([item(119.0)], reverse_charge=False)
    assert len(lines) == 1
    line = lines[0]
    assert (line.account, line.vat_rate) == ("4930", "19")
    assert (line.gross, line.net, line.vat, line.liability) == (
        Decimal("119.00"),
        Decimal("100.00"),
        Decimal("19.00"),
        Decimal("0"),
    )


def test_items_with_the_same_account_and_rate_are_summed_into_one_line():
    items = [item(23.8), item(11.9), item(5.0, vat_rate="7")]
    lines = booking_lines(items, reverse_charge=False)
    assert [(line.account, line.vat_rate, line.gross) for line in lines] == [
        ("4930", "19", Decimal("35.7")),
        ("4930", "7", Decimal("5.0")),
    ]


def test_mixed_rates_give_one_line_per_account_and_rate(platform):
    items = [
        item(23.8, "4930", "19", "Printer paper A4"),
        item(21.4, "4940", "7", "Tax handbook 2026"),
        item(11.9, "4980", "19", "Cleaner"),
        item(10.7, "4980", "7", "Tea"),
    ]
    booking = compute_booking(output(items), "SKR03", platform)
    assert [(line.account, line.vat_rate) for line in booking.lines] == [
        ("4930", "19"),
        ("4940", "7"),
        ("4980", "19"),
        ("4980", "7"),
    ]
    first, second = booking.lines[0], booking.lines[1]
    assert (first.net, first.vat) == (Decimal("20.00"), Decimal("3.80"))
    assert (second.net, second.vat) == (Decimal("20.00"), Decimal("1.40"))
    assert booking.reasons == []
    assert not booking.needs_review


def test_the_net_and_the_vat_always_add_up_to_the_gross():
    for cents in range(1, 3000):
        for rate in ("19", "7", "0"):
            line = booking_lines([item(cents / 100, vat_rate=rate)], False)[0]
            assert line.net + line.vat == line.gross


def test_a_half_cent_rounds_away_from_zero():
    assert round_cents(Decimal("0.005")) == Decimal("0.01")
    assert round_cents(Decimal("-0.005")) == Decimal("-0.01")


def test_a_credit_note_gives_negative_amounts():
    lines = booking_lines([item(-59.5)], reverse_charge=False)
    assert (lines[0].gross, lines[0].net, lines[0].vat) == (
        Decimal("-59.5"),
        Decimal("-50.00"),
        Decimal("-9.50"),
    )


def test_a_credit_note_passes_the_checks(platform):
    assert reasons(platform, [item(-59.5)]) == []


def test_a_discount_line_reduces_the_group(platform):
    items = [item(119.0), item(-11.9)]
    lines = booking_lines(items, reverse_charge=False)
    assert lines[0].gross == Decimal("107.1")
    assert lines[0].net == Decimal("90.00")
    assert reasons(platform, items) == []


def test_reverse_charge_keeps_the_amount_as_net_and_computes_the_liability(platform):
    items = [item(250.0, "4964", "19"), item(100.0, "3300", "7")]
    booking = compute_booking(output(items, reverse_charge=True), "SKR03", platform)
    software, goods = booking.lines
    assert (software.net, software.vat, software.liability) == (
        Decimal("250.0"),
        Decimal("0"),
        Decimal("47.50"),
    )
    assert (goods.net, goods.vat, goods.liability) == (
        Decimal("100.0"),
        Decimal("0"),
        Decimal("7.00"),
    )
    assert booking.reasons == []


def test_reverse_charge_liability_is_rounded_to_cents():
    line = booking_lines([item(33.33, vat_rate="19")], reverse_charge=True)[0]
    assert line.liability == Decimal("6.33")


def test_a_matching_total_is_clean(platform):
    assert reasons(platform, [item(10.0), item(5.5)], total=15.5) == []


def test_a_total_one_cent_off_is_accepted(platform):
    assert reasons(platform, [item(10.0), item(5.5)], total=15.51) == []


def test_a_total_two_cents_off_is_a_mismatch(platform):
    assert reasons(platform, [item(10.0), item(5.5)], total=15.52) == ["total_mismatch"]
    assert reasons(platform, [item(10.0), item(5.5)], total=15.48) == ["total_mismatch"]


def test_items_missing_while_a_total_is_present_is_a_mismatch(platform):
    assert reasons(platform, [], total=85.4) == ["total_mismatch"]


def test_no_items_and_no_total_is_not_a_mismatch(platform):
    assert reasons(platform, [], total=0.0) == []


def test_a_net_amount_near_the_limit_needs_review(platform):
    # gross 975.80 at 19% is net 820.00, which is within 50 of 800
    assert reasons(platform, [item(975.8, "0410")]) == ["near_asset_limit"]


def test_the_margin_edges_count_as_near(platform):
    assert reasons(platform, [item(892.5, "0410")]) == ["near_asset_limit"]  # net 750.00
    assert reasons(platform, [item(1011.5, "0410")]) == ["near_asset_limit"]  # net 850.00


def test_just_outside_the_margin_is_not_near(platform):
    assert reasons(platform, [item(892.49, "0480")]) == []  # net 749.99
    assert reasons(platform, [item(1011.51, "0410")]) == []  # net 850.01


def test_near_the_limit_is_measured_on_the_net_not_the_gross(platform):
    # gross 800.00 is net 672.27, well below the margin
    assert reasons(platform, [item(800.0, "0480")]) == []


def test_near_the_limit_on_a_reverse_charge_invoice_uses_the_printed_amount(platform):
    assert reasons(platform, [item(780.0, "4964")], reverse_charge=True) == ["near_asset_limit"]
    assert reasons(platform, [item(740.0, "0480")], reverse_charge=True) == []


def test_a_credit_note_near_the_limit_needs_review(platform):
    assert reasons(platform, [item(-975.8, "0410")]) == ["near_asset_limit"]


def test_an_asset_booked_as_expense_needs_review(platform):
    # gross 1785.00 at 19% is net 1500.00, booked to office supplies
    assert reasons(platform, [item(1785.0, "4930")]) == ["asset_as_expense"]


def test_the_same_amount_on_an_asset_account_is_clean(platform):
    assert reasons(platform, [item(1785.0, "0410")]) == []
    assert reasons(platform, [item(1785.0, "0650")]) == []


def test_an_asset_booked_as_expense_on_a_reverse_charge_invoice(platform):
    assert reasons(platform, [item(1500.0, "4964")], reverse_charge=True) == ["asset_as_expense"]


def test_just_above_the_limit_as_expense_gives_both_reasons(platform):
    # net 820.00 is above the limit and inside the margin
    assert reasons(platform, [item(975.8, "4930")]) == ["near_asset_limit", "asset_as_expense"]


def test_the_limit_check_looks_at_each_item_not_the_group_sum(platform):
    items = [item(595.0, "4930"), item(595.0, "4930")]  # two items of net 500.00
    assert reasons(platform, items) == []


def test_an_unsupported_chart_needs_review(platform):
    assert reasons(platform, [item(10.0)], chart="SKR04") == ["unsupported_chart"]


def test_an_unknown_account_needs_review(platform):
    assert reasons(platform, [item(10.0, "9999")]) == ["unknown_account"]


def test_an_unknown_account_is_not_also_reported_as_an_expense(platform):
    assert reasons(platform, [item(5000.0, "9999")]) == ["unknown_account"]


def test_several_failures_are_all_listed_once_in_a_fixed_order(platform):
    items = [item(1785.0, "4930"), item(1785.0, "4930"), item(10.0, "9999")]
    assert reasons(platform, items, total=1.0, chart="SKR04") == [
        "total_mismatch",
        "asset_as_expense",
        "unsupported_chart",
        "unknown_account",
    ]


def test_an_unknown_vat_rate_raises(platform):
    with pytest.raises(ValueError, match="unknown VAT rate"):
        booking_lines([item(10.0, vat_rate="16")], reverse_charge=False)
    with pytest.raises(ValueError, match="unknown VAT rate"):
        reasons(platform, [item(10.0, vat_rate="16")])


def test_a_missing_field_raises_instead_of_passing_silently(platform):
    with pytest.raises(KeyError):
        compute_booking({"items": [], "total": 0.0}, "SKR03", platform)


def test_every_generated_target_gets_the_expected_reasons(platform):
    splits = build_splits(platform, 0)
    checked = 0
    for records in splits.values():
        for r in records:
            found = set(check_booking(r["target"], r["chart"], platform))
            # A flagged document has no items, so a printed total cannot be matched.
            unmatched = {"total_mismatch"} if not r["target"]["items"] else set()
            if r["category"] == "unsupported_chart":
                assert "unsupported_chart" in found, r["id"]
                assert found <= {"unsupported_chart"} | unmatched, r["id"]
            elif r["category"] == "near_asset_limit":
                assert "near_asset_limit" in found, r["id"]
            else:
                assert found <= unmatched, r["id"]
            checked += 1
    assert checked > 1000


def test_every_gold_target_gets_the_expected_reasons(platform):
    for line in gold_path("tax-booking").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        target = record["target"]
        found = check_booking(target, record["chart"], platform)
        if not target["needs_review"]:
            assert found == [], record["id"]
