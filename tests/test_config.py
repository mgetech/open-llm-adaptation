import copy
from itertools import chain

import pytest
import yaml
from pydantic import ValidationError

from adapt.config import PLATFORMS_DIR, Platform, Rules, load_platform
from adapt.prompting import system_prompt

LAYOUTS = {
    "table_invoice",
    "till_receipt",
    "letter_invoice",
    "credit_note",
    "eu_invoice",
    "short_receipt",
    "column_invoice",
    "service_invoice",
    "email_invoice",
    "statement",
    "tax_code_receipt",
    "quantity_receipt",
}


@pytest.fixture(scope="module")
def platform() -> Platform:
    return load_platform("tax-booking")


@pytest.fixture
def rules_data() -> dict:
    with (PLATFORMS_DIR / "tax-booking" / "rules.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_platform_loads(platform):
    assert platform.config.name == "tax-booking"
    assert platform.config.charts == ["SKR03"]
    assert platform.rules.chart == "SKR03"


def test_gate_thresholds(platform):
    gate = platform.config.gate
    assert gate.automation_rate_min == 0.60
    assert gate.accuracy_automated_min == 0.95
    assert gate.critical_error_rate_max == 0.005
    assert platform.config.asset_limit.net_eur == 800
    assert platform.config.asset_limit.margin_eur == 50


def test_rules_have_about_the_expected_size(platform):
    assert 24 <= len(platform.rules.accounts) <= 28
    assert 60 <= len(platform.rules.supplier_types) <= 80
    assert sum(len(s.items) for s in platform.rules.supplier_types) >= 600


def test_every_layout_is_used(platform):
    used = set(chain.from_iterable(s.layouts for s in platform.rules.supplier_types))
    assert used == LAYOUTS


def test_letter_invoice_has_enough_suppliers(platform):
    suppliers = [s for s in platform.rules.supplier_types if "letter_invoice" in s.layouts]
    assert len(suppliers) >= 8


def test_reverse_charge_layout_matches_country(platform):
    for supplier in platform.rules.supplier_types:
        if supplier.country == "DE":
            assert "eu_invoice" not in supplier.layouts, supplier.key
        else:
            assert "eu_invoice" in supplier.layouts, supplier.key
            assert "credit_note" not in supplier.layouts, supplier.key
            assert not {"till_receipt", "short_receipt"} & set(supplier.layouts), supplier.key


def test_the_items_of_every_account_have_many_names(platform):
    names = {}
    for supplier in platform.rules.supplier_types:
        for item in supplier.items:
            names.setdefault(item.account, set()).add(item.description)
    assert len(names["1800"]) >= 150
    for account in ("4930", "4940", "4610", "4950", "4964", "3300", "3400", "0480", "4805"):
        assert len(names[account]) >= 15, account


def test_no_item_name_is_used_for_two_accounts_or_two_types(platform):
    seen = {}
    for supplier in platform.rules.supplier_types:
        for item in supplier.items:
            assert item.description not in seen, item.description
            seen[item.description] = supplier.key


def test_private_shops_only_print_receipts(platform):
    receipts = {"till_receipt", "short_receipt", "tax_code_receipt", "quantity_receipt"}
    private = {a.number for a in platform.rules.accounts if a.group == "private"}
    for supplier in platform.rules.supplier_types:
        if any(item.account in private for item in supplier.items):
            assert set(supplier.layouts) <= receipts, supplier.key


def test_prices_stay_clear_of_the_asset_limit(platform):
    limit = platform.config.asset_limit
    low_edge = limit.net_eur - limit.margin_eur
    high_edge = limit.net_eur + limit.margin_eur
    assets = {a.number for a in platform.rules.accounts if a.asset}
    for supplier in platform.rules.supplier_types:
        for item in supplier.items:
            divisor = 1 if supplier.country != "DE" else 1 + int(item.vat_rate) / 100
            net_low, net_high = (price / divisor for price in item.price)
            where = f"{supplier.key}: {item.description}"
            assert net_high <= low_edge or net_low >= high_edge, where
            if item.account not in assets:
                assert net_high <= low_edge, where


def test_prompt_template_renders(platform):
    text = system_prompt("tax-booking", platform.rules.chart, platform.rules)
    assert "SKR03" in text
    for account in platform.rules.accounts:
        assert f"- {account.number} {account.name}\n" in text


def test_load_platform_rejects_wrong_name(tmp_path):
    (tmp_path / "other").mkdir()
    source = PLATFORMS_DIR / "tax-booking"
    for name in ("platform.yaml", "rules.yaml"):
        (tmp_path / "other" / name).write_text((source / name).read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="not 'other'"):
        load_platform("other", root=tmp_path)


def test_platform_rejects_chart_that_is_not_supported(platform):
    data = platform.model_dump()
    data["rules"]["chart"] = "SKR04"
    with pytest.raises(ValidationError, match="not in the platform charts"):
        Platform.model_validate(data)


def test_rules_reject_unknown_account(rules_data):
    rules_data["supplier_types"][0]["items"][0]["account"] = "9999"
    with pytest.raises(ValidationError, match="unknown account 9999"):
        Rules.model_validate(rules_data)


def test_rules_reject_duplicate_account(rules_data):
    rules_data["accounts"].append(copy.deepcopy(rules_data["accounts"][0]))
    with pytest.raises(ValidationError, match="duplicate account"):
        Rules.model_validate(rules_data)


def test_rules_reject_duplicate_supplier_key(rules_data):
    rules_data["supplier_types"].append(copy.deepcopy(rules_data["supplier_types"][0]))
    with pytest.raises(ValidationError, match="duplicate supplier type"):
        Rules.model_validate(rules_data)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("supplier_types", 0, "items", 0, "vat_rate"), "16"),
        (("supplier_types", 0, "items", 0, "price"), [50, 10]),
        (("supplier_types", 0, "layouts"), ["fax"]),
        (("supplier_types", 0, "country"), "FR"),
        (("supplier_types", 0, "weight"), 0),
        (("accounts", 0, "number"), "48"),
        (("accounts", 0, "group"), "tax"),
    ],
)
def test_rules_reject_bad_values(rules_data, path, value):
    target = rules_data
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        Rules.model_validate(rules_data)


def test_rules_reject_unknown_field(rules_data):
    rules_data["accounts"][0]["colour"] = "red"
    with pytest.raises(ValidationError):
        Rules.model_validate(rules_data)


def test_supplier_names_reject_duplicate_syllable(rules_data):
    rules_data["supplier_names"]["syllables"][1] = rules_data["supplier_names"]["syllables"][0]
    with pytest.raises(ValidationError, match="duplicate syllable"):
        Rules.model_validate(rules_data)


def test_supplier_names_reject_missing_suffixes(rules_data):
    del rules_data["supplier_names"]["suffixes"]["EU"]
    with pytest.raises(ValidationError, match="no name suffix for country EU"):
        Rules.model_validate(rules_data)


def test_supplier_names_reject_too_few_syllables(rules_data):
    rules_data["supplier_names"]["syllables"] = rules_data["supplier_names"]["syllables"][:5]
    with pytest.raises(ValidationError):
        Rules.model_validate(rules_data)
