"""Load and validate the platform config and the booking rules."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PLATFORMS_DIR = Path(__file__).resolve().parents[2] / "platforms"

Layout = Literal[
    "table_invoice",
    "till_receipt",
    "letter_invoice",
    "credit_note",
    "eu_invoice",
    "short_receipt",
]
VatRate = Literal["19", "7", "0"]
Country = Literal["DE", "EU", "non-EU"]
AccountGroup = Literal[
    "goods", "operating_expense", "travel_hospitality", "asset", "private", "financial"
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AssetLimit(_Model):
    net_eur: float = Field(gt=0)
    margin_eur: float = Field(ge=0)


class Gate(_Model):
    automation_rate_min: float
    accuracy_automated_min: float
    critical_error_rate_max: float
    mcnemar_p_max: float
    generalisation_gap_max: float
    review_recall_min: float
    invalid_json_rate_max: float
    invented_account_rate_max: float
    injection_success_max: float
    canary_leak_max: float


class PlatformConfig(_Model):
    name: str
    charts: list[str] = Field(min_length=1)
    asset_limit: AssetLimit
    gate: Gate


class Account(_Model):
    number: str = Field(pattern=r"^\d{4}$")
    name: str
    group: AccountGroup
    asset: bool = False


class SupplierItem(_Model):
    description: str
    account: str
    vat_rate: VatRate
    price: tuple[float, float]

    @model_validator(mode="after")
    def check_price(self) -> Self:
        low, high = self.price
        if not 0 < low < high:
            raise ValueError(f"{self.description}: price must satisfy 0 < min < max")
        return self


class SupplierType(_Model):
    key: str
    country: Country
    layouts: list[Layout] = Field(min_length=1)
    items: list[SupplierItem] = Field(min_length=1)

    @model_validator(mode="after")
    def check_layouts(self) -> Self:
        if len(set(self.layouts)) != len(self.layouts):
            raise ValueError(f"{self.key}: duplicate layout")
        return self


class Rules(_Model):
    chart: str
    accounts: list[Account] = Field(min_length=1)
    supplier_types: list[SupplierType] = Field(min_length=1)

    @model_validator(mode="after")
    def check_references(self) -> Self:
        numbers = [account.number for account in self.accounts]
        if len(set(numbers)) != len(numbers):
            raise ValueError("duplicate account number")
        keys = [supplier.key for supplier in self.supplier_types]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate supplier type key")
        for supplier in self.supplier_types:
            for item in supplier.items:
                if item.account not in numbers:
                    raise ValueError(f"{supplier.key}: unknown account {item.account}")
        return self


class Platform(_Model):
    config: PlatformConfig
    rules: Rules

    @model_validator(mode="after")
    def check_chart(self) -> Self:
        if self.rules.chart not in self.config.charts:
            raise ValueError(f"rules chart {self.rules.chart} is not in the platform charts")
        return self


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_platform_config(directory: Path) -> PlatformConfig:
    return PlatformConfig.model_validate(_read_yaml(directory / "platform.yaml"))


def load_rules(directory: Path) -> Rules:
    return Rules.model_validate(_read_yaml(directory / "rules.yaml"))


def load_platform(name: str, root: Path = PLATFORMS_DIR) -> Platform:
    """Load and cross-check the config and rules of the platform called `name`."""
    directory = root / name
    config = load_platform_config(directory)
    if config.name != name:
        raise ValueError(f"platform.yaml names the platform {config.name!r}, not {name!r}")
    return Platform(config=config, rules=load_rules(directory))
