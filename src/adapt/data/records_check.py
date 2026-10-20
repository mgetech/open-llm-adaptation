"""Check newly written evaluation records before they are sent back.

This file uses only the standard library and runs on Python 3.9 or newer. In the folder of the
task it sits next to `audit_check.py`, `rewrite_check.py` and `vat_check.py`, which it uses.

Files in the working folder:
- `new_records.jsonl`: the records, one JSON object per line.
- `config.json`: the accounts, the first and last id, the quotas, and the names already used.

The checks are the ones a program makes later, so a record that passes here passes there.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

try:
    from adapt.data.audit_check import read_jsonl, target_problems
    from adapt.data.rewrite_check import decimal_of, has_amount, has_date
    from adapt.vat_check import vat_mismatch
except ImportError:  # in the folder of the task the files sit side by side
    from audit_check import read_jsonl, target_problems
    from rewrite_check import decimal_of, has_amount, has_date
    from vat_check import vat_mismatch

FIELDS = {"id", "platform", "chart", "layout", "document_text", "target"}
LAYOUTS = (
    "table_invoice",
    "till_receipt",
    "letter_invoice",
    "credit_note",
    "eu_invoice",
    "short_receipt",
)
ASSET_ACCOUNTS = {"0410", "0480", "0650"}
NO_NET_LIMIT = {"0410", "0480", "0650", "1800", "2100", "3300", "3400", "4210", "4360", "4970"}
TOTAL_WORD = re.compile(
    r"(tota[l1I]|amount due|amount payable|balance due|to pay|payable|sum)", re.I
)
FOREIGN = re.compile(r"[$£¥]|\b(?:USD|GBP|CHF|JPY|CNY|CAD|AUD)\b")
UNREADABLE = re.compile(r"(?<![\w?#*.,])(?:[\d?#*]{1,3}[.,])*[\d?#*]{1,3}[.,][\d?#*]{2}(?![\w?#*])")
CENT = Decimal("0.01")


LOOKALIKES = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5"})
NUMBER_LIKE = re.compile(r"(?<![A-Za-z])[0-9OolIS][0-9OolIS.,/\-]*[0-9OolS](?![A-Za-z])")
CONFUSIONS = (("rn", "m"), ("vv", "w"), ("1", "l"), ("0", "o"), ("5", "s"))


def undamage(text: str) -> str:
    """Read the look-alike letters in numbers as digits: 180.OO is 180.00, 2026-O3-15 is a date."""

    def fix(match: re.Match) -> str:
        token = match.group()
        return token.translate(LOOKALIKES) if any(c.isdigit() for c in token) else token

    return NUMBER_LIKE.sub(fix, text)


def canonical(name: str) -> str:
    """A name with the look-alike characters of a scan written in one way, for comparing."""
    name = name.lower()
    for wrong, right in CONFUSIONS:
        name = name.replace(wrong, right)
    return name


def net_of(amount: Decimal, rate: str, reverse_charge: bool) -> Decimal:
    if reverse_charge:
        return abs(amount)
    return abs((amount / (1 + Decimal(rate) / 100)).quantize(CENT, ROUND_HALF_UP))


def record_problems(record: object, accounts: set) -> list[str]:
    if not isinstance(record, dict):
        return ["not a JSON object"]
    if set(record) != FIELDS:
        return [f"the fields must be exactly {sorted(FIELDS)}"]
    problems = []
    if record["platform"] != "tax-booking" or record["chart"] != "SKR03":
        problems.append("platform must be tax-booking and chart SKR03")
    if record["layout"] not in LAYOUTS:
        problems.append(f"layout {record['layout']!r} is not one of {list(LAYOUTS)}")
    text = record["document_text"]
    if not isinstance(text, str) or not text.strip():
        return problems + ["document_text must be a text"]
    found = target_problems(record["target"], accounts)
    if found:
        return problems + found
    target = record["target"]
    credit = target["document_type"] == "credit_note"
    if (target["total"] > 0 and credit) or (target["total"] < 0 and not credit):
        problems.append("the sign of the total does not fit the document type")
    if (record["layout"] == "credit_note") != credit:
        problems.append("only the credit_note layout has document_type credit_note")
    clean = undamage(text)
    if canonical(target["supplier"]) not in canonical(text):
        problems.append("the supplier is not written in the text")
    if not has_date(clean, target["document_date"]):
        problems.append("the date is not written in the text")
    if "4654" in {i["account"] for i in target["items"]}:
        problems.append("account 4654 is not used")
    review_only = target["needs_review"]
    if review_only:
        if target["items"]:
            problems.append("a document for review has no items")
        return problems
    problems += book_problems(record, target, clean)
    return problems


def book_problems(record: dict, target: dict, text: str) -> list[str]:
    problems = []
    reverse = target["reverse_charge"]
    for item in target["items"]:
        amount = decimal_of(item["amount"])
        if not has_amount(text, amount):
            problems.append(f"the amount {amount} is not in the text")
        net = net_of(amount, item["vat_rate"], reverse)
        account = item["account"]
        if account in ("0410", "0650") and net <= 900:
            problems.append(f"{account} needs a net price above 900, not {net}")
        if account == "0480" and net >= 700:
            problems.append(f"0480 needs a net price below 700, not {net}")
        if account in ASSET_ACCOUNTS and 770 <= net <= 830:
            problems.append("a net price near 800 on an asset is a review document")
        if account not in NO_NET_LIMIT and net >= 650:
            problems.append(f"an expense item has a net price below 650, not {net} ({account})")
    if not has_amount(text, decimal_of(target["total"])):
        problems.append(f"the total {target['total']} is not in the text")
    if not TOTAL_WORD.search(text):
        problems.append("the text has no word for the total")
    if FOREIGN.search(text) and target["currency"] == "EUR":
        problems.append("a foreign currency in a document that is booked in EUR")
    smudged = [
        m
        for m in UNREADABLE.finditer(text)
        if any(c.isdigit() for c in m.group()) and any(c in "?#*" for c in m.group())
    ]
    if smudged:
        problems.append("amounts that cannot be read; that is a review document")
    if vat_mismatch(text, target["items"], reverse):
        problems.append("the VAT lines in the text do not match the items")
    if re.search(r"\baccount\s*(no\.?|number)?\s*\d{4}\b", text, re.I) and not re.search(
        r"ignore|system|assistant|instruction|approve|ai\b", text, re.I
    ):
        problems.append("an account number in the text")
    return problems


def quota_problems(records: list[dict], config: dict) -> list[str]:
    rows = [r for r in records if isinstance(r, dict) and isinstance(r.get("target"), dict)]
    targets = [
        r["target"] for r in rows if not target_problems(r["target"], set(config["accounts"]))
    ]
    layouts = Counter(r.get("layout") for r in rows)
    problems = [
        f"layout {name}: {layouts[name]} written, {count} needed"
        for name, count in config["layouts"].items()
        if layouts[name] != count
    ]
    stats = {
        "review documents": sum(t["needs_review"] for t in targets),
        "reverse charge": sum(t["reverse_charge"] for t in targets),
        "documents with 19% and 7% items": sum(
            {"19", "7"} <= {i["vat_rate"] for i in t["items"]} for t in targets
        ),
        "documents with a 0% item": sum(
            any(i["vat_rate"] == "0" for i in t["items"]) for t in targets
        ),
        "documents with a discount line": sum(
            t["document_type"] != "credit_note" and any(i["amount"] < 0 for i in t["items"])
            for t in targets
        ),
        "purchases on 0410, 0480 or 0650": sum(
            any(i["account"] in ASSET_ACCOUNTS for i in t["items"]) for t in targets
        ),
        "private purchases (1800)": sum(
            any(i["account"] == "1800" for i in t["items"]) for t in targets
        ),
        "documents with 4 or more items": sum(len(t["items"]) >= 4 for t in targets),
        "documents with one item": sum(len(t["items"]) == 1 for t in targets),
        "different accounts": len({i["account"] for t in targets for i in t["items"]}),
    }
    for name, minimum in config["minimums"].items():
        if stats[name] < minimum:
            problems.append(f"{name}: {stats[name]} written, at least {minimum} needed")
    if stats["review documents"] != config["review_documents"]:
        written, needed = stats["review documents"], config["review_documents"]
        problems.append(f"review documents: {written} written, {needed} needed")
    return problems


def check_folder(folder: Path) -> list[str]:
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    accounts = set(config["accounts"])
    first, last = config["first_id"], config["last_id"]
    expected = [f"real-{n:03d}" for n in range(first, last + 1)]
    rows = read_jsonl(folder / "new_records.jsonl")
    used = {name.lower() for name in config["used_names"]}
    problems: list[str] = []
    seen_ids: set = set()
    seen_names: set = set()
    seen_texts: set = set()
    good = []
    for row in rows:
        if "_broken" in row:
            problems.append(f"a line is not valid JSON: {row['_broken']!r}")
            continue
        label = row.get("id", "?")
        if label not in expected:
            problems.append(f"{label}: the id must be one of {expected[0]} to {expected[-1]}")
            continue
        if label in seen_ids:
            problems.append(f"{label}: written twice")
            continue
        seen_ids.add(label)
        found = record_problems(row, accounts)
        problems += [f"{label}: {p}" for p in found]
        if not found:
            good.append(row)
        supplier = str(row.get("target", {}).get("supplier", "")).lower()
        if supplier in used or supplier in seen_names:
            problems.append(f"{label}: the supplier name {supplier!r} is already used")
        seen_names.add(supplier)
        text = " ".join(str(row.get("document_text", "")).split()).lower()
        if text in seen_texts:
            problems.append(f"{label}: the same text as another record")
        seen_texts.add(text)
    problems += [f"{i}: not written yet" for i in expected if i not in seen_ids]
    if not problems:
        problems += quota_problems(rows, config)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check newly written records.")
    parser.add_argument("--dir", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    problems = check_folder(args.dir)
    for problem in problems[:60]:
        print(problem)
    if len(problems) > 60:
        print(f"... and {len(problems) - 60} more")
    print("all records are good" if not problems else f"{len(problems)} problems")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
